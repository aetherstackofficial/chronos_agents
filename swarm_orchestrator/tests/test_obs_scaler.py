"""agents/obs_scaler.py (fixed per-input scaler) and the fitting rules of training/fit_scaler.py."""
import json
import warnings

import numpy as np
import pytest

from agents.features_v2 import OBS_SPEC, spec_hash
from agents.obs_scaler import ObsScaler, is_passthrough
from training import fit_scaler as F

WHALE = list(OBS_SPEC["whale"])
MM = list(OBS_SPEC["mm"])


def test_transform_and_clip():
    sc = ObsScaler(["a", "b", "c"], [1.0, 0.0, -2.0], [2.0, 1.0, 0.5], clip=10.0)
    out = sc.transform(np.array([5.0, 3.0, -2.25]))
    assert out.dtype == np.float32 and out.shape == (3,)
    assert np.allclose(out, [2.0, 3.0, -0.5])
    big = sc.transform(np.array([1e6, -1e6, np.nan]))
    assert big[0] == 10.0 and big[1] == -10.0 and np.isfinite(big[2])
    batch = sc.transform(np.array([[1.0, 0.0, -2.0], [3.0, 1.0, -1.5]]))
    assert batch.shape == (2, 3) and np.allclose(batch, [[0, 0, 0], [1, 1, 1]])
    assert np.array_equal(sc(np.zeros(3)), sc.transform(np.zeros(3)))
    with pytest.raises(ValueError):
        sc.transform(np.zeros(4))
    with pytest.raises(ValueError):
        ObsScaler(["a"], [0.0], [0.0])                       # scale must be > 0


def test_passthrough_inputs_are_untouched():
    for role in ("whale", "mm"):
        names = list(OBS_SPEC[role])
        X = np.random.default_rng(0).normal(size=(500, len(names))) * 3.0
        sc, report = F.fit_scaler_from_samples(names, X)
        pt = [i for i, n in enumerate(names) if is_passthrough(n)]
        assert pt and set(report) == {n for n in names if not is_passthrough(n)}
        assert np.all(sc.center[pt] == 0.0) and np.all(sc.scale[pt] == 1.0)
        v = np.zeros(len(names))
        v[pt] = 0.7
        assert np.allclose(sc.transform(v)[pt], 0.7)
    expect = {"sector_Energy", "tod_sin", "tod_cos", "dow_sin", "dow_cos", "to_close", "last_action_0",
              "last_action_3", "inventory", "holding", "risk_appetite", "mem_hist_avail", "mem_range_pos_today",
              "mem_range_pos_5d", "mem_year_range_pos", "mem_lvl_up1_strength", "mem_lvl_dn3_touches",
              "mem_lvl_up2_age"}
    assert all(is_passthrough(n) for n in expect)
    fitted = {"vwap_dist", "bar_range", "volume_ratio", "rsi", "order_flow", "mem_ret_1m", "mem_lvl_up1_dist",
              "open_pnl", "drawdown", "pnl_today", "exposure", "cap_vs_adv", "traded_vs_adv",
              "mem_ema200d_dist_atr"}
    assert not any(is_passthrough(n) for n in fitted)


def test_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, len(MM)))
    sc, _ = F.fit_scaler_from_samples(MM, X)
    p = tmp_path / "s.json"
    sc.save(p, role="mm", n_samples=300, fit_date="2026-01-01T00:00:00")
    d = json.loads(p.read_text())
    assert d["names"] == MM and d["clip"] == 10.0 and d["spec_hash"] == spec_hash("mm")
    assert d["n_samples"] == 300 and d["fit_date"] == "2026-01-01T00:00:00" and len(d["center"]) == len(MM)
    sc2 = ObsScaler.load(p)
    assert sc2.names == MM and np.array_equal(sc2.center, sc.center) and np.array_equal(sc2.scale, sc.scale)
    x = rng.normal(size=len(MM))
    assert np.array_equal(sc.transform(x), sc2.transform(x))
    assert ObsScaler.from_dict(sc.to_dict()).names == MM
    # a stale spec_hash only warns (strict_hash raises)
    d["spec_hash"] = "0" * 64
    p.write_text(json.dumps(d))
    with pytest.warns(UserWarning):
        ObsScaler.load(p)
    with pytest.raises(ValueError):
        ObsScaler.load(p, strict_hash=True)


def test_load_refuses_mismatching_names(tmp_path):
    sc, _ = F.fit_scaler_from_samples(MM, np.random.default_rng(2).normal(size=(200, len(MM))))
    p = tmp_path / "s.json"
    sc.save(p, role="mm")
    d = json.loads(p.read_text())
    swapped = dict(d, names=list(reversed(d["names"])))
    p.write_text(json.dumps(swapped))
    with pytest.raises(ValueError):
        ObsScaler.load(p)
    p.write_text(json.dumps(dict(d, names=d["names"][:-1], center=d["center"][:-1], scale=d["scale"][:-1])))
    with pytest.raises(ValueError):
        ObsScaler.load(p)
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):                           # right file, wrong role asked for
        ObsScaler.load(p, role="whale")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ObsScaler.load(p, role="mm").names == MM
    with pytest.raises(ValueError):                           # save() refuses foreign names too
        ObsScaler(["x", "y"], [0, 0], [1, 1]).save(tmp_path / "bad.json")


def test_fit_input_iqr_rule():
    rng = np.random.default_rng(3)
    x = rng.normal(5.0, 2.0, 50_000)
    c, s, rep = F.fit_input(x)
    assert abs(c - 5.0) < 0.05 and abs(s - 2.0) < 0.05 and rep["method"] == "iqr" and not rep["gated"]
    assert rep["clip_share"] < 0.01
    # std floor: a constant input still gets a positive scale
    c, s, rep = F.fit_input(np.full(1000, 0.25))
    assert c == 0.25 and s >= 1e-6 and s >= 1e-3 * 0.25 - 1e-12


def test_gated_zero_rule():
    rng = np.random.default_rng(4)
    nz = rng.normal(3.0, 0.5, 2_000)
    x = np.concatenate([np.zeros(8_000), nz])                  # 80% exact zeros
    c, s, rep = F.fit_input(x)
    assert rep["gated"] and abs(rep["zero_share"] - 0.8) < 1e-9
    assert abs(c - np.median(nz)) < 1e-9                       # statistics of the non-zero samples only
    assert abs(s - 0.5) < 0.05 and rep["method"] == "iqr"
    sc = ObsScaler(["g"], [c], [s])
    # zero maps to (0 - center) / scale (about -6), inside the clip
    assert np.isclose(sc.transform(np.array([0.0]))[0], (0.0 - c) / s, atol=1e-4)
    assert -9.0 < sc.transform(np.array([0.0]))[0] < -3.0 and rep["clip_share"] <= 0.01
    # not gated at 50% or less zeros: plain median / IQR over everything
    y = np.concatenate([np.zeros(4_000), rng.normal(3.0, 0.5, 6_000)])
    c2, s2, rep2 = F.fit_input(y)
    assert not rep2["gated"] and c2 > 0.0


def test_widening_when_too_many_at_the_clip():
    rng = np.random.default_rng(5)
    # tight core + 5% far tail: the IQR scale puts the tail on the clip, the fit must widen
    x = np.concatenate([rng.normal(0.0, 1.0, 9_500), rng.normal(0.0, 400.0, 500)])
    c, s, rep = F.fit_input(x)
    assert rep["clip_share_iqr"] > 0.01 and rep["method"] != "iqr"
    assert rep["clip_share"] <= 0.01 and s > 1.0 / 1.349 * 0.9


def test_fit_scaler_from_samples_report():
    names = WHALE
    rng = np.random.default_rng(6)
    X = rng.normal(size=(2_000, len(names)))
    X[:, names.index("mem_ema200d_dist_atr")] = 0.0
    X[:300, names.index("mem_ema200d_dist_atr")] = rng.normal(2.0, 1.0, 300)
    sc, rep = F.fit_scaler_from_samples(names, X)
    r = rep["mem_ema200d_dist_atr"]
    assert r["gated"] and r["zero_share"] > 0.8 and sc.meta["fit"]["mem_ema200d_dist_atr"]["gated"]
    assert "inventory" in sc.meta["passthrough"] and "inventory" not in rep
    assert F.format_report(names, rep, sc.clip).count("\n") == len(rep)
    with pytest.raises(ValueError):
        F.fit_scaler_from_samples(names, X[:, :-1])
