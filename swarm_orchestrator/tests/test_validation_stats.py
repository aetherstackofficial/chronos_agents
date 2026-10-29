"""Statistical unit tests of the realism-validation package on synthetic
processes with known answers (so the baseline report can be trusted)."""
import copy
import os

import numpy as np
import pytest
import yaml

from validation.checks.bar_tests import (t1_fat_tails, t2_vol_clustering,
                                         t3_no_autocorr)
from validation.checks.flow_tests import (a2_sign_memory, a8_no_free_lunch,
                                          taker_order_signs)
from validation.data import BarSeries, deseasonalize, intraday_returns
from validation.stats_utils import (ljung_box_p, ljung_box_p_statsmodels,
                                    power_law_fit)

_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "validation", "config", "thresholds.yaml")
with open(_CFG_PATH, "r", encoding="utf-8") as _f:
    CFG = yaml.safe_load(_f)

NB = 375


def make_bars(logp, nd, seed=0):
    """BarSeries from a log-price path of length nd*375 (close = exp(logp))."""
    rng = np.random.default_rng(seed + 1000)
    n = nd * NB
    assert len(logp) == n
    c = np.exp(logp)
    h = c * np.exp(np.abs(rng.normal(0, 1e-4, n)))
    l = c * np.exp(-np.abs(rng.normal(0, 1e-4, n)))
    v = rng.integers(1, 1000, n).astype(float)
    return BarSeries(day=np.repeat(np.arange(nd), NB), minute=np.tile(np.arange(NB), nd),
                     o=c.copy(), h=h, l=l, c=c, v=v, mid=c.copy(), label="SYN")


def bars_from_returns(r, nd):
    return make_bars(np.cumsum(r) + np.log(100.0), nd)


def ar1(n, phi, sigma, seed):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sigma, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def garch(n, omega, alpha, beta, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    r = np.empty(n)
    var = omega / (1 - alpha - beta)
    for i in range(n):
        r[i] = np.sqrt(var) * z[i]
        var = omega + alpha * r[i] ** 2 + beta * var
    return r


# 1 ------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("lags", [5, 10])
def test_ljung_box_matches_statsmodels(seed, lags):
    rng = np.random.default_rng(seed)
    iid = rng.standard_normal(3000)
    ar = ar1(3000, 0.2, 1.0, seed)
    for x in (iid, ar, np.abs(iid)):
        assert ljung_box_p(x, lags) == pytest.approx(
            ljung_box_p_statsmodels(x, lags), rel=1e-8, abs=1e-300)


# 2 ------------------------------------------------------------------------
def test_intraday_returns_drop_day_boundaries():
    nd = 54
    rng = np.random.default_rng(5)
    logp = np.cumsum(rng.normal(0, 1e-3, nd * NB))
    logp[3 * NB:] += 1.0                         # huge overnight jump into day 3
    s = make_bars(logp, nd)
    out, idx = intraday_returns(s)
    assert len(out) == len(s) - nd
    assert np.all(s.day[idx] == s.day[idx - 1])
    assert np.max(np.abs(out)) < 0.1             # jump absent
    assert 3 * NB not in idx
    assert not np.any(s.minute[idx] == 0)


# 3 ------------------------------------------------------------------------
def test_deseasonalize_bin_means_one():
    nd = 60
    rng = np.random.default_rng(7)
    minute = np.tile(np.arange(NB), nd)
    f = 1.0 + 3.0 * np.exp(-minute / 30.0) + 0.5 * (minute > 340)
    x = f * rng.lognormal(0, 0.3, nd * NB)
    y = deseasonalize(x, minute)
    bm = np.bincount(minute, weights=y) / np.bincount(minute)
    np.testing.assert_allclose(bm, 1.0, atol=1e-9)
    assert abs(y.mean() - 1.0) < 1e-9


# 4 ------------------------------------------------------------------------
def test_power_law_fit_recovers_beta():
    k = np.arange(1, 101)
    fit = power_law_fit(0.3 * k ** -0.3)
    assert abs(fit["beta"] - 0.3) < 0.02
    assert fit["n_pos"] == 100


# 5 ------------------------------------------------------------------------
def test_iid_gaussian_t1_fails_t2_fails_t3_passes():
    nd = 54
    rng = np.random.default_rng(11)
    s = bars_from_returns(rng.normal(0, 1e-3, nd * NB), nd)
    cfg = copy.deepcopy(CFG)
    cfg["protocol"]["bootstrap_resamples"] = 200
    t1 = t1_fat_tails(s, cfg, price="c", bootstrap=True, seed=1)
    assert not t1["criteria"]["gate_ci95_lower_gt_0"]
    assert not t1["pass"]
    assert abs(t1["metrics"]["excess_kurtosis"]) < 0.3
    t2 = t2_vol_clustering(s, CFG, price="c")
    assert not t2["pass"]
    assert not t2["criteria"]["rho_1_ge_min"]
    t3 = t3_no_autocorr(s, CFG, price="c")
    assert t3["pass"], t3["metrics"]


# 6 ------------------------------------------------------------------------
def test_student_t_fat_tails():
    nd = 54
    rng = np.random.default_rng(13)
    s = bars_from_returns(rng.standard_t(3, nd * NB) * 5e-4, nd)
    t1 = t1_fat_tails(s, CFG, price="c", bootstrap=False)
    assert t1["metrics"]["excess_kurtosis"] > CFG["t1_fat_tails"]["target_excess_kurtosis"]
    assert t1["criteria"]["target_excess_kurtosis"]
    assert t1["pass"]
    t3 = t3_no_autocorr(s, CFG, price="c")
    assert t3["pass"], t3["metrics"]


# 7 ------------------------------------------------------------------------
def test_garch_vol_clustering():
    nd = 80
    s = bars_from_returns(garch(nd * NB, 1e-7, 0.1, 0.88, seed=17), nd)
    t2 = t2_vol_clustering(s, CFG, price="c")
    assert t2["criteria"]["ljung_box_p_lt_0.01"]
    assert t2["criteria"]["rho_1_10_all_positive"]
    assert t2["metrics"]["rho_1"] >= 0.05


# 8 ------------------------------------------------------------------------
def test_ar1_negative_phi_fails_t3():
    nd = 54
    s = bars_from_returns(ar1(nd * NB, -0.1, 1e-3, seed=19), nd)
    t3 = t3_no_autocorr(s, CFG, price="c")
    assert not t3["criteria"]["max_abs_rho_le_0.03"]
    assert not t3["pass"]
    assert t3["metrics"]["rho_1"] < -0.07


# 9 ------------------------------------------------------------------------
def test_a8_random_walk_no_free_lunch():
    nd = 54
    rng = np.random.default_rng(23)
    s = bars_from_returns(rng.normal(0, 1e-3, nd * NB), nd)
    res = a8_no_free_lunch(s, CFG, seed=3)
    ts = [v for k, v in res["metrics"].items() if k.startswith("t_")]
    assert len(ts) == 8
    assert max(ts) <= 3.0
    assert res["metrics"]["random_mean_net_bps"] < 0


# 10 -----------------------------------------------------------------------
def test_a8_detects_mean_reversion_free_lunch():
    nd = 54
    # Price = slow random walk + large iid noise (bounce): returns are strongly
    # negatively autocorrelated, and multi-bar past returns predict reversal.
    rng = np.random.default_rng(29)
    n = nd * NB
    logp = np.cumsum(rng.normal(0, 1e-3, n)) + rng.normal(0, 1e-2, n)
    s = make_bars(logp, nd)
    res = a8_no_free_lunch(s, CFG, seed=3)
    assert res["metrics"]["max_rule_t"] > 3.0
    assert not res["criteria"]["no_rule_t_gt_3"]


# 11 -----------------------------------------------------------------------
def _trades(bar, sign, buyer=None, seller=None):
    n = len(bar)
    buyer = buyer if buyer is not None else [f"B{i}" for i in range(n)]
    seller = seller if seller is not None else [f"S{i}" for i in range(n)]
    return {"bar": np.asarray(bar), "sign": np.asarray(sign),
            "buyer": np.asarray(buyer, dtype=object),
            "seller": np.asarray(seller, dtype=object)}


def test_taker_order_signs_merge():
    t = _trades([1, 1, 1, 1, 1, 2],
                [1, 1, 1, -1, -1, -1],
                buyer=["a", "a", "a", "b", "c", "d"],
                seller=["x", "y", "z", "s", "s", "t"])
    # buy order a (3 prints) -> 1; sell taker s (2 prints, same bar) -> 1;
    # sell taker t in a new bar -> 1
    assert taker_order_signs(t, min_bar=0).tolist() == [1.0, -1.0, -1.0]
    # same taker/sign in a different bar is not merged
    assert len(taker_order_signs(_trades([1, 2], [1, 1], buyer=["a", "a"]), 0)) == 2
    # different taker, same bar/sign is not merged
    assert len(taker_order_signs(_trades([1, 1], [1, 1], buyer=["a", "b"]), 0)) == 2
    # min_bar filter
    assert len(taker_order_signs(t, min_bar=2)) == 1


def test_a2_sign_memory_persistent_vs_iid():
    n = 20000
    persistent = np.where(ar1(n, 0.95, 1.0, seed=31) > 0, 1, -1)
    iid = np.where(np.random.default_rng(37).random(n) < 0.5, 1, -1)
    bars = np.arange(n)           # one print per bar: nothing merges
    res_p = a2_sign_memory(_trades(bars, persistent), CFG, min_bar=0)
    res_i = a2_sign_memory(_trades(bars, iid), CFG, min_bar=0)
    assert res_p["criteria"]["rho_1_20_all_positive"]
    assert res_p["metrics"]["rho_1"] > 0.5
    assert not res_i["criteria"]["rho_1_20_all_positive"]
