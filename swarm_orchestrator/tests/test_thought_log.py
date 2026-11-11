"""Decision log (agents/thought_log.py) + HTML report (tools/thought_report.py)."""
import json
import math
import time

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, TD3

from agents.thought_log import ThoughtLog, explain

N_IN, N_OUT = 12, 3
NAMES = [f"in{i}" for i in range(N_IN)]


class DummyEnv(gym.Env):
    observation_space = spaces.Box(-5, 5, (N_IN,), np.float32)
    action_space = spaces.Box(-1, 1, (N_OUT,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(N_IN, np.float32), {}

    def step(self, a):
        return np.zeros(N_IN, np.float32), 0.0, False, False, {}


def _make(algo):
    env = DummyEnv()
    kw = dict(policy_kwargs=dict(net_arch=[32, 32]), seed=0, device="cpu")
    if algo == "ppo":
        return PPO("MlpPolicy", env, **kw)
    if algo == "sac":
        return SAC("MlpPolicy", env, **kw)
    return TD3("MlpPolicy", env, **kw)


@pytest.fixture(params=["ppo", "sac", "td3"])
def model(request):
    return _make(request.param)


def _obs(seed=1):
    return np.random.default_rng(seed).normal(size=N_IN).astype(np.float32)


def _params(m):
    return [p.detach().clone() for p in m.policy.parameters()]


def test_explain_shapes_and_finite(model):
    obs = _obs()
    out = explain(model, obs, NAMES, top_k=4, hold_action=np.zeros(N_OUT))
    sure, why = out["sure"], out["why"]
    assert len(sure["action_mean"]) == N_OUT
    assert all(math.isfinite(v) for v in sure["action_mean"])
    if sure["algo"] == "ppo":
        assert math.isfinite(sure["value"])
        assert len(sure["action_std"]) == N_OUT and all(s > 0 for s in sure["action_std"])
    elif sure["algo"] == "sac":
        assert len(sure["action_std"]) == N_OUT and all(s > 0 for s in sure["action_std"])
        assert math.isfinite(sure["q_chosen"]) and math.isfinite(sure["q_hold"])
        assert len(sure["q_all"]) == 2
    else:
        assert sure["action_std"] is None
        assert math.isfinite(sure["q_chosen"]) and math.isfinite(sure["q_hold"])
    assert len(why["per_output"]) == N_OUT
    for po in why["per_output"]:
        assert len(po["top"]) == 4
        for e in po["top"]:
            assert e["name"] in NAMES and math.isfinite(e["contribution"])
    assert len(why["combined"]) == 4
    assert all(e["name"] in NAMES for e in why["combined"])
    tot = [e["total_abs"] for e in why["combined"]]
    assert tot == sorted(tot, reverse=True)


def test_zero_input_has_zero_attribution(model):
    obs = _obs()
    obs[3] = 0.0
    why = explain(model, obs, NAMES, top_k=N_IN)["why"]
    for po in why["per_output"]:
        e = next(e for e in po["top"] if e["name"] == "in3")
        assert e["contribution"] == 0.0 and e["sign"] == 0


def test_explain_does_not_touch_model(model):
    before = _params(model)
    flag = model.policy.training
    model.policy.set_training_mode(True)
    flag_train = model.policy.training
    explain(model, _obs(), NAMES, hold_action=np.zeros(N_OUT))
    assert model.policy.training is flag_train
    model.policy.set_training_mode(False)
    explain(model, _obs(), NAMES, hold_action=np.zeros(N_OUT))
    assert model.policy.training is False
    for a, b in zip(before, _params(model)):
        assert torch.equal(a, b)
    assert all(p.grad is None for p in model.policy.parameters())


def test_phrases_and_raw(model):
    raw = _obs() * 3
    why = explain(model, _obs(), NAMES, top_k=N_IN, obs_raw=raw,
                  phrases={"in0": "nearest support {raw:+.2f} ATR away"})["why"]
    ents = [e for po in why["per_output"] for e in po["top"]]
    e0 = next(e for e in ents if e["name"] == "in0")
    assert e0["phrase"].startswith("nearest support") and e0["raw"] == pytest.approx(float(raw[0]))
    other = next(e for e in ents if e["name"] != "in0")
    assert other["phrase"].startswith(other["name"] + "=")


def _log(tmp_path, algo="ppo", **kw):
    return ThoughtLog("run1", "WHALE_1", "whale_v1", algo, out_dir=str(tmp_path), **kw)


def _rec(tl, model, minute, intent, day=0, changed=None, last_price=100.0, **kw):
    obs = _obs(minute)
    return tl.record(sim_minute=minute, day=day, mode="REPLAY", obs_names=NAMES, obs_raw=obs * 2,
                     obs_norm=obs, model=model, action_raw=np.array([0.1, -0.2, 0.3]),
                     intent=intent, notes=["n"], orders=[{"side": "BUY", "qty": 10}],
                     memory_summary={"last_price": last_price, "levels": {"above": [{"price": 101.0}], "below": [{"price": 99.0}]}},
                     state={"position": 0, "equity": 1000.0}, changed=changed,
                     unix_time=1_700_000_000 + minute * 60, **kw)


def test_write_on_change_and_heartbeat_only(tmp_path):
    m = _make("ppo")
    tl = _log(tmp_path)
    a = {"target": 1}
    r0 = _rec(tl, m, 0, a)                         # first -> change
    assert r0 and r0["kind"] == "change" and r0["sure"] and r0["why"]
    assert _rec(tl, m, 5, a) is None               # same intent, too early
    assert _rec(tl, m, 10, a, changed=False) is None
    r = _rec(tl, m, 12, {"target": 2})             # intent changed (detected)
    assert r and r["kind"] == "change"
    r = _rec(tl, m, 13, {"target": 2}, changed=True)  # caller says changed
    assert r and r["kind"] == "change"
    assert _rec(tl, m, 20, {"target": 2}) is None
    hb = _rec(tl, m, 28, {"target": 2})            # 15 min after last record
    assert hb and hb["kind"] == "heartbeat" and hb["why"] is None and hb["sure"] is None
    assert tl.n_records == 4
    tl.close()
    lines = open(tl.path, encoding="utf-8").read().splitlines()
    assert len(lines) == 4


def test_outcomes_attached_and_valid_jsonl(tmp_path):
    m = _make("sac")
    tl = _log(tmp_path, "sac")
    _rec(tl, m, 0, {"t": 1}, hold_action=np.zeros(N_OUT))
    _rec(tl, m, 30, {"t": 2})
    tl.update_outcome(sim_minute=5, day=0, fills=[{"qty": 5}], equity=1000.0)
    tl.update_outcome(sim_minute=16, day=0, fills=[{"qty": 7}], equity=1010.0)
    tl.update_outcome(sim_minute=40, day=0, fills=[], equity=1020.0)
    tl.update_outcome(sim_minute=61, day=0, fills=[{"qty": 1}], equity=1030.0)  # 60m of rec 0
    assert tl.n_written == 1                        # rec 1 still pending
    tl.close()                                      # flushes the rest
    rows = [json.loads(l) for l in open(tl.path, encoding="utf-8")]
    assert [r["when"]["sim_minute"] for r in rows] == [0, 30]
    t0 = rows[0]["then"]
    assert t0["m15"]["complete"] and t0["m15"]["n_fills"] == 1   # only the fill at +5
    assert t0["m15"]["pnl"] == pytest.approx(10.0)               # equity at +16 vs 1000
    assert t0["m60"]["n_fills"] == 2 and t0["m60"]["pnl"] == pytest.approx(30.0)
    t1 = rows[1]["then"]
    assert t1["m15"]["complete"] and t1["m60"]["complete"] is False
    for r in rows:
        assert set(r) == {"kind", "when", "saw", "state", "chose", "sure", "why", "then"}
        assert set(r["saw"]) == {"memory", "obs"} and set(r["saw"]["obs"]) == set(NAMES)
        assert set(r["chose"]) == {"raw", "intent", "notes", "orders"}
    assert rows[0]["sure"]["q_hold"] is not None


def test_nonfinite_values_stay_valid_json(tmp_path):
    m = _make("td3")
    tl = _log(tmp_path, "td3")
    obs = _obs()
    tl.record(sim_minute=0, day=0, mode="REACTIVE", obs_names=NAMES, obs_raw=obs, obs_norm=obs,
              model=m, action_raw=np.zeros(N_OUT), intent={"x": float("nan")}, notes=[], orders=[],
              memory_summary={"a": float("inf")}, state={})
    tl.close()
    json.loads(open(tl.path, encoding="utf-8").read())


def test_report_builds(tmp_path):
    from tools.thought_report import build_report, main
    m = _make("ppo")
    tl = _log(tmp_path)
    for i, minute in enumerate(range(0, 200, 10)):
        _rec(tl, m, minute, {"t": i % 3}, last_price=100 + math.sin(i))
        tl.update_outcome(sim_minute=minute, day=0, fills=[{"qty": 1}], equity=1000 + i)
    tl.close()
    html = build_report(tl.path)
    assert html.startswith("<!doctype html>") and "<svg" in html and "WHALE_1" in html
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    out = tmp_path / "rep.html"
    main([str(tmp_path / "run1"), "--out", str(out)])
    produced = list(tmp_path.glob("rep*.html"))
    assert produced and all(p.stat().st_size > 1000 for p in produced)
    # also works with no last_price anywhere
    tl2 = ThoughtLog("run2", "MM_1", "mm", "ppo", out_dir=str(tmp_path))
    obs = _obs()
    tl2.record(sim_minute=0, day=0, mode="REPLAY", obs_names=NAMES, obs_raw=obs, obs_norm=obs,
               model=m, action_raw=np.zeros(N_OUT), intent={}, notes=[], orders=[],
               memory_summary={}, state={})
    tl2.close()
    assert "<html" in build_report(tl2.path)


def test_explain_speed(model):
    big_in, big_out = 80, 4

    class E(DummyEnv):
        observation_space = spaces.Box(-5, 5, (big_in,), np.float32)
        action_space = spaces.Box(-1, 1, (big_out,), np.float32)

    algo = type(model).__name__
    cls = {"PPO": PPO, "SAC": SAC, "TD3": TD3}[algo]
    big = cls("MlpPolicy", E(), policy_kwargs=dict(net_arch=[256, 256]), seed=0, device="cpu")
    names = [f"i{i}" for i in range(big_in)]
    obs = np.random.default_rng(0).normal(size=big_in).astype(np.float32)
    hold = np.zeros(big_out)
    for _ in range(5):
        explain(big, obs, names, hold_action=hold)
    n = 100
    t = time.perf_counter()
    for _ in range(n):
        explain(big, obs, names, hold_action=hold)
    ms = (time.perf_counter() - t) / n * 1000
    print(f"\nexplain() {algo}: {ms:.2f} ms")
    assert ms < 50  # target 5 ms; generous bound so CI noise does not fail
