"""training/{train,callbacks,evaluate,baselines,export}.py and the 9 config files.

The scripts are exercised on a small stub env (same spaces, same info keys, the real
AgentBook / MarketMemory / observation builder, synthetic ticks) so the tests stay fast and do
not depend on the data; one extra test runs the real ReplayBookEnv for a single short episode
when it and the parquet data exist.  Nothing is written to agents/ml_models.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import yaml
from gymnasium import spaces

from agents.features_v2 import OBS_SPEC, AgentBook, build_observation_v2, spec_hash
from core.market_memory import MarketMemory
from agents.obs_scaler import ObsScaler
from training import baselines as B
from training import fit_scaler as FS
from training import evaluate as E
from training import export as X
from training import train as T

ALGOS = ("ppo", "sac", "td3")
MODELS = ("whale_swing", "whale_position", "mm")
SESSION = 40            # stub session length in bars
N_SESSIONS = 3


# --------------------------------------------------------------------------- stub env
class StubEnv(gym.Env):
    """Random-walk ticks + the real observation builder; mimics ReplayBookEnv's API."""

    def __init__(self, model_name, split="train", cfg=None, seed=0):
        super().__init__()
        self.model_name = model_name
        self.role = "mm" if model_name == "mm" else "whale"
        self.split = split
        self._init_seed = seed
        self._seeded = False
        n_act = 4 if self.role == "mm" else 3
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(OBS_SPEC[self.role]),), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (n_act,), np.float32)
        self.sampler = type("S", (), {"symbols": ["AAA"]})()
        self.memory = self.book = self.last_tick = None

    def _tick(self):
        self._t += 1
        self._p *= float(np.exp(self.np_random.normal(0, 0.001)))
        p = self._p
        day, minute = 1 + (self._t - 1) // SESSION, 1 + (self._t - 1) % SESSION
        return {"run_id": "stub", "last_price": p, "vwap": p * 0.999, "volume_ma": 1000.0,
                "step_volume": 800.0 + float(self.np_random.uniform(0, 400)), "rsi": 50.0 + float(self.np_random.normal(0, 8)),
                "ofi": float(self.np_random.uniform(-1, 1)), "bar": {"o": p, "h": p * 1.001, "l": p * 0.999, "c": p, "v": 1000.0, "minute": minute},
                "market_minute": minute, "day_count": day, "sector": "Energy", "unix_time": 1_700_000_000 + 86400 * day,
                "tick_size": 0.05, "best_bid": round(p - 0.05, 2), "best_ask": round(p + 0.05, 2)}

    def _obs(self):
        return build_observation_v2(self.role, self.last_tick, self.memory, self.book)

    def reset(self, *, seed=None, options=None):
        if seed is None and not self._seeded:
            seed = self._init_seed
        super().reset(seed=seed)
        self._seeded = True
        self._t, self._p = 0, 500.0
        self.memory = MarketMemory()
        self.book = AgentBook(1_000_000.0, 1000.0, 0.08, n_outputs=self.action_space.shape[0])
        self.last_tick = self._tick()
        self.memory.update(self.last_tick)
        self.book.on_tick(self.last_tick)
        info = {"symbol": "AAA", "memory_sessions": int(self.np_random.choice([5, 150])), "capacity": 1000}
        return self._obs(), info

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        old = self.last_tick
        tick = self._tick()
        eq0 = self.book.equity
        intent, fills = {}, []
        if self.role == "whale":
            target = int(round(a[0] * 1000))
            delta = int(np.clip(target - self.book.position, -100, 100))
            if delta:
                px = tick["best_ask"] if delta > 0 else tick["best_bid"]
                self.book.on_fill("BUY" if delta > 0 else "SELL", abs(delta), px, 0.1)
                fills.append(("BUY" if delta > 0 else "SELL", abs(delta), px))
            intent = {"participation": 0.01 + 0.07 * (a[1] + 1), "child_qty": abs(delta)}
        else:
            bq, aq = int(max(a[2] + 1, 0) * 50), int(max(a[3] + 1, 0) * 50)
            intent = {"bid_qty": bq, "ask_qty": aq}
            if bq and self.np_random.uniform() < 0.3 and self.book.position < 900:
                self.book.on_fill("BUY", bq, old["best_bid"], 0.05)
                fills.append(("BUY", bq, old["best_bid"]))
            if aq and self.np_random.uniform() < 0.3 and self.book.position > -900:
                self.book.on_fill("SELL", aq, old["best_ask"], 0.05)
                fills.append(("SELL", aq, old["best_ask"]))
        self.memory.update(tick)
        self.book.on_tick(tick)
        self.last_tick = tick
        self.book.set_last_action(a)
        truncated = self._t >= SESSION * N_SESSIONS
        r = (self.book.equity - eq0) / 1_000_000.0 * 1e3
        info = {"reward_parts": {"pnl": r, "cost": -0.001}, "equity": self.book.equity, "position": self.book.position,
                "fills": fills, "notes": [], "intent": intent}
        return self._obs(), float(r), False, truncated, info


def make_stub_env(model_name, split, cfg, seed=0):
    return StubEnv(model_name, split, cfg, seed)


FACTORY = "test_training_scripts:make_stub_env"


def _make_scalers():
    """Tiny scaler files fitted on a few stub episodes (the real ones come from training.fit_scaler)."""
    d = Path(tempfile.mkdtemp(prefix="chronos_scalers_"))
    out = {}
    for role, model in (("whale", "whale_swing"), ("mm", "mm")):
        env = StubEnv(model, "train", None, 0)
        rng = np.random.default_rng(0)
        rows, (obs, _) = [], env.reset(seed=0)
        for k in range(300):
            rows.append(obs)
            obs, _, term, trunc, _ = env.step(rng.uniform(-1, 1, env.action_space.shape[0]))
            if term or trunc:
                obs, _ = env.reset()
        sc, _ = FS.fit_scaler_from_samples(list(OBS_SPEC[role]), np.asarray(rows))
        sc.save(d / f"obs_scaler_{role}.json", role=role, n_samples=len(rows), fit_date="2026-01-01T00:00:00")
        out[role] = str(d / f"obs_scaler_{role}.json")
    return out


SCALERS = _make_scalers()


@pytest.fixture(scope="module")
def stub_factory():
    mp = pytest.MonkeyPatch()
    mp.setenv(T.ENV_FACTORY_VAR, FACTORY)
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:                       # spawned workers import the factory by module name
        mp.syspath_prepend(here)
    yield
    mp.undo()


def small_cfg(model, algo, **over):
    cfg = T.load_config(model, algo)
    cfg.update({"steps": 2000, "n_envs": 2, "eval_every": 1000, "eval_episodes": 2,
                "checkpoint_every": 1000, "net_arch": [32, 32], "obs_scaler": SCALERS[T.role_of(model)]})
    cfg["algo_kwargs"] = dict(cfg["algo_kwargs"])
    if algo == "ppo":
        cfg["algo_kwargs"].update(n_steps=256, batch_size=64, n_epochs=2)
    else:
        cfg["algo_kwargs"].update(batch_size=64, gradient_steps=1, learning_starts=100)
    cfg.update(over)
    return cfg


# --------------------------------------------------------------------------- configs
def test_configs_load_and_complete():
    # steps / eval_every / checkpoint_every count DECISIONS: the old per-minute numbers / K
    K = {"whale_swing": 15, "whale_position": 75, "mm": 1}
    steps = {"ppo": 10_000_000, "sac": 1_000_000, "td3": 1_000_000}
    gamma = {"whale_swing": 0.999, "whale_position": 0.9995, "mm": 0.99}
    for m in MODELS:
        for a in ALGOS:
            cfg = T.load_config(m, a)
            assert cfg["model"] == m and cfg["algo"] == a
            assert cfg["steps"] == round(steps[a] / K[m]) and cfg["n_envs"] == 8
            assert cfg["eval_every"] == round(250_000 / K[m]) and cfg["eval_episodes"] == 10
            assert cfg["checkpoint_every"] == round(250_000 / K[m]) and cfg["frame_stack"] == 1
            assert cfg["gamma"] == gamma[m] and cfg["net_arch"] == [256, 256]
            assert cfg["env_overrides"] == {}
            assert cfg["eval_env_overrides"]["sampler"]["length_sessions"][m] == [5, 5]
            kw = cfg["algo_kwargs"]
            if a == "ppo":
                n_steps = 256 if K[m] > 1 else 2048           # whale: 256 decisions per env per rollout
                assert kw == {"learning_rate": 3e-4, "n_steps": n_steps, "batch_size": 256, "n_epochs": 10,
                              "gae_lambda": 0.95, "clip_range": 0.2}
            elif a == "sac":
                assert kw["ent_coef"] == "auto" and kw["buffer_size"] == 1_000_000 and kw["gradient_steps"] == 4
                assert kw["learning_starts"] == round(10_000 / K[m]) and kw["tau"] == 0.005
            else:
                assert kw["action_noise_sigma"] == 0.1 and kw["policy_delay"] == 2 and kw["learning_rate"] == 1e-3
                assert kw["target_policy_noise"] == 0.2 and kw["target_noise_clip"] == 0.5
    req = (Path(T.TRAIN_DIR) / "requirements.txt").read_text()
    assert "stable-baselines3>=2.3" in req and "tensorboard" in req


def test_gamma_is_converted_to_a_per_decision_gamma():
    assert T.decision_gamma(0.999, 15) == pytest.approx(0.999 ** 15)
    assert T.decision_gamma(0.99, 1) == 0.99
    for m, k in (("whale_swing", 15), ("whale_position", 75), ("mm", 1)):
        cfg = T.load_config(m, "ppo")
        assert T.decision_minutes_of(T.load_env_cfg(cfg), m) == k
        assert T.decision_minutes_of(T.load_env_cfg(cfg, eval_env=True), m) == k
    # an env_overrides entry wins
    cfg = T.load_config("whale_swing", "ppo")
    cfg["env_overrides"] = {"env": {"decision_minutes": {"whale_swing": 5}}}
    assert T.decision_minutes_of(T.load_env_cfg(cfg), "whale_swing") == 5
    assert T.decision_minutes_of({}, "whale_swing") == 1          # no entry (old config): 1 minute


def test_load_env_cfg_merges_overrides():
    cfg = T.load_config("mm", "ppo")
    cfg["env_overrides"] = {"env": {"obs_noise": 0.0}}
    tr, ev = T.load_env_cfg(cfg), T.load_env_cfg(cfg, eval_env=True)
    assert tr["env"]["obs_noise"] == 0.0 and tr["sampler"]["length_sessions"]["mm"] == [1, 5]
    assert ev["sampler"]["length_sessions"]["mm"] == [5, 5]
    # the eval env never sees observation noise, the training env keeps it
    base = T.load_config("mm", "ppo")
    assert T.load_env_cfg(base)["env"]["obs_noise"] == 0.05 and T.load_env_cfg(base, eval_env=True)["env"]["obs_noise"] == 0
    assert E.eval_env_cfg(base, cost_mult=2.0, model_name="mm")["env"]["obs_noise"] == 0
    c2 = E.eval_env_cfg(cfg, cost_mult=2.0, episode_sessions=1, memory_sessions=7, model_name="mm")
    assert c2["costs"]["cost_mult_range"] == [2.0, 2.0] and c2["sampler"]["length_sessions"]["mm"] == [1, 1]
    assert c2["sampler"]["memory_mix"] == [[1.0, 7, 7]]


# --------------------------------------------------------------------------- baselines logic
def test_avellaneda_stoikov_inverts_decoder():
    from agents.action_decode import AgentState, decode_mm
    env = StubEnv("mm", "train", None, 0)
    env.reset(seed=0)
    for _ in range(40):                                 # feed some returns
        env.step(np.zeros(4))
    pol = B.AvellanedaStoikovMM()
    a = pol.predict(None, env)
    assert a.shape == (4,) and np.all(np.abs(a) <= 1.0)
    q = pol.quotes(env)
    state = AgentState(position=0, cash=1e9, capacity=1000, max_quote_size=100)
    res = decode_mm(a, state, env.last_tick)
    # the decoder lands on the integer tick offsets the rule asked for
    assert res.intent["offsets_requested"][0] == min(max(round(q["bid_offset"]), -1), 10)
    assert res.intent["offsets_requested"][1] == min(max(round(q["ask_offset"]), -1), 10)
    assert res.intent["bid_qty"] > 0 and res.intent["ask_qty"] > 0
    # long inventory: reservation price falls -> bid further away, ask closer; bid size shrinks
    env.book.on_fill("BUY", 800, 500.0, 0.0)
    q2 = pol.quotes(env)
    assert q2["reservation_ticks"] < q["reservation_ticks"]
    assert q2["bid_offset"] > q["bid_offset"] and q2["ask_offset"] < q["ask_offset"]
    assert pol.predict(None, env)[2] < a[2]


def test_whale_rules():
    env = StubEnv("whale_swing", "train", None, 0)
    env.reset(seed=0)
    assert np.allclose(B.BuyAndHold().predict(None, env), [1.0, 1.0, 1.0])
    for _ in range(80):
        env.step(np.zeros(3))
    a = B.TrendRule().predict(None, env)
    f = env.memory.features()
    assert a[0] == (1.0 if f["ema20_dist"] < f["ema60_dist"] else -1.0)
    assert a[1] == 1.0 and a[2] == 1.0


# --------------------------------------------------------------------------- end to end (stub)
@pytest.fixture(scope="module")
def ppo_run(stub_factory, tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    run_dir = runs / "whale_swing_ppo_test"
    run_dir.mkdir()
    meta = T.train("whale_swing", "ppo", 3, small_cfg("whale_swing", "ppo"), run_dir)
    return run_dir, meta


def test_train_writes_everything(ppo_run):
    run_dir, meta = ppo_run
    for f in ("model.zip", "vecnormalize.pkl", "config.yaml", "run_meta.json", "obs_scaler.json",
              "best/best_model.zip", "best/best_vecnormalize.pkl", "eval/evaluations.npz"):
        assert (run_dir / f).exists(), f
    assert list((run_dir / "checkpoints").glob("ckpt_*_steps.zip"))
    assert list((run_dir / "checkpoints").glob("ckpt_vecnormalize_*_steps.pkl"))
    assert list((run_dir / "tb").rglob("events.out.tfevents.*"))
    m = json.loads((run_dir / "run_meta.json").read_text())
    assert m["seed"] == 3 and m["status"] == "finished" and m["trained_timesteps"] >= 2000
    assert m["spec_hash"] == spec_hash("whale") and len(m["dataset_manifest_sha256"] or "") in (0, 64)
    assert m["git_commit"] is None or len(m["git_commit"]) == 40
    assert m["obs_scaler_sha256"] == T.sha256_file(SCALERS["whale"]) == T.sha256_file(run_dir / "obs_scaler.json")
    assert m["obs_scaler_n_samples"] == 300
    assert m["steps_per_second"] > 0 and m["libraries"]["stable_baselines3"]
    assert m["start_time"] and m["end_time"]
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert cfg["algo"] == "ppo" and cfg["seed"] == 3 and "env_cfg" in cfg
    # per-minute gamma -> per-decision gamma, recorded in both files
    assert m["decision_minutes"] == cfg["decision_minutes"] == 15
    assert m["gamma_per_minute"] == cfg["gamma_per_minute"] == 0.999
    assert m["gamma_per_decision"] == cfg["gamma_per_decision"] == pytest.approx(0.999 ** 15)
    assert m["minutes_per_second"] == pytest.approx(m["steps_per_second"] * 15, rel=0.01)
    from stable_baselines3 import PPO
    assert PPO.load(str(run_dir / "model.zip"), device="cpu").gamma == pytest.approx(0.999 ** 15)
    import pickle
    with open(run_dir / "vecnormalize.pkl", "rb") as fh:
        assert pickle.load(fh).gamma == pytest.approx(0.999 ** 15)
    assert E.run_decision_minutes(cfg) == 15
    assert E.run_decision_minutes({"model": "whale_swing", "env_cfg": {"env": {}}}) == 1     # pre-interval run


def test_vec_env_scales_observations(stub_factory):
    from training.wrappers import ObsScalerVecEnv, find_wrapper
    sc = ObsScaler.load(SCALERS["whale"], "whale")
    venv = T.build_vec_env("whale_swing", "train", {}, 2, 1, 0.999, subproc=False, scaler=sc)
    try:
        vn = T.find_vecnormalize(venv)
        assert vn.norm_obs is False and vn.norm_reward is True and not hasattr(vn, "obs_rms")
        w = find_wrapper(venv, ObsScalerVecEnv)
        obs = venv.reset()
        assert obs.dtype == np.float32 and np.array_equal(obs, sc.transform(w.raw_obs))
        assert not np.array_equal(obs, w.raw_obs) and np.abs(obs).max() <= sc.clip
        for _ in range(SESSION * N_SESSIONS + 2):                 # through a truncation + auto reset
            obs, _, dones, infos = venv.step(np.zeros((2, 3), dtype=np.float32))
            for i, info in enumerate(infos):
                if "terminal_observation" in info:
                    assert np.abs(info["terminal_observation"]).max() <= sc.clip
        assert np.array_equal(obs, sc.transform(w.raw_obs))
    finally:
        venv.close()


def test_train_refuses_mismatching_scaler(stub_factory, tmp_path):
    bad = tmp_path / "bad.json"
    d = json.loads(Path(SCALERS["whale"]).read_text())
    d["names"] = list(reversed(d["names"]))
    bad.write_text(json.dumps(d))
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    with pytest.raises(ValueError):
        T.train("whale_swing", "ppo", 1, small_cfg("whale_swing", "ppo"), run_dir, scaler_path=str(bad))


def test_insights_logged(ppo_run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run_dir, _ = ppo_run
    f = next((run_dir / "tb").rglob("events.out.tfevents.*"))
    ea = EventAccumulator(str(f.parent))
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    for t in ("insights/reward_part/pnl", "insights/action_mean/target", "insights/action_std/target",
              "insights/abs_position_over_capacity", "insights/share_in_position", "insights/turnover_per_session",
              "clip/worst_share", "eval/mean_reward"):
        assert t in tags, t


def test_evaluate_run_and_thoughts(ppo_run, stub_factory, capsys):
    run_dir, _ = ppo_run
    assert E.main(["--run", str(run_dir), "--split", "validation", "--episodes", "2", "--thoughts"]) == 0
    res = json.loads((run_dir / "eval_validation.json").read_text())
    m = res["metrics"]
    for k in ("profit_after_costs_mean", "sharpe_per_session_annualised", "worst_drawdown", "turnover_per_session",
              "share_in_position", "mean_holding_sessions", "share_long", "share_short", "participation_used"):
        assert k in m, k
    names = {c["name"] for c in res["checks"]}
    assert {"outputs_not_stuck", "whale_trades_long_and_short_and_varies", "no_input_at_clip_over_1pct",
            "no_constant_input", "few_memory_sessions_not_wild", "memory_used_in_top5_reasons",
            "whale_position_change_near_strong_levels"} <= names
    mem = next(c for c in res["checks"] if c["name"] == "memory_used_in_top5_reasons")
    assert mem["passed"] is not None and mem["value"]["n_decisions"] > 0
    assert res["steps"] == 2 * (SESSION * N_SESSIONS - 1)
    sc = ObsScaler.load(SCALERS["whale"], "whale")
    jl = list((run_dir / "thoughts").rglob("*.jsonl"))
    assert len(jl) == 2 and jl[0].stat().st_size > 0
    assert list((run_dir / "thoughts").rglob("*.html"))
    rec = json.loads(jl[0].read_text().splitlines()[0])
    assert rec["why"] and rec["saw"]["memory"]["last_price"] > 0 and "equity" in rec["state"]
    # the decision log carries both: obs_raw ("raw") and the scaled obs the policy saw ("norm")
    for e in rec["why"]["combined"]:
        j = sc.names.index(e["name"])
        assert abs(e["norm"]) <= sc.clip
        assert e["norm"] == pytest.approx(float(np.clip((e["raw"] - sc.center[j]) / sc.scale[j], -sc.clip, sc.clip)),
                                          rel=1e-3, abs=1e-3)


def test_evaluate_is_reproducible(ppo_run, stub_factory):
    run_dir, _ = ppo_run
    a = E.main(["--run", str(run_dir), "--episodes", "1", "--out", str(run_dir / "a.json")])
    b = E.main(["--run", str(run_dir), "--episodes", "1", "--out", str(run_dir / "b.json")])
    ja, jb = json.loads((run_dir / "a.json").read_text()), json.loads((run_dir / "b.json").read_text())
    assert ja["metrics"]["profit_after_costs_mean"] == jb["metrics"]["profit_after_costs_mean"]


@pytest.mark.parametrize("algo", ["sac", "td3"])
def test_off_policy_train_and_eval_mm(stub_factory, tmp_path, algo):
    run_dir = tmp_path / f"mm_{algo}"
    run_dir.mkdir()
    T.train("mm", algo, 1, small_cfg("mm", algo, steps=400, n_envs=1, eval_every=10_000, checkpoint_every=10_000),
            run_dir)
    assert (run_dir / "model.zip").exists() and (run_dir / "vecnormalize.pkl").exists()
    out = run_dir / "e.json"
    assert E.main(["--run", str(run_dir), "--episodes", "1", "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    m = res["metrics"]
    for k in ("pnl_spread_markout_total", "pnl_inventory_total", "spread_captured_per_round_trip",
              "abs_inventory_mean", "abs_inventory_max", "minutes_to_return_to_zero_mean",
              "adverse_selection_1m_per_share", "share_quoted_both"):
        assert k in m, k
    assert abs(m["pnl_spread_markout_total"] + m["pnl_inventory_total"] - m["profit_after_costs_total"]) < 1e-6
    assert {"mm_quotes_at_least_80pct_of_time", "mm_inventory_mean_reverts"} <= {c["name"] for c in res["checks"]}


def test_baselines_write_eval_json(stub_factory, tmp_path):
    out = tmp_path / "bl_mm"
    assert B.main(["--model", "mm", "--split", "validation", "--episodes", "1", "--out-dir", str(out)]) == 0
    res = json.loads((out / "eval_validation.json").read_text())
    assert res["policy"].startswith("baseline:") and res["metrics"]["n_fills"] > 0
    for w in ("buyhold", "trend"):
        o = tmp_path / f"bl_{w}"
        assert B.main(["--model", "whale_position", "--episodes", "1", "--policy", w, "--out-dir", str(o)]) == 0
        r = json.loads((o / "eval_validation.json").read_text())
        assert r["metrics"]["episodes"] == 1
    bh = json.loads((tmp_path / "bl_buyhold" / "eval_validation.json").read_text())
    assert bh["metrics"]["share_long"] > 0.5 and bh["metrics"]["share_short"] == 0.0


def test_export_to_temp_dir(ppo_run, tmp_path, monkeypatch):
    run_dir, _ = ppo_run
    if not (run_dir / "eval_validation.json").exists():
        E.main(["--run", str(run_dir), "--episodes", "1"])
    import time
    spec_time = time.time() - 1.0
    root = tmp_path / "v2"
    monkeypatch.setattr(X, "EXPORT_ROOT", root)
    assert X.main(["--run", str(run_dir), "--name", "whale_swing"]) == 0
    d = root / "whale_swing"
    for f in ("model.zip", "vecnormalize.pkl", "config.yaml", "spec.json", "obs_scaler.json"):
        assert (d / f).exists(), f
    spec = json.loads((d / "spec.json").read_text())
    assert spec["obs_names"] == list(OBS_SPEC["whale"]) and spec["spec_hash"] == spec_hash("whale")
    assert spec["outputs"] == ["target", "participation", "aggressiveness"] and spec["frame_stack"] == 1
    assert spec["model"] == "whale_swing" and spec["algo"] == "ppo" and spec["seed"] == 3
    assert spec["obs_scaler"]["sha256"] == T.sha256_file(d / "obs_scaler.json") == T.sha256_file(SCALERS["whale"])
    assert spec["obs_scaler"]["spec_hash"] == spec_hash("whale") and spec["obs_scaler"]["clip"] == 10.0
    assert spec["eval"]["eval_validation"]["metrics"]
    assert spec["dataset_manifest_sha256"] == json.loads((run_dir / "run_meta.json").read_text())["dataset_manifest_sha256"]
    with pytest.raises(FileExistsError):
        X.export_run(run_dir, "whale_swing", export_root=root)
    with pytest.raises(ValueError):
        X.export_run(run_dir, "../escape", export_root=root)
    assert X.export_run(run_dir, "whale_swing", best=True, overwrite=True, export_root=root) == d.resolve()
    # the real model folder was not written to by this test
    real = T.ROOT / "agents" / "ml_models" / "v2" / "whale_swing" / "spec.json"
    assert not real.exists() or real.stat().st_mtime < spec_time


# --------------------------------------------------------------------------- real env (optional)
def _real_env_ok():
    try:
        import training.envs.replay_book_env  # noqa: F401
        from training.envs.sampler import available_symbols
        return bool(available_symbols())
    except Exception:
        return False


@pytest.mark.skipif(not _real_env_ok(), reason="ReplayBookEnv or parquet data not available")
def test_real_env_baseline_episode(monkeypatch, tmp_path):
    monkeypatch.delenv(T.ENV_FACTORY_VAR, raising=False)
    out = tmp_path / "real"
    assert B.main(["--model", "mm", "--episodes", "1", "--episode-sessions", "1", "--out-dir", str(out)]) == 0
    res = json.loads((out / "eval_validation.json").read_text())
    assert res["steps"] > 300 and res["metrics"]["n_fills"] >= 0
    assert res["episode_summaries"][0]["meta"]["symbol"]


@pytest.mark.skipif(not _real_env_ok(), reason="ReplayBookEnv or parquet data not available")
def test_real_env_whale_baseline_metrics_use_minutes(monkeypatch, tmp_path):
    """A whale baseline at a 15-minute interval: steps are decisions, metrics are in minutes/sessions."""
    monkeypatch.delenv(T.ENV_FACTORY_VAR, raising=False)
    out = tmp_path / "w15"
    assert B.main(["--model", "whale_swing", "--policy", "buyhold", "--episodes", "1", "--episode-sessions", "2",
                   "--decision-minutes", "15", "--out-dir", str(out)]) == 0
    res = json.loads((out / "eval_validation.json").read_text())
    assert res["decision_minutes"] == 15 and res["minutes"] == 2 * 375 - 1
    assert res["steps"] == -(-(2 * 375 - 1) // 15)                       # ceil(749 / 15) decisions
    m = res["metrics"]
    assert m["sessions"] == 2 and len(res["episode_summaries"]) == 1
    # buy and hold: a single position held for ~the whole episode = ~2 sessions of holding
    assert 1.5 <= m["mean_holding_sessions"] <= 2.0 and m["share_in_position"] > 0.9


def test_session_returns_use_the_per_minute_session_closes():
    ep = {"equity0": 100.0, "equity": np.array([101.0, 103.0, 102.0]), "day": np.array([1, 1, 2]),
          "session_eq": np.array([102.0, 99.96])}
    sr = E.session_returns(ep)
    assert sr == pytest.approx([0.02, 99.96 / 102.0 - 1.0])
    del ep["session_eq"]                                                   # no log: day change = close
    assert E.session_returns(ep) == pytest.approx([103.0 / 100.0 - 1.0, 102.0 / 103.0 - 1.0])
