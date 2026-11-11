"""training/pretrain_bc.py + the --init-from warm start of training/train.py, on the stub env of
test_training_scripts (fast, no data needed)."""
import json

import numpy as np
import pytest
import torch
import yaml

from agents.obs_scaler import ObsScaler
from test_training_scripts import SCALERS, StubEnv, small_cfg, stub_factory  # noqa: F401  (fixture)
from training import baselines as B
from training import evaluate as E
from training import pretrain_bc as P
from training import train as T


def _cfg():
    cfg = small_cfg("whale_swing", "ppo", log_std_init=-0.5)
    return cfg


@pytest.fixture(scope="module")
def bc(stub_factory, tmp_path_factory):
    out = tmp_path_factory.mktemp("bc") / "bc_policy.pt"
    info = P.pretrain("whale_swing", "trend", 2, out, _cfg(), epochs=40, seed=1, verbose=False)
    return out, info


def test_collect_labels_are_teacher_actions(stub_factory):
    cfg = _cfg()
    env_cfg = T.load_env_cfg(cfg)
    data = P.collect("whale_swing", B.TrendRule(), env_cfg, 2, seed=3, explore_p=0.5)
    assert data["n"] == 2 * 119 and data["obs_raw"].shape == (data["n"], 93) and data["labels"].shape == (data["n"], 3)
    assert np.all(np.abs(data["labels"]) <= 1.0) and data["n_explore"] > 0
    # label = the rule's action for the observed state: participation / aggressiveness constant 1,
    # target in {-1, 0 (cold start), 1}; exploration only changes the executed action, not the label
    assert set(np.unique(data["labels"][:, 0])) <= {-1.0, 0.0, 1.0}
    assert np.all(data["labels"][:, 1:] == 1.0)


def test_pretrain_saves_and_fits(bc):
    out, info = bc
    assert out.exists() and out.with_suffix(".json").exists()
    saved = json.loads(out.with_suffix(".json").read_text())
    for k in ("samples", "train_mse", "holdout_mse", "train_sign_agreement", "holdout_sign_agreement", "teacher",
              "epochs", "obs_scaler_sha256"):
        assert k in saved
    assert saved["samples"] == 238 and saved["samples_holdout"] > 0
    assert info["train_mse"] < 0.35 and info["train_sign_agreement"] > 0.6
    # a loadable run dir sits next to the policy
    for f in ("model.zip", "config.yaml", "obs_scaler.json", "run_meta.json"):
        assert (out.parent / f).exists(), f
    policy, cfg = E.load_run_policy(out.parent)
    assert cfg["algo"] == "ppo" and E.run_decision_minutes(cfg) == 15
    a = policy.predict(np.zeros(93, dtype=np.float32), None)
    assert a.shape == (3,)


def test_bc_policy_is_closer_to_teacher_than_fresh(bc, stub_factory):
    out, _ = bc
    cfg = _cfg()
    data = P.collect("whale_swing", B.TrendRule(), T.load_env_cfg(cfg), 1, seed=9, explore_p=0.0)
    sc = ObsScaler.load(SCALERS["whale"], "whale")
    x = torch.as_tensor(sc.transform(data["obs_raw"]))
    y = torch.as_tensor(data["labels"])
    fresh = P.build_fresh_ppo(cfg, 93, 3, sc.clip, 5, 0.99)
    warm = P.build_fresh_ppo(cfg, 93, 3, sc.clip, 5, 0.99)
    warm.policy.load_state_dict(torch.load(str(out)), strict=True)
    with torch.no_grad():
        e_fresh = float(((fresh.policy.get_distribution(x).distribution.mean - y) ** 2).mean())
        e_warm = float(((warm.policy.get_distribution(x).distribution.mean - y) ** 2).mean())
    assert e_warm < 0.5 * e_fresh
    # same through the public deterministic predict
    obs = x[40:41].numpy()
    a_w, _ = warm.predict(obs, deterministic=True)
    a_f, _ = fresh.predict(obs, deterministic=True)
    assert np.abs(a_w[0] - y[40].numpy()).sum() < np.abs(a_f[0] - y[40].numpy()).sum()


def test_train_with_init_from(bc, stub_factory, tmp_path):
    out, _ = bc
    run_dir = tmp_path / "ft"
    run_dir.mkdir()
    cfg = small_cfg("whale_swing", "ppo", log_std_init=-0.5, init_log_std=-1.5, steps=300, eval_every=300,
                    checkpoint_every=300, n_envs=1)
    meta = T.train("whale_swing", "ppo", 2, cfg, run_dir, init_from=str(out))
    assert meta["status"] == "finished"
    m = json.loads((run_dir / "run_meta.json").read_text())
    assert m["init_from"] == str(out) and m["init_log_std"] == -1.5 and len(m["init_from_sha256"]) == 64
    assert yaml.safe_load((run_dir / "config.yaml").read_text())["init_from"] == str(out)
    from stable_baselines3 import PPO
    model = PPO.load(str(run_dir / "model.zip"), device="cpu")
    # PPO only moved log_std a little in 300 steps from -1.5 (not from -0.5)
    assert float(model.policy.log_std.mean()) < -1.2


def test_train_without_init_from_unchanged(stub_factory, tmp_path):
    run_dir = tmp_path / "plain"
    run_dir.mkdir()
    cfg = small_cfg("whale_swing", "ppo", steps=256, eval_every=256, checkpoint_every=256, n_envs=1,
                    init_log_std=-1.5)                       # init_log_std alone does nothing
    T.train("whale_swing", "ppo", 2, cfg, run_dir)
    m = json.loads((run_dir / "run_meta.json").read_text())
    assert m["init_from"] is None and "init_log_std" not in m


def test_init_from_rejects_off_policy_and_missing_file(stub_factory, tmp_path):
    with pytest.raises(ValueError):
        T.train("whale_swing", "sac", 1, small_cfg("whale_swing", "sac"), tmp_path, init_from="x.pt")
    run_dir = tmp_path / "missing"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        T.train("whale_swing", "ppo", 1, small_cfg("whale_swing", "ppo", steps=256, n_envs=1), run_dir,
                init_from=str(tmp_path / "nope.pt"))
