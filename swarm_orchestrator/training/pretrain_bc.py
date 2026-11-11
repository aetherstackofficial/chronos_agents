"""Behaviour-cloning warm start for a PPO policy: imitate a rule baseline, then fine-tune with PPO.

    python -m training.pretrain_bc --model whale_swing --teacher trend --episodes 60
        [--out training/runs/bc_<model>_<teacher>/bc_policy.pt] [--config path] [--epochs 20]
        [--scaler path.json] [--seed 1] [--explore-p 0.2] [--no-run]

1. The teacher (``training.baselines``: trend | buyhold | as) plays ``--episodes`` episodes of the TRAIN
   split on the same env ``training.train`` would build (``load_env_cfg(cfg)``: env.yaml +
   ``env_overrides``, training sampler, observation noise ON).  At every decision the RAW observation
   and the teacher action are stored.  With probability ``--explore-p`` (0.2) the EXECUTED action is the
   teacher action plus noise or a random action (so the data covers states the teacher alone would not
   reach); the label is always the teacher's action for the observed state.
2. The observations go through the ObsScaler exactly as in training (default
   training/configs/obs_scaler_<role>.json; VecNormalize has ``norm_obs=False`` with a scaler and
   frame_stack is 1).
3. A fresh SB3 PPO with the architecture ``train.build_model`` builds (net_arch, log_std_init, spaces)
   is fitted: MSE between the policy's deterministic mean action and the teacher action, on the actor
   only (policy MLP + action_net; the critic is NOT trained).  Adam lr 1e-3, batch 256.
4. ``<out>`` = ``torch.save(model.policy.state_dict())``, ``<out>`` with suffix .json (bc_policy.json) = settings + metrics.  Unless
   ``--no-run`` the directory of ``<out>`` also becomes a loadable run (model.zip, config.yaml,
   obs_scaler.json, run_meta.json), so ``python -m training.evaluate --run <dir>`` evaluates the BC
   policy alone, and ``python -m training.train ... --init-from <out>`` fine-tunes it with PPO.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

from training import train as T

DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 256
EXPLORE_P = 0.2
EXPLORE_NOISE = 0.3          # std of the noise added to the teacher action in an exploration step


# --------------------------------------------------------------------------- data collection
def collect(model_name: str, teacher, env_cfg: dict, episodes: int, seed: int,
            explore_p: float = EXPLORE_P, explore_noise: float = EXPLORE_NOISE,
            max_steps: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Roll ``teacher`` on the train split; return raw observations, teacher labels, episode ids."""
    env = T.env_factory()(model_name, "train", env_cfg, seed=seed)
    rng = np.random.default_rng(seed + 12345)
    obs_l, lab_l, ep_l = [], [], []
    n_explore = 0
    try:
        for ep in range(int(episodes)):
            obs, _ = env.reset()
            teacher.reset()
            done, t = False, 0
            while not done:
                obs = np.asarray(obs, dtype=np.float32)
                label = np.clip(np.asarray(teacher.predict(obs, env), dtype=np.float32).reshape(-1), -1.0, 1.0)
                obs_l.append(obs)
                lab_l.append(label)
                ep_l.append(ep)
                act = label
                if explore_p > 0 and rng.random() < explore_p:
                    n_explore += 1
                    if rng.random() < 0.5:
                        act = np.clip(label + rng.normal(0.0, explore_noise, label.shape), -1.0, 1.0)
                    else:
                        act = rng.uniform(-1.0, 1.0, label.shape)
                obs, _, term, trunc, _ = env.step(np.asarray(act, dtype=np.float32))
                t += 1
                done = bool(term or trunc) or (max_steps is not None and t >= max_steps)
    finally:
        env.close()
    n = len(obs_l)
    return {"obs_raw": np.asarray(obs_l, dtype=np.float32), "labels": np.asarray(lab_l, dtype=np.float32),
            "episode": np.asarray(ep_l, dtype=np.int64), "n_explore": n_explore, "n": n}


# --------------------------------------------------------------------------- model / fit
class _SpacesEnv:
    """Minimal gym env that only carries the spaces (PPO needs an env to be built)."""

    @staticmethod
    def make(obs_dim: int, act_dim: int, clip: float):
        import gymnasium as gym
        from gymnasium import spaces

        class E(gym.Env):
            observation_space = spaces.Box(-clip, clip, (obs_dim,), np.float32)
            action_space = spaces.Box(-1.0, 1.0, (act_dim,), np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(obs_dim, np.float32), {}

            def step(self, action):
                return np.zeros(obs_dim, np.float32), 0.0, True, False, {}
        return E()


def build_fresh_ppo(cfg: dict, obs_dim: int, act_dim: int, clip: float, seed: int, gamma: float):
    """A PPO with exactly the policy ``train.build_model('ppo', cfg, ...)`` builds."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    venv = DummyVecEnv([lambda: _SpacesEnv.make(obs_dim, act_dim, clip)])
    cfg = dict(cfg)
    cfg["algo_kwargs"] = {k: v for k, v in (cfg.get("algo_kwargs") or {}).items()}
    return T.build_model("ppo", cfg, venv, seed, None, 1, gamma=gamma)


def _mean_action(policy, x):
    return policy.get_distribution(x).distribution.mean


def _eval(policy, x, y) -> dict:
    import torch
    with torch.no_grad():
        pred = _mean_action(policy, x)
        err = (pred - y) ** 2
        t = y[:, 0]
        nz = t.abs() > 1e-6
        sign = float((torch.sign(pred[:, 0][nz]) == torch.sign(t[nz])).float().mean()) if nz.any() else float("nan")
        return {"mse": float(err.mean()), "mse_per_output": [float(v) for v in err.mean(dim=0)],
                "sign_agreement": sign, "n": int(len(y)), "n_sign": int(nz.sum())}


def fit_actor(model, x_tr: np.ndarray, y_tr: np.ndarray, x_ho: Optional[np.ndarray], y_ho: Optional[np.ndarray],
              epochs: int = DEFAULT_EPOCHS, lr: float = DEFAULT_LR, batch: int = DEFAULT_BATCH,
              seed: int = 0, verbose: bool = True) -> dict:
    """MSE-fit the actor (policy MLP + action_net) of ``model.policy`` on (scaled obs, teacher action)."""
    import torch
    pol = model.policy
    torch.manual_seed(seed)
    params = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    xt, yt = torch.as_tensor(x_tr), torch.as_tensor(y_tr)
    xh = torch.as_tensor(x_ho) if x_ho is not None and len(x_ho) else None
    yh = torch.as_tensor(y_ho) if xh is not None else None
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(int(epochs)):
        order = rng.permutation(len(xt))
        tot, nb = 0.0, 0
        for i in range(0, len(order), batch):
            idx = torch.as_tensor(order[i:i + batch])
            loss = ((_mean_action(pol, xt[idx]) - yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            nb += len(idx)
        row = {"epoch": ep + 1, "train_loss": tot / max(nb, 1)}
        if xh is not None:
            row["holdout_mse"] = _eval(pol, xh, yh)["mse"]
        hist.append(row)
        if verbose and ((ep + 1) % 5 == 0 or ep == 0 or ep + 1 == epochs):
            print("  epoch %2d  train mse %.5f%s" % (ep + 1, row["train_loss"],
                  ("  holdout mse %.5f" % row["holdout_mse"]) if "holdout_mse" in row else ""), flush=True)
    out = {"train": _eval(pol, xt, yt), "history": hist}
    out["holdout"] = _eval(pol, xh, yh) if xh is not None else None
    return out


def split_holdout(data: dict, frac: float, seed: int):
    """Boolean mask of the training rows: by episode when there are >= 5 episodes, else by row."""
    n = data["n"]
    rng = np.random.default_rng(seed + 7)
    eps = np.unique(data["episode"])
    if len(eps) >= 5:
        ho_eps = rng.choice(eps, size=max(1, int(round(frac * len(eps)))), replace=False)
        return ~np.isin(data["episode"], ho_eps)
    m = np.ones(n, dtype=bool)
    if n >= 10:
        m[rng.choice(n, size=max(1, int(round(frac * n))), replace=False)] = False
    return m


# --------------------------------------------------------------------------- run dir
def write_run_dir(run_dir: Path, model, cfg: dict, model_name: str, scaler_src: Path, scaler, env_cfg: dict,
                  eval_env_cfg: dict, dec_minutes: int, gamma_minute: float, gamma: float, seed: int,
                  bc_meta: dict) -> None:
    """Make ``run_dir`` loadable by ``training.evaluate`` (same files/keys as ``train.train``)."""
    import shutil
    from agents.features_v2 import OBS_SPEC, spec_hash
    role = T.role_of(model_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scaler_src, run_dir / T.SCALER_NAME)
    cfg = dict(cfg)
    cfg["obs_scaler"] = str(scaler_src)
    resolved = {**cfg, "seed": seed, "role": role, "decision_minutes": dec_minutes,
                "gamma_per_minute": gamma_minute, "gamma_per_decision": gamma, "env_cfg": env_cfg,
                "eval_env_cfg": eval_env_cfg, "run_dir": str(run_dir), "init_from": None,
                "bc_only": True}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    model.save(str(run_dir / "model.zip"))
    meta = {"model": model_name, "algo": "ppo", "seed": seed, "role": role, "git_commit": T.git_commit(),
            "libraries": T.library_versions(), "spec_hash": spec_hash(role), "obs_names": list(OBS_SPEC[role]),
            "frame_stack": 1, "obs_scaler_source": str(scaler_src), "obs_scaler_file": T.SCALER_NAME,
            "obs_scaler_sha256": T.sha256_file(run_dir / T.SCALER_NAME),
            "decision_minutes": dec_minutes, "gamma_per_minute": gamma_minute, "gamma_per_decision": gamma,
            "status": "bc_only", "bc": bc_meta, "end_time": dt.datetime.now().isoformat(timespec="seconds")}
    T.write_json(run_dir / "run_meta.json", meta)


# --------------------------------------------------------------------------- main
def pretrain(model_name: str, teacher_name: str, episodes: int, out: Path, cfg: dict,
             epochs: int = DEFAULT_EPOCHS, scaler_path: Optional[str] = None, seed: int = 1,
             explore_p: float = EXPLORE_P, holdout_frac: float = 0.1, write_run: bool = True,
             max_steps: Optional[int] = None, verbose: bool = True) -> dict:
    import torch
    from agents.obs_scaler import ObsScaler
    from training.baselines import make_baseline
    cfg = dict(cfg)
    cfg["model"], cfg["algo"] = model_name, "ppo"
    role = T.role_of(model_name)
    if int(cfg.get("frame_stack", 1)) != 1:
        raise ValueError("BC warm start supports frame_stack 1 only")
    env_cfg = T.load_env_cfg(cfg)
    eval_env_cfg = T.load_env_cfg(cfg, eval_env=True)
    dec_minutes = T.decision_minutes_of(env_cfg, model_name)
    gamma_minute = float(cfg["gamma"])
    gamma = T.decision_gamma(gamma_minute, dec_minutes)
    src_scaler = T.resolve_scaler_path(role, cfg, scaler_path)
    if not src_scaler.exists():
        raise FileNotFoundError(f"{src_scaler}: no observation scaler")
    scaler = ObsScaler.load(src_scaler, role)
    teacher = make_baseline(model_name, teacher_name)

    t0 = time.time()
    data = collect(model_name, teacher, env_cfg, episodes, seed, explore_p=explore_p, max_steps=max_steps)
    t_collect = time.time() - t0
    if data["n"] < 2:
        raise RuntimeError("no samples collected")
    x = scaler.transform(data["obs_raw"]).astype(np.float32)        # exactly what the PPO policy will see
    y = data["labels"]
    tr = split_holdout(data, holdout_frac, seed)
    if verbose:
        print(f"collected {data['n']} decisions in {t_collect:.1f}s "
              f"({episodes} episodes, {data['n_explore']} exploration steps); "
              f"train {int(tr.sum())} / holdout {int((~tr).sum())}", flush=True)

    model = build_fresh_ppo(cfg, x.shape[1], y.shape[1], scaler.clip, seed, gamma)
    res = fit_actor(model, x[tr], y[tr], x[~tr] if (~tr).any() else None, y[~tr] if (~tr).any() else None,
                    epochs=epochs, seed=seed, verbose=verbose)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.policy.state_dict(), str(out))
    ho = res["holdout"]
    info = {"model": model_name, "teacher": teacher.name, "episodes": int(episodes), "seed": seed,
            "samples": data["n"], "samples_train": int(tr.sum()), "samples_holdout": int((~tr).sum()),
            "explore_p": explore_p, "explore_noise": EXPLORE_NOISE, "n_explore_steps": data["n_explore"],
            "epochs": epochs, "lr": DEFAULT_LR, "batch_size": DEFAULT_BATCH,
            "net_arch": list(cfg["net_arch"]), "log_std_init": cfg.get("log_std_init"),
            "decision_minutes": dec_minutes, "obs_scaler": str(src_scaler), "obs_scaler_sha256": T.sha256_file(src_scaler),
            "train_mse": res["train"]["mse"], "train_mse_per_output": res["train"]["mse_per_output"],
            "train_sign_agreement": res["train"]["sign_agreement"],
            "holdout_mse": ho["mse"] if ho else None,
            "holdout_mse_per_output": ho["mse_per_output"] if ho else None,
            "holdout_sign_agreement": ho["sign_agreement"] if ho else None,
            "label_mean": [float(v) for v in y.mean(axis=0)], "label_std": [float(v) for v in y.std(axis=0)],
            "collect_seconds": round(t_collect, 1), "created": dt.datetime.now().isoformat(timespec="seconds")}
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(info, indent=2, default=T._json_default), encoding="utf-8")
    if write_run:
        write_run_dir(out.parent, model, cfg, model_name, src_scaler, scaler, env_cfg, eval_env_cfg, dec_minutes,
                      gamma_minute, gamma, seed, info)
    info["policy_path"] = str(out)
    info["settings_path"] = str(meta_path)
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="training.pretrain_bc", description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=T.MODELS)
    ap.add_argument("--teacher", required=True, help="trend | buyhold (whale), as (mm)")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--out", help="bc_policy.pt (default training/runs/bc_<model>_<teacher>/bc_policy.pt)")
    ap.add_argument("--config", help="PPO config yaml (default training/configs/<model>_ppo.yaml)")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--scaler", help="observation scaler json (default training/configs/obs_scaler_<role>.json)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--explore-p", type=float, default=EXPLORE_P)
    ap.add_argument("--no-run", action="store_true", help="do not write model.zip / config.yaml next to --out")
    a = ap.parse_args(argv)
    if str(T.ROOT) not in sys.path:
        sys.path.insert(0, str(T.ROOT))
    cfg = T.load_config(a.model, "ppo", a.config)
    out = Path(a.out) if a.out else T.RUNS_DIR / f"bc_{a.model}_{a.teacher}" / "bc_policy.pt"
    info = pretrain(a.model, a.teacher, a.episodes, out, cfg, epochs=a.epochs, scaler_path=a.scaler,
                    seed=a.seed, explore_p=a.explore_p, write_run=not a.no_run)
    ho = info["holdout_mse"]
    print(f"saved {info['policy_path']}  (+ {info['settings_path']})")
    print(f"samples {info['samples']}  train mse {info['train_mse']:.5f}  "
          f"holdout mse {('%.5f' % ho) if ho is not None else 'n/a'}  "
          f"sign agreement train {info['train_sign_agreement']:.3f}  holdout {info['holdout_sign_agreement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
