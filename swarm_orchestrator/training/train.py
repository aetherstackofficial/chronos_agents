"""Train a v2 RL agent (whale_swing / whale_position / mm) with stable-baselines3.

    python -m training.train --model whale_swing --algo sac --seed 1
        [--config path] [--steps N] [--n-envs N] [--run-name X]
        [--eval-every N] [--checkpoint-every N] [--scaler path.json] [--init-from bc_policy.pt]

``steps`` / ``eval_every`` / ``checkpoint_every`` count env steps = DECISIONS: one env step is
``env.decision_minutes[model]`` minutes (training/configs/env.yaml).  The config ``gamma`` is per
MINUTE; the algorithm and VecNormalize get ``gamma ** decision_minutes`` (also in run_meta.json).

Wrapping (training and evaluation envs alike)::

    SubprocVecEnv(ReplayBookEnv x n_envs) -> VecMonitor -> ObsScalerVecEnv
        -> VecNormalize(norm_obs=False, norm_reward=True) -> [VecFrameStack]

Inputs are scaled by the FIXED, data-fitted ``agents.obs_scaler.ObsScaler``
(``training/configs/obs_scaler_<role>.json``, made by ``python -m training.fit_scaler``; override
with ``--scaler`` or the config key ``obs_scaler``), identical in training, evaluation and live.
VecNormalize only normalises the REWARD.  The scaler file is copied into the run dir
(``obs_scaler.json``) and its sha256 is recorded in ``run_meta.json``.

The eval env is the same on the "validation" split (up to 5 processes, no observation noise) with
``training=False`` and ``norm_reward=False``; EvalCallback copies the reward statistics into it
before each evaluation (SB3 does it when both are VecNormalize).  Output goes to
``training/runs/<model>_<algo>_s<seed>_<timestamp>/``.

The env class can be replaced for tests with the environment variable
``CHRONOS_ENV_FACTORY="module:callable"`` (called as ``f(model_name, split, cfg, seed=...)``).
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

TRAIN_DIR = Path(__file__).resolve().parent
ROOT = TRAIN_DIR.parent                      # swarm_orchestrator/
CONFIG_DIR = TRAIN_DIR / "configs"
RUNS_DIR = TRAIN_DIR / "runs"
MANIFEST = TRAIN_DIR / "data" / "parquet" / "manifest.json"
ENV_FACTORY_VAR = "CHRONOS_ENV_FACTORY"
DEFAULT_FACTORY = "training.envs.replay_book_env:ReplayBookEnv"

MODELS = ("whale_swing", "whale_position", "mm")
ALGOS = ("ppo", "sac", "td3")
CLIP_OBS = 10.0
EVAL_ENVS = 3          # eval workers; 5 ran the machine out of RAM next to 8 train workers
EVAL_NO_NOISE = {"env": {"obs_noise": 0}}     # evaluation never sees observation noise
SCALER_NAME = "obs_scaler.json"               # the scaler copy inside a run / export dir
REQUIRED_KEYS = ("model", "algo", "steps", "n_envs", "eval_every", "eval_episodes",
                 "checkpoint_every", "frame_stack", "gamma", "net_arch", "algo_kwargs",
                 "env_overrides")
ACTION_NAMES = {
    "whale": ("target", "participation", "aggressiveness"),
    "mm": ("bid_offset", "ask_offset", "bid_size", "ask_size"),
}


# --------------------------------------------------------------------------- helpers
def role_of(model_name: str) -> str:
    """'whale' for whale_swing / whale_position, 'mm' for mm."""
    if model_name not in MODELS:
        raise ValueError(f"unknown model {model_name!r}, expected one of {MODELS}")
    return "mm" if model_name == "mm" else "whale"


def deep_merge(base: dict, over: Optional[dict]) -> dict:
    out = copy.deepcopy(base or {})
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(model: str, algo: str, path: Optional[str] = None) -> dict:
    p = Path(path) if path else CONFIG_DIR / f"{model}_{algo}.yaml"
    cfg = load_yaml(p)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"{p}: missing keys {missing}")
    return cfg


def load_env_cfg(cfg: dict, eval_env: bool = False, extra: Optional[dict] = None) -> dict:
    """The env config dict: training/configs/env.yaml + cfg['env_overrides'] (+
    eval_env_overrides + ``env.obs_noise: 0`` for the eval env) (+ ``extra``).  Resolved with
    ``training.envs.config.load_env_config`` (fills ``costs.intraday_rt_bps``); {} + the
    overrides when the env package is not there yet."""
    over = deep_merge(cfg.get("env_overrides") or {}, cfg.get("eval_env_overrides") if eval_env else None)
    if eval_env:
        over = deep_merge(over, EVAL_NO_NOISE)         # evaluation: no observation noise
    over = deep_merge(over, extra)
    try:
        from training.envs.config import load_env_config
        return load_env_config(overrides=over)
    except ImportError:
        env_yaml = CONFIG_DIR / "env.yaml"
        base = load_yaml(env_yaml) if env_yaml.exists() else {}
        return deep_merge(base, over)


def decision_minutes_of(env_cfg: dict, model_name: str) -> int:
    """Minutes per env step (``env.decision_minutes[model]`` of the resolved env config; 1 if absent)."""
    k = ((env_cfg or {}).get("env") or {}).get("decision_minutes") or {}
    k = int(k.get(model_name, 1)) if isinstance(k, dict) else int(k)
    if k < 1:
        raise ValueError("env.decision_minutes must be >= 1")
    return k


def decision_gamma(gamma_per_minute: float, decision_minutes: int) -> float:
    """Per-decision discount of a per-minute gamma: one env step = K minutes, so gamma ** K."""
    return float(gamma_per_minute) ** int(decision_minutes)


def git_commit() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
                           text=True, timeout=20)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def manifest_sha256() -> Optional[str]:
    if not MANIFEST.exists():
        return None
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def library_versions() -> Dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("stable_baselines3", "gymnasium", "torch", "numpy", "pandas", "pyarrow", "yaml"):
        try:
            m = importlib.import_module(mod)
            out[mod] = str(getattr(m, "__version__", "?"))
        except Exception:
            out[mod] = "not installed"
    return out


def env_factory():
    spec = os.environ.get(ENV_FACTORY_VAR) or DEFAULT_FACTORY
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


class EnvMaker:
    """Picklable env constructor (SubprocVecEnv uses spawn on Windows)."""

    def __init__(self, model_name: str, split: str, env_cfg: dict, seed: int) -> None:
        self.model_name, self.split, self.env_cfg, self.seed = model_name, split, env_cfg, int(seed)

    def __call__(self):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        return env_factory()(self.model_name, self.split, self.env_cfg, seed=self.seed)


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_scaler_path(role: str, cfg: Optional[dict] = None, override: Optional[str] = None) -> Path:
    """Scaler file of ``role``: ``override`` (CLI), else ``cfg['obs_scaler']`` (relative paths are
    looked up as given, then in training/configs), else training/configs/obs_scaler_<role>.json."""
    name = override or (cfg or {}).get("obs_scaler")
    if not name:
        return CONFIG_DIR / f"obs_scaler_{role}.json"
    p = Path(name)
    return p if p.is_absolute() or p.exists() else CONFIG_DIR / p


def build_vec_env(model_name: str, split: str, env_cfg: dict, n_envs: int, seed: int,
                  gamma: float, frame_stack: int = 1, training: bool = True,
                  norm_reward: bool = True, subproc: bool = True, seed_base: Optional[int] = None,
                  scaler=None):
    """SubprocVecEnv -> VecMonitor -> ObsScalerVecEnv -> VecNormalize(norm_obs=False) -> [VecFrameStack].

    Without ``scaler`` (legacy) the observations are normalised by VecNormalize's running statistics."""
    from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv, VecFrameStack,
                                                  VecMonitor, VecNormalize)
    from training.wrappers import ObsScalerVecEnv
    base = seed * 100 if seed_base is None else seed_base
    makers = [EnvMaker(model_name, split, env_cfg, base + i) for i in range(n_envs)]
    if subproc and n_envs > 1:
        venv = SubprocVecEnv(makers, start_method="spawn")
    else:
        venv = DummyVecEnv(makers)
    venv = VecMonitor(venv)
    if scaler is not None:
        venv = ObsScalerVecEnv(venv, scaler)
    venv = VecNormalize(venv, training=training, norm_obs=scaler is None, norm_reward=norm_reward,
                        clip_obs=CLIP_OBS, gamma=gamma)
    if frame_stack and frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=int(frame_stack))
    return venv


def find_vecnormalize(venv):
    from stable_baselines3.common.vec_env import VecNormalize
    cur = venv
    while cur is not None:
        if isinstance(cur, VecNormalize):
            return cur
        cur = getattr(cur, "venv", None)
    return None


def build_model(algo: str, cfg: dict, venv, seed: int, tb_dir: Optional[str], steps: int,
                gamma: Optional[float] = None):
    """Create the SB3 model of ``algo`` from the config (algo_kwargs + net_arch + gamma).
    ``gamma`` = the PER-DECISION discount (train() passes gamma_per_minute ** K); default: cfg['gamma']."""
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.noise import NormalActionNoise

    kw = dict(cfg.get("algo_kwargs") or {})
    policy_kwargs = {"net_arch": list(cfg["net_arch"])}
    # PPO's Gaussian policy starts with std = exp(log_std_init) = 1.0 by default, i.e. random
    # targets over the whole [-1, 1] range on every decision: the whale explores by flipping
    # its position, and each flip costs ~9 bps of capital. A smaller start std (config key
    # log_std_init, e.g. -1.0 -> 0.37) explores around the current mean instead.
    if algo == "ppo" and cfg.get("log_std_init") is not None:
        policy_kwargs["log_std_init"] = float(cfg["log_std_init"])
    common = dict(policy="MlpPolicy", env=venv, gamma=float(cfg["gamma"] if gamma is None else gamma), seed=seed,
                  device="cpu", verbose=1, tensorboard_log=tb_dir, policy_kwargs=policy_kwargs)
    if algo == "ppo":
        return PPO(**common, **kw)
    # off-policy: never ask for a bigger buffer than the run can fill; learning_starts
    # may not exceed a quarter of the run (matters only for smoke runs)
    if "buffer_size" in kw:
        kw["buffer_size"] = int(min(kw["buffer_size"], max(steps, 1000)))
    if "learning_starts" in kw:
        kw["learning_starts"] = int(min(kw["learning_starts"], max(steps // 4, 1)))
    if algo == "sac":
        return SAC(**common, **kw)
    if algo == "td3":
        sigma = float(kw.pop("action_noise_sigma", 0.1))
        n_act = int(np.prod(venv.action_space.shape))
        noise = NormalActionNoise(mean=np.zeros(n_act), sigma=sigma * np.ones(n_act))
        return TD3(**common, action_noise=noise, **kw)
    raise ValueError(f"unknown algo {algo!r}")


def load_init_from(model, path, init_log_std: Optional[float] = None) -> dict:
    """Warm start (PPO): load a ``training.pretrain_bc`` policy state_dict (strict), then optionally set
    the Gaussian policy's log_std to ``init_log_std`` (a narrow start std keeps the warm-started policy
    from flipping its position at random).  Returns the run_meta entries."""
    import torch
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"init_from {p}: no such file")
    model.policy.load_state_dict(torch.load(str(p), map_location="cpu"), strict=True)
    out = {"init_from": str(p), "init_from_sha256": sha256_file(p)}
    if init_log_std is not None:
        with torch.no_grad():
            model.policy.log_std.fill_(float(init_log_std))
        out["init_log_std"] = float(init_log_std)
    return out


def make_run_dir(model: str, algo: str, seed: int, run_name: Optional[str]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or f"{model}_{algo}_s{seed}_{stamp}"
    d = RUNS_DIR / name
    if d.exists() and any(d.iterdir()):
        d = RUNS_DIR / f"{name}_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


# --------------------------------------------------------------------------- main
def train(model_name: str, algo: str, seed: int, cfg: dict, run_dir: Path,
          steps: Optional[int] = None, n_envs: Optional[int] = None,
          eval_every: Optional[int] = None, checkpoint_every: Optional[int] = None,
          scaler_path: Optional[str] = None, init_from: Optional[str] = None) -> dict:
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from agents.features_v2 import OBS_SPEC, spec_hash
    from agents.obs_scaler import ObsScaler
    from training.callbacks import (FixedSeedEvalCallback, SaveVecNormalizeOnBest,
                                    TrainingInsightsCallback)

    cfg = copy.deepcopy(cfg)
    if steps is not None:
        cfg["steps"] = int(steps)
    if n_envs is not None:
        cfg["n_envs"] = int(n_envs)
    if eval_every is not None:
        cfg["eval_every"] = int(eval_every)
    if checkpoint_every is not None:
        cfg["checkpoint_every"] = int(checkpoint_every)
    cfg["model"], cfg["algo"] = model_name, algo
    if init_from is not None:
        cfg["init_from"] = str(init_from)
    init_from = cfg.get("init_from") or None            # CLI flag / config key: BC warm start (PPO only)
    if init_from and algo != "ppo":
        raise ValueError("init_from (behaviour-cloning warm start) is supported for algo ppo only")
    role = role_of(model_name)
    total = int(cfg["steps"])
    n_envs = int(cfg["n_envs"])
    frame_stack = int(cfg.get("frame_stack", 1))
    train_env_cfg = load_env_cfg(cfg)
    eval_env_cfg = load_env_cfg(cfg, eval_env=True)
    # the run config holds PER-MINUTE gammas; one env step = K minutes, so the algorithm and
    # VecNormalize discount per decision with gamma ** K
    gamma_minute = float(cfg["gamma"])
    dec_minutes = decision_minutes_of(train_env_cfg, model_name)
    if decision_minutes_of(eval_env_cfg, model_name) != dec_minutes:
        raise ValueError("eval_env_overrides may not change env.decision_minutes")
    gamma = decision_gamma(gamma_minute, dec_minutes)

    # fixed observation scaler: load (refuses mismatching input names), copy into the run dir
    src_scaler = resolve_scaler_path(role, cfg, scaler_path)
    if not src_scaler.exists():
        raise FileNotFoundError(f"{src_scaler}: no observation scaler; run "
                                f"`python -m training.fit_scaler --role {role}` first")
    ObsScaler.load(src_scaler, role)
    shutil.copy2(src_scaler, run_dir / SCALER_NAME)
    scaler = ObsScaler.load(run_dir / SCALER_NAME, role)
    cfg["obs_scaler"] = str(src_scaler)

    resolved = {**cfg, "seed": seed, "role": role, "decision_minutes": dec_minutes,
                "gamma_per_minute": gamma_minute, "gamma_per_decision": gamma, "env_cfg": train_env_cfg,
                "eval_env_cfg": eval_env_cfg, "run_dir": str(run_dir)}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    meta: Dict[str, Any] = {
        "model": model_name, "algo": algo, "seed": seed, "role": role,
        "git_commit": git_commit(), "libraries": library_versions(),
        "dataset_manifest_sha256": manifest_sha256(), "spec_hash": spec_hash(role),
        "obs_names": list(OBS_SPEC[role]), "frame_stack": frame_stack,
        "obs_scaler_source": str(src_scaler), "obs_scaler_file": SCALER_NAME,
        "obs_scaler_sha256": sha256_file(run_dir / SCALER_NAME),
        "obs_scaler_n_samples": scaler.meta.get("n_samples"), "obs_scaler_fit_date": scaler.meta.get("fit_date"),
        "env_factory": os.environ.get(ENV_FACTORY_VAR) or DEFAULT_FACTORY,
        "n_envs": n_envs, "total_timesteps": total,
        "init_from": str(init_from) if init_from else None,
        "decision_minutes": dec_minutes, "gamma_per_minute": gamma_minute, "gamma_per_decision": gamma,
        "start_time": dt.datetime.now().isoformat(timespec="seconds"), "end_time": None,
        "steps_per_second": None, "status": "running",
    }
    write_json(run_dir / "run_meta.json", meta)

    venv = build_vec_env(model_name, "train", train_env_cfg, n_envs, seed, gamma, frame_stack,
                         training=True, norm_reward=True, subproc=True, scaler=scaler)
    # eval env: up to EVAL_ENVS processes; FixedSeedEvalCallback re-seeds env i with eval_seed + i
    # before every evaluation (SubprocVecEnv.seed -> reset(seed=...)), so each evaluation replays
    # the same episodes and the curves are comparable.
    # The eval envs run INSIDE the main process (DummyVecEnv) unless CHRONOS_EVAL_SUBPROC=1:
    # every spawned worker imports torch + SB3 (~460 MB measured), and with 2-3 trainings in
    # parallel the 3 eval workers per run were what ran the machine out of RAM. Eval is ~5% of
    # the wall time, so running it sequentially costs little.
    n_eval_envs = max(1, min(EVAL_ENVS, int(cfg["eval_episodes"])))
    eval_subproc = n_eval_envs > 1 and os.environ.get("CHRONOS_EVAL_SUBPROC", "0") == "1"
    eval_env = build_vec_env(model_name, "validation", eval_env_cfg, n_eval_envs, seed, gamma, frame_stack,
                             training=False, norm_reward=False, subproc=eval_subproc,
                             seed_base=seed * 100 + 99, scaler=scaler)
    vn_train = find_vecnormalize(venv)
    model = build_model(algo, cfg, venv, seed, str(run_dir / "tb"), total, gamma=gamma)
    if init_from:
        meta.update(load_init_from(model, init_from, cfg.get("init_log_std")))
        write_json(run_dir / "run_meta.json", meta)

    ckpt = CheckpointCallback(save_freq=max(int(cfg["checkpoint_every"]) // n_envs, 1),
                              save_path=str(run_dir / "checkpoints"), name_prefix="ckpt",
                              save_vecnormalize=True)
    best = SaveVecNormalizeOnBest(vn_train, str(run_dir / "best" / "best_vecnormalize.pkl"))
    evalcb = FixedSeedEvalCallback(
        eval_env, callback_on_new_best=best, n_eval_episodes=int(cfg["eval_episodes"]),
        eval_freq=max(int(cfg["eval_every"]) // n_envs, 1), deterministic=True,
        best_model_save_path=str(run_dir / "best"), log_path=str(run_dir / "eval"),
        eval_seed=seed * 100 + 99)
    insights = TrainingInsightsCallback(OBS_SPEC[role], ACTION_NAMES[role], log_every=1000,
                                        clip_obs=scaler.clip, print_clip=True,
                                        decision_minutes=dec_minutes)

    t0 = time.time()
    try:
        model.learn(total_timesteps=total, callback=CallbackList([ckpt, evalcb, insights]),
                    progress_bar=False)
        status = "finished"
    except KeyboardInterrupt:
        status = "interrupted"
    elapsed = time.time() - t0
    model.save(str(run_dir / "model.zip"))
    vn_train.save(str(run_dir / "vecnormalize.pkl"))
    done = int(model.num_timesteps)
    meta.update({"trained_minutes": done * dec_minutes,
                 "end_time": dt.datetime.now().isoformat(timespec="seconds"), "status": status,
                 "trained_timesteps": done, "wall_seconds": round(elapsed, 2),
                 "steps_per_second": round(done / elapsed, 2) if elapsed > 0 else None,
                 "minutes_per_second": round(done * dec_minutes / elapsed, 2) if elapsed > 0 else None,
                 "eval_seconds": round(evalcb.eval_seconds, 2),
                 "steps_per_second_excl_eval": (round(done / (elapsed - evalcb.eval_seconds), 2)
                                                if elapsed - evalcb.eval_seconds > 0 else None)})
    write_json(run_dir / "run_meta.json", meta)
    venv.close()
    eval_env.close()
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="training.train", description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--algo", required=True, choices=ALGOS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--config", help="config yaml (default training/configs/<model>_<algo>.yaml)")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--n-envs", type=int)
    ap.add_argument("--run-name")
    ap.add_argument("--eval-every", type=int)
    ap.add_argument("--checkpoint-every", type=int)
    ap.add_argument("--scaler", help="observation scaler json (default training/configs/obs_scaler_<role>.json)")
    ap.add_argument("--init-from", help="PPO only: bc_policy.pt from training.pretrain_bc (warm start; config key "
                                        "init_log_std then sets the start log_std)")
    a = ap.parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    cfg = load_config(a.model, a.algo, a.config)
    run_dir = make_run_dir(a.model, a.algo, a.seed, a.run_name)
    print(f"run dir: {run_dir}", flush=True)
    meta = train(a.model, a.algo, a.seed, cfg, run_dir, steps=a.steps, n_envs=a.n_envs,
                 eval_every=a.eval_every, checkpoint_every=a.checkpoint_every, scaler_path=a.scaler,
                 init_from=a.init_from)
    print(f"{meta['status']}: {meta.get('trained_timesteps')} steps (decisions; x{meta.get('decision_minutes')} "
          f"min) in {meta.get('wall_seconds')} s "
          f"({meta.get('steps_per_second')} steps/s = {meta.get('minutes_per_second')} minutes/s, "
          f"{meta.get('steps_per_second_excl_eval')} steps/s excl. "
          f"{meta.get('eval_seconds')} s of eval) -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
