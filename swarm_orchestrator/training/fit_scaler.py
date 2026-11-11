"""Fit the fixed observation scaler (agents/obs_scaler.py) of a role.

    python -m training.fit_scaler --role whale|mm [--episodes N] [--seed S]
        [--samples 200000] [--workers 8] [--subsample 5] [--out path.json]

Random-action episodes of ``ReplayBookEnv`` on the TRAIN split (role whale: whale_swing and
whale_position alternately; mm: mm) with the normal memory-length mix and the observation noise
turned off (``env.obs_noise: 0``); every ``--subsample``-th step is kept until ``--samples``
observations are collected (``--episodes N`` instead runs exactly N episodes in total).  The
episodes are spread over ``--workers`` processes, each with its own env and seed.

Per fitted input (everything except the pass-through inputs, see ``agents.obs_scaler``):

* center = median, scale = max(IQR / 1.349, std_floor), std_floor = max(1e-6, 1e-3 x median|x|);
* gated inputs (more than 50% of the samples exactly 0, e.g. a memory feature while its window is
  unfilled, or ``drawdown`` / ``open_pnl`` while flat): center and scale come from the NON-ZERO
  samples only, so 0 maps to (0 - center) / scale;
* if more than 1% of the samples would sit on the +/-clip bound the scale is widened: first to the
  0.5% .. 99.5% quantile range / (2 x 2.576), then, if that is not enough, so that 99.2% of the
  samples lie inside 95% of the clip.

Writes ``training/configs/obs_scaler_<role>.json`` and prints a per-input table.
"""
from __future__ import annotations

import argparse
import datetime as dt
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:                                   # same import order as tests/conftest.py (Windows DLL quirk)
    import torch  # noqa: F401
except ImportError:
    pass

import numpy as np  # noqa: E402

from agents.obs_scaler import DEFAULT_CLIP, ObsScaler, is_passthrough  # noqa: E402

CONFIG_DIR = os.path.join(ROOT, "training", "configs")
ROLE_MODELS = {"whale": ("whale_swing", "whale_position"), "mm": ("mm",)}
CLIP_SHARE_TARGET = 0.01
GATED_ZERO_SHARE = 0.5
MIN_NONZERO = 50
Z995 = 2.576


def default_path(role: str) -> str:
    return os.path.join(CONFIG_DIR, f"obs_scaler_{role}.json")


# --------------------------------------------------------------------------- fitting maths
def _clip_share(x: np.ndarray, c: float, s: float, clip: float) -> float:
    return float((np.abs(x - c) / s >= clip - 1e-9).mean())


def fit_input(x: np.ndarray, clip: float = DEFAULT_CLIP) -> Tuple[float, float, dict]:
    """(center, scale, report) of one input, rules in the module docstring."""
    x = np.asarray(x, dtype=np.float64)
    zero_share = float((x == 0.0).mean())
    gated = zero_share > GATED_ZERO_SHARE
    xs = x
    if gated:
        nz = x[x != 0.0]
        if nz.size >= MIN_NONZERO:
            xs = nz
        else:
            gated = False                       # (almost) always 0: nothing to fit on
    c = float(np.median(xs))
    q25, q75 = np.percentile(xs, [25, 75])
    floor = max(1e-6, 1e-3 * float(np.median(np.abs(xs))))
    s = max(float(q75 - q25) / 1.349, floor)
    method = "iqr"
    share0 = _clip_share(x, c, s, clip)
    share = share0
    if share > CLIP_SHARE_TARGET:
        lo, hi = np.percentile(xs, [0.5, 99.5])
        s = max(s, float(hi - lo) / (2.0 * Z995), floor)
        method = "q005_q995_range"
        share = _clip_share(x, c, s, clip)
    if share > CLIP_SHARE_TARGET:
        q = float(np.quantile(np.abs(x - c), 0.992))
        s = max(s, q / (0.95 * clip), floor)
        method = "q992_abs"
        share = _clip_share(x, c, s, clip)
    rep = {"center": c, "scale": s, "method": method, "gated": bool(gated), "zero_share": zero_share,
           "clip_share_iqr": share0, "clip_share": share, "n_nonzero_used": int(xs.size)}
    return c, s, rep


def fit_scaler_from_samples(names: Sequence[str], X: np.ndarray, clip: float = DEFAULT_CLIP,
                            ) -> Tuple[ObsScaler, Dict[str, dict]]:
    """Fit an ``ObsScaler`` on the (n_samples, n_inputs) matrix ``X``; also the per-input report
    (fitted inputs only)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError(f"X must be (n, {len(names)}), got {X.shape}")
    center = np.zeros(len(names))
    scale = np.ones(len(names))
    report: Dict[str, dict] = {}
    for j, n in enumerate(names):
        if is_passthrough(n):
            continue
        center[j], scale[j], report[n] = fit_input(X[:, j], clip)
    meta = {"passthrough": [n for n in names if is_passthrough(n)],
            "fit": {k: {kk: (round(vv, 8) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                    for k, v in report.items()}}
    return ObsScaler(names, center, scale, clip, meta), report


# --------------------------------------------------------------------------- data collection
def _env_overrides() -> dict:
    return {"env": {"obs_noise": 0}}


def collect(role: str, seed: int, max_samples: Optional[int], max_episodes: Optional[int],
            subsample: int = 5, split: str = "train", log=None) -> np.ndarray:
    """Random-action episodes in this process; returns the kept observations (n, n_inputs) float32."""
    from training.envs.config import load_env_config
    from training.envs.replay_book_env import ReplayBookEnv
    cfg = load_env_config(overrides=_env_overrides())
    models = ROLE_MODELS[role]
    envs = [ReplayBookEnv(m, split, cfg, seed=seed * 10 + k) for k, m in enumerate(models)]
    rng = np.random.default_rng(seed)
    n_act = envs[0].action_space.shape[0]
    rows: List[np.ndarray] = []
    kept = 0
    ep = 0
    t0 = time.time()
    while True:
        if max_episodes is not None and ep >= max_episodes:
            break
        if max_samples is not None and kept >= max_samples:
            break
        env = envs[ep % len(envs)]
        obs, _ = env.reset()
        step, done = 0, False
        while not done:
            if step % subsample == 0:
                rows.append(np.asarray(obs, dtype=np.float32))
                kept += 1
            obs, _, term, trunc, _ = env.step(rng.uniform(-1.0, 1.0, n_act).astype(np.float32))
            done = bool(term or trunc)
            step += 1
        ep += 1
        if log:
            log(f"  [seed {seed}] episode {ep}: {kept} samples, {time.time() - t0:.0f} s")
    return np.asarray(rows, dtype=np.float32).reshape(len(rows), -1)


def _worker(args) -> np.ndarray:
    role, seed, max_samples, max_episodes, subsample = args
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    return collect(role, seed, max_samples, max_episodes, subsample,
                   log=lambda s: print(s, flush=True))


def collect_parallel(role: str, seed: int, samples: int, episodes: Optional[int], workers: int,
                     subsample: int) -> np.ndarray:
    workers = max(1, int(workers))
    if episodes is not None:
        base, extra = divmod(int(episodes), workers)
        tasks = [(role, seed * 1000 + i, None, base + (1 if i < extra else 0), subsample)
                 for i in range(workers) if base + (1 if i < extra else 0) > 0]
    else:
        per = -(-int(samples) // workers)
        tasks = [(role, seed * 1000 + i, per, None, subsample) for i in range(workers)]
    if len(tasks) == 1:
        return _worker(tasks[0])
    with mp.get_context("spawn").Pool(len(tasks)) as pool:
        parts = pool.map(_worker, tasks, chunksize=1)
    return np.concatenate(parts, axis=0)


# --------------------------------------------------------------------------- report
def format_report(names: Sequence[str], report: Dict[str, dict], clip: float) -> str:
    lines = [f"{'input':32s} {'center':>12s} {'scale':>12s} {'zeros':>7s} {'@clip':>7s}  note"]
    for n in names:
        r = report.get(n)
        if r is None:
            continue
        note = []
        if r["gated"]:
            note.append(f"gated(nonzero n={r['n_nonzero_used']})")
        if r["method"] != "iqr":
            note.append(f"widened:{r['method']} (iqr clip share {r['clip_share_iqr']:.1%})")
        lines.append(f"{n:32s} {r['center']:12.5g} {r['scale']:12.5g} {r['zero_share']:7.1%} "
                     f"{r['clip_share']:7.2%}  {' '.join(note)}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="training.fit_scaler", description=__doc__.split("\n")[0])
    ap.add_argument("--role", required=True, choices=sorted(ROLE_MODELS))
    ap.add_argument("--episodes", type=int, help="run exactly N episodes in total (default: until --samples)")
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--subsample", type=int, default=5, help="keep every k-th step")
    ap.add_argument("--out", help=f"output json (default {default_path('<role>')})")
    a = ap.parse_args(argv)
    from agents.features_v2 import OBS_SPEC
    names = list(OBS_SPEC[a.role])
    t0 = time.time()
    X = collect_parallel(a.role, a.seed, a.samples, a.episodes, a.workers, a.subsample)
    t_collect = time.time() - t0
    print(f"collected {X.shape[0]} observations x {X.shape[1]} inputs in {t_collect:.0f} s", flush=True)
    sc, report = fit_scaler_from_samples(names, X)
    out = a.out or default_path(a.role)
    sc.save(out, role=a.role, n_samples=X.shape[0], fit_date=dt.datetime.now().isoformat(timespec="seconds"),
            extra={"seed": a.seed, "episodes": a.episodes, "subsample": a.subsample,
                   "models": list(ROLE_MODELS[a.role]), "split": "train", "obs_noise": 0.0})
    print(format_report(names, report, sc.clip))
    over = [n for n, r in report.items() if r["clip_share"] > CLIP_SHARE_TARGET]
    print(f"inputs above {CLIP_SHARE_TARGET:.0%} at the clip: {over or 'none'}")
    print(f"wrote {out} (total {time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
