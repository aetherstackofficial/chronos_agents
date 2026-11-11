"""Speed of ReplayBookEnv: one env (random actions, reset excluded), reset time for
M = 250 sessions of memory, and SubprocVecEnv with N envs.

    python -m training.bench_env            (run from swarm_orchestrator/)
    python -m training.bench_env --envs 8 --steps 4000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:                                   # same import order as tests/conftest.py (Windows DLL quirk)
    import torch  # noqa: F401
except ImportError:
    pass

import numpy as np  # noqa: E402

from training.envs.replay_book_env import ReplayBookEnv  # noqa: E402

MODELS = ("whale_swing", "whale_position", "mm")


def make_env(name: str, seed: int, cfg=None):
    return ReplayBookEnv(name, "train", cfg, seed=seed)


def single_env_sps(name: str, n_steps: int = 1500) -> float:
    cfg = {"sampler": {"length_sessions": {name: [5, 5]}}}      # long enough: no reset inside the timing
    env = make_env(name, 1, cfg)
    env.reset(seed=1)
    t = time.perf_counter()
    done = False
    n = 0
    while n < n_steps:
        _, _, te, tr, _ = env.step(env.action_space.sample())
        n += 1
        if te or tr:
            break
    return n / (time.perf_counter() - t)


def reset_time(name: str, memory: int = 250, n: int = 4) -> float:
    cfg = {"sampler": {"memory_mix": [[1.0, memory, memory]]}}
    env = make_env(name, 2, cfg)
    ts = []
    for i in range(n):
        t = time.perf_counter()
        env.reset(seed=10 + i)
        ts.append(time.perf_counter() - t)
    return float(np.mean(ts))


def vec_sps(name: str, n_envs: int, n_steps: int) -> float:
    from stable_baselines3.common.vec_env import SubprocVecEnv
    venv = SubprocVecEnv([partial(make_env, name, 100 + i) for i in range(n_envs)], start_method="spawn")
    try:
        venv.reset()
        acts = np.stack([venv.action_space.sample() for _ in range(n_envs)])
        for _ in range(20):                                    # warm up the workers
            venv.step(acts)
        t = time.perf_counter()
        for _ in range(n_steps):
            venv.step(np.stack([venv.action_space.sample() for _ in range(n_envs)]))
        dt = time.perf_counter() - t
    finally:
        venv.close()
    return n_envs * n_steps / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500, help="vec-env steps (each steps all envs)")
    args = ap.parse_args()
    for name in MODELS:
        print(f"{name:15s} 1 env: {single_env_sps(name):7.0f} steps/s (reset excluded)   "
              f"reset M=250: {reset_time(name):.2f} s", flush=True)
    for name in MODELS:
        sps = vec_sps(name, args.envs, args.steps)
        print(f"{name:15s} SubprocVecEnv x{args.envs}: {sps:7.0f} steps/s total (auto-resets included)", flush=True)


if __name__ == "__main__":
    main()
