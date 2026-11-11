"""VecEnv wrappers of the training pipeline.

``ObsScalerVecEnv`` applies the fixed ``agents.obs_scaler.ObsScaler`` to the observations of the
wrapped vec env (also to ``info["terminal_observation"]``, which SAC / TD3 use to bootstrap
truncated episodes) and keeps the last RAW observation in ``raw_obs`` for the callbacks.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from agents.obs_scaler import ObsScaler


class ObsScalerVecEnv(VecEnvWrapper):
    def __init__(self, venv: VecEnv, scaler: ObsScaler) -> None:
        shape = venv.observation_space.shape
        if shape is None or shape[-1] != scaler.n_inputs:
            raise ValueError(f"env observations have shape {shape}, scaler has {scaler.n_inputs} inputs")
        space = spaces.Box(-scaler.clip, scaler.clip, shape, np.float32)
        super().__init__(venv, observation_space=space)
        self.scaler = scaler
        self.raw_obs: Optional[np.ndarray] = None       # last raw observation (the reset one after a done)

    def reset(self):
        obs = self.venv.reset()
        self.raw_obs = np.array(obs, dtype=np.float32)
        return self.scaler.transform(obs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.raw_obs = np.array(obs, dtype=np.float32)
        for info in infos:
            term = info.get("terminal_observation") if isinstance(info, dict) else None
            if term is not None:
                info["terminal_observation"] = self.scaler.transform(term)
        return self.scaler.transform(obs), rewards, dones, infos


def find_wrapper(venv, cls):
    """First wrapper of type ``cls`` in the ``.venv`` chain of ``venv`` (None if there is none)."""
    cur = venv
    while cur is not None:
        if isinstance(cur, cls):
            return cur
        cur = getattr(cur, "venv", None)
    return None
