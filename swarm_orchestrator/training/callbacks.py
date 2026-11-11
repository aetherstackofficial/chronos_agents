"""Training callbacks for the v2 RL agents.

* ``TrainingInsightsCallback``  - writes behaviour statistics to TensorBoard
  every ``log_every`` env steps (one env step = ``decision_minutes`` minutes): mean of each
  reward component (per MINUTE, so the numbers mean the same at any decision interval; the
  per-decision mean is logged as ``insights/reward_part_per_decision``), mean/std of each
  action output, mean |position|/capacity, turnover, share of steps in a
  position, share of steps both sides quoted (mm) and, per input, the share of
  SCALED values sitting on the +/-clip bound (the 10 worst by name).
* ``SaveVecNormalizeOnBest``  - called by EvalCallback on a new best model; saves
  the training normaliser statistics next to the best model.
* ``FixedSeedEvalCallback``  - EvalCallback that re-seeds the eval env before each
  evaluation, so every evaluation sees the same episodes (comparable curves).

Everything is read from what the env reports (``info`` of each step), from the RAW
observation that ``training.wrappers.ObsScalerVecEnv`` keeps (``raw_obs``; legacy runs:
VecNormalize ``get_original_obs``) and from the observation the policy gets (``new_obs`` = the
output of the fixed ``ObsScaler``), so nothing here needs access to the sub-processes that host
the envs.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

from training.wrappers import ObsScalerVecEnv, find_wrapper

SESSION_MINUTES = 375
FLAT_FRACTION = 0.05          # |position|/capacity below this counts as "not in a position"


def _find(names: Sequence[str], wanted: str) -> Optional[int]:
    try:
        return list(names).index(wanted)
    except ValueError:
        return None


class TrainingInsightsCallback(BaseCallback):
    """Behaviour statistics -> TensorBoard (tags ``insights/...``, ``clip/...``)."""

    def __init__(self, obs_names: Sequence[str], action_names: Sequence[str],
                 log_every: int = 1000, clip_obs: float = 10.0, top_clipped: int = 10,
                 verbose: int = 0, print_clip: bool = False, decision_minutes: int = 1) -> None:
        super().__init__(verbose)
        self.decision_minutes = max(int(decision_minutes), 1)     # default when info has no "minutes"
        self.obs_names = list(obs_names)
        self.action_names = list(action_names)
        self.log_every = int(log_every)
        self.clip_obs = float(clip_obs)
        self.top_clipped = int(top_clipped)
        self.print_clip = bool(print_clip)        # also print the worst clip shares at every flush
        self._i_inv = _find(self.obs_names, "inventory")
        self._last_log = 0
        self._last_inv: Dict[int, float] = {}
        self._reset_window()

    # ------------------------------------------------------------------ #
    def _reset_window(self) -> None:
        self._n = 0
        self._parts_sum: Dict[str, float] = defaultdict(float)
        self._parts_n: Dict[str, int] = defaultdict(int)
        self._parts_minutes = 0.0           # minutes covered by the steps whose parts were summed
        self._act_n = 0
        self._act_sum = np.zeros(len(self.action_names))
        self._act_sq = np.zeros(len(self.action_names))
        self._abs_inv_sum = 0.0
        self._inv_n = 0
        self._turnover_sum = 0.0
        self._turnover_n = 0
        self._turnover_minutes = 0.0        # minutes spanned by the counted position deltas
        self._in_pos = 0
        self._quote_n = 0
        self._both_quoted = 0
        self._clip_hits = np.zeros(len(self.obs_names))
        self._clip_n = 0

    def _tb_writer(self):
        for fmt in self.logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                return fmt.writer
        return None

    # ------------------------------------------------------------------ #
    def _on_step(self) -> bool:
        infos: List[dict] = self.locals.get("infos") or []
        dones = np.asarray(self.locals.get("dones", [False] * len(infos)), dtype=bool)
        actions = self.locals.get("clipped_actions")
        if actions is None:
            actions = self.locals.get("actions")
        self._n += len(infos)

        # reward components
        step_minutes: List[float] = []
        for info in infos:
            mins = info.get("minutes") if isinstance(info, dict) else None
            step_minutes.append(float(mins) if mins else float(self.decision_minutes))
            parts = info.get("reward_parts") if isinstance(info, dict) else None
            if isinstance(parts, dict):
                self._parts_minutes += step_minutes[-1]
                for k, v in parts.items():
                    try:
                        self._parts_sum[k] += float(v)
                        self._parts_n[k] += 1
                    except (TypeError, ValueError):
                        pass
            intent = info.get("intent") if isinstance(info, dict) else None
            if isinstance(intent, dict) and "bid_qty" in intent:        # market maker
                self._quote_n += 1
                if (intent.get("bid_qty") or 0) > 0 and (intent.get("ask_qty") or 0) > 0:
                    self._both_quoted += 1

        # action outputs
        if actions is not None:
            a = np.asarray(actions, dtype=np.float64).reshape(len(infos), -1)[:, :len(self.action_names)]
            self._act_n += a.shape[0]
            self._act_sum[:a.shape[1]] += a.sum(0)
            self._act_sq[:a.shape[1]] += (a * a).sum(0)

        # position / turnover from the RAW observation (inventory input = position/capacity)
        raw = None
        scaler_env = find_wrapper(self.model.get_env(), ObsScalerVecEnv)
        if scaler_env is not None and scaler_env.raw_obs is not None:
            raw = np.asarray(scaler_env.raw_obs)
        else:                                                   # legacy: VecNormalize(norm_obs=True)
            vn = self.model.get_vec_normalize_env()
            if vn is not None and hasattr(vn, "get_original_obs"):
                try:
                    raw = np.asarray(vn.get_original_obs())
                except Exception:
                    raw = None
        if raw is not None and self._i_inv is not None and raw.ndim == 2:
            inv = raw[:, self._i_inv].astype(np.float64)
            self._abs_inv_sum += np.abs(inv).sum()
            self._inv_n += len(inv)
            self._in_pos += int((np.abs(inv) > FLAT_FRACTION).sum())
            for i, x in enumerate(inv):
                if dones[i] if i < len(dones) else False:
                    self._last_inv.pop(i, None)         # obs is the reset obs: no delta
                    continue
                if i in self._last_inv:
                    self._turnover_sum += abs(x - self._last_inv[i])
                    self._turnover_n += 1
                    self._turnover_minutes += step_minutes[i] if i < len(step_minutes) else self.decision_minutes
                self._last_inv[i] = x

        # share of SCALED inputs on the clip bound (new_obs = what the policy gets; newest frame
        # if frames are stacked)
        new_obs = self.locals.get("new_obs")
        if new_obs is not None:
            o = np.asarray(new_obs)
            if o.ndim == 2 and o.shape[1] >= len(self.obs_names):
                o = o[:, -len(self.obs_names):]
                self._clip_hits += (np.abs(o) >= self.clip_obs - 1e-6).sum(0)
                self._clip_n += o.shape[0]

        if self.num_timesteps - self._last_log >= self.log_every:
            self._flush()
            self._last_log = self.num_timesteps
        return True

    # ------------------------------------------------------------------ #
    def _flush(self) -> None:
        if self._n == 0:
            return
        scalars: Dict[str, float] = {}
        for k, s in self._parts_sum.items():
            # sums over K minutes per step: divide by the minutes covered for a per-minute mean
            scalars[f"insights/reward_part/{k}"] = s / max(self._parts_minutes, 1.0)
            scalars[f"insights/reward_part_per_decision/{k}"] = s / max(self._parts_n[k], 1)
        if self._act_n:
            mean = self._act_sum / self._act_n
            std = np.sqrt(np.maximum(self._act_sq / self._act_n - mean * mean, 0.0))
            for i, nm in enumerate(self.action_names):
                scalars[f"insights/action_mean/{nm}"] = float(mean[i])
                scalars[f"insights/action_std/{nm}"] = float(std[i])
        if self._inv_n:
            scalars["insights/abs_position_over_capacity"] = self._abs_inv_sum / self._inv_n
            scalars["insights/share_in_position"] = self._in_pos / self._inv_n
        if self._turnover_n:
            per_step = self._turnover_sum / self._turnover_n
            scalars["insights/turnover_per_step"] = per_step
            # per session = per MINUTE x 375 (a step spans decision_minutes minutes)
            scalars["insights/turnover_per_session"] = (self._turnover_sum / max(self._turnover_minutes, 1.0)
                                                        * SESSION_MINUTES)
        if self._quote_n:
            scalars["insights/share_both_sides_quoted"] = self._both_quoted / self._quote_n
        if self._clip_n:
            share = self._clip_hits / self._clip_n
            order = np.argsort(-share)[:self.top_clipped]
            scalars["clip/worst_share"] = float(share[order[0]]) if len(order) else 0.0
            scalars["clip/inputs_over_1pct"] = float((share > 0.01).sum())
            for j in order:
                if share[j] > 0.0:
                    scalars[f"clip/{self.obs_names[j]}"] = float(share[j])
            if self.print_clip:
                worst = ", ".join(f"{self.obs_names[j]}={share[j]:.2%}" for j in order if share[j] > 0.0)
                print(f"[clip @ {self.num_timesteps}] worst share {scalars['clip/worst_share']:.2%}, "
                      f"inputs over 1%: {int(scalars['clip/inputs_over_1pct'])}"
                      + (f"; top: {worst}" if worst else ""), flush=True)

        writer = self._tb_writer()
        for k, v in scalars.items():
            if writer is not None:
                writer.add_scalar(k, v, self.num_timesteps)
            else:
                self.logger.record(k, v)
        if writer is not None:
            writer.flush()
        self._reset_window()

    def _on_training_end(self) -> None:
        self._flush()


class SaveVecNormalizeOnBest(BaseCallback):
    """Child callback of EvalCallback: on a new best model, save the training
    normaliser next to it (``best_vecnormalize.pkl``)."""

    def __init__(self, vecnormalize, path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.vecnormalize = vecnormalize
        self.path = path

    def _on_step(self) -> bool:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.vecnormalize.save(self.path)
        return True


class FixedSeedEvalCallback(EvalCallback):
    """EvalCallback that re-seeds the eval env before each evaluation so the same
    episodes are replayed every time (needs an env whose ``reset(seed=...)`` seeds
    its episode sampler)."""

    def __init__(self, *args, eval_seed: int = 12345, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eval_seed = int(eval_seed)
        self.eval_seconds = 0.0          # wall time spent in evaluations (for steps/s without eval)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.seed(self.eval_seed)
            t0 = time.time()
            try:
                return super()._on_step()
            finally:
                self.eval_seconds += time.time() - t0
        return super()._on_step()
