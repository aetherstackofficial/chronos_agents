"""Decision log ("thought log") for trained RL agents (whale / market maker).

A neural policy does not think in words.  This module writes, for every
decision that matters, what the agent SAW, what it CHOSE, how SURE it was and
WHICH INPUTS pushed it, as one JSON object per line:

    logs/thoughts/<run_id>/<agent_id>.jsonl

Pieces
------
* ``explain(model, obs_norm, obs_names, ...)``  - works alone (evaluation, notebooks).
  Returns ``{"sure": {...}, "why": {...}}`` for an SB3 PPO/A2C, SAC or TD3 model.
* ``ThoughtLog``  - decides WHEN to write a record (on change + heartbeat),
  builds the record, keeps it pending until its 15/60-minute outcome is known,
  and writes each line once (no file rewriting).

Attribution = gradient x input, taken on the NORMALISED observation (what the
policy really sees), of the deterministic action:
    PPO/A2C : mean of the action distribution (before SB3 clips / squashes it)
    SAC     : tanh(mean)   (the squashed deterministic action, in [-1, 1])
    TD3     : actor output (in [-1, 1])
The model is never modified: no parameter grads, no train()/eval() switch; grad
is enabled only on a detached copy of the observation.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #


def _clean(o: Any) -> Any:
    """Make ``o`` strictly JSON-serialisable (numpy/torch -> python, NaN/inf -> None)."""
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, (int,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, torch.Tensor):
        return _clean(o.detach().cpu().numpy())
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    return str(o)


def _fmt_num(x: Any) -> str:
    try:
        return f"{float(x):.4g}"
    except (TypeError, ValueError):
        return str(x)


# --------------------------------------------------------------------------- #
# explain()
# --------------------------------------------------------------------------- #


def _algo_of(model) -> str:
    """'ppo' (any ActorCriticPolicy: PPO/A2C), 'sac' or 'td3' from the policy class."""
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.sac.policies import SACPolicy
    from stable_baselines3.td3.policies import TD3Policy

    pol = model.policy
    if isinstance(pol, ActorCriticPolicy):
        return "ppo"
    if isinstance(pol, SACPolicy):
        return "sac"
    if isinstance(pol, TD3Policy):
        return "td3"
    raise TypeError(f"unsupported SB3 policy: {type(pol).__name__}")


def _to_tensor(model, arr) -> torch.Tensor:
    a = np.asarray(arr, dtype=np.float32).reshape(1, -1)
    return torch.as_tensor(a, device=model.device)


def _scaled_action(model, action) -> torch.Tensor | None:
    """Env-unit action (what ``predict`` returns) -> policy space [-1, 1] for SAC/TD3."""
    if action is None:
        return None
    a = np.asarray(action, dtype=np.float32).reshape(1, -1)
    try:
        a = model.policy.scale_action(a)
    except Exception:  # unbounded space etc: use as is
        pass
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=model.device)


def _det_action_fn(model, algo):
    """Return f(x[B,n]) -> deterministic action [B,k] (differentiable)."""
    pol = model.policy
    if algo == "ppo":
        return lambda x: pol.get_distribution(x).distribution.mean
    if algo == "sac":
        return lambda x: torch.tanh(pol.actor.get_action_dist_params(x)[0])
    return lambda x: pol.actor(x)


def _q_values(model, x: torch.Tensor, a: torch.Tensor) -> list[float]:
    with torch.no_grad():
        qs = model.policy.critic(x, a)
    return [float(q.reshape(-1)[0]) for q in qs]


def explain(model, obs_norm, obs_names, top_k=5, hold_action=None, *,
            action=None, obs_raw=None, phrases: dict | None = None,
            with_why: bool = True) -> dict:
    """Explain one decision of an SB3 model.

    ``obs_norm``: the normalised observation the policy sees (1-D, len(obs_names)).
    ``action`` / ``hold_action``: actions in env units (as ``predict`` returns); only
    used for SAC/TD3 critics.  ``action`` None -> the deterministic action.
    ``obs_raw`` / ``phrases``: optional, only to fill raw value + wording in ``why``.
    Returns ``{"sure": {...}, "why": {...} | None}``.
    """
    algo = _algo_of(model)
    pol = model.policy
    names = list(obs_names)
    x0 = _to_tensor(model, obs_norm)
    n_in = x0.shape[1]
    if n_in != len(names):
        raise ValueError(f"obs has {n_in} inputs but {len(names)} names")
    f = _det_action_fn(model, algo)

    sure: dict[str, Any] = {"algo": algo, "value": None, "action_mean": None,
                            "action_std": None, "q_chosen": None, "q_hold": None,
                            "q_gap": None, "q_all": None}
    with torch.no_grad():
        if algo == "ppo":
            sure["value"] = float(pol.predict_values(x0).reshape(-1)[0])
            dist = pol.get_distribution(x0).distribution
            sure["action_mean"] = dist.mean[0].tolist()
            sure["action_std"] = dist.stddev[0].tolist()
        elif algo == "sac":
            mean, log_std, _ = pol.actor.get_action_dist_params(x0)
            sure["action_mean"] = torch.tanh(mean)[0].tolist()
            sure["action_std"] = torch.exp(log_std)[0].tolist()  # pre-tanh std
        else:
            sure["action_mean"] = pol.actor(x0)[0].tolist()  # deterministic: no std
    if algo in ("sac", "td3"):
        a_ch = _scaled_action(model, action)
        if a_ch is None:
            a_ch = torch.as_tensor([sure["action_mean"]], dtype=torch.float32, device=model.device)
        qs = _q_values(model, x0, a_ch)
        sure["q_all"] = qs
        sure["q_chosen"] = min(qs)
        a_h = _scaled_action(model, hold_action)
        if a_h is not None:
            sure["q_hold"] = min(_q_values(model, x0, a_h))
            sure["q_gap"] = sure["q_chosen"] - sure["q_hold"]

    why = None
    if with_why:
        k_out = len(sure["action_mean"])
        # one backward pass for all outputs: row i of the batch carries output i
        x = x0.detach().repeat(k_out, 1).requires_grad_(True)
        with torch.enable_grad():
            out = f(x)
            sel = out[torch.arange(k_out), torch.arange(k_out)].sum()
            (grad,) = torch.autograd.grad(sel, x)  # only wrt x: no param .grad touched
        xn = x0.detach().cpu().numpy()[0]
        contrib = (grad.detach().cpu().numpy() * xn[None, :])  # [k_out, n_in]
        raw = None if obs_raw is None else np.asarray(obs_raw, dtype=np.float64).reshape(-1)
        why = _build_why(names, xn, raw, contrib, top_k, phrases or {},
                         sure["action_mean"], algo)
    return {"sure": sure, "why": why}


def _phrase(name, raw, norm, contrib, phrases: dict) -> str:
    tpl = phrases.get(name)
    shown = _fmt_num(raw) if raw is not None else _fmt_num(norm)
    if tpl:
        try:
            return tpl.format(raw=raw if raw is not None else norm, norm=norm,
                              name=name, contrib=contrib)
        except Exception:
            pass
    return f"{name}={shown}"


def _entry(names, xn, raw, i, c, phrases) -> dict:
    r = None if raw is None else float(raw[i])
    return {"name": names[i], "raw": r, "norm": float(xn[i]), "contribution": float(c),
            "sign": int(np.sign(c)), "phrase": _phrase(names[i], r, float(xn[i]), float(c), phrases)}


def _build_why(names, xn, raw, contrib, top_k, phrases, base, algo) -> dict:
    per_output = []
    for o in range(contrib.shape[0]):
        order = np.argsort(-np.abs(contrib[o]))[:top_k]
        per_output.append({"output": o, "action": float(base[o]),
                           "top": [_entry(names, xn, raw, int(i), contrib[o, i], phrases)
                                   for i in order]})
    tot = np.abs(contrib).sum(axis=0)
    comb = []
    for i in np.argsort(-tot)[:top_k]:
        e = _entry(names, xn, raw, int(i), contrib[:, i].sum(), phrases)
        e["total_abs"] = float(tot[i])
        e["per_output"] = [float(v) for v in contrib[:, i]]
        comb.append(e)
    method = {"ppo": "grad*input of distribution mean (pre clip/squash)",
              "sac": "grad*input of tanh(mean) action",
              "td3": "grad*input of actor output"}[algo]
    return {"method": method, "per_output": per_output, "combined": comb}


# --------------------------------------------------------------------------- #
# ThoughtLog
# --------------------------------------------------------------------------- #

_HORIZONS = (15, 60)


class ThoughtLog:
    def __init__(self, run_id: str, agent_id: str, model_name: str, algo: str,
                 out_dir: str = "logs/thoughts", heartbeat_minutes: int = 15,
                 top_k: int = 5, phrases: dict | None = None, *,
                 minutes_per_day: int = 375, explain_on_heartbeat: bool = False):
        # 375 = one NSE session (09:15-15:29), the session length the engine replays.
        self.run_id, self.agent_id = str(run_id), str(agent_id)
        self.model_name, self.algo = model_name, algo
        self.heartbeat = int(heartbeat_minutes)
        self.top_k = int(top_k)
        self.phrases = dict(phrases or {})
        self.mpd = int(minutes_per_day)
        self.explain_on_heartbeat = explain_on_heartbeat
        self.dir = os.path.join(out_dir, self.run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, f"{self.agent_id}.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8")
        self._pending: list[dict] = []   # records in order, with private bookkeeping
        self._last_abs: int | None = None
        self._last_intent: str | None = None
        self._last_equity: float | None = None
        self._last_now: int | None = None
        self.n_records = 0
        self.n_written = 0

    # -- time ---------------------------------------------------------------
    def _abs(self, sim_minute: int, day: int) -> int:
        return int(day) * self.mpd + int(sim_minute)

    # -- main entry ---------------------------------------------------------
    def record(self, *, sim_minute: int, day: int, mode: str, obs_names: list[str],
               obs_raw, obs_norm, model, action_raw, intent: dict, notes: list[str],
               orders: list[dict], memory_summary: dict, state: dict,
               changed: bool | None = None, hold_action=None,
               unix_time: float | None = None) -> dict | None:
        """Maybe log this decision.  Returns the record dict, or None if skipped."""
        now = self._abs(sim_minute, day)
        intent_key = json.dumps(_clean(intent), sort_keys=True)
        is_change = (self._last_abs is None or bool(changed)
                     or intent_key != self._last_intent)
        is_beat = (not is_change and self.heartbeat > 0
                   and now - self._last_abs >= self.heartbeat)
        if not (is_change or is_beat):
            return None
        self._last_abs, self._last_intent = now, intent_key

        raw = np.asarray(obs_raw, dtype=np.float64).reshape(-1)
        names = list(obs_names)
        sure = why = None
        if is_change or self.explain_on_heartbeat:
            ex = explain(model, obs_norm, names, top_k=self.top_k, hold_action=hold_action,
                         action=action_raw, obs_raw=raw, phrases=self.phrases)
            sure, why = ex["sure"], ex["why"]

        baseline = state.get("equity") if isinstance(state, dict) else None
        if not isinstance(baseline, (int, float)):
            baseline = self._last_equity
        rec = {
            "kind": "change" if is_change else "heartbeat",
            "when": {"run_id": self.run_id, "agent_id": self.agent_id,
                     "model": self.model_name, "algo": self.algo, "mode": mode,
                     "sim_minute": int(sim_minute), "day": int(day),
                     "unix_time": unix_time if unix_time is not None else None},
            "saw": {"memory": memory_summary,
                    "obs": {n: float(v) for n, v in zip(names, raw)}},
            "state": state,
            "chose": {"raw": np.asarray(action_raw, dtype=np.float64).reshape(-1).tolist(),
                      "intent": intent, "notes": list(notes), "orders": orders},
            "sure": sure,
            "why": why,
            "then": {"m15": None, "m60": None},
        }
        rec = _clean(rec)
        self._pending.append({"rec": rec, "abs": now, "equity0": baseline, "fills": [],
                              "done": {}})
        self.n_records += 1
        return rec

    # -- outcomes -----------------------------------------------------------
    def update_outcome(self, *, sim_minute: int, day: int,
                       fills: list[dict] | None = None, equity: float | None = None) -> None:
        """Call as time passes.  ``fills`` = fills that arrived since the last call."""
        now = self._abs(sim_minute, day)
        self._last_now = now
        if equity is not None and math.isfinite(float(equity)):
            self._last_equity = float(equity)
        for p in self._pending:
            e = now - p["abs"]
            if e < 0:
                continue
            if p["equity0"] is None and self._last_equity is not None:
                p["equity0"] = self._last_equity   # first equity seen at/after the decision
            if e <= max(_HORIZONS):
                for fl in fills or []:
                    p["fills"].append((e, fl))
            for h in _HORIZONS:
                if h not in p["done"] and e >= h:
                    p["done"][h] = self._outcome(p, h, e, complete=True)
        self._flush(force=False)

    def _outcome(self, p: dict, h: int, e: int, complete: bool) -> dict:
        fl = [f for (t, f) in p["fills"] if t <= h]
        pnl = None
        if self._last_equity is not None and p["equity0"] is not None:
            pnl = self._last_equity - p["equity0"]
        return _clean({"complete": complete, "elapsed_min": e, "n_fills": len(fl),
                       "fills": fl, "pnl": pnl, "equity": self._last_equity})

    def _flush(self, force: bool) -> None:
        # keep file order == decision order: only write from the head of the queue
        while self._pending:
            p = self._pending[0]
            if not force and 60 not in p["done"]:
                break
            rec = p["rec"]
            for h in _HORIZONS:
                o = p["done"].get(h)
                if o is None and force:
                    o = self._outcome(p, h, max(0, (self._last_now if self._last_now is not None else p["abs"]) - p["abs"]), complete=False)
                rec["then"][f"m{h}"] = o
            self._fh.write(json.dumps(rec, allow_nan=False) + "\n")
            self.n_written += 1
            self._pending.pop(0)
        self._fh.flush()

    def close(self) -> None:
        if self._fh.closed:
            return
        self._flush(force=True)
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
