"""Evaluate a trained run (or a rule baseline) on a split and write ``eval_<split>.json``.

    python -m training.evaluate --run training/runs/<dir> [--split validation|test]
        [--episodes 20] [--thoughts] [--cost-mult 2.0] [--stocks ITC,SBIN]
        [--episode-sessions 5] [--seed 7] [--out path.json]

One env step = ``env.decision_minutes`` minutes (the run's own value; runs made before the
decision interval existed are 1).  All time-based metrics (sessions, turnover per session,
holding time, share of time) use MINUTES, so they mean the same at any interval.

Deterministic episodes are played on the raw env (one env, in process) WITHOUT observation
noise (``env.obs_noise: 0``, see ``training.train.load_env_cfg``); the policy sees
``ObsScaler.transform(raw_obs)`` with the run's fixed scaler (``<run>/obs_scaler.json``; runs made
before the scaler existed fall back to their frozen VecNormalize statistics).
Metrics, per role:

* whale: profit after costs, Sharpe of per-session returns (annualised sqrt(250)),
  worst drawdown, turnover (|traded position| / capacity per session), share of time
  in a position, mean holding sessions, long/short share, participation used.
* mm: P&L split into a spread+markout part (edge of every fill measured against the
  mid 1 minute later, minus fees) and an inventory part (the rest), spread captured
  per round trip, mean/max |inventory|/capacity, time for inventory to return to
  zero, adverse selection (1-minute markout), share of time both sides quoted.

Behaviour checks are pass/fail with the measured number (see ``behaviour_checks``).
The metric code is shared with ``training.baselines``.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from training import train as T

SESSION_MINUTES = 375
FLAT_FRACTION = 0.05          # |position|/capacity below this = flat (= decoder whale_flat_fraction)
CLIP = 10.0
EXPLAIN_EVERY = 25            # sampled explain() cadence when no full decision log is written
MEMORY_TOP_K = 5
MEM_SHARE_MIN = 0.10          # "memory used" passes when mem_ inputs make >= 10% of top-5 reasons
LOW_M, HIGH_M = 20, 120
M_KEYS = ("memory_sessions", "mem_sessions", "memory_sessions_M", "M")
CONSTANT_EXEMPT_PREFIX = ("sector_",)
CONSTANT_EXEMPT = ("risk_appetite",)   # fixed per agent in a single-agent eval


# --------------------------------------------------------------------------- tick helpers
def tick_mid(tick: Optional[dict]) -> float:
    if not tick:
        return float("nan")
    bb, ba = tick.get("best_bid"), tick.get("best_ask")
    if bb is not None and ba is not None and float(bb) > 0 and float(ba) > 0:
        return 0.5 * (float(bb) + float(ba))
    return float(tick.get("last_price") or float("nan"))


def tick_abs_minute(tick: Optional[dict]) -> int:
    if not tick:
        return 0
    return int(tick.get("day_count", 0) or 0) * SESSION_MINUTES + int(tick.get("market_minute", 0) or 0)


# --------------------------------------------------------------------------- fills
class FillRecorder:
    """Wraps ``env.book.on_fill`` (instance level) to record every fill with the
    index of the env step it happened in.  Re-attaches when the env makes a new book."""

    def __init__(self) -> None:
        self.fills: List[dict] = []
        self.step = 0
        self._book = None

    def new_episode(self) -> None:
        self.fills = []
        self.step = 0

    def attach(self, env) -> None:
        book = getattr(env, "book", None)
        if book is None or book is self._book:
            return
        orig = book.on_fill

        def wrapped(side, qty, price, fee=0.0, _orig=orig):
            self.fills.append({"step": self.step, "side": str(side).strip().upper()[:1],
                               "qty": float(qty), "price": float(price), "fee": float(fee)})
            return _orig(side, qty, price, fee)

        book.on_fill = wrapped
        self._book = book


# --------------------------------------------------------------------------- policies
class ModelPolicy:
    """SB3 model + observation normaliser (+ optional frame stack).  ``vecnorm`` is the run's fixed
    ``ObsScaler`` (has ``transform``), or - legacy runs - a frozen VecNormalize (``normalize_obs``)."""

    name = "model"

    def __init__(self, model, vecnorm, frame_stack: int = 1, n_in: Optional[int] = None) -> None:
        self.model, self.vn, self.frame_stack = model, vecnorm, max(int(frame_stack), 1)
        self.n_in = n_in
        self._frames: deque = deque(maxlen=self.frame_stack)
        self.last_norm: Optional[np.ndarray] = None       # newest single-frame normalised obs
        self.last_input: Optional[np.ndarray] = None      # what the net saw (stacked)
        self.reset()

    def reset(self) -> None:
        self._frames.clear()

    def _norm(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        if self.vn is None:
            return x
        if hasattr(self.vn, "transform"):                      # ObsScaler
            return np.asarray(self.vn.transform(x))
        return np.asarray(self.vn.normalize_obs(x[None, :]))[0]

    def predict(self, obs, env=None) -> np.ndarray:
        n = self._norm(obs).astype(np.float32)
        self.last_norm = n
        if self.frame_stack > 1:
            if not self._frames:
                for _ in range(self.frame_stack - 1):
                    self._frames.append(np.zeros_like(n))
            self._frames.append(n)
            x = np.concatenate(list(self._frames), axis=-1)
        else:
            x = n
        self.last_input = x
        a, _ = self.model.predict(x[None, :], deterministic=True)
        return np.asarray(a[0], dtype=np.float32)


def load_run_policy(run_dir: Path) -> tuple:
    """(ModelPolicy, resolved_config) from a training run directory."""
    import pickle
    from stable_baselines3 import PPO, SAC, TD3
    cfg = T.load_yaml(run_dir / "config.yaml")
    cls = {"ppo": PPO, "sac": SAC, "td3": TD3}[cfg["algo"]]
    model = cls.load(str(run_dir / "model.zip"), device="cpu")
    scaler_file = run_dir / T.SCALER_NAME
    if scaler_file.exists():
        from agents.obs_scaler import ObsScaler
        norm = ObsScaler.load(scaler_file, T.role_of(cfg["model"]))     # refuses mismatching names
    else:                              # legacy run: VecNormalize(norm_obs=True) statistics
        with open(run_dir / "vecnormalize.pkl", "rb") as fh:
            norm = pickle.load(fh)     # VecNormalize without its venv: enough for normalize_obs
        norm.training = False
    return ModelPolicy(model, norm, int(cfg.get("frame_stack", 1))), cfg


# --------------------------------------------------------------------------- episode runner
def _scalar_meta(d: Optional[dict]) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (int, float, str, bool, np.integer, np.floating)):
            out[k] = v.item() if hasattr(v, "item") else v
    return out


def run_episode(env, policy, seed: Optional[int], recorder: FillRecorder, role: str,
                thought: Optional[dict] = None, explain_every: int = 0,
                max_steps: Optional[int] = None) -> dict:
    """Play one deterministic episode.  Returns arrays (one entry per env step = decision, values AFTER
    the step; ``minutes_per_step`` = the minutes each step spanned) plus, when the env reports its
    ``minute_log``, the equity of every minute (``eq_path``) and of every session close
    (``session_eq``)."""
    obs, info = env.reset(seed=seed) if seed is not None else env.reset()
    recorder.new_episode()
    recorder.attach(env)
    policy.reset()
    book = env.book
    cap = float(book.capacity)
    rows: Dict[str, list] = {k: [] for k in
                             ("obs_raw", "obs_norm", "action", "pos", "equity", "mid", "abs_min",
                              "day", "minute", "participation", "child_qty", "bid_qty", "ask_qty",
                              "explain_names", "last_price", "minutes")}
    eq_path: List[float] = []                   # equity after every minute (env minute_log)
    session_eq: List[float] = []                # equity at every session close (market minute 375)
    have_log = False
    parts: Dict[str, float] = {}
    fees_sum = 0.0                              # ALL costs of the episode (fills, carry, closing)
    meta0 = _scalar_meta(info)
    equity0 = float(book.equity)
    pos0 = float(book.position) / cap
    mid0 = tick_mid(getattr(env, "last_tick", None))
    last_info: dict = info or {}
    tlog = thought["log"] if thought else None
    t = 0
    done = False
    while not done:
        obs = np.asarray(obs, dtype=np.float32)
        a = np.asarray(policy.predict(obs, env), dtype=np.float32).reshape(-1)
        a = np.clip(np.nan_to_num(a), -1.0, 1.0)
        recorder.step = t
        pre_tick = getattr(env, "last_tick", None) or {}
        pre_state = pre_mem = None
        if tlog is not None:                    # the decision is stamped with the state it was taken in
            pre_state = {"equity": float(env.book.equity), "position": float(env.book.position),
                         "capacity": cap, "position_frac": float(env.book.position) / cap,
                         "drawdown": float(env.book.drawdown), "open_pnl": float(env.book.open_pnl),
                         "holding_sessions": int(env.book.holding_sessions)}
            try:
                pre_mem = dict(env.memory.summary())
            except Exception:
                pre_mem = {}
            pre_mem["last_price"] = float(pre_tick.get("last_price") or float("nan"))
        obs2, reward, terminated, truncated, info = env.step(a)
        recorder.attach(env)
        book = env.book
        tick = getattr(env, "last_tick", None) or {}
        last_info = info or {}
        fees_sum += float(last_info.get("fees") or 0.0)
        mlog = last_info.get("minute_log")
        rows["minutes"].append(int(last_info.get("minutes") or (len(mlog) if mlog else 1)))
        if mlog:
            have_log = True
            eq_path.extend(float(e[2]) for e in mlog)
            session_eq.extend(float(e[2]) for e in mlog if e[1] == SESSION_MINUTES)
        rows["obs_raw"].append(obs)
        rows["obs_norm"].append(getattr(policy, "last_norm", None))
        rows["action"].append(a)
        rows["pos"].append(float(book.position) / cap)
        rows["equity"].append(float(book.equity))
        rows["mid"].append(tick_mid(tick))
        rows["last_price"].append(float(tick.get("last_price") or float("nan")))
        rows["abs_min"].append(tick_abs_minute(tick))
        rows["day"].append(int(tick.get("day_count", 0) or 0))
        rows["minute"].append(int(tick.get("market_minute", 0) or 0))
        intent = last_info.get("intent") or {}
        rows["participation"].append(float(intent["participation"]) if intent.get("participation") is not None else float("nan"))
        rows["child_qty"].append(float(intent.get("child_qty") or 0.0))
        rows["bid_qty"].append(float(intent.get("bid_qty") or 0.0))
        rows["ask_qty"].append(float(intent.get("ask_qty") or 0.0))
        for k, v in (last_info.get("reward_parts") or {}).items():
            try:
                parts[k] = parts.get(k, 0.0) + float(v)
            except (TypeError, ValueError):
                pass

        norm = getattr(policy, "last_norm", None)
        model = getattr(policy, "model", None)
        names = None
        if getattr(policy, "frame_stack", 1) > 1:
            norm = None                      # stacked inputs: no per-input explanation
        if tlog is not None and norm is not None and model is not None:
            this_fills = [f for f in recorder.fills if f["step"] == t]
            rec = tlog.record(
                sim_minute=int(pre_tick.get("market_minute", 0) or 0), day=int(pre_tick.get("day_count", 0) or 0),
                mode="evaluate", obs_names=thought["names"], obs_raw=obs, obs_norm=norm, model=model,
                action_raw=a, intent=last_info.get("intent") or {}, notes=list(last_info.get("notes") or []),
                orders=list(last_info.get("orders") or []), memory_summary=pre_mem, state=pre_state,
                unix_time=pre_tick.get("unix_time"))
            # outcomes are tracked minute by minute (a step spans decision_minutes minutes)
            if mlog:
                k0 = 0
                for (d_, m_, eq_, nf_) in mlog:
                    tlog.update_outcome(sim_minute=int(m_), day=int(d_), fills=this_fills[k0:k0 + nf_],
                                        equity=float(eq_))
                    k0 += nf_
            else:
                tlog.update_outcome(sim_minute=int(tick.get("market_minute", 0) or 0),
                                    day=int(tick.get("day_count", 0) or 0), fills=this_fills,
                                    equity=float(book.equity))
            if rec and rec.get("why"):
                names = [e["name"] for e in rec["why"]["combined"][:MEMORY_TOP_K]]
        elif explain_every and norm is not None and model is not None and t % explain_every == 0:
            try:
                from agents.thought_log import explain
                ex = explain(model, norm, thought_names(role), top_k=MEMORY_TOP_K, action=a, obs_raw=obs)
                names = [e["name"] for e in ex["why"]["combined"][:MEMORY_TOP_K]]
            except Exception:
                names = None
        rows["explain_names"].append(names)
        obs = obs2
        t += 1
        done = bool(terminated or truncated) or (max_steps is not None and t >= max_steps)
    if tlog is not None:
        tlog.close()
    ep = {k: (np.asarray(v, dtype=np.float64) if k not in ("obs_raw", "obs_norm", "action", "explain_names") else v)
          for k, v in rows.items()}
    ep["obs_raw"] = np.asarray(rows["obs_raw"], dtype=np.float64)
    ep["action"] = np.asarray(rows["action"], dtype=np.float64)
    ep["obs_norm"] = (np.asarray(rows["obs_norm"], dtype=np.float64)
                      if rows["obs_norm"] and rows["obs_norm"][0] is not None else None)
    sess_eq = None
    if have_log:
        sess_eq = list(session_eq)
        if rows["minute"] and rows["minute"][-1] != SESSION_MINUTES:    # stopped mid-session (drawdown stop)
            sess_eq.append(float(rows["equity"][-1]))
    ep["minutes_per_step"] = np.asarray(rows["minutes"], dtype=np.float64)
    ep.update(equity0=equity0, pos0=pos0, mid0=mid0, capacity=cap, parts=parts, fills=list(recorder.fills),
              eq_path=(np.asarray([equity0] + eq_path) if have_log else None),
              session_eq=(np.asarray(sess_eq) if sess_eq is not None else None),
              decision_minutes=int(getattr(env, "decision_minutes", 1)),
              meta={**meta0, **_scalar_meta(last_info)}, fees_total=float(fees_sum), steps=t, minutes=int(np.sum(rows["minutes"])),
              tick_size=float((getattr(env, "last_tick", None) or {}).get("tick_size") or 0.05))
    return ep


def thought_names(role: str) -> List[str]:
    from agents.features_v2 import OBS_SPEC
    return list(OBS_SPEC[role])


# --------------------------------------------------------------------------- metrics
def _f(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _runs(mask: np.ndarray) -> List[tuple]:
    """(start, end_exclusive) of consecutive True runs."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def session_returns(ep: dict) -> np.ndarray:
    """Return of every session (equity at its close / equity at the previous close - 1).  The closes come
    from the env's per-minute log when it has one (exact at any decision interval); otherwise the step
    on which the day changes is the close."""
    if ep.get("session_eq") is not None:
        eq = np.concatenate([[ep["equity0"]], ep["session_eq"]])
        return np.asarray([c / p - 1.0 for p, c in zip(eq[:-1], eq[1:]) if p > 0])
    eq = np.concatenate([[ep["equity0"]], ep["equity"]])
    day = ep["day"]
    rets, prev = [], eq[0]
    change = np.flatnonzero(np.diff(day) != 0)
    ends = list(change) + [len(day) - 1]
    for e in ends:
        end_eq = eq[e + 1]
        if prev > 0:
            rets.append(end_eq / prev - 1.0)
        prev = end_eq
    return np.asarray(rets)


def max_drawdown(ep: dict) -> float:
    eq = ep["eq_path"] if ep.get("eq_path") is not None else np.concatenate([[ep["equity0"]], ep["equity"]])
    peak = np.maximum.accumulate(eq)
    return float(np.max((peak - eq) / np.where(peak > 0, peak, 1.0)))


def n_sessions(ep: dict) -> int:
    return max(int(len(np.unique(ep["day"]))), 1)


def _step_minutes(ep: dict) -> np.ndarray:
    m = ep.get("minutes_per_step")
    return np.asarray(m, dtype=np.float64) if m is not None else np.ones(len(ep["pos"]))


def whale_episode_metrics(ep: dict) -> dict:
    pos = np.concatenate([[ep["pos0"]], ep["pos"]])
    dpos = np.abs(np.diff(pos))
    in_pos = np.abs(ep["pos"]) > FLAT_FRACTION
    sess = n_sessions(ep)
    w = _step_minutes(ep)                         # minutes each step spans: shares are time shares
    cm = np.concatenate([[0.0], np.cumsum(w)])    # minutes elapsed at the start of every step
    runs = []
    sign = np.sign(ep["pos"])
    for a, b in _runs(in_pos):
        s = a
        for i in range(a + 1, b + 1):
            if i == b or sign[i] != sign[s]:
                runs.append((cm[i] - cm[s]) / SESSION_MINUTES)
                s = i
    part = ep["participation"]
    used = part[(ep["child_qty"] > 0) & np.isfinite(part)]
    if used.size == 0:
        used = part[np.isfinite(part)]
    sr = session_returns(ep)
    return {
        "pnl": float(ep["equity"][-1] - ep["equity0"]),
        "return": float(ep["equity"][-1] / ep["equity0"] - 1.0),
        "session_returns": sr.tolist(),
        "max_drawdown": max_drawdown(ep),
        "turnover_per_session": float(dpos.sum() / sess),
        "share_in_position": float(np.average(in_pos, weights=w)),
        "share_long": float(np.average(ep["pos"] > FLAT_FRACTION, weights=w)),
        "share_short": float(np.average(ep["pos"] < -FLAT_FRACTION, weights=w)),
        "holding_sessions": runs,
        "participation_used": _f(used.mean()) if used.size else None,
        "pos_std": float(np.std(ep["pos"])),
        "n_sessions": sess,
    }


def mm_episode_metrics(ep: dict) -> dict:
    pos, mid, am = ep["pos"], ep["mid"], ep["abs_min"]
    tick = ep["tick_size"]
    fills = ep["fills"]
    n = len(mid)
    mid_prev = np.concatenate([[ep.get("mid0", mid[0])], mid[:-1]])
    edge_1m = spread = adverse = 0.0
    qty_tot = 0.0
    adv_qty_w = []
    fees = 0.0
    n_mark = 0
    for f in fills:
        i = min(f["step"], n - 1)
        sgn = 1.0 if f["side"] == "B" else -1.0
        m0 = mid_prev[i]            # mid of the tick the quote was resting against
        if not math.isfinite(m0):
            m0 = ep["last_price"][i]
        j = int(np.searchsorted(am, am[i] + 1, side="left"))
        if j >= n:
            j, complete = n - 1, False
        else:
            complete = True
        m1 = mid[j] if math.isfinite(mid[j]) else m0
        q = f["qty"]
        spread += sgn * (m0 - f["price"]) * q
        edge_1m += sgn * (m1 - f["price"]) * q
        adverse += -sgn * (m1 - m0) * q            # >0: the mid moved against the maker
        qty_tot += q
        fees += f["fee"]
        n_mark += int(complete)
    total = float(ep["equity"][-1] - ep["equity0"])
    spread_markout = edge_1m - fees
    absmin = np.abs(pos)
    # excursions of inventory away from flat
    side = np.sign(pos) * (absmin > FLAT_FRACTION)
    exc = []                                       # (minutes, completed)
    i = 0
    P = np.concatenate([[ep["pos0"]], pos])
    sg = np.sign(P) * (np.abs(P) > FLAT_FRACTION)
    t_all = np.concatenate([[am[0] - 1], am])
    start = 0 if sg[0] != 0 else None
    for k in range(1, len(sg)):
        if start is not None and sg[k] != sg[start]:
            exc.append((t_all[k] - t_all[start], True))
            start = k if sg[k] != 0 else None
        elif start is None and sg[k] != 0:
            start = k
    if start is not None:
        exc.append((t_all[-1] - t_all[start], False))
    done_t = [m for m, c in exc if c]
    quoted_any = (ep["bid_qty"] > 0) | (ep["ask_qty"] > 0)
    quoted_both = (ep["bid_qty"] > 0) & (ep["ask_qty"] > 0)
    sr = session_returns(ep)
    return {
        "pnl": total,
        "return": float(ep["equity"][-1] / ep["equity0"] - 1.0),
        "session_returns": sr.tolist(),
        "max_drawdown": max_drawdown(ep),
        "pnl_spread_markout": float(spread_markout),
        "pnl_inventory": float(total - spread_markout),
        "fees": float(fees),
        "spread_captured_total": float(spread),
        "shares_traded": float(qty_tot),
        "n_fills": len(fills),
        "adverse_selection_total": float(adverse),
        "abs_inventory_mean": float(absmin.mean()),
        "abs_inventory_max": float(absmin.max()),
        "excursion_minutes_completed": [float(x) for x in done_t],
        "excursions_total": len(exc),
        "excursions_completed": len(done_t),
        "share_quoted_any": float(quoted_any.mean()),
        "share_quoted_both": float(quoted_both.mean()),
        "turnover_per_session": float(np.abs(np.diff(P)).sum() / n_sessions(ep)),
        "n_sessions": n_sessions(ep),
        "tick_size": tick,
    }


def aggregate(role: str, eps: List[dict], per_ep: List[dict]) -> dict:
    """Pool the per-episode metrics into the headline numbers."""
    sr = np.concatenate([np.asarray(m["session_returns"]) for m in per_ep]) if per_ep else np.array([])
    sharpe = None
    if sr.size >= 2 and np.std(sr, ddof=1) > 0:
        sharpe = float(np.mean(sr) / np.std(sr, ddof=1) * math.sqrt(250))
    out: Dict[str, Any] = {
        "episodes": len(per_ep),
        "profit_after_costs_mean": float(np.mean([m["pnl"] for m in per_ep])),
        "profit_after_costs_total": float(np.sum([m["pnl"] for m in per_ep])),
        "return_mean": float(np.mean([m["return"] for m in per_ep])),
        "sharpe_per_session_annualised": sharpe,
        "session_return_mean": _f(sr.mean()) if sr.size else None,
        "sessions": int(sr.size),
        "worst_drawdown": float(np.max([m["max_drawdown"] for m in per_ep])),
        "mean_drawdown": float(np.mean([m["max_drawdown"] for m in per_ep])),
        "turnover_per_session": float(np.mean([m["turnover_per_session"] for m in per_ep])),
        "reward_parts_mean_per_episode": {
            k: float(np.mean([e["parts"].get(k, 0.0) for e in eps]))
            for k in sorted({k for e in eps for k in e["parts"]})},
    }
    if role == "whale":
        hold = [h for m in per_ep for h in m["holding_sessions"]]
        out.update({
            "share_in_position": float(np.mean([m["share_in_position"] for m in per_ep])),
            "mean_holding_sessions": float(np.mean(hold)) if hold else 0.0,
            "n_positions": len(hold),
            "share_long": float(np.mean([m["share_long"] for m in per_ep])),
            "share_short": float(np.mean([m["share_short"] for m in per_ep])),
            "long_share_of_time_in_position": _safe_div(np.mean([m["share_long"] for m in per_ep]),
                                                        np.mean([m["share_in_position"] for m in per_ep])),
            "participation_used": _f(np.nanmean([m["participation_used"] if m["participation_used"] is not None
                                                 else np.nan for m in per_ep])) if per_ep else None,
        })
    else:
        shares = sum(m["shares_traded"] for m in per_ep)
        spread = sum(m["spread_captured_total"] for m in per_ep)
        adverse = sum(m["adverse_selection_total"] for m in per_ep)
        tick = per_ep[0]["tick_size"] if per_ep else 0.05
        exc = [x for m in per_ep for x in m["excursion_minutes_completed"]]
        n_exc = sum(m["excursions_total"] for m in per_ep)
        per_rt = spread / (shares / 2.0) if shares > 0 else None
        out.update({
            "pnl_spread_markout_total": float(sum(m["pnl_spread_markout"] for m in per_ep)),
            "pnl_inventory_total": float(sum(m["pnl_inventory"] for m in per_ep)),
            "fees_total": float(sum(m["fees"] for m in per_ep)),
            "n_fills": int(sum(m["n_fills"] for m in per_ep)),
            "shares_traded": float(shares),
            "spread_captured_per_round_trip": per_rt,
            "spread_captured_per_round_trip_ticks": (per_rt / tick) if per_rt is not None else None,
            "abs_inventory_mean": float(np.mean([m["abs_inventory_mean"] for m in per_ep])),
            "abs_inventory_max": float(np.max([m["abs_inventory_max"] for m in per_ep])),
            "minutes_to_return_to_zero_mean": float(np.mean(exc)) if exc else None,
            "excursions_total": int(n_exc),
            "excursions_returned_share": (len(exc) / n_exc) if n_exc else None,
            "adverse_selection_1m_per_share": (adverse / shares) if shares > 0 else None,
            "adverse_selection_1m_per_share_ticks": (adverse / shares / tick) if shares > 0 else None,
            "share_quoted_any": float(np.mean([m["share_quoted_any"] for m in per_ep])),
            "share_quoted_both": float(np.mean([m["share_quoted_both"] for m in per_ep])),
        })
    return out


def _safe_div(a, b):
    return float(a / b) if b and b > 0 else None


# --------------------------------------------------------------------------- behaviour checks
def _check(name, passed, value, threshold, note="") -> dict:
    return {"name": name, "passed": (None if passed is None else bool(passed)),
            "value": value, "threshold": threshold, "note": note}


def behaviour_checks(role: str, eps: List[dict], per_ep: List[dict], agg: dict,
                     obs_names: List[str]) -> List[dict]:
    checks: List[dict] = []
    # 1. outputs not stuck
    n_out = eps[0]["action"].shape[1]
    stds = np.array([[np.std(e["action"][:, k]) for k in range(n_out)] for e in eps])   # ep x out
    mean_std = stds.mean(axis=0)
    checks.append(_check("outputs_not_stuck", bool((mean_std > 0.05).all()),
                         {f"output_{k}": float(v) for k, v in enumerate(mean_std)},
                         "mean per-episode std of every output > 0.05"))
    # 2. role specific
    if role == "whale":
        sl, ss = agg["share_long"], agg["share_short"]
        pstd = float(np.mean([m["pos_std"] for m in per_ep]))
        checks.append(_check("whale_trades_long_and_short_and_varies",
                             sl >= 0.01 and ss >= 0.01 and pstd > FLAT_FRACTION,
                             {"share_long": sl, "share_short": ss, "position_std": pstd},
                             "long share >= 1%, short share >= 1%, std(position/capacity) > 0.05"))
    else:
        q = agg["share_quoted_any"]
        ret = agg["excursions_returned_share"]
        checks.append(_check("mm_quotes_at_least_80pct_of_time", q >= 0.80, q, ">= 0.80"))
        ok = (agg["abs_inventory_mean"] <= 0.5) and (ret is None or ret >= 0.8)
        checks.append(_check("mm_inventory_mean_reverts", ok,
                             {"abs_inventory_mean": agg["abs_inventory_mean"],
                              "excursions_returned_share": ret,
                              "minutes_to_return_mean": agg["minutes_to_return_to_zero_mean"]},
                             "mean |inv|/cap <= 0.5 and >= 80% of excursions return to flat"))
    # 3. clip + constant inputs
    if eps[0]["obs_norm"] is not None:
        norm = np.concatenate([e["obs_norm"] for e in eps], axis=0)
        share = (np.abs(norm) >= CLIP - 1e-6).mean(axis=0)
        order = np.argsort(-share)[:5]
        worst = {obs_names[i]: float(share[i]) for i in order if share[i] > 0}
        checks.append(_check("no_input_at_clip_over_1pct", bool((share <= 0.01).all()),
                             {"worst_share": float(share.max()), "worst_inputs": worst},
                             "every input at +/-10 on <= 1% of steps"))
    else:
        checks.append(_check("no_input_at_clip_over_1pct", None, None, "n/a: no normaliser (rule baseline)"))
    raw = np.concatenate([e["obs_raw"] for e in eps], axis=0)
    sd = raw.std(axis=0)
    const = [obs_names[i] for i in np.flatnonzero(sd <= 1e-9)
             if not obs_names[i].startswith(CONSTANT_EXEMPT_PREFIX) and obs_names[i] not in CONSTANT_EXEMPT]
    checks.append(_check("no_constant_input", len(const) == 0, {"constant_inputs": const},
                         "every input varies over the evaluation (sector_*/risk_appetite exempt)"))
    # 4. few memory sessions not wild
    checks.append(memory_sessions_check(eps, per_ep))
    # 5. memory used
    checks.extend(memory_used_checks(role, eps, obs_names))
    return checks


def _find_m(meta: dict) -> Optional[int]:
    for k in M_KEYS:
        if k in meta and meta[k] is not None:
            try:
                return int(meta[k])
            except (TypeError, ValueError):
                pass
    return None


def memory_sessions_check(eps: List[dict], per_ep: List[dict]) -> dict:
    ms = [_find_m(e["meta"]) for e in eps]
    if all(m is None for m in ms):
        return _check("few_memory_sessions_not_wild", None, None,
                      "n/a: episodes carry no memory-session count", "use --memory-sweep or an env that reports it")
    lo = [i for i, m in enumerate(ms) if m is not None and m <= LOW_M]
    hi = [i for i, m in enumerate(ms) if m is not None and m >= HIGH_M]
    if not lo or not hi:
        return _check("few_memory_sessions_not_wild", None, {"n_low_M": len(lo), "n_high_M": len(hi)},
                      f"n/a: needs episodes with M <= {LOW_M} and M >= {HIGH_M}")

    def avg(idx, key):
        return float(np.mean([per_ep[i][key] for i in idx]))
    val, ok = {}, True
    for key in ("turnover_per_session", "max_drawdown"):
        a, b = avg(lo, key), avg(hi, key)
        ratio = a / b if b > 0 else (1.0 if a <= 0 else float("inf"))
        val[key] = {"low_M": a, "high_M": b, "ratio": ratio}
        ok = ok and (0.5 <= ratio <= 2.0)
    return _check("few_memory_sessions_not_wild", ok, val, "turnover and drawdown with M<=20 within 2x of M>=120")


def memory_used_checks(role: str, eps: List[dict], obs_names: List[str]) -> List[dict]:
    out = []
    lists = [n for e in eps for n in e["explain_names"] if n]
    if lists:
        share = float(np.mean([any(x.startswith("mem_") for x in names) for names in lists]))
        per_input = float(np.mean([np.mean([x.startswith("mem_") for x in names]) for names in lists]))
        out.append(_check("memory_used_in_top5_reasons", share >= MEM_SHARE_MIN,
                          {"share_of_decisions_with_mem_input_in_top5": share,
                           "share_of_top5_slots_that_are_mem": per_input, "n_decisions": len(lists)},
                          f"share of decisions with a mem_ input in the top-5 reasons >= {MEM_SHARE_MIN}"))
    else:
        out.append(_check("memory_used_in_top5_reasons", None, None,
                          "n/a: no explanations (rule baseline)"))
    if role == "whale":
        near_d, far_d = [], []
        idx = {n: i for i, n in enumerate(obs_names)}
        for e in eps:
            raw = e["obs_raw"]
            near = np.zeros(len(raw), dtype=bool)
            for lv in ("up1", "dn1"):
                d, s = raw[:, idx[f"mem_lvl_{lv}_dist"]], raw[:, idx[f"mem_lvl_{lv}_strength"]]
                near |= (np.abs(d) < 0.5) & (s > 0.5)
            pos = np.concatenate([[e["pos0"]], e["pos"]])
            dp = np.abs(np.diff(pos))
            near_d.append(dp[near])
            far_d.append(dp[~near])
        nd, fd = np.concatenate(near_d), np.concatenate(far_d)
        a = float(nd.mean()) if nd.size else None
        b = float(fd.mean()) if fd.size else None
        out.append(_check("whale_position_change_near_strong_levels", None,
                          {"mean_abs_position_change_near": a, "mean_abs_position_change_elsewhere": b,
                           "steps_near": int(nd.size), "steps_elsewhere": int(fd.size),
                           "ratio": (a / b) if (a is not None and b) else None},
                          "informational: |lvl dist| < 0.5 ATR and strength > 0.5 vs elsewhere"))
    return out


# --------------------------------------------------------------------------- driver
def evaluate_policy(model_name: str, policy, split: str, env_cfg: dict, episodes: int, seed: int,
                    thoughts_dir: Optional[Path] = None, algo: str = "", run_id: str = "eval",
                    explain_every: int = EXPLAIN_EVERY, max_steps: Optional[int] = None,
                    stocks: Optional[List[str]] = None) -> dict:
    role = T.role_of(model_name)
    names = thought_names(role)
    env = T.env_factory()(model_name, split, env_cfg, seed=seed)
    if stocks:
        restrict_stocks(env, stocks)
    rec = FillRecorder()
    eps: List[dict] = []
    t0 = time.time()
    use_explain = hasattr(policy, "model")
    for k in range(episodes):
        thought = None
        if thoughts_dir is not None and use_explain:
            from agents.thought_log import ThoughtLog
            thought = {"log": ThoughtLog(f"{run_id}_{split}_ep{k:02d}", model_name, model_name, algo,
                                         out_dir=str(thoughts_dir)), "names": names}
        ep = run_episode(env, policy, seed + k, rec, role, thought=thought,
                         explain_every=(explain_every if use_explain else 0), max_steps=max_steps)
        eps.append(ep)
    env.close()
    per = [(whale_episode_metrics if role == "whale" else mm_episode_metrics)(e) for e in eps]
    agg = aggregate(role, eps, per)
    checks = behaviour_checks(role, eps, per, agg, names)
    return {
        "model": model_name, "algo": algo, "role": role, "split": split, "episodes": episodes,
        "seed": seed, "steps": int(sum(e["steps"] for e in eps)),
        "minutes": int(sum(e["minutes"] for e in eps)),
        "decision_minutes": int(getattr(env, "decision_minutes", 1)),
        "wall_seconds": round(time.time() - t0, 2),
        "metrics": agg, "checks": checks,
        "checks_failed": [c["name"] for c in checks if c["passed"] is False],
        "episode_summaries": [{"index": i, "steps": e["steps"], "minutes": e["minutes"], "pnl": per[i]["pnl"],
                               "max_drawdown": per[i]["max_drawdown"],
                               # meta["fees"] is the LAST step's fees only; fees_total is the episode's
                               "fees_total": e.get("fees_total"), "meta": e["meta"]}
                              for i, e in enumerate(eps)],
    }


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def write_eval_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")


def eval_env_cfg(cfg: dict, cost_mult: Optional[float] = None, stocks: Optional[List[str]] = None,
                 episode_sessions: Optional[int] = None, memory_sessions: Optional[int] = None,
                 model_name: Optional[str] = None, decision_minutes: Optional[int] = None) -> dict:
    """Env config of the eval env: env.yaml + env_overrides + eval_env_overrides, then
    ``cost_mult`` -> costs.cost_mult_range [m, m]; ``episode_sessions`` ->
    sampler.length_sessions[model] [n, n]; ``memory_sessions`` -> sampler.memory_mix [[1, M, M]];
    ``decision_minutes`` -> env.decision_minutes[model] (a trained run must be evaluated at the interval
    it was trained with, see ``run_decision_minutes``).
    (``stocks`` is applied to the sampler after the env is built, see ``restrict_stocks``.)"""
    model_name = model_name or cfg.get("model")
    extra: Dict[str, Any] = {}
    if cost_mult is not None:
        extra["costs"] = {"cost_mult_range": [float(cost_mult), float(cost_mult)]}
    if episode_sessions is not None:
        extra["sampler"] = {"length_sessions": {model_name: [int(episode_sessions)] * 2}}
    if decision_minutes is not None:
        extra["env"] = {"decision_minutes": {model_name: int(decision_minutes)}}
    cfg2 = T.load_env_cfg(cfg, eval_env=True, extra=extra)
    if memory_sessions is not None:                # memory_mix is a list: replace, do not merge
        cfg2["sampler"]["memory_mix"] = [[1.0, int(memory_sessions), int(memory_sessions)]]
    return cfg2


def run_decision_minutes(cfg: dict) -> int:
    """The decision interval a trained run was made with: the value stored in its config.yaml; runs from
    before the interval existed have none in their stored env config and were trained at 1 minute."""
    if cfg.get("decision_minutes"):
        return int(cfg["decision_minutes"])
    return T.decision_minutes_of(cfg.get("env_cfg") or {}, cfg["model"])


def restrict_stocks(env, stocks: List[str]) -> None:
    sampler = getattr(env, "sampler", None)
    if sampler is None or not hasattr(sampler, "symbols"):
        raise RuntimeError("env has no sampler.symbols: cannot restrict stocks")
    known = set(sampler.symbols) | set(getattr(sampler, "all_symbols", []) or [])
    from training.envs.sampler import available_symbols
    known |= set(available_symbols())
    bad = [s for s in stocks if s not in known]
    if bad:
        raise ValueError(f"unknown stocks {bad}; available: {sorted(known)}")
    sampler.symbols = list(stocks)


def memory_sweep(model_name: str, policy, split: str, cfg: dict, n: int, seed: int, cost_mult=None,
                 stocks=None, episode_sessions=None, algo: str = "", decision_minutes=None) -> dict:
    """Extra check run: ``n`` episodes with M forced to 10 (<= 20) and ``n`` with M forced to 150
    (>= 120); turnover and drawdown of the short-memory group must be within 2x of the long one."""
    runs = {}
    for label, m in (("low_M", 10), ("high_M", 150)):
        ecfg = eval_env_cfg(cfg, cost_mult, None, episode_sessions, memory_sessions=m, model_name=model_name,
                            decision_minutes=decision_minutes)
        runs[label] = evaluate_policy(model_name, policy, split, ecfg, n, seed, algo=algo,
                                      explain_every=0, stocks=stocks)["metrics"]
    val, ok = {}, True
    for key, mkey in (("turnover_per_session", "turnover_per_session"), ("mean_drawdown", "mean_drawdown")):
        a, b = runs["low_M"][mkey], runs["high_M"][mkey]
        ratio = a / b if b > 0 else (1.0 if a <= 0 else float("inf"))
        val[key] = {"M=10": a, "M=150": b, "ratio": ratio}
        ok = ok and 0.5 <= ratio <= 2.0
    return _check("few_memory_sessions_not_wild_sweep", ok, val,
                  "forced M=10 vs M=150: turnover and drawdown within 2x", f"{n} episodes each")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="training.evaluate", description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="training/runs/<dir>")
    ap.add_argument("--split", default="validation", choices=("validation", "test", "train"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--thoughts", action="store_true", help="write a decision log + HTML per episode")
    ap.add_argument("--cost-mult", type=float, help="multiply trading costs (e.g. 2.0)")
    ap.add_argument("--stocks", help="comma-separated stock symbols (e.g. the held-out ones)")
    ap.add_argument("--episode-sessions", type=int, help="override sessions per episode")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--decision-minutes", type=int,
                    help="override the decision interval (default: the one the run was trained with)")
    ap.add_argument("--memory-sweep", type=int, default=0, metavar="N",
                    help="also run N episodes with memory forced to 10 and N with 150 sessions")
    ap.add_argument("--out", help="output json (default <run>/eval_<split>.json)")
    a = ap.parse_args(argv)
    if str(T.ROOT) not in sys.path:
        sys.path.insert(0, str(T.ROOT))
    run_dir = Path(a.run).resolve()
    policy, cfg = load_run_policy(run_dir)
    stocks = [s.strip() for s in a.stocks.split(",")] if a.stocks else None
    dmin = a.decision_minutes or run_decision_minutes(cfg)
    env_cfg = eval_env_cfg(cfg, a.cost_mult, stocks, a.episode_sessions, decision_minutes=dmin)
    thoughts_dir = run_dir / "thoughts" if a.thoughts else None
    res = evaluate_policy(cfg["model"], policy, a.split, env_cfg, a.episodes, a.seed,
                          thoughts_dir=thoughts_dir, algo=cfg["algo"], run_id=run_dir.name, stocks=stocks)
    if a.memory_sweep:
        chk = memory_sweep(cfg["model"], policy, a.split, cfg, a.memory_sweep, a.seed, a.cost_mult, stocks,
                           a.episode_sessions, algo=cfg["algo"], decision_minutes=dmin)
        res["checks"].append(chk)
        res["checks_failed"] = [c["name"] for c in res["checks"] if c["passed"] is False]
    res.update({"run": str(run_dir), "cost_mult": a.cost_mult, "stocks": stocks, "policy": "trained",
                "timestamp": dt.datetime.now().isoformat(timespec="seconds")})
    suffix = ""
    if a.cost_mult is not None:
        suffix += f"_cost{a.cost_mult:g}"
    if stocks:
        suffix += "_stocks"
    out = Path(a.out) if a.out else run_dir / f"eval_{a.split}{suffix}.json"
    write_eval_json(out, res)
    if thoughts_dir is not None and thoughts_dir.exists():
        for d in sorted(p for p in thoughts_dir.iterdir() if p.is_dir()):
            subprocess.run([sys.executable, "-m", "tools.thought_report", str(d)], cwd=str(T.ROOT), check=False)
    print(f"wrote {out}")
    print(json.dumps(_jsonable({k: res["metrics"][k] for k in list(res["metrics"])[:8]}), indent=1))
    print("checks failed:", res["checks_failed"] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
