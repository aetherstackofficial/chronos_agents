"""Reward functions of the replay environment.

Every component is returned separately (``parts``); the reward is their sum.

Whale:  profit (differential Sharpe ratio, or scaled P&L as a fallback)
        - drawdown penalty - termination penalty - optional inaction nudge
        - optional switching penalty (w_switch x |target - position| / capacity).
MM:     (spread captured + 1-minute markout of each fill - its cost)
        - inventory penalty + uptime bonus, / capital, x reward_scale.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple


class DifferentialSharpe:
    """Differential Sharpe ratio (Moody & Saffell 2001).

    A_t = A + eta (R - A),  B_t = B + eta (R^2 - B)
    D_t = (B dA - A dB / 2) / (B - A^2)^{3/2}   with the PREVIOUS A, B
    dA = eta (R - A), dB = eta (R^2 - B). The variance B - A^2 is floored at
    ``var_floor`` and B starts at ``var_floor``; D is clipped to +-clip.
    """

    def __init__(self, eta: float, var_floor: float = 1e-8, clip: float = 10.0) -> None:
        self.eta = float(eta)
        self.var_floor = float(var_floor)
        self.clip = float(clip)
        self.reset()

    def reset(self) -> None:
        self.A = 0.0
        self.B = self.var_floor
        self.n = 0
        self.n_clipped = 0

    def step(self, R: float) -> Tuple[float, bool]:
        A, B, eta = self.A, self.B, self.eta
        dA = eta * (R - A)
        dB = eta * (R * R - B)
        var = max(B - A * A, self.var_floor)
        D = (B * dA - 0.5 * A * dB) / (var ** 1.5)
        self.A = A + dA
        self.B = B + dB
        clipped = abs(D) > self.clip
        if clipped:
            D = math.copysign(self.clip, D)
            self.n_clipped += 1
        self.n += 1
        return D, clipped


class WhaleReward:
    def __init__(self, cfg: dict, model_name: str, risk_tolerance: float) -> None:
        w = cfg["rewards"]["whale"]
        self.profit_term = w["profit_term"]
        if self.profit_term not in ("dsr", "pnl"):
            raise ValueError("profit_term must be 'dsr' or 'pnl'")
        self.pnl_scale = float(w["pnl_scale"])
        self.w_dd = float(w["w_dd"])
        self.dd_free = float(w["dd_free"])
        scale = w["dd_step_scale"]          # a number, or {model_name: number}
        self.dd_step_scale = float(scale[model_name] if isinstance(scale, dict) else scale)
        self.w_term = float(w["w_term"])
        self.w_flat = float(w["w_flat"])
        self.flat_fraction = float(w["flat_fraction"])
        sw = w.get("w_switch", 0.0)         # a number, or {model_name: number}
        self.w_switch = float(sw.get(model_name, 0.0) if isinstance(sw, dict) else sw)
        self.risk_tolerance = float(risk_tolerance)
        self.dsr = DifferentialSharpe(float(w["eta"][model_name]), float(w["dsr_var_floor"]),
                                      float(w["dsr_clip"]))

    def step(self, R: float, dd: float, position: float, capacity: float,
             terminated_dd: bool, switch: float = 0.0) -> Tuple[Dict[str, float], bool]:
        """parts, dsr_clipped. ``R`` = after-cost equity change / capital; ``switch`` =
        |new target - position| / capacity of the decision taken on this minute (0 on the
        minutes without a decision)."""
        clipped = False
        if self.profit_term == "dsr":
            profit, clipped = self.dsr.step(R)
        else:
            profit = self.pnl_scale * R
        pen = self.w_dd * self.dd_step_scale * (max(0.0, dd - self.dd_free) / self.risk_tolerance) ** 2
        flat = abs(position) < self.flat_fraction * capacity
        parts = {"profit": float(profit),
                 "drawdown": -float(pen),
                 "termination": -self.w_term if terminated_dd else 0.0,
                 "inaction": -self.w_flat if flat else 0.0,
                 "switch": -self.w_switch * float(switch)}
        return parts, clipped


class MMReward:
    """Per-fill markout reward. A fill made on step t is settled on step t+1
    against that step's mid (the mid one minute later); at the last step the
    still-open fills are settled against the current mid."""

    def __init__(self, cfg: dict, capital: float, capacity: float, max_quote_size: int) -> None:
        w = cfg["rewards"]["mm"]
        self.scale = float(w["reward_scale"])
        self.w_inv = float(w["w_inv"])
        self.inv_ticks = float(w["inv_penalty_ticks"])
        self.w_up = float(w["w_up"])
        self.w_term = float(w["w_term"])
        self.capital = float(capital)
        self.capacity = float(capacity)
        self.mqs = float(max_quote_size)
        # fills waiting for their markout: (signed_qty, price, fee)
        self.pending: List[Tuple[float, float, float]] = []

    def step(self, new_fills: List[Tuple[float, float, float]], mid: float, extra_cost: float,
             position: float, ofi: float, tick_size: float, two_sided: bool,
             terminated_dd: bool, final: bool) -> Dict[str, float]:
        """``new_fills`` = this step's fills (signed_qty, price, fee); ``extra_cost``
        = carry surcharge / closing cost in money."""
        gross = 0.0
        cost = float(extra_cost)
        for sq, px, fee in self.pending:
            gross += sq * (mid - px)
            cost += fee
        self.pending = []
        if final:
            for sq, px, fee in new_fills:
                gross += sq * (mid - px)
                cost += fee
        else:
            self.pending = list(new_fills)
        s = (position > 0) - (position < 0)
        inv = (position / self.capacity) ** 2 * (1.0 + max(0.0, -s * ofi))
        unit = tick_size * self.mqs                               # one tick on max_quote_size
        k = self.scale / self.capital
        return {"spread_markout": gross * k,
                "cost": -cost * k,
                "inventory": -self.w_inv * self.inv_ticks * unit * inv * k,
                "uptime": self.w_up * unit * k if two_sided else 0.0,
                "termination": -self.w_term if terminated_dd else 0.0}
