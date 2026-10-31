"""Rule-based retail crowd model.

This module contains only the live inference brain for the retail swarm.
Offline backtests of this heuristic (CSV-driven evaluation harnesses) live in
the training repository, not here — the live swarm only needs `predict()`.

Phase 2: every retail agent now owns a `RetailPersonality`, drawn once at
creation from `agents/config/retail_personalities.yaml` (frozen before any
Phase 2 run). The personality is a pure function of (run seed, agent id), so a
seeded run always builds the same crowd.
"""

import math
import os
import random
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import yaml

from agents.execution import snap

_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")

# Fixed order: the type draw walks the cumulative mix in this order.
_TYPES = ("momentum", "mean_reverter", "noise")
_EXIT_STYLES = ("time_based", "target_stop", "trailing_stop", "signal_reversal")


def load_config() -> dict:
    with open(os.path.join(_CFG_DIR, "retail_personalities.yaml")) as fh:
        return yaml.safe_load(fh)


CONFIG = load_config()


def draw(spec: dict, rng: random.Random) -> float:
    """One draw from a distribution spec of the yaml (`dist:` + parameters).

    Shared by the personality sampler and, later, the per-trade exit draws.
    """
    dist = spec["dist"]
    if dist == "uniform":
        return rng.uniform(spec["low"], spec["high"])
    if dist == "uniform_int":
        return rng.randint(spec["low"], spec["high"])
    if dist == "normal":
        value = rng.gauss(spec["mean"], spec["std"])
    elif dist == "lognormal":
        value = rng.lognormvariate(math.log(spec["median"]), spec["sigma"])
    elif dist == "pareto":
        # paretovariate is >= 1, so the draw is >= `minimum`; heavy right tail.
        value = spec["minimum"] * rng.paretovariate(spec["shape"])
        return min(value, spec["clip_max"])
    else:
        raise ValueError(f"unknown distribution: {dist!r}")
    if "clip" in spec:
        value = min(max(value, spec["clip"][0]), spec["clip"][1])
    return value


@dataclass(frozen=True)
class RetailPersonality:
    """Everything that makes one retail agent different from another."""
    agent_id: str
    kind: str                       # momentum | mean_reverter | noise
    rsi_center: float               # RSI points (0-100)
    conviction_threshold: float     # act only when |conviction| passes this
    cooldown_ticks: int             # engine minutes between this agent's orders
    risk_appetite: float            # share of cash risked per order
    start_cash: float               # rupees; the engine seeds the ledger with it
    max_position_cap: int           # hard ceiling in shares (long or short)
    leverage_limit: float           # position value may not exceed cash * this
    exit_preference: tuple          # weights over _EXIT_STYLES, sums to 1

    def max_shares_at_price(self, price: float) -> int:
        """Largest position (shares) this agent may hold at `price`."""
        if price <= 0:
            return 0
        affordable = int(self.start_cash * self.leverage_limit / price)
        return max(0, min(self.max_position_cap, affordable))


def sample_personality(agent_id: str, run_seed, cfg: Optional[dict] = None) -> RetailPersonality:
    """Draw one agent's personality. Deterministic in (run_seed, agent_id).

    String seeding hashes the text, so it is stable across processes and does
    not depend on the order agents are created in.
    """
    cfg = CONFIG if cfg is None else cfg
    rng = random.Random(f"{run_seed}:{agent_id}:personality")
    pers, wealth = cfg["personality"], cfg["wealth"]

    roll, acc, kind = rng.random(), 0.0, _TYPES[-1]
    for name in _TYPES:
        acc += pers["type_mix"][name]
        if roll < acc:
            kind = name
            break

    rsi_center = draw(pers["rsi_center"], rng)
    threshold = draw(pers["conviction_threshold"], rng)
    cooldown = int(draw(pers["cooldown_ticks"], rng))
    risk = draw(cfg["orders"]["risk_appetite"], rng)

    gammas = [rng.gammavariate(a, 1.0) for a in cfg["exits"]["style_preference"]["alpha"]]
    total = sum(gammas)
    exit_pref = tuple(g / total for g in gammas)

    cash_cfg = wealth["starting_cash"]
    if cash_cfg["source"] == "per_agent":
        start_cash = float(round(draw(cash_cfg["per_agent"], rng)))
    else:
        start_cash = float(cash_cfg["engine_default"])

    mp = wealth["max_position"]
    cap = min(int(mp["hard_cap_shares"]),
              int(mp["reference_whale_min_capacity"] * mp["fraction_of_whale"]))

    return RetailPersonality(
        agent_id=agent_id, kind=kind, rsi_center=rsi_center,
        conviction_threshold=threshold, cooldown_ticks=cooldown,
        risk_appetite=risk, start_cash=start_cash, max_position_cap=cap,
        leverage_limit=float(mp["leverage_limit"]), exit_preference=exit_pref)


# --------------------------------------------------------------------------- #
# Phase 2 step 4: entry signal and order size
# --------------------------------------------------------------------------- #

def conviction(rsi: float, p: RetailPersonality, slope: float) -> float:
    """Signed buy pressure in -1..+1 from RSI (points, 0-100).

    A smooth logistic around the agent's own RSI centre, not an on/off line.
    Momentum agents buy strength (RSI above their centre); mean reverters buy
    weakness (RSI below it). Positive = buy, negative = sell/short. Noise
    agents have no signal and return 0.
    """
    if p.kind == "noise":
        return 0.0
    raw = 2.0 / (1.0 + math.exp(-slope * (rsi - p.rsi_center))) - 1.0
    return -raw if p.kind == "mean_reverter" else raw


def order_qty(p: RetailPersonality, conviction_abs: float, price: float,
              cash: float, side: int, cfg: Optional[dict] = None,
              size_multiplier: float = 1.0, max_qty: Optional[int] = None) -> int:
    """Shares for one entry: risk_appetite * cash * |conviction| * multiplier
    / price, at least min_qty, never above the agent's position cap or
    `max_qty` (the participation cap), and a buy never above what its cash
    can pay for at `price`. 0 means "cannot trade"."""
    cfg = CONFIG if cfg is None else cfg
    if price <= 0:
        return 0
    qty = max(int(cfg["orders"]["min_qty"]),
              int(p.risk_appetite * cash * conviction_abs * size_multiplier / price))
    qty = min(qty, p.max_shares_at_price(price))
    if max_qty is not None:
        qty = min(qty, max_qty)
    if side > 0:
        qty = min(qty, int(cash // price))
    return max(0, qty)


def limit_price(side: int, best_bid: Optional[float], best_ask: Optional[float],
                last_price: float, tick_size: float, offset_ticks: int) -> float:
    """Passive limit price on the agent's OWN side of the book: a buy joins the
    best bid (or sits `offset_ticks` below it), a sell joins the best ask (or
    sits above it). Without a quote on that side it falls back to the last
    price. Snapped to the venue tick, never below one tick."""
    if side > 0:
        px = (best_bid or last_price) - offset_ticks * tick_size
    else:
        px = (best_ask or last_price) + offset_ticks * tick_size
    return max(snap(px, tick_size), tick_size)


# --------------------------------------------------------------------------- #
# Phase 2 step 3: open position and exit plan
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ExitPlan:
    """How one trade will be closed. Drawn fresh for every trade, so holding
    time varies trade to trade as well as agent to agent."""
    style: str                      # one of _EXIT_STYLES
    max_hold_ticks: int
    hold_ticks: int = 0             # time_based
    take_profit_pct: float = 0.0    # target_stop
    stop_loss_pct: float = 0.0      # target_stop
    trail_pct: float = 0.0          # trailing_stop
    fear_take_profit_pct: float = 0.0   # >0: "scared of losing its profit", sells at this small gain


@dataclass
class Position:
    """One open trade of one agent (an agent holds at most one at a time)."""
    side: int                       # +1 long, -1 short
    qty: int
    entry_price: float
    entry_t: int                    # engine minutes
    plan: ExitPlan
    best_price: float               # best price since entry (high for a long, low for a short)
    exited_qty: int = 0
    exit_notional: float = 0.0

    def mark(self, price: float) -> None:
        self.best_price = max(self.best_price, price) if self.side > 0 else min(self.best_price, price)


def draw_exit_plan(p: RetailPersonality, rng: random.Random, cfg: Optional[dict] = None) -> ExitPlan:
    """Pick this trade's exit style from the agent's own preference, then draw
    that style's parameters. Noise agents always use their configured style."""
    cfg = CONFIG if cfg is None else cfg
    ex = cfg["exits"]
    if p.kind == "noise":
        style = cfg["personality"]["noise_agent"]["exit_style"]
    else:
        roll, acc, style = rng.random(), 0.0, _EXIT_STYLES[-1]
        for name, weight in zip(_EXIT_STYLES, p.exit_preference):
            acc += weight
            if roll < acc:
                style = name
                break
    cap = int(ex["max_hold_ticks"])
    if style == "time_based":
        plan = ExitPlan(style, cap, hold_ticks=max(1, int(round(draw(ex["time_based"]["hold_ticks"], rng)))))
    elif style == "target_stop":
        plan = ExitPlan(style, cap,
                        take_profit_pct=draw(ex["target_stop"]["take_profit_pct"], rng),
                        stop_loss_pct=draw(ex["target_stop"]["stop_loss_pct"], rng))
    elif style == "trailing_stop":
        plan = ExitPlan(style, cap, trail_pct=draw(ex["trailing_stop"]["trail_pct"], rng))
    else:
        plan = ExitPlan(style, cap)
    # Emotion: with some probability the trader is afraid of giving the profit back.
    fear = cfg["emotions"]["profit_fear"]
    if rng.random() < fear["probability"]:
        plan = replace(plan, fear_take_profit_pct=draw(fear["early_take_profit_pct"], rng))
    return plan


def exit_reason(pos: Position, price: float, t: int, signal: float,
                threshold: float) -> Optional[str]:
    """Why this position should be closed now, or None to keep holding.

    `signal` is the agent's current conviction (positive = buy pressure);
    `threshold` its own conviction threshold. Call pos.mark(price) first.
    """
    plan, held = pos.plan, t - pos.entry_t
    if held >= plan.max_hold_ticks:
        return "max_hold"
    move_pct = pos.side * (price - pos.entry_price) / pos.entry_price * 100.0
    if plan.fear_take_profit_pct and move_pct >= plan.fear_take_profit_pct:
        return "fear_profit"          # overrides the normal exit when it comes sooner
    if plan.style == "time_based":
        return "time" if held >= plan.hold_ticks else None
    if plan.style == "target_stop":
        if move_pct >= plan.take_profit_pct:
            return "take_profit"
        if move_pct <= -plan.stop_loss_pct:
            return "stop_loss"
        return None
    if plan.style == "trailing_stop":
        pullback_pct = pos.side * (pos.best_price - price) / pos.best_price * 100.0
        return "trailing" if pullback_pct >= plan.trail_pct else None
    if plan.style == "signal_reversal":
        return "reversal" if signal * pos.side <= -threshold else None
    return None


class HeuristicRetailSwarm:
    """The OLD shared rule-based retail brain (RSI 30/70 and OFI rules).

    Live retail agents no longer use it: since Phase 2 each agent decides from
    its own personality (`conviction`, `order_qty`, `draw_exit_plan`,
    `exit_reason` above). Kept because the training repo and older harnesses
    still import it.

    The 10% pure-noise rule here was sanctioned behavioural randomness per
    PROTOCOL.md §8 (agent behavioural noise only — never price noise).
    """

    def predict(self, obs: np.ndarray, deterministic: bool = True, rng=None) -> tuple:
        """Mimics the stable-baselines3 predict API: (action, state).

        `obs` is a batched (1, 6) observation; `deterministic` is accepted for
        API compatibility but the 10% noise rule always applies (sanctioned).
        `rng` (a ``random.Random``, one per agent) draws that noise; None
        falls back to the global numpy RNG.
        """
        # DummyVecEnv wraps obs in an extra array, so we grab index 0
        obs_data = obs[0]

        # Unpack the normalized observations from UniversalMarketEnv layout
        vwap_dist = obs_data[0]
        volatility = obs_data[1]
        volume_ratio = obs_data[2]
        rsi_norm = obs_data[3]  # 0.0 to 1.0 (50 RSI = 0.5)
        ofi = obs_data[4]       # Order Flow Imbalance (-1.0 to 1.0)

        action = 0.0  # Default to flat

        # Rule 1: The Basic Technical Analyst (Buy oversold, sell overbought)
        if rsi_norm < 0.30:
            action = 0.5   # Buy with 50% capacity
        elif rsi_norm > 0.70:
            action = -0.5  # Short with 50% capacity

        # Rule 2: The FOMO & Panic Traders (Overrides RSI)
        # If order flow is extremely toxic one way, retail piles in emotionally
        if ofi > 0.8:
            action = 0.8   # Aggressive FOMO Buy
        elif ofi < -0.8:
            action = -1.0  # Absolute Panic Sell

        # Rule 3: Pure Noise (10% of the time, the swarm does something random)
        if rng is None:
            if np.random.rand() < 0.10:
                action = np.random.uniform(-0.5, 0.5)
        elif rng.random() < 0.10:
            action = rng.uniform(-0.5, 0.5)

        return np.array([[action]]), None
