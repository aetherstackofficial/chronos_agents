"""Price levels agents react to (Phase 3, rules in agents/config/retail_personalities.yaml `levels:`).

Support and resistance are never forced onto the price. They appear because
agents put real orders at the same prices, and the book is deeper there.

A level comes from two places:

* round numbers, computed from the price alone (no history needed), and
* history, read from a ``MarketContext`` when one exists and is ``ready``:
  previous-day high, low and close, today's high and low, confirmed swing
  points. Only the attributes the interface promises are read, defensively,
  so a context without some of them just contributes fewer levels.

Pure functions only: no engine, no models, no global random (the caller
passes its own ``random.Random``).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from agents.execution import snap


@dataclass(frozen=True)
class Level:
    price: float
    strength: float
    kind: str                       # "round" | "prev_day_high" | "swing" | ...


# --------------------------------------------------------------------------- #
# building the level map
# --------------------------------------------------------------------------- #
def level_unit(price: float, tick: float) -> float:
    """Base round-number step: 1/1000 of the price's decade, never below a tick
    (Rs 1 at Rs 1,354, Rs 0.1 at Rs 190, Rs 10 at Rs 14,000)."""
    return max(10.0 ** (math.floor(math.log10(price)) - 3), tick)


def round_levels(price: float, tick: float, window_bps: float, cfg: dict) -> List[Level]:
    """Round numbers within `window_bps` of `price`. Strength is that of the
    largest configured multiple of the base unit that divides the level."""
    rn = cfg["levels"]["round_numbers"]
    if price <= 0:
        return []
    u = level_unit(price, tick)
    lo, hi = price * (1.0 - window_bps / 1e4), price * (1.0 + window_bps / 1e4)
    k0, k1 = math.ceil(lo / u - 1e-9), math.floor(hi / u + 1e-9)
    if k1 - k0 + 1 > int(rn["max_levels"]):
        return []                   # unit far too fine for this window; skip rather than flood
    tiers = sorted(zip(rn["multiples"], rn["strengths"]), reverse=True)
    out = []
    for k in range(k0, k1 + 1):
        if k <= 0:
            continue
        strength = next((s for m, s in tiers if k % int(m) == 0), tiers[-1][1])
        out.append(Level(snap(k * u, tick), float(strength), "round"))
    return out


def _price(x) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 else None


def context_levels(context, cfg: dict) -> List[Level]:
    """Levels from a ready MarketContext (empty when there is none)."""
    if context is None or not getattr(context, "ready", False):
        return []
    st, lv = cfg["levels"]["context_strengths"], cfg["levels"]
    out: List[Level] = []

    def add(price, key, kind):
        p = _price(price)
        if p is not None:
            out.append(Level(p, float(st[key]), kind))

    prev = getattr(context, "prev_day", None)
    if isinstance(prev, dict):
        add(prev.get("h"), "prev_day_high", "prev_day_high")
        add(prev.get("l"), "prev_day_low", "prev_day_low")
        add(prev.get("c"), "prev_day_close", "prev_day_close")
    add(getattr(context, "day_high", None), "day_high", "day_high")
    add(getattr(context, "day_low", None), "day_low", "day_low")
    n = int(lv["swing_count"])
    for name in ("swing_highs", "swing_lows"):
        fn = getattr(context, name, None)
        if callable(fn):
            for p in list(fn())[-n:]:
                add(p, "swing", "swing")
    return out


def build_levels(price: float, tick: float, context, cfg: dict,
                 window_bps: Optional[float] = None) -> List[Level]:
    """All levels within the window around `price`, same prices merged (their
    strengths add: a previous-day high that is also a round number is stronger)."""
    w = float(cfg["levels"]["max_distance_bps"] if window_bps is None else window_bps)
    lo, hi = price * (1.0 - w / 1e4), price * (1.0 + w / 1e4)
    merged: Dict[float, Tuple[float, str]] = {}
    for lvl in round_levels(price, tick, w, cfg) + context_levels(context, cfg):
        px = snap(lvl.price, tick)
        if not lo <= px <= hi:
            continue
        s, k = merged.get(px, (0.0, lvl.kind))
        merged[px] = (s + lvl.strength, k if k == lvl.kind else f"{k}+{lvl.kind}")
    return [Level(px, s, k) for px, (s, k) in sorted(merged.items())]


# --------------------------------------------------------------------------- #
# choosing a level
# --------------------------------------------------------------------------- #
def pick_level(side: int, best_bid: Optional[float], best_ask: Optional[float],
               last: float, tick: float, levels: Sequence[Level],
               rng: random.Random, cfg: dict) -> Optional[float]:
    """A passive limit price at a level, or None when no level qualifies.

    A buy may sit anywhere from `max_distance_bps` below the best bid up to
    just under the best ask (inside the spread counts: it would simply become
    the new best bid); a sell mirrors that. Among the candidates, stronger
    levels and levels nearer the touch are likelier: weight = strength *
    exp(-distance_bps / decay_bps).
    """
    lv = cfg["levels"]
    bid, ask = best_bid or last, best_ask or last
    reach = float(lv["max_distance_bps"]) / 1e4
    decay = float(lv["decay_bps"])
    cands, weights = [], []
    for l in levels:
        if side > 0:
            if not bid * (1.0 - reach) <= l.price <= ask - tick:
                continue
            dist = max(0.0, bid - l.price)
            ref = bid
        else:
            if not bid + tick <= l.price <= ask * (1.0 + reach):
                continue
            dist = max(0.0, l.price - ask)
            ref = ask
        cands.append(l.price)
        weights.append(l.strength * math.exp(-(dist / ref * 1e4) / decay))
    if not cands:
        return None
    return snap(rng.choices(cands, weights=weights)[0], tick)


def nearest_level(target: float, levels: Sequence[Level], max_bps: float,
                  lower: Optional[float] = None, upper: Optional[float] = None) -> Optional[Level]:
    """The level closest to `target` within `max_bps`, optionally limited to
    prices in [lower, upper]; the stronger one wins an exact tie."""
    best = None
    for l in levels:
        if (lower is not None and l.price < lower) or (upper is not None and l.price > upper):
            continue
        d = abs(l.price - target)
        if d / target * 1e4 > max_bps:
            continue
        if best is None or (d, -l.strength) < (best[0], -best[1].strength):
            best = (d, l)
    return None if best is None else best[1]


def anchor_exit_prices(side: int, entry: float, take_profit_pct: float, stop_loss_pct: float,
                       tick: float, context, cfg: dict) -> Tuple[float, float]:
    """(take-profit price, stop price) for a position, moved onto levels.

    The target goes just SHORT of the level nearest the drawn target (traders
    want to be filled before the level holds), the stop just BEYOND the level
    nearest the drawn stop (where stops pile up). 0.0 for either one means "no
    level close enough: keep the percentage rule".
    """
    lv = cfg["levels"]
    levels = build_levels(entry, tick, context, cfg, window_bps=float(lv["exit_window_bps"]))
    snap_bps = float(lv["exit_snap_bps"])
    short = float(lv["target_short_of_level_ticks"]) * tick
    beyond = float(lv["stop_beyond_level_ticks"]) * tick
    tp_target = entry * (1.0 + side * take_profit_pct / 100.0)
    sl_target = entry * (1.0 - side * stop_loss_pct / 100.0)
    profit_side = (entry + tick, None) if side > 0 else (None, entry - tick)
    loss_side = (None, entry - tick) if side > 0 else (entry + tick, None)
    tp = nearest_level(tp_target, levels, snap_bps, *profit_side)
    sl = nearest_level(sl_target, levels, snap_bps, *loss_side)
    tp_price = snap(tp.price - side * short, tick) if tp else 0.0
    stop_price = snap(sl.price - side * beyond, tick) if sl else 0.0
    # A level must not turn a target into a loss or a stop into a gain.
    if tp_price and side * (tp_price - entry) <= 0:
        tp_price = 0.0
    if stop_price and side * (stop_price - entry) >= 0:
        stop_price = 0.0
    return tp_price, stop_price
