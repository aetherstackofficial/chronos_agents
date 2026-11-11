"""Shared action decoder: model outputs in [-1, 1] -> engine orders.

The same functions are used in training and live, so one output number always
means one thing. Pure: no models, no engine, no I/O. Orders use the live
order-dict format (``{"cancel_all": True}`` or the §4.1 ORDER fields without
``msg``/``agent_id``). Tick size comes from ``tick["tick_size"]`` (PROTOCOL
§5.1), exactly as live_agents does.

Whale (3 outputs): target position, participation, aggressiveness.
Market maker (4 outputs): bid offset, ask offset, bid size, ask size.
``notes`` lists every rule that changed the raw request, for the decision log.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from agents.execution import snap
from agents.execution_v2 import child_price, child_qty, expected_minute_volume

DEFAULTS: dict = {
    # whale: entry/exit band (fractions of capacity)
    "whale_entry_threshold": 0.20,    # open only if |target| > this
    "whale_exit_threshold": 0.05,     # close if |target| < this
    "whale_flat_fraction": 0.05,      # |position| < this x capacity counts as flat
    "whale_min_change": 0.05,         # ignore a change smaller than this x capacity
    # whale: output 2 -> participation, linear [-1,1] -> [min,max]
    "participation_min": 0.01,
    "participation_max": 0.15,
    # whale: output 3 -> aggressiveness 0..1; 1 = this many ticks past the opposite touch
    "aggr_through_ticks": 3,
    "child_tif_ticks": 1,
    # same cap the Phase 4 slicer applies to a parent's unfilled remainder
    # (agents/config/phase4_execution.yaml whale_pov.max_parent_remaining); None = off
    "max_parent_remaining": 10_000,
    # market maker: output 1/2 -> integer tick offset, linear [-1,1] -> [min,max]
    "mm_offset_min": -1,              # -1 = one tick inside the spread
    "mm_offset_max": 10,
    "mm_inside_min_spread_ticks": 2,  # inside quote allowed only at this spread or wider
    "mm_size_floor": 0.05,            # below this x max_quote_size -> no quote
    "mm_tif_ticks": 1,
    "mm_fallback_half_spread_ticks": 1,   # missing book: last_price +/- this many ticks
    # when the tick carries no tick_size (live code treats it as always present)
    "fallback_tick_size": 0.05,
}


@dataclass
class AgentState:
    position: int
    cash: float
    capacity: int
    max_quote_size: int = 0
    has_resting: bool = False


@dataclass
class DecodeResult:
    orders: list = field(default_factory=list)
    intent: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    raw: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cfg(cfg: Optional[dict]) -> dict:
    return {**DEFAULTS, **(cfg or {})}


def _clip_action(action: Sequence, n: int, notes: list) -> list:
    out = []
    for i in range(n):
        v = float(action[i])
        if math.isnan(v):
            notes.append(f"clip: output {i} is NaN -> 0")
            v = 0.0
        elif v < -1.0 or v > 1.0:
            notes.append(f"clip: output {i} {v:.3g} -> {min(1.0, max(-1.0, v)):g}")
            v = min(1.0, max(-1.0, v))
        out.append(v)
    return out


def _tick_size(tick: dict, c: dict, notes: list) -> float:
    ts = float(tick.get("tick_size") or 0.0)
    if ts <= 0:
        ts = float(c["fallback_tick_size"])
        notes.append(f"tick: no tick_size in tick, using {ts:g}")
    return ts


def _lerp01(x: float) -> float:
    return (x + 1.0) / 2.0


def _round_half_up(v: float) -> int:
    return int(math.floor(v + 0.5))


# --------------------------------------------------------------------------- #
# whale
# --------------------------------------------------------------------------- #
def decode_whale(action, state: AgentState, tick: dict, cfg: Optional[dict] = None) -> DecodeResult:
    c = _cfg(cfg)
    notes: list = []
    raw = _clip_action(action, 3, notes)
    ts = _tick_size(tick, c, notes)
    cap, pos = int(state.capacity), int(state.position)

    target = raw[0]
    participation = round(c["participation_min"] + _lerp01(raw[1]) * (c["participation_max"] - c["participation_min"]), 6)
    aggr = _lerp01(raw[2])
    entry, exit_ = c["whale_entry_threshold"], c["whale_exit_threshold"]

    # a. entry / exit band
    flat = abs(pos) < c["whale_flat_fraction"] * cap
    t = target
    if flat:
        if abs(t) <= entry:
            if t != 0.0:
                notes.append(f"band: |target| {abs(t):.2f} <= entry {entry:.2f} -> stay flat")
            t = 0.0
    else:
        if abs(t) < exit_:
            if t != 0.0:
                notes.append(f"band: |target| {abs(t):.2f} < exit {exit_:.2f} -> close")
            t = 0.0
        elif t * pos < 0 and abs(t) <= entry:
            notes.append(f"band: flip needs |target| > {entry:.2f}, got {abs(t):.2f} -> close to flat")
            t = 0.0
    target_shares = int(round(t * cap))

    # b. ignore a small change
    delta = target_shares - pos
    if delta != 0 and abs(delta) < c["whale_min_change"] * cap:
        notes.append(f"filter: change {abs(delta) / cap * 100:.0f}% < {c['whale_min_change'] * 100:.0f}%")
        target_shares = pos

    # c. caps: capacity, then cash for a long target (shorts allowed)
    if abs(target_shares) > cap:
        target_shares = cap if target_shares > 0 else -cap
        notes.append("cap: capacity")
    ask = tick.get("best_ask") or tick.get("last_price")
    if target_shares > pos and target_shares > 0 and ask and float(ask) > 0:
        affordable = pos + int(max(float(state.cash), 0.0) // float(ask))
        if target_shares > affordable:
            target_shares = max(affordable, pos)
            notes.append("cap: cash")

    delta = target_shares - pos

    # d. parent intent and this minute's child
    side = "BUY" if delta > 0 else "SELL"
    ev = expected_minute_volume(tick)
    if delta != 0 and ev <= 0:
        notes.append("volume: no volume_ma or step_volume -> no child")
    elif delta != 0 and not (tick.get("volume_ma") or 0) > 0:
        notes.append("volume: volume_ma missing, using step_volume")
    qty = child_qty(abs(delta), participation, ev, c["max_parent_remaining"]) if delta else 0
    if delta != 0 and ev > 0 and qty == 0:
        notes.append("child: participation x volume < 1 share -> no child")
    price = None
    bid, ask_px = tick.get("best_bid"), tick.get("best_ask")
    if qty > 0:
        if bid is None or ask_px is None:
            notes.append("book: best bid/ask missing -> no child")
            qty = 0
        else:
            price = child_price(side, aggr, float(bid), float(ask_px), ts, c["aggr_through_ticks"])

    orders: list = []
    if state.has_resting:
        orders.append({"cancel_all": True})
        notes.append("cancel: previous child still resting")
    if qty > 0:
        orders.append({"action": side, "type": "LIMIT", "price": price, "qty": int(qty),
                       "tif_ticks": int(c["child_tif_ticks"])})
    intent = {"target_shares": target_shares, "delta_shares": delta,
              "participation": participation, "aggressiveness": aggr,
              "side": side if delta else None, "child_qty": int(qty), "child_price": price,
              "target_frac": raw[0], "expected_volume": ev, "tick_size": ts}
    return DecodeResult(orders, intent, notes, raw)


# --------------------------------------------------------------------------- #
# market maker
# --------------------------------------------------------------------------- #
def decode_mm(action, state: AgentState, tick: dict, cfg: Optional[dict] = None) -> DecodeResult:
    c = _cfg(cfg)
    notes: list = []
    raw = _clip_action(action, 4, notes)
    ts = _tick_size(tick, c, notes)
    pos, cap = int(state.position), int(state.capacity)
    omin, omax = c["mm_offset_min"], c["mm_offset_max"]

    off_req = [_round_half_up(omin + _lerp01(raw[i]) * (omax - omin)) for i in (0, 1)]
    cancel = [{"cancel_all": True}]

    # book (fallback: last_price +/- half-spread as the touch)
    bb, ba = tick.get("best_bid"), tick.get("best_ask")
    if bb is None or ba is None:
        last = tick.get("last_price")
        if last is None or float(last) <= 0:
            notes.append("book: no best bid/ask and no last_price -> no quotes")
            return DecodeResult(cancel, {"bid_px": None, "ask_px": None, "bid_qty": 0, "ask_qty": 0,
                                         "offsets": off_req, "tick_size": ts}, notes, raw)
        half = c["mm_fallback_half_spread_ticks"] * ts
        bb, ba = snap(float(last) - half, ts), snap(float(last) + half, ts)
        notes.append(f"book: best bid/ask missing, touch = last {float(last):g} +/- {c['mm_fallback_half_spread_ticks']} tick")
    bb, ba = float(bb), float(ba)
    spread_ticks = _round_half_up((ba - bb) / ts)

    # offsets: inside quote only when the spread allows it
    offs = list(off_req)
    for i, name in enumerate(("bid", "ask")):
        if offs[i] < 0 and spread_ticks < c["mm_inside_min_spread_ticks"]:
            notes.append(f"mm: inside quote not allowed, spread {spread_ticks} tick{'s' if spread_ticks != 1 else ''} ({name})")
            offs[i] = 0
    bid_px = snap(bb - offs[0] * ts, ts)
    ask_px = snap(ba + offs[1] * ts, ts)

    # never cross or touch the opposite side
    if bid_px >= ba:
        bid_px = snap(ba - ts, ts)
        notes.append("mm: bid would reach best ask -> pushed back")
    if ask_px <= bb:
        ask_px = snap(bb + ts, ts)
        notes.append("mm: ask would reach best bid -> pushed back")
    if bid_px >= ask_px:     # both inside a 2-tick spread land on the same price
        ask_px = snap(bid_px + ts, ts)
        notes.append("mm: own quotes would meet -> ask pushed back")

    # sizes
    mq = int(state.max_quote_size)
    sizes = []
    for i, (name, room) in enumerate((("bid", cap - pos), ("ask", cap + pos))):
        raw_size = _lerp01(raw[2 + i]) * mq
        size = int(math.floor(raw_size + 1e-9))
        if mq <= 0:
            notes.append(f"mm: max_quote_size 0 -> no {name} quote")
            size = 0
        elif raw_size < c["mm_size_floor"] * mq - 1e-9:
            notes.append(f"mm: {name} size below {c['mm_size_floor'] * 100:.0f}%")
            size = 0
        elif size > max(room, 0):
            notes.append(f"cap: position limit {name} {size} -> {max(room, 0)}")
            size = max(room, 0)
        sizes.append(size)
    bid_qty, ask_qty = sizes

    orders = list(cancel)
    tif = int(c["mm_tif_ticks"])
    if bid_qty > 0:
        orders.append({"action": "BUY", "type": "LIMIT", "price": bid_px, "qty": bid_qty, "tif_ticks": tif})
    if ask_qty > 0:
        orders.append({"action": "SELL", "type": "LIMIT", "price": ask_px, "qty": ask_qty, "tif_ticks": tif})
    intent = {"bid_px": bid_px if bid_qty > 0 else None, "ask_px": ask_px if ask_qty > 0 else None,
              "bid_qty": bid_qty, "ask_qty": ask_qty, "offsets": offs, "offsets_requested": off_req,
              "spread_ticks": spread_ticks, "tick_size": ts}
    return DecodeResult(orders, intent, notes, raw)
