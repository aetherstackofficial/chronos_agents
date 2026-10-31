"""Phase 4 execution logic used by the live agents (pure, no model imports).

* ``VolumeCurve`` -- expected NSE volume per minute from the real-data curve.
* ``PovSlicer``   -- a whale's parent/child order book-keeping and slicing.
* ``skew_ticks`` / ``snap`` -- the market-maker inventory skew rule.

All rules come from agents/config/phase4_execution.yaml (frozen 2026-10-29).

Fill accounting. Every engine ack (ORDER, CANCEL_ALL) carries the engine
ledger's authoritative ``pos`` and ``cash``. A whale trades only through its
parent, so a parent's filled quantity is the position change since the
parent opened and its average price is -cash change / filled (the engine has
no fees). Fills of a child that rests and is hit later in its minute appear
at the next ack. A parent is always closed with CANCEL_ALL, so its final
position comes from that ack and no fill is attributed to two parents.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)
_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")


def load_config() -> dict:
    with open(os.path.join(_CFG_DIR, "phase4_execution.yaml")) as fh:
        return yaml.safe_load(fh)


CONFIG = load_config()


# --------------------------------------------------------------------------- #
# expected volume
# --------------------------------------------------------------------------- #
class VolumeCurve:
    """Expected NSE shares traded in a session bar (0-based minute index)."""

    def __init__(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(_CFG_DIR, CONFIG["whale_pov"]["volume_curves_file"])
        with open(path) as fh:
            d = json.load(fh)
        self._symbols = d["symbols"]
        self._fallback = CONFIG["whale_pov"]["unknown_symbol_fallback"]
        self._warned: set = set()

    def curve_for(self, symbol: str) -> dict:
        c = self._symbols.get(symbol)
        if c is None:
            if symbol not in self._warned:
                logger.warning("no volume curve for %s; using %s", symbol, self._fallback)
                self._warned.add(symbol)
            c = self._symbols[self._fallback]
        return c

    def expected(self, symbol: str, minute_idx: int) -> float:
        c = self.curve_for(symbol)
        n = len(c["minute_weights"])
        return c["adv_shares"] / n * c["minute_weights"][min(max(minute_idx, 0), n - 1)]


# --------------------------------------------------------------------------- #
# parent / child slicing
# --------------------------------------------------------------------------- #
@dataclass
class Parent:
    parent_id: str
    side: str                      # "BUY" | "SELL"
    target_pos: int                # whale position the parent works toward
    opened_day: int
    opened_bar: int
    reason: str
    pos0: Optional[int] = None     # set from the ledger when the parent opens
    cash0: Optional[float] = None
    children: int = 0
    child_qty_sent: int = 0
    revisions: int = 0
    status: str = "active"
    closed_day: Optional[int] = None
    closed_bar: Optional[int] = None
    filled: int = 0
    avg_price: Optional[float] = None
    shortfall: int = 0

    @property
    def sign(self) -> int:
        return 1 if self.side == "BUY" else -1

    def remaining(self, pos: int) -> int:
        """Shares still to trade in the parent's direction (>= 0)."""
        return max(0, self.sign * (self.target_pos - pos))


class PovSlicer:
    """Works one parent at a time as marketable-limit children under a cap.

    The owning agent calls ``retarget`` when its policy produces a new
    position target, ``tick_orders`` on every awake tick, and ``on_ack``
    with every engine ack. Orders are §4.1 dicts (``{"cancel_all": True}`` is
    the orchestrator's CANCEL_ALL marker).
    """

    def __init__(self, agent_id: str, curve: Optional[VolumeCurve] = None,
                 cfg: Optional[dict] = None) -> None:
        c = (cfg or CONFIG)["whale_pov"]
        self.agent_id = agent_id
        self.curve = curve or VolumeCurve()
        self.cap = float(c["participation_cap"])
        self.child_tif = int(c["child_tif_ticks"])
        self.max_remaining = int(c["max_parent_remaining"])
        self.last_child_bar = int(c["last_child_bar"])
        self.reason = c.get("reason", "WHALE_REBALANCE")
        self.parent: Optional[Parent] = None
        self.closing: List[Parent] = []        # closed, waiting for the CANCEL_ALL ack
        self.log: List[Parent] = []            # every closed parent, final numbers
        self.children_log: List[dict] = []     # (day, bar, qty, cap, expected_volume)
        self._n_parent = 0
        self.pos = 0
        self.cash = 0.0
        self._ledger_seen = False

    # -- ledger --------------------------------------------------------- #
    def on_ack(self, ack: dict) -> None:
        if ack.get("pos") is None:
            return                                   # rejects carry no ledger
        self.pos, self.cash = int(ack["pos"]), float(ack["cash"])
        self._ledger_seen = True
        if ack.get("status") == "CANCELLED":
            for p in self.closing:
                self._finalise(p)
            self.closing = []
            if self.parent is not None and self.parent.pos0 is None:
                self.parent.pos0, self.parent.cash0 = self.pos, self.cash

    def _finalise(self, p: Parent) -> None:
        p.filled = p.sign * (self.pos - p.pos0) if p.pos0 is not None else 0
        dq = self.pos - (p.pos0 or 0)
        p.avg_price = (-(self.cash - p.cash0) / dq) if dq and p.cash0 is not None else None
        p.shortfall = 0 if p.status == "completed" else p.remaining(self.pos)
        self.log.append(p)

    # -- parents -------------------------------------------------------- #
    def _close(self, status: str, day: int, bar: int) -> dict:
        p = self.parent
        p.status, p.closed_day, p.closed_bar = status, day, bar
        self.closing.append(p)
        self.parent = None
        return {"cancel_all": True}

    def retarget(self, target_pos: int, side: str, day: int, bar: int) -> List[dict]:
        """New policy target. Same side: update the parent in place (fills
        stay, later children follow the new target). Opposite side: close the
        parent (its fills stay) and open a new one."""
        orders: List[dict] = []
        p = self.parent
        if p is not None and p.side == side:
            if target_pos != p.target_pos:
                p.target_pos, p.revisions = target_pos, p.revisions + 1
            return orders
        if p is not None:
            orders.append(self._close("replaced_opposite_side", day, bar))
        self._n_parent += 1
        np_ = Parent(f"{self.agent_id}:P{self._n_parent}", side, target_pos, day, bar, self.reason)
        if self.closing:
            pass                                 # pos0 comes from the pending CANCEL_ALL ack
        elif not self._ledger_seen:
            # Never seen an ack, so pos/cash are unknown (cash starts at the
            # engine's seed, not 0): read them from a CANCEL_ALL ack first.
            orders.append({"cancel_all": True})
        else:
            np_.pos0, np_.cash0 = self.pos, self.cash
        self.parent = np_
        return orders

    def tick_orders(self, tick: dict) -> List[dict]:
        """Session-close handling, completion and the next child."""
        day, bar = int(tick["day_count"]), int(tick["market_minute"])
        p = self.parent
        if p is None:
            return []
        if p.opened_day != day or bar >= self.last_child_bar:
            return [self._close("session_close", day, bar)]
        rem = min(p.remaining(self.pos), self.max_remaining)
        if rem <= 0:
            return [self._close("completed", day, bar)]
        # The child trades in bar ``bar + 1`` = 0-based minute index ``bar``.
        ev = self.curve.expected(tick.get("symbol", ""), bar)
        cap_qty = int(self.cap * ev)
        qty = min(rem, cap_qty)
        price = tick.get("best_ask") if p.side == "BUY" else tick.get("best_bid")
        if qty <= 0 or price is None:
            return []
        p.children += 1
        p.child_qty_sent += qty
        self.children_log.append({"day": day, "bar": bar + 1, "qty": qty, "cap": cap_qty,
                                  "expected_volume": ev, "parent": p.parent_id})
        return [{"action": p.side, "type": "LIMIT", "price": float(price), "qty": int(qty),
                 "tif_ticks": self.child_tif,
                 "client_order_id": f"{p.parent_id}:C{p.children}"}]


# --------------------------------------------------------------------------- #
# market-maker inventory skew
# --------------------------------------------------------------------------- #
def skew_ticks(inventory: int, max_position: int, cfg: Optional[dict] = None) -> int:
    """shift = -round(k * q / q_max) ticks, clamped to +/- max_ticks."""
    c = (cfg or CONFIG)["mm_inventory_skew"]
    if max_position <= 0:
        return 0
    s = -round(c["k_ticks"] * inventory / max_position)
    return int(max(-c["max_ticks"], min(c["max_ticks"], s)))


def snap(price: float, tick: float) -> float:
    """Round to the engine tick grid (removes the old 2-decimal rounding)."""
    if not tick or tick <= 0:
        return price
    n = round(price / tick)
    dp = max(0, -int(f"{tick:e}".split("e")[1]))
    return round(n * tick, dp + 2)
