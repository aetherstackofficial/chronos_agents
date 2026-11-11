"""Phase 4 slicer with per-parent participation and aggressiveness.

``agents/execution.py`` (PovSlicer) is left untouched. This module adds:

* pure helpers ``expected_minute_volume``, ``child_qty``, ``child_price``
  -- the ONE implementation of "how big and at what price is this minute's
  child". ``agents/action_decode.decode_whale`` calls the same helpers, so
  training and live cannot drift apart.
* ``PovSlicerV2`` -- subclass of PovSlicer. ``retarget`` takes an optional
  ``participation`` and ``aggressiveness`` that apply to the parent;
  everything else (parent bookkeeping, ledger/ack accounting, CANCEL_ALL on
  close, session-close handling, ``max_parent_remaining``, rounding down to
  whole shares, tif, client_order_id) is inherited or copied unchanged.

Defaults reproduce the old behaviour: participation = the 15% cap from
phase4_execution.yaml, aggressiveness = None -> marketable at the touch
(buy at best ask, sell at best bid), expected volume from the NSE curve.
"""
from __future__ import annotations

from typing import List, Optional

from agents.execution import PovSlicer, snap

# Aggressiveness knots (0..1), equally spaced:
#   0   join own best price
#   1/3 one tick inside (own best +/- 1 tick, never beyond the opposite touch)
#   2/3 at the opposite touch (marketable) -- what PovSlicer always did
#   1   `through_ticks` ticks beyond the opposite touch
MARKETABLE_AGGRESSIVENESS = 2.0 / 3.0


def expected_minute_volume(tick: dict) -> float:
    """Expected shares for the coming minute from the tick payload:
    ``volume_ma``, else ``step_volume``, else 0."""
    for key in ("volume_ma", "step_volume"):
        v = tick.get(key)
        if v is not None and float(v) > 0:
            return float(v)
    return 0.0


def child_qty(remaining: int, participation: float, expected_volume: float,
              max_remaining: Optional[int] = None) -> int:
    """min(remaining, participation x expected volume), rounded down."""
    rem = min(remaining, max_remaining) if max_remaining is not None else remaining
    if rem <= 0:
        return 0
    return max(0, min(int(rem), int(participation * expected_volume)))


def child_price(side: str, aggressiveness: float, best_bid: float, best_ask: float,
                tick_size: float, through_ticks: int = 3) -> float:
    """Piecewise-linear price through the four knots, snapped to the tick grid."""
    a = min(1.0, max(0.0, float(aggressiveness)))
    if side == "BUY":
        pts = [best_bid, min(best_bid + tick_size, best_ask), best_ask,
               best_ask + through_ticks * tick_size]
    else:
        pts = [best_ask, max(best_ask - tick_size, best_bid), best_bid,
               best_bid - through_ticks * tick_size]
    t = a * 3.0
    i = min(int(t), 2)
    px = pts[i] + (t - i) * (pts[i + 1] - pts[i])
    return float(snap(px, tick_size))


class PovSlicerV2(PovSlicer):
    """PovSlicer whose children follow per-parent participation/aggressiveness.

    volume_source="curve" (default): expected volume from the NSE curve, as
    the original. volume_source="tick": ``expected_minute_volume(tick)``, the
    rule the shared action decoder uses in training.
    """

    def __init__(self, agent_id: str, curve=None, cfg: Optional[dict] = None,
                 volume_source: str = "curve", through_ticks: int = 3) -> None:
        super().__init__(agent_id, curve, cfg)
        if volume_source not in ("curve", "tick"):
            raise ValueError("volume_source must be 'curve' or 'tick'")
        self.volume_source = volume_source
        self.through_ticks = int(through_ticks)

    def retarget(self, target_pos: int, side: str, day: int, bar: int,
                 participation: Optional[float] = None,
                 aggressiveness: Optional[float] = None) -> List[dict]:
        orders = super().retarget(target_pos, side, day, bar)
        p = self.parent
        p.participation = self.cap if participation is None else min(1.0, max(0.0, float(participation)))
        p.aggressiveness = None if aggressiveness is None else min(1.0, max(0.0, float(aggressiveness)))
        return orders

    def _expected(self, tick: dict, bar: int) -> float:
        if self.volume_source == "tick":
            return expected_minute_volume(tick)
        # The child trades in bar ``bar + 1`` = 0-based minute index ``bar``.
        return self.curve.expected(tick.get("symbol", ""), bar)

    def tick_orders(self, tick: dict) -> List[dict]:
        """Same flow as PovSlicer.tick_orders, with per-parent cap and price."""
        day, bar = int(tick["day_count"]), int(tick["market_minute"])
        p = self.parent
        if p is None:
            return []
        if p.opened_day != day or bar >= self.last_child_bar:
            return [self._close("session_close", day, bar)]
        if p.remaining(self.pos) <= 0:
            return [self._close("completed", day, bar)]
        part = getattr(p, "participation", self.cap)
        aggr = getattr(p, "aggressiveness", None)
        ev = self._expected(tick, bar)
        cap_qty = int(part * ev)
        qty = child_qty(p.remaining(self.pos), part, ev, self.max_remaining)
        bid, ask = tick.get("best_bid"), tick.get("best_ask")
        if aggr is None:
            price = ask if p.side == "BUY" else bid
        elif bid is None or ask is None:
            price = None
        else:
            price = child_price(p.side, aggr, float(bid), float(ask),
                                float(tick.get("tick_size") or 0.0), self.through_ticks)
        if qty <= 0 or price is None:
            return []
        p.children += 1
        p.child_qty_sent += qty
        self.children_log.append({"day": day, "bar": bar + 1, "qty": qty, "cap": cap_qty,
                                  "expected_volume": ev, "parent": p.parent_id,
                                  "participation": part, "aggressiveness": aggr})
        return [{"action": p.side, "type": "LIMIT", "price": float(price), "qty": int(qty),
                 "tif_ticks": self.child_tif,
                 "client_order_id": f"{p.parent_id}:C{p.children}"}]
