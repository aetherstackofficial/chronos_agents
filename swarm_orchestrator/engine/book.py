"""Limit order book with strict price-time priority.

Implements the matching semantics of PROTOCOL §4.2 / §8:
- sortedcontainers.SortedDict per side (asks ascending, bids keyed negated
  so index 0 is always the best price on either side, O(log n) access),
- FIFO deque of order dicts per price level (time priority),
- all limit prices rounded to the venue tick for the price band (tick_for_price),
- self-trade prevention: resting orders of the taker's own agent_id are
  skipped (left resting) during matching,
- order_id -> (key, side) index for O(log n) cancel-by-id.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

from sortedcontainers import SortedDict

# Venue tick bands. A single fixed tick cannot serve every price: ₹0.05 is
# 0.026% of ₹190 but 1.6% of ₹3, coarse enough that a ₹3 instrument cannot
# move at all. Real exchanges solve this with price slabs, reviewed on a
# schedule -- never intraday, because re-gridding would strand resting orders.
#
# NOTE: NSE's real floor is ₹0.01. The sub-₹10 band below is finer than the
# live market so low-priced instruments remain usable in simulation.
TICK_BANDS = (
    #  price below      tick
    (       10.0,      0.001),
    (      100.0,      0.01),
    (     1000.0,      0.05),
    (    10000.0,      0.10),
    (float("inf"),     0.50),
)

DEFAULT_TICK = 0.05     # ₹100-₹1000 band; also the fallback for bare OrderBook()

# Epsilon for float price comparisons; must stay well below the smallest tick.
_PRICE_EPS = 1e-9


def tick_for_price(price: float) -> float:
    """Venue tick size for an instrument trading around ``price``.

    Evaluated at run start and at each session open (see MatchingEngine), never
    per order -- a tick that floated with every trade would put each order on a
    different grid and fragment the book instead of consolidating liquidity.
    """
    if not price or price <= 0:
        return DEFAULT_TICK
    for upper, tick in TICK_BANDS:
        if price < upper:
            return tick
    return TICK_BANDS[-1][1]


def decimals_for_tick(tick: float) -> int:
    """Decimal places needed to display this tick exactly."""
    for dp in range(0, 7):
        if abs(tick * (10 ** dp) - round(tick * (10 ** dp))) < 1e-9:
            return dp
    return 6


def round_to_tick(price: float, tick: float = DEFAULT_TICK) -> float:
    """Snap ``price`` to the ``tick`` grid, rounded for clean display."""
    if tick <= 0:
        tick = DEFAULT_TICK
    return round(round(price / tick) * tick, decimals_for_tick(tick))



class OrderBook:
    """Price-time priority limit order book for a single symbol.

    A fill is ``{"price", "qty", "maker_id", "taker_id", "maker_order_id"}``.
    Resting order entries are ``{"order_id", "agent_id", "qty", "ts_seq"}``.
    """

    def __init__(self, tick:float = DEFAULT_TICK) -> None:
        self.tick = tick
        self._asks: SortedDict = SortedDict()   # price -> deque, ascending
        self._bids: SortedDict = SortedDict()   # -price -> deque, so best first
        # order_id -> (book key, side) for O(log n) cancel.
        self._index: Dict[int, Tuple[float, str]] = {}
        self._agent_orders: Dict[str, Set[int]] = {}

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _book_for(self, side: str) -> SortedDict:
        return self._bids if side == "BUY" else self._asks

    @staticmethod
    def _key_for(side: str, price: float) -> float:
        return -price if side == "BUY" else price

    def _drop_index(self, order_id: int, agent_id: str) -> None:
        self._index.pop(order_id, None)
        owned = self._agent_orders.get(agent_id)
        if owned is not None:
            owned.discard(order_id)
            if not owned:
                del self._agent_orders[agent_id]

    def _match(self, taker_id: str, side: str, qty: int,
               limit_price: Optional[float]) -> List[dict]:
        """Walk the opposite side of the book in price-time order.

        Resting orders belonging to ``taker_id`` are skipped in place
        (self-trade prevention); matching continues past them.
        """
        opposite = self._asks if side == "BUY" else self._bids
        remaining = qty
        fills: List[dict] = []
        for key in list(opposite.keys()):
            if remaining <= 0:
                break
            price = key if side == "BUY" else -key
            if limit_price is not None:
                if side == "BUY" and price > limit_price + _PRICE_EPS:
                    break
                if side == "SELL" and price < limit_price - _PRICE_EPS:
                    break
            queue: Deque[dict] = opposite[key]
            i = 0
            while i < len(queue) and remaining > 0:
                resting = queue[i]
                if resting["agent_id"] == taker_id:
                    i += 1  # self-trade prevention: leave it resting
                    continue
                take = min(remaining, resting["qty"])
                resting["qty"] -= take
                remaining -= take
                fills.append({
                    "price": price,
                    "qty": take,
                    "maker_id": resting["agent_id"],
                    "taker_id": taker_id,
                    "maker_order_id": resting["order_id"],
                })
                if resting["qty"] == 0:
                    self._drop_index(resting["order_id"], resting["agent_id"])
                    del queue[i]
            if not queue:
                del opposite[key]
        return fills

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def add_limit(self, order: dict) -> List[dict]:
        """Add a limit order: execute the marketable part, rest the remainder.

        ``order`` requires order_id, agent_id, side ("BUY"/"SELL"), price,
        qty, ts_seq. The price is tick-rounded in place. Returns fills.
        """
        price = round_to_tick(order["price"], self.tick)
        order["price"] = price
        fills = self._match(order["agent_id"], order["side"], order["qty"], price)
        executed = sum(f["qty"] for f in fills)
        remaining = order["qty"] - executed
        if remaining > 0:
            key = self._key_for(order["side"], price)
            book = self._book_for(order["side"])
            if key not in book:
                book[key] = deque()
            book[key].append({
                "order_id": order["order_id"],
                "agent_id": order["agent_id"],
                "qty": remaining,
                "ts_seq": order["ts_seq"],
            })
            self._index[order["order_id"]] = (key, order["side"])
            self._agent_orders.setdefault(order["agent_id"], set()).add(order["order_id"])
        return fills

    def match_market(self, agent_id: str, side: str, qty: int) -> List[dict]:
        """Execute a market order against the whole opposite side.

        Any unfilled remainder is discarded (never rests). Returns fills.
        """
        return self._match(agent_id, side, qty, None)

    def cancel(self, order_id: int) -> bool:
        """Remove a resting order by id. Returns False if unknown/filled."""
        info = self._index.get(order_id)
        if info is None:
            return False
        key, side = info
        book = self._book_for(side)
        queue = book.get(key)
        if queue is not None:
            for i, resting in enumerate(queue):
                if resting["order_id"] == order_id:
                    self._drop_index(order_id, resting["agent_id"])
                    del queue[i]
                    if not queue:
                        del book[key]
                    return True
        # Index/book desync should be impossible; fail closed.
        self._index.pop(order_id, None)
        return False

    def cancel_all(self, agent_id: str) -> int:
        """Cancel every resting order of an agent. Returns count cancelled."""
        order_ids = list(self._agent_orders.get(agent_id, ()))
        return sum(1 for oid in order_ids if self.cancel(oid))

    def has_order(self, order_id: int) -> bool:
        """True if the order is still resting in the book."""
        return order_id in self._index

    def best_bid(self) -> Optional[float]:
        """Highest bid price, or None if the bid side is empty."""
        if not self._bids:
            return None
        return -self._bids.peekitem(0)[0]

    def best_ask(self) -> Optional[float]:
        """Lowest ask price, or None if the ask side is empty."""
        if not self._asks:
            return None
        return self._asks.peekitem(0)[0]

    def depth(self, n: int = 10) -> Tuple[List[List[float]], List[List[float]]]:
        """Top-n aggregated (bids, asks) as [[price, qty], ...], best first."""
        bids: List[List[float]] = []
        for key in self._bids.keys()[:n]:
            bids.append([-key, sum(o["qty"] for o in self._bids[key])])
        asks: List[List[float]] = []
        for key in self._asks.keys()[:n]:
            asks.append([key, sum(o["qty"] for o in self._asks[key])])
        return bids, asks

    def cum_qty_within(self, taker_side: str, limit_price: float,
                       exclude_agent: Optional[str] = None) -> int:
        """Total opposite-side quantity executable by a limit at limit_price.

        For a BUY taker: sum of ask qty priced <= limit_price; for SELL:
        bid qty priced >= limit_price. Used by the replayer to size ghost
        orders that walk the book exactly to a historical target price.
        """
        opposite = self._asks if taker_side == "BUY" else self._bids
        total = 0
        for key in opposite.keys():
            price = key if taker_side == "BUY" else -key
            if taker_side == "BUY" and price > limit_price + _PRICE_EPS:
                break
            if taker_side == "SELL" and price < limit_price - _PRICE_EPS:
                break
            total += sum(o["qty"] for o in opposite[key]
                         if o["agent_id"] != exclude_agent)
        return total
