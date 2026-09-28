"""Example strategy: VWAP mean reversion with an RSI filter.

Logic (per PROTOCOL section 5.1 tick fields):

* BUY  when ``last_price < vwap * (1 - 0.002)`` AND ``rsi < 35``
  (price stretched below fair value while momentum is oversold).
* SELL / flatten when ``last_price > vwap * (1 + 0.002)`` OR ``rsi > 65``.

Execution style: passive-aggressive LIMIT orders one tick inside the touch
(buy at best_bid + 0.05, sell at best_ask - 0.05) so we improve the quote
instead of crossing the spread. Position is tracked from the authoritative
``pos`` field on every fill ack (see ``on_fill``).

Run standalone:
    python -m runner.sdk runner/examples/mean_reversion.py --name mean_rev
"""

from __future__ import annotations

from runner.sdk import Strategy

TICK = 0.05


class UserStrategy(Strategy):
    """Buy dips below VWAP when oversold; exit above VWAP or when overbought."""

    ENTRY_BAND = 0.002   # 0.2% displacement from VWAP triggers a signal
    RSI_OVERSOLD = 35.0
    RSI_OVERBOUGHT = 65.0
    ORDER_QTY = 50
    MAX_POSITION = 200   # never build beyond +/-200 shares

    def on_start(self, config: dict) -> None:
        """Reset inventory tracking."""
        self.position: int = 0
        self.agent_id: str = config.get("agent_id", "?")

    def on_tick(self, state: dict) -> list[dict]:
        """Emit at most one LIMIT order per tick based on VWAP displacement."""
        last = state.get("last_price")
        vwap = state.get("vwap")
        rsi = state.get("rsi")
        best_bid = state.get("best_bid")
        best_ask = state.get("best_ask")
        if not all(isinstance(v, (int, float)) and v > 0
                   for v in (last, vwap, rsi, best_bid, best_ask)):
            return []  # warm-up tick or one-sided book — stand down

        lower = vwap * (1.0 - self.ENTRY_BAND)
        upper = vwap * (1.0 + self.ENTRY_BAND)

        # Entry: cheap AND oversold, with room under the position cap.
        if last < lower and rsi < self.RSI_OVERSOLD and self.position < self.MAX_POSITION:
            qty = min(self.ORDER_QTY, self.MAX_POSITION - self.position)
            return [{
                "action": "BUY",
                "type": "LIMIT",
                "price": best_bid + TICK,   # improve the bid, stay passive
                "qty": qty,
                "tif_ticks": 10,            # stale reversion quotes expire fast
            }]

        # Exit: rich OR overbought — flatten longs (or lean short one clip).
        if last > upper or rsi > self.RSI_OVERBOUGHT:
            if self.position > 0:
                qty = min(self.position, self.MAX_POSITION)  # flatten
            elif self.position > -self.MAX_POSITION:
                qty = min(self.ORDER_QTY, self.MAX_POSITION + self.position)
            else:
                return []
            if qty <= 0:
                return []
            return [{
                "action": "SELL",
                "type": "LIMIT",
                "price": best_ask - TICK,   # improve the offer
                "qty": qty,
                "tif_ticks": 10,
            }]

        return []

    def on_fill(self, ack: dict) -> None:
        """Track inventory from the engine's authoritative post-trade ledger."""
        pos = ack.get("pos")
        if isinstance(pos, int):
            self.position = pos

    def on_stop(self) -> None:
        """Final inventory report (CANCEL_ALL already sent by the SDK)."""
        print(f"[mean_reversion] {self.agent_id} stopped with position {self.position}")
