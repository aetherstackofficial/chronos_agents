"""Example strategy: bar-close momentum (trend following).

Logic: keep the closes of the last 10 completed 1-minute bars (each engine
physics tick = one sim-minute, PROTOCOL section 3). When the newest close is
meaningfully above the oldest close in the window, the tape is trending up —
join it with a MARKET buy; when below, sell. A hard inventory cap of +/-500
shares bounds the position, tracked from the authoritative ``pos`` field on
each fill ack.

Bar completion is detected by watching ``state["bar"]["minute"]`` roll over:
the previously observed in-progress bar's close is then final.

Run standalone:
    python -m runner.sdk runner/examples/momentum.py --name momentum
"""

from __future__ import annotations

from collections import deque

from runner.sdk import Strategy


class UserStrategy(Strategy):
    """Trend-follow bar closes over a 10-tick lookback with MARKET orders."""

    LOOKBACK = 10          # completed bars in the momentum window
    THRESHOLD = 0.001      # 0.1% move across the window = a trend
    ORDER_QTY = 25
    MAX_POSITION = 500     # hard cap, long and short

    def on_start(self, config: dict) -> None:
        """Reset the bar window and inventory tracking."""
        self.closes: deque[float] = deque(maxlen=self.LOOKBACK)
        self.position: int = 0
        self.last_bar_minute: int | None = None
        self.last_bar_close: float | None = None
        self.agent_id: str = config.get("agent_id", "?")

    def _record_completed_bar(self, state: dict) -> None:
        """Push the close of any just-completed bar into the window."""
        bar = state.get("bar") or {}
        minute = bar.get("minute")
        close = bar.get("c")
        if not isinstance(minute, int) or not isinstance(close, (int, float)):
            return
        if self.last_bar_minute is not None and minute != self.last_bar_minute:
            # The bar we were watching has closed — its last seen close is final.
            if self.last_bar_close is not None and self.last_bar_close > 0:
                self.closes.append(float(self.last_bar_close))
        self.last_bar_minute = minute
        self.last_bar_close = float(close)

    def on_tick(self, state: dict) -> list[dict]:
        """Trade the direction of the 10-bar momentum, capped at +/-500."""
        self._record_completed_bar(state)
        if len(self.closes) < self.LOOKBACK:
            return []  # window still warming up

        oldest, newest = self.closes[0], self.closes[-1]
        if oldest <= 0:
            return []
        momentum = (newest - oldest) / oldest

        if momentum > self.THRESHOLD and self.position < self.MAX_POSITION:
            qty = min(self.ORDER_QTY, self.MAX_POSITION - self.position)
            return [{"action": "BUY", "qty": qty}]          # shorthand = MARKET

        if momentum < -self.THRESHOLD and self.position > -self.MAX_POSITION:
            qty = min(self.ORDER_QTY, self.MAX_POSITION + self.position)
            return [{"action": "SELL", "qty": qty}]

        return []

    def on_fill(self, ack: dict) -> None:
        """Track inventory from the engine's authoritative post-trade ledger."""
        pos = ack.get("pos")
        if isinstance(pos, int):
            self.position = pos

    def on_stop(self) -> None:
        """Final inventory report (CANCEL_ALL already sent by the SDK)."""
        print(f"[momentum] {self.agent_id} stopped with position {self.position}")
