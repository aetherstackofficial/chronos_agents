"""Example strategy: Big Block Trader -- one big order, sent in slices.

What it does, in plain words
  1. Waits WARMUP minutes so VWAP means something.
  2. If the price is ENTRY_GAP below VWAP it BUYS a big block (expects a bounce up); if it is ENTRY_GAP above
     VWAP it SELLS a big block short (expects a fall). If neither happens within MAX_WAIT minutes it buys anyway,
     so you always get to see a big trade.
  3. The block (PARENT_QTY shares) is never sent at once: one MARKET slice of SLICE_QTY per minute, like a real
     desk working a large order. Each slice crosses the spread, so you can watch the price impact build up.
  4. Holds until the price moves TAKE_PROFIT in its favour, STOP_LOSS against it, or MAX_HOLD minutes pass.
  5. Gets out the same way, in slices, then rests COOLDOWN minutes and starts again.

Sizes, for RELIANCE (real data, Nov 2026): about 1.7 crore shares trade in a day and about 28,000 in a typical
minute. So 100,000 shares is ~0.6% of the day (~Rs 13 crore), and a 10,000-share slice is about a third of a
normal minute's volume: a big order. Change the numbers at the top to make it bigger or smaller.

Note: the engine starts a strategy with Rs 10 lakh of cash but does not stop it from trading more (it behaves
like unlimited margin), so the cash shown in Results goes negative while the block is held.

Where its trades show up
  * Market page: an arrow on each candle where it traded (B = bought, S = sold) -- hover it for the details;
    the tape marks its prints with YOU; the leaderboard has its STRAT_xxxx row.
  * Results page: this strategy's P&L, equity curve, every trade, CSV and PDF downloads.
  * Strategies page -> Running Strategies -> Logs: the lines this file prints.

Run standalone:
    python -m runner.sdk runner/examples/big_block.py --name big_block
"""

from runner.sdk import Strategy


class UserStrategy(Strategy):
    PARENT_QTY = 100_000     # shares in one big order
    SLICE_QTY = 10_000       # shares sent each minute
    WARMUP = 10              # minutes to wait before the first trade
    ENTRY_GAP = 0.001        # 0.1% away from VWAP = a signal
    MAX_WAIT = 20            # minutes to wait for a signal before buying anyway
    TAKE_PROFIT = 0.004      # +0.4% from the average entry price
    STOP_LOSS = 0.004        # -0.4%
    MAX_HOLD = 45            # minutes
    COOLDOWN = 20            # minutes of rest after getting out
    MAX_EMPTY_SLICES = 5     # give up on a phase after this many slices that fill nothing

    def on_start(self, config):
        self.me = config.get("agent_id", "?")
        self.pos = 0
        self.phase = "WAIT"          # WAIT -> ENTER -> HOLD -> EXIT -> REST -> WAIT
        self.side = 0                # +1 long block, -1 short block
        self.minutes_in_phase = 0
        self.minutes_seen = 0
        self.last_minute = None
        self.entry_cost = 0.0        # money paid (or received) for the block, to get the average entry
        self.entry_qty = 0
        self.empty = 0
        print(f"[big_block] {self.me} ready: {self.PARENT_QTY:,} shares per block, "
              f"{self.SLICE_QTY:,} per minute")

    # ------------------------------------------------------------------ helpers
    def _go(self, phase):
        self.phase, self.minutes_in_phase, self.empty = phase, 0, 0

    def _slice(self, action, qty):
        return [{"action": action, "type": "MARKET", "qty": int(qty)}]

    # ------------------------------------------------------------------ hooks
    def on_tick(self, state):
        minute = state.get("unix_time")
        if minute is None or minute == self.last_minute or state.get("paused"):
            return []                # at most one decision per market minute
        self.last_minute = minute
        self.minutes_seen += 1
        self.minutes_in_phase += 1

        px = state.get("last_price") or 0.0
        vwap = state.get("vwap") or 0.0
        if px <= 0:
            return []

        if self.phase == "WAIT":
            if self.minutes_seen < self.WARMUP or vwap <= 0:
                return []
            if px <= vwap * (1 - self.ENTRY_GAP):
                self.side = +1
                why = f"price {px:.2f} is below VWAP {vwap:.2f}"
            elif px >= vwap * (1 + self.ENTRY_GAP):
                self.side = -1
                why = f"price {px:.2f} is above VWAP {vwap:.2f}"
            elif self.minutes_in_phase >= self.MAX_WAIT:
                self.side = +1
                why = f"no signal for {self.MAX_WAIT} minutes"
            else:
                return []
            self.entry_cost, self.entry_qty = 0.0, 0
            print(f"[big_block] {'BUY' if self.side > 0 else 'SELL'} block of {self.PARENT_QTY:,} starts: {why}")
            self._go("ENTER")

        if self.phase == "ENTER":
            left = self.PARENT_QTY - abs(self.pos)
            if left <= 0 or self.empty >= self.MAX_EMPTY_SLICES:
                avg = self.entry_cost / self.entry_qty if self.entry_qty else px
                print(f"[big_block] block done: position {self.pos:,} at avg {avg:.2f}")
                self._go("HOLD")
                return []
            self.empty += 1          # reset by on_fill when the slice fills
            return self._slice("BUY" if self.side > 0 else "SELL", min(self.SLICE_QTY, left))

        if self.phase == "HOLD":
            avg = self.entry_cost / self.entry_qty if self.entry_qty else px
            move = self.side * (px - avg) / avg
            if move >= self.TAKE_PROFIT or move <= -self.STOP_LOSS or self.minutes_in_phase >= self.MAX_HOLD:
                why = ("take profit" if move >= self.TAKE_PROFIT else
                       "stop loss" if move <= -self.STOP_LOSS else "time is up")
                print(f"[big_block] exit starts ({why}): price {px:.2f}, entry {avg:.2f}, "
                      f"move {move * 100:+.2f}%")
                self._go("EXIT")
            else:
                return []

        if self.phase == "EXIT":
            if self.pos == 0 or self.empty >= self.MAX_EMPTY_SLICES:
                print(f"[big_block] out; position {self.pos:,}. Resting {self.COOLDOWN} minutes")
                self._go("REST")
                return []
            self.empty += 1
            return self._slice("SELL" if self.pos > 0 else "BUY", min(self.SLICE_QTY, abs(self.pos)))

        if self.phase == "REST" and self.minutes_in_phase >= self.COOLDOWN:
            self._go("WAIT")
        return []

    def on_fill(self, ack):
        done = int(ack.get("executed_qty") or 0)
        before, self.pos = self.pos, int(ack.get("pos", self.pos))
        if done > 0:
            self.empty = 0
            if self.phase == "ENTER":
                self.entry_cost += done * float(ack.get("average_price") or 0.0)
                self.entry_qty += done
        print(f"[big_block] {ack.get('status')}: {done:,} @ {ack.get('average_price')} "
              f"(position {before:,} -> {self.pos:,})")
