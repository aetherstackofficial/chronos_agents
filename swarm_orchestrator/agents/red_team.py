"""Red-team predatory taker (PROTOCOL_V2.md §8) — real heuristic adversary.

HONEST SCOPE: this is a deterministic *heuristic* predator, not an RL-trained
one (an RL red-team is roadmap). It models adverse selection against user
strategies with two real behaviours driven purely by public tick data:

1. **Fade the crowd** — when order flow is strongly one-sided (``|ofi| > 0.5``)
   AND price has extended in the same direction over the recent window, it hits
   the move with a MARKET order *against* the crowd, punishing late chasers.
2. **Sweep the thin side** — when the spread is wide (thin book), it
   occasionally fires a small MARKET taker into the thinner side, consuming
   scarce liquidity exactly when it hurts most.

Interface is identical to the live agents (``agents/live_agents.py``) so the
``SwarmOrchestrator`` can run it unmodified: ``on_tick(tick) -> list`` of §4.1
order dicts minus ``msg``/``agent_id``; ``on_ack(ack)`` adopts the engine's
authoritative ledger. Sanctioned randomness only (seeded per-agent RNG for
cooldown jitter and the probabilistic sweep trigger — PROTOCOL.md §8).
Pure heuristic: no torch, no model files.
"""
from __future__ import annotations

import logging
import random
import time
from collections import deque
from typing import Deque, List

logger = logging.getLogger("chronos.redteam")


class RedTeamAgent:
    """Heuristic predatory taker: fades extended one-sided flow, sweeps thin books."""

    # -- policy constants ------------------------------------------------- #
    MAX_POS = 5_000                # hard position cap (long and short)
    COOLDOWN_S = 1.5               # base cooldown between actions
    COOLDOWN_JITTER = 0.4          # +/-40% seeded jitter on the cooldown
    MOMO_WINDOW = 12               # mid-price history length (ticks)
    MOMO_MIN_TICKS = 5             # minimum history before momentum is trusted
    OFI_TRIGGER = 0.5              # |ofi| beyond this = crowded one-sided flow
    EXTENSION_PCT = 0.002          # 0.2% move over the window = "extended"
    FADE_BASE_QTY = 200            # fade size at the OFI trigger
    FADE_MAX_QTY = 800             # fade size at |ofi| = 1.0
    WIDE_SPREAD_PCT = 0.003        # spread >= 0.3% of mid = thin/wide book
    SWEEP_PROB = 0.25              # per-eligible-tick chance of a sweep
    SWEEP_QTY = 100                # small taker clip for sweeps
    DEPTH_LEVELS = 3               # top-of-book levels compared for thinness

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.cash: float = 0.0
        self.inventory: int = 0
        self.is_asleep: bool = False
        self.last_trade: float = 0.0
        # Seeded per-agent RNG — sanctioned behavioural randomness only
        # (cooldown jitter + probabilistic sweep trigger, PROTOCOL.md §8).
        self._rng = random.Random(f"redteam:{agent_id}")
        self._cooldown_s = self.COOLDOWN_S
        self._mids: Deque[float] = deque(maxlen=self.MOMO_WINDOW)
        logger.info("[REDTEAM] %s online (heuristic predator, cap +/-%d)",
                    agent_id, self.MAX_POS)

    # ------------------------------------------------------------------ tick
    def on_tick(self, tick: dict) -> List[dict]:
        """Return this tick's orders (§4.1 minus ``msg``/``agent_id``)."""
        if self.is_asleep:
            return []

        mid = float(tick.get("mid_price") or tick.get("last_price") or 0.0)
        if mid <= 0.0:
            return []
        self._mids.append(mid)

        now = time.time()
        if now - self.last_trade < self._cooldown_s:
            return []

        ofi = float(tick.get("ofi") or 0.0)
        spread = float(tick.get("spread") or 0.0)

        order = self._fade_order(mid, ofi) or self._sweep_order(tick, mid, spread)
        if order is None:
            return []

        self.last_trade = now
        self._cooldown_s = self.COOLDOWN_S * (
            1.0 + self.COOLDOWN_JITTER * (2.0 * self._rng.random() - 1.0))
        logger.info("[REDTEAM] %s %s %s %d (ofi %+.2f, spread %.2f, pos %d)",
                    self.agent_id, order["action"], order["type"],
                    order["qty"], ofi, spread, self.inventory)
        return [order]

    # ------------------------------------------------------------- behaviours
    def _fade_order(self, mid: float, ofi: float) -> "dict | None":
        """Fade crowded, extended flow with a MARKET order against the crowd."""
        if abs(ofi) <= self.OFI_TRIGGER or len(self._mids) < self.MOMO_MIN_TICKS:
            return None
        anchor = self._mids[0]
        if anchor <= 0.0:
            return None
        move = (mid - anchor) / anchor
        # Price must be extended in the SAME direction as the flow: a
        # buy-heavy tape that already ran up is the crowd we punish.
        if ofi > 0 and move < self.EXTENSION_PCT:
            return None
        if ofi < 0 and move > -self.EXTENSION_PCT:
            return None

        side = "SELL" if ofi > 0 else "BUY"
        # Size scales deterministically with how one-sided the flow is.
        intensity = min(1.0, (abs(ofi) - self.OFI_TRIGGER) / (1.0 - self.OFI_TRIGGER))
        qty = int(self.FADE_BASE_QTY
                  + intensity * (self.FADE_MAX_QTY - self.FADE_BASE_QTY))
        qty = self._cap_qty(side, qty)
        if qty <= 0:
            return None
        return {"action": side, "type": "MARKET", "qty": qty}

    def _sweep_order(self, tick: dict, mid: float, spread: float) -> "dict | None":
        """Occasionally sweep the thinner side of a wide (thin) book."""
        if spread < mid * self.WIDE_SPREAD_PCT:
            return None
        if self._rng.random() >= self.SWEEP_PROB:  # sanctioned probabilistic trigger
            return None

        bid_depth = self._depth(tick.get("lob_bids"))
        ask_depth = self._depth(tick.get("lob_asks"))
        if bid_depth is not None and ask_depth is not None and bid_depth != ask_depth:
            # BUY consumes the ask side; hit whichever side is thinner.
            side = "BUY" if ask_depth < bid_depth else "SELL"
        else:
            # No depth data: work against our own inventory so the sweep also
            # mean-reverts position (deterministic given the ledger).
            side = "SELL" if self.inventory > 0 else "BUY"

        qty = self._cap_qty(side, self.SWEEP_QTY)
        if qty <= 0:
            return None
        return {"action": side, "type": "MARKET", "qty": qty}

    # --------------------------------------------------------------- helpers
    def _depth(self, levels: "list | None") -> "int | None":
        """Total resting qty on the top ``DEPTH_LEVELS`` of one book side."""
        if not levels:
            return None
        total = 0
        for level in levels[:self.DEPTH_LEVELS]:
            try:
                total += int(level[1])
            except (TypeError, ValueError, IndexError):
                return None
        return total

    def _cap_qty(self, side: str, qty: int) -> int:
        """Clamp ``qty`` so the post-fill position stays within ±MAX_POS."""
        if side == "BUY":
            room = self.MAX_POS - self.inventory
        else:
            room = self.MAX_POS + self.inventory
        return max(0, min(qty, room))

    # ---------------------------------------------------------------- ledger
    def on_ack(self, ack: dict) -> None:
        """Adopt the engine's post-action ledger (§4.2 — authoritative)."""
        cash = ack.get("cash")
        if cash is not None:
            self.cash = float(cash)
        pos = ack.get("pos")
        if pos is not None:
            self.inventory = int(pos)
