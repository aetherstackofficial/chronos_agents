"""Transaction costs of the replay environment.

* every fill pays HALF of the intraday round trip: (intraday_rt_bps / 2) x notional x cost_mult
* shares still held at a session close pay the delivery surcharge
  (delivery_rt_bps - intraday_rt_bps) x value, ONCE per share (``CarryTracker``)
* at the end of an episode the open position is closed out at
  |pos| x price x (intraday_rt_bps / 2 + half the spread in bps) x cost_mult
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    intraday_rt_bps: float
    delivery_rt_bps: float
    cost_mult: float = 1.0

    @classmethod
    def from_config(cls, cfg: dict, cost_mult: float = 1.0) -> "CostModel":
        c = cfg["costs"]
        return cls(float(c["intraday_rt_bps"]), float(c["delivery_rt_bps"]), float(cost_mult))

    def fill_fee(self, qty: float, price: float) -> float:
        """Fee of one fill, in money."""
        return abs(qty) * price * (self.intraday_rt_bps / 2.0) / 1e4 * self.cost_mult

    def carry_rate(self) -> float:
        """Delivery surcharge as a fraction of value, cost_mult applied."""
        return max(self.delivery_rt_bps - self.intraday_rt_bps, 0.0) / 1e4 * self.cost_mult

    def closing_cost(self, position: float, price: float, spread: float) -> float:
        """Cost of flattening ``position`` at the episode end. ``spread`` is in price units."""
        if not position or price <= 0.0:
            return 0.0
        half_spread_bps = 0.5 * max(float(spread), 0.0) / price * 1e4
        return abs(position) * price * (self.intraday_rt_bps / 2.0 + half_spread_bps) / 1e4 * self.cost_mult


class CarryTracker:
    """Which shares of the position have already paid the delivery surcharge.

    ``charged`` counts shares (same sign as the position) that already paid.
    It shrinks when the position shrinks and resets to 0 when the position
    flips sign or goes flat, so a share is charged once however many sessions
    it is carried, and a re-opened position pays again.
    """

    def __init__(self) -> None:
        self.charged = 0.0
        self._sign = 0

    def on_position(self, pos: float) -> None:
        """Call after every fill with the new position."""
        s = (pos > 0) - (pos < 0)
        if s == 0 or s != self._sign:
            self.charged = 0.0
        else:
            self.charged = min(self.charged, abs(pos))
        self._sign = s

    def session_close(self, pos: float, price: float, model: CostModel) -> float:
        """Surcharge (money) for the not-yet-charged shares at a session close."""
        self.on_position(pos)
        new = abs(pos) - self.charged
        if new <= 0.0 or price <= 0.0:
            return 0.0
        self.charged = abs(pos)
        return new * price * model.carry_rate()
