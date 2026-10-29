"""Phase 4 mechanism tests: whale POV slicer and market-maker inventory skew.

The slicer runs against the real MatchingEngine (SIM_LP provides liquidity
every physics tick), so caps, fills and shortfall are checked against the
engine's own trade tape, not against the slicer's view of itself.
"""
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.execution import CONFIG, PovSlicer, VolumeCurve, skew_ticks, snap  # noqa: E402
from engine.matching_engine import MatchingEngine  # noqa: E402

WHALE = "WHALE_T"


class Harness:
    """Engine + one slicer, driven the way the orchestrator drives an agent."""

    def __init__(self, price=1350.0, symbol="RELIANCE", seed=7):
        self.eng = MatchingEngine(db=None)
        self.eng.init_sim(symbol, "SIM", price, seed=seed)
        self.slicer = PovSlicer(WHALE)
        self.trades = []          # (bar, side_for_whale, price, qty)
        self.sent = []
        self.tick = None

    def _absorb(self, pubs):
        for topic, p in pubs:
            if topic == "trade" and WHALE in (p["buyer"], p["seller"]):
                side = 1 if p["buyer"] == WHALE else -1
                self.trades.append((self.eng.tick_count + 1, side, p["price"], p["qty"]))

    def physics(self):
        pubs = self.eng.physics_tick()
        for topic, p in pubs:
            if topic == "trade" and WHALE in (p["buyer"], p["seller"]):
                side = 1 if p["buyer"] == WHALE else -1
                self.trades.append((self.eng.tick_count, side, p["price"], p["qty"]))
        self.tick = [p for t, p in pubs if t == "tick"][-1]
        return self.tick

    def send(self, orders):
        for o in orders:
            msg = ({"msg": "CANCEL_ALL", "agent_id": WHALE} if o.get("cancel_all")
                   else {"msg": "ORDER", "agent_id": WHALE, **o})
            self.sent.append(msg)
            ack, pubs = self.eng.handle_message(msg)
            self._absorb(pubs)
            self.slicer.on_ack(ack)

    def whale_pos(self):
        return self.eng.agent_ledger.get(WHALE, {"pos": 0})["pos"]


def _run(h, bars, target_fn):
    """target_fn(bar) -> (target_pos, side) or None; slicer acts every bar."""
    for _ in range(bars):
        t = h.physics()
        tgt = target_fn(int(t["market_minute"]))
        if tgt is not None:
            h.send(h.slicer.retarget(tgt[0], tgt[1], int(t["day_count"]), int(t["market_minute"])))
        h.send(h.slicer.tick_orders(t))


def test_children_are_marketable_limits_never_market():
    h = Harness()
    _run(h, 40, lambda b: (10_000, "BUY") if b == 1 else None)
    orders = [m for m in h.sent if m["msg"] == "ORDER"]
    assert orders, "slicer sent no children"
    assert all(m["type"] == "LIMIT" and m["tif_ticks"] == 1 for m in orders)
    assert all(":P1:C" in m["client_order_id"] for m in orders)


def test_cap_holds_every_minute_against_the_tape():
    h = Harness()
    _run(h, 120, lambda b: (10_000, "BUY") if b == 1 else None)
    curve = VolumeCurve()
    cap = CONFIG["whale_pov"]["participation_cap"]
    per_bar = {}
    for bar, side, _px, q in h.trades:
        per_bar[bar] = per_bar.get(bar, 0) + q
    for c in h.slicer.children_log:
        assert c["qty"] <= int(cap * curve.expected("RELIANCE", c["bar"] - 1))
    for bar, q in per_bar.items():
        assert q <= int(cap * curve.expected("RELIANCE", bar - 1)), (bar, q)


def test_fills_match_tape_exactly_and_shortfall_carries():
    h = Harness()
    _run(h, 60, lambda b: (10_000, "BUY") if b == 1 else None)
    h.send([h.slicer._close("test_close", 1, 61)]) if h.slicer.parent else None
    tape_qty = sum(s * q for _, s, _, q in h.trades)
    tape_notional = sum(s * q * px for _, s, px, q in h.trades)
    p = h.slicer.log[-1]
    assert p.filled == tape_qty == h.whale_pos()
    assert p.avg_price == pytest.approx(tape_notional / tape_qty, rel=1e-12)
    # The visible best ask is usually thinner than the child: some child
    # quantity must have gone unfilled and rolled forward, never a market order.
    assert p.child_qty_sent > p.filled
    assert p.shortfall == 10_000 - p.filled


def test_same_side_retarget_keeps_fills_and_updates_parent():
    h = Harness()
    _run(h, 10, lambda b: (10_000, "BUY") if b == 1 else ((4_000, "BUY") if b == 8 else None))
    p = h.slicer.parent or h.slicer.log[-1]
    assert p.parent_id.endswith(":P1") and p.revisions == 1 and p.target_pos == 4_000
    assert h.whale_pos() <= 4_000 + max(c["qty"] for c in h.slicer.children_log)


def test_opposite_side_closes_parent_with_cancel_all_and_reopens():
    h = Harness()
    _run(h, 12, lambda b: (10_000, "BUY") if b == 1 else ((-3_000, "SELL") if b == 10 else None))
    first = h.slicer.log[0]
    assert first.status == "replaced_opposite_side" and first.filled > 0
    assert h.slicer.parent.side == "SELL" and h.slicer.parent.pos0 == first.pos0 + first.filled
    cancel_idx = [i for i, m in enumerate(h.sent) if m["msg"] == "CANCEL_ALL"]
    assert cancel_idx, "a parent must be closed through CANCEL_ALL"


def test_session_close_cancels_parent_and_nothing_trades_at_1529():
    h = Harness()
    # Opened at 15:25 (bar 370): four minutes cannot finish 10,000 shares.
    _run(h, 375, lambda b: (10_000, "BUY") if b == 370 else None)
    p = h.slicer.log[-1]
    assert p.status == "session_close" and p.closed_bar == CONFIG["whale_pov"]["last_child_bar"]
    assert all(bar < 375 for bar, *_ in h.trades), "whale traded in the 15:29 bar"
    assert p.shortfall == p.remaining(h.whale_pos())


def test_unknown_symbol_falls_back_to_reliance_and_logs(caplog):
    curve = VolumeCurve()
    with caplog.at_level(logging.WARNING, logger="agents.execution"):
        assert curve.expected("NOSUCH", 100) == curve.expected("RELIANCE", 100)
    assert "using RELIANCE" in caplog.text


def test_open_auction_excluded_from_curve():
    c = VolumeCurve().curve_for("RELIANCE")
    assert c["minute_weights"][0] == c["minute_weights"][1]
    assert c["adv_shares_including_open_auction"] > c["adv_shares"]


@pytest.mark.parametrize("q,qmax,expected", [
    (0, 10_000, 0), (10_000, 10_000, -3), (-10_000, 10_000, 3), (5_000, 10_000, -2),
    (1_000, 10_000, 0), (40_000, 10_000, -3), (-40_000, 10_000, 3), (500, 0, 0)])
def test_skew_rule(q, qmax, expected):
    assert skew_ticks(q, qmax) == expected


@pytest.mark.parametrize("px,tick", [(1352.3349, 0.10), (231.017, 0.01), (612.37, 0.05), (7123.3, 0.5)])
def test_snap_is_on_grid(px, tick):
    s = snap(px, tick)
    assert abs(s / tick - round(s / tick)) < 1e-9 and abs(s - px) <= tick / 2 + 1e-9
