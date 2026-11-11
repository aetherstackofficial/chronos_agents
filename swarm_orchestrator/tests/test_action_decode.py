"""Shared action decoder (agents/action_decode.py) and PovSlicerV2 (agents/execution_v2.py)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.action_decode import DEFAULTS, AgentState, decode_mm, decode_whale  # noqa: E402
from agents.execution import PovSlicer  # noqa: E402
from agents.execution_v2 import MARKETABLE_AGGRESSIVENESS, PovSlicerV2  # noqa: E402

CAP = 100_000


def tk(bid=100.00, ask=100.10, vol=10_000.0, **kw):
    t = {"day_count": 1, "market_minute": 5, "symbol": "RELIANCE", "tick_size": 0.05,
         "best_bid": bid, "best_ask": ask, "last_price": 100.05, "volume_ma": vol, "step_volume": 5000}
    t.update(kw)
    return t


def ws(pos=0, cash=1e9, **kw):
    return AgentState(position=pos, cash=cash, capacity=CAP, **kw)


def on_grid(px, tick=0.05):
    return abs(px / tick - round(px / tick)) < 1e-6


def x_for(frac):          # model output giving a 0..1 value `frac`
    return frac * 2 - 1


# ----------------------------------------------------------------- whale: band
def test_enter_from_flat_when_target_above_entry():
    r = decode_whale([0.5, 1, 1], ws(0), tk())
    assert r.intent["target_shares"] == 50_000 and r.intent["delta_shares"] == 50_000
    assert r.orders[0]["action"] == "BUY"
    assert not any(n.startswith("band") for n in r.notes)


def test_stay_flat_below_entry_has_note():
    r = decode_whale([0.12, 1, 1], ws(0), tk())
    assert r.intent["target_shares"] == 0 and r.orders == []
    assert any(n.startswith("band") and "stay flat" in n for n in r.notes)


def test_entry_threshold_is_strict():
    assert decode_whale([0.20, 0, 0], ws(0), tk()).intent["target_shares"] == 0
    assert decode_whale([0.21, 0, 0], ws(0), tk()).intent["target_shares"] == 21_000
    assert decode_whale([-0.21, 0, 0], ws(0), tk()).intent["target_shares"] == -21_000


def test_small_position_counts_as_flat():
    r = decode_whale([0.10, 0, 0], ws(4_000), tk())          # 4% < 5% -> flat band applies
    assert r.intent["target_shares"] == 4_000                # band -> 0, then 4% change filtered
    assert any(n.startswith("band") for n in r.notes)


def test_hold_same_side_keeps_target():
    r = decode_whale([0.5, 0, 0], ws(30_000), tk())
    assert r.intent["target_shares"] == 50_000 and r.intent["delta_shares"] == 20_000


def test_exit_when_target_near_zero():
    r = decode_whale([0.02, 0, 0], ws(30_000), tk())
    assert r.intent["target_shares"] == 0 and r.intent["delta_shares"] == -30_000
    assert r.orders[0]["action"] == "SELL"
    assert any("close" in n for n in r.notes)


def test_opposite_small_target_closes_to_flat():
    r = decode_whale([-0.15, 0, 0], ws(30_000), tk())
    assert r.intent["target_shares"] == 0
    assert any("flip" in n for n in r.notes)


def test_flip_when_opposite_target_beyond_entry():
    r = decode_whale([-0.5, 0, 0], ws(30_000), tk())
    assert r.intent["target_shares"] == -50_000 and r.intent["delta_shares"] == -80_000
    assert r.orders[0]["action"] == "SELL"


# ------------------------------------------------------------ whale: filter
def test_small_change_is_ignored():
    r = decode_whale([0.33, 0, 0], ws(30_000), tk())          # 3% change
    assert r.intent["target_shares"] == 30_000 and r.intent["delta_shares"] == 0 and r.orders == []
    assert any(n.startswith("filter: change 3% < 5%") for n in r.notes)


def test_change_of_exactly_five_percent_passes():
    r = decode_whale([0.35, 0, 0], ws(30_000), tk())
    assert r.intent["target_shares"] == 35_000


# --------------------------------------------------------------- whale: caps
def test_out_of_range_clipped_and_capped_at_capacity():
    r = decode_whale([3.0, 9.0, -9.0], ws(0), tk())
    assert r.raw == [1.0, 1.0, -1.0]
    assert r.intent["target_shares"] == CAP
    assert r.intent["participation"] == pytest.approx(0.15) and r.intent["aggressiveness"] == 0.0
    assert any(n.startswith("clip") for n in r.notes)
    r = decode_whale([-3.0, 0, 0], ws(0), tk())
    assert r.intent["target_shares"] == -CAP


def test_cash_caps_long_target_by_ask():
    r = decode_whale([0.5, 0, 0], ws(0, cash=1_000_000.0), tk())   # ask 100.10 -> 9990 shares
    assert r.intent["target_shares"] == 9_990
    assert "cap: cash" in r.notes


def test_cash_cap_counts_existing_position():
    r = decode_whale([0.9, 0, 0], ws(30_000, cash=100_100.0), tk())   # 1000 more shares
    assert r.intent["target_shares"] == 31_000


def test_shorts_not_limited_by_cash():
    r = decode_whale([-0.5, 0, 0], ws(0, cash=0.0), tk())
    assert r.intent["target_shares"] == -50_000 and "cap: cash" not in r.notes


# ------------------------------------------------- whale: participation / child
def test_participation_mapping_endpoints():
    assert decode_whale([0.5, -1, 0], ws(), tk()).intent["participation"] == pytest.approx(0.01)
    assert decode_whale([0.5, 1, 0], ws(), tk()).intent["participation"] == pytest.approx(0.15)
    assert decode_whale([0.5, 0, 0], ws(), tk()).intent["participation"] == pytest.approx(0.08)


def test_child_qty_limited_by_participation_times_volume_ma():
    assert decode_whale([0.5, 1, 0], ws(), tk(vol=10_000)).orders[0]["qty"] == 1_500
    assert decode_whale([0.5, -1, 0], ws(), tk(vol=10_000)).orders[0]["qty"] == 100
    assert decode_whale([0.5, 1, 0], ws(), tk(vol=999)).orders[0]["qty"] == 149   # rounded down


def test_child_qty_limited_by_remaining_delta():
    s = AgentState(position=0, cash=1e9, capacity=1_000)
    r = decode_whale([0.5, 1, 0], s, tk(vol=1e6))
    assert r.orders[0]["qty"] == 500 == r.intent["child_qty"]


def test_child_qty_uses_parent_remaining_cap():
    r = decode_whale([1.0, 1, 0], ws(), tk(vol=1e6))
    assert r.orders[0]["qty"] == DEFAULTS["max_parent_remaining"]
    r = decode_whale([1.0, 1, 0], ws(), tk(vol=1e6), {"max_parent_remaining": None})
    assert r.orders[0]["qty"] == 100_000


def test_volume_fallbacks():
    t = tk()
    t["volume_ma"] = None
    r = decode_whale([0.5, 1, 0], ws(), t)
    assert r.orders[0]["qty"] == 750                                    # step_volume 5000
    assert any("step_volume" in n for n in r.notes)
    t["step_volume"] = None
    r = decode_whale([0.5, 1, 0], ws(), t)
    assert r.orders == [] and any("no child" in n for n in r.notes)


def test_missing_book_gives_no_whale_child():
    r = decode_whale([0.5, 1, 1], ws(), tk(bid=None, ask=None))
    assert r.orders == [] and any(n.startswith("book") for n in r.notes)


# ------------------------------------------------------ whale: price mapping
@pytest.mark.parametrize("frac,expected", [(0.0, 100.00), (1 / 3, 100.05), (2 / 3, 100.10), (1.0, 100.25)])
def test_aggressiveness_buy_prices(frac, expected):
    r = decode_whale([0.5, 1, x_for(frac)], ws(0), tk())
    o = r.orders[0]
    assert o["action"] == "BUY" and o["price"] == pytest.approx(expected)
    assert o["type"] == "LIMIT" and o["tif_ticks"] == 1


@pytest.mark.parametrize("frac,expected", [(0.0, 100.10), (1 / 3, 100.05), (2 / 3, 100.00), (1.0, 99.85)])
def test_aggressiveness_sell_prices(frac, expected):
    r = decode_whale([-0.5, 1, x_for(frac)], ws(0), tk())
    o = r.orders[0]
    assert o["action"] == "SELL" and o["price"] == pytest.approx(expected)


def test_child_price_snapped_to_tick_grid():
    for a in [-1, -0.77, -0.5, -0.123, 0, 0.2, 0.41, 0.9, 1]:
        for tick_size, bid, ask in [(0.05, 100.00, 100.10), (0.01, 50.00, 50.03), (0.10, 2000.0, 2000.3)]:
            r = decode_whale([0.5, 1, a], ws(), tk(bid=bid, ask=ask, tick_size=tick_size))
            assert on_grid(r.orders[0]["price"], tick_size), (a, tick_size, r.orders[0]["price"])


def test_one_tick_spread_inside_equals_touch():
    r = decode_whale([0.5, 1, x_for(1 / 3)], ws(), tk(bid=100.00, ask=100.05))
    assert r.orders[0]["price"] == pytest.approx(100.05)


def test_cancel_first_when_previous_child_resting():
    r = decode_whale([0.5, 1, 1], ws(has_resting=True), tk())
    assert r.orders[0] == {"cancel_all": True} and r.orders[1]["action"] == "BUY"
    assert any(n.startswith("cancel") for n in r.notes)
    r = decode_whale([0.5, 1, 1], ws(has_resting=False), tk())
    assert r.orders[0].get("action") == "BUY"
    assert decode_whale([0.02, 1, 1], ws(30_000, has_resting=True), tk()).orders[0] == {"cancel_all": True}


def test_tick_size_fallback_noted():
    t = tk()
    del t["tick_size"]
    r = decode_whale([0.5, 1, 1], ws(), t)
    assert any(n.startswith("tick") for n in r.notes) and r.intent["tick_size"] == 0.05


# ------------------------------------------------------------------- MM
def mm(action, pos=0, cap=10_000, mq=1_000, **tick_kw):
    return decode_mm(action, AgentState(position=pos, cash=1e9, capacity=cap, max_quote_size=mq),
                     tk(**{"bid": 100.00, "ask": 100.50, **tick_kw}))


def quote(r, action):
    return next((o for o in r.orders if o.get("action") == action), None)


def test_mm_output_shape():
    r = mm([0, 0, 0, 0])
    assert r.orders[0] == {"cancel_all": True}
    b, a = quote(r, "BUY"), quote(r, "SELL")
    assert b["type"] == a["type"] == "LIMIT" and b["tif_ticks"] == a["tif_ticks"] == 1
    assert len(r.orders) == 3


def test_mm_offset_endpoints():
    r = mm([-1, -1, 0, 0])
    assert r.intent["offsets"] == [-1, -1]
    assert quote(r, "BUY")["price"] == pytest.approx(100.05) and quote(r, "SELL")["price"] == pytest.approx(100.45)
    r = mm([1, 1, 0, 0])
    assert r.intent["offsets"] == [10, 10]
    assert quote(r, "BUY")["price"] == pytest.approx(99.50) and quote(r, "SELL")["price"] == pytest.approx(101.00)
    r = mm([-9 / 11, -9 / 11, 0, 0])
    assert r.intent["offsets"] == [0, 0]
    assert quote(r, "BUY")["price"] == pytest.approx(100.00) and quote(r, "SELL")["price"] == pytest.approx(100.50)


def test_mm_offsets_independent_per_side():
    r = mm([-1, 1, 0, 0])
    assert r.intent["offsets"] == [-1, 10]


def test_mm_inside_quote_refused_at_one_tick_spread():
    r = mm([-1, -1, 0, 0], bid=100.00, ask=100.05)
    assert r.intent["offsets"] == [0, 0] and r.intent["offsets_requested"] == [-1, -1]
    assert quote(r, "BUY")["price"] == pytest.approx(100.00) and quote(r, "SELL")["price"] == pytest.approx(100.05)
    assert any("inside quote not allowed, spread 1 tick" in n for n in r.notes)


def test_mm_inside_quotes_never_meet_in_two_tick_spread():
    r = mm([-1, -1, 0, 0], bid=100.00, ask=100.10)
    assert quote(r, "BUY")["price"] == pytest.approx(100.05)
    assert quote(r, "SELL")["price"] > quote(r, "BUY")["price"]
    assert any("meet" in n for n in r.notes)


def test_mm_never_crosses_locked_or_crossed_book():
    r = mm([-1, -1, 0, 0], bid=100.00, ask=100.00)
    assert quote(r, "BUY")["price"] < 100.00 < quote(r, "SELL")["price"]
    assert any("pushed back" in n for n in r.notes)
    r = mm([-9 / 11, -9 / 11, 0, 0], bid=100.10, ask=100.00)
    assert quote(r, "BUY")["price"] < 100.00 and quote(r, "SELL")["price"] > 100.10


def test_mm_prices_on_grid():
    for a in [-1, -0.6, -0.1, 0.3, 0.77, 1]:
        r = mm([a, a, 0, 0])
        assert all(on_grid(o["price"]) for o in r.orders if "price" in o)


def test_mm_size_mapping():
    r = mm([0, 0, 0, 1])
    assert quote(r, "BUY")["qty"] == 500 and quote(r, "SELL")["qty"] == 1_000


def test_mm_size_floor():
    r = mm([0, 0, -0.95, 0])
    assert quote(r, "BUY") is None and quote(r, "SELL") is not None
    assert "mm: bid size below 5%" in r.notes and r.intent["bid_qty"] == 0 and r.intent["bid_px"] is None
    r = mm([0, 0, -1, -1])
    assert r.orders == [{"cancel_all": True}]
    r = mm([0, 0, -0.9, 0])          # 50 = exactly the floor -> quoted
    assert quote(r, "BUY")["qty"] == 50


def test_mm_position_limit_caps():
    r = mm([0, 0, 1, 1], pos=9_800)
    assert quote(r, "BUY")["qty"] == 200 and quote(r, "SELL")["qty"] == 1_000
    assert any(n.startswith("cap: position limit bid") for n in r.notes)
    r = mm([0, 0, 1, 1], pos=-9_900)
    assert quote(r, "SELL")["qty"] == 100 and quote(r, "BUY")["qty"] == 1_000
    r = mm([0, 0, 1, 1], pos=10_000)
    assert quote(r, "BUY") is None and quote(r, "SELL") is not None


def test_mm_no_inventory_skew():
    a = mm([0, 0, 0, 0], pos=0)
    b = mm([0, 0, 0, 0], pos=5_000)
    assert quote(a, "BUY")["price"] == quote(b, "BUY")["price"]
    assert quote(a, "SELL")["price"] == quote(b, "SELL")["price"]


def test_mm_missing_book_quotes_around_last_price():
    r = mm([-9 / 11, -9 / 11, 0, 0], bid=None, ask=None, last_price=100.0)
    assert quote(r, "BUY")["price"] == pytest.approx(99.95) and quote(r, "SELL")["price"] == pytest.approx(100.05)
    assert any(n.startswith("book") for n in r.notes)
    r = mm([-1, -1, 0, 0], bid=None, ask=None, last_price=100.0)     # 2-tick fallback spread
    assert quote(r, "BUY")["price"] == pytest.approx(100.0)
    assert quote(r, "SELL")["price"] > quote(r, "BUY")["price"]
    r = mm([0, 0, 0, 0], bid=100.0, ask=None, last_price=100.0)       # only one side missing
    assert any(n.startswith("book") for n in r.notes)


def test_mm_out_of_range_clipped():
    r = mm([-7, 7, 7, -7])
    assert r.raw == [-1.0, 1.0, 1.0, -1.0]
    assert r.intent["offsets"] == [-1, 10] and quote(r, "BUY")["qty"] == 1_000 and quote(r, "SELL") is None
    assert sum(n.startswith("clip") for n in r.notes) == 4


def test_mm_max_quote_size_zero_means_no_quotes():
    r = mm([0, 0, 1, 1], mq=0)
    assert r.orders == [{"cancel_all": True}]


# ------------------------------------------------- execution_v2 vs original
def _ticks(n=8, bid=1349.95, ask=1350.05):
    return [{"day_count": 1, "market_minute": b, "symbol": "RELIANCE", "tick_size": 0.05,
             "best_bid": bid, "best_ask": ask, "last_price": 1350.0, "volume_ma": 4000.0}
            for b in range(1, n + 1)]


def _drive(slicer, side, target, kwargs):
    slicer.on_ack({"pos": 0, "cash": 1e6, "status": "CANCELLED"})
    out, pos = [slicer.retarget(target, side, 1, 1, **kwargs)], 0
    for t in _ticks():
        orders = slicer.tick_orders(t)
        out.append(orders)
        for o in orders:
            if o.get("action"):
                pos += (1 if o["action"] == "BUY" else -1) * (o["qty"] // 2)
        slicer.on_ack({"pos": pos, "cash": 1e6 - pos * 1350.0, "status": "FILLED"})
    return out


@pytest.mark.parametrize("side,target", [("BUY", 10_000), ("SELL", -10_000)])
def test_v2_matches_original_at_defaults(side, target):
    old = _drive(PovSlicer("W"), side, target, {})
    new = _drive(PovSlicerV2("W"), side, target, {})
    explicit = _drive(PovSlicerV2("W"), side, target,
                      {"participation": 0.15, "aggressiveness": MARKETABLE_AGGRESSIVENESS})
    assert any(o.get("action") for step in old for o in step)
    assert old == new == explicit


def test_v2_per_parent_participation_changes_child_size():
    s = PovSlicerV2("W")
    s.on_ack({"pos": 0, "cash": 1e6, "status": "CANCELLED"})
    s.retarget(10_000, "BUY", 1, 1, participation=0.01, aggressiveness=0.0)
    o = s.tick_orders(_ticks()[1])[0]
    assert o["qty"] == int(0.01 * s.curve.expected("RELIANCE", 2)) and o["price"] == 1349.95
    s.retarget(10_000, "BUY", 1, 1, participation=0.15, aggressiveness=1.0)     # same side: updates parent
    o = s.tick_orders(_ticks()[1])[0]
    assert o["qty"] == min(10_000, int(0.15 * s.curve.expected("RELIANCE", 2))) and o["price"] == pytest.approx(1350.20)


@pytest.mark.parametrize("pos,target_frac,part,aggr", [(0, 0.5, 0.3, 0.2), (30_000, -0.6, -1.0, 0.9), (0, 0.9, 1.0, 1.0)])
def test_decoder_child_equals_slicer_child(pos, target_frac, part, aggr):
    """Training decoder and the live slicer share the child size/price code."""
    t = tk(bid=1349.95, ask=1350.05, vol=4000.0)
    r = decode_whale([target_frac, part, aggr], ws(pos), t)
    s = PovSlicerV2("W", volume_source="tick")
    s.on_ack({"pos": pos, "cash": 1e9, "status": "CANCELLED"})
    s.retarget(r.intent["target_shares"], r.intent["side"], 1, 4,
               r.intent["participation"], r.intent["aggressiveness"])
    live = s.tick_orders(t)[0]
    dec = r.orders[0]
    assert (live["action"], live["type"], live["price"], live["qty"], live["tif_ticks"]) == \
           (dec["action"], dec["type"], dec["price"], dec["qty"], dec["tif_ticks"])
