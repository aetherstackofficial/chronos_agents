"""Phase 2 steps 1-2: retail personality, seeding, and per-agent starting cash."""
import copy
import logging
import math
import os
import statistics
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

from agents import heuristic_retail as hr  # noqa: E402
from engine.matching_engine import MatchingEngine  # noqa: E402


def make_engine(seed=42):
    e = MatchingEngine()
    e.init_sim("TCS", "TECH", 190.0, seed=seed)
    return e


def cancel_all(agent, **extra):
    return {"msg": "CANCEL_ALL", "agent_id": agent, **extra}


# --- engine: per-agent starting cash -------------------------------------------

def test_retail_first_message_sets_starting_cash():
    e = make_engine()
    ack, _ = e.handle_message(cancel_all("RETAIL_1", start_cash=123_456.0))
    assert ack["cash"] == 123_456.0
    assert e.agent_ledger["RETAIL_1"]["start_cash"] == 123_456.0


def test_start_cash_works_on_an_order_too():
    e = make_engine()
    e.handle_message({"msg": "ORDER", "agent_id": "MM_1", "action": "SELL",
                      "type": "LIMIT", "qty": 100, "price": 190.10})
    ack, _ = e.handle_message({"msg": "ORDER", "agent_id": "RETAIL_2", "action": "BUY",
                               "type": "MARKET", "qty": 10, "start_cash": 200_000.0})
    assert ack["cash"] == pytest.approx(200_000.0 - 10 * 190.10)


def test_start_cash_is_ignored_after_the_ledger_exists():
    e = make_engine()
    e.handle_message(cancel_all("RETAIL_1", start_cash=100_000.0))
    ack, _ = e.handle_message(cancel_all("RETAIL_1", start_cash=999_999.0))
    assert ack["cash"] == 100_000.0


@pytest.mark.parametrize("agent", ["MM_1", "WHALE_1", "STRAT_x", "USER_x"])
def test_start_cash_is_ignored_for_non_retail(agent):
    e = make_engine()
    fixed = e._classify_agent(agent)[1]
    ack, _ = e.handle_message(cancel_all(agent, start_cash=12_345.0))
    assert ack["cash"] == fixed


@pytest.mark.parametrize("bad", [-5.0, 0, 9_999.0, 1e9 + 1, float("nan"),
                                 float("inf"), "100000", True, None, [1]])
def test_bad_start_cash_falls_back_to_the_default(bad):
    e = make_engine()
    ack, _ = e.handle_message(cancel_all("RETAIL_9", start_cash=bad))
    assert ack["cash"] == MatchingEngine.CASH_RETAIL


def test_liquidation_resets_to_the_agents_own_starting_cash():
    e = make_engine()
    e.handle_message(cancel_all("RETAIL_1", start_cash=200_000.0))
    e.handle_message(cancel_all("RETAIL_2"))  # default 50k
    for aid in ("RETAIL_1", "RETAIL_2"):
        e.agent_ledger[aid]["cash"], e.agent_ledger[aid]["pos"] = 500.0, 0
    e._check_liquidations(["RETAIL_1", "RETAIL_2"], [])
    assert e.agent_ledger["RETAIL_1"]["cash"] == 200_000.0
    assert e.agent_ledger["RETAIL_2"]["cash"] == MatchingEngine.CASH_RETAIL


# --- personality ---------------------------------------------------------------

def test_same_seed_and_id_give_the_same_personality():
    assert hr.sample_personality("RETAIL_3", 7) == hr.sample_personality("RETAIL_3", 7)


def test_different_seed_or_id_gives_a_different_personality():
    base = hr.sample_personality("RETAIL_3", 7)
    assert base != hr.sample_personality("RETAIL_3", 8)
    assert base != hr.sample_personality("RETAIL_4", 7)


def test_personality_does_not_depend_on_creation_order():
    forward = [hr.sample_personality(f"RETAIL_{i}", 5) for i in range(20)]
    backward = [hr.sample_personality(f"RETAIL_{i}", 5) for i in reversed(range(20))]
    assert forward == list(reversed(backward))


def test_personality_fields_stay_inside_the_frozen_bounds():
    cfg = hr.CONFIG
    p = cfg["personality"]
    for i in range(500):
        s = hr.sample_personality(f"RETAIL_{i}", 1)
        assert p["rsi_center"]["clip"][0] <= s.rsi_center <= p["rsi_center"]["clip"][1]
        assert p["conviction_threshold"]["low"] <= s.conviction_threshold <= p["conviction_threshold"]["high"]
        assert p["cooldown_ticks"]["low"] <= s.cooldown_ticks <= p["cooldown_ticks"]["high"]
        assert cfg["orders"]["risk_appetite"]["low"] <= s.risk_appetite <= cfg["orders"]["risk_appetite"]["high"]
        assert math.isclose(sum(s.exit_preference), 1.0)
        assert s.start_cash >= cfg["wealth"]["starting_cash"]["per_agent"]["minimum"] - 1
        assert s.start_cash <= cfg["wealth"]["starting_cash"]["per_agent"]["clip_max"]
        assert s.max_position_cap == 12_500  # 25% of the smallest whale (50,000)


def test_type_mix_is_about_40_40_20():
    kinds = [hr.sample_personality(f"RETAIL_{i}", 11).kind for i in range(4000)]
    share = {k: kinds.count(k) / len(kinds) for k in set(kinds)}
    assert share["momentum"] == pytest.approx(0.40, abs=0.03)
    assert share["mean_reverter"] == pytest.approx(0.40, abs=0.03)
    assert share["noise"] == pytest.approx(0.20, abs=0.03)


def test_starting_cash_is_heavy_tailed_with_few_rich_agents():
    spec = hr.CONFIG["wealth"]["starting_cash"]["per_agent"]
    cash = [hr.sample_personality(f"RETAIL_{i}", 3).start_cash for i in range(4000)]
    assert statistics.median(cash) == pytest.approx(spec["minimum"] * 2 ** (1 / spec["shape"]), rel=0.10)
    assert statistics.mean(cash) > 1.5 * statistics.median(cash)  # heavy right tail
    cap_cash = hr.CONFIG["wealth"]["max_position"]["hard_cap_shares"] * 1354.0   # cash needed for the ceiling
    assert sum(c >= cap_cash for c in cash) / len(cash) < 0.03                   # the ceiling is rare


def test_engine_default_cash_source_gives_everyone_50k():
    cfg = copy.deepcopy(hr.CONFIG)
    cfg["wealth"]["starting_cash"]["source"] = "engine_default"
    assert {hr.sample_personality(f"RETAIL_{i}", 2, cfg).start_cash for i in range(50)} == {50_000.0}


def test_max_position_is_limited_by_what_the_agent_can_afford():
    p = hr.sample_personality("RETAIL_1", 1)
    assert p.max_shares_at_price(0) == 0
    assert p.max_shares_at_price(1354.0) == min(p.max_position_cap, int(p.start_cash / 1354.0))
    assert p.max_shares_at_price(0.01) == p.max_position_cap  # cheap stock: the cap binds


# --- live agent (loads the PPO models, ~10 s) -----------------------------------

la = pytest.importorskip("agents.live_agents")


def _tick(t_minute, **kw):
    t = {"seq": 1, "last_price": 1350.0, "mid_price": 1350.0, "best_bid": 1349.9,
         "best_ask": 1350.1, "tick_size": 0.1, "vwap": 1350.0, "volume_ma": 1000.0,
         "step_volume": 1000, "volatility": 0.001, "rsi": 20.0, "ofi": 0.0,
         "day_count": 0, "market_minute": t_minute, "symbol": "RELIANCE"}
    t.update(kw)
    return t


def _run(agent, minutes=200):
    out = []
    for m in range(1, minutes + 1):
        out += [(m, o) for o in agent.on_tick(_tick(m))]
    return out


def test_retail_agent_is_reproducible_for_a_seed():
    assert _run(la.LiveRetailAgent("RETAIL_5", seed=9)) == _run(la.LiveRetailAgent("RETAIL_5", seed=9))


def test_retail_agent_starts_with_its_personal_cash_and_sends_it():
    a = la.LiveRetailAgent("RETAIL_5", seed=9)
    assert a.cash == a.personality.start_cash
    orders = _run(a)
    assert orders, "an agent that sees RSI 20 should buy at least once in 200 minutes"
    assert all(o["start_cash"] == a.personality.start_cash for _, o in orders)


def test_retail_cooldown_counts_engine_minutes_not_wall_clock(monkeypatch):
    def boom():
        raise AssertionError("retail must not read the wall clock")
    monkeypatch.setattr(la.time, "time", boom)
    a = la.LiveRetailAgent("RETAIL_5", seed=9)
    placed = [m for m, _ in _run(a)]
    assert placed
    gaps = [b - a_ for a_, b in zip(placed, placed[1:])]
    assert all(g >= a.personality.cooldown_ticks for g in gaps)


def test_retail_cooldown_spans_the_day_boundary():
    a = la.LiveRetailAgent("RETAIL_5", seed=9)
    assert a.personality.cooldown_ticks >= 3
    a._last_trade_t = a._tick_index(_tick(375))          # last minute of day 0
    assert a.on_tick(_tick(1, day_count=1)) == []        # one minute later: still cooling down
    assert a._tick_index(_tick(1, day_count=1)) == 376


def test_liquidation_mirror_restores_own_cash():
    a = la.LiveRetailAgent("RETAIL_5", seed=9)
    a.cash, a.inventory = 10.0, 77
    a.on_liquidation()
    assert (a.cash, a.inventory) == (a.personality.start_cash, 0)


def test_worker_builds_the_same_crowd_for_the_same_seed():
    import worker
    one = [x.personality for x in worker.build_agents("retail", 10, seed=21)]
    two = [x.personality for x in worker.build_agents("retail", 10, seed=21)]
    other = [x.personality for x in worker.build_agents("retail", 10, seed=22)]
    assert one == two and one != other


# --- step 4: signal and order size (pure logic, no engine) ----------------------

from dataclasses import replace  # noqa: E402


def _pers(kind="momentum", **kw):
    base = hr.sample_personality("RETAIL_T", 1)
    fields = dict(kind=kind, rsi_center=50.0, conviction_threshold=0.20, cooldown_ticks=3,
                  risk_appetite=0.05, start_cash=100_000.0,
                  exit_preference=(1.0, 0.0, 0.0, 0.0), level_exits=False)
    fields.update(kw)
    return replace(base, **fields)


def test_momentum_buys_strength_and_mean_reverter_buys_weakness():
    mom, rev = _pers("momentum"), _pers("mean_reverter")
    slope = hr.CONFIG["personality"]["logistic_slope"]
    assert hr.conviction(60, mom, slope) > 0 > hr.conviction(40, mom, slope)
    assert hr.conviction(60, rev, slope) < 0 < hr.conviction(40, rev, slope)
    assert hr.conviction(50, mom, slope) == pytest.approx(0.0)
    assert hr.conviction(60, mom, slope) == pytest.approx(-hr.conviction(60, rev, slope))


def test_conviction_is_smooth_bounded_and_zero_for_noise_agents():
    mom = _pers("momentum")
    values = [hr.conviction(r, mom, 0.25) for r in range(0, 101, 5)]
    assert all(-1.0 <= v <= 1.0 for v in values)
    assert values == sorted(values)                       # monotone in RSI
    assert hr.conviction(55, mom, 0.25) > 0               # trades at RSI 55 too, just weakly
    assert hr.conviction(55, mom, 0.25) < hr.conviction(65, mom, 0.25)
    assert hr.conviction(80, _pers("noise"), 0.25) == 0.0


def test_order_qty_respects_minimum_cap_and_cash():
    p = _pers(risk_appetite=0.05, start_cash=100_000.0)
    assert hr.order_qty(p, 0.01, 1354.0, 100_000.0, +1) == 1                      # tiny signal still 1 share
    assert hr.order_qty(p, 1.0, 1354.0, 100_000.0, +1) == int(0.05 * 100_000 / 1354.0)
    assert hr.order_qty(p, 1.0, 1354.0, 500.0, +1) == 0                           # cannot pay for a share
    assert hr.order_qty(p, 1.0, 1354.0, 500.0, -1) >= 1                           # a short needs no cash
    assert hr.order_qty(p, 1.0, 0.0, 100_000.0, +1) == 0
    rich = _pers(risk_appetite=0.05, start_cash=1e9)
    assert hr.order_qty(rich, 1.0, 10.0, 1e9, +1) == rich.max_position_cap         # hard cap binds


# --- step 3: exit plan and exit rules (pure logic) --------------------------------

def _pos(side=1, entry=100.0, t=0, plan=None):
    plan = plan or hr.ExitPlan("time_based", 750, hold_ticks=10)
    return hr.Position(side=side, qty=5, entry_price=entry, entry_t=t, plan=plan, best_price=entry)


def test_exit_plan_follows_the_agents_preference_and_bounds():
    import random
    ex = hr.CONFIG["exits"]
    rng = random.Random(3)
    p = _pers(exit_preference=(0.7, 0.1, 0.1, 0.1))
    plans = [hr.draw_exit_plan(p, rng) for _ in range(3000)]
    share = {s: sum(x.style == s for x in plans) / len(plans) for s in
             ("time_based", "target_stop", "trailing_stop", "signal_reversal")}
    assert share["time_based"] == pytest.approx(0.7, abs=0.04)
    assert share["target_stop"] == pytest.approx(0.1, abs=0.03)
    for x in plans:
        assert x.max_hold_ticks == ex["max_hold_ticks"]
        if x.style == "time_based":
            clip = ex["time_based"]["hold_ticks"]["clip"]
            assert clip[0] <= x.hold_ticks <= clip[1]
        if x.style == "target_stop":
            assert 0.30 <= x.take_profit_pct <= 2.00 and 0.30 <= x.stop_loss_pct <= 1.50
        if x.style == "trailing_stop":
            assert 0.30 <= x.trail_pct <= 1.00


def test_holding_time_varies_from_trade_to_trade():
    import random
    rng = random.Random(4)
    p = _pers(exit_preference=(1.0, 0.0, 0.0, 0.0))
    assert len({hr.draw_exit_plan(p, rng).hold_ticks for _ in range(200)}) > 20


def test_noise_agents_always_use_their_configured_exit_style():
    import random
    rng = random.Random(5)
    p = _pers("noise", exit_preference=(0.0, 0.0, 0.0, 1.0))
    assert {hr.draw_exit_plan(p, rng).style for _ in range(100)} == {"time_based"}


def test_time_based_exit():
    pos = _pos(plan=hr.ExitPlan("time_based", 750, hold_ticks=10))
    assert hr.exit_reason(pos, 100.0, 9, 0.0, 0.2) is None
    assert hr.exit_reason(pos, 100.0, 10, 0.0, 0.2) == "time"


@pytest.mark.parametrize("side,price,expected", [
    (+1, 101.5, "take_profit"), (+1, 98.5, "stop_loss"), (+1, 100.5, None),
    (-1, 98.5, "take_profit"), (-1, 101.5, "stop_loss"), (-1, 99.5, None)])
def test_target_stop_exit_is_mirrored_for_shorts(side, price, expected):
    plan = hr.ExitPlan("target_stop", 750, take_profit_pct=1.0, stop_loss_pct=1.0)
    assert hr.exit_reason(_pos(side, plan=plan), price, 1, 0.0, 0.2) == expected


def test_trailing_stop_follows_the_best_price():
    plan = hr.ExitPlan("trailing_stop", 750, trail_pct=1.0)
    long_pos = _pos(+1, plan=plan)
    for price in (100.0, 103.0):
        long_pos.mark(price)
    assert long_pos.best_price == 103.0
    assert hr.exit_reason(long_pos, 102.5, 5, 0.0, 0.2) is None
    assert hr.exit_reason(long_pos, 101.5, 5, 0.0, 0.2) == "trailing"
    short_pos = _pos(-1, plan=plan)
    for price in (100.0, 97.0):
        short_pos.mark(price)
    assert short_pos.best_price == 97.0
    assert hr.exit_reason(short_pos, 97.5, 5, 0.0, 0.2) is None
    assert hr.exit_reason(short_pos, 98.5, 5, 0.0, 0.2) == "trailing"


def test_signal_reversal_exit_and_max_hold_safety_net():
    plan = hr.ExitPlan("signal_reversal", 750)
    assert hr.exit_reason(_pos(+1, plan=plan), 100.0, 5, -0.1, 0.2) is None
    assert hr.exit_reason(_pos(+1, plan=plan), 100.0, 5, -0.2, 0.2) == "reversal"
    assert hr.exit_reason(_pos(-1, plan=plan), 100.0, 5, +0.2, 0.2) == "reversal"
    assert hr.exit_reason(_pos(+1, plan=plan), 100.0, 750, +0.9, 0.2) == "max_hold"
    slow = hr.ExitPlan("target_stop", 100, take_profit_pct=50.0, stop_loss_pct=50.0)
    assert hr.exit_reason(_pos(+1, plan=slow), 100.0, 100, 0.0, 0.2) == "max_hold"


# --- steps 3 + 4 on the live agent, against a stand-in venue ----------------------

class Venue:
    """Fills every market order at the given price and answers like an ack."""
    def __init__(self, cash, fill_fraction=1.0):
        self.cash, self.pos, self.n, self.frac = cash, 0, 0, fill_fraction

    def send(self, order, price):
        self.n += 1
        qty = max(1, int(order["qty"] * self.frac)) if self.frac < 1 else order["qty"]
        sign = 1 if order["action"] == "BUY" else -1
        self.cash -= sign * qty * price
        self.pos += sign * qty
        return {"msg": "ACK", "status": "FILLED" if qty == order["qty"] else "PARTIAL",
                "order_id": self.n, "executed_qty": qty, "average_price": price,
                "resting_qty": 0, "cash": round(self.cash, 2), "pos": self.pos,
                "client_order_id": order["client_order_id"]}


def _agent(kind="momentum", allow_short=True, **kw):
    cfg = copy.deepcopy(hr.CONFIG)
    cfg["personality"]["base_trade_prob"] = 1.0           # remove the entry dice
    cfg["personality"]["noise_agent"]["trade_prob_per_tick"] = 1.0
    cfg["wealth"]["allow_short"] = allow_short
    cfg["orders"]["type_mix"] = {"market": 1.0, "limit": 0.0}          # market orders only
    cfg["levels"]["anchor_prob"] = 0.0                                 # levels have their own tests
    cfg["emotions"]["loss_overtrade"]["probability"] = 0.0             # emotions have their own tests
    cfg["emotions"]["profit_fear"]["probability"] = 0.0
    cfg["emotions"]["capitulation"]["consecutive_losses"] = 99
    a = la.LiveRetailAgent("RETAIL_T", seed=1, cfg=cfg)
    a.personality = _pers(kind, **kw)
    a.cash = a.personality.start_cash
    return a, Venue(a.cash)


def _drive(agent, venue, script):
    out = []
    for minute, price, rsi in script:
        for order in agent.on_tick(_tick(minute, last_price=price, rsi=rsi)):
            out.append((minute, order))
            agent.on_ack(venue.send(order, price))
    return out


def test_agent_enters_holds_exits_and_books_the_result():
    a, venue = _agent("momentum")
    # Strong RSI -> buy; the price rises while it holds, then the time exit sells.
    out = _drive(a, venue, [(m, 1350.0 + m, 70.0) for m in range(1, 200)])
    entry_min, entry = out[0]
    assert entry["action"] == "BUY" and entry["qty"] >= 1
    assert a.closed_trades, "the time-based exit should have closed the trade"
    trade = a.closed_trades[0]
    assert trade["reason"] == "time" and trade["side"] == 1
    assert trade["pnl"] == pytest.approx((trade["exit_price"] - trade["entry_price"]) * trade["qty"])
    assert trade["pnl"] > 0                                    # price rose while held
    exit_min, exit_order = next((m, o) for m, o in out if o["action"] == "SELL")
    assert exit_min - entry_min == trade["held_ticks"] and exit_order["qty"] == trade["qty"]


def test_agent_does_not_pile_into_a_second_position_while_holding():
    a, venue = _agent("momentum")
    a.personality = replace(a.personality, exit_preference=(0.0, 0.0, 1.0, 0.0))   # trailing: holds while price rises
    out = _drive(a, venue, [(m, 1350.0 + m, 70.0) for m in range(1, 60)])
    assert [o["action"] for _, o in out] == ["BUY"]
    assert a.position is not None and a.position.side == 1


def test_agent_can_go_short_and_profits_when_price_falls():
    a, venue = _agent("momentum")
    out = _drive(a, venue, [(m, 1350.0 - m, 30.0) for m in range(1, 200)])   # weak RSI -> a momentum agent shorts
    assert out[0][1]["action"] == "SELL"
    assert a.closed_trades and a.closed_trades[0]["side"] == -1
    assert a.closed_trades[0]["pnl"] > 0


def test_agent_never_shorts_when_shorting_is_off():
    a, venue = _agent("momentum", allow_short=False)
    assert _drive(a, venue, [(m, 1350.0, 30.0) for m in range(1, 100)]) == []


def test_agent_stays_flat_below_its_conviction_threshold():
    a, venue = _agent("momentum", conviction_threshold=0.9)
    assert _drive(a, venue, [(m, 1350.0, 55.0) for m in range(1, 100)]) == []


def test_asleep_agent_sends_nothing_and_cooldown_separates_orders():
    a, venue = _agent("momentum")
    a.is_asleep = True
    assert _drive(a, venue, [(m, 1350.0, 70.0) for m in range(1, 20)]) == []
    a.is_asleep = False
    out = _drive(a, venue, [(m, 1350.0, 70.0) for m in range(1, 400)])
    minutes = [m for m, _ in out]
    assert len(minutes) > 3
    assert all(b - x >= a.personality.cooldown_ticks for x, b in zip(minutes, minutes[1:]))


def test_unacked_order_blocks_the_agent_until_it_times_out():
    a, venue = _agent("momentum")
    assert len(a.on_tick(_tick(1, rsi=70.0))) == 1               # sent, ack never delivered
    for m in range(2, 5):
        assert a.on_tick(_tick(m, rsi=70.0)) == []               # still waiting
    a.on_tick(_tick(1 + a.PENDING_TIMEOUT_TICKS + 1, rsi=70.0))
    assert a._pending is None or a._pending["t"] > 1             # forgotten, free to act again


def test_shares_that_appear_without_a_tracked_position_are_adopted():
    a, venue = _agent("momentum")
    a.inventory = 7
    a.on_tick(_tick(1, rsi=50.0))
    assert a.position is not None and a.position.side == 1 and a.position.qty == 7


def test_liquidation_clears_the_position():
    a, venue = _agent("momentum")
    _drive(a, venue, [(m, 1350.0, 70.0) for m in range(1, 5)])
    assert a.position is not None
    a.on_liquidation()
    assert a.position is None and a._pending is None
    assert (a.cash, a.inventory) == (a.personality.start_cash, 0)


def test_partial_exit_fill_keeps_the_position_until_it_is_flat():
    a, venue = _agent("momentum", risk_appetite=0.05, start_cash=5_000_000.0)
    a.cash = venue.cash = 5_000_000.0
    _drive(a, venue, [(1, 1350.0, 70.0)])
    assert a.position is not None and a.position.qty > 2
    entry_qty = a.position.qty
    venue.frac = 0.5
    _drive(a, venue, [(m, 1350.0, 70.0) for m in range(2, 400)])
    assert a.closed_trades, "the remainder should be sold on a later tick"
    assert a.closed_trades[0]["qty"] == entry_qty


def test_noise_agent_trades_both_sides_without_a_signal():
    a, venue = _agent("noise")
    out = _drive(a, venue, [(m, 1350.0, 50.0) for m in range(1, 3000)])
    assert {o["action"] for _, o in out} == {"BUY", "SELL"}
    assert {t["plan"] for t in a.closed_trades} == {"time_based"}


def test_same_seed_gives_identical_orders_on_the_same_market():
    script = [(m, 1350.0 + (m % 7), 30.0 + (m * 13) % 40) for m in range(1, 370)]
    runs = []
    for _ in range(2):
        a, venue = _agent("mean_reverter")
        runs.append(_drive(a, venue, script))
    assert runs[0] == runs[1] and len(runs[0]) > 3


# --- limit orders, participation cap, profit fear, overtrading, capitulation -----

def test_limit_price_sits_on_the_agents_own_side_and_on_the_tick_grid():
    assert hr.limit_price(+1, 1349.9, 1350.1, 1350.0, 0.1, 0) == pytest.approx(1349.9)   # join the bid
    assert hr.limit_price(+1, 1349.9, 1350.1, 1350.0, 0.1, 3) == pytest.approx(1349.6)   # three ticks behind
    assert hr.limit_price(-1, 1349.9, 1350.1, 1350.0, 0.1, 0) == pytest.approx(1350.1)   # join the ask
    assert hr.limit_price(-1, 1349.9, 1350.1, 1350.0, 0.1, 2) == pytest.approx(1350.3)
    assert hr.limit_price(+1, None, None, 1350.04, 0.1, 0) == pytest.approx(1350.0)      # no quote: last price, snapped
    assert hr.limit_price(+1, 0.05, 0.1, 0.05, 0.05, 5) == pytest.approx(0.05)           # never below one tick


def test_order_qty_size_multiplier_and_participation_cap():
    p = _pers(risk_appetite=0.05, start_cash=1_000_000.0)
    base = hr.order_qty(p, 1.0, 1000.0, 1_000_000.0, +1)
    assert hr.order_qty(p, 1.0, 1000.0, 1_000_000.0, +1, size_multiplier=2.0) == 2 * base
    assert hr.order_qty(p, 1.0, 1000.0, 1_000_000.0, +1, size_multiplier=2.0, max_qty=7) == 7


def test_profit_fear_exit_overrides_the_normal_plan():
    plan = hr.ExitPlan("time_based", 750, hold_ticks=500, fear_take_profit_pct=0.30)
    assert hr.exit_reason(_pos(+1, plan=plan), 100.29, 5, 0.0, 0.2) is None
    assert hr.exit_reason(_pos(+1, plan=plan), 100.31, 5, 0.0, 0.2) == "fear_profit"
    assert hr.exit_reason(_pos(-1, plan=plan), 99.69, 5, 0.0, 0.2) == "fear_profit"
    calm = hr.ExitPlan("time_based", 750, hold_ticks=500)                       # 0 = not afraid
    assert hr.exit_reason(_pos(+1, plan=calm), 150.0, 5, 0.0, 0.2) is None


def test_about_40_percent_of_plans_are_afraid_of_losing_their_profit():
    import random
    rng = random.Random(8)
    fear = hr.CONFIG["emotions"]["profit_fear"]
    plans = [hr.draw_exit_plan(_pers(), rng) for _ in range(4000)]
    afraid = [x.fear_take_profit_pct for x in plans if x.fear_take_profit_pct]
    assert len(afraid) / len(plans) == pytest.approx(fear["probability"], abs=0.03)
    assert all(fear["early_take_profit_pct"]["low"] <= v <= fear["early_take_profit_pct"]["high"] for v in afraid)


class BookVenue(Venue):
    """Like Venue, but a LIMIT order rests until the test fills it."""
    def __init__(self, cash):
        super().__init__(cash)
        self.resting = None

    def _ack(self, status, order=None, **kw):
        ack = {"msg": "ACK", "status": status, "order_id": self.n, "executed_qty": 0,
               "average_price": 0.0, "resting_qty": 0, "cash": round(self.cash, 2), "pos": self.pos}
        if order is not None:
            ack["client_order_id"] = order["client_order_id"]
        ack.update(kw)
        return ack

    def send(self, order, price):
        if order.get("cancel_all"):
            n, self.resting = (1 if self.resting else 0), None
            return self._ack("CANCELLED", order_id=0, cancelled=n)
        if order["type"] == "LIMIT":
            self.n += 1
            self.resting = dict(order)
            return self._ack("RESTING", order, resting_qty=order["qty"])
        return super().send(order, price)

    def fill_resting(self, fraction=1.0):
        o = self.resting
        qty = o["qty"] if fraction >= 1 else max(1, int(o["qty"] * fraction))
        sign = 1 if o["action"] == "BUY" else -1
        self.cash -= sign * qty * o["price"]
        self.pos += sign * qty
        o["qty"] -= qty
        if o["qty"] == 0:
            self.resting = None


def _agent2(kind="momentum", market=1.0, emotions=False, hold=5, edit=None, **kw):
    """Agent on a BookVenue with the entry dice removed. Emotions are off
    unless asked for; `market` is the share of MARKET orders (1 = all)."""
    cfg = copy.deepcopy(hr.CONFIG)
    cfg["personality"]["base_trade_prob"] = 1.0
    cfg["personality"]["noise_agent"]["trade_prob_per_tick"] = 1.0
    cfg["orders"]["type_mix"] = {"market": market, "limit": 1.0 - market}
    cfg["levels"]["anchor_prob"] = 0.0                                 # levels have their own tests
    if hold is not None:
        cfg["exits"]["time_based"]["hold_ticks"] = {"dist": "uniform_int", "low": hold, "high": hold}
    if not emotions:
        cfg["emotions"]["loss_overtrade"]["probability"] = 0.0
        cfg["emotions"]["profit_fear"]["probability"] = 0.0
        cfg["emotions"]["capitulation"]["consecutive_losses"] = 99
    if edit:
        edit(cfg)
    a = la.LiveRetailAgent("RETAIL_T", seed=1, cfg=cfg)
    a.personality = _pers(kind, **kw)
    a.cash = a.personality.start_cash
    return a, BookVenue(a.cash)


def _drive2(agent, venue, script):
    """script items: (minute, price, rsi) or (minute, price, rsi, day)."""
    out = []
    for item in script:
        minute, price, rsi = item[:3]
        day = item[3] if len(item) > 3 else 0
        for order in agent.on_tick(_tick(minute, last_price=price, rsi=rsi, day_count=day)):
            out.append((minute, order))
            agent.on_ack(venue.send(order, price))
    return out


def test_participation_cap_limits_every_entry_and_exit_piece(monkeypatch):
    monkeypatch.setattr(la._RETAIL_CURVE, "expected", lambda symbol, minute: 100.0)   # cap = 15% of 100 = 15
    a, venue = _agent2("momentum", risk_appetite=0.05, start_cash=5_000_000.0)
    a.cash = venue.cash = 5_000_000.0
    assert hr.order_qty(a.personality, 0.99, 1350.0, 5_000_000.0, +1) > 15           # the cap really bites
    out = _drive2(a, venue, [(m, 1350.0, 70.0) for m in range(1, 80)])
    assert out and all(o["qty"] <= 15 for _, o in out if "qty" in o)
    assert a.closed_trades and a.closed_trades[0]["qty"] == 15


def test_limit_entry_rests_then_is_resolved_with_cancel_all():
    a, venue = _agent2("momentum", market=0.0)
    out = _drive2(a, venue, [(m, 1350.0, 70.0) for m in range(1, 5)])
    first = out[0][1]
    assert first["type"] == "LIMIT" and first["action"] == "BUY"
    assert first["price"] <= 1349.9 + 1e-9                                  # own side: at or behind the best bid
    assert 5 <= first["tif_ticks"] <= 30
    assert a._working is not None
    sent_at, resolve_t = out[0][0], a._working["resolve_t"]
    assert resolve_t == sent_at + first["tif_ticks"]
    assert len(out) == 1                                                    # quiet while the order rests
    out2 = _drive2(a, venue, [(m, 1350.0, 70.0) for m in range(5, resolve_t + 1)])
    assert [o for _, o in out2] == [{"cancel_all": True}]                    # exactly one resolve, at the TTL
    assert a._working is None and a.position is None                         # nothing filled: flat again


def test_late_limit_fill_is_found_from_the_cash_and_position_change():
    a, venue = _agent2("momentum", market=0.0)
    out = _drive2(a, venue, [(1, 1350.0, 70.0)])
    order = out[0][1]
    venue.fill_resting(1.0)                                                  # the book hits it
    _drive2(a, venue, [(m, 1350.0, 70.0) for m in range(2, 40)])
    assert a.position is not None and a.position.side == 1
    assert a.position.qty == order["qty"]
    assert a.position.entry_price == pytest.approx(order["price"])


def test_partially_filled_limit_entry_holds_only_what_traded():
    a, venue = _agent2("momentum", market=0.0, risk_appetite=0.05, start_cash=5_000_000.0)
    a.cash = venue.cash = 5_000_000.0
    order = _drive2(a, venue, [(1, 1350.0, 70.0)])[0][1]
    assert order["qty"] >= 4
    venue.fill_resting(0.5)
    _drive2(a, venue, [(m, 1350.0, 70.0) for m in range(2, 40)])
    assert a.position is not None and a.position.qty == max(1, int(order["qty"] * 0.5))


def _open_via_limit(a, venue, price=1350.0, rsi=70.0):
    """Place a limit entry, let the book fill it, and advance to the minute the
    agent resolves it (the fill is only learned then). Returns that minute."""
    _drive2(a, venue, [(1, price, rsi)])
    venue.fill_resting(1.0)
    resolve_t = a._working["resolve_t"]
    _drive2(a, venue, [(m, price, rsi) for m in range(2, resolve_t + 1)])
    assert a.position is not None
    return resolve_t


def test_urgent_exits_are_market_orders_even_when_every_other_order_is_a_limit():
    a, venue = _agent2("momentum", market=0.0, hold=None, exit_preference=(0.0, 1.0, 0.0, 0.0))
    resolve_t = _open_via_limit(a, venue)
    out = _drive2(a, venue, [(resolve_t + 1, 1350.0 * 0.95, 70.0)])           # -5%: past any stop-loss
    sells = [o for _, o in out if o.get("action") == "SELL"]
    assert sells and sells[0]["type"] == "MARKET"
    assert a.closed_trades[-1]["reason"] == "stop_loss"


def test_a_calm_exit_goes_out_as_a_passive_limit_order_on_the_ask_side():
    a, venue = _agent2("momentum", market=0.0, hold=5)
    resolve_t = _open_via_limit(a, venue)
    out = _drive2(a, venue, [(m, 1350.0, 70.0) for m in range(resolve_t + 1, resolve_t + 30)])
    sells = [o for _, o in out if o.get("action") == "SELL"]
    assert sells and sells[0]["type"] == "LIMIT" and sells[0]["price"] >= 1350.1 - 1e-9


def test_limit_orders_never_commit_more_than_the_agent_can_pay_or_hold():
    a, venue = _agent2("momentum", market=0.0, risk_appetite=0.05, start_cash=20_000.0)
    a.cash = venue.cash = 20_000.0
    out = _drive2(a, venue, [(m, 1350.0, 70.0 if m % 2 else 30.0) for m in range(1, 600)])
    buys = [o for _, o in out if o.get("action") == "BUY" and "price" in o]
    sells = [o for _, o in out if o.get("action") == "SELL" and "price" in o]
    assert buys and sells
    assert all(o["qty"] * o["price"] <= 20_000.0 for o in buys)
    assert all(o["qty"] <= a.personality.max_shares_at_price(o["price"]) for o in buys + sells)


def _losing_script(days=1, minutes=370, start=1350.0, day0=0):
    """RSI stays high (a momentum agent buys) while the price slides: every trade loses."""
    return [(m, start - 0.5 * m, 70.0, d) for d in range(day0, day0 + days) for m in range(1, minutes)]


def test_after_a_loss_the_agent_trades_harder_for_two_entries_then_calms_down():
    def edit(cfg):
        cfg["emotions"]["loss_overtrade"]["probability"] = 1.0
        cfg["emotions"]["loss_overtrade"]["size_multiplier"] = {"dist": "uniform", "low": 2.0, "high": 2.0}
        cfg["emotions"]["capitulation"]["consecutive_losses"] = 99
    a, venue = _agent2("momentum", emotions=True, edit=edit, cooldown_ticks=10,
                       risk_appetite=0.05, start_cash=2_000_000.0)
    a.cash = venue.cash = 2_000_000.0
    # The first trade loses (price slides until minute 6); every later one wins (price climbs).
    script = [(m, 1350.0 - 0.5 * m if m <= 6 else 1347.0 + 2.0 * (m - 6), 70.0) for m in range(1, 60)]
    out = _drive2(a, venue, script)
    entries = [(m, o) for m, o in out if ":E" in o["client_order_id"]]
    exits = [(m, o) for m, o in out if ":X" in o["client_order_id"]]
    assert a.closed_trades[0]["pnl"] < 0 and all(t["pnl"] > 0 for t in a.closed_trades[1:3])
    q1, q2, q3, q4 = (e[1]["qty"] for e in entries[:4])
    assert q2 >= 1.8 * q1 and q3 >= 1.8 * q4                      # two harder entries, then normal size
    assert q4 == pytest.approx(q1, rel=0.15)
    assert entries[1][0] - exits[0][0] == 5                         # cooldown halved: 10 -> 5
    assert entries[2][0] - exits[1][0] == 5
    assert entries[3][0] - exits[2][0] == 10                        # back to the agent's own cooldown


def test_three_losses_in_a_row_end_the_agents_day_and_the_next_day_starts_fresh():
    a, venue = _agent2("momentum", emotions=True, edit=lambda cfg: (
        cfg["emotions"]["loss_overtrade"].update(probability=0.0),
        cfg["emotions"]["profit_fear"].update(probability=0.0)))
    day0 = _drive2(a, venue, _losing_script(minutes=370))
    assert len(a.closed_trades) == 3 and a.loss_streak == 3
    assert a.halted_day == 0 and a.position is None
    last_order_min = max(m for m, _ in day0)
    assert last_order_min < 369, "no orders may follow the third loss on that day"
    day1 = _drive2(a, venue, _losing_script(days=1, minutes=8, day0=1))
    assert day1 and day1[0][0] == 1, "the agent trades again from the first minute of the next session"
    assert a.loss_streak == 1 and a.halted_day == 0          # the streak restarted from zero on day 1


def test_a_win_resets_the_loss_streak_and_a_new_day_does_too():
    a, _ = _agent2("momentum", emotions=True, edit=lambda cfg: (
        cfg["emotions"]["loss_overtrade"].update(probability=0.0)))
    a._day = 0
    a._feel({"pnl": -1.0}); a._feel({"pnl": -1.0})
    assert a.loss_streak == 2 and a.halted_day is None
    a._feel({"pnl": 2.0})
    assert a.loss_streak == 0
    a._feel({"pnl": -1.0}); a._feel({"pnl": -1.0})
    a.on_tick(_tick(1, day_count=1))
    assert a.loss_streak == 0                                    # new session, fresh start


def test_capitulation_dumps_any_shares_still_open_at_market():
    a, venue = _agent2("momentum", market=0.0)
    a.inventory, a.halted_day, a._day = 9, 0, 0
    orders = a.on_tick(_tick(5, rsi=50.0, day_count=0))
    assert len(orders) == 1 and orders[0]["action"] == "SELL"
    assert orders[0]["type"] == "MARKET" and orders[0]["qty"] == 9


def test_profit_fear_closes_a_trade_early_at_market():
    def edit(cfg):
        cfg["emotions"]["profit_fear"]["probability"] = 1.0
        cfg["emotions"]["profit_fear"]["early_take_profit_pct"] = {"dist": "uniform", "low": 0.30, "high": 0.30}
    a, venue = _agent2("momentum", emotions=True, hold=500, edit=edit)
    a.personality = replace(a.personality, exit_preference=(1.0, 0.0, 0.0, 0.0))
    script = [(1, 1350.0, 70.0)] + [(m, 1350.0 + 0.5 * m, 70.0) for m in range(2, 40)]   # +0.3% within a few minutes
    out = _drive2(a, venue, script)
    assert a.closed_trades and a.closed_trades[0]["reason"] == "fear_profit"
    assert a.closed_trades[0]["pnl"] > 0
    first_exit = next(o for _, o in out if ":X" in o["client_order_id"])
    assert first_exit["type"] == "MARKET"


def test_same_seed_gives_identical_orders_with_limit_orders_and_emotions_on():
    script = [(m, 1350.0 + (m % 7), 30.0 + (m * 13) % 40) for m in range(1, 370)]
    runs = []
    for _ in range(2):
        a, venue = _agent2("mean_reverter", market=0.5, emotions=True, hold=None)
        runs.append(_drive2(a, venue, script))
    assert runs[0] == runs[1] and len(runs[0]) > 3
