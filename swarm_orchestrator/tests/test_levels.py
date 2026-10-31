"""Phase 3: price levels (agents/levels.py) and retail orders placed at them."""
import copy
import logging
import os
import random
import sys
from collections import Counter
from dataclasses import replace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

from agents import heuristic_retail as hr  # noqa: E402
from agents import levels as lv  # noqa: E402

CFG = hr.CONFIG


class StubContext:
    """The part of the MarketContext interface the levels read."""
    def __init__(self, ready=True, prev_day=None, day_high=None, day_low=None, highs=(), lows=()):
        self.ready, self.prev_day = ready, prev_day
        self.day_high, self.day_low = day_high, day_low
        self._h, self._l = list(highs), list(lows)

    def swing_highs(self):
        return list(self._h)

    def swing_lows(self):
        return list(self._l)


# --- round numbers ------------------------------------------------------------------

@pytest.mark.parametrize("price,tick,unit", [
    (1354.0, 0.1, 1.0), (190.0, 0.05, 0.1), (14_000.0, 0.5, 10.0),
    (3.0, 0.001, 0.001), (55.0, 0.01, 0.01), (1354.0, 5.0, 5.0)])
def test_level_unit_is_a_thousandth_of_the_decade_and_never_below_a_tick(price, tick, unit):
    assert lv.level_unit(price, tick) == pytest.approx(unit)


def test_round_levels_cover_the_window_with_tiered_strengths():
    levels = lv.round_levels(1354.0, 0.1, 40, CFG)
    prices = [l.price for l in levels]
    assert prices == sorted(prices) and prices[0] >= 1354 * (1 - 0.004) - 1e-9 and prices[-1] <= 1354 * 1.004 + 1e-9
    assert 1350.0 in prices and 1355.0 in prices and 1349.0 in prices
    s = {l.price: l.strength for l in levels}
    st = CFG["levels"]["round_numbers"]["strengths"]
    assert s[1349.0] == st[0]                                    # plain whole rupee
    assert s[1355.0] == st[1]                                    # multiple of 5
    assert s[1350.0] == st[3]                                    # multiple of 50
    big = {l.price: l.strength for l in lv.round_levels(1354.0, 0.1, 400, CFG)}
    assert big[1400.0] == st[4] and big[1360.0] == st[2]         # 100 and 10 tiers


def test_round_levels_follow_the_price_scale():
    cheap = [l.price for l in lv.round_levels(190.0, 0.05, 100, CFG)]
    assert all(abs(p * 10 - round(p * 10)) < 1e-6 for p in cheap)       # multiples of 0.1
    dear = [l.price for l in lv.round_levels(14_000.0, 0.5, 100, CFG)]
    assert all(p % 10 == 0 for p in dear)                               # multiples of 10


def test_round_levels_never_flood():
    cfg = copy.deepcopy(CFG)
    cfg["levels"]["round_numbers"]["max_levels"] = 5
    assert lv.round_levels(1354.0, 0.1, 400, cfg) == []
    assert lv.round_levels(0.0, 0.1, 40, CFG) == []


# --- history levels -----------------------------------------------------------------

def test_context_levels_read_the_promised_attributes():
    ctx = StubContext(prev_day={"o": 1, "h": 1360.0, "l": 1340.0, "c": 1352.0}, day_high=1356.0,
                      day_low=1348.0, highs=[1357.0, 1358.0], lows=[1346.0])
    kinds = Counter(l.kind for l in lv.context_levels(ctx, CFG))
    assert kinds == Counter({"prev_day_high": 1, "prev_day_low": 1, "prev_day_close": 1,
                             "day_high": 1, "day_low": 1, "swing": 3})


def test_context_levels_are_empty_without_a_ready_context():
    assert lv.context_levels(None, CFG) == []
    assert lv.context_levels(StubContext(ready=False, day_high=1356.0), CFG) == []


def test_context_levels_ignore_missing_or_bad_values():
    ctx = StubContext(prev_day={"h": None, "l": float("nan"), "c": -1}, day_high=None)
    assert lv.context_levels(ctx, CFG) == []
    assert lv.context_levels(object(), CFG) == []                  # an object with none of the attributes


def test_build_levels_adds_strengths_of_levels_at_the_same_price_and_respects_the_window():
    ctx = StubContext(prev_day={"h": 1350.0, "l": 1200.0, "c": 1300.0})     # 1200 / 1300 are out of the window
    levels = {l.price: l for l in lv.build_levels(1354.0, 0.1, ctx, CFG)}
    st = CFG["levels"]
    assert levels[1350.0].strength == pytest.approx(st["round_numbers"]["strengths"][3] + st["context_strengths"]["prev_day_high"])
    assert "round" in levels[1350.0].kind and "prev_day_high" in levels[1350.0].kind
    assert 1200.0 not in levels and 1300.0 not in levels
    assert all(1354 * 0.996 - 1e-9 <= p <= 1354 * 1.004 + 1e-9 for p in levels)


# --- choosing a level ---------------------------------------------------------------

def _pick_many(side, n=4000, seed=1, bid=1349.9, ask=1350.1, levels=None, cfg=CFG):
    levels = levels if levels is not None else lv.build_levels(1350.0, 0.1, None, cfg)
    rng = random.Random(seed)
    return [lv.pick_level(side, bid, ask, 1350.0, 0.1, levels, rng, cfg) for _ in range(n)]


def test_a_buy_level_is_on_its_own_side_and_never_crosses_the_ask():
    picks = [p for p in _pick_many(+1) if p is not None]
    assert picks and max(picks) <= 1350.0 + 1e-9            # ask 1350.1 minus a tick
    assert min(picks) >= 1349.9 * (1 - 0.004) - 1e-9
    assert all(abs(p - round(p)) < 1e-6 for p in picks)     # every level is a whole rupee at this price


def test_a_sell_level_is_on_its_own_side_and_never_crosses_the_bid():
    picks = [p for p in _pick_many(-1) if p is not None]
    assert picks and min(picks) >= 1350.0 - 1e-9            # bid 1349.9 plus a tick
    assert max(picks) <= 1350.1 * 1.004 + 1e-9


def test_levels_near_the_touch_and_stronger_levels_are_picked_more_often():
    picks = Counter(_pick_many(+1, n=20000))
    assert picks[1350.0] > picks[1346.0]                    # near and strong beats far and weak
    two = Counter(_pick_many(+1, n=20000, levels=[lv.Level(1347.0, 1.0, "r"), lv.Level(1346.9, 3.0, "r")]))
    assert two[1346.9] > two[1347.0] * 2                     # 3x strength at almost the same distance


def test_no_level_in_reach_means_no_anchor():
    far = [lv.Level(1300.0, 1.0, "round"), lv.Level(1400.0, 1.0, "round")]
    assert set(_pick_many(+1, n=50, levels=far)) == {None}
    assert set(_pick_many(-1, n=50, levels=far)) == {None}
    assert set(_pick_many(+1, n=50, levels=[])) == {None}


def test_picking_works_without_quotes():
    levels = lv.build_levels(1350.0, 0.1, None, CFG)
    px = lv.pick_level(+1, None, None, 1350.0, 0.1, levels, random.Random(2), CFG)
    assert px is not None and px <= 1349.9 + 1e-9


def test_picking_is_deterministic_for_a_seed():
    assert _pick_many(+1, n=200, seed=5) == _pick_many(+1, n=200, seed=5)
    assert _pick_many(+1, n=200, seed=5) != _pick_many(+1, n=200, seed=6)


# --- exits at levels ----------------------------------------------------------------

def test_a_long_target_goes_just_short_of_a_level_and_the_stop_just_beyond_one():
    # entry 1350.0; drawn target +1.0% = 1363.5, drawn stop -1.0% = 1336.5
    tp, stop = lv.anchor_exit_prices(+1, 1350.0, 1.0, 1.0, 0.1, None, CFG)
    assert tp > 1350.0 and abs((tp + 0.1) - round(tp + 0.1)) < 1e-6              # one tick short of a whole rupee
    assert stop < 1350.0 and abs((stop + 0.2) - round(stop + 0.2)) < 1e-6        # two ticks beyond (below) a whole rupee


def test_a_short_mirrors_a_long():
    tp, stop = lv.anchor_exit_prices(-1, 1350.0, 1.0, 1.0, 0.1, None, CFG)
    assert tp < 1350.0 < stop
    assert abs((tp - 0.1) - round(tp - 0.1)) < 1e-6                              # one tick short (above) a whole rupee
    assert abs((stop - 0.2) - round(stop - 0.2)) < 1e-6                          # two ticks beyond (above) a whole rupee


def test_no_level_close_enough_keeps_the_percentage_rule():
    cfg = copy.deepcopy(CFG)
    cfg["levels"]["round_numbers"]["max_levels"] = 1            # no round levels at all
    assert lv.anchor_exit_prices(+1, 1350.0, 1.0, 1.0, 0.1, None, cfg) == (0.0, 0.0)


def test_history_levels_can_pull_a_target_onto_a_previous_day_high():
    ctx = StubContext(prev_day={"h": 1367.3, "l": 1300.0, "c": 1340.0})
    cfg = copy.deepcopy(CFG)
    cfg["levels"]["round_numbers"]["max_levels"] = 1
    tp, _ = lv.anchor_exit_prices(+1, 1350.0, 1.2, 1.0, 0.1, ctx, cfg)   # drawn target 1366.2, prev-day high 1367.3
    assert tp == pytest.approx(1367.2)


def test_exit_reason_uses_level_prices_when_the_plan_has_them():
    plan = hr.ExitPlan("target_stop", 750, take_profit_pct=1.0, stop_loss_pct=1.0,
                       take_profit_price=1360.0, stop_price=1345.0)
    pos = hr.Position(side=1, qty=3, entry_price=1350.0, entry_t=0, plan=plan, best_price=1350.0)
    assert hr.exit_reason(pos, 1359.9, 1, 0.0, 0.2) is None          # +0.7%: the percentage rule would also say no
    assert hr.exit_reason(pos, 1360.0, 1, 0.0, 0.2) == "take_profit"
    assert hr.exit_reason(pos, 1345.1, 1, 0.0, 0.2) is None
    assert hr.exit_reason(pos, 1345.0, 1, 0.0, 0.2) == "stop_loss"
    short = hr.Position(side=-1, qty=3, entry_price=1350.0, entry_t=0, plan=replace(
        plan, take_profit_price=1340.0, stop_price=1356.0), best_price=1350.0)
    assert hr.exit_reason(short, 1340.0, 1, 0.0, 0.2) == "take_profit"
    assert hr.exit_reason(short, 1356.0, 1, 0.0, 0.2) == "stop_loss"


# --- the trait: only some agents set exits at levels -----------------------------------

def test_about_half_the_agents_have_the_level_exit_trait():
    flags = [hr.sample_personality(f"RETAIL_{i}", 4).level_exits for i in range(3000)]
    assert sum(flags) / len(flags) == pytest.approx(CFG["levels"]["level_exit_share"], abs=0.03)


def test_adding_the_trait_did_not_move_any_other_draw():
    a = hr.sample_personality("RETAIL_9", 4)
    cfg = copy.deepcopy(CFG)
    cfg["levels"]["level_exit_share"] = 0.0
    b = hr.sample_personality("RETAIL_9", 4, cfg)
    assert replace(a, level_exits=False) == b


# --- retail agents placing orders at levels (loads the PPO models, ~10 s) -------------------

la = pytest.importorskip("agents.live_agents")


def _tick(minute=1, **kw):
    t = {"seq": 1, "last_price": 1350.0, "mid_price": 1350.0, "best_bid": 1349.9,
         "best_ask": 1350.1, "tick_size": 0.1, "vwap": 1350.0, "volume_ma": 1000.0,
         "step_volume": 1000, "volatility": 0.001, "rsi": 70.0, "ofi": 0.0,
         "day_count": 0, "market_minute": minute, "symbol": "RELIANCE"}
    t.update(kw)
    return t


def _agent(anchor_prob=1.0, level_exits=False, exit_pref=(0.0, 1.0, 0.0, 0.0), context=None, seed=1):
    cfg = copy.deepcopy(CFG)
    cfg["personality"]["base_trade_prob"] = 1.0
    cfg["orders"]["type_mix"] = {"market": 0.0, "limit": 1.0}
    cfg["levels"]["anchor_prob"] = anchor_prob
    cfg["emotions"]["loss_overtrade"]["probability"] = 0.0
    cfg["emotions"]["profit_fear"]["probability"] = 0.0
    cfg["emotions"]["capitulation"]["consecutive_losses"] = 99
    a = la.LiveRetailAgent("RETAIL_L", seed=seed, cfg=cfg)
    a.personality = replace(hr.sample_personality("RETAIL_L", 1), kind="momentum", rsi_center=50.0,
                            conviction_threshold=0.2, cooldown_ticks=3, risk_appetite=0.05,
                            start_cash=1_000_000.0, exit_preference=exit_pref, level_exits=level_exits)
    a.cash = a.personality.start_cash
    a.context = context
    return a


def _first_limit(agent, side_rsi=70.0):
    """The agent's first order (the entry dice may skip a minute or two)."""
    for minute in range(1, 8):
        orders = agent.on_tick(_tick(minute, rsi=side_rsi))
        if orders:
            assert orders[0]["type"] == "LIMIT"
            return orders[0]
    raise AssertionError("the agent never sent an order")


def test_anchored_buy_orders_sit_at_whole_rupee_levels_inside_the_reach():
    prices = []
    for seed in range(60):
        a = _agent(anchor_prob=1.0, seed=seed)
        prices.append(_first_limit(a)["price"])
    assert all(abs(p - round(p)) < 1e-6 for p in prices)               # whole rupees
    assert all(1344.5 <= p <= 1350.0 + 1e-9 for p in prices)
    assert len(set(prices)) > 3                                         # spread over several levels


def test_unanchored_orders_use_the_old_offset_rule():
    prices = [_first_limit(_agent(anchor_prob=0.0, seed=s))["price"] for s in range(60)]
    assert all(p <= 1349.9 + 1e-9 for p in prices)                      # at or behind the best bid
    assert any(abs(p - round(p)) > 1e-6 for p in prices)                # not all on whole rupees


def test_share_of_anchored_orders_matches_the_configured_probability():
    # The old offset rule puts a buy at 1349.9 - k*0.1 (k = 0,1,2,3,5), never on a whole rupee,
    # so at this price every whole-rupee order is an anchored one.
    n = 800
    whole = sum(abs(p - round(p)) < 1e-6 for p in
                (_first_limit(_agent(anchor_prob=0.3, seed=s))["price"] for s in range(n)))
    assert whole / n == pytest.approx(0.3, abs=0.05)


def test_a_ready_context_adds_history_levels_to_the_choices():
    ctx = StubContext(prev_day={"h": 1346.5, "l": 1300.0, "c": 1340.0})
    seen = set()
    for s in range(300):
        seen.add(_first_limit(_agent(anchor_prob=1.0, context=ctx, seed=s))["price"])
    assert 1346.5 in seen                                               # the previous-day high, not a round number


def test_a_context_that_is_not_ready_changes_nothing_and_trading_goes_on():
    cold = StubContext(ready=False, prev_day={"h": 1346.5})
    a = _agent(anchor_prob=1.0, context=cold)
    assert _first_limit(a)["price"] <= 1350.0 + 1e-9


def test_anchoring_is_deterministic_for_a_seed():
    one = [_first_limit(_agent(anchor_prob=0.5, seed=s))["price"] for s in range(40)]
    two = [_first_limit(_agent(anchor_prob=0.5, seed=s))["price"] for s in range(40)]
    assert one == two


class Venue:
    def __init__(self, cash):
        self.cash, self.pos = cash, 0

    def market_fill(self, order, price):
        qty, sign = order["qty"], (1 if order["action"] == "BUY" else -1)
        self.cash -= sign * qty * price
        self.pos += sign * qty
        return {"msg": "ACK", "status": "FILLED", "order_id": 1, "executed_qty": qty, "average_price": price,
                "resting_qty": 0, "cash": self.cash, "pos": self.pos, "client_order_id": order["client_order_id"]}


def _enter(agent):
    cfg_mix = agent._cfg["orders"]["type_mix"]
    cfg_mix.update(market=1.0, limit=0.0)                               # fill the entry at once, at market
    venue = Venue(agent.cash)
    order = agent.on_tick(_tick(1, rsi=70.0))[0]
    agent.on_ack(venue.market_fill(order, 1350.0))
    return agent.position


def test_agents_with_the_trait_get_level_prices_for_target_and_stop():
    pos = _enter(_agent(level_exits=True))
    plan = pos.plan
    assert plan.style == "target_stop"
    assert plan.take_profit_price > 1350.0 and 0.0 < plan.stop_price < 1350.0
    assert abs((plan.take_profit_price + 0.1) - round(plan.take_profit_price + 0.1)) < 1e-6   # one tick short of a level
    assert abs((plan.stop_price + 0.2) - round(plan.stop_price + 0.2)) < 1e-6                 # two ticks beyond a level


def test_agents_without_the_trait_keep_percentage_exits():
    plan = _enter(_agent(level_exits=False)).plan
    assert plan.take_profit_price == 0.0 and plan.stop_price == 0.0


def test_the_trait_does_not_touch_other_exit_styles():
    plan = _enter(_agent(level_exits=True, exit_pref=(1.0, 0.0, 0.0, 0.0))).plan
    assert plan.style == "time_based" and plan.take_profit_price == 0.0 and plan.stop_price == 0.0
