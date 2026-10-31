"""The portfolio-return input of the PPO agents is measured per session, like a training episode.

Found 2026-10-31: measured from the start of the RUN, a whale that had built its
position over days showed +143% on average, its input sat on the +10 clip 93% of
the time, and its policy froze at "maximum long" and stopped trading once full.
"""
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

la = pytest.importorskip("agents.live_agents")


def _tick(day, price):
    return {"day_count": day, "market_minute": 10, "last_price": price}


def _agent(cash=1_000_000.0, inventory=0):
    a = la.LiveMarketMaker("MM_PR", rng=__import__("random").Random(1))
    a.cash, a.inventory = cash, inventory
    return a


def test_return_is_measured_from_the_first_look_within_a_session():
    a = _agent()
    assert a._port_return(_tick(0, 100.0)) == 0.0
    a.cash, a.inventory = 900_000.0, 1000                  # buys 1000 shares at 100
    assert a._port_return(_tick(0, 100.0)) == pytest.approx(0.0)
    assert a._port_return(_tick(0, 110.0)) == pytest.approx(0.01)       # +10 on 1000 shares of 1,000,000


def test_each_new_session_starts_again_from_zero():
    a = _agent()
    a._port_return(_tick(0, 100.0))
    a.cash, a.inventory = 900_000.0, 1000
    assert a._port_return(_tick(0, 110.0)) == pytest.approx(0.01)
    assert a._port_return(_tick(1, 110.0)) == 0.0          # new day: equity 1,010,000 is the new start
    assert a._port_return(_tick(1, 121.0)) == pytest.approx(11_000 / 1_010_000)


def test_a_position_built_over_many_days_does_not_blow_the_input_up():
    a = _agent(cash=5_000_000.0)
    price = 1354.0
    for day in range(30):
        a._port_return(_tick(day, price))                  # first look of the session
        a.cash -= 10_000 * price                           # 10,000 more shares a day, financed on margin
        a.inventory += 10_000
        price *= 1.005                                     # and the market rises 0.5% a day, as the whales' own buying did
        assert abs(a._port_return(_tick(day, price))) < 1.0
    # Measured from the start of the run, the same book reads several hundred percent.
    since_run_start = (a.cash + a.inventory * price - 5_000_000.0) / 5_000_000.0
    assert since_run_start > 5.0


def test_no_start_is_latched_before_the_engine_has_told_the_agent_its_cash():
    a = _agent(cash=0.0)
    assert a._port_return(_tick(0, 100.0)) == 0.0
    assert a.start_equity is None
    a.cash = 1_000_000.0
    assert a._port_return(_tick(0, 100.0)) == 0.0
    assert a.start_equity == 1_000_000.0
