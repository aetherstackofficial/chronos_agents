"""Phase 4 on the real live agents (loads the PPO models, ~10 s)."""
import logging
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

la = pytest.importorskip("agents.live_agents")
from agents.execution import skew_ticks  # noqa: E402


def _tick(**kw):
    t = {"seq": 1, "last_price": 1350.0, "mid_price": 1350.0, "best_bid": 1349.9,
         "best_ask": 1350.1, "tick_size": 0.1, "vwap": 1350.0, "volume_ma": 1000.0,
         "step_volume": 1000, "volatility": 0.001, "rsi": 50.0, "ofi": 0.0,
         "day_count": 1, "market_minute": 100, "symbol": "RELIANCE"}
    t.update(kw)
    return t


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_mm_quotes_shift_against_inventory_and_snap(monkeypatch):
    # Fixed policy output so only the skew differs between the two calls.
    monkeypatch.setattr(la.mm_model, "predict",
                        lambda obs, deterministic=True: (np.array([[0.0, 0.0, 0.0]]), None))
    mm = la.LiveMarketMaker("MM_T", clock=Clock(), rng=random.Random(1))
    flat = {o["action"]: o["price"] for o in mm.on_tick(_tick()) if "action" in o}
    # 90% long: skew -round(2.7) = -3 ticks, and the bid is still allowed.
    mm.last_trade, mm.inventory = 0.0, int(0.9 * mm.max_position)
    long_ = {o["action"]: o["price"] for o in mm.on_tick(_tick()) if "action" in o}
    assert skew_ticks(mm.inventory, mm.max_position) == -3
    for side in ("BUY", "SELL"):
        assert long_[side] == pytest.approx(flat[side] - 0.3, abs=1e-9)
        assert abs(long_[side] / 0.1 - round(long_[side] / 0.1)) < 1e-9


def test_whale_uses_injected_clock_and_rng_and_sends_limits(monkeypatch):
    monkeypatch.setattr(la.whale_model, "predict",
                        lambda obs, deterministic=True: (np.array([[0.5]]), None))
    a = la.LiveWhaleAgent("WHALE_T", clock=Clock(), rng=random.Random(3))
    b = la.LiveWhaleAgent("WHALE_T", clock=Clock(), rng=random.Random(3))
    assert a.max_capacity == b.max_capacity
    oa, ob = a.on_tick(_tick()), b.on_tick(_tick())
    assert oa == ob and oa[0] == {"cancel_all": True}        # first parent reads the ledger
    child = oa[-1]
    assert child["type"] == "LIMIT" and child["price"] == 1350.1 and child["tif_ticks"] == 1
    assert child["qty"] <= 10_000


def test_same_seed_same_event_hash_with_phase4_agents():
    from validation.sim_driver import Simulation

    def run():
        sim = Simulation(5, 1350.0, symbol="RELIANCE", hash_events=True)
        for _ in range(300):
            sim.step()
        return sim.event_hash(), [p.parent_id for a in sim.orch.agents
                                  if hasattr(a, "slicer") for p in a.slicer.log]
    h1, p1 = run()
    h2, p2 = run()
    assert h1 == h2 and p1 == p2
