"""Deployed-swarm determinism: tick clock, seeded per-agent RNGs, fixed submit order."""
import asyncio

import pytest

la = pytest.importorskip("agents.live_agents")
from agents.swarm_factory import TickClock, agent_rng, build_swarm  # noqa: E402
from core.loop_manager import SwarmOrchestrator  # noqa: E402

COUNTS = {"mm": 3, "whale": 2, "retail": 5, "redteam": 2}


def _params(agents):
    return [(a.agent_id, getattr(a, "max_position", None), a.cash,
             getattr(a, "_cooldown_s", None)) for a in agents]


def test_tick_clock_counts_physics_ticks_and_stands_still_when_paused():
    c = TickClock()
    c.observe({"day_count": 1, "market_minute": 7})
    assert c() == 7.0
    c.observe({"day_count": 3, "market_minute": 2})
    assert c() == 2 * 375 + 2
    c.observe({"day_count": 3, "market_minute": 2})      # paused: same minute again
    assert c() == 752.0


def test_same_seed_same_swarm_and_order_independent():
    a = build_swarm(COUNTS, 11, TickClock())
    b = build_swarm(COUNTS, 11, TickClock())
    assert _params(a) == _params(b)
    assert _params(build_swarm(COUNTS, 12, TickClock())) != _params(a)
    # An agent's draws depend on (seed, agent_id) only, not on who was built first.
    assert agent_rng(11, "RETAIL_3").random() == agent_rng(11, "RETAIL_3").random()
    only_retail = build_swarm({"retail": 5}, 11, TickClock())
    assert _params(only_retail) == [p for p in _params(a) if p[0].startswith("RETAIL")]


class _RecordingClient:
    def __init__(self):
        self.order = []

    async def send_order(self, msg):
        await asyncio.sleep(0.001 * (hash(msg["agent_id"]) % 3))   # uneven latency
        self.order.append(msg["agent_id"])
        return {"msg": "ACK", "status": "FILLED", "pos": 0, "cash": 0.0}


class _Agent:
    def __init__(self, aid):
        self.agent_id, self.is_asleep = aid, False

    def on_tick(self, tick):
        return [{"action": "BUY", "type": "MARKET", "qty": 1}] * 2

    def on_ack(self, ack):
        pass


def test_orders_reach_the_engine_in_agent_order():
    client = _RecordingClient()
    orch = SwarmOrchestrator(client, seed=1, clock=TickClock())
    orch.NEWS_WINDOW_TICKS, orch._last_news_tick = 10 ** 9, 0     # everyone awake
    agents = [_Agent(f"A{i}") for i in range(8)]
    orch.load_agents(agents)
    asyncio.run(orch._on_tick({"day_count": 1, "market_minute": 1}))
    assert client.order == [a.agent_id for a in agents for _ in range(2)]


def test_harness_swarm_never_reads_wall_time(monkeypatch):
    from validation.sim_driver import Simulation
    sim = Simulation(3, 1350.0, symbol="RELIANCE")

    def boom():
        raise AssertionError("an agent read wall time")
    monkeypatch.setattr(la.time, "time", boom)
    for _ in range(60):
        sim.step()


def test_restored_agents_follow_the_live_clock():
    from validation.sim_driver import Simulation
    sim = Simulation(4, 1350.0, symbol="RELIANCE")
    for _ in range(5):
        sim.step()
    snap = sim.snapshot()
    for _ in range(5):
        sim.step()
    sim.restore(snap)
    sim.step()
    assert all(a._clock is sim.clock for a in sim.orch.agents)
    assert sim.clock() == 6.0


def test_unseeded_swarm_still_builds():
    agents = build_swarm({"retail": 2, "mm": 1}, None, TickClock())
    assert len(agents) == 3 and all(isinstance(a.cash, float) for a in agents)
