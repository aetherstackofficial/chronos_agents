"""The swarm stays silent while a replay is running (core/loop_manager.py)."""
import asyncio

from core.loop_manager import SwarmOrchestrator


class FakeAgent:
    agent_id = "MM_1"
    is_asleep = False

    def __init__(self):
        self.calls = 0

    def on_tick(self, tick):
        self.calls += 1
        return [{"action": "BUY", "qty": 1}]

    def on_ack(self, ack):
        pass


class FakeClient:
    def __init__(self):
        self.sent = []

    async def send_order(self, msg):
        self.sent.append(msg)
        return {"status": "FILLED"}


def _tick(mode):
    agent, client = FakeAgent(), FakeClient()
    orch = SwarmOrchestrator(client, seed=1)
    orch.QUIET_AWAKE_RATIO = 1.0                      # everyone awake, so only the mode decides
    orch.load_agents([agent])
    asyncio.run(orch._on_tick({"mode": mode}))
    return agent, client


def test_agents_do_nothing_during_replay():
    agent, client = _tick("REPLAY")
    assert agent.calls == 0 and client.sent == []


def test_agents_trade_in_live_and_reactive_modes():
    for mode in ("LIVE", "REACTIVE"):
        agent, client = _tick(mode)
        assert agent.calls == 1 and len(client.sent) == 1
