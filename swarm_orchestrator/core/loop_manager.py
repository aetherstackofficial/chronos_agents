import asyncio
import logging
import random
from typing import Iterable, List

from core.engine_client import EngineClient

logger = logging.getLogger(__name__)

class SwarmOrchestrator:
    NEWS_WINDOW_TICKS = 60
    QUIET_AWAKE_RATIO = 0.30
    NEWS_AWAKE_RATIO = 1.0

    def __init__(self, client: EngineClient, seed: int | None = None) -> None:
        self.client = client
        self.agents: List = []
        self.running = False
        self._rng = random.Random(seed)
        self._tick_count = 0
        self._last_news_tick = -self.NEWS_WINDOW_TICKS

    def load_agents(self, agents_list: Iterable) -> None:
        self.agents = list(agents_list)

    @property
    def news_active(self) -> bool:
        return (self._tick_count - self._last_news_tick) <= self.NEWS_WINDOW_TICKS

    async def run(self) -> None:
        self.running = True
        logger.info("Orchestrator online: %d agents, push-driven.", len(self.agents))
        
        async for topic, payload in self.client.iter_messages():
            if not self.running:
                break
            if topic == "tick":
                await self._on_tick(payload)
            elif topic == "event":
                self._on_event(payload)

    async def _on_tick(self, tick: dict) -> None:
        self._tick_count += 1
        awake_ratio = self.NEWS_AWAKE_RATIO if self.news_active else self.QUIET_AWAKE_RATIO
        
        batches = []
        for agent in self.agents:
            agent.is_asleep = self._rng.random() > awake_ratio
            if not agent.is_asleep and (orders := agent.on_tick(tick)):
                batches.append((agent, orders))

        if not batches:
            return

        for result in await asyncio.gather(*(self._submit_batch(a, o) for a, o in batches), return_exceptions=True):
            if isinstance(result, Exception):
                logger.error("Unexpected submit failure: %r", result)

    async def _submit_batch(self, agent, orders: List[dict]) -> None:
        for order in orders:
            msg = {"msg": "CANCEL_ALL", "agent_id": agent.agent_id} if order.get("cancel_all") else {"msg": "ORDER", "agent_id": agent.agent_id, **order}
            try:
                ack = await self.client.send_order(msg)
            except TimeoutError as exc:
                logger.warning("%s ack timeout (transient, skipped): %s", agent.agent_id, exc)
                continue
            agent.on_ack(ack)

    def _on_event(self, event: dict) -> None:
        if (kind := event.get("kind")) == "news":
            self._last_news_tick = self._tick_count
            logger.info("News event (score=%s) — full participation for %d ticks.", event.get("score"), self.NEWS_WINDOW_TICKS)
        elif kind == "liquidation":
            self._mirror_liquidation(event.get("agent_id"))

    def _mirror_liquidation(self, agent_id: str | None) -> None:
        for agent in self.agents:
            if agent.agent_id == agent_id and hasattr(agent, "on_liquidation"):
                agent.on_liquidation()
                logger.info("Mirrored engine liquidation reset for %s.", agent_id)
                break

    async def shutdown(self) -> None:
        self.running = False
        results = await asyncio.gather(*(self.client.send_order({"msg": "CANCEL_ALL", "agent_id": a.agent_id}) for a in self.agents), return_exceptions=True)
        
        if failed := sum(1 for r in results if isinstance(r, Exception)):
            logger.warning("%d/%d shutdown CANCEL_ALLs got no ack.", failed, len(self.agents))
            
        self.client.close()
        logger.info("Orchestrator shut down cleanly.")