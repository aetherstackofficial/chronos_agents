import asyncio
import logging
import random
from typing import Iterable, List

from core.engine_client import EngineClient
from core.market_context import MarketContext

logger = logging.getLogger(__name__)

class SwarmOrchestrator:
    NEWS_WINDOW_TICKS = 60
    QUIET_AWAKE_RATIO = 0.30
    NEWS_AWAKE_RATIO = 1.0

    def __init__(self, client: EngineClient, seed: int | None = None, clock=None) -> None:
        self.client = client
        self.agents: List = []
        self.running = False
        self._rng = random.Random(seed)
        # agents.swarm_factory.TickClock shared with the agents; set from each
        # tick before any agent acts.
        self.clock = clock
        # One shared market-history object (core/market_context.py). Fed once per
        # tick below, before any agent acts; agents read it as ``agent.context``.
        # It only reports what happened and never gates an agent.
        self.context = MarketContext()
        self._tick_count = 0
        self._last_news_tick = -self.NEWS_WINDOW_TICKS

    def load_agents(self, agents_list: Iterable) -> None:
        self.agents = list(agents_list)
        for agent in self.agents:
            agent.context = self.context

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
        # Ignores paused/repeated ticks itself, so it is safe to call every time.
        self.context.update(tick)
        if tick.get("paused"):
            # The market is closed: acting would only burn RNG draws and move
            # cooldowns by however many paused seconds happened to pass.
            return
        self._tick_count += 1
        if self.clock is not None:
            self.clock.observe(tick)
        awake_ratio = self.NEWS_AWAKE_RATIO if self.news_active else self.QUIET_AWAKE_RATIO
        
        batches = []
        for agent in self.agents:
            agent.is_asleep = self._rng.random() > awake_ratio
            if not agent.is_asleep and (orders := agent.on_tick(tick)):
                batches.append((agent, orders))

        # One agent at a time, in agent order: the engine then sees the same
        # order sequence every run (concurrent submits over the socket pool
        # arrive in whatever order the scheduler and network produce).
        for agent, orders in batches:
            try:
                await self._submit_batch(agent, orders)
            except Exception as exc:  # one agent's failure must not stop the tick
                logger.error("Unexpected submit failure for %s: %r", agent.agent_id, exc)

        # Lockstep (engine/lockstep.py): tell the engine this tick is done so
        # it may run the next one. Clients without it (the in-process
        # validation harness) are in lockstep by construction.
        if (tick_done := getattr(self.client, "tick_done", None)) is not None and "seq" in tick:
            try:
                await tick_done(tick["seq"])
            except TimeoutError as exc:
                logger.warning("TICK_DONE not acknowledged: %s", exc)

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