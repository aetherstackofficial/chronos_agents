"""Push-driven swarm orchestrator (replaces the Phase-1 serial pull loop).

Consumes the market-data plane via `EngineClient.iter_messages()` and reacts:

* topic ``tick``  — dynamic participation, agent decisions, CONCURRENT order
  submission over the pooled DEALER gateway (asyncio.gather), ack routing.
* topic ``event`` — `news` refreshes the participation window; `liquidation`
  mirrors the engine's retail ledger reset (engine is the ONLY liquidation
  authority — no local bankruptcy logic, PROTOCOL.md §8.6).
"""

import asyncio
import logging
import random
from typing import Iterable, List, Tuple

from core.engine_client import EngineClient

logger = logging.getLogger(__name__)


class SwarmOrchestrator:
    """Drives a set of live agents off the engine's push feed."""

    NEWS_WINDOW_TICKS = 60   # a news event keeps the swarm fully awake this long
    QUIET_AWAKE_RATIO = 0.30
    NEWS_AWAKE_RATIO = 1.0

    def __init__(self, client: EngineClient, seed: "int | None" = None) -> None:
        """Args:
            client: connected EngineClient (owned by this orchestrator).
            seed: optional seed for the participation RNG (sanctioned
                behavioural randomness, PROTOCOL.md §8).
        """
        self.client = client
        self.agents: List = []
        self.running = False
        self._rng = random.Random(seed)
        self._tick_count = 0
        self._last_news_tick = -self.NEWS_WINDOW_TICKS  # no news seen yet

    def load_agents(self, agents_list: Iterable) -> None:
        """Register the live agents this orchestrator drives."""
        self.agents = list(agents_list)

    @property
    def news_active(self) -> bool:
        """True while a news event was seen within the last 60 ticks."""
        return (self._tick_count - self._last_news_tick) <= self.NEWS_WINDOW_TICKS

    async def run(self) -> None:
        """Consume the push feed until `shutdown()` flips `running`."""
        self.running = True
        logger.info("Orchestrator online: %d agents, push-driven.",
                    len(self.agents))
        async for topic, payload in self.client.iter_messages():
            if not self.running:
                break
            if topic == "tick":
                await self._on_tick(payload)
            elif topic == "event":
                self._on_event(payload)
            # topic "trade": individual prints are not needed by swarm agents.

    # ------------------------------------------------------------------ tick

    async def _on_tick(self, tick: dict) -> None:
        self._tick_count += 1

        # (1) Dynamic participation: 30% awake when quiet, 100% on news (§8).
        awake_ratio = self.NEWS_AWAKE_RATIO if self.news_active else self.QUIET_AWAKE_RATIO
        for agent in self.agents:
            agent.is_asleep = self._rng.random() > awake_ratio

        # (2) Collect every awake agent's orders, then submit ALL agents'
        # batches concurrently over the pooled gateway. Order is preserved
        # per agent (its batch is sequential); agents run in parallel.
        batches: List[Tuple[object, List[dict]]] = []
        for agent in self.agents:
            if agent.is_asleep:
                continue
            orders = agent.on_tick(tick)
            if orders:
                batches.append((agent, orders))
        if not batches:
            return

        results = await asyncio.gather(
            *(self._submit_batch(agent, orders) for agent, orders in batches),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                logger.error("Unexpected submit failure: %r", result)

    async def _submit_batch(self, agent, orders: List[dict]) -> None:
        """Send one agent's orders sequentially; route each ack back to it."""
        for order in orders:
            if order.get("cancel_all"):
                msg = {"msg": "CANCEL_ALL", "agent_id": agent.agent_id}
            else:
                msg = {"msg": "ORDER", "agent_id": agent.agent_id, **order}
            try:
                ack = await self.client.send_order(msg)
            except TimeoutError as exc:
                logger.warning("%s ack timeout (transient, skipped): %s",
                               agent.agent_id, exc)
                continue
            agent.on_ack(ack)

    # ----------------------------------------------------------------- event

    def _on_event(self, event: dict) -> None:
        kind = event.get("kind")
        if kind == "news":
            self._last_news_tick = self._tick_count
            logger.info("News event (score=%s) — full participation "
                        "for %d ticks.", event.get("score"), self.NEWS_WINDOW_TICKS)
        elif kind == "liquidation":
            self._mirror_liquidation(event.get("agent_id"))

    def _mirror_liquidation(self, agent_id: "str | None") -> None:
        """Mirror the engine's retail liquidation reset (engine-authoritative)."""
        for agent in self.agents:
            if agent.agent_id == agent_id and hasattr(agent, "on_liquidation"):
                agent.on_liquidation()
                logger.info("Mirrored engine liquidation reset for %s.",
                            agent_id)
                return

    # -------------------------------------------------------------- shutdown

    async def shutdown(self) -> None:
        """Cancel every agent's resting orders, then close the sockets."""
        self.running = False
        results = await asyncio.gather(
            *(self.client.send_order({"msg": "CANCEL_ALL", "agent_id": a.agent_id})
              for a in self.agents),
            return_exceptions=True,
        )
        failed = sum(1 for r in results if isinstance(r, Exception))
        if failed:
            logger.warning("%d/%d shutdown CANCEL_ALLs got no ack.",
                           failed, len(self.agents))
        self.client.close()
        logger.info("Orchestrator shut down cleanly.")
