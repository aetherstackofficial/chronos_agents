"""Async client for the Chronos engine's two planes (PROTOCOL.md §1, §4, §5).

Order entry rides a pooled DEALER gateway (concurrent request/ack); market data
arrives on a SUB socket subscribed to `tick`, `trade` and `event`.
"""

import json
import logging
from typing import AsyncIterator, Tuple

from core.zmq_helpers import AsyncOrderGateway, env_data_port, env_host, make_sub

logger = logging.getLogger(__name__)

DATA_TOPICS = ["tick", "trade", "event"]


class EngineClient:
    """One process's connection to the engine: orders out, market data in."""

    def __init__(
        self,
        identity_prefix: str,
        pool_size: int = 4,
        host: "str | None" = None,
        order_port: "int | None" = None,
        data_port: "int | None" = None,
    ) -> None:
        """Build the pooled DEALER gateway and the SUB socket.

        Args:
            identity_prefix: unique DEALER identity prefix for this process
                (pool sockets get suffixes "-0".."-{pool_size-1}").
            pool_size: number of pooled DEALER sockets (concurrent in-flight acks).
            host / order_port / data_port: optional overrides; default to
                ZMQ_HOST / ZMQ_ORDER_PORT / ZMQ_DATA_PORT (PROTOCOL.md §2).
        """
        resolved_host = host if host is not None else env_host()
        self.gateway = AsyncOrderGateway(
            identity_prefix, host=resolved_host, port=order_port, pool_size=pool_size
        )
        resolved_data_port = data_port if data_port is not None else env_data_port()
        self._sub = make_sub(resolved_host, resolved_data_port, DATA_TOPICS)
        logger.info(
            "EngineClient '%s' subscribed to %s on tcp://%s:%d",
            identity_prefix, DATA_TOPICS, resolved_host, resolved_data_port,
        )

    async def send_order(self, order: dict) -> dict:
        """Send one trading message (§4.1) and return the engine's ack (§4.2).

        Raises TimeoutError (transient-safe) if no ack arrives within 5s.
        """
        return await self.gateway.request(order)

    async def control(self, msg: dict) -> dict:
        """Send a control message (§4.3) and return the ack."""
        return await self.gateway.request(msg)

    async def fetch_state(self) -> dict:
        """Poll-fallback: fetch the current tick payload via FETCH_STATE."""
        return await self.control({"msg": "FETCH_STATE"})

    async def iter_messages(self) -> AsyncIterator[Tuple[str, dict]]:
        """Yield (topic, payload) pairs from the market-data plane, forever.

        Malformed frames are logged and skipped — a bad publisher frame must
        never kill the swarm's consume loop.
        """
        while True:
            frames = await self._sub.recv_multipart()
            if len(frames) != 2:
                logger.warning("Dropping malformed PUB frame (%d parts)",
                               len(frames))
                continue
            topic_raw, payload_raw = frames
            try:
                payload = json.loads(payload_raw)
            except (ValueError, UnicodeDecodeError):
                logger.warning("Dropping undecodable payload on topic %r",
                               topic_raw)
                continue
            yield topic_raw.decode("utf-8", errors="replace"), payload

    def close(self) -> None:
        """Close both planes' sockets. Idempotent."""
        self.gateway.close()
        self._sub.close(linger=0)
