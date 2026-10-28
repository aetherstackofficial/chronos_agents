import json
import logging
from typing import AsyncIterator, Tuple

from core.zmq_helpers import AsyncOrderGateway, env_data_port, env_host, make_sub

logger = logging.getLogger(__name__)
DATA_TOPICS = ["tick", "trade", "event"]

class EngineClient:
    def __init__(self, identity_prefix: str, pool_size: int = 4, host: str | None = None, order_port: int | None = None, data_port: int | None = None) -> None:
        resolved_host = host if host is not None else env_host()
        self.gateway = AsyncOrderGateway(identity_prefix, host=resolved_host, port=order_port, pool_size=pool_size)
        
        resolved_data = data_port if data_port is not None else env_data_port()
        self._sub = make_sub(resolved_host, resolved_data, DATA_TOPICS)
        logger.info("EngineClient '%s' subscribed to %s on tcp://%s:%d", identity_prefix, DATA_TOPICS, resolved_host, resolved_data)

    async def send_order(self, order: dict) -> dict:
        return await self.gateway.request(order)

    async def control(self, msg: dict) -> dict:
        return await self.gateway.request(msg)

    async def fetch_state(self) -> dict:
        return await self.control({"msg": "FETCH_STATE"})

    async def iter_messages(self) -> AsyncIterator[Tuple[str, dict]]:
        while True:
            if len(frames := await self._sub.recv_multipart()) != 2:
                logger.warning("Dropping malformed PUB frame (%d parts)", len(frames))
                continue
            try:
                yield frames[0].decode("utf-8", errors="replace"), json.loads(frames[1])
            except (ValueError, UnicodeDecodeError):
                logger.warning("Dropping undecodable payload on topic %r", frames[0])

    def close(self) -> None:
        self.gateway.close()
        self._sub.close(linger=0)