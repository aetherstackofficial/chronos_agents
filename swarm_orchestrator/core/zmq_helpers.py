import asyncio
import json
import logging
import os

import zmq
import zmq.asyncio

logger = logging.getLogger(__name__)
ACK_TIMEOUT_S = 5.0

def env_host() -> str: return os.getenv("ZMQ_HOST", "127.0.0.1")
def env_order_port() -> int: return int(os.getenv("ZMQ_ORDER_PORT", "5555"))
def env_data_port() -> int: return int(os.getenv("ZMQ_DATA_PORT", "5556"))

_context: zmq.asyncio.Context | None = None

def get_context() -> zmq.asyncio.Context:
    global _context
    if _context is None:
        _context = zmq.asyncio.Context.instance()
    return _context

def make_dealer(identity: str, host: str, port: int) -> zmq.asyncio.Socket:
    sock = get_context().socket(zmq.DEALER)
    sock.setsockopt_string(zmq.IDENTITY, identity)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{host}:{port}")
    return sock

def make_sub(host: str, port: int, topics: list[str]) -> zmq.asyncio.Socket:
    sock = get_context().socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{host}:{port}")
    for topic in topics:
        sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    return sock

class AsyncOrderGateway:
    def __init__(self, identity_prefix: str, host: str | None = None, port: int | None = None, pool_size: int = 4, timeout: float = ACK_TIMEOUT_S) -> None:
        if pool_size < 1: 
            raise ValueError("pool_size must be >= 1")
            
        self._host = host or env_host()
        self._port = port or env_order_port()
        self._prefix, self._timeout, self._next, self._closed = identity_prefix, timeout, 0, False
        
        self._sockets = [make_dealer(f"{identity_prefix}-{i}", self._host, self._port) for i in range(pool_size)]
        self._locks = [asyncio.Lock() for _ in range(pool_size)]
        
        logger.info("Order gateway '%s' up: %d DEALER sockets -> tcp://%s:%d", identity_prefix, pool_size, self._host, self._port)

    async def request(self, msg: dict) -> dict:
        if self._closed: 
            raise RuntimeError("AsyncOrderGateway is closed")

        idx = self._next
        self._next = (self._next + 1) % len(self._sockets)

        async with self._locks[idx]:
            sock = self._sockets[idx]
            
            # Manually encode to bypass pyzmq's inaccurate send_json type hint
            await sock.send(json.dumps(msg).encode("utf-8"))

            poller = zmq.asyncio.Poller()
            poller.register(sock, zmq.POLLIN)
            
            if not await poller.poll(timeout=int(self._timeout * 1000)):
                self._recycle(idx)
                raise TimeoutError(f"No ack from engine ROUTER at tcp://{self._host}:{self._port} within {self._timeout:.0f}s (identity {self._prefix}-{idx}).")
                
            return json.loads(await sock.recv())

    def _recycle(self, idx: int) -> None:
        self._sockets[idx].close(linger=0)
        self._sockets[idx] = make_dealer(f"{self._prefix}-{idx}", self._host, self._port)
        logger.warning("Recycled DEALER socket %s-%d after timeout", self._prefix, idx)

    def close(self) -> None:
        if self._closed: return
        self._closed = True
        for sock in self._sockets:
            sock.close(linger=0)
            