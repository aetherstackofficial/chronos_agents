"""Shared ZeroMQ transport utilities for the two-plane exchange model.

Order-entry plane:  DEALER -> engine ROUTER :5555 (reliable, per-identity acks).
Market-data plane:  SUB    <- engine PUB    :5556 (push; topics tick/trade/event).

See PROTOCOL.md §1-§5 for the binding schemas.

Windows note: callers must set `asyncio.WindowsSelectorEventLoopPolicy()`
BEFORE creating any zmq.asyncio socket/loop — worker.py and main.py do this
in their entrypoints.
"""

import asyncio
import logging
import os

import zmq
import zmq.asyncio

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_ORDER_PORT = 5555
DEFAULT_DATA_PORT = 5556

ACK_TIMEOUT_S = 5.0


def env_host() -> str:
    """Engine host from the environment (PROTOCOL.md §2)."""
    return os.getenv("ZMQ_HOST", DEFAULT_HOST)


def env_order_port() -> int:
    """Engine ROUTER (order-entry) port from the environment."""
    return int(os.getenv("ZMQ_ORDER_PORT", str(DEFAULT_ORDER_PORT)))


def env_data_port() -> int:
    """Engine PUB (market-data) port from the environment."""
    return int(os.getenv("ZMQ_DATA_PORT", str(DEFAULT_DATA_PORT)))


_context: "zmq.asyncio.Context | None" = None


def get_context() -> zmq.asyncio.Context:
    """Return the process-wide shared zmq.asyncio context (lazily created)."""
    global _context
    if _context is None:
        _context = zmq.asyncio.Context.instance()
    return _context


def make_dealer(identity: str, host: str, port: int) -> zmq.asyncio.Socket:
    """Create a connected DEALER socket with the given session identity.

    The engine ROUTER addresses replies by this identity, so it must be unique
    per socket across all connected clients.
    """
    sock = get_context().socket(zmq.DEALER)
    sock.setsockopt_string(zmq.IDENTITY, identity)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{host}:{port}")
    return sock


def make_sub(host: str, port: int, topics: "list[str]") -> zmq.asyncio.Socket:
    """Create a connected SUB socket subscribed to the given topics."""
    sock = get_context().socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{host}:{port}")
    for topic in topics:
        sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    return sock


class AsyncOrderGateway:
    """Pooled DEALER gateway for concurrent request/ack round-trips.

    The engine ROUTER sends exactly ONE ack per request, addressed to the
    sending socket's identity. A single DEALER therefore serializes the whole
    process to one in-flight request at a time. To allow concurrency, we pool
    N DEALER sockets (identities `<prefix>-0` .. `<prefix>-{N-1}`), guard each
    with its own asyncio.Lock, and round-robin across them — up to N acks can
    be in flight simultaneously (the non-blocking fix).
    """

    def __init__(
        self,
        identity_prefix: str,
        host: "str | None" = None,
        port: "int | None" = None,
        pool_size: int = 4,
        timeout: float = ACK_TIMEOUT_S,
    ) -> None:
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        self._host = host if host is not None else env_host()
        self._port = port if port is not None else env_order_port()
        self._prefix = identity_prefix
        self._timeout = timeout
        self._sockets: "list[zmq.asyncio.Socket]" = [
            make_dealer(f"{identity_prefix}-{i}", self._host, self._port)
            for i in range(pool_size)
        ]
        self._locks: "list[asyncio.Lock]" = [asyncio.Lock() for _ in range(pool_size)]
        self._next = 0
        self._closed = False
        logger.info(
            "Order gateway '%s' up: %d DEALER sockets -> tcp://%s:%d",
            identity_prefix, pool_size, self._host, self._port,
        )

    async def request(self, msg: dict) -> dict:
        """Send one request on a pooled DEALER socket and await its ack.

        Raises:
            TimeoutError: no ack within the configured timeout. The socket is
                recycled (closed and reconnected with the same identity) so a
                late, stale ack can never be mistaken for the next reply —
                transient timeouts are safe to catch and retry.
            RuntimeError: the gateway has been closed.
        """
        if self._closed:
            raise RuntimeError("AsyncOrderGateway is closed")

        idx = self._next
        self._next = (self._next + 1) % len(self._sockets)

        async with self._locks[idx]:
            sock = self._sockets[idx]
            await sock.send_json(msg)

            poller = zmq.asyncio.Poller()
            poller.register(sock, zmq.POLLIN)
            events = dict(await poller.poll(timeout=int(self._timeout * 1000)))
            if sock not in events:
                # Recycle the socket so a late ack cannot desynchronize it.
                self._recycle(idx)
                raise TimeoutError(
                    f"No ack from engine ROUTER at "
                    f"tcp://{self._host}:{self._port} within {self._timeout:.0f}s "
                    f"(identity {self._prefix}-{idx}, msg={msg.get('msg')}). "
                    f"Is the engine running?"
                )
            return await sock.recv_json()

    def _recycle(self, idx: int) -> None:
        """Close and reconnect one pooled socket (same identity)."""
        self._sockets[idx].close(linger=0)
        self._sockets[idx] = make_dealer(
            f"{self._prefix}-{idx}", self._host, self._port
        )
        logger.warning("Recycled DEALER socket %s-%d after timeout",
                       self._prefix, idx)

    def close(self) -> None:
        """Close every pooled socket. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for sock in self._sockets:
            sock.close(linger=0)
