"""Thin ZeroMQ transport shell around MatchingEngine (PROTOCOL §1, §12.1).

Single thread: ROUTER for order entry (per-identity acks), PUB for market
data, and a drift-free 1 Hz wall-clock timer for physics ticks. All market
logic lives in engine.matching_engine — this file only moves bytes.

Run with: python -m engine.server
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Tuple

import zmq
from dotenv import load_dotenv

from .matching_engine import MatchingEngine
from .persistence import Persistence

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - [ENGINE] - %(message)s")
logger = logging.getLogger("engine.server")

POLL_TIMEOUT_MS = 50
TICK_INTERVAL_S = 1.0
# Auto-init defaults so the engine is immediately alive (INIT_SIM re-inits).
DEFAULT_SYMBOL = "TCS"
DEFAULT_SECTOR = "TECH"
DEFAULT_PRICE = 190.0
DEFAULT_SEED = 42


def _publish(pub: zmq.Socket, pubs: List[Tuple[str, dict]]) -> None:
    for topic, payload in pubs:
        pub.send_multipart([topic.encode(), json.dumps(payload).encode()])


def _handle_request(router: zmq.Socket, pub: zmq.Socket,
                    engine: MatchingEngine, frames: List[bytes]) -> None:
    """Decode one ROUTER request, tolerating both DEALER ([identity,
    payload]) and REQ ([identity, empty, payload]) envelopes, and reply
    with a matching envelope."""
    identity = frames[0]
    if len(frames) >= 3 and frames[1] == b"":
        envelope = [identity, b""]
    else:
        envelope = [identity]
    try:
        msg = json.loads(frames[-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        ack: dict = {"msg": "ACK", "status": "REJECTED", "order_id": 0,
                     "reason": "invalid JSON payload", "seq": 0}
        publications: List[Tuple[str, dict]] = []
    else:
        ack, publications = engine.handle_message(msg)
    router.send_multipart(envelope + [json.dumps(ack).encode()])
    _publish(pub, publications)


def main() -> None:
    load_dotenv()
    host = os.getenv("ZMQ_HOST", "127.0.0.1")
    order_port = os.getenv("ZMQ_ORDER_PORT", "5555")
    data_port = os.getenv("ZMQ_DATA_PORT", "5556")

    ctx = zmq.Context()
    router = ctx.socket(zmq.ROUTER)
    router.bind(f"tcp://{host}:{order_port}")
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{host}:{data_port}")

    db = Persistence()
    engine = MatchingEngine(db=db)
    run_id = engine.init_sim(DEFAULT_SYMBOL, DEFAULT_SECTOR, DEFAULT_PRICE,
                             seed=DEFAULT_SEED)
    logger.info("engine online: ROUTER tcp://%s:%s | PUB tcp://%s:%s | %s",
                host, order_port, host, data_port, run_id)

    poller = zmq.Poller()
    poller.register(router, zmq.POLLIN)
    next_tick = time.monotonic() + TICK_INTERVAL_S
    try:
        while True:
            events = dict(poller.poll(POLL_TIMEOUT_MS))
            if router in events:
                while True:  # drain everything queued this wakeup
                    try:
                        frames = router.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    _handle_request(router, pub, engine, frames)
            now = time.monotonic()
            if now >= next_tick:
                _publish(pub, engine.physics_tick())
                next_tick += TICK_INTERVAL_S
                if next_tick <= now:  # fell behind; don't burst-tick
                    next_tick = now + TICK_INTERVAL_S
    except KeyboardInterrupt:
        logger.info("shutdown requested")
    finally:
        db.close()
        router.close(0)
        pub.close(0)
        ctx.term()
        logger.info("engine offline")


if __name__ == "__main__":
    main()
