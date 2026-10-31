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

from .lockstep import Lockstep
from .matching_engine import MatchingEngine
from .persistence import Persistence

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - [ENGINE] - %(message)s")
logger = logging.getLogger("engine.server")

POLL_TIMEOUT_MS = 50
TICK_INTERVAL_S = 1.0
# Auto-init defaults so the engine is immediately alive (INIT_SIM re-inits).
# DEFAULT_SYMBOL = "TCS"
# DEFAULT_SECTOR = "TECH"
# DEFAULT_PRICE = 190.0
# DEFAULT_SEED = 42


def _publish(pub: zmq.Socket, pubs: List[Tuple[str, dict]]) -> None:
    for topic, payload in pubs:
        pub.send_multipart([topic.encode(), json.dumps(payload).encode()])


def _handle_request(router: zmq.Socket, pub: zmq.Socket,
                    engine: MatchingEngine, frames: List[bytes],
                    lock: Lockstep) -> None:
    """Decode one ROUTER request, tolerating both DEALER ([identity,
    payload]) and REQ ([identity, empty, payload]) envelopes, and reply
    with a matching envelope. TICK_DONE is a scheduling message for the
    server's lockstep, not for the engine core."""
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
        if isinstance(msg, dict) and msg.get("msg") == "TICK_DONE":
            counted = lock.tick_done(msg.get("agent_id", ""), msg.get("tick_seq"))
            ack = {"msg": "ACK", "status": "TICK_DONE" if counted else "TICK_DONE_STALE",
                   "seq": engine.seq}
            publications = []
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
    # run_id = engine.init_sim(DEFAULT_SYMBOL, DEFAULT_SECTOR, DEFAULT_PRICE,
    #                          seed=DEFAULT_SEED)
    logger.info("engine online: ROUTER tcp://%s:%s | PUB tcp://%s:%s | idle (start ka intezaar)",
                host, order_port, host, data_port)


    # Opt-in lockstep (engine/lockstep.py): hold each physics tick until this
    # many swarm workers have sent TICK_DONE for the previous one.
    lock = Lockstep(int(os.getenv("CHRONOS_LOCKSTEP_WORKERS", "0")),
                    float(os.getenv("CHRONOS_LOCKSTEP_TIMEOUT_S", "10")))
    if lock.enabled:
        logger.info("lockstep on: waiting for %d worker(s) per tick, timeout %.1fs",
                    lock.workers, lock.timeout_s)

    engine.paused = True
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
                    _handle_request(router, pub, engine, frames, lock)
            now = time.monotonic()
            if now >= next_tick and lock.ready(now):
                if lock.timed_out(now):
                    logger.warning("lockstep timeout: %d worker(s) silent for %.1fs; "
                                   "this run is no longer reproducible",
                                   lock.missing(), lock.timeout_s)
                    _publish(pub, [("event", {"seq": engine._next_seq(), "kind": "lockstep",
                                              "action": "timeout",
                                              "missing_workers": lock.missing()})])
                lock.release()
                pubs = engine.physics_tick()
                _publish(pub, pubs)
                tick = next((p for t, p in reversed(pubs) if t == "tick"), None)
                if tick is not None:
                    lock.published(tick["seq"], bool(tick.get("paused")), time.monotonic())
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
