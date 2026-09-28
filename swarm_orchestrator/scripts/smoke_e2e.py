"""End-to-end smoke test — boots the real engine server and drives it over ZeroMQ.

Verifies the two-plane architecture end to end:
  • ORDER plane (DEALER -> ROUTER :5555): INIT_SIM, resting LIMIT, crossing MARKET, acks.
  • DATA plane  (SUB  <- PUB    :5556): tick/trade push, book-driven price.

Run from the swarm_orchestrator directory:  ../venv/Scripts/python scripts/smoke_e2e.py
Exit code 0 = all checks passed.
"""
import json
import os
import subprocess
import sys
import time

import zmq

HOST = os.getenv("ZMQ_HOST", "127.0.0.1")
ORDER_PORT = os.getenv("ZMQ_ORDER_PORT", "5555")
DATA_PORT = os.getenv("ZMQ_DATA_PORT", "5556")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAILS = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        FAILS.append(name)


def request(sock, msg, timeout_ms=3000):
    sock.send_json(msg)
    if sock.poll(timeout_ms):
        return sock.recv_json()
    return None


def main():
    print("Booting engine server ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "engine.server"], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(3.0)  # let it bind
        ctx = zmq.Context()

        # DATA plane: subscribe first so we catch the pushed ticks.
        sub = ctx.socket(zmq.SUB)
        sub.connect(f"tcp://{HOST}:{DATA_PORT}")
        for topic in (b"tick", b"trade", b"event"):
            sub.setsockopt(zmq.SUBSCRIBE, topic)

        # ORDER plane: DEALER for reliable order entry.
        dealer = ctx.socket(zmq.DEALER)
        dealer.setsockopt(zmq.IDENTITY, b"SMOKE")
        dealer.connect(f"tcp://{HOST}:{ORDER_PORT}")

        print("Order plane:")
        ack = request(dealer, {"msg": "INIT_SIM", "symbol": "TCS", "sector": "TECH",
                               "price": 190.0, "seed": 7})
        check("INIT_SIM acked", ack is not None and "run_id" in (ack or {}))

        ack = request(dealer, {"msg": "ORDER", "agent_id": "MM_1", "action": "SELL",
                               "type": "LIMIT", "qty": 100, "price": 190.10})
        check("resting LIMIT acked", ack is not None and ack.get("status") in ("RESTING", "PARTIAL", "FILLED"))

        ack = request(dealer, {"msg": "ORDER", "agent_id": "WHALE_1", "action": "BUY",
                               "type": "MARKET", "qty": 60})
        check("MARKET buy filled @190.10", ack is not None
              and ack.get("executed_qty") == 60
              and abs(ack.get("average_price", 0) - 190.10) < 1e-6)
        check("ack carries authoritative ledger", ack is not None
              and ack.get("pos") == 60 and "cash" in ack)

        print("Data plane:")
        ticks, trades = [], []
        deadline = time.time() + 4.0
        while time.time() < deadline and len(ticks) < 3:
            if sub.poll(500):
                topic, payload = sub.recv_multipart()
                data = json.loads(payload)
                if topic == b"tick":
                    ticks.append(data)
                elif topic == b"trade":
                    trades.append(data)

        check("received tick snapshots (push)", len(ticks) >= 2)
        if ticks:
            t = ticks[-1]
            required = {"seq", "last_price", "best_bid", "best_ask", "mid_price",
                        "micro_price", "rsi", "vwap", "ofi", "regime", "lob_bids",
                        "lob_asks", "leaderboard", "bar"}
            check("tick has full §5.1 schema", required.issubset(t))
            check("price is book-driven (== a real trade print)",
                  any(abs(t["last_price"] - tr["price"]) < 1e-6 for tr in trades)
                  or t["last_price"] == 190.10)
            check("sequence numbers present & increasing",
                  [x["seq"] for x in ticks] == sorted(x["seq"] for x in ticks))

        sub.close(); dealer.close(); ctx.term()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print()
    if FAILS:
        print(f"SMOKE FAILED — {len(FAILS)} check(s): {FAILS}")
        sys.exit(1)
    print("SMOKE PASSED — engine is book-driven and both planes work.")
    sys.exit(0)


if __name__ == "__main__":
    main()
