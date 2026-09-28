"""Full-system integration smoke — boots engine + AI swarm + bridge together.

Proves the production stack works end to end:
  • engine server (both ZMQ planes) + SQLite persistence
  • the AI swarm actually connecting and trading via its trained PPO models
  • the FastAPI bridge serving the dashboard + REST + reading the persisted run

Run from swarm_orchestrator:  ../venv/Scripts/python scripts/integration_smoke.py
Exit 0 = the whole system is alive and the swarm is trading.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

import zmq

HOST = os.getenv("ZMQ_HOST", "127.0.0.1")
DATA_PORT = os.getenv("ZMQ_DATA_PORT", "5556")
BRIDGE = f"http://127.0.0.1:{os.getenv('BRIDGE_PORT', '8000')}"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
FAILS = []
procs = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILS.append(name)


def spawn(label, args):
    print(f"  starting {label} ...")
    p = subprocess.Popen([PY] + args, cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append((label, p))
    return p


def get(path):
    with urllib.request.urlopen(BRIDGE + path, timeout=4) as r:
        return json.loads(r.read())


def main():
    try:
        spawn("engine", ["-m", "engine.server"])
        time.sleep(3)
        spawn("bridge", ["bridge.py"])
        time.sleep(3)
        # Swarm loads torch models at import — give it time.
        spawn("swarm", ["worker.py", "--role", "all"])
        print("  waiting for swarm to load PPO models & trade (25s) ...")
        time.sleep(25)

        print("Bridge:")
        try:
            health = get("/api/health")
            check("bridge /api/health reports engine up", health.get("engine") == "up", str(health))
        except Exception as e:
            check("bridge /api/health reachable", False, repr(e))

        try:
            with urllib.request.urlopen(BRIDGE + "/", timeout=4) as r:
                html = r.read().decode("utf-8", "replace")
            check("bridge serves dashboard", "CHRONOS" in html.upper() or "<html" in html.lower())
        except Exception as e:
            check("bridge serves dashboard", False, repr(e))

        print("Persistence:")
        try:
            runs = get("/api/runs")
            check("a run is persisted to SQLite", isinstance(runs, list) and len(runs) >= 1,
                  f"{len(runs) if isinstance(runs, list) else '?'} run(s)")
        except Exception as e:
            check("runs endpoint", False, repr(e))

        print("Swarm trading (data plane):")
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.connect(f"tcp://{HOST}:{DATA_PORT}")
        sub.setsockopt(zmq.SUBSCRIBE, b"tick")
        sub.setsockopt(zmq.SUBSCRIBE, b"trade")
        swarm_ids, trade_count, ticks = set(), 0, 0
        deadline = time.time() + 8
        while time.time() < deadline and (len(swarm_ids) < 1 or ticks < 3):
            if sub.poll(500):
                topic, payload = sub.recv_multipart()
                d = json.loads(payload)
                if topic == b"tick":
                    ticks += 1
                    for row in d.get("leaderboard", []):
                        if row["id"].startswith(("MM_", "WHALE_", "RETAIL_")):
                            swarm_ids.add(row["id"])
                elif topic == b"trade":
                    trade_count += 1
        sub.close(); ctx.term()
        check("AI swarm agents present in leaderboard", len(swarm_ids) >= 1,
              f"{len(swarm_ids)} agents e.g. {sorted(swarm_ids)[:5]}")
        check("trades are flowing", trade_count >= 1, f"{trade_count} prints in 8s")
    finally:
        print("Tearing down ...")
        for label, p in reversed(procs):
            p.terminate()
        for label, p in reversed(procs):
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    print()
    if FAILS:
        print(f"INTEGRATION FAILED — {FAILS}")
        sys.exit(1)
    print("INTEGRATION PASSED — engine + AI swarm + bridge run together, swarm is trading.")
    sys.exit(0)


if __name__ == "__main__":
    main()
