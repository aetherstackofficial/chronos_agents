"""v2 end-to-end smoke — boots engine + bridge, exercises every new endpoint and
the full strategy -> results -> CSV/PDF flow. Exit 0 = the whole v2 layer works."""
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
BASE = f"http://127.0.0.1:{os.getenv('BRIDGE_PORT', '8000')}"
FAILS, procs = [], []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + str(detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def req(method, path, body=None, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=30) as resp:
        blob = resp.read()
        return blob if raw else json.loads(blob)


def spawn(label, args):
    p = subprocess.Popen([PY] + args, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append((label, p)); return p


def main():
    try:
        spawn("engine", ["-m", "engine.server"]); time.sleep(3)
        spawn("bridge", ["bridge.py"]); time.sleep(4)

        print("Frontend & health:")
        html = req("GET", "/", raw=True).decode("utf-8", "replace")
        check("dashboard shell served", "CHRONOS" in html.upper() and "app.js" in html)
        check("engine up via bridge", req("GET", "/api/health").get("engine") == "up")

        print("Script library:")
        code = ("from runner.sdk import Strategy\nclass UserStrategy(Strategy):\n"
                "    def on_tick(self, state):\n        return [{'action':'BUY','type':'MARKET','qty':5}]\n")
        saved = req("POST", "/api/scripts", {"name": "smoke_buy", "code": code})
        sid = saved["script_id"]
        check("save script to file", bool(sid))
        check("list scripts", any(s["script_id"] == sid for s in req("GET", "/api/scripts")))
        check("get script code", req("GET", f"/api/scripts/{sid}")["code"] == code)

        print("New features:")
        cop = req("POST", "/api/copilot", {"description": "buy when rsi below 30 qty 40"})
        check("copilot returns valid code", "class UserStrategy" in cop.get("code", ""), cop.get("source"))
        nar = req("POST", "/api/narrate", {})
        check("narrate returns text", len(nar.get("narration", "")) > 0, nar.get("source"))
        scen = req("GET", "/api/timemachine/scenarios")
        check("timemachine 5 scenarios", len(scen) == 5)
        mc = req("POST", "/api/montecarlo", {"symbol": "TCS", "price": 190.0, "n_paths": 12, "horizon": 25})
        check("montecarlo fan", len(mc.get("percentiles", {}).get("p50", [])) == 25,
              f"terminal p50={mc.get('terminal', {}).get('p50')}")

        print("Strategy -> Results -> CSV/PDF:")
        run = req("POST", f"/api/scripts/{sid}/run", {})
        strat_id = run["strategy_id"]
        check("run script (Pillar 2 + results link)", bool(run.get("agent_id", "").startswith("STRAT_")),
              run.get("backend"))
        time.sleep(8)  # let it trade against the live engine
        results = req("GET", "/api/results")
        check("strategy appears in results", any(r["strategy_id"] == strat_id for r in results),
              f"{len(results)} strateg(ies)")
        detail = req("GET", f"/api/results/{strat_id}")
        check("results detail has days/trades", "days" in detail)
        csv = req("GET", f"/api/results/{strat_id}/trades.csv", raw=True)
        check("trades CSV downloads", csv[:1] and len(csv) > 0, f"{len(csv)} bytes")
        pdf = req("GET", f"/api/results/{strat_id}/report.pdf", raw=True)
        check("PDF report downloads", pdf[:4] == b"%PDF", f"{len(pdf)} bytes")
        req("POST", "/api/strategy/stop", {"strategy_id": strat_id})

        print("Time Machine replay + Lab red-team:")
        tm = req("POST", "/api/timemachine/start", {"scenario_id": scen[0]["id"], "seed": 1})
        check("crisis replay starts", tm.get("status") and tm.get("bars", 0) > 0, tm.get("scenario"))
        rt = req("POST", "/api/lab/redteam", {"count": 2})
        check("red-team launches", rt.get("status") == "launched", f"pid={rt.get('pid')}")
        req("POST", "/api/lab/redteam/stop", {})

        print("Pop-out pages served:")
        for pg in ("trades", "agents", "agent", "results"):
            h = req("GET", f"/pages/{pg}.html", raw=True).decode("utf-8", "replace")
            check(f"pages/{pg}.html served", "CHRONOS" in h.upper() and "../lib/" in h)
    finally:
        for _l, p in reversed(procs):
            p.terminate()
        for _l, p in reversed(procs):
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    print()
    if FAILS:
        print(f"V2 SMOKE FAILED — {FAILS}"); sys.exit(1)
    print("V2 SMOKE PASSED — script library, results+CSV/PDF, and all new features work end to end.")
    sys.exit(0)


if __name__ == "__main__":
    main()
