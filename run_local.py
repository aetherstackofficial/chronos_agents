#!/usr/bin/env python3
"""Run the whole Chronos stack natively -- no Docker.

    python run_local.py                  # engine + oracle + swarm + bridge, then open the dashboard
    python run_local.py --skip oracle    # leave a service out (comma separated)
    python run_local.py --only engine,bridge
    python run_local.py --check          # start everything, verify it is healthy, stop (exit 0/1)

Ctrl+C stops everything, engine last so it can flush its database.

What it does
  1. picks a Python (venv first), and checks the dependencies are installed
  2. checks the ports are free (a leftover `docker compose up` is the usual culprit)
  3. starts the engine, waits for it, then starts oracle, swarm and bridge
  4. prefixes every line of output with the service name and also saves it to
     swarm_orchestrator/.run/logs/<service>.log

Local runs use their own database, swarm_orchestrator/.run/chronos.db, so the old
chronos.db is never touched. Pass --db to use another file.

Only the standard library is needed to run this launcher. The services themselves
use the interpreter chosen in step 1.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parent
APP = REPO / "swarm_orchestrator"
RUN_DIR = APP / ".run"
IS_WIN = os.name == "nt"

# Runs inside each service process. On Windows a CTRL_BREAK is the only signal we can
# send to a process in its own group, and Python treats it as "terminate now" unless we
# map it to KeyboardInterrupt -- which is what the services already handle for Ctrl+C
# (the engine flushes its database in a `finally`).
WRAPPER = (
    "import runpy, signal, sys\n"
    "if hasattr(signal, 'SIGBREAK'):\n"
    "    signal.signal(signal.SIGBREAK, signal.default_int_handler)\n"
    "kind, target = sys.argv[1], sys.argv[2]\n"
    "sys.argv = [target]\n"
    "if kind == 'module':\n"
    "    runpy.run_module(target, run_name='__main__', alter_sys=True)\n"
    "else:\n"
    "    runpy.run_path(target, run_name='__main__')\n"
)

# Asked of the chosen interpreter without importing anything (torch takes seconds).
PREFLIGHT = (
    "import importlib.util, json, sys\n"
    "def has(m):\n"
    "    try:\n"
    "        return importlib.util.find_spec(m) is not None\n"
    "    except Exception:\n"
    "        return False\n"
    "print(json.dumps({m: has(m) for m in sys.argv[1:]}))\n"
)
REQUIRED = ["zmq", "numpy", "pandas", "sortedcontainers", "torch", "stable_baselines3",
            "gymnasium", "fastapi", "uvicorn", "dotenv", "httpx", "psutil"]
OPTIONAL = {"google.genai": "oracle tier 1 (Gemini)", "simplefix": "FIX gateway",
            "reportlab": "PDF reports"}


class Service:
    def __init__(self, name: str, kind: str, target: str, color: str, essential: bool,
                 what: str, ready_marker: str = "") -> None:
        self.name, self.kind, self.target = name, kind, target
        self.color, self.essential, self.what = color, essential, what
        self.ready_marker = ready_marker  # a log line that means "fully up"
        self.ready = False
        self.proc: "subprocess.Popen | None" = None
        self.log = None
        self.tail: "collections.deque[str]" = collections.deque(maxlen=12)
        self.reported_exit = False


SERVICES = [
    Service("engine", "module", "engine.server", "36", True, "matching engine"),
    Service("oracle", "path", "oracle.py", "35", False, "news scoring"),
    Service("swarm", "path", "main.py", "33", False, "AI agents", "Orchestrator online"),
    Service("bridge", "path", "bridge.py", "32", True, "API + dashboard"),
]
BY_NAME = {s.name: s for s in SERVICES}

_print_lock = threading.Lock()
_color = False


def say(text: str = "") -> None:
    with _print_lock:
        print(text, flush=True)


def paint(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if _color else text


_stopping = False


def emit(svc: Service, line: str) -> None:
    # While shutting down, children print their own KeyboardInterrupt tracebacks (that is
    # how Python reports Ctrl+C). Dim them so a real error is still easy to spot.
    text = paint("2", line) if _stopping else line
    say(f"{paint(svc.color, f'{svc.name:<6}')} | {text}")


# --------------------------------------------------------------------------- config
def read_env_file(path: Path) -> dict:
    """Minimal KEY=VALUE reader (the launcher must not depend on python-dotenv)."""
    values: dict = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #")[0].strip().strip("\"'")
        values[key.strip()] = value
    return values


def find_python(explicit: "str | None") -> str:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("CHRONOS_PYTHON"):
        candidates.append(Path(os.environ["CHRONOS_PYTHON"]))
    sub = Path("Scripts/python.exe") if IS_WIN else Path("bin/python")
    for folder in (REPO / "venv", APP / "venv", REPO / ".venv", APP / ".venv"):
        candidates.append(folder / sub)
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


def preflight(py: str) -> bool:
    wanted = REQUIRED + list(OPTIONAL)
    try:
        out = subprocess.run([py, "-c", PREFLIGHT, *wanted], capture_output=True, text=True,
                             timeout=60, cwd=APP)
        found = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001 - any failure here means "cannot use this python"
        say(paint("31", f"cannot run {py}: {exc}"))
        return False
    missing = [m for m in REQUIRED if not found.get(m)]
    if missing:
        say(paint("31", f"missing packages for {py}: {', '.join(missing)}"))
        say(f"  install with:  \"{py}\" -m pip install -r swarm_orchestrator/requirements.txt")
        return False
    for mod, why in OPTIONAL.items():
        if not found.get(mod):
            say(paint("33", f"note: optional package '{mod}' not installed ({why} will be unavailable)"))
    return True


# --------------------------------------------------------------------------- networking
def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def port_owner(port: int) -> str:
    if not IS_WIN:
        return ""
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True,
                             timeout=10).stdout
        for line in out.splitlines():
            p = line.split()
            if len(p) >= 5 and p[3] == "LISTENING" and p[1].endswith(f":{port}"):
                name = subprocess.run(["tasklist", "/FI", f"PID eq {p[4]}", "/FO", "CSV", "/NH"],
                                      capture_output=True, text=True, timeout=10
                                      ).stdout.split(",")[0].strip('"')
                return f"  [PID {p[4]}, {name}]"
    except Exception:  # noqa: BLE001
        pass
    return ""


def bridge_healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as r:
            return r.status == 200 and json.loads(r.read()).get("engine") == "up"
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- processes
def start(svc: Service, py: str, env: dict) -> None:
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    svc.log = open(RUN_DIR / "logs" / f"{svc.name}.log", "w", encoding="utf-8")
    kwargs: dict = dict(cwd=APP, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                        errors="replace", bufsize=1)
    if IS_WIN:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    svc.proc = subprocess.Popen([py, "-u", "-c", WRAPPER, svc.kind, svc.target], **kwargs)
    threading.Thread(target=pump, args=(svc,), daemon=True).start()
    emit(svc, paint("2", f"started (pid {svc.proc.pid})"))


def pump(svc: Service) -> None:
    for raw in svc.proc.stdout:  # type: ignore[union-attr]
        line = raw.rstrip("\r\n")
        svc.tail.append(line)
        if svc.ready_marker and svc.ready_marker in line:
            svc.ready = True
        try:
            svc.log.write(line + "\n")
            svc.log.flush()
        except ValueError:  # log already closed during shutdown
            pass
        emit(svc, line)


def alive(svc: Service) -> bool:
    return svc.proc is not None and svc.proc.poll() is None


def wait_until(check, svc: "Service | None", timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if svc is not None and not alive(svc):
            return False
        if check():
            return True
        time.sleep(0.25)
    return False


def request_stop(svc: Service) -> None:
    try:
        if IS_WIN:
            svc.proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[union-attr]
        else:
            os.killpg(svc.proc.pid, signal.SIGINT)  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - already gone, or no console to signal
        pass


def force_stop(svc: Service) -> None:
    try:
        if IS_WIN:
            subprocess.run(["taskkill", "/PID", str(svc.proc.pid), "/T", "/F"],  # type: ignore[union-attr]
                           capture_output=True)
        else:
            os.killpg(svc.proc.pid, signal.SIGKILL)  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass


def wait_exit(group: "list[Service]", timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(alive(s) for s in group):
        time.sleep(0.2)


def stop_all() -> None:
    global _stopping
    running = [s for s in SERVICES if alive(s)]
    if not running:
        return
    _stopping = True
    say(paint("1", "stopping ..."))
    others = [s for s in running if s.name != "engine"]
    engine = [s for s in running if s.name == "engine"]
    for group, grace in ((others, 6.0), (engine, 8.0)):  # engine last: it flushes the database
        for s in group:
            request_stop(s)
        wait_exit(group, grace)
        for s in group:
            if alive(s):
                emit(s, paint("33", "did not stop in time, killing"))
                force_stop(s)
    time.sleep(0.4)  # let the pump threads drain the last lines
    for s in SERVICES:
        if s.log:
            s.log.close()


# --------------------------------------------------------------------------- main
def main() -> int:
    global _color
    ap = argparse.ArgumentParser(description="Run the whole Chronos stack without Docker.")
    ap.add_argument("--skip", default="", help="comma separated services to leave out")
    ap.add_argument("--only", default="", help="comma separated services to run (others skipped)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the dashboard")
    ap.add_argument("--check", action="store_true",
                    help="start, verify the stack is healthy, stop; exit code 0 = healthy")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address the dashboard listens on (0.0.0.0 to reach it from your phone)")
    ap.add_argument("--port", type=int, default=0, help="dashboard port (default 8000)")
    ap.add_argument("--db", default="", help="SQLite file (default swarm_orchestrator/.run/chronos.db)")
    ap.add_argument("--python", default="", help="interpreter for the services (default: venv)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    _color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    if _color and IS_WIN:
        os.system("")  # switches the Windows console to ANSI mode

    names = {n.strip() for n in (args.only or "").split(",") if n.strip()}
    skip = {n.strip() for n in args.skip.split(",") if n.strip()}
    unknown = (names | skip) - set(BY_NAME)
    if unknown:
        say(paint("31", f"unknown service(s): {', '.join(sorted(unknown))}  (choose from {', '.join(BY_NAME)})"))
        return 2
    chosen = [s for s in SERVICES if (not names or s.name in names) and s.name not in skip]
    if not chosen:
        say(paint("31", "nothing to run"))
        return 2

    py = find_python(args.python or None)
    say(f"python : {py}")
    if not preflight(py):
        return 2

    file_env = read_env_file(APP / ".env")

    def pick(key: str, default: str) -> str:
        return os.environ.get(key) or file_env.get(key) or default

    order_port = int(pick("ZMQ_ORDER_PORT", "5555"))
    data_port = int(pick("ZMQ_DATA_PORT", "5556"))
    oracle_port = int(pick("ORACLE_PORT", "5557"))
    bridge_port = args.port or int(pick("BRIDGE_PORT", "8000"))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    db_path = str(Path(args.db).resolve()) if args.db else str(RUN_DIR / "chronos.db")

    wants = {"engine": [order_port, data_port], "oracle": [oracle_port], "bridge": [bridge_port]}
    busy = [(p, s.name) for s in chosen for p in wants.get(s.name, []) if port_open(p)]
    if busy:
        for port, name in busy:
            say(paint("31", f"port {port} ({name}) is already in use{port_owner(port)}"))
        say("  Is `docker compose up` still running?  ->  docker compose down")
        say("  Or a previous run left a process behind.  ->  taskkill /PID <pid> /T /F")
        return 2

    env = dict(os.environ)
    env.update({
        "ZMQ_HOST": "127.0.0.1", "ORACLE_HOST": "127.0.0.1",   # local run: never the docker names
        "ZMQ_ORDER_PORT": str(order_port), "ZMQ_DATA_PORT": str(data_port),
        "ORACLE_PORT": str(oracle_port),
        "BRIDGE_HOST": args.host, "BRIDGE_PORT": str(bridge_port),
        "CHRONOS_DB": db_path,
        "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
        "TF_CPP_MIN_LOG_LEVEL": "2", "TF_ENABLE_ONEDNN_OPTS": "0",
    })
    keys = {k: bool(os.environ.get(k) or file_env.get(k)) and "your_" not in (os.environ.get(k) or file_env.get(k, ""))
            for k in ("GEMINI_API_KEY", "GROQ_API_KEY")}
    if any(s.name == "oracle" for s in chosen):
        tiers = [n for n, k in (("Gemini", "GEMINI_API_KEY"), ("Groq", "GROQ_API_KEY")) if keys[k]]
        say(f"oracle : {' -> '.join(tiers + ['word list'])}")
    say(f"db     : {db_path}")
    say(f"logs   : {RUN_DIR / 'logs'}")
    say("")

    def _interrupt(*_: object) -> None:
        raise KeyboardInterrupt

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGBREAK", None)):
        if sig is not None:
            signal.signal(sig, _interrupt)

    exit_code = 0
    try:
        engine = BY_NAME["engine"]
        if engine in chosen:
            start(engine, py, env)
            if not wait_until(lambda: port_open(order_port), engine, 40):
                say(paint("31", "engine did not come up"))
                return 1
        for svc in chosen:
            if svc is not engine:
                start(svc, py, env)

        bridge = BY_NAME["bridge"]
        if bridge in chosen:
            if not wait_until(lambda: bridge_healthy(bridge_port), bridge, 90):
                say(paint("31", "bridge did not become healthy (see its log above)"))
                return 1
        if BY_NAME["oracle"] in chosen:
            wait_until(lambda: port_open(oracle_port), BY_NAME["oracle"], 30)

        url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{bridge_port}"
        say("")
        say(paint("1", "Chronos is up"))
        for s in chosen:
            if s.name == "engine" or s.name == "bridge":
                state = "running" if alive(s) else "STOPPED"
            elif s.name == "oracle":
                state = "running" if alive(s) and port_open(oracle_port) else (
                    "starting" if alive(s) else "STOPPED (see its log)")
            else:
                state = ("ready" if s.ready else "loading AI models") if alive(s) else "STOPPED (see its log)"
            say(f"  {s.name:<7}{state:<26}{s.what}")
        if bridge in chosen:
            say(f"  dashboard  {url}")
            say("  The market starts PAUSED -- press START in the dashboard.")
        say("  Ctrl+C stops everything.")
        say("")

        if args.check:
            swarm = BY_NAME["swarm"]
            if swarm in chosen:
                if wait_until(lambda: swarm.ready, swarm, 120):
                    say(paint("32", "check: swarm ready"))
                else:
                    say(paint("33", "check: swarm did not report ready"))
            time.sleep(2)  # give a crashing service time to crash
            bad = [s.name for s in chosen if s.essential and not alive(s)]
            late = [s.name for s in chosen if not s.essential and not alive(s)]
            if late:
                say(paint("33", f"check: not running: {', '.join(late)}"))
            say(paint("32" if not bad else "31", "check: essential services healthy" if not bad
                      else f"check: essential service stopped: {', '.join(bad)}"))
            return 1 if bad else 0
        if bridge in chosen and not args.no_browser:
            webbrowser.open(url)

        while True:
            time.sleep(0.5)
            for s in chosen:
                if s.proc is not None and s.proc.poll() is not None and not s.reported_exit:
                    s.reported_exit = True
                    code = s.proc.returncode
                    if s.essential:
                        say(paint("31", f"{s.name} exited (code {code}) -- shutting everything down"))
                        exit_code = 1
                        raise KeyboardInterrupt
                    say(paint("33", f"{s.name} exited (code {code}); the rest keeps running. "
                                    f"Log: {RUN_DIR / 'logs' / (s.name + '.log')}"))
    except KeyboardInterrupt:
        pass
    finally:
        stop_all()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
