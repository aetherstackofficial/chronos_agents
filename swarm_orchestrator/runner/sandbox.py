"""Strategy sandbox manager (PROTOCOL.md section 12.2) — imported by the bridge.

Launches user strategy code as a supervised process, preferring an isolated
Docker container and falling back to a plain subprocess.

Backends
--------
* **docker** — ``python:3.11-slim`` container with hard resource limits
  (``--memory 512m --cpus 1 --pids-limit 64``), the repository mounted
  read-only, and network access to the engine. This is the only backend that
  provides an actual isolation boundary and is REQUIRED for untrusted code.
* **subprocess** — ``sys.executable -m runner.sdk`` with a psutil watchdog that
  kills the process tree if it exceeds 512 MB RSS or 15 minutes of CPU time.

HONESTY REQUIREMENT (No-Dummy charter): the subprocess backend is a *resource
guard*, **NOT a security boundary**. The strategy runs as the same OS user with
full filesystem and network access; a malicious strategy can read/write
anything this user can. Docker mode is required for untrusted code — subprocess
mode exists so the system stays usable on hosts without Docker, and it says so
loudly in the logs. We do not pretend it is secure.

Artifacts live under ``swarm_orchestrator/.strategies/<strategy_id>/``
(``strategy.py``, ``run.log``, ``meta.json``) plus an append-only
``.strategies/audit.log`` of JSON lines.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Optional

import psutil

log = logging.getLogger("runner.sandbox")
if not log.handlers:  # dedicated handler so the [SANDBOX] tag survives co-imports
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - [SANDBOX] - %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False

# --------------------------------------------------------------------------- #
# Policy constants                                                             #
# --------------------------------------------------------------------------- #
_ROOT = Path(__file__).resolve().parent.parent  # swarm_orchestrator/
RSS_LIMIT_BYTES: int = 512 * 1024 * 1024  # 512 MB (matches docker --memory 512m)
CPU_LIMIT_SECONDS: float = 15 * 60  # 15 minutes of CPU time
WATCHDOG_POLL_S: float = 2.0
DOCKER_PROBE_TIMEOUT_S: float = 3.0
DOCKER_IMAGE: str = "python:3.11-slim"


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class SandboxManager:
    """Launch, supervise, and stop sandboxed strategy processes.

    API contract (PROTOCOL section 12.2 — the bridge imports and calls this):

    * ``launch(name, code) -> {"strategy_id", "agent_id", "backend"}``
    * ``stop(strategy_id) -> bool``
    * ``list() -> [{"strategy_id","agent_id","name","status","started_at"}]``
    * ``logs(strategy_id, tail=200) -> str``
    """

    def __init__(self, base_dir: str = ".strategies") -> None:
        """Create the manager; ``base_dir`` resolves relative to swarm_orchestrator."""
        base = Path(base_dir)
        self.base_dir: Path = base if base.is_absolute() else (_ROOT / base)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # strategy_id -> {meta fields, "proc": Popen|None, "log_handle": IO|None}
        self._registry: dict[str, dict] = {}
        self._docker_available = self._probe_docker()
        if not self._docker_available:
            log.warning(
                "Docker unavailable — subprocess backend will be used. Subprocess "
                "mode is a resource guard, NOT a security boundary; do not run "
                "untrusted strategy code without Docker.")
        self._reconstruct()

    # ------------------------------------------------------------------ #
    # Backend probing                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _probe_docker() -> bool:
        """True iff the docker CLI exists and the daemon answers within 3s."""
        if shutil.which("docker") is None:
            return False
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=DOCKER_PROBE_TIMEOUT_S,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    # ------------------------------------------------------------------ #
    # Registry reconstruction (survive bridge restarts)                    #
    # ------------------------------------------------------------------ #
    def _reconstruct(self) -> None:
        """Rebuild the in-memory registry from ``.strategies/<id>/meta.json``."""
        for entry_dir in sorted(self.base_dir.iterdir() if self.base_dir.exists() else []):
            meta_path = entry_dir / "meta.json"
            if not entry_dir.is_dir() or not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                log.warning("Skipping unreadable meta %s: %s", meta_path, exc)
                continue
            meta.setdefault("proc", None)
            meta.setdefault("log_handle", None)
            self._registry[meta["strategy_id"]] = meta
        if self._registry:
            log.info("Reconstructed %d strategy record(s) from %s",
                     len(self._registry), self.base_dir)

    # ------------------------------------------------------------------ #
    # Audit trail                                                          #
    # ------------------------------------------------------------------ #
    def _audit(self, action: str, strategy_id: str, name: str, backend: str) -> None:
        """Append one JSON line to ``.strategies/audit.log``."""
        record = {
            "ts": _utc_now_iso(),
            "action": action,
            "strategy_id": strategy_id,
            "name": name,
            "backend": backend,
        }
        try:
            with open(self.base_dir / "audit.log", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            log.warning("Audit write failed: %s", exc)

    def _write_meta(self, entry: dict) -> None:
        """Persist the serialisable part of a registry entry to meta.json."""
        serialisable = {k: v for k, v in entry.items() if k not in ("proc", "log_handle")}
        meta_path = self.base_dir / entry["strategy_id"] / "meta.json"
        try:
            meta_path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
        except OSError as exc:
            log.warning("Meta write failed for %s: %s", entry["strategy_id"], exc)

    # ------------------------------------------------------------------ #
    # Public API — launch                                                  #
    # ------------------------------------------------------------------ #
    def launch(self, name: str, code: str) -> dict:
        """Validate ``code`` and start it in the best available backend.

        Raises:
            ValueError: if the code has a syntax error or does not define
                ``class UserStrategy``.

        Returns:
            ``{"strategy_id": <8 hex>, "agent_id": "STRAT_<id>", "backend": ...}``
        """
        try:
            compile(code, "<strategy>", "exec")
        except SyntaxError as exc:
            raise ValueError(f"strategy code has a syntax error: {exc}") from exc
        if "class UserStrategy" not in code:
            raise ValueError("strategy code must define `class UserStrategy(Strategy)`")

        strategy_id = uuid.uuid4().hex[:8]
        agent_id = f"STRAT_{strategy_id}"
        strategy_dir = self.base_dir / strategy_id
        strategy_dir.mkdir(parents=True, exist_ok=True)
        strategy_file = strategy_dir / "strategy.py"
        strategy_file.write_text(code, encoding="utf-8")

        entry: dict = {
            "strategy_id": strategy_id,
            "agent_id": agent_id,
            "name": name,
            "status": "running",
            "started_at": _utc_now_iso(),
            "backend": None,
            "pid": None,
            "container_id": None,
            "proc": None,
            "log_handle": None,
        }

        if self._docker_available:
            try:
                self._launch_docker(entry, strategy_id, agent_id, name)
            except Exception as exc:  # any docker failure -> honest fallback
                log.warning(
                    "Docker launch failed (%s) — falling back to subprocess backend. "
                    "Reminder: subprocess mode is NOT a security boundary.", exc)
                self._launch_subprocess(entry, strategy_file, agent_id, name)
        else:
            self._launch_subprocess(entry, strategy_file, agent_id, name)

        with self._lock:
            self._registry[strategy_id] = entry
        self._write_meta(entry)
        self._audit("launch", strategy_id, name, entry["backend"])
        log.info("Launched strategy %r as %s (backend=%s)", name, agent_id, entry["backend"])
        return {"strategy_id": strategy_id, "agent_id": agent_id, "backend": entry["backend"]}

    # ------------------------------------------------------------------ #
    # subprocess backend                                                   #
    # ------------------------------------------------------------------ #
    def _launch_subprocess(self, entry: dict, strategy_file: Path,
                           agent_id: str, name: str) -> None:
        """Start the strategy via ``sys.executable -m runner.sdk`` + watchdog.

        WARNING: this is a resource guard (RSS/CPU watchdog), NOT a security
        boundary — the strategy runs unconfined as the current OS user.
        """
        log.warning(
            "Strategy %s starting in SUBPROCESS mode: resource-guarded but NOT "
            "security-isolated. Use Docker for untrusted code.", agent_id)
        run_log = strategy_file.parent / "run.log"
        log_handle: IO[bytes] = open(run_log, "ab")
        env = os.environ.copy()
        env.setdefault("ZMQ_HOST", "127.0.0.1")
        env.setdefault("ZMQ_ORDER_PORT", "5555")
        env.setdefault("ZMQ_DATA_PORT", "5556")
        proc = subprocess.Popen(
            [sys.executable, "-m", "runner.sdk", str(strategy_file),
             "--name", name, "--agent-id", agent_id],
            cwd=str(_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        entry.update(backend="subprocess", pid=proc.pid, proc=proc, log_handle=log_handle)
        watchdog = threading.Thread(
            target=self._watchdog, args=(entry,), daemon=True,
            name=f"watchdog-{entry['strategy_id']}")
        watchdog.start()

    def _watchdog(self, entry: dict) -> None:
        """Kill the subprocess tree if it exceeds the RSS or CPU-time budget."""
        proc: subprocess.Popen = entry["proc"]
        try:
            ps = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            self._mark_exited(entry)
            return
        while proc.poll() is None:
            try:
                rss = ps.memory_info().rss
                cpu = ps.cpu_times()
                cpu_total = cpu.user + cpu.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            if rss > RSS_LIMIT_BYTES:
                log.warning("Watchdog: %s exceeded RSS limit (%.0f MB) — killing",
                            entry["agent_id"], rss / 1e6)
                self._kill_tree(proc.pid)
                break
            if cpu_total > CPU_LIMIT_SECONDS:
                log.warning("Watchdog: %s exceeded CPU-time limit (%.0fs) — killing",
                            entry["agent_id"], cpu_total)
                self._kill_tree(proc.pid)
                break
            time.sleep(WATCHDOG_POLL_S)
        self._mark_exited(entry)

    @staticmethod
    def _kill_tree(pid: int) -> None:
        """Terminate a process and all of its children (kill after grace)."""
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        procs = parent.children(recursive=True) + [parent]
        for p in procs:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(procs, timeout=3)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass

    def _mark_exited(self, entry: dict) -> None:
        """Flip a registry entry to exited and persist it."""
        if entry.get("status") != "exited":
            entry["status"] = "exited"
            self._write_meta(entry)
        handle = entry.get("log_handle")
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
            entry["log_handle"] = None

    # ------------------------------------------------------------------ #
    # docker backend                                                       #
    # ------------------------------------------------------------------ #
    def _launch_docker(self, entry: dict, strategy_id: str,
                       agent_id: str, name: str) -> None:
        """Start the strategy inside a resource-limited Docker container."""
        order_port = os.environ.get("ZMQ_ORDER_PORT", "5555")
        data_port = os.environ.get("ZMQ_DATA_PORT", "5556")
        container_name = f"chronos_strat_{strategy_id}"
        strategy_rel = f".strategies/{strategy_id}/strategy.py"
        inner_cmd = (
            "pip install --quiet pyzmq python-dotenv && "
            f"python -m runner.sdk {strategy_rel} --name {name} --agent-id {agent_id}"
        )
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", container_name,
            "--memory", "512m",
            "--cpus", "1",
            "--pids-limit", "64",
            "--network", "host",
            "-v", f"{_ROOT}:/chronos:ro",
            "-w", "/chronos",
            "-e", "ZMQ_HOST=host.docker.internal",
            "-e", f"ZMQ_ORDER_PORT={order_port}",
            "-e", f"ZMQ_DATA_PORT={data_port}",
            DOCKER_IMAGE,
            "sh", "-c", inner_cmd,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr.strip()}")
        entry.update(backend="docker", container_id=result.stdout.strip(),
                     container_name=container_name)

    def _docker_running(self, container_name: str) -> bool:
        """True iff a container with this exact name is currently running."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-q", "--filter", f"name=^{container_name}$"],
                capture_output=True, text=True, timeout=DOCKER_PROBE_TIMEOUT_S)
            return result.returncode == 0 and bool(result.stdout.strip())
        except (OSError, subprocess.TimeoutExpired):
            return False

    # ------------------------------------------------------------------ #
    # Public API — stop / list / logs                                      #
    # ------------------------------------------------------------------ #
    def stop(self, strategy_id: str) -> bool:
        """Stop a running strategy. Returns True if something was stopped.

        Sends nothing to the engine: for graceful stops the SDK's own signal
        handler issues CANCEL_ALL; for hard kills (docker stop timeout /
        watchdog kill) the engine's order TTL (PROTOCOL section 8.8) expires
        any resting quotes, so no stale depth lingers.
        """
        with self._lock:
            entry = self._registry.get(strategy_id)
        if entry is None:
            log.warning("stop(%s): unknown strategy", strategy_id)
            return False
        if entry.get("status") == "exited":
            return False

        stopped = False
        if entry.get("backend") == "docker" and entry.get("container_name"):
            try:
                result = subprocess.run(
                    ["docker", "stop", entry["container_name"]],
                    capture_output=True, text=True, timeout=30)
                stopped = result.returncode == 0
                if not stopped:
                    log.warning("docker stop %s failed: %s",
                                entry["container_name"], result.stderr.strip())
            except (OSError, subprocess.TimeoutExpired) as exc:
                log.warning("docker stop error: %s", exc)
        else:
            pid = entry.get("pid")
            proc: Optional[subprocess.Popen] = entry.get("proc")
            if proc is not None and proc.poll() is None:
                self._kill_tree(proc.pid)
                stopped = True
            elif pid and psutil.pid_exists(pid):
                self._kill_tree(pid)
                stopped = True

        self._mark_exited(entry)
        self._audit("stop", strategy_id, entry.get("name", "?"),
                    entry.get("backend", "?"))
        log.info("Stopped strategy %s (%s)", strategy_id, entry.get("agent_id"))
        return stopped

    def _refresh_status(self, entry: dict) -> None:
        """Recompute running/exited for one entry (cheap liveness check)."""
        if entry.get("status") == "exited":
            return
        backend = entry.get("backend")
        if backend == "docker":
            if not self._docker_running(entry.get("container_name", "")):
                self._mark_exited(entry)
            return
        proc: Optional[subprocess.Popen] = entry.get("proc")
        if proc is not None:
            if proc.poll() is not None:
                self._mark_exited(entry)
            return
        pid = entry.get("pid")  # reconstructed entry — only the pid survives
        if not (pid and psutil.pid_exists(pid)):
            self._mark_exited(entry)

    def list(self) -> list[dict]:
        """All known strategies with refreshed status, newest first."""
        with self._lock:
            entries = list(self._registry.values())
        for entry in entries:
            self._refresh_status(entry)
        entries.sort(key=lambda e: e.get("started_at", ""), reverse=True)
        return [
            {
                "strategy_id": e["strategy_id"],
                "agent_id": e["agent_id"],
                "name": e.get("name", ""),
                "status": e.get("status", "exited"),
                "started_at": e.get("started_at", ""),
            }
            for e in entries
        ]

    def logs(self, strategy_id: str, tail: int = 200) -> str:
        """Last ``tail`` lines of the strategy's stdout/stderr ('' if none)."""
        with self._lock:
            entry = self._registry.get(strategy_id)
        if entry is None:
            return ""
        if entry.get("backend") == "docker" and entry.get("container_name"):
            try:
                result = subprocess.run(
                    ["docker", "logs", "--tail", str(tail), entry["container_name"]],
                    capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return result.stdout + result.stderr
            except (OSError, subprocess.TimeoutExpired) as exc:
                log.warning("docker logs error: %s", exc)
            # fall through to run.log (e.g. container already removed by --rm)
        run_log = self.base_dir / strategy_id / "run.log"
        if not run_log.is_file():
            return ""
        try:
            with open(run_log, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError as exc:
            log.warning("run.log read error for %s: %s", strategy_id, exc)
            return ""
        return "".join(lines[-tail:])
