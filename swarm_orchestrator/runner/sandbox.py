"""Strategy sandbox manager (PROTOCOL.md section 12.2) -- imported by the bridge.

Checks and launches user strategy code. Strategy code is written by other
people, so it runs only where it cannot reach the project's files or secrets.

Backends, in order of preference
--------------------------------
* **runner** -- ``RUNNER_URL`` is set (Docker Compose). The bridge asks the
  runner service (runner/service.py), the only container with the Docker
  socket, to run the strategy in a sealed container. The bridge itself stays
  unprivileged.
* **docker** -- the bridge runs on a host with the ``docker`` CLI and the
  ``chronos-sandbox`` image; it starts the sealed container itself.
* **subprocess** -- a plain process with a psutil watchdog. It is a resource
  guard, **not** a security boundary: the strategy runs as the bridge's own OS
  user. It is OFF by default and only used when the operator opts in with
  ``CHRONOS_ALLOW_UNSAFE_SANDBOX=1`` (local development without Docker). The old
  behaviour of silently falling back to it is gone: with no sealed backend and
  no opt-in, ``launch()`` raises :class:`SandboxUnavailable`.

Every launch first runs :func:`runner.check.static_check` and a dry run
(runner/dryrun.py, executed inside the same kind of sandbox), so a typo is
reported in a second with its line number instead of after the replay is half
over. :meth:`SandboxManager.check` exposes the same checks without launching.

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
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Optional

import psutil

from runner.check import DRYRUN_SENTINEL as SENTINEL, build_sample_ticks, static_check
from runner.docker_backend import DockerSandbox
from runner.remote import RemoteSandbox

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
LOCAL_DRYRUN_TIMEOUT_S: float = 10.0
# Environment a strategy process may see in the opt-in unsafe mode. Everything
# else (API keys, tokens, the bridge's settings) is withheld.
_SAFE_ENV_KEYS = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE",
                  "ZMQ_HOST", "ZMQ_ORDER_PORT", "ZMQ_DATA_PORT")

NO_SANDBOX_MESSAGE = (
    "no sealed sandbox is available, so strategies cannot be run. Start the stack with "
    "docker compose (it includes the runner service), or build the image with "
    "`docker build -f Dockerfile.sandbox -t chronos-sandbox .` on a host with Docker. "
    "For local development only, set CHRONOS_ALLOW_UNSAFE_SANDBOX=1 to accept running "
    "strategies as a plain, unsealed process.")


_AUTO = object()     # "choose the backend from the environment" (None means: no backend)


class SandboxUnavailable(RuntimeError):
    """No sealed backend can run the strategy (and unsafe mode was not enabled)."""


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

    def __init__(self, base_dir: str = ".strategies", sealed=_AUTO,
                 allow_unsafe: Optional[bool] = None) -> None:
        """Create the manager; ``base_dir`` resolves relative to swarm_orchestrator.

        ``sealed`` and ``allow_unsafe`` exist for tests; normally they come from
        the environment (``RUNNER_URL``/``RUNNER_TOKEN``, docker on PATH,
        ``CHRONOS_ALLOW_UNSAFE_SANDBOX``). Pass ``sealed=None`` to mean "no sealed
        backend" (auto-detection is skipped).
        """
        base = Path(base_dir)
        self.base_dir: Path = base if base.is_absolute() else (_ROOT / base)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # strategy_id -> {meta fields, "proc": Popen|None, "log_handle": IO|None}
        self._registry: dict[str, dict] = {}
        if allow_unsafe is None:
            allow_unsafe = os.environ.get("CHRONOS_ALLOW_UNSAFE_SANDBOX", "").strip().lower() \
                in ("1", "true", "yes")
        self.allow_unsafe: bool = allow_unsafe
        self._sealed = self._choose_sealed_backend() if sealed is _AUTO else sealed
        self._image_checked = False
        if self._sealed is None:
            log.warning("No Docker and no RUNNER_URL: strategies %s.",
                        "will run UNSEALED (CHRONOS_ALLOW_UNSAFE_SANDBOX is set)"
                        if self.allow_unsafe else "cannot be run (sealed sandbox required)")
        self._reconstruct()

    # ------------------------------------------------------------------ #
    # Backend selection                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _choose_sealed_backend():
        """The runner service if configured, else local Docker, else None."""
        runner_url = os.environ.get("RUNNER_URL", "").strip()
        if runner_url:
            return RemoteSandbox(runner_url, os.environ.get("RUNNER_TOKEN", ""))
        if shutil.which("docker"):
            return DockerSandbox()
        return None

    def _sealed_ready(self) -> bool:
        """True if a sealed backend can run strategies right now."""
        sealed = self._sealed
        if sealed is None:
            return False
        if isinstance(sealed, DockerSandbox) and not self._image_checked:
            self._image_checked = True       # try to build the image once, if we can
            try:
                sealed.ensure_image(_ROOT.parent)
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                log.warning("%s", exc)
        return sealed.available()

    def mode(self) -> str:
        """``runner``, ``docker``, ``unsafe-subprocess`` or ``unavailable``."""
        if self._sealed_ready():
            return "runner" if isinstance(self._sealed, RemoteSandbox) else "docker"
        return "unsafe-subprocess" if self.allow_unsafe else "unavailable"

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
    def check(self, code: str) -> dict:
        """Static check + dry run, without launching anything.

        Returns ``{"ok", "errors", "warnings", "orders", "ticks_tested",
        "output", "backend", "dry_run", "duration_ms"}``. ``ok`` is about the
        code only; ``backend``/``dry_run`` say how (or whether) it was exercised.
        """
        started = time.perf_counter()
        static = static_check(code)
        report: dict = {"ok": False, "errors": static["errors"], "warnings": static["warnings"],
                        "orders": [], "ticks_tested": 0, "output": ""}
        if static["errors"]:
            report.update(dry_run="skipped: fix the errors first", backend=self.mode())
        elif self._sealed_ready():
            dry = self._sealed.dry_run(code, build_sample_ticks())
            dry["warnings"] = self._merge_warnings(static["warnings"], dry.get("warnings", []))
            report = dict(dry, dry_run="sealed sandbox", backend=self.mode())
        elif self.allow_unsafe:
            dry = self._dry_run_local(code)
            dry["warnings"] = self._merge_warnings(static["warnings"], dry.get("warnings", []))
            report = dict(dry, dry_run="UNSEALED local process (unsafe mode)", backend=self.mode())
        else:
            report.update(
                ok=True, backend=self.mode(),
                dry_run="skipped: no sealed sandbox is available",
                warnings=static["warnings"] + [{"stage": "dry-run", "line": None,
                                                "message": NO_SANDBOX_MESSAGE}])
        report["duration_ms"] = int((time.perf_counter() - started) * 1000)
        return report

    @staticmethod
    def _merge_warnings(*groups: list) -> list:
        """Concatenate warning lists, dropping exact repeats (the runner already
        adds the static warnings to its own report)."""
        seen, merged = set(), []
        for item in (w for group in groups for w in group):
            key = (item.get("stage"), item.get("line"), item.get("message"))
            if key not in seen:
                seen.add(key)
                merged.append(item)
        return merged

    @staticmethod
    def _format_errors(errors: list) -> str:
        parts = [f"line {e['line']}: {e['message']}" if e.get("line") else e["message"]
                 for e in errors]
        return "strategy check failed: " + "; ".join(parts)

    def _dry_run_local(self, code: str) -> dict:
        """Dry run in a plain subprocess. ONLY for the opt-in unsafe mode."""
        env = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}
        with tempfile.TemporaryDirectory(prefix="chronos_dry_") as tmp:
            strategy_file = Path(tmp) / "strategy.py"
            ticks_file = Path(tmp) / "ticks.json"
            strategy_file.write_text(code, encoding="utf-8")
            ticks_file.write_text(json.dumps(build_sample_ticks()), encoding="utf-8")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "runner.dryrun", str(strategy_file), str(ticks_file)],
                    capture_output=True, text=True, timeout=LOCAL_DRYRUN_TIMEOUT_S,
                    cwd=str(_ROOT), env=env)
            except subprocess.TimeoutExpired:
                return {"ok": False, "warnings": [], "orders": [], "ticks_tested": 0, "output": "",
                        "errors": [{"stage": "dry-run", "line": None, "message":
                                    f"the strategy did not finish within {LOCAL_DRYRUN_TIMEOUT_S:.0f} s "
                                    "(an infinite loop, or a very slow on_tick)"}]}
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(SENTINEL):
                return json.loads(line[len(SENTINEL):])
        tail = " | ".join((result.stderr or result.stdout).strip().splitlines()[-3:])
        return {"ok": False, "warnings": [], "orders": [], "ticks_tested": 0, "output": "",
                "errors": [{"stage": "dry-run", "line": None,
                            "message": f"the dry run ended without a result: {tail}"[:400]}]}

    def launch(self, name: str, code: str) -> dict:
        """Check ``code`` and start it in a sealed sandbox.

        Raises:
            ValueError: the code fails the static check or the dry run; the
                message lists each problem with its line number.
            SandboxUnavailable: no sealed backend can run it (and unsafe mode is
                not enabled), or the sealed launch itself failed.

        Returns:
            ``{"strategy_id": <8 hex>, "agent_id": "STRAT_<id>", "backend": ...}``
        """
        report = self.check(code)
        if report["errors"]:
            raise ValueError(self._format_errors(report["errors"]))
        sealed = self._sealed_ready()
        if not sealed and not self.allow_unsafe:
            raise SandboxUnavailable(NO_SANDBOX_MESSAGE)

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

        try:
            if sealed:
                try:
                    container_id = self._sealed.launch(strategy_id, agent_id, name, code)
                except Exception as exc:                      # noqa: BLE001
                    if not self.allow_unsafe:
                        raise SandboxUnavailable(
                            f"could not start the sealed sandbox: {exc}") from exc
                    log.warning("Sealed launch failed (%s); unsafe mode is on, so running "
                                "UNSEALED.", exc)
                    self._launch_subprocess(entry, strategy_file, agent_id, name)
                else:
                    entry.update(backend=self.mode(), container_id=container_id,
                                 container_name=f"chronos_strat_{strategy_id}")
            else:
                self._launch_subprocess(entry, strategy_file, agent_id, name)
        except Exception:
            shutil.rmtree(strategy_dir, ignore_errors=True)   # leave nothing behind
            raise

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
        # Withhold the bridge's secrets (API keys, tokens): the strategy sees only
        # what it needs to reach the engine.
        env = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}
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
    # sealed backend (runner service or local docker)                      #
    # ------------------------------------------------------------------ #
    def _finalize_sealed(self, entry: dict) -> None:
        """Keep the container's output in run.log, then remove the container."""
        strategy_id = entry["strategy_id"]
        if self._sealed is None:
            return
        text = self._sealed.logs(strategy_id, 2000)
        if text:
            try:
                (self.base_dir / strategy_id / "run.log").write_text(text, encoding="utf-8")
            except OSError as exc:
                log.warning("run.log write failed for %s: %s", strategy_id, exc)
        self._sealed.remove(strategy_id)

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
        if entry.get("backend") in ("docker", "runner") and self._sealed is not None:
            stopped = self._sealed.stop(strategy_id)
            if not stopped:
                log.warning("sealed stop of %s failed", strategy_id)
            self._finalize_sealed(entry)
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
        if backend in ("docker", "runner"):
            # is_running() is None when the backend cannot be reached: then we do
            # not know, so we leave the status alone (a network blip must not
            # mark a live strategy as exited).
            if self._sealed is not None and self._sealed.is_running(entry["strategy_id"]) is False:
                self._finalize_sealed(entry)
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
        if entry.get("backend") in ("docker", "runner") and self._sealed is not None:
            text = self._sealed.logs(strategy_id, tail)
            if text is not None:
                return text
            # fall through to run.log (the container was removed after it stopped)
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
