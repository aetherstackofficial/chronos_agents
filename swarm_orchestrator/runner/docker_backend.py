"""Sealed Docker execution of strategy code.

One class, :class:`DockerSandbox`, used in two places:

* directly by ``SandboxManager`` when the bridge runs on a host that has Docker;
* by the **runner service** (runner/service.py), the only container that holds
  the Docker socket when the whole project runs under Docker Compose. The bridge
  then talks to the runner over HTTP and never touches the socket itself.

What "sealed" means here -- every strategy container gets:

* an image (``Dockerfile.sandbox``) that holds ONLY the SDK and pyzmq: no
  project files, no ``.env``, no API keys, no database;
* ``--cap-drop ALL``, ``no-new-privileges``, a read-only root filesystem with a
  small non-executable ``/tmp``, and a non-root user;
* hard limits: 512 MB memory (swap included), 1 CPU, 64 processes;
* a network that can reach the engine and nothing else. Under Compose that is
  an ``internal`` network (no internet either); a dry run uses ``--network none``.

The strategy code is passed as a base64 environment variable and written to
``/tmp`` by runner/sandbox_entry.py, so no host path is ever mounted.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import List, Optional

from runner.check import DRYRUN_SENTINEL as SENTINEL, MAX_CODE_BYTES

log = logging.getLogger("runner.docker")

DEFAULT_IMAGE = "chronos-sandbox:latest"
DRYRUN_TIMEOUT_S = 10.0   # container start ~0.5 s; the hooks themselves should take milliseconds
_ID_RE = re.compile(r"^[0-9a-f]{8}$")
_NAME_CLEAN = re.compile(r"[^A-Za-z0-9 _.\-]+")
_PROBE_TTL_S = 10.0


def clean_name(name: str) -> str:
    """Display name: printable characters only, at most 48 of them."""
    return (_NAME_CLEAN.sub("", name or "") or "strategy")[:48]


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


class DockerSandbox:
    """Launch, stop and inspect strategy containers through the ``docker`` CLI."""

    def __init__(self, image: Optional[str] = None, network: Optional[str] = None,
                 zmq_host: Optional[str] = None, order_port: Optional[str] = None,
                 data_port: Optional[str] = None, docker_bin: str = "docker") -> None:
        self.image = image or os.environ.get("SANDBOX_IMAGE", DEFAULT_IMAGE)
        # Under Compose: SANDBOX_NETWORK=chronos_strategies, SANDBOX_ZMQ_HOST=engine.
        # On a plain host: the default bridge network and host.docker.internal.
        self.network = network if network is not None else os.environ.get("SANDBOX_NETWORK") or None
        self.zmq_host = zmq_host or os.environ.get("SANDBOX_ZMQ_HOST") or "host.docker.internal"
        self.order_port = order_port or os.environ.get("ZMQ_ORDER_PORT", "5555")
        self.data_port = data_port or os.environ.get("ZMQ_DATA_PORT", "5556")
        self.docker = docker_bin
        self._probe: tuple = (0.0, False)

    # -- plumbing -----------------------------------------------------------
    def _run(self, args: List[str], timeout: float = 30.0, **kw) -> subprocess.CompletedProcess:
        return subprocess.run([self.docker, *args], capture_output=True, text=True,
                              timeout=timeout, **kw)

    @staticmethod
    def container_name(strategy_id: str) -> str:
        return f"chronos_strat_{strategy_id}"

    def available(self) -> bool:
        """True if the docker CLI answers and the sandbox image exists (cached 10 s)."""
        checked, ok = self._probe
        if time.monotonic() - checked < _PROBE_TTL_S:
            return ok
        try:
            ok = (self._run(["info"], timeout=5).returncode == 0
                  and self._run(["image", "inspect", self.image], timeout=5).returncode == 0)
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        self._probe = (time.monotonic(), ok)
        return ok

    # -- docker run arguments (pure, so they can be unit-tested) -------------
    def _hardening(self, memory: str = "512m") -> List[str]:
        return ["--memory", memory, "--memory-swap", memory, "--cpus", "1",
                "--pids-limit", "64", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
                "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
                "--user", "65534:65534"]

    def launch_args(self, strategy_id: str, agent_id: str, name: str, code: str) -> List[str]:
        args = ["run", "-d", "--name", self.container_name(strategy_id),
                "--label", "chronos.role=strategy", "--label", f"chronos.strategy_id={strategy_id}",
                "--stop-timeout", "10",
                "--log-opt", "max-size=1m", "--log-opt", "max-file=2",
                *self._hardening()]
        if self.network:
            args += ["--network", self.network]
        else:
            args += ["--add-host", "host.docker.internal:host-gateway"]
        args += ["-e", f"ZMQ_HOST={self.zmq_host}",
                 "-e", f"ZMQ_ORDER_PORT={self.order_port}",
                 "-e", f"ZMQ_DATA_PORT={self.data_port}",
                 "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONUNBUFFERED=1",
                 "-e", f"STRATEGY_B64={_b64(code)}",
                 "-e", f"STRATEGY_NAME={clean_name(name)}",
                 "-e", f"STRATEGY_AGENT_ID={agent_id}",
                 self.image, "python", "-m", "runner.sandbox_entry", "run"]
        return args

    def dryrun_args(self, container: str, code: str, ticks: List[dict]) -> List[str]:
        return ["run", "--rm", "--name", container, "--network", "none",
                *self._hardening(memory="256m"),
                "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-e", f"STRATEGY_B64={_b64(code)}",
                "-e", f"TICKS_B64={_b64(json.dumps(ticks))}",
                self.image, "python", "-m", "runner.sandbox_entry", "dryrun"]

    # -- strategy lifecycle -------------------------------------------------
    def launch(self, strategy_id: str, agent_id: str, name: str, code: str) -> str:
        """Start the strategy in a sealed container; returns the container id."""
        if not _ID_RE.match(strategy_id):
            raise ValueError("bad strategy id")
        if len(code.encode("utf-8")) > MAX_CODE_BYTES:
            raise ValueError("strategy code is too large")
        result = self._run(self.launch_args(strategy_id, agent_id, name, code), timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr.strip()}")
        return result.stdout.strip()

    def is_running(self, strategy_id: str) -> Optional[bool]:
        """True/False, or None if docker cannot be asked (unknown, not "stopped")."""
        try:
            result = self._run(["ps", "-q", "--filter",
                                f"name=^{self.container_name(strategy_id)}$"], timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        return bool(result.stdout.strip())

    def stop(self, strategy_id: str) -> bool:
        try:
            return self._run(["stop", self.container_name(strategy_id)], timeout=30).returncode == 0
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("docker stop error: %s", exc)
            return False

    def logs(self, strategy_id: str, tail: int = 200) -> Optional[str]:
        """Container output, or None if the container no longer exists."""
        try:
            result = self._run(["logs", "--tail", str(int(tail)),
                                self.container_name(strategy_id)], timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout + result.stderr if result.returncode == 0 else None

    def remove(self, strategy_id: str) -> None:
        try:
            self._run(["rm", "-f", self.container_name(strategy_id)], timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def cleanup_stale(self) -> int:
        """Remove every strategy container, running or stopped; returns how many.

        Run once when the runner starts. The bridge's record of what it launched
        does not survive a restart, so a leftover strategy would keep trading
        with nothing able to stop it.
        """
        try:
            ids = self._run(["ps", "-aq", "--filter", "label=chronos.role=strategy"],
                            timeout=10).stdout.split()
            if ids:
                self._run(["rm", "-f", *ids], timeout=60)
            return len(ids)
        except (OSError, subprocess.TimeoutExpired):
            return 0

    # -- dry run ------------------------------------------------------------
    def dry_run(self, code: str, ticks: List[dict], timeout: float = DRYRUN_TIMEOUT_S) -> dict:
        """Run the strategy's hooks once in a throwaway container with no network.

        Returns the report from runner/dryrun.py. A timeout or crash becomes an
        ordinary error entry in the same format, so the caller handles one shape.
        """
        container = f"chronos_dry_{uuid.uuid4().hex[:8]}"
        try:
            result = self._run(self.dryrun_args(container, code, ticks), timeout=timeout)
        except subprocess.TimeoutExpired:
            self._run(["rm", "-f", container], timeout=15)   # kills a runaway loop
            return _failed(f"the strategy did not finish within {timeout:.0f} s "
                           "(an infinite loop, or a very slow on_tick)")
        except OSError as exc:
            return _failed(f"could not start the sandbox: {exc}")
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(SENTINEL):
                try:
                    return json.loads(line[len(SENTINEL):])
                except ValueError:
                    break
        tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        return _failed("the dry run ended without a result "
                       f"(exit code {result.returncode}). {' | '.join(tail)}"[:400])

    # -- image --------------------------------------------------------------
    def ensure_image(self, build_context: Optional[Path] = None, timeout: float = 600.0) -> None:
        """Build the sandbox image if it is missing and a Dockerfile.sandbox is at hand."""
        if self._run(["image", "inspect", self.image], timeout=10).returncode == 0:
            return
        dockerfile = (build_context / "Dockerfile.sandbox") if build_context else None
        if not dockerfile or not dockerfile.is_file():
            raise RuntimeError(
                f"sandbox image {self.image!r} is missing; build it with: "
                f"docker build -f Dockerfile.sandbox -t {self.image} .")
        log.info("building sandbox image %s ...", self.image)
        built = self._run(["build", "-f", str(dockerfile), "-t", self.image, str(build_context)],
                          timeout=timeout)
        if built.returncode != 0:
            raise RuntimeError(f"building {self.image} failed: {built.stderr.strip()[-300:]}")
        self._probe = (0.0, False)


def _failed(message: str) -> dict:
    return {"ok": False, "errors": [{"stage": "dry-run", "line": None, "message": message}],
            "warnings": [], "orders": [], "ticks_tested": 0, "output": ""}
