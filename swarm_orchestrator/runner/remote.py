"""HTTP client for the runner service (runner/service.py).

Gives the bridge the same methods as :class:`runner.docker_backend.DockerSandbox`
(``launch``, ``stop``, ``is_running``, ``logs``, ``remove``, ``dry_run``,
``available``), but every call goes over HTTP to the runner container. The bridge
therefore never needs the Docker socket: only the runner has it, and the runner
accepts nothing except "run this strategy in the sealed sandbox".

Standard library only (urllib), so the bridge image needs no extra package.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import List, Optional

from runner.docker_backend import _failed

log = logging.getLogger("runner.remote")
_PROBE_TTL_S = 5.0


class RunnerError(RuntimeError):
    """The runner refused or failed a request; ``str(exc)`` is safe to show a user."""


class RemoteSandbox:
    """Talks to ``RUNNER_URL`` with the shared secret in ``X-Runner-Token``."""

    def __init__(self, base_url: str, token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._probe: tuple = (0.0, False)

    def _call(self, method: str, path: str, body: Optional[dict] = None,
              timeout: float = 30.0) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers={"Content-Type": "application/json", "X-Runner-Token": self.token})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("detail", exc.reason)
            except Exception:                                  # noqa: BLE001
                detail = exc.reason
            raise RunnerError(f"runner refused the request ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise RunnerError(f"runner unreachable: {exc}") from exc

    def available(self) -> bool:
        """True if the runner answers and its sandbox image exists (cached 5 s)."""
        checked, ok = self._probe
        if time.monotonic() - checked < _PROBE_TTL_S:
            return ok
        try:
            ok = bool(self._call("GET", "/health", timeout=3).get("image_present"))
        except RunnerError:
            ok = False
        self._probe = (time.monotonic(), ok)
        return ok

    def launch(self, strategy_id: str, agent_id: str, name: str, code: str) -> str:
        # agent_id is derived by the runner from strategy_id; it is not accepted from the caller.
        return self._call("POST", "/launch", {"strategy_id": strategy_id, "name": name,
                                              "code": code}, timeout=90).get("container_id", "")

    def is_running(self, strategy_id: str) -> Optional[bool]:
        """True/False, or None if the runner cannot be reached (unknown, not "stopped")."""
        try:
            return bool(self._call("GET", f"/status/{strategy_id}", timeout=5).get("running"))
        except RunnerError:
            return None

    def stop(self, strategy_id: str) -> bool:
        try:
            return bool(self._call("POST", "/stop", {"strategy_id": strategy_id},
                                   timeout=40).get("stopped"))
        except RunnerError as exc:
            log.warning("runner stop failed: %s", exc)
            return False

    def logs(self, strategy_id: str, tail: int = 200) -> Optional[str]:
        try:
            return self._call("GET", f"/logs/{strategy_id}?tail={int(tail)}", timeout=15).get("logs")
        except RunnerError:
            return None

    def remove(self, strategy_id: str) -> None:
        try:
            self._call("POST", "/remove", {"strategy_id": strategy_id}, timeout=20)
        except RunnerError:
            pass

    def dry_run(self, code: str, ticks: List[dict], timeout: float = 10.0) -> dict:
        try:
            return self._call("POST", "/check", {"code": code, "ticks": ticks}, timeout=timeout + 15)
        except RunnerError as exc:
            return _failed(str(exc))
