"""The runner service: the ONLY container that holds the Docker socket.

Why it exists. Strategy code is written by strangers and must run sealed. Running
Docker needs the Docker socket, and whoever holds the socket controls the whole
machine. Giving the socket to the bridge (the public web server) would make any
bug in the bridge a full takeover. So the bridge stays unprivileged and asks this
small service instead. It is deliberately narrow:

* it offers six operations, all about strategy containers (launch, stop, remove,
  status, logs, check);
* it builds the ``docker run`` command itself, from a fixed hardening recipe
  (runner/docker_backend.py), so a caller cannot ask for a privileged container,
  a host mount or another image;
* it derives the agent id from the strategy id (``STRAT_<id>``) and re-runs the
  static check, so it does not trust the bridge's word that code is valid;
* it is reachable only on the internal Compose network (no published port) and
  requires a shared secret in ``X-Runner-Token``. Sandbox containers sit on a
  different network and cannot reach it at all.

Run: ``uvicorn runner.service:app --host 0.0.0.0 --port 8100``
"""
from __future__ import annotations

import hmac
import logging
import os
import re
import uuid
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from runner.check import MAX_CODE_BYTES, static_check
from runner.docker_backend import DockerSandbox

log = logging.getLogger("runner.service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [RUNNER] - %(message)s")

_ID_RE = re.compile(r"^[0-9a-f]{8}$")
MAX_TICKS = 5
MAX_TICKS_BYTES = 64 * 1024

app = FastAPI(title="Chronos strategy runner", docs_url=None, redoc_url=None, openapi_url=None)
sandbox = DockerSandbox()


def _auth(x_runner_token: str = Header(default="")) -> None:
    expected = os.environ.get("RUNNER_TOKEN", "")
    if not expected or not hmac.compare_digest(x_runner_token, expected):
        raise HTTPException(status_code=401, detail="bad or missing runner token")


def _strategy_id(value: str) -> str:
    if not _ID_RE.match(value or ""):
        raise HTTPException(status_code=400, detail="strategy_id must be 8 hex characters")
    return value


class LaunchReq(BaseModel):
    strategy_id: str
    name: str = "strategy"
    code: str


class IdReq(BaseModel):
    strategy_id: str


class CheckReq(BaseModel):
    code: str
    ticks: List[dict]


@app.on_event("startup")
def _startup() -> None:
    if not os.environ.get("RUNNER_TOKEN"):
        log.error("RUNNER_TOKEN is not set: every request will be refused")
    try:
        sandbox.ensure_image(Path(os.environ.get("SANDBOX_BUILD_CONTEXT", "/nonexistent")))
    except RuntimeError as exc:
        log.error("%s", exc)
    removed = sandbox.cleanup_stale()
    if removed:
        log.info("removed %d strategy container(s) left from a previous session", removed)


@app.get("/health", dependencies=[Depends(_auth)])
def health() -> dict:
    return {"status": "ok", "image": sandbox.image, "image_present": sandbox.available()}


@app.post("/launch", dependencies=[Depends(_auth)])
def launch(req: LaunchReq) -> dict:
    strategy_id = _strategy_id(req.strategy_id)
    problems = static_check(req.code)["errors"]
    if problems:
        raise HTTPException(status_code=400, detail=problems[0]["message"])
    try:
        container_id = sandbox.launch(strategy_id, f"STRAT_{strategy_id}", req.name, req.code)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    log.info("launched %s", strategy_id)
    return {"container_id": container_id, "agent_id": f"STRAT_{strategy_id}"}


@app.post("/stop", dependencies=[Depends(_auth)])
def stop(req: IdReq) -> dict:
    return {"stopped": sandbox.stop(_strategy_id(req.strategy_id))}


@app.post("/remove", dependencies=[Depends(_auth)])
def remove(req: IdReq) -> dict:
    sandbox.remove(_strategy_id(req.strategy_id))
    return {"removed": True}


@app.get("/status/{strategy_id}", dependencies=[Depends(_auth)])
def status(strategy_id: str) -> dict:
    return {"running": sandbox.is_running(_strategy_id(strategy_id))}


@app.get("/logs/{strategy_id}", dependencies=[Depends(_auth)])
def logs(strategy_id: str, tail: int = Query(200, ge=1, le=2000)) -> dict:
    return {"logs": sandbox.logs(_strategy_id(strategy_id), tail)}


@app.post("/check", dependencies=[Depends(_auth)])
def check(req: CheckReq) -> dict:
    """Dry run in a throwaway container with no network. The caller supplies the ticks."""
    if len(req.code.encode("utf-8")) > MAX_CODE_BYTES:
        raise HTTPException(status_code=400, detail="strategy code is too large")
    if len(req.ticks) > MAX_TICKS or len(str(req.ticks)) > MAX_TICKS_BYTES:
        raise HTTPException(status_code=400, detail="too many or too large sample ticks")
    problems = static_check(req.code)
    if problems["errors"]:
        return {"ok": False, "errors": problems["errors"], "warnings": problems["warnings"],
                "orders": [], "ticks_tested": 0, "output": ""}
    report = sandbox.dry_run(req.code, req.ticks)
    report["warnings"] = problems["warnings"] + report.get("warnings", [])
    return report
