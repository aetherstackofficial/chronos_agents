"""Feature-layer API router (PROTOCOL_V2.md §3-§6, §8).

One ``APIRouter`` shared by all four features plus the red-team lab control.
The integrator mounts it on the bridge with ``app.include_router(router)``;
shared singletons are read from ``request.app.state`` per the PROTOCOL_V2
integration preamble (``control``, ``latest_tick``, ``recent_trades``).
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from typing import Any, Dict, IO, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from features.copilot import generate_strategy
from features.montecarlo import run_fan
from features.narrator import narrate
from features.timemachine import (
    SCENARIOS,
    ensure_scenarios,
    get_scenario,
    scenario_csv_path,
)

log = logging.getLogger("chronos.features")

router = APIRouter()

# Where worker.py lives (the swarm_orchestrator directory).
_SWARM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REDTEAM_LOG = os.path.join(_SWARM_DIR, ".strategies", "redteam.log")
_REDTEAM_DEFAULT_COUNT = 3
_REDTEAM_MAX_COUNT = 10

# Module-global handle on the (single) red-team worker subprocess.
_redteam_proc: Optional[subprocess.Popen] = None
_redteam_log_fh: Optional[IO[bytes]] = None
_redteam_count: int = 0


# --------------------------------------------------------------------------- #
# Request bodies                                                               #
# --------------------------------------------------------------------------- #
class MonteCarloBody(BaseModel):
    """POST /api/montecarlo body (PROTOCOL_V2 §3).

    Range handling lives in :func:`features.montecarlo.run_fan`: out-of-range
    ``n_paths``/``horizon`` are CLAMPED, semantically bad input (empty symbol,
    non-positive price, malformed shock) raises ``ValueError`` → HTTP 400.
    """

    symbol: str = Field(max_length=64)
    price: float
    n_paths: int = 60
    horizon: int = 90
    seed_base: int = 1000
    shock: Optional[Dict[str, Any]] = None


class CopilotBody(BaseModel):
    """POST /api/copilot body (PROTOCOL_V2 §4)."""

    description: str = ""


class TimeMachineStartBody(BaseModel):
    """POST /api/timemachine/start body (PROTOCOL_V2 §5)."""

    scenario_id: str = Field(min_length=1)
    seed: Optional[int] = None


class RedTeamBody(BaseModel):
    """POST /api/lab/redteam body (PROTOCOL_V2 §8)."""

    count: Optional[int] = Field(default=None, ge=1, le=_REDTEAM_MAX_COUNT)


# --------------------------------------------------------------------------- #
# §3 Monte-Carlo fan                                                           #
# --------------------------------------------------------------------------- #
@router.post("/api/montecarlo")
async def api_montecarlo(body: MonteCarloBody) -> dict:
    """Run the real in-process Monte-Carlo fan (off the event loop)."""
    try:
        return await asyncio.to_thread(
            run_fan,
            body.symbol.strip(),
            body.price,
            n_paths=body.n_paths,
            horizon=body.horizon,
            seed_base=body.seed_base,
            shock=body.shock,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# §4 Strategy copilot                                                          #
# --------------------------------------------------------------------------- #
@router.post("/api/copilot")
async def api_copilot(body: CopilotBody) -> dict:
    """NL description → runnable UserStrategy code (Gemini or template)."""
    description = body.description.strip()
    if not description:
        raise HTTPException(status_code=400, detail="description must not be empty")
    return await generate_strategy(description, os.getenv("GEMINI_API_KEY"))


# --------------------------------------------------------------------------- #
# §5 Time Machine                                                              #
# --------------------------------------------------------------------------- #
@router.get("/api/timemachine/scenarios")
async def api_timemachine_scenarios() -> list:
    """List crisis presets, generating any missing scenario tapes first."""
    await asyncio.to_thread(ensure_scenarios)
    return SCENARIOS


@router.post("/api/timemachine/start")
async def api_timemachine_start(body: TimeMachineStartBody, request: Request) -> dict:
    """Start a crisis replay: forward REPLAY_START for the scenario tape."""
    await asyncio.to_thread(ensure_scenarios)
    scenario = get_scenario(body.scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404,
                            detail=f"unknown scenario: {body.scenario_id!r}")
    csv_path = scenario_csv_path(body.scenario_id)

    ack = await request.app.state.control({
        "msg": "REPLAY_START",
        "csv_path": csv_path,
        "symbol": scenario["symbol"],
        "sector": scenario["sector"],
        "seed": body.seed if body.seed is not None else 42,
    })
    if not ack:
        raise HTTPException(status_code=502,
                            detail="engine did not acknowledge REPLAY_START")
    log.info("timemachine: started %s (run_id=%s, bars=%s)",
             body.scenario_id, ack.get("run_id"), ack.get("bars"))
    return {
        "status": ack.get("status", "REPLAY_READY"),
        "run_id": ack.get("run_id"),
        "bars": ack.get("bars", scenario["bars"]),
        "scenario": scenario,
    }


# --------------------------------------------------------------------------- #
# §6 Narration                                                                 #
# --------------------------------------------------------------------------- #
@router.post("/api/narrate")
async def api_narrate(request: Request) -> dict:
    """Explain the current market state from the bridge's live view."""
    tick = request.app.state.latest_tick()
    trades = request.app.state.recent_trades(60)
    return await narrate(tick, trades, os.getenv("GEMINI_API_KEY"))


# --------------------------------------------------------------------------- #
# §8 Red-team lab (real subprocess; the worker role is wired by the integrator)
# --------------------------------------------------------------------------- #
def _redteam_running() -> bool:
    return _redteam_proc is not None and _redteam_proc.poll() is None


def _close_redteam_log() -> None:
    global _redteam_log_fh
    if _redteam_log_fh is not None:
        try:
            _redteam_log_fh.close()
        except OSError:
            pass
        _redteam_log_fh = None


@router.post("/api/lab/redteam")
async def api_redteam_start(body: RedTeamBody) -> dict:
    """Launch a red-team worker subprocess (``worker.py --role redteam``)."""
    global _redteam_proc, _redteam_log_fh, _redteam_count
    if _redteam_running():
        raise HTTPException(
            status_code=409,
            detail=f"red-team already running (pid {_redteam_proc.pid})")  # type: ignore[union-attr]

    count = body.count if body.count is not None else _REDTEAM_DEFAULT_COUNT
    os.makedirs(os.path.dirname(_REDTEAM_LOG), exist_ok=True)
    _close_redteam_log()
    _redteam_log_fh = open(_REDTEAM_LOG, "ab")
    try:
        _redteam_proc = subprocess.Popen(
            [sys.executable, "worker.py", "--role", "redteam",
             "--count", str(count)],
            cwd=_SWARM_DIR,
            stdout=_redteam_log_fh,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        _close_redteam_log()
        raise HTTPException(status_code=500,
                            detail=f"failed to launch red-team worker: {exc}") from exc
    _redteam_count = count
    log.info("[REDTEAM] launched worker pid=%d count=%d (log: %s)",
             _redteam_proc.pid, count, _REDTEAM_LOG)
    return {"status": "launched", "pid": _redteam_proc.pid, "count": count}


@router.post("/api/lab/redteam/stop")
async def api_redteam_stop() -> dict:
    """Terminate the red-team worker subprocess, if running."""
    global _redteam_proc
    if not _redteam_running():
        _close_redteam_log()
        _redteam_proc = None
        raise HTTPException(status_code=404, detail="no red-team process running")

    proc = _redteam_proc
    proc.terminate()
    try:
        await asyncio.to_thread(proc.wait, 5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        await asyncio.to_thread(proc.wait)
    log.info("[REDTEAM] stopped worker pid=%d", proc.pid)
    _redteam_proc = None
    _close_redteam_log()
    return {"status": "stopped"}
