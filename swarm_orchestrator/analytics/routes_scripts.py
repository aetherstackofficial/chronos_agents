"""Script-library API router (PROTOCOL_V2 section 1).

Endpoints operate on shared singletons the integrator places on
``request.app.state`` (see PROTOCOL_V2 "Integration model"):

* ``app.state.scripts`` — ``runner.script_store.ScriptStore``
* ``app.state.sandbox`` — ``runner.sandbox.SandboxManager`` (may be absent)
* ``app.state.results`` — ``analytics.results.ResultsStore``

The integrator wires this router into ``bridge.py`` with
``app.include_router(analytics.routes_scripts.router)``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger("chronos.analytics")

router = APIRouter()


# --------------------------------------------------------------------------- #
# Request models                                                               #
# --------------------------------------------------------------------------- #
class ScriptCreateReq(BaseModel):
    """Body of ``POST /api/scripts``."""

    name: str
    code: str
    language: str = "python"


class ScriptUpdateReq(BaseModel):
    """Body of ``PUT /api/scripts/{script_id}`` (both fields optional)."""

    name: Optional[str] = None
    code: Optional[str] = None


# --------------------------------------------------------------------------- #
# Shared-state accessors                                                       #
# --------------------------------------------------------------------------- #
def _scripts(request: Request) -> Any:
    """The shared ``ScriptStore`` singleton, or HTTP 503 if not wired yet."""
    store = getattr(request.app.state, "scripts", None)
    if store is None:
        raise HTTPException(status_code=503, detail="script store not initialised")
    return store


def _results(request: Request) -> Any:
    """The shared ``ResultsStore`` singleton, or HTTP 503 if not wired yet."""
    results = getattr(request.app.state, "results", None)
    if results is None:
        raise HTTPException(status_code=503, detail="results store not initialised")
    return results


def _sandbox(request: Request) -> Any:
    """The shared ``SandboxManager``, or HTTP 503 when unavailable.

    The sandbox is optional infrastructure: it may be missing from
    ``app.state`` entirely, be ``None``, or raise on access (e.g. a lazy
    import of ``runner.sandbox`` failing). All three map to 503.
    """
    try:
        sandbox = getattr(request.app.state, "sandbox", None)
    except Exception as exc:  # noqa: BLE001 — lazy accessor may raise anything
        raise HTTPException(
            status_code=503, detail=f"strategy sandbox unavailable: {exc}"
        ) from exc
    if sandbox is None:
        raise HTTPException(status_code=503, detail="strategy sandbox unavailable")
    return sandbox


# --------------------------------------------------------------------------- #
# Endpoints                                                                    #
# --------------------------------------------------------------------------- #
@router.post("/api/scripts")
async def create_script(req: ScriptCreateReq, request: Request) -> dict:
    """Save a new strategy script to the library. 400 on empty code/name."""
    store = _scripts(request)
    try:
        return await asyncio.to_thread(store.save, req.name, req.code, req.language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/scripts")
async def list_scripts(request: Request) -> list[dict]:
    """List saved scripts (metadata only, newest first)."""
    store = _scripts(request)
    return await asyncio.to_thread(store.list)


@router.get("/api/scripts/{script_id}")
async def get_script(script_id: str, request: Request) -> dict:
    """Fetch one script including its code. 404 if unknown."""
    store = _scripts(request)
    try:
        return await asyncio.to_thread(store.get, script_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown script_id: {script_id}"
        ) from exc


@router.put("/api/scripts/{script_id}")
async def update_script(script_id: str, req: ScriptUpdateReq, request: Request) -> dict:
    """Update a script's name and/or code. Returns the refreshed metadata."""
    store = _scripts(request)
    try:
        return await asyncio.to_thread(store.update, script_id, req.name, req.code)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown script_id: {script_id}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/api/scripts/{script_id}")
async def delete_script(script_id: str, request: Request) -> dict:
    """Delete a script from the library. 404 if unknown."""
    store = _scripts(request)
    deleted = await asyncio.to_thread(store.delete, script_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"unknown script_id: {script_id}")
    return {"status": "deleted"}


@router.post("/api/scripts/{script_id}/run")
async def run_script(script_id: str, request: Request) -> dict:
    """Launch a saved script in the strategy sandbox and track its results.

    Loads the code from the library, launches it via ``SandboxManager`` and
    registers the run with the results store so the strategy's trades and
    outcomes are captured from the live market-data stream.
    """
    store = _scripts(request)
    results = _results(request)     # fail fast, before spawning anything
    sandbox = _sandbox(request)
    try:
        script = await asyncio.to_thread(store.get, script_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown script_id: {script_id}"
        ) from exc
    try:
        launched = await asyncio.to_thread(sandbox.launch, script["name"], script["code"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — docker/subprocess failure
        logger.exception("sandbox launch failed for script %s", script_id)
        raise HTTPException(
            status_code=503, detail=f"strategy launch failed: {exc}"
        ) from exc
    results.attach_strategy(
        launched["strategy_id"], launched["agent_id"], script["name"], script_id
    )
    logger.info(
        "script %s launched as strategy %s (agent %s, backend %s)",
        script_id,
        launched["strategy_id"],
        launched["agent_id"],
        launched.get("backend"),
    )
    return {
        "strategy_id": launched["strategy_id"],
        "agent_id": launched["agent_id"],
        "backend": launched.get("backend"),
        "script_id": script_id,
    }
