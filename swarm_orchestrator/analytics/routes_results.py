"""Results / reports API router (PROTOCOL_V2 section 2).

Reads the shared ``analytics.results.ResultsStore`` singleton from
``request.app.state.results`` (wired by the integrator via
``app.include_router(analytics.routes_results.router)``).

CSV / PDF endpoints return downloads with
``Content-Disposition: attachment; filename="<safe name>_<sid>.csv|pdf"``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from analytics.reports import outcomes_csv, strategy_pdf, trades_csv

logger = logging.getLogger("chronos.analytics")

router = APIRouter()

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _results(request: Request) -> Any:
    """The shared ``ResultsStore`` singleton, or HTTP 503 if not wired yet."""
    results = getattr(request.app.state, "results", None)
    if results is None:
        raise HTTPException(status_code=503, detail="results store not initialised")
    return results


def _detail_or_404(results: Any, strategy_id: str) -> dict:
    """Full strategy detail, mapping an unknown id to HTTP 404."""
    try:
        return results.strategy_detail(strategy_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown strategy_id: {strategy_id}"
        ) from exc


def _safe_filename(name: str) -> str:
    """Reduce a strategy name to a filesystem/header-safe token."""
    token = _SAFE_NAME_RE.sub("_", (name or "").strip()).strip("._")
    return token or "strategy"


def _attachment(payload: bytes, media_type: str, filename: str) -> Response:
    """A download response with the proper Content-Disposition header."""
    return Response(
        content=payload,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --------------------------------------------------------------------------- #
# Endpoints                                                                    #
# --------------------------------------------------------------------------- #
@router.get("/api/results")
async def list_results(request: Request) -> list[dict]:
    """Summary rows for every tracked strategy."""
    return _results(request).list_strategies()


@router.get("/api/results/{strategy_id}")
async def strategy_detail(strategy_id: str, request: Request) -> dict:
    """Full detail (per-day outcomes + trades) for one strategy. 404 if unknown."""
    return _detail_or_404(_results(request), strategy_id)


@router.get("/api/results/{strategy_id}/trades")
async def strategy_trades(
    strategy_id: str, request: Request, day: Optional[int] = None
) -> dict:
    """Trade log for one strategy, optionally filtered to ``?day=N``."""
    results = _results(request)
    try:
        rows = results.strategy_trades(strategy_id, day)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown strategy_id: {strategy_id}"
        ) from exc
    return {"trades": rows}


@router.get("/api/results/{strategy_id}/trades.csv")
async def strategy_trades_csv(
    strategy_id: str, request: Request, day: Optional[int] = None
) -> Response:
    """Download the trade log as an RFC-4180 CSV attachment."""
    results = _results(request)
    detail = _detail_or_404(results, strategy_id)
    rows = results.strategy_trades(strategy_id, day)
    payload = await asyncio.to_thread(
        trades_csv, rows, {"name": detail["name"], "strategy_id": strategy_id}
    )
    filename = f"{_safe_filename(detail['name'])}_{strategy_id}.csv"
    return _attachment(payload, "text/csv", filename)


@router.get("/api/results/{strategy_id}/outcomes.csv")
async def strategy_outcomes_csv(strategy_id: str, request: Request) -> Response:
    """Download the per-day outcome summary as a CSV attachment."""
    results = _results(request)
    detail = _detail_or_404(results, strategy_id)
    payload = await asyncio.to_thread(
        outcomes_csv,
        detail.get("days") or [],
        {"name": detail["name"], "strategy_id": strategy_id},
    )
    filename = f"{_safe_filename(detail['name'])}_{strategy_id}_outcomes.csv"
    return _attachment(payload, "text/csv", filename)


@router.get("/api/results/{strategy_id}/report.pdf")
async def strategy_report_pdf(strategy_id: str, request: Request) -> Response:
    """Download the full strategy report as a PDF attachment."""
    results = _results(request)
    detail = _detail_or_404(results, strategy_id)
    detail["generated_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    payload = await asyncio.to_thread(strategy_pdf, detail)
    filename = f"{_safe_filename(detail['name'])}_{strategy_id}.pdf"
    return _attachment(payload, "application/pdf", filename)
