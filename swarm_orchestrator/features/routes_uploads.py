"""Bring-your-own replay data: CSV upload and Yahoo Finance download.

* ``POST /api/replay/upload`` -- upload your own 1-minute bar CSV.
* ``POST /api/replay/yahoo``  -- download 15-minute bars from Yahoo Finance
  (features/yahoo.py), saved as a CSV in the same folder.

Both end the same way: a CSV on disk, checked, and a ``csv_path`` to pass to
``POST /api/replay/start``. The replay always reads that saved file, never
Yahoo directly.

Lets a user upload their own 1-minute bar CSV and replay it through the real
book. The file is validated with the SAME checks the engine runs at
REPLAY_START (:meth:`engine.replay.HistoricalReplayer.load_csv`), so a file
that uploads cleanly is guaranteed to load in the engine -- there is no second,
drifting copy of the column rules here.

Where files go
--------------
Uploads are written to ``CHRONOS_UPLOAD_DIR``. Under Docker Compose that is
``/data/uploads`` on the ``chronos_data`` volume, which BOTH the bridge and the
engine mount: the bridge writes the file, and the engine reads it from the
same absolute path when the returned ``csv_path`` is passed to
``POST /api/replay/start``. Locally (no Docker) it defaults to
``swarm_orchestrator/data/uploads``, which the engine sees directly.

Safety
------
* Size cap ``CHRONOS_UPLOAD_MAX_MB`` (default 10 MB), enforced twice: early
  from the Content-Length header, and again while copying, so a client that
  lies about its length is still cut off.
* The client's filename is never used as a path: it is reduced to a safe stem
  and given a random suffix, so uploads cannot escape the directory
  (``../../x.csv``) or overwrite each other.
* The file is written to a ``.part`` temp name and only renamed into place
  after validation passes; a rejected upload leaves nothing behind.
"""
from __future__ import annotations

import asyncio
import csv
import logging
import os
import re
import uuid

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from engine.replay import HistoricalReplayer
from features.yahoo import YahooError, fetch_bars

log = logging.getLogger("chronos.uploads")

router = APIRouter()

# Package parent = swarm_orchestrator, so the local default resolves to the
# same place regardless of the bridge's cwd (same convention as timemachine.py).
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MAX_UPLOAD_MB = 10.0

# Copy in 1 MB chunks: bounded memory regardless of file size.
_CHUNK_BYTES = 1024 * 1024
# A multipart body is a little larger than the file inside it (boundaries and
# part headers), so the early Content-Length check allows this much slack.
_MULTIPART_SLACK_BYTES = 64 * 1024
# Anything outside this set is replaced, e.g. "My Data (v2).csv" -> "My_Data_v2".
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


# Settings are read per request, not at import: bridge.py calls load_dotenv()
# AFTER importing its routers, so import-time reads would miss values in .env.
def _upload_dir() -> str:
    return os.path.abspath(os.getenv("CHRONOS_UPLOAD_DIR")
                           or os.path.join(_BASE_DIR, "data", "uploads"))


def _max_upload_mb() -> float:
    return float(os.getenv("CHRONOS_UPLOAD_MAX_MB") or DEFAULT_MAX_UPLOAD_MB)


def _too_large(limit_mb: float) -> HTTPException:
    return HTTPException(
        status_code=413,
        detail=f"file too large: the limit is {limit_mb:g} MB")


def _safe_stem(filename: str) -> str:
    """Reduce a client filename to a safe stem: no directories, no odd chars."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = _UNSAFE_NAME_CHARS.sub("_", stem).strip("_")
    return stem[:40] or "upload"


def _validate(path: str, shown_name: str) -> int:
    """Run the engine's own CSV checks on ``path``; return the bar count.

    Raises HTTPException(400) with a readable message. load_csv() prefixes
    its errors with the file path, so the server-side temp path is swapped for
    the name the user uploaded -- they see "prices.csv: line 7: ...", and the
    server's directory layout is never leaked.
    """
    try:
        bars = HistoricalReplayer.load_csv(path)
    except UnicodeDecodeError:
        # Checked before ValueError: UnicodeDecodeError is a ValueError subclass,
        # and its default message (byte offsets, codec names) means nothing to a user.
        raise HTTPException(
            status_code=400,
            detail=f"{shown_name}: not a UTF-8 text file; save it as CSV (UTF-8)")
    except csv.Error as exc:
        # Malformed CSV structure, e.g. NUL bytes from a binary file renamed .csv.
        raise HTTPException(status_code=400,
                            detail=f"{shown_name}: not a readable CSV ({exc})")
    except ValueError as exc:
        # Missing columns, unparseable numbers, bad OHLC, no rows, ... (load_csv)
        raise HTTPException(status_code=400,
                            detail=str(exc).replace(path, shown_name))
    return len(bars)


@router.post("/api/replay/upload", status_code=201)
async def upload_replay_csv(request: Request,
                            file: UploadFile = File(...)) -> dict:
    """Upload a 1-minute bar CSV for replay.

    Send it as multipart form data in a field named ``file``, e.g.
    ``curl -F "file=@prices.csv" http://localhost:8000/api/replay/upload``.

    Required columns (any case): open, high, low, close, volume, plus either
    date + time or a single datetime/timestamp column.

    Returns 201 with ``csv_path`` -- pass it to ``POST /api/replay/start``.
    Errors: 400 bad file (message says which line and why), 413 too large.
    """
    shown_name = os.path.basename(file.filename or "")
    if not shown_name.lower().endswith(".csv"):
        raise HTTPException(status_code=400,
                            detail="only .csv files are accepted")

    limit_mb = _max_upload_mb()
    max_bytes = int(limit_mb * 1024 * 1024)

    # Early reject from the header, before reading a single byte of the body.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and \
            int(declared) > max_bytes + _MULTIPART_SLACK_BYTES:
        raise _too_large(limit_mb)

    upload_dir = _upload_dir()
    os.makedirs(upload_dir, exist_ok=True)
    file_id = uuid.uuid4().hex[:8]
    final_path = os.path.join(upload_dir, f"{_safe_stem(shown_name)}_{file_id}.csv")
    part_path = final_path + ".part"

    try:
        # Copy with a running byte count: the header can be missing or wrong,
        # so this is the check that actually enforces the limit.
        size = 0
        with open(part_path, "wb") as out:
            while chunk := await file.read(_CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise _too_large(limit_mb)
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail=f"{shown_name}: file is empty")

        # load_csv is blocking file I/O; keep it off the bridge's event loop.
        bars = await asyncio.to_thread(_validate, part_path, shown_name)
        os.replace(part_path, final_path)   # atomic: only valid files appear
    finally:
        # Any failure above (400, 413, disk error) leaves no partial file behind.
        if os.path.exists(part_path):
            os.remove(part_path)
        await file.close()

    log.info("replay upload: %s -> %s (%d bars, %d bytes)",
             shown_name, final_path, bars, size)
    return {
        "file_id": file_id,
        "filename": shown_name,
        "csv_path": final_path,
        "bars": bars,
        "size_bytes": size,
    }


# --------------------------------------------------------------------------- #
# Yahoo Finance download -> saved CSV -> replay from that file                 #
# --------------------------------------------------------------------------- #
class YahooReq(BaseModel):
    symbol: str                 # e.g. "RELIANCE.NS" (NSE) / "RELIANCE.BO" (BSE)
    interval: str = "15m"       # the only supported interval (last ~59 days)
    days: int = 5               # how many calendar days back from now


# One download at a time: Yahoo is unofficial and blocks heavy use, so the
# bridge never fires parallel requests at it, even if users click repeatedly.
_yahoo_lock = asyncio.Lock()


def _write_rows(path: str, rows: list) -> int:
    """Write rows in the replay CSV format; return the file size in bytes."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["date", "time", "open", "high",
                                                "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(rows)
    return os.path.getsize(path)


@router.post("/api/replay/yahoo", status_code=201)
async def download_yahoo_csv(req: YahooReq) -> dict:
    """Download bars from Yahoo Finance and save them as a replay CSV.

    Every download is saved to disk (same folder as uploads) and checked with
    the engine's own CSV rules; replay then reads that file, so Yahoo being
    slow, down or blocking us never affects a running replay, and the same
    data can be replayed again later.

    Returns 201 with ``csv_path`` -- pass it to ``POST /api/replay/start``.
    Errors: 400 bad symbol/interval/days, 404 no data (with a .NS hint for
    Indian stocks), 502 Yahoo failed or blocked us, 503 yfinance missing.
    """
    async with _yahoo_lock:
        try:
            # yfinance is blocking network I/O; keep it off the event loop.
            data = await asyncio.to_thread(fetch_bars, req.symbol,
                                           req.interval, req.days)
        except YahooError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.message)

    rows = data["rows"]
    symbol = data["symbol"]
    first, last = rows[0], rows[-1]
    upload_dir = _upload_dir()
    os.makedirs(upload_dir, exist_ok=True)
    file_id = uuid.uuid4().hex[:8]
    # The name records what was downloaded, e.g.
    # yahoo_RELIANCE_NS_1m_2026-10-20_to_2026-10-24_3f9a1c2e.csv
    # (Not _safe_stem: that strips an "extension", which would cut ".NS_1m_...".)
    stem = _UNSAFE_NAME_CHARS.sub(
        "_", f"yahoo_{symbol}_{req.interval}_{first['date']}_to_{last['date']}")
    shown_name = f"{stem}.csv"
    final_path = os.path.join(upload_dir, f"{stem}_{file_id}.csv")
    part_path = final_path + ".part"

    try:
        size = await asyncio.to_thread(_write_rows, part_path, rows)
        # Same engine checks as an upload, so this file is known to replay.
        bars = await asyncio.to_thread(_validate, part_path, shown_name)
        os.replace(part_path, final_path)
    finally:
        if os.path.exists(part_path):
            os.remove(part_path)

    log.info("yahoo download: %s %s %dd -> %s (%d bars)",
             symbol, req.interval, req.days, final_path, bars)
    return {
        "file_id": file_id,
        "filename": os.path.basename(final_path),
        "csv_path": final_path,
        "bars": bars,
        "size_bytes": size,
        "symbol": symbol,
        "interval": req.interval,
        "first_bar": f"{first['date']} {first['time']}",
        "last_bar": f"{last['date']} {last['time']}",
        "repaired_bars": data["repaired"],
        "requests": data["requests"],
        "source": "yahoo",
    }
