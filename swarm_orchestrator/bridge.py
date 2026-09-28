"""Chronos FastAPI web bridge (Phase 2).

A single-event-loop, fully-async bridge between the browser dashboard and the
Chronos engine + oracle over ZeroMQ. This replaces the Phase-1 Flask +
``flask_socketio`` bridge and its flaw **R7**: a blocking ``zmq.REQ`` socket
shared across a background thread and Socket.IO handler threads (ZeroMQ sockets
are *not* thread-safe). Here every socket is created on, and touched only by,
the uvicorn asyncio loop -- there are no threads.

Transport map (see ``PROTOCOL.md`` sections 1-6)::

    SUB   <- engine PUB   (ZMQ_DATA_PORT)   topics tick/trade/event -> WebSockets
    DEALER "BRIDGE"        -> engine ROUTER (ZMQ_ORDER_PORT)  control plane
    DEALER "BRIDGE_ORACLE" -> oracle ROUTER (ORACLE_PORT)     news scoring

The oracle scores a headline and, when material, submits its order to the
engine carrying ``meta.news``; the engine then publishes an ``event kind=news``
that reaches the UI over this bridge's WebSocket. The bridge never scores news
itself and never publishes on the data plane.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import sqlite3
import sys
from collections import deque
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator, Optional

import uvicorn
import zmq
import zmq.asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# v2 feature layer (PROTOCOL_V2): script library, session results, and the four
# new capabilities. Routers read shared singletons from ``app.state`` (wired in
# the lifespan below); the ResultsStore is fed by the pump.
from analytics.results import ResultsStore
from analytics.routes_results import router as results_router
from analytics.routes_scripts import router as scripts_router
from features.routes_features import router as features_router
from runner.script_store import ScriptStore

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [BRIDGE] - %(message)s"
)
log = logging.getLogger("chronos.bridge")

# --------------------------------------------------------------------------- #
# Configuration (PROTOCOL section 2)                                          #
# --------------------------------------------------------------------------- #
ZMQ_HOST = os.getenv("ZMQ_HOST", "127.0.0.1")
ZMQ_ORDER_PORT = os.getenv("ZMQ_ORDER_PORT", "5555")
ZMQ_DATA_PORT = os.getenv("ZMQ_DATA_PORT", "5556")
# Defaults to ZMQ_HOST (single-machine dev, where engine/oracle share a host). Set
# ORACLE_HOST explicitly when the oracle lives on a different host/container than
# the engine (e.g. Docker Compose, where each service gets its own hostname).
ORACLE_HOST = os.getenv("ORACLE_HOST", ZMQ_HOST)
ORACLE_PORT = os.getenv("ORACLE_PORT", "5557")
BRIDGE_HOST = os.getenv("BRIDGE_HOST", "0.0.0.0")
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "8000"))
CHRONOS_DB = os.getenv("CHRONOS_DB", "chronos.db")

CONTROL_TIMEOUT = 3.0      # engine control ack budget (POST /api/sim/*)
HEALTH_TIMEOUT = 1.5       # PING roundtrip budget (GET /api/health)
WS_QUEUE_MAX = 500         # per-client buffer; drop-oldest beyond this
MAX_EQUITY_POINTS = 5000   # equity series cap (stride-sampled beyond this)

# --------------------------------------------------------------------------- #
# Shared runtime state (sockets are created in the lifespan startup so they    #
# bind to the running uvicorn loop; importing this module has no side effects) #
# --------------------------------------------------------------------------- #
_ctx: Optional[zmq.asyncio.Context] = None
_sub: Optional[zmq.asyncio.Socket] = None
_control: Optional[zmq.asyncio.Socket] = None
_oracle: Optional[zmq.asyncio.Socket] = None
_control_lock = asyncio.Lock()
_oracle_lock = asyncio.Lock()

_ws_clients: set[asyncio.Queue] = set()
_latest_tick: Optional[dict] = None
_recent_trades: "deque[dict]" = deque(maxlen=300)  # for narrator / agent views

_sandbox: Any = None  # lazy runner.sandbox.SandboxManager singleton
_results: Optional[ResultsStore] = None   # session results (fed by the pump)
_scripts: Optional[ScriptStore] = None    # strategy script library (files)


def _latest_tick_data() -> Optional[dict]:
    """The most recent §5.1 tick payload (data only), or None. Used by app.state."""
    return _latest_tick["data"] if _latest_tick else None


def _recent_trades_list(n: int = 200) -> list[dict]:
    """The last ``n`` trade payloads (§5.2). Used by app.state for narration."""
    return list(_recent_trades)[-n:]


# --------------------------------------------------------------------------- #
# DEALER request/reply helper                                                  #
# --------------------------------------------------------------------------- #
async def _request(
    sock: Optional[zmq.asyncio.Socket],
    lock: asyncio.Lock,
    msg: dict,
    timeout: float,
) -> Optional[dict]:
    """Send ``msg`` on a DEALER socket and await one JSON reply.

    Serialized by ``lock`` so concurrent HTTP requests never interleave frames
    on the same socket. Any stale reply left by a previously timed-out request
    is drained first. Returns the decoded reply, or ``None`` on timeout /
    undecodable reply (callers translate ``None`` into an HTTP error).
    """
    if sock is None:  # startup not complete
        return None
    async with lock:
        while await sock.poll(0, zmq.POLLIN):      # drop stale replies
            await sock.recv_multipart()
        await sock.send_json(msg)
        if await sock.poll(int(timeout * 1000), zmq.POLLIN):
            frames = await sock.recv_multipart()
            try:
                return json.loads(frames[-1])
            except (ValueError, IndexError):
                return None
        return None


async def _control_request(msg: dict, timeout: float = CONTROL_TIMEOUT) -> Optional[dict]:
    """Issue a control-plane request to the engine (DEALER identity ``BRIDGE``)."""
    return await _request(_control, _control_lock, msg, timeout)


async def _oracle_request(msg: dict, timeout: float = CONTROL_TIMEOUT) -> Optional[dict]:
    """Issue a request to the oracle (DEALER identity ``BRIDGE_ORACLE``)."""
    return await _request(_oracle, _oracle_lock, msg, timeout)


# --------------------------------------------------------------------------- #
# Strategy sandbox (lazy module-level singleton)                               #
# --------------------------------------------------------------------------- #
def _get_sandbox() -> Any:
    """Return the process-wide ``SandboxManager`` singleton, importing lazily.

    The import is deferred so ``import bridge`` succeeds before the RUNNER
    agent's ``runner/sandbox.py`` lands; the singleton is created on first use.
    """
    global _sandbox
    if _sandbox is None:
        from runner.sandbox import SandboxManager  # noqa: WPS433 (intentional lazy import)

        _sandbox = SandboxManager()
    return _sandbox


def _require_sandbox() -> Any:
    """Like :func:`_get_sandbox` but maps a missing runtime to HTTP 503."""
    try:
        return _get_sandbox()
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"strategy sandbox unavailable: {exc}")


# --------------------------------------------------------------------------- #
# Read-only SQLite access (PROTOCOL section 7). Callers wrap these in          #
# ``asyncio.to_thread`` so the sqlite blocking calls never stall the loop.     #
# --------------------------------------------------------------------------- #
def _ro_connect() -> sqlite3.Connection:
    """Open the engine DB strictly read-only via a ``file:...?mode=ro`` URI."""
    uri = pathlib.Path(CHRONOS_DB).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=2.0)


def _read_runs() -> list[dict]:
    """Return up to 50 most-recent runs, newest first. ``[]`` if DB absent."""
    if not os.path.exists(CHRONOS_DB):
        return []
    try:
        con = _ro_connect()
    except sqlite3.Error:
        return []
    try:
        rows = con.execute(
            "SELECT run_id, started_at, symbol, mode, seed FROM runs "
            "ORDER BY started_at DESC LIMIT 50"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    return [
        {"run_id": r[0], "started_at": r[1], "symbol": r[2], "mode": r[3], "seed": r[4]}
        for r in rows
    ]


def _read_equity(run_id: str, agent_id: Optional[str]) -> dict:
    """Equity curve for ``(run_id, agent_id)``, or the run's agent list.

    With ``agent_id`` -> ``{"points": [[seq, equity], ...]}`` (stride-sampled to
    at most :data:`MAX_EQUITY_POINTS`). Without it -> ``{"agents": [...]}`` so
    the UI can populate a picker. Missing DB/tables degrade to empty results.
    """
    empty: dict = {"points": []} if agent_id else {"agents": []}
    if not os.path.exists(CHRONOS_DB):
        return empty
    try:
        con = _ro_connect()
    except sqlite3.Error:
        return empty
    try:
        if agent_id is None:
            rows = con.execute(
                "SELECT DISTINCT agent_id FROM agent_snapshots "
                "WHERE run_id=? ORDER BY agent_id",
                (run_id,),
            ).fetchall()
            return {"agents": [r[0] for r in rows]}
        rows = con.execute(
            "SELECT seq, equity FROM agent_snapshots "
            "WHERE run_id=? AND agent_id=? ORDER BY seq",
            (run_id, agent_id),
        ).fetchall()
    except sqlite3.Error:
        return empty
    finally:
        con.close()
    points = [[r[0], r[1]] for r in rows]
    if len(points) > MAX_EQUITY_POINTS:
        stride = len(points) // MAX_EQUITY_POINTS + 1
        points = points[::stride]
    return {"points": points}


# --------------------------------------------------------------------------- #
# Market-data pump: engine PUB -> per-client queues                            #
# --------------------------------------------------------------------------- #
async def _pump() -> None:
    """Fan out every engine PUB message to all WebSocket clients.

    Each client owns a bounded queue; on overflow the oldest message is dropped
    so a single slow browser can never stall the pump or the other clients. The
    latest ``tick`` is cached and replayed to late joiners.
    """
    global _latest_tick
    assert _sub is not None
    log.info("pump started (SUB tcp://%s:%s)", ZMQ_HOST, ZMQ_DATA_PORT)
    while True:
        try:
            frames = await _sub.recv_multipart()
        except asyncio.CancelledError:
            break
        except Exception as exc:  # pragma: no cover - transport hiccup
            log.warning("pump recv error: %s", exc)
            continue
        if len(frames) < 2:
            continue
        topic = frames[0].decode("utf-8", "replace")
        try:
            data = json.loads(frames[-1])
        except ValueError:
            continue
        message = {"topic": topic, "data": data}
        if topic == "tick":
            _latest_tick = message
        elif topic == "trade":
            _recent_trades.append(data)
        # Feed the session results recorder (never raises on the hot path).
        if _results is not None:
            _results.ingest(topic, data)
        for q in list(_ws_clients):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                with suppress(asyncio.QueueEmpty):
                    q.get_nowait()          # drop oldest
                with suppress(asyncio.QueueFull):
                    q.put_nowait(message)


# --------------------------------------------------------------------------- #
# Application lifespan: bring sockets up on the running loop, tear down clean.  #
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create ZeroMQ sockets and the pump task on startup; close on shutdown."""
    global _ctx, _sub, _control, _oracle, _results, _scripts
    _ctx = zmq.asyncio.Context()

    # v2 feature-layer singletons, exposed to routers via app.state.
    _results = ResultsStore()
    _scripts = ScriptStore()
    app.state.results = _results
    app.state.scripts = _scripts
    app.state.control = _control_request
    app.state.latest_tick = _latest_tick_data
    app.state.recent_trades = _recent_trades_list
    try:
        app.state.sandbox = _get_sandbox()
    except Exception as exc:  # runner.sandbox missing/unavailable -> routers 503
        log.warning("sandbox unavailable at startup: %s", exc)
        app.state.sandbox = None

    _sub = _ctx.socket(zmq.SUB)
    _sub.connect(f"tcp://{ZMQ_HOST}:{ZMQ_DATA_PORT}")
    for topic in ("tick", "trade", "event"):
        _sub.setsockopt_string(zmq.SUBSCRIBE, topic)

    _control = _ctx.socket(zmq.DEALER)
    _control.setsockopt(zmq.IDENTITY, b"BRIDGE")
    _control.setsockopt(zmq.LINGER, 0)
    _control.connect(f"tcp://{ZMQ_HOST}:{ZMQ_ORDER_PORT}")

    _oracle = _ctx.socket(zmq.DEALER)
    _oracle.setsockopt(zmq.IDENTITY, b"BRIDGE_ORACLE")
    _oracle.setsockopt(zmq.LINGER, 0)
    _oracle.connect(f"tcp://{ORACLE_HOST}:{ORACLE_PORT}")

    pump_task = asyncio.create_task(_pump())
    log.info("bridge online, serving http://%s:%s", BRIDGE_HOST, BRIDGE_PORT)
    try:
        yield
    finally:
        pump_task.cancel()
        with suppress(asyncio.CancelledError):
            await pump_task
        for sock in (_sub, _control, _oracle):
            if sock is not None:
                sock.close(0)
        if _ctx is not None:
            _ctx.term()
        _sub = _control = _oracle = _ctx = None
        log.info("bridge shut down")


app = FastAPI(title="Chronos Bridge", version="2.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Request models                                                               #
# --------------------------------------------------------------------------- #
class SimStartReq(BaseModel):
    symbol: str
    sector: str
    price: float
    seed: Optional[int] = None


class NewsReq(BaseModel):
    headline: str


class ReplayStartReq(BaseModel):
    csv_path: str
    symbol: Optional[str] = None
    sector: Optional[str] = None
    seed: Optional[int] = None


class StrategyRunReq(BaseModel):
    name: str
    code: str


class StrategyStopReq(BaseModel):
    strategy_id: str


# --------------------------------------------------------------------------- #
# WebSocket: /ws                                                               #
# --------------------------------------------------------------------------- #
@app.websocket("/ws")
async def ws_stream(websocket: WebSocket) -> None:
    """Stream ``{"topic","data"}`` messages; replay the cached tick on connect."""
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=WS_QUEUE_MAX)
    _ws_clients.add(queue)
    log.info("ws client connected (%d total)", len(_ws_clients))
    try:
        if _latest_tick is not None:
            await websocket.send_json(_latest_tick)
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - client vanished mid-send
        log.debug("ws send ended: %s", exc)
    finally:
        _ws_clients.discard(queue)
        log.info("ws client disconnected (%d total)", len(_ws_clients))


# --------------------------------------------------------------------------- #
# REST: simulation control                                                     #
# --------------------------------------------------------------------------- #
@app.post("/api/sim/start")
async def sim_start(req: SimStartReq) -> dict:
    """Initialise a live simulation on the engine (INIT_SIM)."""
    msg: dict = {
        "msg": "INIT_SIM",
        "symbol": req.symbol,
        "sector": req.sector,
        "price": req.price,
    }
    if req.seed is not None:
        msg["seed"] = req.seed
    ack = await _control_request(msg)
    if ack is None:
        raise HTTPException(status_code=502, detail="engine did not acknowledge INIT_SIM")
    return {"status": ack.get("status"), "run_id": ack.get("run_id")}


@app.post("/api/sim/stop")
async def sim_stop() -> dict:
    """Pause the running simulation (PAUSE)."""
    ack = await _control_request({"msg": "PAUSE"})
    if ack is None:
        raise HTTPException(status_code=502, detail="engine did not acknowledge PAUSE")
    return {"status": ack.get("status")}


@app.post("/api/sim/resume")
async def sim_resume() -> dict:
    """Resume a paused simulation (RESUME)."""
    ack = await _control_request({"msg": "RESUME"})
    if ack is None:
        raise HTTPException(status_code=502, detail="engine did not acknowledge RESUME")
    return {"status": ack.get("status")}


@app.post("/api/news")
async def inject_news(req: NewsReq) -> dict:
    """Forward a headline to the oracle. Returns its immediate accept ack.

    The oracle scores asynchronously; the scored result reaches the UI later as
    an ``event kind=news`` published by the engine when the oracle's order (with
    ``meta.news``) executes.
    """
    reply = await _oracle_request({"msg": "NEWS", "headline": req.headline})
    if reply is None:
        raise HTTPException(status_code=502, detail="oracle did not acknowledge news")
    return reply


@app.post("/api/replay/start")
async def replay_start(req: ReplayStartReq) -> dict:
    """Start a historical replay from a 1-minute-bar CSV (REPLAY_START)."""
    msg: dict = {"msg": "REPLAY_START", "csv_path": req.csv_path}
    if req.symbol is not None:
        msg["symbol"] = req.symbol
    if req.sector is not None:
        msg["sector"] = req.sector
    if req.seed is not None:
        msg["seed"] = req.seed
    ack = await _control_request(msg)
    if ack is None:
        raise HTTPException(status_code=502, detail="engine did not acknowledge REPLAY_START")
    return {"status": ack.get("status"), "run_id": ack.get("run_id"), "bars": ack.get("bars")}


# --------------------------------------------------------------------------- #
# REST: strategy sandbox                                                       #
# --------------------------------------------------------------------------- #
@app.post("/api/strategy/run")
async def strategy_run(req: StrategyRunReq) -> dict:
    """Launch user strategy code in a sandbox. 400 if the code is invalid."""
    mgr = _require_sandbox()
    try:
        result = await asyncio.to_thread(mgr.launch, req.name, req.code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # Register the direct-run strategy with the session results store (the
    # library-run path does the same inside analytics.routes_scripts).
    if _results is not None:
        _results.attach_strategy(
            result["strategy_id"], result["agent_id"], req.name, None)
    return {"strategy_id": result["strategy_id"], "agent_id": result["agent_id"],
            "backend": result.get("backend")}


@app.post("/api/strategy/stop")
async def strategy_stop(req: StrategyStopReq) -> dict:
    """Stop a running sandboxed strategy."""
    mgr = _require_sandbox()
    stopped = await asyncio.to_thread(mgr.stop, req.strategy_id)
    if stopped and _results is not None:
        _results.mark_status(req.strategy_id, "stopped")
    return {"status": "STOPPED" if stopped else "NOT_FOUND"}


@app.get("/api/strategy/list")
async def strategy_list() -> list[dict]:
    """List launched strategies and their status."""
    mgr = _require_sandbox()
    return await asyncio.to_thread(mgr.list)


@app.get("/api/strategy/{strategy_id}/logs")
async def strategy_logs(strategy_id: str) -> dict:
    """Return the tail of a strategy's sandboxed stdout/stderr."""
    mgr = _require_sandbox()
    logs = await asyncio.to_thread(mgr.logs, strategy_id)
    return {"logs": logs}


# --------------------------------------------------------------------------- #
# REST: history / persistence (read-only)                                      #
# --------------------------------------------------------------------------- #
@app.get("/api/runs")
async def runs() -> list[dict]:
    """List recent runs from the engine's SQLite store."""
    return await asyncio.to_thread(_read_runs)


@app.get("/api/runs/{run_id}/equity")
async def run_equity(run_id: str, agent_id: Optional[str] = None) -> dict:
    """Equity curve for one agent, or the run's agent list when omitted."""
    return await asyncio.to_thread(_read_equity, run_id, agent_id)


@app.get("/api/health")
async def health() -> dict:
    """Report bridge health and whether the engine answers a PING in time."""
    ack = await _control_request({"msg": "PING"}, timeout=HEALTH_TIMEOUT)
    return {"status": "ok", "engine": "up" if ack is not None else "down"}


# --------------------------------------------------------------------------- #
# v2 feature routers (script library, session results/reports, and the four    #
# new capabilities). Included BEFORE the static mount so /api takes precedence. #
# --------------------------------------------------------------------------- #
app.include_router(scripts_router)
app.include_router(results_router)
app.include_router(features_router)


# --------------------------------------------------------------------------- #
# Static frontend -- mounted LAST so /api and /ws take precedence.             #
# check_dir=False lets this module import before the FRONTEND agent's files    #
# land; the directory is resolved per-request at serve time.                   #
# --------------------------------------------------------------------------- #
app.mount("/", StaticFiles(directory="dashboard", html=True, check_dir=False), name="dashboard")


if __name__ == "__main__":
    # zmq.asyncio requires a selector loop. On Windows, uvicorn.run() installs its
    # own Proactor loop (which zmq cannot use), so a policy set here would be
    # ignored. Instead we build the server with loop="asyncio" and run it inside a
    # selector loop we create via asyncio.run() — this is the loop zmq.asyncio uses.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    _config = uvicorn.Config(app, host=BRIDGE_HOST, port=BRIDGE_PORT,
                             log_level="info", loop="asyncio")
    asyncio.run(uvicorn.Server(_config).serve())
