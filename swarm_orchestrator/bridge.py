from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import sqlite3
import sys
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator, Optional

import uvicorn
import zmq
import zmq.asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analytics.results import ResultsStore
from analytics.routes_results import router as results_router
from analytics.routes_scripts import router as scripts_router
from features.routes_features import router as features_router
from runner.script_store import ScriptStore

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s[BRIDGE] - %(message)s")
log = logging.getLogger("chronos.bridge")

ZMQ_HOST = os.getenv("ZMQ_HOST", "127.0.0.1")
ZMQ_ORDER_PORT = os.getenv("ZMQ_ORDER_PORT", "5555")
ZMQ_DATA_PORT = os.getenv("ZMQ_DATA_PORT", "5556")
ORACLE_HOST = os.getenv("ORACLE_HOST", ZMQ_HOST)
ORACLE_PORT = os.getenv("ORACLE_PORT", "5557")
BRIDGE_HOST = os.getenv("BRIDGE_HOST", "0.0.0.0")
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "8000"))
CHRONOS_DB = os.getenv("CHRONOS_DB", "chronos.db")

CONTROL_TIMEOUT, HEALTH_TIMEOUT = 3.0, 1.5
WS_QUEUE_MAX, MAX_EQUITY_POINTS = 500, 5000

_ctx: Optional[zmq.asyncio.Context] = None
_sub: Optional[zmq.asyncio.Socket] = None
_control: Optional[zmq.asyncio.Socket] = None
_oracle: Optional[zmq.asyncio.Socket] = None
_control_lock = asyncio.Lock()
_oracle_lock = asyncio.Lock()

_ws_clients: set[asyncio.Queue] = set()
_latest_tick: Optional[dict] = None

_sandbox: Any = None
_results: Optional[ResultsStore] = None
_scripts: Optional[ScriptStore] = None


def _latest_tick_data() -> Optional[dict]:
    return _latest_tick["data"] if _latest_tick else None

async def _request(sock: Optional[zmq.asyncio.Socket], lock: asyncio.Lock, msg: dict, timeout: float) -> Optional[dict]:
    if not sock:
        return None
    async with lock:
        while await sock.poll(0, zmq.POLLIN):
            await sock.recv_multipart()
            
        # Manually encode to bypass pyzmq's inaccurate send_json type hint
        await sock.send(json.dumps(msg).encode("utf-8"))
        
        if await sock.poll(int(timeout * 1000), zmq.POLLIN):
            try:
                return json.loads((await sock.recv_multipart())[-1])
            except (ValueError, IndexError):
                pass
    return None

async def _control_request(msg: dict, timeout: float = CONTROL_TIMEOUT) -> Optional[dict]:
    return await _request(_control, _control_lock, msg, timeout)

async def _oracle_request(msg: dict, timeout: float = CONTROL_TIMEOUT) -> Optional[dict]:
    return await _request(_oracle, _oracle_lock, msg, timeout)

def _get_sandbox() -> Any:
    global _sandbox
    if _sandbox is None:
        from runner.sandbox import SandboxManager
        _sandbox = SandboxManager()
    return _sandbox

def _require_sandbox() -> Any:
    try:
        return _get_sandbox()
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"strategy sandbox unavailable: {exc}")

def _ro_connect() -> sqlite3.Connection:
    uri = pathlib.Path(CHRONOS_DB).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=2.0)

def _read_runs() -> list[dict]:
    if not os.path.exists(CHRONOS_DB):
        return []
    try:
        con = _ro_connect()
        rows = con.execute("SELECT run_id, started_at, symbol, mode, seed FROM runs ORDER BY started_at DESC LIMIT 50").fetchall()
        con.close()
        return [{"run_id": r[0], "started_at": r[1], "symbol": r[2], "mode": r[3], "seed": r[4]} for r in rows]
    except sqlite3.Error:
        return []

def _read_equity(run_id: str, agent_id: Optional[str]) -> dict:
    empty: dict = {"points": []} if agent_id else {"agents": []}
    if not os.path.exists(CHRONOS_DB):
        return empty
    try:
        con = _ro_connect()
        if agent_id is None:
            rows = con.execute("SELECT DISTINCT agent_id FROM agent_snapshots WHERE run_id=? ORDER BY agent_id", (run_id,)).fetchall()
            con.close()
            return {"agents": [r[0] for r in rows]}
        
        rows = con.execute("SELECT seq, equity FROM agent_snapshots WHERE run_id=? AND agent_id=? ORDER BY seq", (run_id, agent_id)).fetchall()
        con.close()
        
        points = [[r[0], r[1]] for r in rows]
        if len(points) > MAX_EQUITY_POINTS:
            points = points[:: len(points) // MAX_EQUITY_POINTS + 1]
        return {"points": points}
    except sqlite3.Error:
        return empty

async def _pump() -> None:
    global _latest_tick
    log.info("pump started (SUB tcp://%s:%s)", ZMQ_HOST, ZMQ_DATA_PORT)
    while True:
        try:
            frames = await _sub.recv_multipart()
            if len(frames) < 2:
                continue
            topic = frames[0].decode("utf-8", "replace")
            data = json.loads(frames[-1])
        except asyncio.CancelledError:
            break
        except Exception:
            continue
            
        message = {"topic": topic, "data": data}
        if topic == "tick":
            _latest_tick = message
            
        if _results:
            _results.ingest(topic, data)
            
        for q in list(_ws_clients):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                with suppress(asyncio.QueueEmpty): q.get_nowait()
                with suppress(asyncio.QueueFull): q.put_nowait(message)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _ctx, _sub, _control, _oracle, _results, _scripts
    _ctx = zmq.asyncio.Context()
    _results, _scripts = ResultsStore(), ScriptStore()
    
    app.state.results, app.state.scripts = _results, _scripts
    app.state.control = _control_request
    app.state.latest_tick = _latest_tick_data
    
    try:
        app.state.sandbox = _get_sandbox()
    except Exception as exc:
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
            if sock: sock.close(0)
        if _ctx: _ctx.term()
        _sub = _control = _oracle = _ctx = None
        log.info("bridge shut down")

app = FastAPI(title="Chronos Bridge", version="2.0", lifespan=lifespan)

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

@app.websocket("/ws")
async def ws_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=WS_QUEUE_MAX)
    _ws_clients.add(queue)
    try:
        if _latest_tick:
            await websocket.send_json(_latest_tick)
        while True:
            await websocket.send_json(await queue.get())
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _ws_clients.discard(queue)

@app.post("/api/sim/start")
async def sim_start(req: SimStartReq) -> dict:
    msg = {"msg": "INIT_SIM", "symbol": req.symbol, "sector": req.sector, "price": req.price}
    if req.seed is not None: msg["seed"] = req.seed
    
    if not (ack := await _control_request(msg)):
        raise HTTPException(status_code=502, detail="engine did not acknowledge INIT_SIM")
    return {"status": ack.get("status"), "run_id": ack.get("run_id")}

@app.post("/api/sim/stop")
async def sim_stop() -> dict:
    if not (ack := await _control_request({"msg": "PAUSE"})):
        raise HTTPException(status_code=502, detail="engine did not acknowledge PAUSE")
    return {"status": ack.get("status")}

@app.post("/api/sim/resume")
async def sim_resume() -> dict:
    if not (ack := await _control_request({"msg": "RESUME"})):
        raise HTTPException(status_code=502, detail="engine did not acknowledge RESUME")
    return {"status": ack.get("status")}

@app.post("/api/news")
async def inject_news(req: NewsReq) -> dict:
    if not (reply := await _oracle_request({"msg": "NEWS", "headline": req.headline})):
        raise HTTPException(status_code=502, detail="oracle did not acknowledge news")
    return reply

@app.post("/api/replay/start")
async def replay_start(req: ReplayStartReq) -> dict:
    msg = {"msg": "REPLAY_START", "csv_path": req.csv_path}
    if req.symbol: msg["symbol"] = req.symbol
    if req.sector: msg["sector"] = req.sector
    if req.seed is not None: msg["seed"] = req.seed
        
    if not (ack := await _control_request(msg)):
        raise HTTPException(status_code=502, detail="engine did not acknowledge REPLAY_START")
    return {"status": ack.get("status"), "run_id": ack.get("run_id"), "bars": ack.get("bars")}

@app.post("/api/strategy/run")
async def strategy_run(req: StrategyRunReq) -> dict:
    mgr = _require_sandbox()
    try:
        result = await asyncio.to_thread(mgr.launch, req.name, req.code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
        
    if _results:
        _results.attach_strategy(result["strategy_id"], result["agent_id"], req.name, None)
    return {"strategy_id": result["strategy_id"], "agent_id": result["agent_id"], "backend": result.get("backend")}

@app.post("/api/strategy/stop")
async def strategy_stop(req: StrategyStopReq) -> dict:
    mgr = _require_sandbox()
    stopped = await asyncio.to_thread(mgr.stop, req.strategy_id)
    if stopped and _results:
        _results.mark_status(req.strategy_id, "stopped")
    return {"status": "STOPPED" if stopped else "NOT_FOUND"}

@app.get("/api/strategy/list")
async def strategy_list() -> list[dict]:
    return await asyncio.to_thread(_require_sandbox().list)

@app.get("/api/strategy/{strategy_id}/logs")
async def strategy_logs(strategy_id: str) -> dict:
    return {"logs": await asyncio.to_thread(_require_sandbox().logs, strategy_id)}

@app.get("/api/runs")
async def runs() -> list[dict]:
    return await asyncio.to_thread(_read_runs)

@app.get("/api/runs/{run_id}/equity")
async def run_equity(run_id: str, agent_id: Optional[str] = None) -> dict:
    return await asyncio.to_thread(_read_equity, run_id, agent_id)

@app.get("/api/health")
async def health() -> dict:
    ack = await _control_request({"msg": "PING"}, timeout=HEALTH_TIMEOUT)
    return {"status": "ok", "engine": "up" if ack else "down"}

app.include_router(scripts_router)
app.include_router(results_router)
app.include_router(features_router)
app.mount("/", StaticFiles(directory="dashboard", html=True, check_dir=False), name="dashboard")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    _config = uvicorn.Config(app, host=BRIDGE_HOST, port=BRIDGE_PORT, log_level="info", loop="asyncio")
    asyncio.run(uvicorn.Server(_config).serve())
    