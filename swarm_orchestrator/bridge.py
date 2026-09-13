import os
import zmq
import zmq.asyncio
import asyncio
import random
import logging
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BRIDGE] - %(message)s')

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logging.error(f"Error broadcasting: {e}")
                dead_connections.append(connection)
                
        for dead in dead_connections:
            try:
                self.active_connections.remove(dead)
            except ValueError:
                pass

manager = ConnectionManager()
sim_running = False
last_close = None

# ZMQ Setup
context = zmq.asyncio.Context()
engine_socket = context.socket(zmq.REQ)
# Fetch from environment
zmq_host = os.getenv("ZMQ_HOST", "127.0.0.1")
zmq_port = os.getenv("ZMQ_ORDER_PORT", "5555")
engine_socket.connect(f"tcp://{zmq_host}:{zmq_port}")

oracle_host = os.getenv("ORACLE_HOST", "127.0.0.1" if zmq_host == "127.0.0.1" else "oracle")
oracle_socket = context.socket(zmq.REQ)
oracle_socket.connect(f"tcp://{oracle_host}:5557")

async def background_state_fetcher():
    global sim_running, last_close
    while True:
        if sim_running:
            try:
                await engine_socket.send_json({"action": "FETCH_STATE"})
                state = await engine_socket.recv_json()
                
                current_price = float(state.get("current_price", 0.0))
                if last_close is None:
                    last_close = current_price
                    
                asks_list = state.get("lob_asks", [])
                bids_list = state.get("lob_bids", [])
                
                spread = 0.0
                if asks_list and bids_list:
                    spread = round(asks_list[0][0] - bids_list[0][0], 2)
                
                volatility = state.get("step_volume", 1000) / 100000.0
                jitter_high = random.uniform(0.01, 0.05 + volatility)
                jitter_low = random.uniform(0.01, 0.05 + volatility)
                
                ui_payload = {
                    "type": "sim_update",
                    "unix_time": state.get("unix_time"),
                    "day_count": state.get("day_count", 1),
                    "step_volume": state.get("step_volume", 0),
                    "ohlc": {
                        "open": last_close,
                        "high": max(last_close, current_price) + jitter_high, 
                        "low": min(last_close, current_price) - jitter_low,
                        "close": current_price
                    },
                    "book": {
                        "spread": spread,
                        "asks": [{"price": p[0], "size": p[1]} for p in asks_list],
                        "bids": [{"price": p[0], "size": p[1]} for p in bids_list]
                    },
                    "trades": state.get("trades", []), 
                    "agents": state.get("agents", [])  
                }
                
                await manager.broadcast(ui_payload)
                last_close = current_price 
                
            except Exception as e:
                logging.error(f"Error fetching state: {e}")
        
        await asyncio.sleep(0.05)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup zmq asyncio context for non-blocking if needed, but we'll run fetcher as a task
    task = asyncio.create_task(background_state_fetcher())
    yield
    task.cancel()
    engine_socket.close()
    oracle_socket.close()
    context.term()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def get():
    return HTMLResponse("Backend is running")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    global sim_running, last_close
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            action = message.get("action")
            
            if action == "start_sim":
                sim_running = True
                last_close = float(message.get("start_price", 190.0))
                try:
                    await engine_socket.send_json({
                        "action": "INIT_SIM",
                        "stock": message.get("stock_name", "TCS"),
                        "sector": message.get("sector", "TECH"),
                        "price": last_close
                    })
                    await engine_socket.recv_json()
                except Exception as e:
                    logging.error(f"Failed to initialize Engine: {e}")
                    
            elif action == "stop_sim":
                sim_running = False
                try:
                    await engine_socket.send_json({"action": "STOP_SIM"})
                    await engine_socket.recv_json()
                except Exception as e:
                    logging.error(f"Failed to stop Engine: {e}")
                
            elif action == "inject_news":
                await oracle_socket.send_json({
                    "headline": message.get("headline")
                })
                await oracle_socket.recv_json()
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == '__main__':
    uvicorn.run("bridge:app", host="0.0.0.0", port=8000, reload=False)