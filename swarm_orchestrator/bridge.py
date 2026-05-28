import zmq
import threading
import random
import logging
from flask import Flask, render_template_string
from flask_socketio import SocketIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BRIDGE] - %(message)s')

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
context = zmq.Context()

engine_socket = context.socket(zmq.REQ)
engine_socket.connect("tcp://127.0.0.1:5555")

oracle_socket = context.socket(zmq.REQ)
oracle_socket.connect("tcp://127.0.0.1:5557")

sim_running = False
last_close = None  

def background_state_fetcher():
    global sim_running, last_close
    while True:
        if sim_running:
            try:
                engine_socket.send_json({"action": "FETCH_STATE"})
                state = engine_socket.recv_json()
                
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
                    "unix_time": state.get("unix_time"),      # FIX: Exact engine time
                    "day_count": state.get("day_count", 1),   # FIX: Track the day
                    "step_volume": state.get("step_volume", 0), # FIX: Passes volume to Chart
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
                
                socketio.emit('sim_update', ui_payload)
                last_close = current_price 
                
            except Exception as e:
                logging.error(f"Error fetching state: {e}")
        
        socketio.sleep(1)

@app.route('/')
def index():
    with open('dashboard.html', 'r') as f:
        return render_template_string(f.read())

@socketio.on('start_sim')
def handle_start(data):
    global sim_running, last_close
    sim_running = True
    last_close = float(data.get("start_price", 190.0)) 
    try:
        engine_socket.send_json({
            "action": "INIT_SIM",
            "stock": data.get("stock_name", "TCS"),
            "sector": data.get("sector", "TECH"),
            "price": last_close
        })
        engine_socket.recv_json()
    except Exception as e:
        logging.error(f"Failed to initialize Engine: {e}")

@socketio.on('stop_sim')
def handle_stop():
    global sim_running
    sim_running = False

@socketio.on('inject_news')
def handle_news(data):
    oracle_socket.send_json(data)
    oracle_socket.recv_json()

if __name__ == '__main__':
    threading.Thread(target=background_state_fetcher, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)