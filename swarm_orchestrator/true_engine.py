import zmq
import time
import random
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [TRUE_ENGINE] - %(message)s')

class MatchingEngine:
    def __init__(self):
        self.bids = {}  
        self.asks = {}  
        self.share_name = "TCS"
        self.sector = "TECH"
        self.current_price = 190.00
        
        self.agent_ledger = {}
        self.recent_trades = []
        
        # MACRO REGIME & MASTER CLOCK
        self.macro_regime = "RANGING"
        self.regime_ticks_left = 60
        self.market_minute = 0  
        self.day_count = 1
        # Start exactly at 2026-01-01 09:15:00 UTC (1767258900)
        self.sim_unix_time = 1767258900 
        self.u_shape_multiplier = 2.5
        
        self.recent_volume = 10000.0 * self.u_shape_multiplier
        self.price_history = [self.current_price] * 15 
        
    def initialize_sim(self, stock, sector, price):
        self.share_name = stock
        self.sector = sector
        self.current_price = float(price)
        self.price_history = [self.current_price] * 15
        self.bids.clear()
        self.asks.clear()
        self.agent_ledger.clear()
        self.recent_trades.clear()
        self.macro_regime = "RANGING"
        self.regime_ticks_left = 60
        
        self.market_minute = 0
        self.day_count = 1
        self.sim_unix_time = 1767258900
        self.u_shape_multiplier = 2.5
        self.recent_volume = 10000.0 * self.u_shape_multiplier
        logging.warning(f"=== ENGINE RESET: {self.share_name} | Starting @ ${self.current_price} | DAY 1, 09:15 AM ===")

    def _get_agent(self, agent_id):
        if agent_id not in self.agent_ledger:
            a_type = "Institution" if "WHALE" in agent_id else "Market Maker" if "MM" in agent_id else "Retail"
            if "ORACLE" in agent_id: a_type = "Macro Oracle"
            if "GHOST" in agent_id: a_type = "Dark Pool"
            
            start_cash = 5000000.0 if a_type == "Institution" else 250000.0 if a_type == "Market Maker" else 50000.0
            if a_type == "Dark Pool": start_cash = 1000000000.0 
            
            self.agent_ledger[agent_id] = {"id": agent_id, "type": a_type, "cash": start_cash, "pos": 0}
        else:
            if self.agent_ledger[agent_id]["type"] == "Retail":
                net_value = self.agent_ledger[agent_id]["cash"] + (self.agent_ledger[agent_id]["pos"] * self.current_price)
                if net_value < 1000:
                    logging.warning(f"LIQUIDATION: {agent_id} Bankrupt! Respawning with fresh $50k.")
                    self.agent_ledger[agent_id]["cash"] = 50000.0
                    self.agent_ledger[agent_id]["pos"] = 0
        return self.agent_ledger[agent_id]

    def add_order(self, agent_id, action, price, qty):
        self._get_agent(agent_id) 
        price = round(price, 2)
        if action == "BUY":
            if price not in self.bids: self.bids[price] = []
            self.bids[price].append({"agent_id": agent_id, "qty": qty})
        elif action == "SELL":
            if price not in self.asks: self.asks[price] = []
            self.asks[price].append({"agent_id": agent_id, "qty": qty})

    def calculate_rsi(self):
        if len(self.price_history) < 15: return 50.0
        gains = losses = 0
        for i in range(1, 15):
            change = self.price_history[-i] - self.price_history[-(i+1)]
            if change > 0: gains += change
            else: losses += abs(change)
        if losses == 0: return 100.0
        rs = (gains / 14) / (losses / 14)
        return round(100 - (100 / (1 + rs)), 2)

    def process_market_order(self, taker_id, action, qty):
        self._get_agent(taker_id)
        executed_qty = 0
        total_cost = 0.0
        
        if action == "BUY":
            sorted_asks = sorted(self.asks.keys())
            for price in sorted_asks:
                orders = self.asks[price]
                orders_to_remove = []
                for order in orders:
                    maker_id = order["agent_id"]
                    available = order["qty"]
                    if available == 0: continue
                    fill_qty = min(qty - executed_qty, available)
                    order["qty"] -= fill_qty
                    
                    self.agent_ledger[taker_id]["pos"] += fill_qty
                    self.agent_ledger[taker_id]["cash"] -= (fill_qty * price)
                    self.agent_ledger[maker_id]["pos"] -= fill_qty
                    self.agent_ledger[maker_id]["cash"] += (fill_qty * price)
                    
                    executed_qty += fill_qty
                    total_cost += (fill_qty * price)
                    
                    self.recent_trades.insert(0, {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "price": price, "size": fill_qty,
                        "buyer": taker_id, "seller": maker_id
                    })
                    if order["qty"] == 0: orders_to_remove.append(order)
                    if executed_qty >= qty: break
                for o in orders_to_remove: orders.remove(o)
                if len(orders) == 0: del self.asks[price]
                if executed_qty >= qty: break
                
        elif action == "SELL":
            sorted_bids = sorted(self.bids.keys(), reverse=True)
            for price in sorted_bids:
                orders = self.bids[price]
                orders_to_remove = []
                for order in orders:
                    maker_id = order["agent_id"]
                    available = order["qty"]
                    if available == 0: continue
                    fill_qty = min(qty - executed_qty, available)
                    order["qty"] -= fill_qty
                    
                    self.agent_ledger[taker_id]["pos"] -= fill_qty
                    self.agent_ledger[taker_id]["cash"] += (fill_qty * price)
                    self.agent_ledger[maker_id]["pos"] += fill_qty
                    self.agent_ledger[maker_id]["cash"] -= (fill_qty * price)
                    
                    executed_qty += fill_qty
                    total_cost += (fill_qty * price)
                    
                    self.recent_trades.insert(0, {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "price": price, "size": fill_qty,
                        "buyer": maker_id, "seller": taker_id
                    })
                    if order["qty"] == 0: orders_to_remove.append(order)
                    if executed_qty >= qty: break
                for o in orders_to_remove: orders.remove(o)
                if len(orders) == 0: del self.bids[price]
                if executed_qty >= qty: break
                
        self.recent_trades = self.recent_trades[:50] 
        avg_price = total_cost / executed_qty if executed_qty > 0 else self.current_price
        
        if executed_qty > 0:
            if qty > 50000:
                impact_multiplier = (qty / 100000) * 2.00 
                if action == "BUY": self.current_price = avg_price + impact_multiplier
                else: self.current_price = avg_price - impact_multiplier
                
                self.bids.clear()
                self.asks.clear()
                logging.warning(f"💥 BLACK SWAN EVENT! Order book cleared. Price gapped to ${self.current_price:.2f}")
            else:
                self.current_price = avg_price
                
            self.price_history.append(self.current_price)
            if len(self.price_history) > 100: self.price_history.pop(0)
                
        return executed_qty, round(avg_price, 2)

    def apply_market_regime_physics(self):
        # ========================================================
        # THE MASTER CLOCK: Advances exactly 1 minute per tick
        # ========================================================
        self.market_minute += 1
        
        if self.market_minute >= 375:
            # IT'S 3:30 PM! Close the market and jump to 9:15 AM tomorrow
            self.market_minute = 0
            self.day_count += 1
            self.sim_unix_time += int(17.5 * 3600)  # Jump 17.5 hours
            logging.warning(f"🔔 DING DING DING! Market Closed. Opening DAY {self.day_count} at 09:15 AM.")
        else:
            self.sim_unix_time += 60 # Normal tick adds 60 seconds
            
        normalized_time = (self.market_minute - 187.5) / 187.5
        self.u_shape_multiplier = 0.2 + (2.3 * (normalized_time ** 2))

        # ========================================================
        # REGIME SHIFTS (Bull/Bear intact and highly visible)
        # ========================================================
        self.regime_ticks_left -= 1
        if self.regime_ticks_left <= 0:
            self.macro_regime = random.choices(["BULL", "BEAR", "RANGING"], weights=[0.4, 0.4, 0.2])[0]
            self.regime_ticks_left = random.randint(60, 180) 
            logging.info(f"🌊 MACRO TIDE SHIFT: Market entering {self.macro_regime} phase.")

        nearest_level = round(self.current_price / 5.0) * 5.0
        distance = self.current_price - nearest_level
        bounce_qty = int(random.randint(2500, 5000) * self.u_shape_multiplier)

        if self.macro_regime == "BEAR" and 0 < distance < 0.20:
            if random.random() < 0.3: 
                self.process_market_order("GHOST_REVERSAL", "BUY", bounce_qty)
                self.macro_regime = "BULL"
                self.regime_ticks_left = random.randint(30, 90)

        elif self.macro_regime == "BULL" and -0.20 < distance < 0:
            if random.random() < 0.3: 
                self.process_market_order("GHOST_REVERSAL", "SELL", bounce_qty)
                self.macro_regime = "BEAR"
                self.regime_ticks_left = random.randint(30, 90)

        # ORGANIC TREND CREATION
        trend_qty = int(random.randint(300, 1000) * self.u_shape_multiplier)
        if self.macro_regime == "BULL" and random.random() < 0.5:
            self.process_market_order("GHOST_TREND", "BUY", trend_qty)
        elif self.macro_regime == "BEAR" and random.random() < 0.5:
            self.process_market_order("GHOST_TREND", "SELL", trend_qty)

        prune_threshold = 0.50 
        stale_bids = [p for p in self.bids.keys() if p < self.current_price - prune_threshold]
        for p in stale_bids: del self.bids[p]
        stale_asks = [p for p in self.asks.keys() if p > self.current_price + prune_threshold]
        for p in stale_asks: del self.asks[p]

    def generate_lob_state(self):
        agg_bids = {p: sum(o["qty"] for o in orders) for p, orders in self.bids.items()}
        agg_asks = {p: sum(o["qty"] for o in orders) for p, orders in self.asks.items()}
        top_bids = sorted([[p, q] for p, q in agg_bids.items() if q > 0], key=lambda x: x[0], reverse=True)[:5]
        top_asks = sorted([[p, q] for p, q in agg_asks.items() if q > 0], key=lambda x: x[0])[:5]
        
        display_volume = int(self.recent_volume * self.u_shape_multiplier)
        volatility_scale = max(1.0, display_volume / 10000.0)
        
        regime_drift = 0.0
        if self.macro_regime == "BULL": regime_drift = random.uniform(0.01, 0.04)
        elif self.macro_regime == "BEAR": regime_drift = random.uniform(-0.04, -0.01)
        
        price_drift = (random.gauss(0, 0.02) * volatility_scale) + regime_drift
        self.current_price = max(0.01, self.current_price + price_drift)
        
        synthetic_ofi = max(-1.0, min(1.0, random.uniform(-0.3, 0.3) + price_drift))

        if not top_bids: top_bids = [[round(self.current_price - 0.1, 2), 1000]]
        if not top_asks: top_asks = [[round(self.current_price + 0.1, 2), 1000]]
        
        leaderboard = []
        for a_id, data in self.agent_ledger.items():
            if "GHOST" in a_id: continue 
            
            pnl = data["cash"] + (data["pos"] * self.current_price) 
            start_cash = 5000000.0 if data["type"] == "Institution" else 250000.0 if data["type"] == "Market Maker" else 50000.0
            leaderboard.append({"id": a_id, "type": data["type"], "pnl": round(pnl - start_cash, 2), "pos": data["pos"]})
        
        leaderboard = sorted(leaderboard, key=lambda x: x["pnl"], reverse=True)[:15] 

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "unix_time": self.sim_unix_time,     # <--- PASSING THE MASTER CLOCK
            "day_count": self.day_count,         # <--- PASSING THE DAY TRACKER
            "asset_id": self.share_name,
            "current_price": round(self.current_price, 2),
            "mid_price": round(self.current_price, 2),
            "vwap": round(self.current_price, 2),
            "rsi": self.calculate_rsi(),
            "step_volume": display_volume,       # <--- PASSING THE VOLUME
            "lob_asks": top_asks,
            "lob_bids": top_bids,
            "order_flow_imbalance": round(synthetic_ofi, 3),
            "agents": leaderboard,
            "trades": self.recent_trades 
        }

def run_true_engine():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{os.getenv('ZMQ_HOST', '127.0.0.1')}:{os.getenv('ZMQ_ORDER_PORT', '5555')}")

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    logging.info("TRUE Matching Engine Online. Waiting for Swarm...")

    engine = MatchingEngine()
    total_requests = 0
    start_time = time.time()

    try:
        while True:
            socks = dict(poller.poll(timeout=50))
            if socket in socks and socks[socket] == zmq.POLLIN:
                message = socket.recv_json()

                if message.get("action") == "FETCH_STATE":
                    socket.send_json(engine.generate_lob_state())
                    total_requests += 1
                elif message.get("action") == "INIT_SIM":
                    engine.initialize_sim(message.get("stock", "TCS"), message.get("sector", "TECH"), message.get("price", 190.0))
                    socket.send_json({"status": "ENGINE_READY"})
                else:
                    agent_id = message.get("agent_id", "UNKNOWN")
                    action = message.get("action")
                    order_type = message.get("type", "MARKET")
                    qty = message.get("qty", 0)
                    price = message.get("price", engine.current_price)
                    
                    if order_type == "LIMIT":
                        engine.add_order(agent_id, action, price, qty)
                        socket.send_json({"status": "ACKNOWLEDGED", "executed_qty": 0, "average_price": 0})
                    else: 
                        executed_qty, avg_price = engine.process_market_order(agent_id, action, qty)
                        engine.recent_volume = max(2000.0, (engine.recent_volume * 0.8) + (executed_qty * 0.2))
                        socket.send_json({
                            "status": "FILLED" if executed_qty == qty else "PARTIAL",
                            "executed_qty": executed_qty,
                            "average_price": avg_price
                        })
                    total_requests += 1

            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                engine.apply_market_regime_physics()
                
                if total_requests > 0:
                    logging.info(f"{engine.share_name} | Price=${engine.current_price:.2f} | Vol={int(engine.recent_volume * engine.u_shape_multiplier)} | TICK={engine.market_minute}/375")
                total_requests = 0
                start_time = time.time()

    except KeyboardInterrupt:
        socket.close()
        context.term()

if __name__ == "__main__":
    run_true_engine()