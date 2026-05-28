# import random
# import numpy as np
# import pandas as pd
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# from agents.agents import SmartMoneyEnv, MarketMakerEnv
# from agents.backtest_retail import HeuristicRetailSwarm

# print("\n[AI_LOADER] Initializing AI Models into memory. Please wait...")

# # Dummy data required just to initialize the environment wrappers
# dummy_df = pd.DataFrame({
#     'date': ['2026-01-01']*10, 'time': ['09:30:00']*10,
#     'Open': [100]*10, 'High': [100]*10, 'Low': [100]*10, 'Close': [100]*10,
#     'Volume': [1000]*10, 'VWAP': [100]*10, 'Volume_MA': [1000]*10,
#     'Order_Flow_Imbalance': [0]*10, 'RSI_14': [50]*10
# })

# # 1. Load Market Maker Brain
# mm_raw_env = DummyVecEnv([lambda: MarketMakerEnv(dummy_df, max_capacity=5000)])
# mm_norm_env = VecNormalize.load("agents/ml_models/mm_vec_normalize_stats.pkl", mm_raw_env)
# mm_norm_env.training = False
# mm_norm_env.norm_reward = False
# mm_model = PPO.load("agents/ml_models/market_maker_us_base.zip")

# # 2. Load Institutional Whale Brain
# whale_raw_env = DummyVecEnv([lambda: SmartMoneyEnv(dummy_df, max_capacity=10000)])
# whale_norm_env = VecNormalize.load("agents/ml_models/sm_vec_normalize_stats.pkl", whale_raw_env)
# whale_norm_env.training = False
# whale_norm_env.norm_reward = False
# whale_model = PPO.load("agents/ml_models/smart_money_us_base.zip")

# # 3. Load Retail Brain
# retail_model = HeuristicRetailSwarm()

# print("[AI_LOADER] All Neural Networks loaded successfully!\n")

# def extract_features(tick_data, inventory, max_capacity):
#     """Safely extracts the 6 features regardless of how the Engine formats the JSON."""
#     ms = tick_data.get("market_state", tick_data)
    
#     vwap_dist = ms.get("vwap_dist", 0.01)
#     volatility = ms.get("volatility", 0.005)
#     volume_ratio = ms.get("volume_ratio", 1.0)
#     rsi_norm = ms.get("rsi_norm", 0.5)
#     ofi = ms.get("order_flow_imbalance", ms.get("ofi", 0.0))
#     pos_norm = inventory / max_capacity
    
#     return np.array([[vwap_dist, volatility, volume_ratio, rsi_norm, ofi, pos_norm]], dtype=np.float32)

# import random # Add this to the top of live_agents.py if not there

# class LiveWhaleAgent:
#     def __init__(self, agent_id):
#         self.agent_id = agent_id
#         self.inventory = 0
#         # PDF Specification: Whales between $5M - $20M
#         self.max_capacity = random.randint(5_000_000, 20_000_000)
#         self.is_asleep = False # For dynamic participation
        
#     def on_price_update(self, tick_data):
#         if self.is_asleep:
#             return []
            
#         raw_obs = extract_features(tick_data, self.inventory, self.max_capacity)
#         normalized_obs = whale_norm_env.normalize_obs(raw_obs)
        
#         action, _ = whale_model.predict(normalized_obs, deterministic=True)
#         target_pct = np.clip(action[0][0], -1.0, 1.0)
#         desired_inventory = int(target_pct * self.max_capacity)
#         shares_to_trade = desired_inventory - self.inventory
        
#         if shares_to_trade == 0:
#             return []
            
#         side = "BUY" if shares_to_trade > 0 else "SELL"
#         mid_price = float(tick_data.get("order_book", tick_data).get("last_traded_price", tick_data.get("mid_price", 2500)))
#         self.inventory += shares_to_trade 
        
#         return [{"agent_id": self.agent_id, "action": side, "price": float(mid_price), "qty": int(abs(shares_to_trade))}]

# class LiveMarketMaker:
#     def __init__(self, agent_id):
#         self.agent_id = agent_id
#         self.inventory = 0
#         # PDF Specification: Market Makers between $1M - $5M
#         self.max_capacity = random.randint(1_000_000, 5_000_000)
#         self.is_asleep = False
        
#     def on_price_update(self, tick_data):
#         if self.is_asleep:
#             return []
            
#         raw_obs = extract_features(tick_data, self.inventory, self.max_capacity)
#         normalized_obs = mm_norm_env.normalize_obs(raw_obs)
#         action, _ = mm_model.predict(normalized_obs, deterministic=True)
        
#         bid_bps = max(2.0, ((action[0][0] + 1) / 2) * 50)
#         ask_bps = max(2.0, ((action[0][1] + 1) / 2) * 50)
#         quote_qty = int(((action[0][2] + 1) / 2) * self.max_capacity)
        
#         if quote_qty == 0:
#             return []

#         mid_price = float(tick_data.get("order_book", tick_data).get("last_traded_price", tick_data.get("mid_price", 2500)))
#         bid_price = float(round(mid_price * (1 - (float(bid_bps) / 10000)), 2))
#         ask_price = float(round(mid_price * (1 + (float(ask_bps) / 10000)), 2))
        
#         return [
#             {"agent_id": self.agent_id, "action": "BUY", "price": bid_price, "qty": int(quote_qty), "type": "LIMIT"},
#             {"agent_id": self.agent_id, "action": "SELL", "price": ask_price, "qty": int(quote_qty), "type": "LIMIT"}
#         ]

# class LiveRetailAgent:
#     def __init__(self, agent_id):
#         self.agent_id = agent_id
#         # PDF Specification: Retail between $10k - $100k
#         self.cash = random.randint(10_000, 100_000)
#         self.inventory = 0
#         self.is_asleep = False
        
#     def on_price_update(self, tick_data):
#         if self.is_asleep or self.cash <= 1000: # Handled by liquidation loop next
#             return []
            
#         raw_obs = extract_features(tick_data, self.inventory, self.cash) 
#         action, _ = retail_model.predict(raw_obs)
        
#         target_pct = action[0][0]
#         if target_pct == 0.0:
#             return []
            
#         side = "BUY" if target_pct > 0 else "SELL"
#         mid_price = float(tick_data.get("order_book", tick_data).get("last_traded_price", tick_data.get("mid_price", 2500)))
        
#         # Simplified execution for retail: assume market order fills instantly at mid
#         qty = int(abs(target_pct) * (self.cash / mid_price) * 0.5) # Use max 50% of cash per trade
        
#         if qty == 0:
#             return []
            
#         if side == "BUY":
#             self.cash -= (qty * mid_price)
#             self.inventory += qty
#         elif side == "SELL" and self.inventory > 0:
#             qty = min(qty, self.inventory)
#             self.cash += (qty * mid_price)
#             self.inventory -= qty
        
#         return [{"agent_id": self.agent_id, "action": side, "price": float(mid_price), "qty": int(qty)}]
















































import random
import time
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.agents import SmartMoneyEnv, MarketMakerEnv
from agents.backtest_retail import HeuristicRetailSwarm

print("\n[AI_LOADER] Initializing AI Models into memory. Please wait...")

# =========================================================
# DUMMY DATA FOR ENVIRONMENT INITIALIZATION
# =========================================================

dummy_df = pd.DataFrame({
    'date': ['2026-01-01'] * 10,
    'time': ['09:30:00'] * 10,
    'Open': [100] * 10,
    'High': [100] * 10,
    'Low': [100] * 10,
    'Close': [100] * 10,
    'Volume': [1000] * 10,
    'VWAP': [100] * 10,
    'Volume_MA': [1000] * 10,
    'Order_Flow_Imbalance': [0] * 10,
    'RSI_14': [50] * 10
})

# =========================================================
# LOAD MARKET MAKER MODEL
# =========================================================

mm_raw_env = DummyVecEnv([
    lambda: MarketMakerEnv(dummy_df, max_capacity=5000)
])

mm_norm_env = VecNormalize.load(
    "agents/ml_models/mm_vec_normalize_stats.pkl",
    mm_raw_env
)

mm_norm_env.training = False
mm_norm_env.norm_reward = False

mm_model = PPO.load(
    "agents/ml_models/market_maker_us_base.zip"
)

# =========================================================
# LOAD WHALE MODEL
# =========================================================

whale_raw_env = DummyVecEnv([
    lambda: SmartMoneyEnv(dummy_df, max_capacity=10000)
])

whale_norm_env = VecNormalize.load(
    "agents/ml_models/sm_vec_normalize_stats.pkl",
    whale_raw_env
)

whale_norm_env.training = False
whale_norm_env.norm_reward = False

whale_model = PPO.load(
    "agents/ml_models/smart_money_us_base.zip"
)

# =========================================================
# LOAD RETAIL MODEL
# =========================================================

retail_model = HeuristicRetailSwarm()

print("[AI_LOADER] All Neural Networks loaded successfully!\n")


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(tick_data, inventory, max_capacity):

    ms = tick_data.get("market_state", tick_data)

    vwap_dist = ms.get("vwap_dist", 0.01)
    volatility = ms.get("volatility", 0.005)
    volume_ratio = ms.get("volume_ratio", 1.0)
    rsi_norm = ms.get("rsi_norm", 0.5)

    ofi = ms.get(
        "order_flow_imbalance",
        ms.get("ofi", 0.0)
    )

    pos_norm = inventory / max_capacity

    return np.array([
        [
            vwap_dist,
            volatility,
            volume_ratio,
            rsi_norm,
            ofi,
            pos_norm
        ]
    ], dtype=np.float32)


# =========================================================
# WHALE AGENT
# =========================================================

class LiveWhaleAgent:

    def __init__(self, agent_id):

        self.agent_id = agent_id

        self.inventory = 0

        # REALISTIC SIZE
        self.max_capacity = random.randint(
            50000,
            200000
        )

        self.is_asleep = False

        self.last_trade = 0

    def on_price_update(self, tick_data):

        if self.is_asleep:
            return []

        # COOLDOWN
        if time.time() - self.last_trade < 2:
            return []

        raw_obs = extract_features(
            tick_data,
            self.inventory,
            self.max_capacity
        )

        normalized_obs = whale_norm_env.normalize_obs(raw_obs)

        action, _ = whale_model.predict(
            normalized_obs,
            deterministic=True
        )

        target_pct = np.clip(
            action[0][0],
            -1.0,
            1.0
        )

        desired_inventory = int(
            target_pct * self.max_capacity
        )

        shares_to_trade = desired_inventory - self.inventory

        # CLAMP TRADE SIZE
        shares_to_trade = max(
            -10000,
            min(shares_to_trade, 10000)
        )

        if shares_to_trade == 0:
            return []

        side = "BUY" if shares_to_trade > 0 else "SELL"

        mid_price = float(
            tick_data.get("order_book", tick_data)
            .get(
                "last_traded_price",
                tick_data.get("mid_price", 2500)
            )
        )

        self.inventory += shares_to_trade

        self.last_trade = time.time()

        return [
            {
                "agent_id": self.agent_id,
                "action": side,
                "price": round(mid_price, 2),
                "qty": int(abs(shares_to_trade)),
                "type": "MARKET"
            }
        ]


# =========================================================
# MARKET MAKER AGENT
# =========================================================

class LiveMarketMaker:

    def __init__(self, agent_id):

        self.agent_id = agent_id

        self.inventory = 0

        # REDUCED INVENTORY
        self.max_capacity = random.randint(
            5000,
            25000
        )

        self.is_asleep = False

        self.last_trade = 0

    def on_price_update(self, tick_data):

        if self.is_asleep:
            return []

        # COOLDOWN
        if time.time() - self.last_trade < 1:
            return []

        raw_obs = extract_features(
            tick_data,
            self.inventory,
            self.max_capacity
        )

        normalized_obs = mm_norm_env.normalize_obs(raw_obs)

        action, _ = mm_model.predict(
            normalized_obs,
            deterministic=True
        )

        mid_price = float(
            tick_data.get("order_book", tick_data)
            .get(
                "last_traded_price",
                tick_data.get("mid_price", 2500)
            )
        )

        # REALISTIC SPREADS
        bid_offset = random.uniform(0.001, 0.003)
        ask_offset = random.uniform(0.001, 0.003)

        bid_price = round(
            mid_price * (1 - bid_offset),
            2
        )

        ask_price = round(
            mid_price * (1 + ask_offset),
            2
        )

        # CONTROLLED QUANTITY
        raw_qty = ((action[0][2] + 1) / 2)

        quote_qty = int(
            max(
                10,
                min(raw_qty * 1000, 3000)
            )
        )

        self.last_trade = time.time()

        return [
            {
                "agent_id": self.agent_id,
                "action": "BUY",
                "price": bid_price,
                "qty": quote_qty,
                "type": "LIMIT"
            },
            {
                "agent_id": self.agent_id,
                "action": "SELL",
                "price": ask_price,
                "qty": quote_qty,
                "type": "LIMIT"
            }
        ]


# =========================================================
# RETAIL AGENT
# =========================================================

class LiveRetailAgent:

    def __init__(self, agent_id):

        self.agent_id = agent_id

        self.cash = random.randint(
            10000,
            100000
        )

        self.inventory = 0

        self.is_asleep = False

        self.last_trade = 0

    def on_price_update(self, tick_data):

        if self.is_asleep:
            return []

        # COOLDOWN
        if time.time() - self.last_trade < random.uniform(2, 5):
            return []

        raw_obs = extract_features(
            tick_data,
            self.inventory,
            self.cash
        )

        action, _ = retail_model.predict(raw_obs)

        target_pct = action[0][0]

        if target_pct == 0.0:
            return []

        side = "BUY" if target_pct > 0 else "SELL"

        mid_price = float(
            tick_data.get("order_book", tick_data)
            .get(
                "last_traded_price",
                tick_data.get("mid_price", 2500)
            )
        )

        # SMALL RETAIL SIZE
        qty = random.randint(1, 20)

        if side == "BUY":

            cost = qty * mid_price

            if self.cash < cost:
                return []

            self.cash -= cost
            self.inventory += qty

        else:

            if self.inventory <= 0:
                return []

            qty = min(qty, self.inventory)

            self.cash += qty * mid_price
            self.inventory -= qty

        self.last_trade = time.time()

        return [
            {
                "agent_id": self.agent_id,
                "action": side,
                "price": round(mid_price, 2),
                "qty": qty,
                "type": "MARKET"
            }
        ]