import logging
import os
import random
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.environments import MarketMakerEnv, SmartMoneyEnv
from agents.execution import PovSlicer, skew_ticks, snap
from agents.features import build_observation
from agents.heuristic_retail import HeuristicRetailSwarm

logger = logging.getLogger(__name__)
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models")

_dummy_df = pd.DataFrame({k: v * 10 for k, v in {
    "date": ["2026-01-01"], "time": ["09:30:00"], "Open": [100.0], "High": [100.0],
    "Low": [100.0], "Close": [100.0], "Volume": [1000], "VWAP": [100.0],
    "Volume_MA": [1000.0], "Order_Flow_Imbalance": [0.0], "RSI_14": [50.0]
}.items()})

mm_norm_env = VecNormalize.load(os.path.join(_MODELS_DIR, "mm_vec_normalize_stats.pkl"), DummyVecEnv([lambda: MarketMakerEnv(_dummy_df, max_capacity=5000)]))
mm_norm_env.training = mm_norm_env.norm_reward = False
mm_model = PPO.load(os.path.join(_MODELS_DIR, "market_maker_us_base.zip"))

whale_norm_env = VecNormalize.load(os.path.join(_MODELS_DIR, "sm_vec_normalize_stats.pkl"), DummyVecEnv([lambda: SmartMoneyEnv(_dummy_df, max_capacity=10000)]))
whale_norm_env.training = whale_norm_env.norm_reward = False
whale_model = PPO.load(os.path.join(_MODELS_DIR, "smart_money_us_base.zip"))

retail_model = HeuristicRetailSwarm()

def _mid_price(tick: dict) -> float:
    return float(tick.get("mid_price") or tick["last_price"])


class _LiveAgentBase:
    max_position: int = 0

    def __init__(self, agent_id: str, clock=None) -> None:
        self.agent_id = agent_id
        # last_trade = -inf: "never traded", so a cooldown never depends on
        # where the clock happens to start.
        self.cash, self.inventory, self.is_asleep, self.last_trade = 0.0, 0, False, float("-inf")
        self.start_equity: float | None = None
        # agents.swarm_factory injects a TickClock; None reads wall time.
        self._clock = clock

    def _now(self) -> float:
        return self._clock() if self._clock is not None else time.time()

    def on_tick(self, tick: dict) -> list[dict]:
        raise NotImplementedError

    def on_ack(self, ack: dict) -> None:
        if (cash := ack.get("cash")) is not None: self.cash = float(cash)
        if (pos := ack.get("pos")) is not None: self.inventory = int(pos)

    def _port_return(self, tick: dict) -> float:
        equity = self.cash + self.inventory * float(tick["last_price"])
        if self.start_equity is None:
            if self.cash <= 0: return 0.0
            self.start_equity = equity
        return (equity - self.start_equity) / self.start_equity if self.start_equity else 0.0

    def _cap_qty(self, side: str, qty: int) -> int:
        if self.max_position <= 0 or qty <= 0: return max(0, qty)
        return max(0, min(qty, int(self.max_position - self.inventory if side == "BUY" else self.max_position + self.inventory)))


class LiveMarketMaker(_LiveAgentBase):
    COOLDOWN_S, MAX_BPS, MIN_BPS, QUOTE_TIF_TICKS = 1.0, 50.0, 2.0, 5

    def __init__(self, agent_id: str, clock=None, rng=None) -> None:
        super().__init__(agent_id, clock)
        self.max_position = self.max_capacity = (rng or random).randint(5_000, 25_000)

    def on_tick(self, tick: dict) -> list[dict]:
        if self.is_asleep or (now := self._now()) - self.last_trade < self.COOLDOWN_S: return []

        obs = mm_norm_env.normalize_obs(build_observation(tick, self._port_return(tick)))
        a = mm_model.predict(obs, deterministic=True)[0][0]
        
        bid_bps = max(self.MIN_BPS, ((float(a[0]) + 1.0) / 2.0) * self.MAX_BPS)
        ask_bps = max(self.MIN_BPS, ((float(a[1]) + 1.0) / 2.0) * self.MAX_BPS)
        quote_qty = int(max(10.0, min(((float(a[2]) + 1.0) / 2.0) * self.max_capacity, 3000.0)))

        mid = _mid_price(tick)
        bid_qty, ask_qty = self._cap_qty("BUY", quote_qty), self._cap_qty("SELL", quote_qty)

        # Phase 4 inventory skew (uncalibrated fixed rule): both quotes move
        # against inventory by up to 3 ticks, then snap to the engine tick.
        tick_size = float(tick.get("tick_size") or 0.0)
        self.last_skew_ticks = skew_ticks(self.inventory, self.max_position)
        shift = self.last_skew_ticks * tick_size

        self.last_trade = now
        orders: list[dict] = [{"cancel_all": True}]
        if bid_qty > 0: orders.append({"action": "BUY", "type": "LIMIT", "price": snap(mid * (1.0 - bid_bps / 10_000.0) + shift, tick_size), "qty": bid_qty, "tif_ticks": self.QUOTE_TIF_TICKS})
        if ask_qty > 0: orders.append({"action": "SELL", "type": "LIMIT", "price": snap(mid * (1.0 + ask_bps / 10_000.0) + shift, tick_size), "qty": ask_qty, "tif_ticks": self.QUOTE_TIF_TICKS})
        return orders


class LiveWhaleAgent(_LiveAgentBase):
    COOLDOWN_S, MAX_CLIP = 2.0, 10_000

    def __init__(self, agent_id: str, clock=None, rng=None, slicer: PovSlicer | None = None) -> None:
        super().__init__(agent_id, clock)
        self.max_position = self.max_capacity = (rng or random).randint(50_000, 200_000)
        # Phase 4: the policy sets a target; the slicer works it as POV
        # children (agents/execution.py, agents/config/phase4_execution.yaml).
        self.slicer = slicer or PovSlicer(agent_id)

    def on_ack(self, ack: dict) -> None:
        super().on_ack(ack)
        self.slicer.on_ack(ack)

    def on_tick(self, tick: dict) -> list[dict]:
        if self.is_asleep: return []
        orders: list[dict] = []
        day, bar = int(tick["day_count"]), int(tick["market_minute"])

        if (now := self._now()) - self.last_trade >= self.COOLDOWN_S and bar < self.slicer.last_child_bar:
            obs = whale_norm_env.normalize_obs(build_observation(tick, self._port_return(tick)))
            action = whale_model.predict(obs, deterministic=True)[0]

            delta = int(float(np.clip(action[0][0], -1.0, 1.0)) * self.max_capacity) - self.inventory
            delta = max(-self.MAX_CLIP, min(delta, self.MAX_CLIP))
            qty = self._cap_qty(side := "BUY" if delta > 0 else "SELL", abs(delta)) if delta else 0
            if qty:
                self.last_trade = now
                orders += self.slicer.retarget(self.inventory + (qty if side == "BUY" else -qty), side, day, bar)
            elif self.slicer.parent is not None:
                # The policy now wants no change: stop the parent where it is.
                self.slicer.parent.target_pos = self.inventory
                self.slicer.parent.revisions += 1

        return orders + self.slicer.tick_orders(tick)


class LiveRetailAgent(_LiveAgentBase):
    LIQUIDATION_RESET_CASH, MAX_QTY = 50_000.0, 100

    def __init__(self, agent_id: str, clock=None, rng=None) -> None:
        super().__init__(agent_id, clock)
        self._rng = rng or random
        self.cash, self._cooldown_s = float(self._rng.randint(10_000, 100_000)), self._rng.uniform(2.0, 5.0)

    def on_liquidation(self) -> None:
        self.cash, self.inventory = self.LIQUIDATION_RESET_CASH, 0

    def on_tick(self, tick: dict) -> list[dict]:
        if self.is_asleep or (now := self._now()) - self.last_trade < self._cooldown_s or (last_price := float(tick["last_price"])) <= 0:
            return []

        action, _ = retail_model.predict(build_observation(tick, self._port_return(tick)), rng=self._rng)
        if not (conviction := abs(raw := float(action[0][0]))):
            return []

        qty = max(1, min(self.MAX_QTY, int(conviction * self.cash * 0.5 / last_price)))
        
        if raw > 0:
            if self.cash < qty * last_price: return []
            side = "BUY"
        else:
            if self.inventory <= 0: return []
            qty, side = min(qty, self.inventory), "SELL"

        self.last_trade, self._cooldown_s = now, self._rng.uniform(2.0, 5.0)
        return [{"action": side, "type": "MARKET", "qty": qty}]
        