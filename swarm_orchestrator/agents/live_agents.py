"""Live swarm agents: PPO market makers, PPO whales, heuristic retail.

Models are loaded ONCE at import time (module level) and shared by every agent
instance in the process. Order dicts returned by `on_tick` follow PROTOCOL.md
§4.1 minus the `msg`/`agent_id` fields (the orchestrator injects those), plus
one sentinel: `{"cancel_all": True}` which the orchestrator translates to a
CANCEL_ALL before that agent's subsequent orders.

Single-truth ledger (PROTOCOL.md §4.2): agents NEVER book their own fills.
`on_ack` copies `cash`/`pos` from the engine ack — the engine ledger is
authoritative, which eliminates the Phase-1 partial-fill inventory drift.

Sanctioned randomness (PROTOCOL.md §8): capacity/cash draws at construction,
retail cooldown jitter, and the retail heuristic's 10% noise. Nothing else.
"""

import logging
import os
import random
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.environments import MarketMakerEnv, SmartMoneyEnv
from agents.features import build_observation
from agents.heuristic_retail import HeuristicRetailSwarm

logger = logging.getLogger(__name__)

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models")

logger.info("Loading AI models into memory (once per process)...")

# ---------------------------------------------------------------------------
# Constructor stub dataframe.
#
# The gym environments require a dataframe in their constructor, but here they
# exist ONLY as VecNormalize wrappers around the trained normalisation stats —
# they are never reset or stepped, and this frame never generates market data.
# This is NOT a No-Dummy violation (PROTOCOL.md §8): it is a required
# constructor stub, not a data source.
# ---------------------------------------------------------------------------
_dummy_df = pd.DataFrame({
    "date": ["2026-01-01"] * 10,
    "time": ["09:30:00"] * 10,
    "Open": [100.0] * 10,
    "High": [100.0] * 10,
    "Low": [100.0] * 10,
    "Close": [100.0] * 10,
    "Volume": [1000] * 10,
    "VWAP": [100.0] * 10,
    "Volume_MA": [1000.0] * 10,
    "Order_Flow_Imbalance": [0.0] * 10,
    "RSI_14": [50.0] * 10,
})

# --- Market Maker brain (PPO + frozen VecNormalize stats) ---
_mm_raw_env = DummyVecEnv([lambda: MarketMakerEnv(_dummy_df, max_capacity=5000)])
mm_norm_env = VecNormalize.load(
    os.path.join(_MODELS_DIR, "mm_vec_normalize_stats.pkl"), _mm_raw_env
)
mm_norm_env.training = False
mm_norm_env.norm_reward = False
mm_model = PPO.load(os.path.join(_MODELS_DIR, "market_maker_us_base.zip"))

# --- Institutional Whale brain (PPO + frozen VecNormalize stats) ---
_whale_raw_env = DummyVecEnv([lambda: SmartMoneyEnv(_dummy_df, max_capacity=10000)])
whale_norm_env = VecNormalize.load(
    os.path.join(_MODELS_DIR, "sm_vec_normalize_stats.pkl"), _whale_raw_env
)
whale_norm_env.training = False
whale_norm_env.norm_reward = False
whale_model = PPO.load(os.path.join(_MODELS_DIR, "smart_money_us_base.zip"))

# --- Retail brain (rule-based, no normalisation) ---
retail_model = HeuristicRetailSwarm()

logger.info("All agent brains loaded.")


def _mid_price(tick: dict) -> float:
    """Reference price for quoting: engine mid, falling back to last trade."""
    mid = tick.get("mid_price")
    if mid:
        return float(mid)
    return float(tick["last_price"])


class _LiveAgentBase:
    """Common ledger plumbing for every live agent.

    Attributes:
        agent_id: engine-facing identity (PROTOCOL.md §4.1 `agent_id`).
        cash / inventory: mirror of the ENGINE ledger, updated only via acks.
        is_asleep: dynamic-participation flag set by the orchestrator.
        last_trade: wall-clock time of the last emitted action (cooldowns).
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.cash: float = 0.0
        self.inventory: int = 0
        self.is_asleep: bool = False
        self.last_trade: float = 0.0

    def on_tick(self, tick: dict) -> "list[dict]":
        """Return this tick's orders (§4.1 minus msg/agent_id). Override."""
        raise NotImplementedError

    def on_ack(self, ack: dict) -> None:
        """Adopt the engine's post-action ledger (§4.2 — authoritative)."""
        cash = ack.get("cash")
        if cash is not None:
            self.cash = float(cash)
        pos = ack.get("pos")
        if pos is not None:
            self.inventory = int(pos)


class LiveMarketMaker(_LiveAgentBase):
    """PPO liquidity provider: model-driven cancel/replace two-sided quoting.

    Uses the REAL 3-D model output [bid_offset, ask_offset, quote_size]
    (fixes flaws R6/M5 where the model's spread was discarded for random
    offsets). Refreshes quotes at most once per second, like a real MM's
    throttled cancel/replace cycle.
    """

    COOLDOWN_S = 1.0
    MAX_BPS = 50.0
    MIN_BPS = 2.0
    QUOTE_TIF_TICKS = 5

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id)
        self.max_capacity = random.randint(5_000, 25_000)

    def on_tick(self, tick: dict) -> "list[dict]":
        if self.is_asleep:
            return []
        now = time.time()
        if now - self.last_trade < self.COOLDOWN_S:
            return []

        obs = build_observation(tick, self.inventory, self.max_capacity)
        norm_obs = mm_norm_env.normalize_obs(obs)
        action, _ = mm_model.predict(norm_obs, deterministic=True)
        a = action[0]

        bid_bps = max(self.MIN_BPS, ((float(a[0]) + 1.0) / 2.0) * self.MAX_BPS)
        ask_bps = max(self.MIN_BPS, ((float(a[1]) + 1.0) / 2.0) * self.MAX_BPS)
        quote_qty = int(max(10.0, min(((float(a[2]) + 1.0) / 2.0) * self.max_capacity, 3000.0)))

        mid = _mid_price(tick)
        bid_price = round(mid * (1.0 - bid_bps / 10_000.0), 2)
        ask_price = round(mid * (1.0 + ask_bps / 10_000.0), 2)

        self.last_trade = now
        return [
            {"cancel_all": True},
            {"action": "BUY", "type": "LIMIT", "price": bid_price,
             "qty": quote_qty, "tif_ticks": self.QUOTE_TIF_TICKS},
            {"action": "SELL", "type": "LIMIT", "price": ask_price,
             "qty": quote_qty, "tif_ticks": self.QUOTE_TIF_TICKS},
        ]


class LiveWhaleAgent(_LiveAgentBase):
    """PPO institutional whale: model-driven target position via MARKET orders.

    Inventory truth comes exclusively from engine acks (`on_ack` sets
    `inventory = ack['pos']`), so partial fills can never drift the whale's
    view of its own position.
    """

    COOLDOWN_S = 2.0
    MAX_CLIP = 10_000  # per-action share clamp

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id)
        self.max_capacity = random.randint(50_000, 200_000)

    def on_tick(self, tick: dict) -> "list[dict]":
        if self.is_asleep:
            return []
        now = time.time()
        if now - self.last_trade < self.COOLDOWN_S:
            return []

        obs = build_observation(tick, self.inventory, self.max_capacity)
        norm_obs = whale_norm_env.normalize_obs(obs)
        action, _ = whale_model.predict(norm_obs, deterministic=True)

        target_pct = float(np.clip(action[0][0], -1.0, 1.0))
        desired = int(target_pct * self.max_capacity)
        delta = desired - self.inventory
        delta = max(-self.MAX_CLIP, min(delta, self.MAX_CLIP))
        if delta == 0:
            return []

        self.last_trade = now
        side = "BUY" if delta > 0 else "SELL"
        return [{"action": side, "type": "MARKET", "qty": abs(delta)}]


class LiveRetailAgent(_LiveAgentBase):
    """Heuristic retail trader with conviction-scaled sizing.

    Sizing is conviction * 50% of cash (capped 1..100 shares) — replaces the
    Phase-1 'random 1-20 shares' dummy. Cash/inventory mirror the engine ledger
    via acks; on an engine liquidation event the orchestrator calls
    `on_liquidation` to mirror the engine's ledger reset (PROTOCOL.md §8.6).
    Cooldown jitter and the heuristic's internal 10% noise are sanctioned
    behavioural randomness (§8).
    """

    LIQUIDATION_RESET_CASH = 50_000.0  # engine resets retail to ₹50k / 0 pos
    MAX_QTY = 100

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id)
        self.cash = float(random.randint(10_000, 100_000))
        self._cooldown_s = random.uniform(2.0, 5.0)

    def on_liquidation(self) -> None:
        """Mirror the engine's liquidation reset (engine is authoritative)."""
        self.cash = self.LIQUIDATION_RESET_CASH
        self.inventory = 0

    def on_tick(self, tick: dict) -> "list[dict]":
        if self.is_asleep:
            return []
        now = time.time()
        if now - self.last_trade < self._cooldown_s:
            return []

        last_price = float(tick["last_price"])
        if last_price <= 0:
            return []

        # Retail normalises position by its own bankroll (capacity proxy).
        obs = build_observation(tick, self.inventory, max(self.cash, 1.0))
        action, _ = retail_model.predict(obs)
        raw = float(action[0][0])
        conviction = abs(raw)
        if conviction == 0.0:
            return []

        qty = max(1, min(self.MAX_QTY, int(conviction * self.cash * 0.5 / last_price)))
        if raw > 0:
            side = "BUY"
            if self.cash < qty * last_price:
                return []  # cannot cover the buy
        else:
            side = "SELL"
            if self.inventory <= 0:
                return []  # nothing to sell (no naked retail shorts)
            qty = min(qty, self.inventory)

        self.last_trade = now
        self._cooldown_s = random.uniform(2.0, 5.0)  # sanctioned jitter (§8)
        return [{"action": side, "type": "MARKET", "qty": qty}]
