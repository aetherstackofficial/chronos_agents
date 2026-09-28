"""Rule-based retail crowd model.

This module contains only the live inference brain for the retail swarm.
Offline backtests of this heuristic (CSV-driven evaluation harnesses) live in
the training repository, not here — the live swarm only needs `predict()`.
"""

import numpy as np


class HeuristicRetailSwarm:
    """A hardcoded, rule-based agent that simulates the chaotic,
    emotional trading of a retail crowd. Zero training required.

    The 10% pure-noise rule is sanctioned behavioural randomness per
    PROTOCOL.md §8 (agent behavioural noise only — never price noise).
    """

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple:
        """Mimics the stable-baselines3 predict API: (action, state).

        `obs` is a batched (1, 6) observation; `deterministic` is accepted for
        API compatibility but the 10% noise rule always applies (sanctioned).
        """
        # DummyVecEnv wraps obs in an extra array, so we grab index 0
        obs_data = obs[0]

        # Unpack the normalized observations from UniversalMarketEnv layout
        vwap_dist = obs_data[0]
        volatility = obs_data[1]
        volume_ratio = obs_data[2]
        rsi_norm = obs_data[3]  # 0.0 to 1.0 (50 RSI = 0.5)
        ofi = obs_data[4]       # Order Flow Imbalance (-1.0 to 1.0)

        action = 0.0  # Default to flat

        # Rule 1: The Basic Technical Analyst (Buy oversold, sell overbought)
        if rsi_norm < 0.30:
            action = 0.5   # Buy with 50% capacity
        elif rsi_norm > 0.70:
            action = -0.5  # Short with 50% capacity

        # Rule 2: The FOMO & Panic Traders (Overrides RSI)
        # If order flow is extremely toxic one way, retail piles in emotionally
        if ofi > 0.8:
            action = 0.8   # Aggressive FOMO Buy
        elif ofi < -0.8:
            action = -1.0  # Absolute Panic Sell

        # Rule 3: Pure Noise (10% of the time, the swarm does something random)
        if np.random.rand() < 0.10:
            action = np.random.uniform(-0.5, 0.5)

        return np.array([[action]]), None
