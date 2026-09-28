import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

class UniversalMarketEnv(gym.Env):
    """
    The Base Physics Engine. Handles data loading, random day sampling,
    and portfolio math.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, df: pd.DataFrame, initial_capital=1000000.0, episode_length=390):
        super(UniversalMarketEnv, self).__init__()

        self.initial_capital = initial_capital
        self.episode_length = episode_length

        # Filter out incomplete trading days
        day_counts = df['date'].value_counts()
        valid_dates = day_counts[day_counts >= self.episode_length].index
        self.df = df[df['date'].isin(valid_dates)].copy()

        self.unique_dates = self.df['date'].unique()

        # Scale-Invariant Observation Space (Percentages and Ratios only)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # START AT 1: The agent looks at T-1, and trades on T.
        self.current_step = 1
        self.day_data = None
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.position = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        random_date = random.choice(self.unique_dates)
        self.day_data = self.df[self.df['date'] == random_date].copy()
        self.day_data.reset_index(drop=True, inplace=True)

        # Reset to Step 1 so we have a Step 0 to look at
        self.current_step = 1
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.position = 0

        return self._get_obs(), {}

    def _get_obs(self):
        """Translates T-1 data into Neural-Network friendly normalized values."""
        # CRITICAL FIX: The agent only sees the minute that just finished
        row = self.day_data.iloc[self.current_step - 1]

        vwap_dist = (row['Close'] - row['VWAP']) / row['VWAP'] if row['VWAP'] > 0 else 0.0
        volatility = (row['High'] - row['Low']) / row['Close']
        volume_ratio = row['Volume'] / row['Volume_MA'] if row.get('Volume_MA', 0) > 0 else 1.0
        rsi_norm = row.get('RSI_14', 50.0) / 100.0
        ofi = row.get('Order_Flow_Imbalance', 0.0)
        port_return = (self.portfolio_value - self.initial_capital) / self.initial_capital

        obs = np.array([
            vwap_dist,
            volatility,
            volume_ratio,
            rsi_norm,
            ofi,
            port_return
        ], dtype=np.float32)

        return np.nan_to_num(obs)
