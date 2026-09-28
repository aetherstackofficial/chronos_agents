from agents.base_environment import UniversalMarketEnv
from gymnasium import spaces
import numpy as np


class SmartMoneyEnv(UniversalMarketEnv):
    """
    The Institutional Whale — v2
    Hysteresis deadzone, normalized rewards, progressive drawdown penalty,
    inaction nudge, and carry bonus. Exploit-free and jitter-proof.
    """

    # ── Deadzone thresholds ──────────────────────────────────────────────────
    # Two separate values create the hysteresis band:
    #   Signal must exceed ENTRY_THRESH to open a position.
    #   Signal must fall below EXIT_THRESH before the agent is forced flat.
    # The gap between them (0.05 to 0.20) is the "sticky zone" where the agent
    # keeps whatever position it already has, without paying entry/exit fees.
    ENTRY_THRESH = 0.20   # must exceed this to initiate a trade
    EXIT_THRESH  = 0.05   # must fall below this to be forced flat

    # ── Reward scaling ───────────────────────────────────────────────────────
    CARRY_BONUS_SCALE  = 0.10   # fraction of step PnL added as carry bonus
    INACTION_PENALTY   = 0.002  # penalty (as fraction of initial_capital) per
                                # flat step when there was a tradeable signal
    DD_PENALTY_ONSET   = 0.05   # drawdown fraction at which penalties start
    DD_PENALTY_SCALE   = 5.0    # multiplier on excess drawdown for penalty
    TERMINAL_DD        = 0.30   # hard stop: terminate episode
    TERMINAL_PENALTY   = 300.0  # fixed penalty (in normalized units × 1000)

    def __init__(self, df, max_capacity=10_000):
        super().__init__(df)
        self.max_capacity   = max_capacity
        self.action_space   = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        # Tracks whether the agent is currently in the "sticky zone"
        self._in_position_band = False

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _apply_hysteresis(self, raw_pct: float) -> float:
        """
        Hysteresis gate.

        States:
          • Flat  → only open if |raw_pct| ≥ ENTRY_THRESH
          • Open  → stay open until |raw_pct| <  EXIT_THRESH
          • Sticky zone (EXIT_THRESH ≤ |raw_pct| < ENTRY_THRESH):
              keep whatever band state we're already in.
        """
        abs_pct = abs(raw_pct)

        if self._in_position_band:
            # Already in a position — only exit band if signal collapses
            if abs_pct < self.EXIT_THRESH:
                self._in_position_band = False
                return 0.0
        else:
            # Currently flat — only enter if conviction is strong enough
            if abs_pct >= self.ENTRY_THRESH:
                self._in_position_band = True
            else:
                return 0.0  # stay flat

        return raw_pct

    def _normalize(self, dollar_value: float) -> float:
        """Convert a dollar PnL to a fraction of initial capital."""
        return dollar_value / self.initial_capital

    def _drawdown_penalty(self) -> float:
        """
        Progressive penalty that grows as drawdown deepens.
        Returns a NEGATIVE normalized value.
        """
        dd = 1.0 - (self.portfolio_value / self.initial_capital)
        excess = dd - self.DD_PENALTY_ONSET
        if excess <= 0:
            return 0.0
        return -self.DD_PENALTY_SCALE * (excess ** 2)

    # ── Main step ────────────────────────────────────────────────────────────

    def step(self, action):
        row      = self.day_data.iloc[self.current_step]
        prev_row = self.day_data.iloc[self.current_step - 1]

        base_exec_price = row["Open"]
        prev_value      = self.portfolio_value

        # ── 1. Decode action through hysteresis gate ─────────────────────────
        raw_pct    = float(np.clip(action[0], -1.0, 1.0))
        target_pct = self._apply_hysteresis(raw_pct)

        desired_shares  = int(target_pct * self.max_capacity)
        shares_to_trade = desired_shares - self.position

        # ── 2. Anti-jitter: ignore micro-adjustments < 5 % of capacity ───────
        if abs(shares_to_trade) < int(self.max_capacity * 0.05):
            shares_to_trade = 0

        # ── 3. Volume cap (15 % participation limit) ──────────────────────────
        actual_volume = row.get("Volume", 1e-5)
        max_tradeable = int(actual_volume * 0.15)

        if abs(shares_to_trade) > max_tradeable:
            shares_to_trade = (
                max_tradeable if shares_to_trade > 0 else -max_tradeable
            )

        # ── 4. Execution with market impact ───────────────────────────────────
        traded_this_step = shares_to_trade != 0
        if traded_this_step:
            participation   = abs(shares_to_trade) / actual_volume
            slippage_bps    = (participation / 0.15) * 10.0
            price_impact    = base_exec_price * (slippage_bps / 10_000.0)

            actual_exec_price = (
                base_exec_price + price_impact
                if shares_to_trade > 0
                else base_exec_price - price_impact
            )

            trade_value = abs(shares_to_trade) * actual_exec_price
            commission  = trade_value * 0.0003

            self.position += shares_to_trade
            if shares_to_trade > 0:
                self.cash -= trade_value + commission
            else:
                self.cash += trade_value - commission

        # ── 5. Mark-to-market ─────────────────────────────────────────────────
        current_close       = row["Close"]
        self.portfolio_value = self.cash + self.position * current_close

        # ── 6. Reward — normalized delta PnL ──────────────────────────────────
        step_pnl = self._normalize(self.portfolio_value - prev_value)
        reward   = step_pnl

        # ── 6a. Carry bonus ───────────────────────────────────────────────────
        # Reward the agent for *holding* a position that moves in its favour.
        # This encourages trend-riding rather than quick in-and-out trades.
        if self.position != 0 and not traded_this_step:
            position_direction = 1 if self.position > 0 else -1
            price_move         = current_close - prev_row["Close"]
            pnl_direction      = 1 if price_move * position_direction > 0 else -1
            carry              = self.CARRY_BONUS_SCALE * abs(step_pnl) * pnl_direction
            reward            += carry

        # ── 6b. Inaction penalty ──────────────────────────────────────────────
        # If the agent is flat but the raw signal was strong enough to trade,
        # apply a small nudge so it doesn't hide behind entry costs forever.
        if self.position == 0 and not traded_this_step:
            if abs(raw_pct) >= self.ENTRY_THRESH:
                reward -= self.INACTION_PENALTY

        # ── 6c. Progressive drawdown penalty ──────────────────────────────────
        reward += self._drawdown_penalty()

        # ── 7. Progression ────────────────────────────────────────────────────
        self.current_step += 1
        truncated  = self.current_step >= (self.episode_length - 1)
        terminated = (
            1.0 - self.portfolio_value / self.initial_capital
        ) >= self.TERMINAL_DD

        if terminated:
            reward -= self.TERMINAL_PENALTY / 1000.0  # keep in normalized scale

        # ── 8. EOD forced exit (exploit-free) ─────────────────────────────────
        if truncated and self.position != 0:
            trade_value          = abs(self.position) * current_close
            commission           = trade_value * 0.0003
            self.cash           += self.position * current_close - commission
            self.position        = 0
            self.portfolio_value = self.cash
            self._in_position_band = False

            # Recalculate reward so the agent feels the forced-exit commission
            reward = self._normalize(self.portfolio_value - prev_value)
            reward += self._drawdown_penalty()

        info = {
            "position":        self.position,
            "portfolio_value": self.portfolio_value,
            "drawdown":        1.0 - self.portfolio_value / self.initial_capital,
            "in_band":         self._in_position_band,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._in_position_band = False
        return obs, info
    
class MarketMakerEnv(UniversalMarketEnv):
    """
    High-Frequency Liquidity Provider.
    Upgraded: Eliminates Lookahead Bias, enforces tick-size rounding, and strictly simulates queue position.
    """
    def __init__(self, df, max_capacity=5000, max_bps_spread=50, min_bps_spread=2.0, tick_size=0.05):
        super().__init__(df)
        self.max_capacity = max_capacity
        
        self.max_bps_spread = max_bps_spread
        self.min_bps_spread = min_bps_spread 
        self.tick_size = tick_size # NSE standard tick size
        
        # Action: [Bid_Offset, Ask_Offset, Quote_Size]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.lambda_penalty = 500.0 
        self.panic_multiplier = 3.0

    def step(self, action):
        # row = The candle we are executing IN (Minute T)
        # prev_row = The candle the agent SAW (Minute T-1)
        row = self.day_data.iloc[self.current_step]
        prev_row = self.day_data.iloc[self.current_step - 1]
        
        # Agent places quotes based on the previous close
        ref_mid_price = prev_row['Close']
        
        # 1. Enforce a Minimum Spread Floor
        raw_bid_bps = ((action[0] + 1) / 2) * self.max_bps_spread
        raw_ask_bps = ((action[1] + 1) / 2) * self.max_bps_spread
        
        bid_bps = max(self.min_bps_spread, raw_bid_bps)
        ask_bps = max(self.min_bps_spread, raw_ask_bps)
        quote_qty = int(((action[2] + 1) / 2) * self.max_capacity)
        
        # 2. Tick Size Rounding (The Phantom Touch Fix)
        bid_price = round((ref_mid_price * (1 - (bid_bps / 10000))) / self.tick_size) * self.tick_size
        ask_price = round((ref_mid_price * (1 + (ask_bps / 10000))) / self.tick_size) * self.tick_size

        # 3. Execution Physics against Minute T
        shares_bought = 0
        shares_sold = 0
        eps = 0.001 # Buffer for floating point equality
        
        # Buy Fill Logic
        if row['Low'] < (bid_price - eps):
            shares_bought = min(quote_qty, int(row['Volume'] * 0.20))
        elif abs(row['Low'] - bid_price) <= eps:
            shares_bought = min(quote_qty, int(row['Volume'] * 0.02))

        # Sell Fill Logic
        if row['High'] > (ask_price + eps):
            shares_sold = min(quote_qty, int(row['Volume'] * 0.20))
        elif abs(row['High'] - ask_price) <= eps:
            shares_sold = min(quote_qty, int(row['Volume'] * 0.02))

        # 4. Portfolio Math
        prev_value = self.portfolio_value
        self.cash += (shares_sold * ask_price) - (shares_bought * bid_price)
        self.position += (shares_bought - shares_sold)
        
        # Mark to market using the newly closed candle (Minute T)
        current_close = row['Close']
        self.portfolio_value = self.cash + (self.position * current_close)
        
        # 5. Reward Calculation
        delta_pnl = self.portfolio_value - prev_value
        inventory_pct_squared = (abs(self.position) / self.max_capacity) ** 2
        
        ofi = row.get('Order_Flow_Imbalance', 0.0)
        is_underwater = (self.position > 0 and ofi < -0.5) or (self.position < 0 and ofi > 0.5)
        
        gamma = self.panic_multiplier if is_underwater else 1.0
        reward = delta_pnl - (self.lambda_penalty * inventory_pct_squared * gamma)

        # 6. Progression & Termination
        self.current_step += 1
        truncated = self.current_step >= (self.episode_length - 1)
        terminated = self.portfolio_value < (self.initial_capital * 0.5)

        if terminated:
            reward -= 100.0 

        if truncated:
            self.cash += self.position * current_close
            if abs(self.position) > 0:
                reward -= (abs(self.position) * current_close * 0.005)
            self.position = 0
            self.portfolio_value = self.cash

        info = {"position": self.position, "portfolio_value": self.portfolio_value}
        
        return self._get_obs(), float(reward), terminated, truncated, info