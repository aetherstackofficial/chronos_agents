# Chronos — Refurbish Plan & Implementation Roadmap (Track 1)

**Owners:** Krishna (@krishna) & Akash (@akash)  
**Track:** Track 1 — Simulator, Models, and Validation  
**Core Goal:** A simulated market whose statistical behavior and price impact can be mathematically proven realistic.  
**Governing Rule:** *"The backtest is the product. If the simulated market behaves wrongly, every backtest result is wrong."*

---

## 1. Work Division & Fair Workload Balance

The workload between Krishna and Akash is strictly **50% / 50%**, organized into equal, focused phases.

- **Krishna**: Owns the **Matching Engine, Retail Crowd Microstructure & Realism Validation** (the deterministic physics of the market).
- **Akash**: Owns the **Data Engineering, Analytical Baseline, RL Model Training & LLM News Engine** (the intelligence and learning pipelines).

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          KRISHNA (Engine, Microstructure & Tests)                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Phase 1: Realism Test Suite (The 8 statistical validation checks & report card)    │
│  Phase 2: Retail Microstructure (Heterogeneous personalities & smooth sigmoid curves)│
│  Phase 3: Retail Memory & Support/Resistance (Revenge trading, disposition effect)  │
│  Phase 4: POV Execution Engine (Parent/child large order splitting & 15% cap)       │
│  Phase 5: Engine Hand-Over Command (Replay -> Reactive mode latch for Banti)        │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                           │
         (Krishna pushes to GitHub: Engine microphysics & test harness ready)
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                             AKASH (Data, RL Models & LLM)                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Phase 1: Intraday Data Pipeline (Bypass 1M Excel row limit, Nifty 500 dataset)     │
│  Phase 2: 9-Feature Observation Pipeline (Parity between Gym Env & live inference)  │
│  Phase 3: Avellaneda–Stoikov Baseline (Closed-form inventory risk formula)          │
│  Phase 4: RL Model Training (SAC vs. PPO policy training & "Beat the Formula")     │
│  Phase 5: Jev LLM News Engine & Jump Test Integration (Typed structured decisions)  │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                      SHARED: FastAPI Modular Bridge Restructure (with Aryan)
```

---

## 2. Krishna's Detailed Workflow (5 Core Phases)

Krishna implements, tests, and pushes these 5 deterministic engine/microstructure phases:

### Phase 1: Realism Test Suite (The Ground Truth)
*File: `swarm_orchestrator/tests/test_realism.py`*

Build the 8 empirical market validation tests before touching agent mechanics. Every subsequent modification must run against this suite.

1. **Fat Tails Test**:
   - Computes kurtosis of 1-minute simulated log returns: $r_t = \ln(P_t / P_{t-1})$.
   - **Pass Criteria**: Excess Kurtosis $\kappa > 3.0$ (rejecting thin-tailed normal Gaussian distribution).
2. **Volatility Clustering Test (ARCH Effect)**:
   - Computes autocorrelation of absolute returns $\text{Corr}(|r_t|, |r_{t-k}|)$ for lags $k \in [1, 10]$.
   - **Pass Criteria**: Statistically significant positive autocorrelation ($p < 0.01$).
3. **No Return Autocorrelation**:
   - Computes autocorrelation of raw price returns $\text{Corr}(r_t, r_{t-k})$ for lags $k \in [1, 5]$.
   - **Pass Criteria**: Return autocorrelation indistinguishable from 0 (within $[-2/\sqrt{N}, +2/\sqrt{N}]$ $95\%$ confidence bounds), proving no artificial trend steering.
4. **Volume & Volatility Coupling**:
   - Computes Pearson correlation between bar volume $V_t$ and realized volatility $\sigma_t$.
   - **Pass Criteria**: Positive correlation coefficient $\rho(V_t, \sigma_t) > 0.25$.
5. **Order Book Shape (Convex Depth)**:
   - Measures cumulative limit order depth across price distance from mid-quote.
   - **Pass Criteria**: Cumulative volume increases progressively away from mid price (non-flat, authentic liquidity curve).
6. **Replay Determinism (Zero-Trade Check)**:
   - Runs `HistoricalReplayer` without any active user strategy trades.
   - **Pass Criteria**: Replayed simulated candles match historical OHLCV candles with $0.00\%$ deviation.
7. **Square-Root Price Impact Law**:
   - Injects market orders of varying quantities $Q \in [100, 500, 1000, 5000, 20000]$ into the active book.
   - Fits temporary price impact $\Delta P$ against order size $Q$.
   - **Pass Criteria**: $\Delta P \approx Y \cdot \sigma \cdot \sqrt{Q / V}$ with regression fit $R^2 \ge 0.85$.
8. **Monotonic Impact**:
   - Sanity check: $Q_2 > Q_1 \implies |\Delta P(Q_2)| \ge |\Delta P(Q_1)|$ with 0 exceptions.
- **Deliverable**: Produces a standardized JSON report card (`pass`/`fail` per metric) for Aryan's landing page.

---

### Phase 2: Retail Swarm Microstructure & Personalities
*Files: `swarm_orchestrator/agents/heuristic_retail.py`, `swarm_orchestrator/agents/live_agents.py`*

Replace the frozen, identical 80 retail agents with an active, heterogeneous crowd that crosses the spread for real reasons.

1. **Heterogeneous Agent Personalities**:
   - Initialize each retail agent with unique behavioral traits:
     - `personality_type`: `MOMENTUM_TREND_FOLLOWER` (40%), `MEAN_REVERTER` (40%), `NOISE_TRADER` (20%).
     - `rsi_center`: Gaussian $\mathcal{N}(50, 5)$.
     - `conviction_threshold`: Uniform $\mathcal{U}(0.10, 0.35)$.
     - `holding_time_ticks`: Poisson distributed $\lambda \in [10, 60]$.
     - `risk_appetite`: Uniform $\mathcal{U}(0.01, 0.05)$ of available cash.
2. **Smooth Logistic Response (Replacing On/Off Step Rules)**:
   - Eliminate hard cliff logic (`if rsi < 30 buy, else 0`).
   - Implement continuous logistic trade probability and conviction scaling:
     $$\text{Conviction}_{\text{reversion}}(RSI) = \frac{2}{1 + e^{-\alpha (RSI - 50)}} - 1$$
     $$\text{Conviction}_{\text{momentum}}(RSI) = -\text{Conviction}_{\text{reversion}}(RSI)$$
   - Allows agents to trade at RSI 55 with lower size/probability instead of complete paralysis.
3. **Cash-Needs & Rebalancing Liquidity Flow**:
   - At each physics tick, a small stochastic fraction of agents ($p \sim 0.05$) execute unaligned cash-flow trades (e.g. portfolio rebalancing, liquidity withdrawal).
   - Equal probability of buy/sell, creating non-directional turnover and eliminating frozen books without inducing fake trends.
4. **ADV-Based Agent Sizing & 15% Participation Cap**:
   - Replace fixed `MAX_QTY = 100` with sizing dynamically scaled to Average Daily Volume (ADV):
     $$\text{BaseQty} = \text{ADV} \times f_{\text{agent}}$$
   - Enforce maximum single-order participation cap: $\text{OrderQty} \le 0.15 \times \text{Volume}_{\text{minute}}$.

---

### Phase 3: Natural Support/Resistance & Behavioral Memory
*Files: `swarm_orchestrator/agents/heuristic_retail.py`, `swarm_orchestrator/engine/matching_engine.py`*

1. **Natural Price Level Clustering**:
   - Remove hardcoded ₹5 price bounce rules from the engine.
   - Retail and resting limit orders cluster around psychological round numbers (multiples of ₹1, ₹5, ₹10, ₹100 depending on tick band) and local $N$-tick rolling swing highs and lows.
   - Support and resistance emerge naturally from order density in the book.
2. **Retail Psychological Memory**:
   - Track per-agent trading history (`pnl_streak`, `last_trade_pnl`, `open_trade_entry_price`).
   - **Revenge Trading**: Following a realized loss ($PnL < 0$), apply temporary risk multiplier ($1.5\times - 2.0\times$) to next order.
   - **Disposition Effect**: When holding winning positions ($P_{\text{current}} > P_{\text{entry}}$), urgency to take profit increases rapidly; when holding losers, holding time increases.
   - **Capitulation**: Following $N \ge 3$ consecutive losses or extreme drawdown, execute full market panic liquidation.

---

### Phase 4: Large-Order Splitting (POV Execution Engine) & Quote Skew
*Files: `swarm_orchestrator/engine/execution.py`, `swarm_orchestrator/engine/matching_engine.py`, `swarm_orchestrator/agents/live_agents.py`*

1. **Parent–Child Order Architecture**:
   - In `matching_engine.py`, support structured order payloads:
     - `parent_order_id`: UUID of parent execution.
     - `child_order_id`: UUID of individual slice.
     - `total_target_qty`: Cumulative size.
     - `cum_filled_qty`: Running executed count.
     - `avg_fill_price`: Running VWAP of executed slices.
     - `reason`: Strategy / institutional intent (`WHALE_REBALANCE`, `POV_TWAP`).
2. **Percentage of Volume (POV) Slicer**:
   - Whale orders execute as child slices capped at $10\%$ to $15\%$ of traded bar volume, preventing huge single-order market dislocations.
3. **Market Maker Quote Skewing**:
   - Market maker quotes shift downward when holding long inventory to attract buyers, and shift upward when short to attract sellers.

---

### Phase 5: Engine Control Hand-Over (Banti Hand-Off)
*Files: `swarm_orchestrator/engine/matching_engine.py`, `swarm_orchestrator/engine/replay.py`*

1. **Deterministic Hand-Over Command**:
   - Implement explicit engine command `handover_to_reactive(at_bar: int)` callable via REST API and engine physics tick.
   - Switches engine mode from `REPLAY` to `REACTIVE` at a preset bar without requiring a trade from `STRAT_` or `USER_`.
   - Preserves identical random seed and starting order book state, enabling Banti's control simulation (Section 2.8).

---

## 3. Akash's Detailed Workflow (5 Core Phases)

Akash pulls Krishna's verified engine code from GitHub and executes the 5 ML/RL phases:

### Phase 1: Intraday Data Pipeline & Re-Export
- Re-export intraday data without Excel's 1,048,576 row cutoff across 20-50 Nifty 500 stocks.
- Handle missing trading days, calculate rolling 14-period RSI, volume moving averages, and order flow imbalance.

### Phase 2: 9-Feature Parity & Gym Environment Setup
*Files: `swarm_orchestrator/agents/features.py`, `swarm_orchestrator/agents/environments.py`*
- Expand observation vector from 6 to 9 dimensions across both training and live inference:
  1. Distance to VWAP
  2. Realized Volatility
  3. Volume Ratio
  4. RSI (normalized $0..1$)
  5. Order Flow Imbalance (OFI)
  6. Portfolio Return
  7. **Inventory** (normalized)
  8. **Current Spread** (bps)
  9. **Book Imbalance** (Level-1 / Top-N bid-ask ratio)
- Recompute observation normalizer statistics (`VecNormalize`).

### Phase 3: Avellaneda–Stoikov Analytical Baseline
*File: `swarm_orchestrator/agents/avellaneda_stoikov.py`*
- Implement the published closed-form Avellaneda–Stoikov market maker formula:
  $$r(s, q, t) = s - q \gamma \sigma^2 (T - t)$$
  $$\delta^a + \delta^b = \gamma \sigma^2 (T - t) + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{\kappa}\right)$$
- Acts as the mathematical benchmark that any trained RL agent must beat before deployment.

### Phase 4: RL Model Training (SAC vs. PPO) & "Beat the Formula"
*Files: `swarm_orchestrator/agents/ml_models/`*
- Implement Soft Actor-Critic (SAC) for `MarketMakerEnv` and `SmartMoneyEnv`.
- Train comparative PPO models with inventory penalties and spread-capture rewards.
- Benchmark trained models against the Phase 3 Avellaneda–Stoikov baseline.

### Phase 5: Jev LLM Structured News Engine & Jump Detection
*Files: `swarm_orchestrator/oracle.py`, `swarm_orchestrator/features/jumps.py`*
- Replace slow (10-13s) free-text LLM prompt parsing with typed JSON schemas (sentiment, category, urgency).
- Implement the Lee–Mykland statistical jump test to detect price shocks in historical bars, using Jev only to label the shock type.

---

## 4. Shared Task: FastAPI Modular Code Restructure
*Files: `swarm_orchestrator/bridge.py` -> `swarm_orchestrator/api/`*
- Krishna and Akash collaborate with Aryan to split `bridge.py` into standard modular routers (`routes/replay.py`, `routes/market.py`, `routes/agents.py`, `routes/health.py`).

---

## 5. Krishna's Pre-Push Verification Checklist

Before pushing to GitHub for Akash:
- [ ] Existing 34 tests pass (`pytest swarm_orchestrator/tests`).
- [ ] Realism test suite (`test_realism.py`) passes regression checks.
- [ ] Retail swarm behavioral model is active in `live_agents.py`.
- [ ] POV parent-child order splitting works in matching engine.
- [ ] Engine handover command is functional for Banti's control run.
