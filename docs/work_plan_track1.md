# Chronos — Refurbish Plan & Implementation Roadmap (Track 1)

**Owners:** Krishna (@krishna) & Akash (@akash)  
**Track:** Track 1 — Simulator, Models, and Validation  
**Core Goal:** A simulated market whose statistical behavior and price impact can be mathematically proven realistic.  
**Governing Rule:** *"The backtest is the product. If the simulated market behaves wrongly, every backtest result is wrong."*

---

## 1. Executive Summary & Work Division

The implementation is split into two complementary tracks:
- **Krishna (Algorithmic / Microstructure / Execution / Validation)**: ~50% total effort (higher code volume, deterministic algorithms, mathematical formulations, test harnesses, and architecture).
- **Akash (ML / RL Training / Data Pipelines / LLM)**: ~50% total effort (higher research uncertainty, data sourcing, compute-heavy SAC/PPO training iterations, reward shaping, and LLM structured parsing).

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             KRISHNA'S WORKFLOW (Algorithmic)                     │
│  Phase 1: Realism Test Suite (8 statistical verification tests & report card)    │
│  Phase 2: Retail Swarm Microstructure (Personalities, smooth sigmoid, noise flow)│
│  Phase 3: Behavioral Memory & Support/Resistance Order Clustering                │
│  Phase 4: Market Maker Quote Skewing & Avellaneda-Stoikov Formula Baseline       │
│  Phase 5: POV Large-Order Splitting Engine (Parent/Child & 15% participation)    │
│  Phase 6: Engine Hand-Over Command (Replay -> Reactive latch for Banti)          │
│  Phase 7: Statistical Jump Detection (Lee-Mykland test)                          │
│  Phase 8: Modular FastAPI Refactor of bridge.py (Agreed layout with Aryan)       │
└──────────────────────────────────────────────────────────────────────────────────┘
                                         │
        (Git Commit & Push: Engine & Testing Harness Fully Deterministic)
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              AKASH'S WORKFLOW (ML & Data)                        │
│  Phase 1: Clean Historical Data Ingestion (Bypass Excel 1M row cut, Nifty 500)   │
│  Phase 2: 9-Feature Parity Engineering (Live build_observation <-> Gym Env)     │
│  Phase 3: RL Model Training (SAC vs. PPO for Whales and Market Makers)           │
│  Phase 4: Benchmark RL Models against Krishna's Avellaneda-Stoikov Baseline      │
│  Phase 5: Jev LLM Structured News Engine (Typed decisions: sentiment & urgency)  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Akash's Scope (ML, Training & LLM News)

Akash will pick up the repository after Krishna's commit and execute the following:

1. **Training Data Pipeline Overhaul**:
   - Re-export multi-year Indian intraday stock data without Excel's 1,048,576 row truncation.
   - Expand asset universe to representative Nifty 500 stocks (20, 50, or 100 tickers) across 20 years.
   - Clean missing trading days and precompute rolling 14-period RSI and volume moving averages.
2. **Feature Engineering & Parity**:
   - Align the gym training environment (`UniversalMarketEnv` in `swarm_orchestrator/agents/environments.py`) with live inference (`features.py`).
   - Add the 3 missing observation inputs (expanding observation vector to 9 dimensions):
     1. Normalized inventory position.
     2. Current bid-ask spread (in bps).
     3. Order book imbalance (Level-1 / Top-5 bid-ask volume ratio).
   - Recompute observation normalizer statistics (`VecNormalize`) with updated distribution ranges.
3. **Reinforcement Learning Training (SAC vs. PPO)**:
   - Implement Soft Actor-Critic (SAC) for `MarketMakerEnv` and `SmartMoneyEnv`.
   - Train comparison models using PPO.
   - Shape reward functions to penalize excess inventory and optimize spread capture / execution slippage.
   - **Quality Gate ("Beat the Formula")**: Evaluate trained RL market makers against Krishna's closed-form Avellaneda–Stoikov model. If the RL model does not beat the formula, the formula will be preferred.
4. **Jev LLM News Integration**:
   - Replace slow (10–13s) free-text LLM prompt parsing in `oracle.py` with typed, structured JSON decision schemas.
   - Extract: `sentiment` (float $-1.0 .. +1.0$), `category` (`earnings`, `macro`, `liquidity`, `regulatory`, `rumour`), and `urgency` ($1 .. 5$).
   - Implement deterministic response caching and rate-limiting.

---

## 3. Krishna's Detailed Workflow & Implementation Blueprint

Krishna executes and pushes the following 8 phases before handing off to Akash:

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
- **Output**: Produces a standardized JSON report card (`pass`/`fail` per metric) for Aryan's landing page.

---

### Phase 2: Real Order Flow & Retail Heterogeneity
*Files: `swarm_orchestrator/agents/heuristic_retail.py`, `swarm_orchestrator/agents/live_agents.py`*

Replace the frozen, identical 80 retail agents with a heterogeneous crowd that crosses the spread for natural reasons.

1. **Heterogeneous Agent Personalities**:
   - Initialize each of the 80 retail agents with randomized parameters:
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
   - Support and resistance emerge endogenously from order density in the book.
2. **Retail Psychological Memory**:
   - Track per-agent trading history (`pnl_streak`, `last_trade_pnl`, `open_trade_entry_price`).
   - **Revenge Trading**: Following a realized loss ($PnL < 0$), temporary multiplier applied to risk appetite ($1.5\times - 2.0\times$) for the next order.
   - **Disposition Effect**: When holding an open winning position ($P_{\text{current}} > P_{\text{entry}}$), urgency to take profit increases rapidly; when holding a losing position ($P_{\text{current}} < P_{\text{entry}}$), holding time increases.
   - **Capitulation**: Following $N \ge 3$ consecutive losses or drawdown exceeding threshold, agent executes full market panic liquidation.

---

### Phase 4: Market Maker Quote Skew & Avellaneda–Stoikov Baseline
*Files: `swarm_orchestrator/agents/live_agents.py`, `swarm_orchestrator/agents/avellaneda_stoikov.py`*

1. **Inventory-Based Quote Skewing**:
   - Update `LiveMarketMaker` so quote prices skew based on current inventory $q$ relative to limit $Q_{\text{max}}$:
     $$\Delta_{\text{skew}} = \frac{q}{Q_{\text{max}}} \times \kappa_{\text{skew}}$$
     $$P_{\text{bid}} = P_{\text{mid}} \cdot (1 - \text{bid\_bps} - \Delta_{\text{skew}})$$
     $$P_{\text{ask}} = P_{\text{mid}} \cdot (1 + \text{ask\_bps} - \Delta_{\text{skew}})$$
   - When holding excess long inventory ($q > 0$), both bid and ask shift down to deter sellers and attract buyers, offloading inventory naturally.
2. **Avellaneda–Stoikov Baseline Model**:
   - Implement closed-form market maker logic to serve as Akash’s training benchmark:
     $$r(s, q, t) = s - q \gamma \sigma^2 (T - t)$$
     $$\delta^a + \delta^b = \gamma \sigma^2 (T - t) + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{\kappa}\right)$$
     $$r^a = r + \frac{\delta^a + \delta^b}{2}, \quad r^b = r - \frac{\delta^a + \delta^b}{2}$$
     *(Where $s$ is mid price, $q$ is inventory, $\gamma$ is risk aversion, $\sigma$ is volatility, and $\kappa$ is book liquidity parameter).*

---

### Phase 5: Large-Order Splitting & POV Execution Engine
*Files: `swarm_orchestrator/engine/execution.py`, `swarm_orchestrator/engine/matching_engine.py`*

1. **Parent–Child Order Architecture**:
   - In `matching_engine.py`, support structured order payloads:
     - `parent_order_id`: UUID of parent execution.
     - `child_order_id`: UUID of individual slice.
     - `total_target_qty`: Cumulative size.
     - `cum_filled_qty`: Running executed count.
     - `avg_fill_price`: Running VWAP of executed slices.
     - `reason`: Strategy / institutional intent (`WHALE_REBALANCE`, `POV_TWAP`, `NEWS_SHOCK`).
2. **Percentage of Volume (POV) Slicer**:
   - Institutional whale orders are placed into an execution schedule capped at a target participation rate (e.g. $10\%$ to $15\%$ of traded bar volume).
   - Child slices adapt dynamically: if market volume dries up, slice sizes shrink to avoid excessive market impact.

---

### Phase 6: Engine Control Hand-Over & Replay Hooks (Banti Hand-Off)
*Files: `swarm_orchestrator/engine/matching_engine.py`, `swarm_orchestrator/engine/replay.py`*

1. **Deterministic Hand-Over Command**:
   - Implement explicit engine command `handover_to_reactive(at_bar: int)` callable via REST API and engine physics tick.
   - Switches engine mode from `REPLAY` to `REACTIVE` at a preset bar without requiring a trade from `STRAT_` or `USER_`.
   - Preserves identical random seed and starting order book state, enabling Banti's control simulation (2.8).

---

### Phase 7: Lee–Mykland Jump Detection Algorithm
*File: `swarm_orchestrator/features/jumps.py`*

1. **Intraday Jump Test**:
   - Detects abnormal price movements by comparing instantaneous return shocks against local realized bipower variation:
     $$\mathcal{L}_t = \frac{|r_t|}{\sigma_{\text{bipower}}}$$
   - When test statistic $\mathcal{L}_t$ exceeds the threshold $S_n$, tag the bar as an endogenous jump shock.
   - Direction originates strictly from price data; Jev LLM is only called to assign semantic classification.

---

### Phase 8: FastAPI Modular Code Restructure
*Files: `swarm_orchestrator/bridge.py` -> `swarm_orchestrator/api/`*

1. Refactor monolithic 564-line `bridge.py` into clean modular structure:
   - `api/routers/replay.py`: Historical replay upload and start endpoints.
   - `api/routers/market.py`: Order book snapshot, trade tape, ticker.
   - `api/routers/agents.py`: Swarm configuration, pause/resume, inspector.
   - `api/routers/health.py`: Service health and system telemetry.
   - `api/dependencies.py`: Dependency injection for engine state.
2. Aligns directory structure with Aryan for authentication and multi-user database routes.

---

## 4. Git Delivery & Akash Hand-Off Checklist

Before Akash begins model training, Krishna completes and verifies:

- [ ] All 34 existing unit tests pass (`pytest swarm_orchestrator/tests`).
- [ ] Realism test suite (`test_realism.py`) is committed and executes.
- [ ] Retail swarm behavioral model is active in `live_agents.py`.
- [ ] Avellaneda–Stoikov formula baseline is verified and documented.
- [ ] POV parent-child execution schema is supported in matching engine.
- [ ] Engine handover command is functional for Banti's control run.
- [ ] Clean commit pushed to GitHub repository with commit message:  
  `feat(engine): implement realism test suite, microstructure order flow, and Avellaneda-Stoikov baseline`

