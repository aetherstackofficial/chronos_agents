# Phase 1 baseline: realism suite vs current simulator

Run 2026-10-29, commit `d3162a4` (suite uncommitted). Spec: "Market Simulator Realism Validation Spec" v1.0.
Report card: `swarm_orchestrator/validation/reports/report_card_full_RELIANCE_20261029T151146.json`.

> **Status update (2026-10-30).** This page is the baseline as measured; its numbers stay as
> recorded. Since then:
> - gap 5 (replay fidelity) is fixed: T6 Part A replays 0 of 7,125 bars wrong;
> - gap 6 (deployed determinism) is fixed: seeded worker plus opt-in lockstep;
> - the tick table is now NSE's.
>
> See `swarm_orchestrator/PROTOCOL.md` §2, §3.1 and §9.

**Result: 5 of 19 tests pass, 14 fail, 0 errors.** The simulator has heavy tails and a volume-volatility link, but:
- its price falls in every seed;
- volume is about 14× too low and very lumpy;
- intraday seasonality is absent;
- price impact never decays;
- replay does not reproduce the tape.

As agreed, the simulator mechanics were not changed. The failures are the measurement.

## What was run

| Item | Value |
|---|---|
| Simulator | Live default swarm, unchanged: 15 PPO market makers, 4 PPO whales, 80 heuristic retail, SIM_LP. No news or oracle. |
| Harness | In-process client in place of ZMQ, virtual clock (1 s per tick, as live), seeded `random`/`numpy`/orchestrator RNGs, snapshot/restore |
| Size | 20 seeds × 100,232 returns (268 NSE sessions plus 2 warm-up days), started at ₹1,354 (RELIANCE last close) |
| Impact harness | 8 seeds × 200 single-order snapshots and 50 metaorder snapshots. The spec asks for 20 seeds × ≥200 snapshots. |
| Reference data | Kaggle NIFTY-100 1-minute bars, 2015-02-02 to 2026-04-08. RELIANCE is the gating instrument; HDFCBANK, INFY, SBIN, ITC and MARUTI are calibrated too. |
| Calibration | 268-day windows stepped by 20 days. Calibrate on dates before 2022-12-05 (82 windows), confirm on later dates (27 windows). Band = 5–95% widened by 10%. |
| Wall time | 70 min on 5 workers (12-core, 5.8 GB RAM machine) |

**Symmetry rules** (the real data has trade-price bars only):
- Comparisons with real data use trade-price closes on both sides.
- σ_t uses the Parkinson range estimator on both sides.
- The spec-default T1–T4 criteria are judged on the simulator's mid-quote, as the spec defines them.

## Scorecard

| Test | Status | Seeds passing | Simulator vs real NSE (median) |
|---|---|---|---|
| T6 Replay and determinism | **fail** | – | Replay: 7,124 of 7,125 bars mismatch. Seeded harness: bit-identical. |
| T0 Book invariants | pass | 100% | Zero crossed, off-grid, negative or duplicate orders |
| T1 Fat tails | pass (spec) / **fail** (real) | 100% | Excess kurtosis **192** vs 12.5 (band 7.2–27). Zero-return fraction 0.26 vs 0.11. |
| T2 Volatility clustering | **fail** | 50% | ρ₁ of \|r\| **0.13** vs 0.24. Decay exponent β 0.11 vs 0.27. |
| T3 No return autocorrelation | **fail** | 0% | Mid lag-1 −0.073. Trade lag-1 **−0.097** vs −0.034 (band −0.061 to −0.017). |
| T4 Volume–volatility | **pass** | 100% | Spearman 0.62 vs 0.53, in band. Whale ablation drops it 56%, so the link comes from trading. |
| A1 Aggregational Gaussianity | pass (spec) / fail (real) | 100% | Falls with scale as it should, but starts from 192 |
| A3 Volume clustering | **fail** | 0% | β **0.01** vs 0.29: volume autocorrelation does not decay at all |
| A4 Extreme intermittency* | fail | – | Interarrival CV 4.3 vs 2.0 (band 1.6–6.0), 75% of seeds in band |
| A5 Intraday seasonality* | **fail** | – | Profile correlation with real: \|r\| **−0.08**, volume 0.25 (real ≈0.96). Open/mid-day volume 1.1 vs 3.4. |
| T5 Book shape† | **fail** | 0% | 55% of the 20 tick levels empty. Depth falls away from the touch. γ 0.90 (needs > 1). Median spread 3 ticks. |
| T7 Square-root impact† | **fail** | 0% | δ 0.21 (T=30) and 0.36 (T=120). Y 1.4 vs 3.5, so impact grows with horizon. Buy/sell sign wrong in some Q buckets. |
| T7a Book walk† | fail | 12.5% | δ 0.97: linear, consistent with T5's flat book |
| T8 Monotonic impact† | fail | 12.5% | 0 violations, conservation exact. Fails only buy/sell symmetry. Q = 20,000 empties the book in 50% of trials. |
| A6 Impact decay† | **fail** | 12.5% | Impact after T+5/30/120 is 1.04, 1.00 and 1.14 of impact at T: permanent, no decay |
| A7 Spread/depth recovery† | fail | 12.5% | Median 2 min, but 1.5% of sweeps do not recover in 60 min |
| A2 Order-sign memory† | fail | 85% | ρ₁ 0.12, decay exponent 0.58. Positive memory exists (just under the 90% rule). |
| A8 No free lunch | pass (spec) / fail (real) | 95% | No rule beats costs. Intraday buy-and-hold t = **−7.9** (real ≈ −0.8): systematic intraday decline. |
| A9 Leverage/asymmetry* | **fail** | – | 5-day skew **−1.48** vs 0.01 |

\* PDF row truncated: provisional, judged on the real band only.<br>
† No real order-book data: spec defaults only, uncalibrated.

## Gaps, ranked by what to fix first

1. **The price collapses (most urgent).** Every one of the 20 seeds ends lower, with a median of −57% and a worst case of −89% over 268 days. Real RELIANCE over the same window length has a median of +18% and a worst of −16%. This trend also distorts A3, T2, A8 and A9, so other results can't be trusted until it is fixed.
   - Not yet diagnosed. Candidate causes to test:
     - retail can only sell inventory it holds and never shorts;
     - the liquidation reset injects fresh cash;
     - SIM_LP re-centres on `last_price` every minute, so nothing anchors value;
     - the US-trained whale policy may lean to selling.
2. **Volume scale and lumpiness.** The median minute trades 3 shares in the sim versus 17,350 in real data. 10% of sim minutes have no trades, versus 0% in real data. Daily volume is 0.86M versus 12.1M.
   - This explains the extreme kurtosis (192 vs 12.5) and the zero-return fraction: a few whale clips of up to 10,000 shares carry the whole tape.
3. **No intraday seasonality.** Nothing in the simulator varies with time of day (the A5 profiles are flat). Real NSE volume is 3.4× higher at the open and 3.6× higher at the close than at mid-day.
4. **The book and impact mechanism.** SIM_LP sets the shape: 8 flat levels spaced 1–3 ticks apart and refreshed every minute. Results:
   - a flat or concave depth profile (γ 0.90);
   - linear book-walk impact (δ 0.97);
   - permanent metaorder impact (A6 ≈ 1.0);
   - impact amplitude that grows with horizon.
5. **Replay fidelity (T6 Part A).** 7,124 of 7,125 March 2026 bars mismatch. Volume fails on every bar, because 70% of real bars exceed the 20,000-share `MAX_BAR_QTY` cap. 3,891 opens mismatch, because the engine's bar opens at the previous close.
   - Closes match exactly. RELIANCE's ₹0.10 tick happens to agree between NSE and the engine; a ₹100–250 stock would also fail on the tick grid.
6. **Deployed determinism.** With the harness seeds in place, runs are bit-identical (same seed, with or without a null strategy). The deployed `worker.py` seeds neither the orchestrator nor `random`/`numpy`, so two live runs with the same seed differ.
   - In-process, the PPO model load consumed global RNG state; the harness works around this by importing before it seeds.
7. **Return autocorrelation.** Lag-1 is −0.097 on trade closes versus a real band of −0.061 to −0.017. Short-horizon reversal is about 3× too strong, consistent with the thin, wide book (median spread 3 ticks).
8. **Volatility clustering too weak.** ρ₁ is 0.13 versus 0.24.

**What already works:**
- the matching core: T0 invariants, T8 monotonicity and conservation;
- an emergent volume–volatility link (T4, which the ablation shows comes from trading);
- positive order-sign memory (A2);
- costs that dominate random trading exactly as on real data (A8 random mean −3.54 vs −3.55 bps).

## What the real NSE data says about the spec itself

The same tests on real RELIANCE windows give these pass rates against the spec defaults (calibration / holdout):
- T1, T4, A3, A4, A8: 100% / 100%.
- T2: 96% / 100%.
- A1: 76% / 100%.
- **T3: 0% / 0%.** Trade-price bars always show bid-ask bounce (lag-1 −0.02 to −0.08 across all six stocks), so the spec's ±0.03 bound can't be judged on trade-price data. It needs mid-quotes, or the real band should replace it.
- T2's decay exponent β is 0.26–0.33 on all six stocks, inside the literature's 0.2–0.4.

## Limits of this baseline

- **One-minute engine.** There is no sub-minute data: no 10-second book snapshots, TWAP children one per minute, Parkinson σ in place of sub-bar realised volatility.
- **Impact harness is smaller than the spec:** 8 × 50–200 instead of 20 × 200. Throughput is about 160 ticks/s per core.
- **Smoke mode takes about 7 minutes,** not the spec's "under a minute".
- **Uncalibrated tests:** T5, T7, A2 and A7 have no real order-book or trade data to compare against.
- **Unverified cost input:** the A8 cost of 3.55 bps uses the NSE transaction charge of 0.00307% from broker pages. NSE's own circular wasn't verified; the alternative 0.00297% changes the cost by about 0.04 bps.
- **Data handling:** Kaggle prices are vendor-adjusted (fine for intraday returns). 34 incomplete or special sessions were dropped.

## Reproduce

```
cd swarm_orchestrator
pip install -r validation/requirements.txt          # scipy 1.14.1, statsmodels 0.14.6
python -m validation.fetch_data
python -m validation.calibrate --n-days 268
python -m validation.runner --mode full              # ~70 min, 5 workers
python -m validation.summarize validation/reports/report_card_full_*.json
python -m pytest tests/test_validation_stats.py      # 17 tests on known processes
```
