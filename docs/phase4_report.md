# Phase 4: parent/child orders, whale volume-capped slicer, market-maker inventory skew

Run 2026-10-30, commit `f8ed0a1` (Phase 4 code uncommitted). Same suite, mode, seeds and reference data as the Phase 1 baseline.
- Report card: `swarm_orchestrator/validation/reports/report_card_full_RELIANCE_20261030T012931.json`
- Baseline: `report_card_full_RELIANCE_20261029T151146.json`
- Mechanism diagnostics: `phase4_diagnostics_20261029T232849.json` (5 seeds × 20 days, skew on vs off)

**Result: 1 of 19 tests pass, 18 fail, 0 errors (Phase 1: 5 pass).**
- The mechanisms work as specified. 23 unit tests pass against the real engine and its trade tape.
- The median price outcome improves a lot, from −57% to −14%. Most metrics measured against real-data bands move toward real NSE.
- Four tests that passed before now fail: T1, T4, A1 and A8.
- Each regression has a traced cause. All of them come from one of two things: whales now dominate a book that is too thin, or children never walk the book. As agreed, they are recorded, not patched. Each is assigned to the later step that owns it.

## Definition of done: status

| Criterion (agreed) | Status |
|---|---|
| Mechanisms proven by unit tests | **Met.** 20 slicer/skew tests plus 3 live-agent tests pass (74/74 in the suite, excluding the known `test_oracle_fallback.py` collection issue). |
| Tests expected to improve move the right way against the baseline | **Mostly met.** 15 of 23 real-band metrics are closer to real. T8 pass rate goes from 0.125 to 0.75, but for a mechanical reason, not a realism gain (see "The order book changed"). |
| Nothing that passed before regresses | **Not met.** T1, T4, A1 and A8 regress (causes below). |
| Failures labelled with the later phase that owns them | Done (last section). |

## What was built

| Piece | Rule (frozen in `agents/config/phase4_execution.yaml`) |
|---|---|
| Parent/child IDs | `WHALE_n:P<k>:C<m>` in `client_order_id` |
| Whale volume-capped slicer | Each child ≤ 15% × *expected* NSE volume for that minute (real ADV 11.97M × real intraday shape, opening auction excluded). **The 15% is the plan's own number. It matches the whale training env; no NSE source.** |
| Children | Marketable LIMIT at best ask/bid, expires at the end of its minute. Unfilled quantity carries to the next child. **No market-order fallback.** |
| Retarget | Same side: update the target, fills stay. Opposite side: close via CANCEL_ALL, open a new parent. |
| Session end | Unfinished parents cancelled at 15:29 and shortfall recorded. No whale trades in the 15:29 bar (tested). |
| Parent remainder cap | MAX_CLIP 10,000 |
| MM inventory skew | Quotes shifted −round(3·q/q_max) ticks, clamped to ±3, snapped to the tick grid (also fixes the old 2-decimal rounding). **Uncalibrated; fixed rule, not tuned. Not Avellaneda–Stoikov.** |
| Determinism | Clock and RNG injected into the live agents. Same seed gives the same event hash and parent IDs (tested). |
| Unchanged, as agreed | Orchestrator (awake 0.3, 1.0 for 60 ticks after news), SIM_LP, retail, user strategies |

## Mechanism findings (diagnostics, skew k=3)

| Measure | Value | Reading |
|---|---|---|
| Whale participation vs expected NSE volume, quiet minutes | **0.14%** per whale | Far below the 15% cap. Expected was ~4.5%. |
| Same, news windows | **0.54%** per whale | About 4× quiet, as the orchestrator intends |
| Parent fill ratio / shortfall | 0.30 / 0.71 | Most planned size is never filled |
| Child fill rate | 0.17 | The visible best level is much thinner than the child |
| Cap binding | 78% of children | The cap is usually the size limit, not the remaining quantity |
| Parent outcomes | 78% replaced by an opposite-side target, 19% completed, 2% session close | The PPO policy flips direction faster than the parent can fill |
| Whale MARKET orders sent | **0** | The no-market-order rule holds |
| Whale share of simulated volume | **96%** | Everyone else barely trades |
| Simulated / expected NSE volume | **0.0069** | Scale mismatch, kept visible as agreed |
| MM inventory, mean \|q\|/q_max | 3% | The MMs barely trade |
| MM inventory AR(1) | 1.0000 | No reversion, because inventory hardly moves |
| Mean \|skew\| | 0.00 ticks | **The skew is inert.** k=3 and k=0 give bit-identical runs in 3/5 seeds. |

### The order book changed (found in the cross-check)

The cross-check measured resting depth by owner, sampled every 15 minutes after the physics tick (the moment impact snapshots are taken), seeds 1 and 3, days 3–6. Phase 1 was rebuilt from the committed `live_agents.py` in a scratch copy.

| Resting shares per side (median) | Phase 1 | Phase 4 |
|---|---|---|
| Market makers | ~110 | **~33,000** |
| SIM_LP | ~20,000–29,000 | **~6,000** |
| 20-minute volume average (sets SIM_LP size) | ~4,100–6,500 | ~650 |

- **Phase 1:** whale market orders kept filling the MMs up to their position limits, so the MMs quoted almost nothing. SIM_LP was the book, which confirms Phase 1's "SIM_LP sets the shape".
- **Phase 4:** whales no longer hit the MMs, so the MMs keep full-size quotes on both sides. They are almost never filled: their quotes sit at least 2 bps (~3 ticks) from the mid, SIM_LP's first level is 1–3 ticks away, and whale children only take the best price.
- **SIM_LP shrank** because its size is 400 + 0.5 × the 20-minute volume average, and volume fell about 8×.

Consequences:
- **T8's improvement is mechanical.** The 20,000-share probe no longer empties the book (exhausted trials fell from 50% to 0%) because idle MM quotes now sit behind the touch. The same explains T7's flatter impact (δ 0.21 → 0.03) and A6.
- **The impact tests now mostly measure idle MM quotes,** not a realistic book.
- **This sharpens the skew explanation.** The MMs quote, but nothing fills them, so their inventory never moves.

**The MM skew rule cannot be evaluated yet.** It only acts when inventory exceeds 1/6 of the limit, and the MMs never get there. SIM_LP (8 levels, unlimited cash, refreshed every minute) takes the flow that MMs would otherwise absorb.

## Suite results: before → after

| Test | Phase 1 | Phase 4 | Seed pass rate |
|---|---|---|---|
| T0 invariants | pass | pass | 1 → 1 |
| T1 fat tails | pass | **fail** | 1 → 0 |
| T2 vol clustering | fail | fail | 0.5 → 0 |
| T3 no return autocorr | fail | fail | 0 → 0 (real NSE also fails, bid-ask bounce) |
| T4 vol–volume | pass | **fail** | 1 → 0.3 |
| T5 book shape | fail | fail | 0 → 0 |
| T6 determinism / replay | fail | fail | 0 → 0 |
| T7 square-root impact | fail | fail | 0 → 0 |
| T7a book walk | fail | fail | 0.125 → 0 |
| T8 monotonic impact | fail | fail | **0.125 → 0.75** |
| A1 aggregational gaussianity | pass | **fail** | 1 → 0 |
| A2 order-flow sign memory | fail | fail | 0.85 → 0.6 |
| A3 volume clustering | fail | fail | 0 → 0.05 |
| A4 extreme intermittency | fail | fail | 1 → 1 (real-band fail) |
| A5 intraday seasonality | fail | fail | – |
| A6 impact decay | fail | fail | 0.125 → 0 |
| A7 spread/depth recovery | fail | fail | 0.125 → 0 |
| A8 no free lunch | pass | **fail** | 0.95 → 0.6 |
| A9 leverage asymmetry | fail | fail | – |

### Key metrics against real NSE (sim median over 20 seeds)

| Metric | Real | Phase 1 | Phase 4 | Closer? |
|---|---|---|---|---|
| Excess kurtosis, 1 min | 12.55 | 191.7 | −0.12 | yes, but now overshoots: tails too **thin** |
| Zero-return fraction | 0.109 | 0.260 | 0.194 | yes |
| \|r\| autocorr ρ₁ (vol clustering) | 0.238 | 0.134 | 0.150 | yes |
| Return autocorr ρ₁ | −0.034 | −0.097 | −0.057 | yes |
| Vol–volume Spearman | 0.530 | 0.621 | 0.317 | no |
| Kurtosis, 60 min | 6.28 | 7.30 | 13.96 | no |
| Volume autocorr ρ₁ | 0.426 | 0.271 | 0.435 | yes |
| Interarrival CV of extremes | 1.99 | 4.32 | 10.25 | no |
| Intraday \|r\| profile corr with real | 0.965 | −0.08 | 0.03 | yes (still none) |
| Intraday volume profile corr with real | 0.963 | 0.25 | 0.27 | yes (still weak) |
| Best rule t-stat (A8) | −0.77 | −5.01 | −0.78 | yes (median), but see runaway seeds |
| Leverage, lags 1–10 | −0.0034 | −0.0061 | −0.0052 | yes |
| Multi-day skew | 0.010 | −1.48 | −1.14 | yes |

### Price path (268 sessions per seed)

| | Phase 1 | Phase 4 |
|---|---|---|
| Median last / first price | 0.43 | **0.86** |
| Range | 0.11 – 0.92 | 0.69 – **4.34** |
| Seeds ending below start | 20 / 20 | 15 / 20 |
| Median daily volume V_D | 862,544 | 21,180 |

The seeds split into two regimes:
- **15 seeds** end 2–31% down, with ±40% swings inside the year. Seed 4 ends +4% after peaking at 2.5×.
- **4 seeds (1, 2, 13, 19)** rise to about 4.2× and end at their high. They never trade below their start price and have the lowest volume (V_D 4–8k, 19% zero-volume minutes).

These four are the whale-buys-against-SIM_LP ratchet seen in the smoke run: SIM_LP re-centres on each new mid, so persistent net buying walks the price up with nothing to push back.

## Why the four regressions happened

| Test | Cause | Owner |
|---|---|---|
| **T1 fat tails / A1** | Phase 1's heavy tails came from whale MARKET orders sweeping a thin book. Children are now marketable limits at the best price and **never walk the book**, so 1-minute jumps disappear. Kurtosis goes from 192 to −0.1. Kurtosis at 60 minutes rises (14) because of the slow trends, so A1's "kurtosis falls with scale" fails. The tails were an artifact before. Real tails need real depth and real aggressive flow. | Liquidity step (SIM_LP removal, MMs as real liquidity) + Phase 2 (ADV-scaled participants) |
| **T4 vol–volume** | Whales are 96% of volume. Their size follows the expected-volume curve and the cap, not volatility, so volume no longer co-moves with volatility. The whale ablation confirms it: removing whales now lowers the Spearman by only 8% (0.36 → 0.33), against 56% in Phase 1. The remaining link comes from MMs and retail. | Phase 2 (more independent flow, at scale) |
| **A8 no free lunch** | Median is fine (t = −0.78, matches real). Fails in 8/20 seeds, split evenly. In the 4 runaway seeds (1, 2, 13, 19), intraday buy-and-hold has t = 27–33. In 4 other seeds (3, 4, 5, 20), **30-minute momentum** has t = 20–22: the trends are slow and persistent enough to follow profitably after costs. | Liquidity step (removing the SIM_LP ratchet; real liquidity to absorb persistent whale flow) |

## Labels that stay attached

- The 15% cap is the plan's own number. It matches the whale training env; there is no NSE source.
- The MM skew is uncalibrated and was never tuned. Its effect is currently zero because MMs quote but are not filled, so they never hold inventory.
- The Phase 4 impact results (T7, T7a, T8, A6) measure a book made mostly of idle MM quotes. Don't read them as impact realism.
- The volume scale mismatch (sim = 0.7% of NSE) is shown, not corrected. The cap uses real expected volume on purpose.
- All Phase 4 results sit on top of Phase 1's open problems (SIM_LP, volume scale, event clock, replay).

## Next step (to plan together before coding)

Combined liquidity step:
1. Remove SIM_LP.
2. Make the MMs the real liquidity providers. This needs Akash, since their quoting model is his Phase 3.
3. Scale participants to ADV (Phase 2).

Phase 4's mechanisms need this before they can be judged on realism. The skew needs MMs that carry inventory. The slicer needs a book deep enough for 15% of real volume to be fillable.
