# Chronos Protocol & Architecture Contract (Phase 2)

This file is the **single source of truth** for every inter-process interface in Chronos.
All modules MUST code against these schemas exactly. Any deviation must be recorded here.

---

## 1 · Process topology (two-plane exchange model)

```
                       ┌──────────────────────────────────────────┐
                       │  ENGINE  (engine/server.py, 1 process)   │
                       │  • matching core: single-threaded,       │
                       │    deterministic, sequenced              │
   ORDER-ENTRY PLANE   │  • ROUTER :5555  (reliable, addressed)   │   MARKET-DATA PLANE
   DEALER ──────────────▶                                          ──────────────▶ PUB :5556
   (agents, strategies,│  • 1 Hz physics tick = 1 sim-minute      │   (push; topics: tick,
    bridge, oracle,    │  • SQLite persistence (authoritative)    │    trade, event)
    FIX gateway)       └──────────────────────────────────────────┘        │
                                                                            ▼
        workers (worker.py / main.py) · bridge (FastAPI :8000) · strategy SDK · oracle
        — every consumer SUBscribes to :5556 for data and DEALs to :5555 for orders.
```

- **Order entry = ZeroMQ ROUTER/DEALER over TCP** (like FIX/OUCH over TCP at a real venue:
  reliable, per-session identity, acks). Every client uses a **DEALER** socket with a unique
  identity; the engine replies to that identity.
- **Market data = ZeroMQ PUB/SUB over TCP** (like ITCH over UDP multicast: push, one-to-many,
  sequence-numbered). Subscribers detect gaps via `seq`; the 1 Hz `tick` snapshot doubles as the
  recovery/snapshot channel.
- **Sequencer:** the engine stamps a monotonically increasing `seq` (int, starts 1, never resets
  within a run) on every accepted order, fill, and published message. Total order = replayable.

## 2 · Ports & environment variables (.env)

| Var | Default | Meaning |
|---|---|---|
| `ZMQ_HOST` | `127.0.0.1` | Engine host |
| `ZMQ_ORDER_PORT` | `5555` | ROUTER (order entry) |
| `ZMQ_DATA_PORT` | `5556` | PUB (market data) |
| `ORACLE_PORT` | `5557` | Oracle ROUTER (news scoring requests from bridge) |
| `BRIDGE_HOST` / `BRIDGE_PORT` | `0.0.0.0` / `8000` | FastAPI bridge |
| `CHRONOS_DB` | `chronos.db` | SQLite file (engine writes, bridge reads — WAL mode) |
| `GEMINI_API_KEY` | — | Oracle scoring tier 1 (Gemini; free tier is 20 req/day/model) |
| `GROQ_API_KEY` | — | Oracle scoring tier 2, used when tier 1 fails or its quota is spent |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Override if Groq retires the default model |
| `CHRONOS_SEED` | — | Swarm worker seed (same as `worker.py --seed`); unset = unseeded swarm |
| `CHRONOS_LOCKSTEP_WORKERS` | `0` | Engine waits for `TICK_DONE` from this many swarm workers before each physics tick (§4.3); `0` = off |
| `CHRONOS_LOCKSTEP_TIMEOUT_S` | `10` | Longest the engine waits for a silent worker before ticking anyway (publishes a `lockstep` event) |

**Reproducible live runs** need all three of: the same `INIT_SIM`/`REPLAY_START` seed, the same
worker `--seed`, and lockstep on. The swarm then counts cooldowns in engine ticks, draws from
per-agent RNGs seeded from the run seed, submits orders one agent at a time, and ignores paused
ticks (`agents/swarm_factory.py`, `core/loop_manager.py`). Two such runs give the same trade tape,
even if a worker stalls. User strategies (`STRAT_`/`USER_`) are outside lockstep by design.

## 3 · Venue model (ONE consistent venue — India/NSE style)

| Parameter | Value |
|---|---|
| Currency | INR, display symbol `₹` |
| Tick size | **Derived from the instrument's price band** (see below) — all limit prices rounded to tick by the engine. Published every tick as `tick_size` (§5.1); clients must not hardcode it. |
| Session | 09:15 → 15:30 = **375 sim-minutes/day** |
| Time model | **1 real second = 1 sim-minute** (one physics tick) |
| Sim epoch | day 1 starts at unix `1767258900` (2026-01-01 09:15 IST-as-UTC) |
| Overnight | after minute 375, clock jumps to 09:15 of the next **weekday** (Sat/Sun skipped), `day_count += 1`. Minute `m` is stamped `09:15 + m min` on its session's date (bar-close time). Exchange holidays are not modelled. |
| Lot size | 1 share; `qty` is always an int > 0 |

### 3.1 · Tick size bands

A single fixed tick cannot serve every price: `0.05` is 0.026 % of ₹190 but 1.6 % of ₹3 — coarse
enough that a low-priced instrument cannot move at all, which flattens every candle and saturates
the agents' `volatility` / `volume_ratio` features.

The bands are NSE's cash-market tick table, in force since 2025-04-15:

| Price | Tick |
|---|---|
| < ₹250 | `0.01` |
| ₹250 – ₹1 000 | `0.05` |
| ₹1 000 – ₹5 000 | `0.10` |
| ₹5 000 – ₹10 000 | `0.50` |
| ₹10 000 – ₹20 000 | `1.00` |
| ₹20 000 + | `5.00` |

Every band was checked against real NSE 1-minute closes (Kaggle NIFTY-100 data, 2025-07 onward):
all closes of the tested stocks sit on their band's grid.

**Review cadence.** The tick is derived at `INIT_SIM` / `REPLAY_START` from the start price, then
re-checked **only on the first session of each calendar month**, from the previous month's last
close. This matches NSE: in the real data a stock that crossed a band edge changed tick on the
1st of the next month (HDFCBANK, SBIN, 2025–26). It is never re-derived intraday: re-gridding
mid-session would strand every resting order on a price that is no longer valid. When a review
does change the tick, the engine starts that session with a fresh book.

## 4 · Order-entry plane — message schemas (JSON, UTF-8, one frame after identity)

Clients are **DEALER** sockets; set `IDENTITY` to a unique string (e.g. `MM_3`, `BRIDGE`,
`STRAT_ab12`). Engine sends replies/acks back to the same identity, single JSON frame.
All requests carry `"msg"`. Unknown fields are ignored; missing required fields → REJECTED ack.

### 4.1 Trading messages

```jsonc
// New order
{ "msg": "ORDER", "agent_id": "WHALE_0", "action": "BUY" | "SELL",
  "type": "MARKET" | "LIMIT", "qty": 100,
  "price": 190.05,            // required for LIMIT; ignored for MARKET
  "client_order_id": "w0-17", // optional echo field
  "tif_ticks": 20,            // optional time-to-live in physics ticks (default 60; 0 = day)
  "start_cash": 85000.0 }     // optional, Retail agents only, honoured on the agent's FIRST message
                              // (ORDER or CANCEL_ALL) and ignored afterwards; must be in
                              // [10_000, 1e9], otherwise the default 50_000 ledger is used

// Cancel one / all resting orders
{ "msg": "CANCEL", "agent_id": "MM_3", "order_id": 4711 }
{ "msg": "CANCEL_ALL", "agent_id": "MM_3" }
```

An ORDER may carry an optional `"meta"` object. If `meta.news` is present
(`{"headline", "score", "sector", "reasoning", "source"}`), the engine publishes an
`event kind=news` with that payload **before** executing the order (this is how the oracle's
scored shock reaches the UI — the oracle cannot publish on the engine's PUB socket itself).

### 4.2 Ack schema (engine → sender, exactly one per request)

```jsonc
{ "msg": "ACK", "status": "FILLED" | "PARTIAL" | "RESTING" | "CANCELLED" | "REJECTED",
  "order_id": 4711,           // engine-assigned, 0 if rejected
  "client_order_id": "w0-17", // echoed if provided
  "executed_qty": 60, "average_price": 190.10, "resting_qty": 40,
  "cash": 4988600.0, "pos": 160,   // sender's ledger AFTER this action (authoritative)
  "reason": "...",            // only when REJECTED/CANCELLED
  "seq": 88123 }
```

Semantics: a **marketable LIMIT executes immediately** up to its limit price; the remainder
rests. MARKET walks the book; unfilled remainder is discarded (status PARTIAL). Matching is
strict **price–time priority**. Self-trades are prevented (an order never matches the same
`agent_id`; those resting orders are skipped).

### 4.3 Control messages (bridge/tools only)

```jsonc
{ "msg": "INIT_SIM", "symbol": "TCS", "sector": "TECH", "price": 190.0, "seed": 42 }
→ { "msg": "ACK", "status": "ENGINE_READY", "run_id": "run_2026...","seq": 1 }

{ "msg": "REPLAY_START", "csv_path": "data/TCS_1min.csv", "seed": 42,
  "symbol": "TCS", "sector": "TECH" }        // CSV columns: date,time,open,high,low,close,volume
→ { "msg": "ACK", "status": "REPLAY_READY", "run_id": "...", "bars": 375, "seq": 1 }

{ "msg": "HANDOVER", "at_bar": 120 }        // REPLAY only; at_bar omitted = bars replayed so far
→ { "msg": "ACK", "status": "HANDOVER_ARMED", "at_bar": 120, "bars_replayed": 37, "seq": ... }
   // REJECTED if not replaying, at_bar already replayed, beyond the tape, or not an int (§9)

{ "msg": "PAUSE" }  |  { "msg": "RESUME" }  → ACK status "PAUSED" | "RUNNING"
{ "msg": "FETCH_STATE" }                    → the current tick payload (§5.1) — poll fallback
{ "msg": "FETCH_BARS", "n": 400 }           → { "msg": "ACK", "status": "BARS", "run_id": "...", "n": 400,
                                                 "bars": [{"day_count", "minute", "o", "h", "l", "c", "v"}, ...] }
   // the last n closed bars, oldest first (n = 1-1000, default 375; REJECTED otherwise). Read-only, works while
   // paused. Late joiners use it to rebuild core/market_context.py (MarketContext.seed). The engine keeps the last 1000.
{ "msg": "PING" }                           → { "msg": "ACK", "status": "PONG", ... }

// Swarm worker → engine server, after acting on every unpaused tick (lockstep, §2).
{ "msg": "TICK_DONE", "agent_id": "SWARM_ALL_1234", "tick_seq": 812 }   // tick_seq = that tick's seq
→ { "msg": "ACK", "status": "TICK_DONE" | "TICK_DONE_STALE", "seq": ... }
// Handled by engine/server.py, not the engine core; harmless (STALE) when lockstep is off.
```

## 5 · Market-data plane — PUB topics (multipart: [topic-bytes, json-bytes])

### 5.1 Topic `tick` — full snapshot, once per physics tick (1 Hz)

```jsonc
{ "seq": 88124, "run_id": "run_...", "mode": "LIVE" | "REPLAY" | "REACTIVE",
  "paused": false,
  "unix_time": 1767258960, "day_count": 1, "market_minute": 2,
  "symbol": "TCS", "sector": "TECH",
  "last_price": 190.10,          // last TRADE price — the only price authority
  "tick_size": 0.05,             // venue tick for the current price band (§3.1).
                                 // Clients MUST snap LIMIT prices to this — it is
                                 // re-derived at each session open, so a hardcoded
                                 // client-side tick will drift out of sync.
  "best_bid": 190.05, "best_ask": 190.15, "mid_price": 190.10,
  "micro_price": 190.0862,       // size-weighted mid
  "spread": 0.10,
  "bar": { "minute": 2, "o": 190.0, "h": 190.2, "l": 189.95, "c": 190.10, "v": 5400 },
  "vwap": 190.04,                // cumulative day VWAP from real trades
  "rsi": 51.2,                   // 14-period Wilder on minute closes
  "ofi": 0.18,                   // real: (buy_vol - sell_vol)/(total+1) last minute, book-blended
  "volatility": 0.0013,          // (bar.h - bar.l) / bar.c of last completed bar
  "volume_ma": 6100.0,           // 20-minute rolling mean of per-minute traded volume
  "step_volume": 5400,           // real traded volume this tick
  "regime": "BULL" | "BEAR" | "RANGING",
  "lob_bids": [[190.05, 800], ...],   // top 10 aggregated levels, best first
  "lob_asks": [[190.15, 650], ...],
  "leaderboard": [ { "id": "WHALE_0", "type": "Institution", "pnl": 1250.5,
                     "pos": 160, "cash": 4988600.0 }, ... ]  // top 15 by pnl, no GHOST/LP
}
```

### 5.2 Topic `trade` — every print, immediately

```jsonc
{ "seq": 88125, "price": 190.10, "qty": 60, "buyer": "WHALE_0", "seller": "SIM_LP",
  "aggressor": "BUY" | "SELL", "unix_time": 1767258960, "market_minute": 2 }
```

### 5.3 Topic `event` — everything else

```jsonc
{ "seq": ..., "kind": "news",        "headline": "...", "score": -0.62,
  "sector": "TECH", "reasoning": "...", "source": "gemini" | "fallback" }
{ "seq": ..., "kind": "liquidation", "agent_id": "RETAIL_12" }   // engine reset ledger to 50k
{ "seq": ..., "kind": "mode_change", "mode": "REACTIVE", "trigger_agent": "STRAT_ab12" | "HANDOVER",
               "bars_replayed": 120, "bars_remaining": 255 }
{ "seq": ..., "kind": "session",     "action": "day_open" | "day_close" | "reset" | "paused" | "resumed",
               "day_count": 2 }
{ "seq": ..., "kind": "lockstep",    "action": "timeout", "missing_workers": 1 }   // run no longer reproducible
```

## 6 · Bridge (FastAPI :8000)

- `GET /` → serves `dashboard/index.html` (static mount of `dashboard/`).
- `WS /ws` → pushes every PUB message as `{"topic": "tick"|"trade"|"event", "data": {...}}`.
- REST (all JSON):

| Endpoint | Body → Response |
|---|---|
| `POST /api/sim/start` | `{symbol, sector, price, seed?}` → `{status, run_id}` |
| `POST /api/sim/stop` | `{}` → `{status}` (PAUSE) |
| `POST /api/sim/resume` | `{}` → `{status}` |
| `POST /api/news` | `{headline}` → `{status:"accepted"}` (score arrives via `event`) |
| `POST /api/replay/start` | `{csv_path, symbol?, sector?, seed?}` → `{status, run_id, bars}` |
| `POST /api/strategy/check` | `{code}` → `{ok, errors:[{stage,line,message,code?}], warnings, orders, ticks_tested, output, backend, dry_run, duration_ms}`. Static check + dry run in the sealed sandbox; never trades. Always 200 |
| `GET /api/strategy/sandbox` | → `{mode: runner\|docker\|unsafe-subprocess\|unavailable, sealed}` |
| `POST /api/replay/handover` | `{at_bar?}` → `{status, at_bar, bars_replayed}`; 409 with the reason if rejected (§9) |
| `POST /api/strategy/run` | `{name, code}` → `{strategy_id, agent_id, backend}` (python only). Runs the same check first: `400` lists each problem with its line, `503` if no sealed sandbox is available |
| `POST /api/strategy/stop` | `{strategy_id}` → `{status}` |
| `GET /api/strategy/list` | → `[{strategy_id, agent_id, name, status, started_at}]` |
| `GET /api/strategy/{id}/logs` | → `{logs: "..."}` (tail of sandboxed stdout/stderr) |
| `GET /api/runs` | → `[{run_id, started_at, symbol, mode, seed}]` |
| `GET /api/runs/{run_id}/equity?agent_id=X` | → `{points: [[seq, equity], ...]}` |
| `GET /api/health` | → `{status:"ok", engine:"up"|"down"}` |

## 7 · Persistence (SQLite, engine-owned, WAL)

```sql
runs(run_id TEXT PK, started_at TEXT, symbol TEXT, sector TEXT, mode TEXT,
     seed INTEGER, config_json TEXT);
ticks(run_id TEXT, seq INTEGER, unix_time INTEGER, market_minute INTEGER, day_count INTEGER,
      last_price REAL, best_bid REAL, best_ask REAL, vwap REAL, rsi REAL, ofi REAL,
      step_volume INTEGER, regime TEXT, PRIMARY KEY(run_id, seq));
trades(run_id TEXT, seq INTEGER, unix_time INTEGER, price REAL, qty INTEGER,
       buyer TEXT, seller TEXT, aggressor TEXT, PRIMARY KEY(run_id, seq));
agent_snapshots(run_id TEXT, seq INTEGER, agent_id TEXT, agent_type TEXT,
                cash REAL, pos INTEGER, equity REAL, PRIMARY KEY(run_id, seq, agent_id));
```
Batched writes once per tick; readers (bridge) open read-only.

## 8 · Engine behavioral spec

1. **Price is ONLY produced by trades.** `last_price` changes exclusively when a fill occurs in
   the book. NO `random.gauss` price drift. Mid/micro derive from real best bid/ask.
2. **Impact is emergent** — market orders walk and deplete real depth. No magic-constant gap,
   no book-wipe. Large orders simply eat many levels.
3. **Regime physics inject FLOW, not price:** GHOST_TREND / GHOST_REVERSAL submit **real
   orders** through the same matching path (BULL → buy flow, BEAR → sell flow, sized by the
   U-shape multiplier). Regime shifts on a timer as before.
4. **Designated liquidity (`SIM_LP`)**: an engine-internal agent that keeps ~8 price levels per
   side quoted around last_price (size scaled to recent volume), refreshed each tick via real
   cancel/replace. Guarantees baseline depth like a real venue's designated MM. Its ledger is
   tracked but excluded from the leaderboard.
5. **OFI, VWAP, RSI, OHLC bars, volume — all computed from real trades/book only.**
6. **Single liquidation authority:** engine checks retail equity < ₹1,000 → resets ledger to
   the agent's own starting cash (₹50,000 unless it sent `start_cash`) / 0 pos, publishes
   `event kind=liquidation`. Workers only react to the event.
7. **Determinism:** engine owns `self.rng = random.Random(seed)`; NO module-level `random.*`
   in engine code. Same seed + same order stream ⇒ identical run.
8. **Order TTL:** resting orders expire after `tif_ticks` physics ticks (default 60) via the
   real cancel path (prevents infinite stale depth; simulation policy, documented).
9. **Ledger conservation invariant:** across any fill, sum of cash and sum of position over
   all agents is unchanged (tested).

### Sanctioned randomness (the ONLY allowed randomness, all via seeded RNGs)
- Agent *behavioural* noise: retail 10% irrationality, sleep/wake participation, cooldown jitter.
- Regime timer durations & ghost-flow sizes (seeded, engine RNG).
- NOTHING else — no random prices, spreads, wicks, colors, OFI, or fills.

## 9 · Replay / history-repeater (Pillar 1)

- `REPLAY_START` loads a CSV of 1-minute bars. Engine enters `mode="REPLAY"`, seeds RNG.
- Each physics tick = next historical minute. The engine's **HistoricalReplayer** drives the
  tape THROUGH the real book: it manages `SIM_LP`-style bracketing depth around the historical
  path and fires ghost orders so trades print the bar's prices inside the minute. Swarm agents
  may trade during replay; their fills are real but the replayer re-anchors price to history.
- **Fidelity (swarm off):** every bar is reproduced exactly — o, h, l, c to the paisa and the full
  volume (no cap). The first print is at the open (bars open at their first trade in every mode,
  not at the previous close); every print stays inside [l, h]. The path is o→h→l→c or o→l→h→c,
  whichever needs fewer prints (a bar does not record which extreme came first). A zero-volume
  bar flat at the previous close prints nothing. A tape bar with fewer shares than the prints its
  prices need (seen in thin names' vendor data) keeps its prices and prints the minimum volume;
  such bars are listed in `HistoricalReplayer.inexact_bars`.
- Replay timestamps and tick reviews follow the sim calendar (§3), not the CSV's own dates.
- **The latch:** the FIRST fill whose buyer or seller `agent_id` starts with `STRAT_` or `USER_`
  flips `mode` to `"REACTIVE"` permanently (published as `event kind=mode_change`). From then on
  the replayer stops re-anchoring; the book + swarm own the price. State carries over seamlessly
  (book, ledgers, RSI window, regime inferred from recent drift) — no price gap.
- Same seed + same CSV + no user trade ⇒ bit-identical history (tested).
- **Hand-over (control runs):** `HANDOVER {at_bar: N}` (or `POST /api/replay/handover`) flips to
  REACTIVE with no user trade, at the start of the physics tick that would replay bar N+1 — the
  same point at which a user trade placed after bar N latches. Book, ledgers, RSI window and the
  RNG stream carry over (no reseed). A control run (HANDOVER at N) and a treatment run (user
  trades after bar N) are therefore identical up to the switch and differ afterwards only by
  the user's own orders. A later HANDOVER replaces an armed one; the user-trade latch still
  applies if it comes first; `at_bar` = tape length hands over instead of pausing at tape end.
  The rest of the tape is not replayed; the CSV itself is the record of what history did next.
  For a reproducible control run with the swarm on, use a seeded worker with lockstep (§2) or the
  in-process harness (`validation/sim_driver.py`).

## 10 · Strategy SDK (runner/sdk.py)

```python
class Strategy:                      # user subclasses this
    def on_tick(self, state: dict) -> list[dict]: ...   # state = §5.1 payload
    def on_fill(self, ack: dict) -> None: ...           # optional

run_strategy(MyStrategy, name="mean_rev")   # blocking; SUB loop + DEALER orders
```
- Orders returned use §4.1 schema minus `agent_id`/`msg` (SDK injects; agent_id = `STRAT_<id>`).
- SDK validates schema, throttles to ≤ 5 orders/tick, handles SIGTERM cleanly (CANCEL_ALL).
- Sandbox (`runner/sandbox.py`): `docker` backend (resource-limited container) when available,
  else `subprocess` backend with psutil memory/CPU watchdog + wall-clock kill. Subprocess mode
  is honestly documented as NOT a security boundary.

## 11 · File ownership (Phase-2 build)

| Module | Owner agent | Files |
|---|---|---|
| Engine | ENGINE | `engine/__init__.py, engine/book.py, engine/matching_engine.py, engine/replay.py, engine/persistence.py, engine/server.py` |
| Swarm | SWARM | `core/zmq_helpers.py, core/engine_client.py, core/loop_manager.py, core/__init__.py, agents/features.py, agents/live_agents.py, agents/environments.py, agents/base_environment.py, agents/heuristic_retail.py, worker.py, main.py` |
| Runner | RUNNER | `runner/__init__.py, runner/sdk.py, runner/sandbox.py, runner/fix_gateway.py, runner/examples/*` |
| Bridge+Oracle | BRIDGE | `bridge.py, oracle.py` |
| Frontend | FRONTEND | `dashboard/index.html, dashboard/styles.css, dashboard/app.js` |
| Meta/tests | INTEGRATOR | `tests/*, README.md, requirements.txt, .env.example, PROTOCOL.md` |

Python: 3.11+. Windows-compatible (use `asyncio.WindowsSelectorEventLoopPolicy` before any
`zmq.asyncio` loop). Logging via `logging`, format `'%(asctime)s - [<MODULE>] - %(message)s'`.
Type hints on public functions. No file outside your ownership row.

## 12 · Internal Python APIs (cross-module contracts)

### 12.1 Engine core (`engine/matching_engine.py`) — transport-free, fully testable

```python
class MatchingEngine:
    def __init__(self, db: Persistence | None = None): ...
    def init_sim(self, symbol: str, sector: str, price: float, seed: int = 42) -> str: ...
    def start_replay(self, csv_path: str, symbol: str | None, sector: str, seed: int = 42) -> tuple[str, int]: ...
    # Handles every §4 message (ORDER/CANCEL/CANCEL_ALL/FETCH_STATE/PING/PAUSE/RESUME/
    # INIT_SIM/REPLAY_START). Returns (ack, publications) where publications is a list of
    # (topic, payload) to push on the data plane (e.g. immediate `trade` prints, `event`s).
    def handle_message(self, msg: dict) -> tuple[dict, list[tuple[str, dict]]]: ...
    # One physics tick (1 sim-minute): clock, regime flow, SIM_LP refresh, TTL expiry,
    # bar close, persistence flush. Returns publications (always ends with one `tick`).
    def physics_tick(self) -> list[tuple[str, dict]]: ...
```
`engine/server.py` is a thin transport shell around this class (ROUTER + PUB + 1 Hz timer).

### 12.2 Sandbox manager (`runner/sandbox.py`) — used by the bridge

```python
class SandboxManager:
    def __init__(self, base_dir: str = ".strategies"): ...
    def check(self, code: str) -> dict:   # static check + dry run; never launches
    def launch(self, name: str, code: str) -> dict:   # {"strategy_id","agent_id","backend"}
                                          # ValueError: failed check; SandboxUnavailable: no sealed backend
    def mode(self) -> str:                # runner | docker | unsafe-subprocess | unavailable
    def stop(self, strategy_id: str) -> bool: ...
    def list(self) -> list[dict]:  # [{"strategy_id","agent_id","name","status","started_at"}]
    def logs(self, strategy_id: str, tail: int = 200) -> str: ...
```
Strategy processes run user code that defines `class UserStrategy(Strategy)` (see §10);
artifacts live under `swarm_orchestrator/.strategies/<strategy_id>/` (gitignored).

**Sealed execution.** Strategy code never runs in the bridge's own process space. It runs in a
container from the `chronos-sandbox` image (SDK + pyzmq only, no project files or keys), with all
capabilities dropped, a read-only root, 512 MB / 1 CPU / 64 processes, and a network that reaches the
engine only. Under Compose the bridge asks the **runner service** (`runner/service.py`, the only
container holding the Docker socket, port 8100 on the internal network, `X-Runner-Token`) to start it.
With no sealed backend, `launch()` refuses; the old plain-subprocess mode runs only if
`CHRONOS_ALLOW_UNSAFE_SANDBOX=1` (local development). Every launch first runs `runner.check.static_check`
and a dry run (`runner/dryrun.py`: `on_start`, `on_tick` on calm/dip/spike sample ticks, `on_fill`,
`on_stop`), and reports problems with line numbers. User guide: `docs/STRATEGY_GUIDE.md`.
FIX gateway listens on `FIX_PORT` (default `9878`).
