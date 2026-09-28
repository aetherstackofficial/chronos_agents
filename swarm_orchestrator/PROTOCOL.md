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
| `GEMINI_API_KEY` | — | Oracle LLM key |

## 3 · Venue model (ONE consistent venue — India/NSE style)

| Parameter | Value |
|---|---|
| Currency | INR, display symbol `₹` |
| Tick size | `0.05` — all limit prices rounded to tick by the engine |
| Session | 09:15 → 15:30 = **375 sim-minutes/day** |
| Time model | **1 real second = 1 sim-minute** (one physics tick) |
| Sim epoch | day 1 starts at unix `1767258900` (2026-01-01 09:15 IST-as-UTC) |
| Overnight | after minute 375, clock jumps +17.5 h to next 09:15, `day_count += 1` |
| Lot size | 1 share; `qty` is always an int > 0 |

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
  "tif_ticks": 20 }           // optional time-to-live in physics ticks (default 60; 0 = day)

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

{ "msg": "PAUSE" }  |  { "msg": "RESUME" }  → ACK status "PAUSED" | "RUNNING"
{ "msg": "FETCH_STATE" }                    → the current tick payload (§5.1) — poll fallback
{ "msg": "PING" }                           → { "msg": "ACK", "status": "PONG", ... }
```

## 5 · Market-data plane — PUB topics (multipart: [topic-bytes, json-bytes])

### 5.1 Topic `tick` — full snapshot, once per physics tick (1 Hz)

```jsonc
{ "seq": 88124, "run_id": "run_...", "mode": "LIVE" | "REPLAY" | "REACTIVE",
  "paused": false,
  "unix_time": 1767258960, "day_count": 1, "market_minute": 2,
  "symbol": "TCS", "sector": "TECH",
  "last_price": 190.10,          // last TRADE price — the only price authority
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
{ "seq": ..., "kind": "mode_change", "mode": "REACTIVE", "trigger_agent": "STRAT_ab12" }
{ "seq": ..., "kind": "session",     "action": "day_open" | "day_close" | "reset" | "paused" | "resumed",
               "day_count": 2 }
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
| `POST /api/strategy/run` | `{name, code}` → `{strategy_id, agent_id}` (python only) |
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
   ₹50,000 cash / 0 pos, publishes `event kind=liquidation`. Workers only react to the event.
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
  path and fires ghost market orders so trades print o→h→l→c inside the minute. Swarm agents
  may trade during replay; their fills are real but the replayer re-anchors price to history.
- **The latch:** the FIRST fill whose buyer or seller `agent_id` starts with `STRAT_` or `USER_`
  flips `mode` to `"REACTIVE"` permanently (published as `event kind=mode_change`). From then on
  the replayer stops re-anchoring; the book + swarm own the price. State carries over seamlessly
  (book, ledgers, RSI window, regime inferred from recent drift) — no price gap.
- Same seed + same CSV + no user trade ⇒ bit-identical history (tested).

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
    def launch(self, name: str, code: str) -> dict:   # {"strategy_id","agent_id","backend"}
    def stop(self, strategy_id: str) -> bool: ...
    def list(self) -> list[dict]:  # [{"strategy_id","agent_id","name","status","started_at"}]
    def logs(self, strategy_id: str, tail: int = 200) -> str: ...
```
Strategy processes run user code that defines `class UserStrategy(Strategy)` (see §10);
artifacts live under `swarm_orchestrator/.strategies/<strategy_id>/` (gitignored).
FIX gateway listens on `FIX_PORT` (default `9878`).
