# Chronos PROTOCOL v2 — Feature Layer (additive to PROTOCOL.md)

This document extends `PROTOCOL.md` (unchanged, still authoritative for the engine, the two
network planes, and §1–§12). v2 adds the **product feature layer**: a script library, a
session results/reporting store, and four new capabilities (Monte-Carlo counterfactual fan,
LLM strategy copilot, Time-Machine crisis replay, explainable narration).

**Storage model:** everything in v2 is **session-scoped** — held in memory in the bridge process
(the "session"), plus scripts saved as real files on disk so users can refer to them again.
A durable DB comes later; the code is written so swapping the in-memory stores for a DB is a
localized change.

**Integration model (no merge conflicts):** backend agents write **self-contained logic modules
+ one `APIRouter` each**. The integrator wires them into `bridge.py` with `app.include_router(...)`
and a single pump hook. Routers read shared singletons from `request.app.state`:

```python
request.app.state.results        # analytics.results.ResultsStore
request.app.state.scripts        # runner.script_store.ScriptStore
request.app.state.sandbox        # runner.sandbox.SandboxManager (may raise 503 if unavailable)
request.app.state.control        # async callable: await control(msg: dict, timeout=3.0) -> ack|None
request.app.state.latest_tick    # callable: latest_tick() -> dict|None (most recent §5.1 tick)
request.app.state.recent_trades  # callable: recent_trades(n=200) -> list[trade payloads §5.2]
```

---

## 1 · Script Library (save/refer strategy scripts as files)

**Module (analytics agent):** `runner/script_store.py`

```python
class ScriptStore:
    def __init__(self, base_dir: str = "scripts_library"): ...
    def save(self, name: str, code: str, language: str = "python",
             script_id: str | None = None) -> dict:
        # writes scripts_library/<script_id>.py + updates index.json; validates non-empty code.
        # returns {script_id, name, language, saved_at, updated_at, size, path}
    def update(self, script_id: str, name: str | None, code: str | None) -> dict: ...
    def get(self, script_id: str) -> dict:      # {script_id, name, code, language, saved_at, updated_at, size}
    def list(self) -> list[dict]:               # meta only (no code), newest first
    def delete(self, script_id: str) -> bool: ...
```
IDs are 8 hex. Files persist across bridge restarts (they are real files); the index is
`scripts_library/index.json`. `scripts_library/` is gitignored.

**Router (analytics agent):** `analytics/routes_scripts.py` exposing `router = APIRouter()`:

| Method + path | Body → Response |
|---|---|
| `POST /api/scripts` | `{name, code, language?}` → `{script_id, name, saved_at, ...}` (400 empty code) |
| `GET /api/scripts` | → `[{script_id, name, language, saved_at, updated_at, size}]` |
| `GET /api/scripts/{id}` | → `{script_id, name, code, ...}` (404) |
| `PUT /api/scripts/{id}` | `{name?, code?}` → updated meta |
| `DELETE /api/scripts/{id}` | → `{status:"deleted"}` |
| `POST /api/scripts/{id}/run` | `{}` → `{strategy_id, agent_id, backend, script_id}` — loads code, launches via `sandbox`, and **registers the run with the results store** (`results.attach_strategy(strategy_id, agent_id, name, script_id)`) |

## 2 · Results / Session store + reports (CSV + PDF)

**Module (analytics agent):** `analytics/results.py`

```python
class ResultsStore:
    """Session-scoped record of every strategy's trades & outcomes, organized
    strategy -> run -> day. Fed live by the bridge pump; authoritative trade
    source is the engine's `trade` stream filtered to the strategy's agent_id."""
    def __init__(self): ...
    def attach_strategy(self, strategy_id: str, agent_id: str, name: str,
                        script_id: str | None) -> None: ...
    def on_run(self, run_id: str, symbol: str, mode: str, seed: int) -> None:  # from event/tick
    def ingest(self, topic: str, payload: dict) -> None:
        # called for EVERY pump message. On 'trade': if buyer or seller is a tracked
        # strategy agent_id, record a fill (side, price, qty, cash impact, unix_time,
        # market_minute, day_count, run_id). On 'tick': snapshot each tracked strategy's
        # equity/pos/cash from leaderboard for per-day P&L & drawdown. On 'event'
        # kind=session day_close: close the day bucket.
    def list_strategies(self) -> list[dict]:
        # [{strategy_id, agent_id, name, script_id, status, started_at, total_trades,
        #   total_pnl, realized_pnl, n_days, n_runs}]
    def strategy_detail(self, strategy_id: str) -> dict:
        # {..., days: [{run_id, day_count, symbol, trades:[...], n_trades, gross_pnl,
        #   start_equity, end_equity, max_drawdown, win_rate}]}
    def strategy_trades(self, strategy_id: str, day: int | None = None) -> list[dict]:
    def mark_status(self, strategy_id: str, status: str) -> None:  # running/stopped
```
A "trade" row: `{ts, unix_time, day_count, market_minute, run_id, side, price, qty,
cash_delta, counterparty}`. Equity/P&L come from the engine leaderboard (authoritative ledger).

**Module (analytics agent):** `analytics/reports.py`
```python
def trades_csv(rows: list[dict], meta: dict) -> bytes            # RFC-4180 CSV
def outcomes_csv(days: list[dict], meta: dict) -> bytes          # per-day summary CSV
def strategy_pdf(detail: dict) -> bytes                          # reportlab PDF: header, summary
                                                                 # table, per-day table, trade log
```

**Router (analytics agent):** `analytics/routes_results.py` `router = APIRouter()`:

| Method + path | Response |
|---|---|
| `GET /api/results` | `[strategy summary ...]` (from `list_strategies`) |
| `GET /api/results/{sid}` | full `strategy_detail` |
| `GET /api/results/{sid}/trades` | `{trades: [...]}` (optional `?day=N`) |
| `GET /api/results/{sid}/trades.csv` | `text/csv` download (Content-Disposition attachment) |
| `GET /api/results/{sid}/outcomes.csv` | per-day summary CSV download |
| `GET /api/results/{sid}/report.pdf` | `application/pdf` download |

CSV/PDF responses set `Content-Disposition: attachment; filename="<name>_<sid>.csv|pdf"`.

## 3 · Monte-Carlo Counterfactual Fan (real, in-process)

**Module (features agent):** `features/montecarlo.py`
```python
def run_fan(symbol: str, price: float, n_paths: int = 100, horizon: int = 120,
            seed_base: int = 1000, shock: dict | None = None,
            max_paths: int = 500, max_horizon: int = 750) -> dict:
    """Spin up n_paths transport-free MatchingEngine instances (engine.matching_engine),
    each seeded seed_base+i, run `horizon` physics ticks, collect the last_price path.
    Optional shock={minute:int, side:'BUY'|'SELL', qty:int} injects a real market order
    at that tick (a 'what if I trade this size' counterfactual). Uses a process/thread
    pool for speed. NO sockets. Returns:"""
    return {
      "symbol": symbol, "n_paths": ..., "horizon": ...,
      "percentiles": {"p5": [...], "p25": [...], "p50": [...], "p75": [...], "p95": [...]},
      "mean": [...], "sample_paths": [[...], ...],   # up to 30 thin sample paths for the fan
      "terminal": {"p5":..,"p25":..,"p50":..,"p75":..,"p95":..,"mean":..,"min":..,"max":..},
      "start_price": price
    }
```
Clamp n_paths<=max_paths, horizon<=max_horizon. Deterministic given seed_base.

**Router (features agent):** `features/routes_features.py` (shared by all four features):
`POST /api/montecarlo` `{symbol, price, n_paths?, horizon?, seed_base?, shock?}` → the dict above.

## 4 · LLM Strategy Copilot (NL → strategy code)

**Module (features agent):** `features/copilot.py`
```python
async def generate_strategy(description: str, api_key: str | None = None) -> dict:
    """Gemini (gemini-2.5-flash, async, 12s timeout) turns a natural-language description
    into a valid `class UserStrategy(Strategy)` body (SDK contract, PROTOCOL §10). Validates
    the result: compile() + 'class UserStrategy' present + defines on_tick. On any failure or
    missing key, returns a REAL deterministic template built from parsed hints (RSI/VWAP
    thresholds, side, qty) — never a stub. Returns:"""
    return {"code": "<python>", "explanation": "<1-3 sentences>", "source": "gemini"|"template"}
```
`POST /api/copilot` `{description}` → the dict. The UI shows the code in Monaco for review; the
user saves/runs it via the normal script endpoints (copilot never auto-runs code).

## 5 · Time Machine (crisis replay presets)

**Module (features agent):** `features/timemachine.py`
```python
SCENARIOS: list[dict]  # [{id, name, description, symbol, sector, bars, tag}]
def ensure_scenarios(data_dir: str = "data/scenarios") -> None:
    """Deterministically GENERATE real 1-minute CSV tapes for each scenario if absent:
    flash_crash_2010, covid_crash_2020, gme_squeeze_2021, black_monday_1987, rate_shock.
    Each CSV has columns date,time,open,high,low,close,volume and a plausible shaped path
    (pre-event drift, the event move, aftermath). Files are real inputs to engine replay."""
def get_scenario(scenario_id: str) -> dict | None:
def scenario_csv_path(scenario_id: str) -> str:
```
Router: `GET /api/timemachine/scenarios` → `SCENARIOS`; `POST /api/timemachine/start`
`{scenario_id, seed?}` → forwards a `REPLAY_START` (csv_path = scenario file) to the engine via
`request.app.state.control` → `{status, run_id, bars, scenario}`. Replaying a scenario means: the
market re-lives that crisis, and the moment a user strategy trades, it latches to REACTIVE
(PROTOCOL §9) — history diverges. That IS the product thesis, made demoable.

## 6 · Explainable Market Narration

**Module (features agent):** `features/narrator.py`
```python
async def narrate(tick: dict, recent_trades: list[dict], api_key: str | None = None) -> dict:
    """Explain WHY the market is doing what it is, from the latest §5.1 tick + recent trades
    (regime, OFI, spread, who's winning on the leaderboard, recent aggressor balance). Gemini
    (async, 8s) with a REAL deterministic heuristic fallback that composes a sentence from the
    numbers (never a stub). Returns {narration: str, source: 'gemini'|'heuristic'}."""
```
Router: `POST /api/narrate` `{}` → uses `request.app.state.latest_tick()` + `recent_trades()`.

## 7 · Frontend shared library API (so shell & page agents align)

All pages are plain ES modules importing from `dashboard/lib/`. **Shell agent owns `lib/` and
`design-tokens.css`; pages agent imports them.** Exact API:

```js
// lib/store.js  — single WS /ws connection, auto-reconnect (2s backoff)
export const store = {
  state: { tick, connected, mode, day, price, regime },  // live, updated in place
  onTick(fn), onTrade(fn), onEvent(fn), onStatus(fn),     // subscribe; returns unsubscribe()
  latestTick(),                                           // last tick or null
  recentTrades(n),                                        // ring buffer of last 200 trades
};
// lib/api.js — every REST endpoint as a promise-returning method (throws ApiError on !ok)
export const api = {
  sim:{start,stop,resume}, news(headline), replay(cfg), health(),
  strategy:{run(name,code), stop(id), list(), logs(id)},
  scripts:{list(), get(id), save(name,code), update(id,p), del(id), run(id)},
  results:{list(), detail(id), trades(id,day), csvUrl(id), outcomesCsvUrl(id), pdfUrl(id)},
  montecarlo(cfg), copilot(description), timemachine:{scenarios(), start(id,seed)}, narrate(),
  runs(), equity(runId, agentId),
};
// lib/format.js
export const fmt = { inr(n), signed(n), pct(n), compact(n), time(unix), dtime(unix), num(n,d) };
// lib/ui.js
export const ui = { toast(msg,kind), modal({title,body,actions}), drawer(...), icon(name), confirm(...) };
// lib/charts.js  — lightweight-charts wrappers themed to tokens
export function candleChart(el), lineChart(el,{color}), fanChart(el, fanData), depthBars(...);
// lib/splitter.js — draggable resize + collapse
export function hSplit(container, {sizes, min, collapsible}), vSplit(...), makeResizable(...);
```
Pop-out pages live in `dashboard/pages/*.html` and import `../lib/*.js`. The shell links to them
with `window.open('pages/trades.html', ...)` so agents/trades/results open in **new browser tabs**.

## 8 · worker.py addition (features agent, optional real feature)

`--role redteam --count N`: a `RedTeamAgent` (new `agents/red_team.py`) — a **real heuristic
predatory taker** that watches OFI/price momentum and aggressively fades extended moves and
sweeps thin books (models adverse selection against a user's strategy). Honest scope: heuristic,
not RL-trained (RL red-team is roadmap). Included in `--role all`? NO — opt-in only, launched from
the UI "Lab". It connects like any worker (SUB data + DEALER orders, agent_id `REDTEAM_i`).
