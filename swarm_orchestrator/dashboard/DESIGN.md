# Chronos Dashboard — Design & Build Spec (v2)

A modern financial-terminal SPA. Goal: it must look like a **premium trading platform**
(Bloomberg / TradingView caliber), not an AI demo. Everything below is binding. Colors,
spacing, radii, typography come **only** from `design-tokens.css` — never hardcode.

## App shell (index.html + app.js + app-shell.css — FE-CORE agent)

```
┌────────────────────────────────────────────────────────────────────────────┐
│ TOPBAR (52px): ◆ CHRONOS · <mode chip> · DAY n · hh:mm · ₹price(±color) ·    │
│                <regime chip> · conn dot ·  [START][PAUSE][RESET]  · ⤢pop     │
├────┬──────────────────────────────────────────────────────────────┬─────────┤
│ L  │                     CENTER  (active view)                     │  R      │
│ H  │                                                              │  H      │
│ S  │   views mount here; some views host resizable sub-panels     │  S      │
│ n  │                                                              │ actions │
│ a  │                                                              │         │
│ v  │                                                              │         │
└────┴──────────────────────────────────────────────────────────────┴─────────┘
      ↕ col-splitter                                          ↕ col-splitter
```

- **LHS nav** (collapsible → icon rail): vertical list of views, grouped:
  MARKET · *TRADING*: Strategies, Results · *LAB*: Monte-Carlo, Time Machine, Copilot,
  Narration, Lab · *DATA*: Runs. Active item highlighted (left accent bar + glow). A ⟨⟨ collapse
  toggle at the bottom persists state to localStorage.
- **RHS panel** (collapsible → hidden with a floating ⟩ re-open tab): top has **two big primary
  action buttons** — **① Ingest News** (📰) and **② Upload Script** (⤴), each opens a modal
  (see below). Below: live context sections — "Running Strategies" (mini list w/ status dots),
  "Event Feed" (last 8 news/mode_change/liquidation/session events, color-coded), and a compact
  "Market Pulse" narration line with a ↻ refresh. Collapsible, resizable.
- **Both side panels resizable** via `lib/splitter.js` col-splitters; widths persist to localStorage.
  Min widths enforced; double-click a splitter to collapse/restore.
- **Topbar controls** drive `api.sim.start/stop/resume`; a ⤢ button pops the current view's data
  out (Trades/Agents/Results) into a new browser tab.

## Views (each is `dashboard/views/<id>.js`, default export)

```js
export default {
  id:'market', title:'Market', icon:'candles', group:'',   // group label for nav sections
  async mount(el, ctx){ /* build DOM into el; subscribe via ctx.store */ },
  unmount(){ /* remove listeners, dispose charts */ }
}
```
`ctx = { store, api, fmt, ui, charts, splitter }` (the lib singletons). app.js imports this
FIXED list and builds the nav from it:
```js
import market from './views/market.js';        // FE-CORE
import strategies from './views/strategies.js'; // FE-CORE
import results from './views/results.js';       // FE-CORE
import montecarlo from './views/montecarlo.js'; // FE-FEAT
import timemachine from './views/timemachine.js';//FE-FEAT
import copilot from './views/copilot.js';       // FE-FEAT
import narration from './views/narration.js';   // FE-FEAT
import runs from './views/runs.js';             // FE-FEAT
import lab from './views/lab.js';               // FE-FEAT
```
Routing is hash-based (`#/market`, `#/results`, …) so views are linkable and back/forward work.

### MARKET (FE-CORE) — the hero view
Resizable grid: large **candle+volume chart** (real `tick.bar`, OHLCV crosshair readout),
**order book** (10 levels/side, depth bars scaled to in-view max, prices up/down-colored),
**trade tape** (per-`trade` msg, colored by REAL `aggressor`), **leaderboard** (from
`tick.leaderboard`, ₹ pnl sign-colored). Tape header has ⤢ → `pages/trades.html`; leaderboard
header ⤢ → `pages/agents.html`; clicking an agent row → `pages/agent.html?id=<id>`. Empty state
before START.

### STRATEGIES (FE-CORE) — Pillar-2 front door + script library
Two columns (resizable): left = **Monaco** editor (python, vs-dark) with the SDK template
prefilled; a Name field; **Save to Library**, **Save & Run**, **Run (unsaved)** buttons.
Right = **Script Library** (`api.scripts.list()`): each saved script → Load / Run / Rename /
Delete; plus **Running Strategies** table (`api.strategy.list()` poll 3s) with Stop + View Logs
(logs drawer). Backend chip shows docker/subprocess with the honest tooltip. Saving writes a
real file (user can refer to it again). Running a script registers it with Results.

### RESULTS (FE-CORE) — the reports page (also poppable)
Master-detail: left list of strategies (`api.results.list()`: name, agent_id, status dot,
total trades, total P&L sign-colored). Select → detail: a **summary stat row** (total trades,
realized P&L, win rate, # days, max drawdown), an **equity curve** (line, accent), a **per-day
outcomes table** (day, run, trades, gross P&L, start→end equity, max DD, win rate), and a
**trade log table** (ts, side, price, qty, cash Δ, counterparty). Header buttons: **Download
Trades CSV**, **Download Outcomes CSV**, **Download PDF Report** (hit `api.results.csvUrl/…/pdfUrl`
in a new tab — real files stream from the bridge). Multiple strategies organized clearly; a day
selector filters the trade log. Empty state if no strategies have run.

### MONTE-CARLO (FE-FEAT) — counterfactual fan
Config form (symbol, start price, # paths, horizon, optional shock: minute/side/qty). **Run**
→ `api.montecarlo(cfg)`. Render a **fan chart**: p5–p95 band (shaded), p25–p75 (darker band),
p50 median line (accent), plus a few thin sample paths (muted). Below: a **terminal-price
distribution** (histogram/bars from the returned terminal percentiles) and a stat row
(median, P5, P95, mean, min, max). Explain copy: "1,000 futures your order could have caused —
a distribution, not a point estimate." Use `ctx.charts.fanChart`.

### TIME MACHINE (FE-FEAT) — crisis replay
Cards for each scenario (`api.timemachine.scenarios()`): name, description, bars, a tag chip
(crash/squeeze/shock). **Replay This Crisis** → `api.timemachine.start(id, seed)` → switches the
topbar to REPLAY mode; an info banner explains the latch ("the moment your strategy trades, the
market diverges from history — go to Strategies to intervene"). Show live progress via `store`.

### COPILOT (FE-FEAT) — NL → strategy
A prompt textarea ("describe your strategy in plain English") + examples chips. **Generate**
→ `api.copilot(description)` → shows returned code in a **Monaco** viewer + the explanation +
a `source` chip (gemini/template). Buttons: **Save to Library**, **Open in Strategies** (hands
the code to the Strategies view via `sessionStorage` handoff key `chronos.copilot.code`).

### NARRATION (FE-FEAT) — explainable market
A live feed: **Explain Now** button → `api.narrate()` → appends a timestamped narration card
(source chip). Auto-refresh toggle (every 15s while on). Reads the same live tick context.

### RUNS (FE-FEAT) — persistence explorer
`api.runs()` table (run_id, started_at, symbol, mode, seed). Select → `api.equity(runId)` (no
agent → agent picker, prefer STRAT_*) → equity line chart + stat row (start/end/ΔPnL/maxDD/points).

### LAB (FE-FEAT) — experimental
**Adversarial Red-Team**: a launcher explaining the heuristic predatory agent + a "Launch
Red-Team (N agents)" control that shells `worker.py --role redteam` is a BACKEND concern; the UI
POSTs `api.strategy`? No — provide an informational panel + a `POST /api/lab/redteam` if the
features backend exposes it, else show it as an available `worker.py --role redteam` command with
a copy button (honest). **Cross-venue fragmentation**: an honest "Roadmap" card (not faked).

## Pop-out standalone pages (FE-FEAT) — open in NEW browser tabs
Plain HTML pages under `dashboard/pages/` that import `../lib/*.js`, each with the topbar-lite
(logo + conn dot) and their own WS connection via `lib/store.js`:
- `pages/trades.html` — full-height live tape: filter by side/agent, running count, **Download
  visible as CSV** (client-side blob), colored by real aggressor.
- `pages/agents.html` — all agents grid from `tick.leaderboard`: id, type, P&L, pos, cash; click
  → `agent.html?id=`; sortable columns.
- `pages/agent.html?id=<id>` — single agent detail: header (type, P&L, pos, cash), its equity
  curve (via `api.runs` + `api.equity(run,id)` newest run), and its recent trades filtered from
  the tape.
- `pages/results.html` — the Results view as a standalone full page (may import the same render
  helpers or re-fetch via `api.results`). Download buttons stream from the bridge.

## Interaction & polish (both agents)
- Micro-transitions on hover/active (tokens `--dur-1/2`), never janky. Smooth panel collapse.
- Skeleton loaders while fetching; spinners on buttons mid-request; toasts on success/failure.
- Everything keyboard-reachable; real `<button>`; aria-labels; `prefers-reduced-motion` honored.
- Numbers tabular-nums + mono. ₹ via `fmt.inr`. Times via `fmt.time` (UTC hh:mm from unix).
- NO random values, NO lorem, NO placeholder data — every pixel driven by real API/WS data.
- Charts: lightweight-charts@4.2.0 (unpkg), Monaco@0.52.0 (jsdelivr AMD). One value axis per
  chart; recessive grid; semantic up/down only for polarity; categorical palette for multi-series.

## File ownership
- **FE-CORE**: `index.html`, `app-shell.css`, `app.js`, `lib/{store,api,format,ui,charts,splitter}.js`,
  `views/{market,strategies,results}.js`.
- **FE-FEAT**: `views/{montecarlo,timemachine,copilot,narration,runs,lab}.js`,
  `pages/{trades,agents,agent,results}.html` (+ any small per-page JS under `pages/`).
- Shared, do not edit: `design-tokens.css` (integrator-owned).
