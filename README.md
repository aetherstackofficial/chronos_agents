# Project Chronos — Reactive Market Digital Twin

Chronos is a **counterfactual, impact-aware market simulator**. Instead of replaying a frozen
historical tape and assuming your trades had no effect (the "price-taker fallacy" every classic
backtester makes), Chronos runs a **living limit-order-book** populated by AI and rule-based
agents. When your strategy submits an order, it matches against real resting liquidity, **moves
the price**, and the agent population re-prices on the next tick. The backtest stops being *"what
happened"* and becomes *"what would have happened, with you in it."*

> **Phase-2 status — honest snapshot.** This is a **single-node** system: four cooperating
> processes on one machine, talking over ZeroMQ. It is not multi-datacenter "distributed." What
> *is* true and working: the price is now **produced only by real trades in the order book** (no
> random walk), market impact is **emergent from real depth consumption** (no magic constants),
> the AI market-makers **quote from their trained models** (not random spreads), and a user can
> **submit a strategy that trades as a first-class participant** (Pillar 2) against a **replay of
> real history that diverges the moment they trade** (Pillar 1).

## Architecture — a two-plane exchange model

Chronos mirrors how a real exchange separates its network traffic:

```
  ORDER-ENTRY PLANE (reliable, TCP)          MARKET-DATA PLANE (push, one-to-many)
  DEALER ──▶ engine ROUTER :5555             engine PUB :5556 ──▶ SUB subscribers
  (agents, strategies, bridge, oracle,       (every book delta, trade print, and
   FIX gateway — per-session identity,        event — sequence-numbered, like an
   acks, sequence numbers — like FIX/TCP)     ITCH multicast feed, but over TCP)
```

The oracle adds a **third bind point** — a *service* socket rather than a plane. The bridge asks
it to score a headline; it answers, and when the news is material it submits its own orders to the
engine like any other participant:

```
  bridge DEALER "BRIDGE_ORACLE" ──▶ oracle ROUTER :5557
                                      └─▶ DEALER "GEMINI_ORACLE" ──▶ engine ROUTER :5555
```

| Port | Socket | Bound by | Carries |
|------|--------|----------|---------|
| `5555` | `ROUTER` | engine | order entry + control — per-session identity, one ack per request |
| `5556` | `PUB` | engine | market data — `tick` / `trade` / `event`, sequence-numbered |
| `5557` | `ROUTER` | oracle | news-scoring requests from the bridge |
| `8000` | HTTP/WS | bridge | dashboard, REST API, and the WebSocket fan-out of `5556` |

All four are overridable via `ZMQ_ORDER_PORT` / `ZMQ_DATA_PORT` / `ORACLE_PORT` / `BRIDGE_PORT`
(see [`.env.example`](swarm_orchestrator/.env.example)).

- **Engine** (`engine/`) — a single-threaded, deterministic, **sequenced** matching core
  (price-time priority, real depth, SQLite persistence). Determinism = replayable runs.
- **Swarm** (`worker.py`, `core/`, `agents/`) — 15 PPO market-makers + 4 PPO whales + 80
  heuristic retail, **push-driven** (subscribe to data, submit orders concurrently over a pooled
  gateway — no more one-at-a-time serial loop).
- **Runner** (`runner/`) — Pillar 2: a strategy **SDK** (`on_tick(state) -> orders`), a
  **sandbox** (Docker / resource-guarded subprocess), and a **FIX 4.4 gateway**.
- **Bridge** (`bridge.py`) — FastAPI + native WebSockets (ASGI, async — no thread-unsafe
  sockets), serves the dashboard and the REST/WS API.
- **Oracle** (`oracle.py`) — an **async** Gemini macro-news engine with a **real** local fallback
  scorer (keyword-polarity matrix), sizing shocks from actual book depth (not magic constants).
  Both a server (`ROUTER :5557`) and a client (`DEALER` into the engine) — the bridge never scores
  news itself and never publishes on the data plane.
- **Dashboard** (`dashboard/`) — a professional trading terminal (MARKET / STRATEGY / RUNS tabs).

The full interface contract is **[`swarm_orchestrator/PROTOCOL.md`](swarm_orchestrator/PROTOCOL.md)** —
every message schema, port, and invariant. Read it before changing any interface.

## Tech stack

Python 3.11+ · ZeroMQ (ROUTER/DEALER + PUB/SUB) · PyTorch + Stable-Baselines3 (PPO) ·
Gymnasium · FastAPI + Uvicorn · SQLite · Google GenAI (Gemini) · simplefix · lightweight-charts
+ Monaco.

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (source venv/bin/activate on Unix)
cd swarm_orchestrator
pip install -r requirements.txt
cp .env.example .env             # then edit .env — add your GEMINI_API_KEY
```

> **Security:** never commit `.env`. It is gitignored. A previously-committed key must be treated
> as compromised — rotate it and purge it from git history (`git filter-repo` / BFG).

## Running

Each component is its own process. From `swarm_orchestrator/`, in separate terminals:

```bash
# 1 · The matching engine (order plane :5555, data plane :5556)
python -m engine.server

# 2 · The AI swarm  (15 MM + 4 whales + 80 retail)
python main.py                    #  == python worker.py --role all
#     or run a single cohort:  python worker.py --role mm --count 15

# 3 · The web bridge + dashboard   ->  http://localhost:8000
python bridge.py

# 4 · The Gemini macro-oracle
python oracle.py

# optional · institutional FIX gateway (:9878)
python -m runner.fix_gateway
```

Open **http://localhost:8000**, set a symbol & price, and **START SIMULATION**. Inject a news
headline to watch the oracle gap the book. Load a historical CSV under **REPLAY** to run
Pillar-1 mode; submit a strategy under the **STRATEGY** tab (Pillar 2).

## Testing

```bash
cd swarm_orchestrator
python -m pytest -q                      # unit + integration suite
python scripts/smoke_e2e.py              # end-to-end: boots the engine, drives both planes
```

The suite guards the Phase-2 promises: order-book price-time priority, **ledger conservation**,
**price is book-driven only**, **deterministic runs under a fixed seed**, the **replay→reactive
latch**, and the oracle's real fallback matrix.

## What changed from Phase 1

| Phase-1 (demo) | Phase-2 (this build) |
|---|---|
| Price = `random.gauss` on every fetch | Price = **last real trade** in the book |
| Impact = one magic constant, book wiped on >50k | **Emergent** from real depth consumption |
| MM quoted **random** spreads | MM quotes from its **trained PPO** 3-D action |
| Retail traded **random** 1–20 shares | Retail sizes by heuristic **conviction** |
| 99 agents **pulled** state over one shared socket | Agents **subscribe** to a pushed feed, submit concurrently |
| Flask-SocketIO, thread-unsafe socket | FastAPI + async WebSockets, single event loop |
| No persistence | SQLite runs / ticks / trades / equity |
| No way for a user to trade | Strategy **SDK + sandbox + FIX gateway** (Pillar 2) |
| — | **History-repeater** replay that latches to reactive (Pillar 1) |
| Broken `run_*.py`, dead code, leaked key | Parameterized `worker.py`, cleaned, key untracked |

## Roadmap (not yet in this build)

- **Model retraining** (real RSI feature, best-checkpoint selection, multi-asset) — deferred.
- **New product features** (Monte-Carlo counterfactual fan, adversarial red-team agent, LLM
  strategy copilot, crisis "time machine") — deferred.
- **Calibration study** (simulated vs realised impact) and true multi-node distribution.

See `Chronos_Phase2_Master_Plan.html` for the complete plan.

## Docker & the two front-ends

The Docker track and the Vite/React `frontend/` were built in parallel against the old Phase-1
`true_engine.py`, before this Phase-2 rewrite landed, and were merged in once both branches were
reconciled. What that merge settled:

- **Backend is this build's.** Push-based PUB/SUB two-plane transport, `DEALER` + `asyncio.Lock`
  control sockets, and no synthesized OHLC — price and bars come only from real fills
  (PROTOCOL §8.1). The parallel branch's REQ/REP polling bridge and its `random.uniform` candle
  jitter were both dropped by agreement.
- **Containers point at the real services.** `Dockerfile.engine` ran `python true_engine.py`,
  which no longer exists — repointed at `python -m engine.server`. The other four (`bridge`,
  `oracle`, `agents`, `frontend`) already matched this build's entry points.
- **`frontend/` is not wired to PROTOCOL.md yet.** It still opens `ws://localhost:8000/ws` and
  sends `{"action": "start_sim", ...}` — the old bridge's contract. This build expects REST for
  control (`POST /api/sim/start|stop`, `/api/news`, …) and `{"topic","data"}` envelopes over `/ws`
  for market data. Rewiring it is owned by the author of the React app.

**Direction:** React becomes the primary shell over time, with the V2 feature views (copilot,
Monte-Carlo, time machine, script library, results/PDF) ported across incrementally. Until that
port is done, `swarm_orchestrator/dashboard/` remains the full-featured UI and the bridge keeps
serving it — the bridge container copies `swarm_orchestrator/`, so both UIs ship side by side.

```bash
docker-compose up --build
```
brings up `engine`, `oracle`, `swarm`, `bridge` (REST/WS API + the full dashboard on `:8000`), and
`frontend` (the React shell on `:3000`, pending the protocol rewire).
