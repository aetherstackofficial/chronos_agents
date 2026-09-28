"""Chronos matching engine core (PROTOCOL §4, §5, §8, §12.1).

Transport-free and deterministic: ``handle_message`` and ``physics_tick``
return ``(ack, publications)`` / ``publications`` so the class is fully
unit-testable without sockets. All randomness flows through ``self.rng``
(seeded ``random.Random``) — no module-level ``random.*`` anywhere in the
engine package (§8.7). ``last_price`` changes ONLY when a real fill occurs
in the book (§8.1).
"""
from __future__ import annotations

import json
import logging
import math
import random
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

from .book import OrderBook, TICK_SIZE, round_to_tick
from .persistence import Persistence
from .replay import HistoricalReplayer

logger = logging.getLogger("engine.core")

Publication = Tuple[str, dict]


class MatchingEngine:
    """Single-threaded, sequenced matching core for one symbol."""

    # -- venue constants (PROTOCOL §3) ---------------------------------- #
    SESSION_MINUTES = 375
    SIM_EPOCH = 1767258900                    # 2026-01-01 09:15 IST-as-UTC
    OVERNIGHT_SECONDS = int(17.5 * 3600)
    SECONDS_PER_MINUTE = 60

    # -- policy constants (§8, all named per quality bar) ---------------- #
    DEFAULT_TIF_TICKS = 60                    # §8.8 default resting TTL
    TRADES_KEPT = 50                          # FETCH_STATE compat window
    LEADERBOARD_SIZE = 15
    DEPTH_LEVELS = 10
    RSI_PERIOD = 14
    VOLUME_MA_WINDOW = 20
    LIQUIDATION_THRESHOLD = 1000.0            # §8.6 retail equity floor
    # U-shape intraday activity curve: u = U_BASE + U_AMPL * t^2.
    U_BASE = 0.2
    U_AMPL = 2.3
    U_MIDPOINT = 187.5
    # Regime physics (§8.3).
    REGIME_CHOICES = ("BULL", "BEAR", "RANGING")
    REGIME_WEIGHTS = (0.4, 0.4, 0.2)
    REGIME_MIN_TICKS = 60
    REGIME_MAX_TICKS = 180
    INITIAL_REGIME_TICKS = 60
    GHOST_TREND_PROB = 0.5
    GHOST_TREND_MIN_QTY = 300
    GHOST_TREND_MAX_QTY = 1000
    REVERSAL_PROB = 0.3
    REVERSAL_MIN_QTY = 2500
    REVERSAL_MAX_QTY = 5000
    REVERSAL_MIN_TICKS = 30                   # regime length after a bounce
    REVERSAL_MAX_TICKS = 90
    ROUND_LEVEL = 5.0                         # ₹5 psychological levels
    LEVEL_BAND = 0.20
    # Designated liquidity (§8.4).
    LP_AGENT = "SIM_LP"
    LP_LEVELS = 8
    LP_BASE_SIZE = 400
    LP_VOLUME_FACTOR = 0.5                    # size += factor * volume_ma
    LP_TIF_TICKS = 3
    LP_MAX_TICK_STEP = 3                      # levels spaced 1-3 ticks
    LP_SIZE_JITTER = 0.2                      # +/-20% via engine rng
    # Ledger seeding (§ ledger rules).
    CASH_INSTITUTION = 5_000_000.0
    CASH_MARKET_MAKER = 250_000.0
    CASH_RETAIL = 50_000.0
    CASH_STRATEGY = 1_000_000.0
    CASH_ORACLE = 1e9
    CASH_INTERNAL = 1e12                      # SIM_LP / GHOST flow agents
    # Regime inference after the replay latch (§9).
    DRIFT_LOOKBACK = 15
    DRIFT_THRESHOLD = 0.002

    def __init__(self, db: Optional[Persistence] = None) -> None:
        self.db = db
        self.rng: random.Random = random.Random(42)
        self.run_id: Optional[str] = None
        self.replayer: Optional[HistoricalReplayer] = None
        self._reset_state("TCS", "TECH", 190.0, 42, "LIVE")

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def _reset_state(self, symbol: str, sector: str, price: float,
                     seed: int, mode: str) -> None:
        """Reset every piece of run state; called by init_sim/start_replay."""
        self.symbol = symbol
        self.sector = sector
        self.seed = seed
        self.mode = mode
        self.paused = False
        self.rng = random.Random(seed)
        self.book = OrderBook()
        self.last_price = round_to_tick(price)
        self.seq = 0
        self._order_counter = 0
        self.tick_count = 0
        self.market_minute = 0
        self.day_count = 1
        self.sim_unix_time = self.SIM_EPOCH
        self.regime = "RANGING"
        self.regime_ticks_left = self.INITIAL_REGIME_TICKS
        self.u_shape = self.U_BASE + self.U_AMPL  # value at minute 0
        self.agent_ledger: Dict[str, dict] = {}
        # order_id -> {"agent_id", "expiry_tick": int | None}; None = day order.
        self._resting: Dict[int, dict] = {}
        self.trades: Deque[dict] = deque(maxlen=self.TRADES_KEPT)
        self._pending_trades: List[dict] = []
        self.current_bar = self._new_bar()
        self._last_bar: Optional[dict] = None
        self._closes: Deque[float] = deque(maxlen=64)
        self._vol_window: Deque[int] = deque(maxlen=self.VOLUME_MA_WINDOW)
        self._min_buy_vol = 0
        self._min_sell_vol = 0
        self._last_min_buy = 0
        self._last_min_sell = 0
        self._day_pv = 0.0
        self._day_vol = 0
        # Incremental Wilder RSI state.
        self._rsi_prev_close: Optional[float] = None
        self._rsi_avg_gain: Optional[float] = None
        self._rsi_avg_loss: Optional[float] = None
        self._rsi_seed: List[Tuple[float, float]] = []
        self.replayer = None

    def _make_run_id(self) -> str:
        return "run_{}_{:04x}".format(
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            self.rng.randrange(16 ** 4))

    def init_sim(self, symbol: str, sector: str, price: float,
                 seed: int = 42) -> str:
        """Reset everything and start a fresh LIVE run. Returns run_id."""
        self._reset_state(symbol, sector, float(price), seed, "LIVE")
        self.run_id = self._make_run_id()
        if self.db is not None:
            self.db.start_run(
                self.run_id, datetime.now().isoformat(), symbol, sector,
                "LIVE", seed,
                json.dumps({"symbol": symbol, "sector": sector,
                            "price": float(price), "seed": seed}))
        logger.info("init_sim %s %s @ %.2f seed=%d run_id=%s",
                    symbol, sector, self.last_price, seed, self.run_id)
        return self.run_id

    def start_replay(self, csv_path: str, symbol: Optional[str],
                     sector: str, seed: int = 42) -> Tuple[str, int]:
        """Load a 1-minute bar CSV and enter REPLAY mode (PROTOCOL §9)."""
        bars = HistoricalReplayer.load_csv(csv_path)
        sym = symbol or self.symbol
        self._reset_state(sym, sector, bars[0]["o"], seed, "REPLAY")
        self.run_id = self._make_run_id()
        self.replayer = HistoricalReplayer(self, bars)
        if self.db is not None:
            self.db.start_run(
                self.run_id, datetime.now().isoformat(), sym, sector,
                "REPLAY", seed,
                json.dumps({"csv_path": csv_path, "symbol": sym,
                            "sector": sector, "seed": seed}))
        logger.info("start_replay %s (%d bars) run_id=%s",
                    csv_path, len(bars), self.run_id)
        return self.run_id, len(bars)

    # ------------------------------------------------------------------ #
    # sequencing / ledger
    # ------------------------------------------------------------------ #
    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def _next_order_id(self) -> int:
        self._order_counter += 1
        return self._order_counter

    @classmethod
    def _classify_agent(cls, agent_id: str) -> Tuple[str, float, bool]:
        """(type, start_cash, internal). Internal agents are excluded from
        the leaderboard and snapshots (SIM_LP designated liquidity, ghost
        regime flow)."""
        if agent_id.startswith(("STRAT_", "USER_")):
            return "Strategy", cls.CASH_STRATEGY, False
        if "GEMINI" in agent_id or "ORACLE" in agent_id:
            return "Macro Oracle", cls.CASH_ORACLE, False
        if "SIM_LP" in agent_id:
            return "Designated LP", cls.CASH_INTERNAL, True
        if "GHOST" in agent_id:
            return "Sim Flow", cls.CASH_INTERNAL, True
        if "WHALE" in agent_id:
            return "Institution", cls.CASH_INSTITUTION, False
        if "MM" in agent_id:
            return "Market Maker", cls.CASH_MARKET_MAKER, False
        return "Retail", cls.CASH_RETAIL, False

    def _ensure_agent(self, agent_id: str) -> dict:
        led = self.agent_ledger.get(agent_id)
        if led is None:
            a_type, cash, internal = self._classify_agent(agent_id)
            led = {"id": agent_id, "type": a_type, "cash": cash, "pos": 0,
                   "start_cash": cash, "internal": internal}
            self.agent_ledger[agent_id] = led
        return led

    # ------------------------------------------------------------------ #
    # fills / trades
    # ------------------------------------------------------------------ #
    def _apply_fill(self, fill: dict, taker_side: str,
                    pubs: List[Publication]) -> None:
        """Settle one fill: ledger transfer (conserving cash and position
        exactly), last_price update (§8.1), bar/VWAP/flow accounting, trade
        publication, and the replay latch (§9)."""
        price = fill["price"]
        qty = fill["qty"]
        taker, maker = fill["taker_id"], fill["maker_id"]
        buyer, seller = (taker, maker) if taker_side == "BUY" else (maker, taker)
        buy_led = self._ensure_agent(buyer)
        sell_led = self._ensure_agent(seller)
        notional = price * qty
        buy_led["cash"] -= notional
        buy_led["pos"] += qty
        sell_led["cash"] += notional
        sell_led["pos"] -= qty

        self.last_price = price  # the ONLY price authority

        bar = self.current_bar
        bar["h"] = max(bar["h"], price)
        bar["l"] = min(bar["l"], price)
        bar["c"] = price
        bar["v"] += qty

        self._day_pv += notional
        self._day_vol += qty
        if taker_side == "BUY":
            self._min_buy_vol += qty
        else:
            self._min_sell_vol += qty

        trade = {"seq": self._next_seq(), "price": round(price, 2),
                 "qty": qty, "buyer": buyer, "seller": seller,
                 "aggressor": taker_side, "unix_time": self.sim_unix_time,
                 "market_minute": self.market_minute}
        self.trades.append(trade)
        self._pending_trades.append(trade)
        pubs.append(("trade", trade))

        if self.mode == "REPLAY":
            for aid in (buyer, seller):
                if aid.startswith(("STRAT_", "USER_")):
                    self._latch_reactive(aid, pubs)
                    break

    def _latch_reactive(self, trigger_agent: str,
                        pubs: List[Publication]) -> None:
        """Flip REPLAY -> REACTIVE permanently on the first swarm/user fill.

        Book, ledgers and RSI window carry over untouched; the regime is
        inferred from recent close drift so LIVE physics resumes seamlessly.
        """
        self.mode = "REACTIVE"
        self.regime = self._infer_regime()
        self.regime_ticks_left = self.INITIAL_REGIME_TICKS
        pubs.append(("event", {"seq": self._next_seq(), "kind": "mode_change",
                               "mode": "REACTIVE",
                               "trigger_agent": trigger_agent}))
        logger.info("LATCH: mode=REACTIVE trigger=%s regime=%s",
                    trigger_agent, self.regime)

    def _infer_regime(self) -> str:
        closes = list(self._closes)
        if len(closes) >= self.DRIFT_LOOKBACK and closes[-self.DRIFT_LOOKBACK]:
            drift = ((closes[-1] - closes[-self.DRIFT_LOOKBACK])
                     / closes[-self.DRIFT_LOOKBACK])
            if drift > self.DRIFT_THRESHOLD:
                return "BULL"
            if drift < -self.DRIFT_THRESHOLD:
                return "BEAR"
        return "RANGING"

    # ------------------------------------------------------------------ #
    # order submission (shared by external ORDER messages and internal
    # ghost/LP/replay flow — one matching path, §8.3)
    # ------------------------------------------------------------------ #
    def _submit_order(self, agent_id: str, action: str, order_type: str,
                      qty: int, price: Optional[float], tif_ticks: int,
                      pubs: List[Publication],
                      client_order_id: Optional[str] = None) -> dict:
        led = self._ensure_agent(agent_id)
        order_id = self._next_order_id()
        resting_qty = 0
        if order_type == "LIMIT":
            fills = self.book.add_limit({
                "order_id": order_id, "agent_id": agent_id, "side": action,
                "price": price, "qty": qty, "ts_seq": order_id})
            executed = sum(f["qty"] for f in fills)
            resting_qty = qty - executed
            if resting_qty > 0:
                expiry = (None if tif_ticks == 0
                          else self.tick_count + tif_ticks)
                self._resting[order_id] = {"agent_id": agent_id,
                                           "expiry_tick": expiry}
        else:
            fills = self.book.match_market(agent_id, action, qty)
            executed = sum(f["qty"] for f in fills)

        for fill in fills:
            self._apply_fill(fill, action, pubs)

        notional = sum(f["price"] * f["qty"] for f in fills)
        avg_price = round(notional / executed, 4) if executed else 0.0
        if executed == qty:
            status = "FILLED"
        elif order_type == "LIMIT" and executed == 0:
            status = "RESTING"
        else:
            status = "PARTIAL"  # includes MARKET remainders (discarded)

        involved = {agent_id} | {f["maker_id"] for f in fills}
        self._check_liquidations(involved, pubs)

        ack = {"msg": "ACK", "status": status, "order_id": order_id,
               "executed_qty": executed, "average_price": avg_price,
               "resting_qty": resting_qty, "cash": round(led["cash"], 2),
               "pos": led["pos"], "seq": self._next_seq()}
        if client_order_id is not None:
            ack["client_order_id"] = client_order_id
        return ack

    def submit_flow_order(self, agent_id: str, action: str, order_type: str,
                          qty: int, price: Optional[float],
                          pubs: List[Publication],
                          tif_ticks: int = DEFAULT_TIF_TICKS) -> dict:
        """Engine-internal order entry (ghost flow, SIM_LP, replayer).

        Uses the exact same matching path as external orders so every fill
        hits the real book, ledger and trade tape; no ack is published.
        """
        return self._submit_order(agent_id, action, order_type, qty, price,
                                  tif_ticks, pubs)

    def flush_agent_orders(self, agent_id: str) -> int:
        """Cancel every resting order of an agent via the real cancel path."""
        cancelled = self.book.cancel_all(agent_id)
        for oid in [oid for oid, info in self._resting.items()
                    if info["agent_id"] == agent_id]:
            del self._resting[oid]
        return cancelled

    def _check_liquidations(self, agent_ids, pubs: List[Publication]) -> None:
        """Retail bankruptcy reset (§8.6): equity < threshold -> fresh 50k."""
        for aid in agent_ids:
            led = self.agent_ledger.get(aid)
            if led is None or led["type"] != "Retail":
                continue
            equity = led["cash"] + led["pos"] * self.last_price
            if equity < self.LIQUIDATION_THRESHOLD:
                self.flush_agent_orders(aid)
                led["cash"] = self.CASH_RETAIL
                led["pos"] = 0
                pubs.append(("event", {"seq": self._next_seq(),
                                       "kind": "liquidation",
                                       "agent_id": aid}))
                logger.info("liquidation: %s reset to fresh retail ledger", aid)

    # ------------------------------------------------------------------ #
    # message plane (PROTOCOL §4)
    # ------------------------------------------------------------------ #
    def handle_message(self, msg: dict) -> Tuple[dict, List[Publication]]:
        """Handle one order-entry message; returns (ack, publications)."""
        pubs: List[Publication] = []
        if not isinstance(msg, dict):
            return self._reject("malformed message"), pubs
        mtype = msg.get("msg")
        if mtype == "ORDER":
            return self._handle_order(msg, pubs)
        if mtype == "CANCEL":
            return self._handle_cancel(msg), pubs
        if mtype == "CANCEL_ALL":
            return self._handle_cancel_all(msg), pubs
        if mtype == "FETCH_STATE":
            payload = self._build_tick_payload(
                dict(self.current_bar), self.current_bar["v"])
            payload["msg"] = "ACK"
            payload["status"] = "STATE"
            payload["trades"] = list(self.trades)
            return payload, pubs
        if mtype == "PING":
            return {"msg": "ACK", "status": "PONG",
                    "seq": self._next_seq()}, pubs
        if mtype == "PAUSE":
            self.paused = True
            pubs.append(("event", {"seq": self._next_seq(), "kind": "session",
                                   "action": "paused",
                                   "day_count": self.day_count}))
            return {"msg": "ACK", "status": "PAUSED",
                    "seq": self._next_seq()}, pubs
        if mtype == "RESUME":
            self.paused = False
            pubs.append(("event", {"seq": self._next_seq(), "kind": "session",
                                   "action": "resumed",
                                   "day_count": self.day_count}))
            return {"msg": "ACK", "status": "RUNNING",
                    "seq": self._next_seq()}, pubs
        if mtype == "INIT_SIM":
            return self._handle_init_sim(msg), pubs
        if mtype == "REPLAY_START":
            return self._handle_replay_start(msg), pubs
        return self._reject(f"unknown msg type: {mtype!r}"), pubs

    def _reject(self, reason: str,
                client_order_id: Optional[str] = None) -> dict:
        ack = {"msg": "ACK", "status": "REJECTED", "order_id": 0,
               "executed_qty": 0, "average_price": 0.0, "resting_qty": 0,
               "reason": reason, "seq": self._next_seq()}
        if client_order_id is not None:
            ack["client_order_id"] = client_order_id
        return ack

    @staticmethod
    def _as_int_qty(value) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float) and value.is_integer() and value > 0:
            return int(value)
        return None

    @staticmethod
    def _as_price(value) -> Optional[float]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        price = float(value)
        return price if math.isfinite(price) and price > 0 else None

    def _handle_order(self, msg: dict,
                      pubs: List[Publication]) -> Tuple[dict, List[Publication]]:
        client_order_id = msg.get("client_order_id")
        agent_id = msg.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id:
            return self._reject("missing agent_id", client_order_id), pubs
        action = msg.get("action")
        if action not in ("BUY", "SELL"):
            return self._reject(f"bad action: {action!r}", client_order_id), pubs
        order_type = msg.get("type")
        if order_type not in ("MARKET", "LIMIT"):
            return self._reject(f"bad type: {order_type!r}", client_order_id), pubs
        qty = self._as_int_qty(msg.get("qty"))
        if qty is None:
            return self._reject("qty must be a positive integer",
                                client_order_id), pubs
        price: Optional[float] = None
        if order_type == "LIMIT":
            price = self._as_price(msg.get("price"))
            if price is None:
                return self._reject("LIMIT requires a positive price",
                                    client_order_id), pubs
        tif = msg.get("tif_ticks", self.DEFAULT_TIF_TICKS)
        if not isinstance(tif, int) or isinstance(tif, bool) or tif < 0:
            return self._reject("tif_ticks must be an int >= 0",
                                client_order_id), pubs

        meta = msg.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("news"), dict):
            news = meta["news"]
            pubs.append(("event", {
                "seq": self._next_seq(), "kind": "news",
                "headline": news.get("headline", ""),
                "score": news.get("score", 0.0),
                "sector": news.get("sector", self.sector),
                "reasoning": news.get("reasoning", ""),
                "source": news.get("source", "fallback")}))

        ack = self._submit_order(agent_id, action, order_type, qty, price,
                                 tif, pubs, client_order_id)
        return ack, pubs

    def _handle_cancel(self, msg: dict) -> dict:
        agent_id = msg.get("agent_id")
        order_id = msg.get("order_id")
        if not isinstance(agent_id, str) or not agent_id:
            return self._reject("missing agent_id")
        if not isinstance(order_id, int) or isinstance(order_id, bool):
            return self._reject("missing order_id")
        info = self._resting.get(order_id)
        if info is None or not self.book.has_order(order_id):
            self._resting.pop(order_id, None)
            return self._reject(f"unknown or filled order_id {order_id}")
        if info["agent_id"] != agent_id:
            return self._reject("order belongs to another agent")
        self.book.cancel(order_id)
        del self._resting[order_id]
        led = self._ensure_agent(agent_id)
        return {"msg": "ACK", "status": "CANCELLED", "order_id": order_id,
                "executed_qty": 0, "average_price": 0.0, "resting_qty": 0,
                "cash": round(led["cash"], 2), "pos": led["pos"],
                "seq": self._next_seq()}

    def _handle_cancel_all(self, msg: dict) -> dict:
        agent_id = msg.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id:
            return self._reject("missing agent_id")
        count = self.flush_agent_orders(agent_id)
        led = self._ensure_agent(agent_id)
        return {"msg": "ACK", "status": "CANCELLED", "order_id": 0,
                "executed_qty": 0, "average_price": 0.0, "resting_qty": 0,
                "cancelled": count, "cash": round(led["cash"], 2),
                "pos": led["pos"], "seq": self._next_seq()}

    def _handle_init_sim(self, msg: dict) -> dict:
        symbol = msg.get("symbol", "TCS")
        sector = msg.get("sector", "TECH")
        price = self._as_price(msg.get("price", 190.0))
        seed = msg.get("seed", 42)
        if not isinstance(symbol, str) or not symbol or price is None \
                or not isinstance(seed, int) or isinstance(seed, bool):
            return self._reject("INIT_SIM requires symbol, positive price, int seed")
        run_id = self.init_sim(symbol, sector, price, seed)
        return {"msg": "ACK", "status": "ENGINE_READY", "run_id": run_id,
                "seq": self._next_seq()}

    def _handle_replay_start(self, msg: dict) -> dict:
        csv_path = msg.get("csv_path")
        if not isinstance(csv_path, str) or not csv_path:
            return self._reject("REPLAY_START requires csv_path")
        seed = msg.get("seed", 42)
        if not isinstance(seed, int) or isinstance(seed, bool):
            return self._reject("seed must be an int")
        try:
            run_id, n_bars = self.start_replay(
                csv_path, msg.get("symbol"), msg.get("sector", self.sector),
                seed)
        except (OSError, ValueError) as exc:
            return self._reject(f"replay load failed: {exc}")
        return {"msg": "ACK", "status": "REPLAY_READY", "run_id": run_id,
                "bars": n_bars, "seq": self._next_seq()}

    # ------------------------------------------------------------------ #
    # physics tick (PROTOCOL §12.1)
    # ------------------------------------------------------------------ #
    def physics_tick(self) -> List[Publication]:
        """One physics tick = one sim-minute.

        Clock advance, regime flow (or replay drive), SIM_LP refresh, TTL
        expiry, bar close, liquidation sweep, persistence flush. Always ends
        with one ('tick', §5.1 payload). When paused, only the tick snapshot
        is emitted — no clock/flow advance.
        """
        if self.paused:
            bar = self._last_bar or dict(self.current_bar)
            return [("tick", self._build_tick_payload(bar, 0))]

        pubs: List[Publication] = []
        self.tick_count += 1
        self._advance_clock(pubs)
        self._update_u_shape()
        self._update_regime()

        was_replay = self.mode == "REPLAY"
        if was_replay and self.replayer is not None:
            self.replayer.step(pubs)
            if self.mode != "REPLAY":
                # Latch fired mid-step: restore resting LP depth so the
                # swarm inherits a quoted book with no gap.
                self._refresh_sim_lp(pubs)
        else:
            self._ghost_flow(pubs)
            self._refresh_sim_lp(pubs)

        self._expire_ttl()
        closed_bar = self._close_bar()
        self._check_liquidations(list(self.agent_ledger.keys()), pubs)

        payload = self._build_tick_payload(closed_bar, closed_bar["v"])
        self._persist(payload)
        pubs.append(("tick", payload))
        return pubs

    def _advance_clock(self, pubs: List[Publication]) -> None:
        self.market_minute += 1
        if self.market_minute > self.SESSION_MINUTES:
            pubs.append(("event", {"seq": self._next_seq(), "kind": "session",
                                   "action": "day_close",
                                   "day_count": self.day_count}))
            self._expire_day_orders()
            self.day_count += 1
            self.market_minute = 1
            self.sim_unix_time += self.OVERNIGHT_SECONDS
            self._day_pv = 0.0
            self._day_vol = 0
            pubs.append(("event", {"seq": self._next_seq(), "kind": "session",
                                   "action": "day_open",
                                   "day_count": self.day_count}))
            logger.info("session roll: day %d open", self.day_count)
        else:
            self.sim_unix_time += self.SECONDS_PER_MINUTE

    def _update_u_shape(self) -> None:
        t = (self.market_minute - self.U_MIDPOINT) / self.U_MIDPOINT
        self.u_shape = self.U_BASE + self.U_AMPL * (t * t)

    def _update_regime(self) -> None:
        self.regime_ticks_left -= 1
        if self.regime_ticks_left <= 0:
            self.regime = self.rng.choices(
                list(self.REGIME_CHOICES),
                weights=list(self.REGIME_WEIGHTS))[0]
            self.regime_ticks_left = self.rng.randint(
                self.REGIME_MIN_TICKS, self.REGIME_MAX_TICKS)
            logger.info("regime shift -> %s for %d ticks",
                        self.regime, self.regime_ticks_left)

    def _ghost_flow(self, pubs: List[Publication]) -> None:
        """Regime physics inject FLOW, not price (§8.3): real market orders
        through the normal matching path, sized by the U-shape curve."""
        if self.regime == "BULL" and self.rng.random() < self.GHOST_TREND_PROB:
            qty = int(self.rng.randint(self.GHOST_TREND_MIN_QTY,
                                       self.GHOST_TREND_MAX_QTY) * self.u_shape)
            if qty > 0:
                self.submit_flow_order("GHOST_TREND", "BUY", "MARKET",
                                       qty, None, pubs)
        elif self.regime == "BEAR" and self.rng.random() < self.GHOST_TREND_PROB:
            qty = int(self.rng.randint(self.GHOST_TREND_MIN_QTY,
                                       self.GHOST_TREND_MAX_QTY) * self.u_shape)
            if qty > 0:
                self.submit_flow_order("GHOST_TREND", "SELL", "MARKET",
                                       qty, None, pubs)

        # Level-bounce near round ₹5 levels: a real reversal order + regime flip.
        level = round(self.last_price / self.ROUND_LEVEL) * self.ROUND_LEVEL
        distance = self.last_price - level
        if (self.regime == "BEAR" and 0 < distance < self.LEVEL_BAND
                and self.rng.random() < self.REVERSAL_PROB):
            qty = int(self.rng.randint(self.REVERSAL_MIN_QTY,
                                       self.REVERSAL_MAX_QTY) * self.u_shape)
            if qty > 0:
                self.submit_flow_order("GHOST_REVERSAL", "BUY", "MARKET",
                                       qty, None, pubs)
            self.regime = "BULL"
            self.regime_ticks_left = self.rng.randint(
                self.REVERSAL_MIN_TICKS, self.REVERSAL_MAX_TICKS)
        elif (self.regime == "BULL" and -self.LEVEL_BAND < distance < 0
                and self.rng.random() < self.REVERSAL_PROB):
            qty = int(self.rng.randint(self.REVERSAL_MIN_QTY,
                                       self.REVERSAL_MAX_QTY) * self.u_shape)
            if qty > 0:
                self.submit_flow_order("GHOST_REVERSAL", "SELL", "MARKET",
                                       qty, None, pubs)
            self.regime = "BEAR"
            self.regime_ticks_left = self.rng.randint(
                self.REVERSAL_MIN_TICKS, self.REVERSAL_MAX_TICKS)

    def _refresh_sim_lp(self, pubs: List[Publication],
                        reference: Optional[float] = None) -> None:
        """Designated liquidity (§8.4): cancel/replace 8 limit levels per
        side around the reference price via the normal order path."""
        self.flush_agent_orders(self.LP_AGENT)
        ref = round_to_tick(reference if reference is not None
                            else self.last_price)
        base = self.LP_BASE_SIZE + self._volume_ma() * self.LP_VOLUME_FACTOR
        bid_offset = 0
        ask_offset = 0
        for _ in range(self.LP_LEVELS):
            bid_offset += self.rng.randint(1, self.LP_MAX_TICK_STEP)
            ask_offset += self.rng.randint(1, self.LP_MAX_TICK_STEP)
            bid_px = round_to_tick(ref - bid_offset * TICK_SIZE)
            ask_px = round_to_tick(ref + ask_offset * TICK_SIZE)
            bid_sz = max(1, int(base * self.rng.uniform(
                1 - self.LP_SIZE_JITTER, 1 + self.LP_SIZE_JITTER)))
            ask_sz = max(1, int(base * self.rng.uniform(
                1 - self.LP_SIZE_JITTER, 1 + self.LP_SIZE_JITTER)))
            if bid_px > 0:
                self.submit_flow_order(self.LP_AGENT, "BUY", "LIMIT",
                                       bid_sz, bid_px, pubs,
                                       tif_ticks=self.LP_TIF_TICKS)
            self.submit_flow_order(self.LP_AGENT, "SELL", "LIMIT",
                                   ask_sz, ask_px, pubs,
                                   tif_ticks=self.LP_TIF_TICKS)

    def _expire_ttl(self) -> None:
        """Expire resting orders past their tif via the real cancel path
        (§8.8); no ack is published for internal expiry."""
        expired = [oid for oid, info in self._resting.items()
                   if info["expiry_tick"] is not None
                   and info["expiry_tick"] <= self.tick_count]
        for oid in expired:
            self.book.cancel(oid)
            del self._resting[oid]
        # Prune registry entries whose orders were fully filled meanwhile.
        stale = [oid for oid in self._resting
                 if not self.book.has_order(oid)]
        for oid in stale:
            del self._resting[oid]

    def _expire_day_orders(self) -> None:
        """Cancel tif_ticks=0 (good-for-day) orders at the session close."""
        day_orders = [oid for oid, info in self._resting.items()
                      if info["expiry_tick"] is None]
        for oid in day_orders:
            self.book.cancel(oid)
            del self._resting[oid]

    # ------------------------------------------------------------------ #
    # bars & indicators (all from real trades/book only, §8.5)
    # ------------------------------------------------------------------ #
    def _new_bar(self) -> dict:
        p = self.last_price
        return {"minute": self.market_minute, "o": p, "h": p, "l": p,
                "c": p, "v": 0}

    def _close_bar(self) -> dict:
        """Close the current minute bar; a zero-trade minute yields a flat
        bar at last_price with v=0. Rolls RSI/volume/OFI windows."""
        bar = self.current_bar
        bar["minute"] = self.market_minute
        self._closes.append(bar["c"])
        self._rsi_on_close(bar["c"])
        self._vol_window.append(bar["v"])
        self._last_min_buy = self._min_buy_vol
        self._last_min_sell = self._min_sell_vol
        self._min_buy_vol = 0
        self._min_sell_vol = 0
        self._last_bar = bar
        self.current_bar = self._new_bar()
        return bar

    def _rsi_on_close(self, close: float) -> None:
        if self._rsi_prev_close is None:
            self._rsi_prev_close = close
            return
        change = close - self._rsi_prev_close
        self._rsi_prev_close = close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        if self._rsi_avg_gain is None:
            self._rsi_seed.append((gain, loss))
            if len(self._rsi_seed) == self.RSI_PERIOD:
                self._rsi_avg_gain = (sum(g for g, _ in self._rsi_seed)
                                      / self.RSI_PERIOD)
                self._rsi_avg_loss = (sum(l for _, l in self._rsi_seed)
                                      / self.RSI_PERIOD)
                self._rsi_seed = []
        else:
            n = self.RSI_PERIOD
            self._rsi_avg_gain = (self._rsi_avg_gain * (n - 1) + gain) / n
            self._rsi_avg_loss = (self._rsi_avg_loss * (n - 1) + loss) / n

    def _rsi(self) -> float:
        """14-period Wilder RSI on minute closes; 50.0 until 15 closes."""
        if self._rsi_avg_gain is None or self._rsi_avg_loss is None:
            return 50.0
        if self._rsi_avg_loss == 0:
            return 100.0
        rs = self._rsi_avg_gain / self._rsi_avg_loss
        return round(100.0 - 100.0 / (1.0 + rs), 2)

    def _volume_ma(self) -> float:
        """20-minute rolling mean of per-minute volume; never 0."""
        if self._vol_window:
            ma = sum(self._vol_window) / len(self._vol_window)
        else:
            ma = float(self.current_bar["v"])
        return ma if ma > 0 else 1.0

    def _vwap(self) -> float:
        """Cumulative day VWAP from real trades; last_price before any print."""
        if self._day_vol > 0:
            return self._day_pv / self._day_vol
        return self.last_price

    def _ofi(self) -> float:
        """Signed trade flow of the last completed minute blended 50/50 with
        top-10 book imbalance, clamped to [-1, 1]."""
        total = self._last_min_buy + self._last_min_sell
        trade_ofi = (self._last_min_buy - self._last_min_sell) / (total + 1)
        bids, asks = self.book.depth(self.DEPTH_LEVELS)
        bid_depth = sum(q for _, q in bids)
        ask_depth = sum(q for _, q in asks)
        book_imb = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1)
        return max(-1.0, min(1.0, 0.5 * trade_ofi + 0.5 * book_imb))

    def _volatility(self) -> float:
        if self._last_bar is None or not self._last_bar["c"]:
            return 0.0
        return (self._last_bar["h"] - self._last_bar["l"]) / self._last_bar["c"]

    # ------------------------------------------------------------------ #
    # snapshot payload (PROTOCOL §5.1) & persistence
    # ------------------------------------------------------------------ #
    def _build_tick_payload(self, bar: dict, step_volume: int) -> dict:
        best_bid = self.book.best_bid()
        best_ask = self.book.best_ask()
        bids, asks = self.book.depth(self.DEPTH_LEVELS)
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
            bid_sz = bids[0][1] if bids else 0
            ask_sz = asks[0][1] if asks else 0
            if bid_sz + ask_sz > 0:
                micro = ((best_bid * ask_sz + best_ask * bid_sz)
                         / (bid_sz + ask_sz))
            else:
                micro = mid
            spread = round(best_ask - best_bid, 2)
        else:
            # Empty side: mid/micro fall back to last_price for display.
            mid = self.last_price
            micro = self.last_price
            spread = 0.0
        return {
            "seq": self._next_seq(),
            "run_id": self.run_id,
            "mode": self.mode,
            "paused": self.paused,
            "unix_time": self.sim_unix_time,
            "day_count": self.day_count,
            "market_minute": self.market_minute,
            "symbol": self.symbol,
            "sector": self.sector,
            "last_price": round(self.last_price, 2),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": round(mid, 4),
            "micro_price": round(micro, 4),
            "spread": spread,
            "bar": bar,
            "vwap": round(self._vwap(), 4),
            "rsi": self._rsi(),
            "ofi": round(self._ofi(), 4),
            "volatility": round(self._volatility(), 6),
            "volume_ma": round(self._volume_ma(), 2),
            "step_volume": step_volume,
            "regime": self.regime,
            "lob_bids": bids,
            "lob_asks": asks,
            "leaderboard": self._leaderboard(),
        }

    def _leaderboard(self) -> List[dict]:
        rows = []
        for led in self.agent_ledger.values():
            if led["internal"]:
                continue
            equity = led["cash"] + led["pos"] * self.last_price
            rows.append({"id": led["id"], "type": led["type"],
                         "pnl": round(equity - led["start_cash"], 2),
                         "pos": led["pos"], "cash": round(led["cash"], 2)})
        rows.sort(key=lambda r: r["pnl"], reverse=True)
        return rows[:self.LEADERBOARD_SIZE]

    def _persist(self, payload: dict) -> None:
        pending, self._pending_trades = self._pending_trades, []
        if self.db is None or self.run_id is None:
            return
        self.db.write_tick(self.run_id, {
            "seq": payload["seq"], "unix_time": payload["unix_time"],
            "market_minute": payload["market_minute"],
            "day_count": payload["day_count"],
            "last_price": payload["last_price"],
            "best_bid": payload["best_bid"],
            "best_ask": payload["best_ask"], "vwap": payload["vwap"],
            "rsi": payload["rsi"], "ofi": payload["ofi"],
            "step_volume": payload["step_volume"],
            "regime": payload["regime"]})
        self.db.write_trades(self.run_id, pending)
        snapshots = []
        for led in self.agent_ledger.values():
            if led["internal"]:
                continue
            equity = led["cash"] + led["pos"] * self.last_price
            snapshots.append({"agent_id": led["id"],
                              "agent_type": led["type"],
                              "cash": round(led["cash"], 2),
                              "pos": led["pos"],
                              "equity": round(equity, 2)})
        self.db.write_snapshots(self.run_id, payload["seq"], snapshots)
