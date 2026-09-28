"""Session-scoped results store for user strategies (PROTOCOL_V2 section 2).

``ResultsStore`` is an in-memory record of every tracked strategy's trades and
per-day outcomes, organized strategy -> run -> day. It is fed live by the
bridge's market-data pump, which calls :meth:`ResultsStore.ingest` for EVERY
PUB message (topics ``tick`` / ``trade`` / ``event``, PROTOCOL.md section 5).

Data provenance (No-Dummy charter):
* Fills come from the engine's ``trade`` stream filtered to tracked agent ids —
  the authoritative print tape.
* Equity / mark-to-market P&L come from the engine ``tick`` leaderboard
  (authoritative ledger): ``equity = cash + pos * last_price``.
* Realized P&L is recomputed from the fill log with the average-cost method.

Everything here is session-scoped (lives and dies with the bridge process);
scripts themselves persist separately via ``runner.script_store``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("chronos.analytics")

_ALLOWED_STATUSES = ("idle", "running", "stopped", "error")


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _iso_from_unix(unix_time: object) -> str:
    """ISO-8601 UTC string for an engine ``unix_time`` (empty string if bad)."""
    try:
        return datetime.fromtimestamp(int(unix_time), tz=timezone.utc).isoformat()  # type: ignore[arg-type]
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


@dataclass
class _DayBucket:
    """One (run, day) of a strategy's activity: fills + equity path."""

    run_id: str
    day_count: int
    symbol: str
    trades: list[dict] = field(default_factory=list)
    equity_seq: list[tuple[int, float]] = field(default_factory=list)  # (seq, equity)
    start_equity: Optional[float] = None
    end_equity: Optional[float] = None
    closed: bool = False


@dataclass
class _StrategyRecord:
    """Everything tracked for one attached strategy."""

    strategy_id: str
    agent_id: str
    name: str
    script_id: Optional[str]
    started_at: str
    status: str = "idle"
    days: dict[tuple[str, int], _DayBucket] = field(default_factory=dict)
    latest: Optional[dict] = None  # last leaderboard snapshot {seq,day_count,equity,pnl,pos,cash}
    runs: list[str] = field(default_factory=list)  # insertion-ordered distinct run ids

    def touch_run(self, run_id: str) -> None:
        """Record that this strategy was active in ``run_id``."""
        if run_id not in self.runs:
            self.runs.append(run_id)


class ResultsStore:
    """Session-scoped record of strategy trades & outcomes (strategy -> run -> day).

    Thread-safety: the bridge feeds and reads this store on one asyncio loop,
    but every entry point still takes a re-entrant lock so occasional
    ``asyncio.to_thread`` access (e.g. report rendering) stays safe.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._strategies: dict[str, _StrategyRecord] = {}
        self._by_agent: dict[str, str] = {}  # agent_id -> strategy_id
        self._runs_meta: dict[str, dict] = {}  # run_id -> {symbol, mode, seed, first_seen}
        # Current market context, maintained from the tick stream.
        self._run_id: Optional[str] = None
        self._symbol: Optional[str] = None
        self._day_count: int = 1
        self._last_price: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Registration / lifecycle                                            #
    # ------------------------------------------------------------------ #
    def attach_strategy(
        self,
        strategy_id: str,
        agent_id: str,
        name: str,
        script_id: Optional[str],
    ) -> None:
        """Register a launched strategy so its market activity is tracked.

        Status starts as ``'idle'`` and flips to ``'running'`` the first time
        the agent is observed on the tape or leaderboard. Re-attaching an
        existing ``strategy_id`` refreshes its identity fields but keeps its
        accumulated history.
        """
        with self._lock:
            record = self._strategies.get(strategy_id)
            if record is None:
                record = _StrategyRecord(
                    strategy_id=strategy_id,
                    agent_id=agent_id,
                    name=name,
                    script_id=script_id,
                    started_at=_utc_now_iso(),
                )
                self._strategies[strategy_id] = record
                logger.info(
                    "tracking strategy %s (agent %s, name %r, script %s)",
                    strategy_id, agent_id, name, script_id,
                )
            else:
                record.agent_id = agent_id
                record.name = name
                record.script_id = script_id
                logger.info("re-attached strategy %s (agent %s)", strategy_id, agent_id)
            self._by_agent[agent_id] = strategy_id

    def mark_status(self, strategy_id: str, status: str) -> None:
        """Set a strategy's lifecycle status (``running`` / ``stopped`` / ...).

        Raises :class:`KeyError` for an unknown strategy.
        """
        with self._lock:
            record = self._strategies.get(strategy_id)
            if record is None:
                raise KeyError(f"unknown strategy_id: {strategy_id}")
            if status not in _ALLOWED_STATUSES:
                logger.warning("unusual strategy status %r for %s", status, strategy_id)
            record.status = status
            logger.info("strategy %s status -> %s", strategy_id, status)

    def on_run(self, run_id: str, symbol: str, mode: str, seed: int) -> None:
        """Record a new engine run's metadata and reset the day context."""
        with self._lock:
            if run_id not in self._runs_meta:
                self._runs_meta[run_id] = {
                    "symbol": symbol,
                    "mode": mode,
                    "seed": seed,
                    "first_seen": _utc_now_iso(),
                }
                logger.info("run registered: %s (%s, mode=%s)", run_id, symbol, mode)
            self._run_id = run_id
            self._symbol = symbol
            self._day_count = 1

    # ------------------------------------------------------------------ #
    # Live ingest (called by the bridge pump for EVERY message)           #
    # ------------------------------------------------------------------ #
    def ingest(self, topic: str, payload: dict) -> None:
        """Consume one market-data message. Never raises (hot pump path)."""
        try:
            if not isinstance(payload, dict):
                return
            with self._lock:
                if topic == "tick":
                    self._on_tick(payload)
                elif topic == "trade":
                    self._on_trade(payload)
                elif topic == "event":
                    self._on_event(payload)
        except Exception:  # noqa: BLE001 — the pump must never die on our account
            logger.exception("results ingest failed for topic %r", topic)

    def _on_tick(self, payload: dict) -> None:
        """Refresh market context and snapshot every tracked agent's ledger."""
        run_id = str(payload.get("run_id") or self._run_id or "unknown")
        if run_id != self._run_id:
            self._run_id = run_id
            self._runs_meta.setdefault(
                run_id,
                {
                    "symbol": payload.get("symbol"),
                    "mode": payload.get("mode"),
                    "seed": None,
                    "first_seen": _utc_now_iso(),
                },
            )
        symbol = payload.get("symbol")
        if symbol:
            self._symbol = str(symbol)
        try:
            self._day_count = int(payload.get("day_count") or self._day_count)
        except (TypeError, ValueError):
            pass
        last_price = payload.get("last_price")
        if isinstance(last_price, (int, float)):
            self._last_price = float(last_price)
        try:
            seq = int(payload.get("seq") or 0)
        except (TypeError, ValueError):
            seq = 0

        leaderboard = payload.get("leaderboard")
        if not isinstance(leaderboard, list):
            return
        price = self._last_price if self._last_price is not None else 0.0
        for row in leaderboard:
            if not isinstance(row, dict):
                continue
            strategy_id = self._by_agent.get(str(row.get("id")))
            if strategy_id is None:
                continue
            record = self._strategies[strategy_id]
            try:
                cash = float(row.get("cash") or 0.0)
                pos = int(row.get("pos") or 0)
                pnl = float(row.get("pnl") or 0.0)
            except (TypeError, ValueError):
                continue
            equity = cash + pos * price
            record.latest = {
                "seq": seq,
                "day_count": self._day_count,
                "equity": equity,
                "pnl": pnl,
                "pos": pos,
                "cash": cash,
            }
            record.touch_run(run_id)
            bucket = self._bucket(record, run_id, self._day_count)
            if bucket.start_equity is None:
                bucket.start_equity = equity
            bucket.end_equity = equity
            bucket.equity_seq.append((seq, equity))
            if record.status == "idle":
                record.status = "running"

    def _on_trade(self, payload: dict) -> None:
        """Record a fill row for each tracked side of a print."""
        buyer = payload.get("buyer")
        seller = payload.get("seller")
        for agent_id, side in ((buyer, "BUY"), (seller, "SELL")):
            strategy_id = self._by_agent.get(str(agent_id)) if agent_id else None
            if strategy_id is None:
                continue
            record = self._strategies[strategy_id]
            try:
                price = float(payload.get("price") or 0.0)
                qty = int(payload.get("qty") or 0)
            except (TypeError, ValueError):
                continue
            run_id = self._run_id or "unknown"
            day_count = self._day_count
            row = {
                "ts": _iso_from_unix(payload.get("unix_time")),
                "unix_time": payload.get("unix_time"),
                "day_count": day_count,
                "market_minute": payload.get("market_minute"),
                "run_id": run_id,
                "side": side,
                "price": price,
                "qty": qty,
                "cash_delta": -price * qty if side == "BUY" else price * qty,
                "counterparty": seller if side == "BUY" else buyer,
            }
            record.touch_run(run_id)
            self._bucket(record, run_id, day_count).trades.append(row)
            if record.status == "idle":
                record.status = "running"

    def _on_event(self, payload: dict) -> None:
        """Handle session events; ``day_close`` finalizes the day buckets."""
        if payload.get("kind") != "session":
            return
        action = payload.get("action")
        if action != "day_close":
            return
        try:
            day_count = int(payload.get("day_count") or self._day_count)
        except (TypeError, ValueError):
            day_count = self._day_count
        run_id = self._run_id or "unknown"
        closed = 0
        for record in self._strategies.values():
            bucket = record.days.get((run_id, day_count))
            if bucket is not None and not bucket.closed:
                bucket.closed = True
                closed += 1
        if closed:
            logger.info(
                "day %d closed for run %s (%d strategy bucket(s) finalized)",
                day_count, run_id, closed,
            )

    def _bucket(self, record: _StrategyRecord, run_id: str, day_count: int) -> _DayBucket:
        """Get or create the (run, day) bucket for ``record``."""
        key = (run_id, day_count)
        bucket = record.days.get(key)
        if bucket is None:
            bucket = _DayBucket(
                run_id=run_id, day_count=day_count, symbol=self._symbol or ""
            )
            record.days[key] = bucket
        return bucket

    # ------------------------------------------------------------------ #
    # Read API                                                            #
    # ------------------------------------------------------------------ #
    def list_strategies(self) -> list[dict]:
        """Summaries of every tracked strategy, in attach order."""
        with self._lock:
            return [self._summary(record) for record in self._strategies.values()]

    def strategy_detail(self, strategy_id: str) -> dict:
        """Full detail with per-day outcomes for one strategy.

        Raises :class:`KeyError` for an unknown ``strategy_id``.
        """
        with self._lock:
            record = self._require(strategy_id)
            realized_total, per_bucket = self._fill_stats(record)
            days: list[dict] = []
            for key, bucket in record.days.items():
                stats = per_bucket.get(key, {"wins": 0, "losses": 0, "realized": 0.0})
                decided = stats["wins"] + stats["losses"]
                start = bucket.start_equity if bucket.start_equity is not None else 0.0
                end = bucket.end_equity if bucket.end_equity is not None else start
                days.append(
                    {
                        "run_id": bucket.run_id,
                        "day_count": bucket.day_count,
                        "symbol": bucket.symbol,
                        "trades": [dict(t) for t in bucket.trades],
                        "n_trades": len(bucket.trades),
                        "gross_pnl": end - start,
                        "start_equity": start,
                        "end_equity": end,
                        "max_drawdown": self._max_drawdown(
                            [e for _, e in bucket.equity_seq]
                        ),
                        "win_rate": (stats["wins"] / decided) if decided else 0.0,
                        "realized_pnl": stats["realized"],
                        "closed": bucket.closed,
                    }
                )
            detail = self._summary(record)
            detail["realized_pnl"] = realized_total
            detail["days"] = days
            return detail

    def strategy_trades(self, strategy_id: str, day: Optional[int] = None) -> list[dict]:
        """All fill rows for a strategy in chronological order.

        ``day`` filters to a single ``day_count``. Raises :class:`KeyError`
        for an unknown strategy.
        """
        with self._lock:
            record = self._require(strategy_id)
            rows: list[dict] = []
            for bucket in record.days.values():
                for trade in bucket.trades:
                    if day is None or trade.get("day_count") == day:
                        rows.append(dict(trade))
            return rows

    # ------------------------------------------------------------------ #
    # Computation helpers                                                 #
    # ------------------------------------------------------------------ #
    def _require(self, strategy_id: str) -> _StrategyRecord:
        """Fetch a record or raise :class:`KeyError`."""
        record = self._strategies.get(strategy_id)
        if record is None:
            raise KeyError(f"unknown strategy_id: {strategy_id}")
        return record

    def _summary(self, record: _StrategyRecord) -> dict:
        """One strategy's summary row (PROTOCOL_V2 `list_strategies` shape)."""
        realized_total, _ = self._fill_stats(record)
        total_trades = sum(len(b.trades) for b in record.days.values())
        total_pnl = record.latest["pnl"] if record.latest else 0.0
        return {
            "strategy_id": record.strategy_id,
            "agent_id": record.agent_id,
            "name": record.name,
            "script_id": record.script_id,
            "status": record.status,
            "started_at": record.started_at,
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "realized_pnl": realized_total,
            "n_days": len(record.days),
            "n_runs": len(record.runs),
        }

    def _fill_stats(
        self, record: _StrategyRecord
    ) -> tuple[float, dict[tuple[str, int], dict]]:
        """Average-cost realized P&L over the fill log, per (run, day) bucket.

        Positions carry across days within a run but reset between runs (the
        engine re-initializes ledgers per run). A fill that reduces existing
        exposure realizes ``(exit - avg_cost) * closed_qty`` (sign-adjusted);
        wins/losses count only those realizing fills.
        """
        rows_by_run: dict[str, list[tuple[tuple[str, int], dict]]] = {}
        for key, bucket in record.days.items():
            rows_by_run.setdefault(key[0], []).extend(
                (key, trade) for trade in bucket.trades
            )
        total_realized = 0.0
        per_bucket: dict[tuple[str, int], dict] = {}
        for rows in rows_by_run.values():
            pos = 0
            avg_cost = 0.0
            for key, trade in rows:
                qty = int(trade["qty"])
                price = float(trade["price"])
                realized = 0.0
                closed_qty = 0
                if trade["side"] == "BUY":
                    if pos < 0:  # buying back a short
                        closed_qty = min(qty, -pos)
                        realized = (avg_cost - price) * closed_qty
                        pos += closed_qty
                        remainder = qty - closed_qty
                        if remainder > 0:  # flipped long
                            pos = remainder
                            avg_cost = price
                    else:
                        avg_cost = (
                            (avg_cost * pos + price * qty) / (pos + qty)
                            if (pos + qty) > 0
                            else price
                        )
                        pos += qty
                else:  # SELL
                    if pos > 0:  # selling down a long
                        closed_qty = min(qty, pos)
                        realized = (price - avg_cost) * closed_qty
                        pos -= closed_qty
                        remainder = qty - closed_qty
                        if remainder > 0:  # flipped short
                            pos = -remainder
                            avg_cost = price
                    else:
                        short = -pos
                        avg_cost = (
                            (avg_cost * short + price * qty) / (short + qty)
                            if (short + qty) > 0
                            else price
                        )
                        pos -= qty
                if pos == 0:
                    avg_cost = 0.0
                total_realized += realized
                stats = per_bucket.setdefault(
                    key, {"wins": 0, "losses": 0, "realized": 0.0}
                )
                stats["realized"] += realized
                if closed_qty > 0:
                    if realized > 0:
                        stats["wins"] += 1
                    else:
                        stats["losses"] += 1
        return total_realized, per_bucket

    @staticmethod
    def _max_drawdown(equities: list[float]) -> float:
        """Maximum peak-to-trough drop over an equity series (>= 0)."""
        peak = float("-inf")
        drawdown = 0.0
        for equity in equities:
            if equity > peak:
                peak = equity
            elif peak - equity > drawdown:
                drawdown = peak - equity
        return drawdown
