"""SQLite persistence for the Chronos engine (PROTOCOL §7).

Engine-owned, WAL mode so the bridge can attach concurrent read-only
readers. All writes are batched once per physics tick via executemany.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from typing import List, Optional

logger = logging.getLogger("engine.persistence")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
    run_id TEXT PRIMARY KEY, started_at TEXT, symbol TEXT, sector TEXT,
    mode TEXT, seed INTEGER, config_json TEXT);
CREATE TABLE IF NOT EXISTS ticks(
    run_id TEXT, seq INTEGER, unix_time INTEGER, market_minute INTEGER,
    day_count INTEGER, last_price REAL, best_bid REAL, best_ask REAL,
    vwap REAL, rsi REAL, ofi REAL, step_volume INTEGER, regime TEXT,
    PRIMARY KEY(run_id, seq));
CREATE TABLE IF NOT EXISTS trades(
    run_id TEXT, seq INTEGER, unix_time INTEGER, price REAL, qty INTEGER,
    buyer TEXT, seller TEXT, aggressor TEXT,
    PRIMARY KEY(run_id, seq));
CREATE TABLE IF NOT EXISTS agent_snapshots(
    run_id TEXT, seq INTEGER, agent_id TEXT, agent_type TEXT,
    cash REAL, pos INTEGER, equity REAL,
    PRIMARY KEY(run_id, seq, agent_id));
"""


class Persistence:
    """Authoritative run/tick/trade/snapshot store (SQLite, WAL)."""

    BUSY_TIMEOUT_MS = 5000

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or os.getenv("CHRONOS_DB", "chronos.db")
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(f"PRAGMA busy_timeout={self.BUSY_TIMEOUT_MS}")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("persistence open at %s (WAL)", self.path)

    def start_run(self, run_id: str, started_at: str, symbol: str,
                  sector: str, mode: str, seed: int, config_json: str) -> None:
        """Register a new run row (idempotent on run_id)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO runs(run_id, started_at, symbol, sector,"
            " mode, seed, config_json) VALUES(?,?,?,?,?,?,?)",
            (run_id, started_at, symbol, sector, mode, seed, config_json))
        self._conn.commit()

    def write_tick(self, run_id: str, tick_row: dict) -> None:
        """Persist one per-tick market summary row."""
        self._conn.execute(
            "INSERT OR REPLACE INTO ticks(run_id, seq, unix_time,"
            " market_minute, day_count, last_price, best_bid, best_ask,"
            " vwap, rsi, ofi, step_volume, regime)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, tick_row["seq"], tick_row["unix_time"],
             tick_row["market_minute"], tick_row["day_count"],
             tick_row["last_price"], tick_row["best_bid"],
             tick_row["best_ask"], tick_row["vwap"], tick_row["rsi"],
             tick_row["ofi"], tick_row["step_volume"], tick_row["regime"]))
        self._conn.commit()

    def write_trades(self, run_id: str, trades: List[dict]) -> None:
        """Persist a batch of prints (one per fill, seq-keyed)."""
        if not trades:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO trades(run_id, seq, unix_time, price,"
            " qty, buyer, seller, aggressor) VALUES(?,?,?,?,?,?,?,?)",
            [(run_id, t["seq"], t["unix_time"], t["price"], t["qty"],
              t["buyer"], t["seller"], t["aggressor"]) for t in trades])
        self._conn.commit()

    def write_snapshots(self, run_id: str, seq: int,
                        snapshots: List[dict]) -> None:
        """Persist per-agent equity snapshots for one tick seq."""
        if not snapshots:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO agent_snapshots(run_id, seq, agent_id,"
            " agent_type, cash, pos, equity) VALUES(?,?,?,?,?,?,?)",
            [(run_id, seq, s["agent_id"], s["agent_type"], s["cash"],
              s["pos"], s["equity"]) for s in snapshots])
        self._conn.commit()

    def close(self) -> None:
        """Flush and close the connection."""
        try:
            self._conn.commit()
        finally:
            self._conn.close()
        logger.info("persistence closed")
