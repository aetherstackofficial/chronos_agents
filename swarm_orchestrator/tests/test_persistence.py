"""Persistence round-trip tests (PROTOCOL §7) — runs, ticks, and equity snapshots."""
import sqlite3

import pytest

from engine.persistence import Persistence


def test_run_and_tick_roundtrip(tmp_path):
    db_path = str(tmp_path / "chronos_test.db")
    db = Persistence(db_path)
    db.start_run("run_x", "2026-01-01T09:15:00", "TCS", "TECH", "LIVE", 42, "{}")
    db.write_tick("run_x", {
        "seq": 1, "unix_time": 1767258900, "market_minute": 1, "day_count": 1,
        "last_price": 190.0, "best_bid": 189.95, "best_ask": 190.05,
        "vwap": 190.0, "rsi": 50.0, "ofi": 0.0, "step_volume": 100, "regime": "RANGING",
    })
    db.write_snapshots("run_x", 1, [
        {"agent_id": "WHALE_0", "agent_type": "Institution",
         "cash": 5_000_000.0, "pos": 0, "equity": 5_000_000.0},
    ])
    db.close()

    con = sqlite3.connect(db_path)
    assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0] == 1
    equity = con.execute(
        "SELECT equity FROM agent_snapshots WHERE run_id='run_x' AND agent_id='WHALE_0'"
    ).fetchone()[0]
    assert equity == pytest.approx(5_000_000.0)
    con.close()


def test_readonly_reader_can_open_while_engine_writes(tmp_path):
    db_path = str(tmp_path / "chronos_ro.db")
    db = Persistence(db_path)
    db.start_run("run_y", "2026-01-01T09:15:00", "TCS", "TECH", "LIVE", 1, "{}")
    # A read-only connection (as the bridge uses) must succeed concurrently.
    ro = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    assert ro.execute("SELECT symbol FROM runs WHERE run_id='run_y'").fetchone()[0] == "TCS"
    ro.close()
    db.close()
