"""Start an engine REPLAY run from bars already in memory."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Tuple

from engine.replay import HistoricalReplayer

logger = logging.getLogger("training.replay_from_bars")


def to_replay_bar(b: dict) -> dict:
    """A bar in exactly the shape ``HistoricalReplayer.load_csv`` produces:
    {o, h, l, c (float), v (int), ts (int epoch seconds or None)}."""
    return {"o": float(b["o"]), "h": float(b["h"]), "l": float(b["l"]),
            "c": float(b["c"]), "v": int(float(b["v"])),
            "ts": None if b.get("ts") is None else int(b["ts"])}


def start_replay_from_bars(engine, bars: List[dict], symbol, sector: str,
                           seed: int = 42) -> Tuple[str, int]:
    """Same steps as ``MatchingEngine.start_replay`` (engine/matching_engine.py)
    but the tape is a list of bar dicts instead of a CSV path. Returns
    (run_id, n_bars). The engine is not modified; its private reset/run-id
    helpers are called the way start_replay calls them."""
    if not bars:
        raise ValueError("bars is empty")
    tape = [to_replay_bar(b) for b in bars]
    if any(b["ts"] is None for b in tape):          # load_csv: one bad date -> all None
        for b in tape:
            b["ts"] = None
    sym = symbol or engine.symbol
    engine._reset_state(sym, sector, tape[0]["o"], seed, "REPLAY")
    engine.run_id = engine._make_run_id()
    engine.replayer = HistoricalReplayer(engine, tape)
    if tape[0].get("ts") is not None:
        engine.sim_unix_time = tape[0]["ts"]
    if engine.db is not None:
        engine.db.start_run(
            engine.run_id, datetime.now().isoformat(), sym, sector,
            "REPLAY", seed,
            json.dumps({"csv_path": None, "source": "bars", "symbol": sym,
                        "sector": sector, "seed": seed}))
    logger.info("start_replay_from_bars %s (%d bars) run_id=%s", sym, len(tape), engine.run_id)
    return engine.run_id, len(tape)
