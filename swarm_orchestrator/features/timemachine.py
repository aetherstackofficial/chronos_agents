"""Time Machine crisis presets (PROTOCOL_V2.md §5).

Five historical crisis scenarios, each backed by a REAL generated 1-minute CSV
tape (``date,time,open,high,low,close,volume`` — the exact REPLAY_START input
format, PROTOCOL.md §4.3/§9). Tapes are generated deterministically on first
use: each scenario seeds its own ``random.Random(scenario_id)`` (string seeding
is stable across processes/platforms in CPython), and the path is shaped in
three phases — pre-event drift, the characteristic event move, aftermath — with
volume swelling through the event. OHLC bars are internally consistent
(``high >= max(o, c)``, ``low <= min(o, c)``). No external data downloads.

Replaying a scenario re-lives the crisis through the real book; the moment a
user strategy trades, the engine latches to REACTIVE and history diverges
(PROTOCOL.md §9) — that IS the product thesis, made demoable.
"""
from __future__ import annotations

import csv
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("chronos.features")

SESSION_MINUTES = 375                 # one full venue session (PROTOCOL §3)
SESSION_OPEN = (9, 15)                # 09:15 first bar
TICK_SIZE = 0.05
DEFAULT_DATA_DIR = "data/scenarios"

# Package parent = swarm_orchestrator; relative data dirs resolve against it so
# the tapes land in the same place regardless of the caller's cwd.
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCENARIOS: List[dict] = [
    {
        "id": "flash_crash_2010",
        "name": "Flash Crash — May 6, 2010",
        "description": ("A calm drift snaps into a nine-percent intraday plunge "
                        "in minutes as liquidity evaporates, then partially "
                        "recovers into the close."),
        "symbol": "DJIX",
        "sector": "BANK",
        "bars": SESSION_MINUTES,
        "tag": "crash",
    },
    {
        "id": "covid_crash_2020",
        "name": "COVID Crash — March 12, 2020",
        "description": ("A multi-leg pandemic selloff: successive down legs with "
                        "failed bounces compress the pandemic collapse into one "
                        "session, bleeding roughly a quarter off the tape."),
        "symbol": "NIFTX",
        "sector": "BANK",
        "bars": SESSION_MINUTES,
        "tag": "crash",
    },
    {
        "id": "gme_squeeze_2021",
        "name": "GME Squeeze — January 27, 2021",
        "description": ("A parabolic short squeeze: price ramps roughly 300% on "
                        "exploding volume before fading hard off the peak."),
        "symbol": "GME",
        "sector": "TECH",
        "bars": SESSION_MINUTES,
        "tag": "squeeze",
    },
    {
        "id": "black_monday_1987",
        "name": "Black Monday — October 19, 1987",
        "description": ("Relentless capitulation: a full-session cascade of "
                        "selling takes the index down more than twenty percent "
                        "with only a feeble late bounce."),
        "symbol": "SPX87",
        "sector": "BANK",
        "bars": SESSION_MINUTES,
        "tag": "crash",
    },
    {
        "id": "rate_shock",
        "name": "Surprise Rate Shock",
        "description": ("A quiet session gaps four percent lower on a surprise "
                        "central-bank hike, then grinds steadily down through "
                        "the afternoon on elevated volume."),
        "symbol": "BNKX",
        "sector": "BANK",
        "bars": SESSION_MINUTES,
        "tag": "macro",
    },
]

_SCENARIO_INDEX: Dict[str, dict] = {sc["id"]: sc for sc in SCENARIOS}

# Real-event trading dates give the tapes an honest identity in the UI.
_SCENARIO_DATES: Dict[str, str] = {
    "flash_crash_2010": "2010-05-06",
    "covid_crash_2020": "2020-03-12",
    "gme_squeeze_2021": "2021-01-27",
    "black_monday_1987": "1987-10-19",
    "rate_shock": "2022-09-21",
}

_START_PRICES: Dict[str, float] = {
    "flash_crash_2010": 260.0,
    "covid_crash_2020": 310.0,
    "gme_squeeze_2021": 95.0,
    "black_monday_1987": 225.0,
    "rate_shock": 180.0,
}

# Per-minute phase spec: (n_bars, total_drift_pct, noise_pct, volume_mult).
# total_drift_pct is the cumulative log-return of the phase, spread evenly.
_Phase = Tuple[int, float, float, float]

_SCENARIO_PHASES: Dict[str, List[_Phase]] = {
    "flash_crash_2010": [
        (265, -0.004, 0.0009, 1.0),    # calm pre-event drift
        (16, -0.105, 0.0060, 9.0),     # the nine-minute-scale plunge
        (34, 0.055, 0.0035, 5.0),      # violent partial recovery
        (60, -0.004, 0.0018, 2.0),     # nervous aftermath
    ],
    "covid_crash_2020": [
        (55, -0.008, 0.0012, 1.2),     # heavy open
        (45, -0.055, 0.0035, 4.0),     # leg one down
        (35, 0.018, 0.0025, 2.5),      # failed bounce
        (70, -0.075, 0.0040, 5.0),     # leg two capitulates
        (40, 0.015, 0.0028, 3.0),      # second failed bounce
        (70, -0.055, 0.0038, 4.5),     # leg three
        (60, 0.008, 0.0020, 2.0),      # exhausted stabilization
    ],
    "gme_squeeze_2021": [
        (95, 0.050, 0.0020, 1.0),      # word gets around: slow bid
        (170, 1.450, 0.0080, 8.0),     # parabolic squeeze (~4x = +300%)
        (110, -0.510, 0.0090, 6.0),    # brutal fade off the peak (~-40%)
    ],
    "black_monday_1987": [
        (50, -0.012, 0.0018, 1.5),     # ominous open
        (280, -0.215, 0.0035, 4.0),    # all-day cascade to about -22%
        (45, 0.012, 0.0030, 3.0),      # feeble late bounce
    ],
    "rate_shock": [
        (180, 0.001, 0.0008, 1.0),     # quiet morning
        (7, -0.041, 0.0050, 8.0),      # the shock: sharp gap lower
        (188, -0.030, 0.0022, 3.0),    # afternoon grind on elevated volume
    ],
}

_BASE_VOLUME = 6000  # shares per calm minute before phase multipliers


def _resolve_dir(data_dir: str) -> str:
    """Absolute scenario directory; relative paths anchor at swarm_orchestrator."""
    if os.path.isabs(data_dir):
        return data_dir
    return os.path.abspath(os.path.join(_BASE_DIR, data_dir))


def _bar_time(minute: int) -> str:
    """HH:MM:SS of the ``minute``-th session bar (09:15 open)."""
    total = SESSION_OPEN[0] * 60 + SESSION_OPEN[1] + minute
    return f"{total // 60:02d}:{total % 60:02d}:00"


def _generate_tape(scenario_id: str) -> List[dict]:
    """Deterministically generate a full session of shaped 1-minute bars."""
    rng = random.Random(scenario_id)  # string seed → stable everywhere
    phases = _SCENARIO_PHASES[scenario_id]
    date = _SCENARIO_DATES[scenario_id]
    price = _START_PRICES[scenario_id]

    bars: List[dict] = []
    minute = 0
    for n_bars, total_drift, noise, vol_mult in phases:
        per_bar_drift = total_drift / n_bars
        for _ in range(n_bars):
            if minute >= SESSION_MINUTES:
                break
            open_ = price
            ret = per_bar_drift + rng.gauss(0.0, noise)
            close = max(TICK_SIZE, open_ * (1.0 + ret))
            body_high = max(open_, close)
            body_low = min(open_, close)
            wick = abs(rng.gauss(0.0, noise * 0.8))
            high = body_high * (1.0 + wick)
            low = max(TICK_SIZE / 2, body_low * (1.0 - wick))
            volume = int(_BASE_VOLUME * vol_mult *
                         (0.6 + 0.8 * rng.random()) *
                         (1.0 + 40.0 * abs(ret)))  # volume swells on the move
            bars.append({
                "date": date,
                "time": _bar_time(minute),
                "open": round(open_, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": max(1, volume),
            })
            price = close
            minute += 1

    # Top up (rounding of phase lengths) so every tape is exactly one session.
    while minute < SESSION_MINUTES:
        open_ = price
        ret = rng.gauss(0.0, 0.0008)
        close = max(TICK_SIZE, open_ * (1.0 + ret))
        bars.append({
            "date": date,
            "time": _bar_time(minute),
            "open": round(open_, 2),
            "high": round(max(open_, close) * (1.0 + abs(rng.gauss(0, 0.0005))), 2),
            "low": round(min(open_, close) * (1.0 - abs(rng.gauss(0, 0.0005))), 2),
            "close": round(close, 2),
            "volume": max(1, int(_BASE_VOLUME * (0.6 + 0.8 * rng.random()))),
        })
        price = close
        minute += 1

    # Rounding can break OHLC consistency by a cent — repair invariants.
    for bar in bars:
        bar["high"] = round(max(bar["high"], bar["open"], bar["close"]), 2)
        bar["low"] = round(min(bar["low"], bar["open"], bar["close"]), 2)
    return bars[:SESSION_MINUTES]


def ensure_scenarios(data_dir: str = DEFAULT_DATA_DIR) -> None:
    """Generate any missing scenario CSV tapes under ``data_dir``.

    Idempotent and deterministic: an existing file is never rewritten, and a
    regenerated file is byte-identical for the same scenario id.
    """
    directory = _resolve_dir(data_dir)
    os.makedirs(directory, exist_ok=True)
    for sc in SCENARIOS:
        path = os.path.join(directory, f"{sc['id']}.csv")
        if os.path.exists(path):
            continue
        bars = _generate_tape(sc["id"])
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["date", "time", "open", "high", "low",
                                "close", "volume"])
            writer.writeheader()
            writer.writerows(bars)
        log.info("timemachine: generated %s (%d bars) -> %s",
                 sc["id"], len(bars), path)


def get_scenario(scenario_id: str) -> Optional[dict]:
    """Return the scenario preset dict for ``scenario_id``, or ``None``."""
    return _SCENARIO_INDEX.get(scenario_id)


def scenario_csv_path(scenario_id: str, data_dir: str = DEFAULT_DATA_DIR) -> str:
    """Absolute CSV path for a scenario tape (input to REPLAY_START).

    :raises KeyError: if ``scenario_id`` is unknown.
    """
    if scenario_id not in _SCENARIO_INDEX:
        raise KeyError(f"unknown scenario: {scenario_id!r}")
    return os.path.join(_resolve_dir(data_dir), f"{scenario_id}.csv")
