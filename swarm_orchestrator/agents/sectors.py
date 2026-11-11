"""Sector vocabulary shared by the live agents and the training pipeline.

``SECTORS`` is a fixed tuple (NSE macro-economic sectors): its order defines the
one-hot layout of the model inputs, so never reorder or insert in the middle
(append only, and retrain if you do).

The live engine uses its own coarse labels in ``tick["sector"]`` (set by the
dashboard / bridge: TECH, BANK, OIL, PHARMA, DEFENSE). ``LIVE_ALIASES`` maps
those onto this vocabulary so a trained policy sees the same sector id in
training and in a live run. ``canonical_sector`` applies the mapping.

Lives in ``agents/`` (not ``training/``) because live agent code imports it and
the runtime images ship ``agents/`` but not necessarily ``training/``.
"""
from __future__ import annotations

from typing import Dict, List

SECTORS = (
    "Commodities",
    "Consumer Discretionary",
    "Energy",
    "Fast Moving Consumer Goods",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Information Technology",
    "Services",
    "Telecommunication",
    "Utilities",
    "Diversified",
    "UNKNOWN",
)
UNKNOWN = "UNKNOWN"

# Strings a live run puts in tick["sector"] (found in dashboard/app.js sector
# picker, dashboard/views/replaydata.js STOCKS groups, oracle._SECTOR_KEYWORDS,
# features/timemachine.py scenarios; engine default "TECH").
LIVE_ALIASES: Dict[str, str] = {
    "TECH": "Information Technology",
    "IT": "Information Technology",
    "BANK": "Financial Services",
    "OIL": "Energy",
    "PHARMA": "Healthcare",
    "DEFENSE": "Industrials",     # HAL / BEL are capital goods -> Industrials
    "DEFENCE": "Industrials",
}

_BY_LOWER = {s.lower(): s for s in SECTORS}


def canonical_sector(name) -> str:
    """Map any sector string (vocabulary name, live alias, any case) to ``SECTORS``.
    Anything unrecognised becomes ``"UNKNOWN"``."""
    if not isinstance(name, str):
        return UNKNOWN
    key = name.strip()
    if key.upper() in LIVE_ALIASES:
        return LIVE_ALIASES[key.upper()]
    return _BY_LOWER.get(key.lower(), UNKNOWN)


def one_hot(sector: str) -> List[float]:
    """One-hot over ``SECTORS``. An unknown name maps to the UNKNOWN slot."""
    name = sector if sector in SECTORS else UNKNOWN
    return [1.0 if s == name else 0.0 for s in SECTORS]
