"""Symbol -> sector mapping for the training pipeline.

The vocabulary itself (SECTORS, LIVE_ALIASES, canonical_sector, one_hot) lives in
``agents/sectors.py`` and is re-exported here.

``sectors.csv`` (symbol,sector) is provisional: the final symbol->sector
source is still being decided.
"""
from __future__ import annotations

import csv
import os
from functools import lru_cache
from typing import Dict, List

from agents.sectors import (  # noqa: F401  (re-exported: one vocabulary for live and training)
    LIVE_ALIASES, SECTORS, UNKNOWN, canonical_sector, one_hot,
)

DEFAULT_SECTORS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sectors.csv")


def _norm_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    for suffix in (".NS", ".BO"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def load_sector_map(path: str = DEFAULT_SECTORS_CSV) -> Dict[str, str]:
    """Read a ``symbol,sector`` CSV. Names outside ``SECTORS`` become UNKNOWN."""
    out: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            sym = _norm_symbol(row["symbol"])
            name = (row.get("sector") or "").strip()
            out[sym] = name if name in SECTORS else UNKNOWN
    return out


@lru_cache(maxsize=1)
def _default_map() -> Dict[str, str]:
    return load_sector_map(DEFAULT_SECTORS_CSV)


def sector_of(symbol: str) -> str:
    """Sector of ``symbol`` from the default sectors.csv; UNKNOWN if not listed."""
    return _default_map().get(_norm_symbol(symbol), UNKNOWN)
