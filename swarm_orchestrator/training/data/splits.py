"""Train / validation / test date splits, identical for every stock.

    train       2015-02-01 .. 2022-12-31
    validation  2023-01-01 .. 2024-06-30
    test        2024-07-01 .. 2026-04-30

Embargo: the first ``EMBARGO_DAYS`` (5) weekdays of validation and of test are
excluded (``split_of`` returns None for them). It is calendar based, not data
based, so every stock gets the same cut (a holiday inside the window makes the
embargo one trading day shorter; that is accepted). A training episode must lie
wholly inside its split, so none can span two sets.

An episode's MEMORY (sessions before its first traded session) may reach back
before the split start; only the traded sessions have to be inside the split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EMBARGO_DAYS = 5
SPLIT_RANGES = {
    "train": ("2015-02-01", "2022-12-31"),
    "validation": ("2023-01-01", "2024-06-30"),
    "test": ("2024-07-01", "2026-04-30"),
}
SPLIT_NAMES = tuple(SPLIT_RANGES)
HELD_OUT_STOCKS: tuple = ()      # ~10 of 100 stocks later; empty with only 4 stocks


def _effective_ranges() -> dict:
    out = {}
    for name, (a, b) in SPLIT_RANGES.items():
        start = np.datetime64(a, "D")
        if name != "train":
            # first EMBARGO_DAYS weekdays are dropped: start at the 6th weekday
            start = np.busday_offset(start, EMBARGO_DAYS, roll="forward")
        out[name] = (start, np.datetime64(b, "D"))
    return out


_RANGES = _effective_ranges()


def is_held_out(symbol: str) -> bool:
    return str(symbol).upper() in {s.upper() for s in HELD_OUT_STOCKS}


def _split_of_days(days: np.ndarray) -> np.ndarray:
    """Object array of split names / None for a datetime64[D] array."""
    res = np.full(len(days), None, dtype=object)
    for name, (a, b) in _RANGES.items():
        res[(days >= a) & (days <= b)] = name
    return res


def split_of(date):
    """'train' | 'validation' | 'test' | None (outside every range or embargoed)."""
    day = np.datetime64(pd.Timestamp(date).date(), "D")
    return _split_of_days(np.array([day]))[0]


def _session_table(df: pd.DataFrame):
    """(session_ids, dates, gap_before) with one entry per session, sorted."""
    first = df[df["minute"] == 0]
    return (first["session_id"].to_numpy(),
            first["datetime"].dt.normalize().to_numpy().astype("datetime64[D]"),
            first["gap_before"].to_numpy(dtype=bool))


def sessions_in_split(df: pd.DataFrame, split: str) -> np.ndarray:
    """session_ids whose date belongs to ``split`` (embargo already applied)."""
    if split not in SPLIT_RANGES:
        raise ValueError(f"unknown split {split!r}")
    sid, days, _ = _session_table(df)
    return sid[_split_of_days(days) == split]


def valid_episode_starts(df: pd.DataFrame, split: str, length_sessions: int) -> np.ndarray:
    """session_ids s where sessions s .. s+length-1 all exist, all lie in ``split``
    and none of s+1 .. s+length-1 has ``gap_before`` (no missing trading day
    inside the episode). A gap before the FIRST session only breaks the memory,
    which is allowed."""
    if split not in SPLIT_RANGES:
        raise ValueError(f"unknown split {split!r}")
    L = int(length_sessions)
    if L < 1:
        raise ValueError("length_sessions must be >= 1")
    sid, days, gap = _session_table(df)
    n = len(sid)
    if n < L:
        return np.empty(0, dtype=sid.dtype)
    inside = (_split_of_days(days) == split).astype(np.int64)
    # a "break" at position i means session i does not directly follow i-1
    brk = np.zeros(n, dtype=np.int64)
    brk[1:] = (np.diff(sid) != 1) | gap[1:]
    c_in = np.concatenate([[0], np.cumsum(inside)])
    c_brk = np.concatenate([[0], np.cumsum(brk)])
    starts = np.arange(n - L + 1)
    all_in = (c_in[starts + L] - c_in[starts]) == L
    no_break = (c_brk[starts + L] - c_brk[starts + 1]) == 0      # positions s+1 .. s+L-1
    return sid[starts[all_in & no_break]]
