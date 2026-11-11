"""Split boundaries, embargo, and episode-start validity."""
import numpy as np
import pandas as pd
import pytest

from training.data import splits
from training.data.splits import split_of, sessions_in_split, valid_episode_starts


def _frame(dates, gaps=None):
    """One row per session at minute 0 (enough for the split helpers)."""
    gaps = gaps or {}
    return pd.DataFrame({
        "datetime": pd.to_datetime(pd.Series(dates)) + pd.to_timedelta("9h15min"),
        "minute": 0,
        "session_id": np.arange(len(dates)),
        "gap_before": [i in gaps for i in range(len(dates))],
    })


def test_split_of_boundaries_and_embargo():
    assert split_of("2015-01-30") is None
    assert split_of("2015-02-02") == "train"
    assert split_of("2022-12-30") == "train"
    # validation starts Sun 2023-01-01: embargo = Jan 2-6 (5 weekdays)
    for d in ("2022-12-31", ):
        assert split_of(d) == "train"
    for d in ("2023-01-01", "2023-01-02", "2023-01-06"):
        assert split_of(d) is None
    assert split_of("2023-01-09") == "validation"
    assert split_of("2024-06-28") == "validation"
    # test starts Mon 2024-07-01: embargo Jul 1-5
    assert split_of("2024-06-30") == "validation"
    for d in ("2024-07-01", "2024-07-05"):
        assert split_of(d) is None
    assert split_of("2024-07-08") == "test"
    assert split_of("2026-04-30") == "test"
    assert split_of("2026-05-04") is None
    assert split_of(pd.Timestamp("2024-07-08 09:15")) == "test"


def test_held_out_default_empty():
    assert splits.HELD_OUT_STOCKS == ()
    assert not splits.is_held_out("RELIANCE")


def test_sessions_in_split_excludes_embargo():
    days = pd.bdate_range("2022-12-26", "2023-01-20")
    df = _frame(days)
    train = sessions_in_split(df, "train")
    val = sessions_in_split(df, "validation")
    assert set(days[train].strftime("%Y-%m-%d")) == {"2022-12-26", "2022-12-27", "2022-12-28",
                                                      "2022-12-29", "2022-12-30"}
    assert days[val][0] == pd.Timestamp("2023-01-09")
    assert len(set(train) & set(val)) == 0
    # embargoed sessions (Jan 2-6) belong to nothing
    embargoed = [i for i, d in enumerate(days) if pd.Timestamp("2023-01-02") <= d <= pd.Timestamp("2023-01-06")]
    assert len(embargoed) == 5
    assert not set(embargoed) & set(train) and not set(embargoed) & set(val)


def test_no_episode_crosses_split_end_or_start():
    days = pd.bdate_range("2022-12-19", "2023-01-31")
    df = _frame(days)
    L = 3
    for split in ("train", "validation"):
        starts = valid_episode_starts(df, split, L)
        ins = set(sessions_in_split(df, split))
        assert len(starts) > 0
        for s in starts:
            assert all((s + k) in ins for k in range(L))
    # last train day is Fri 2022-12-30: no train episode of 3 may start after Wed 12-28
    train_starts = valid_episode_starts(df, "train", 3)
    assert days[train_starts].max() == pd.Timestamp("2022-12-28")
    # the first validation episode starts on the first non-embargoed day, memory may precede it
    assert days[valid_episode_starts(df, "validation", 3)].min() == pd.Timestamp("2023-01-09")


def test_no_episode_crosses_gap_day():
    days = pd.bdate_range("2023-02-01", "2023-02-28")
    gap_at = 8                       # session 8 follows a missing trading day
    df = _frame(days, gaps={gap_at})
    L = 4
    starts = set(valid_episode_starts(df, "validation", L).tolist())
    for s in range(len(days) - L + 1):
        crosses = any((s + k) == gap_at for k in range(1, L))
        assert (s in starts) == (not crosses), s
    # starting exactly ON the gap day is fine (only the memory crosses it)
    assert gap_at in starts
    assert (gap_at - 1) not in starts and (gap_at - L + 1) not in starts


def test_too_short_and_bad_args():
    df = _frame(pd.bdate_range("2023-02-01", "2023-02-03"))
    assert len(valid_episode_starts(df, "validation", 10)) == 0
    with pytest.raises(ValueError):
        valid_episode_starts(df, "nope", 2)
    with pytest.raises(ValueError):
        sessions_in_split(df, "nope")
    with pytest.raises(ValueError):
        valid_episode_starts(df, "validation", 0)
