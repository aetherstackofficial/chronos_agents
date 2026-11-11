"""Load one stock's parquet and turn session ranges into replay/seed bar dicts."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd

# CHRONOS_PARQUET_DIR selects another dataset build (e.g. training/data/parquet_n100) without code changes.
DEFAULT_PARQUET_DIR = (os.environ.get("CHRONOS_PARQUET_DIR")
                       or os.path.join(os.path.dirname(os.path.abspath(__file__)), "parquet"))
SESSION_BARS = 375


@lru_cache(maxsize=1)
def _load(symbol: str, parquet_dir: str) -> pd.DataFrame:
    # Memory matters: every training worker holds one stock, and 8 workers plus
    # eval workers ran the machine out of RAM. Measured per worker:
    # pd.read_parquet kept an extra arrow copy (+~100 MB RSS); reading the table
    # and converting with self_destruct frees it. One sector string repeated on
    # ~1M rows was half of the frame's 119 MB; as a category it is ~1 MB.
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pq.read_table(os.path.join(parquet_dir, f"{symbol}.parquet"))
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    del table
    if "sector" in df.columns:
        df["sector"] = df["sector"].astype("category")
    pa.default_memory_pool().release_unused()
    return df


def load_stock(symbol: str, parquet_dir: str = DEFAULT_PARQUET_DIR) -> pd.DataFrame:
    """The stock's parquet as a DataFrame. Cached for ONE stock only (the last
    one asked for), so looping over many stocks never holds more than one."""
    return _load(str(symbol).upper(), os.path.abspath(parquet_dir))


def bars_for_sessions(df: pd.DataFrame, first_session: int, n_sessions: int) -> List[dict]:
    """Bars of sessions ``first_session .. first_session+n_sessions-1`` as
    ``{day_count, minute, o, h, l, c, v, ts}`` (day_count = session_id + 1,
    ts = epoch seconds of the bar's wall-clock time read as UTC, the engine's
    convention), the format of MarketContext.seed() and the engine replayer."""
    sid = df["session_id"].to_numpy()
    lo = int(np.searchsorted(sid, first_session, side="left"))
    hi = int(np.searchsorted(sid, first_session + n_sessions, side="left"))
    if hi - lo != n_sessions * SESSION_BARS:
        raise ValueError(f"sessions {first_session}..{first_session + n_sessions - 1} "
                         f"not fully present ({hi - lo} bars)")
    part = df.iloc[lo:hi]
    ts = part["datetime"].to_numpy().astype("datetime64[s]").astype(np.int64).tolist()
    return [
        {"day_count": int(d) + 1, "minute": int(m), "o": float(o), "h": float(h),
         "l": float(l), "c": float(c), "v": int(v), "ts": int(t)}
        for d, m, o, h, l, c, v, t in zip(
            part["session_id"].tolist(), part["minute"].tolist(), part["open"].tolist(),
            part["high"].tolist(), part["low"].tolist(), part["close"].tolist(),
            part["volume"].tolist(), ts)
    ]
