"""Build per-stock parquet files from the Kaggle NSE 1-minute CSVs.

    python -m training.data.build_dataset [--src DIR] [--out DIR] [--symbols A,B]

Cleaning is exactly ``validation.data.load_real`` (imported, not copied): drop
non-positive prices, floor timestamps to the minute, drop duplicate minutes,
keep 09:15-15:29, keep only complete 375-bar days, count OHLC-inconsistent
rows (kept, only counted).

Output columns: datetime, open, high, low, close, volume, minute (0..374),
session_id (0.. over kept days), gap_before, sector. ``gap_before`` is True when
the kept day before this one is not the previous trading day, i.e. at least one
date of the trading calendar lies strictly between them. The calendar is the
union of kept dates over all processed stocks (so weekends and
holidays are not gaps, but a special weekend session some stock kept is a trading day). With a single
stock the calendar is that stock's own days, so no gap can be seen: build the
stocks together. The first session has no predecessor and gets False.

Stocks are processed one at a time (pass 1: clean + write; pass 2: add
gap_before once the calendar is known), so memory stays at one stock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow

from validation.data import SESSION_BARS, load_real
from training.data.sectors import load_sector_map, DEFAULT_SECTORS_CSV, UNKNOWN

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SRC = os.path.join(PKG_ROOT, "data", "real", "nse_kaggle")
DEFAULT_OUT = os.path.join(PKG_ROOT, "training", "data", "parquet")
OPEN_MINUTE_OF_DAY = 9 * 60 + 15


def clean_file(csv_path: str, symbol: str) -> tuple:
    """Clean one CSV with load_real. Returns (DataFrame without gap_before, info dict)."""
    s = load_real(csv_path, label=symbol)
    info = dict(s.info["cleaning"])
    dates = pd.to_datetime(pd.Series(s.info["dates"]))
    day_start = dates.values.astype("datetime64[ns]")[s.day]
    mins = (OPEN_MINUTE_OF_DAY + s.minute).astype("timedelta64[m]").astype("timedelta64[ns]")
    df = pd.DataFrame({
        "datetime": day_start + mins,
        "open": s.o, "high": s.h, "low": s.l, "close": s.c,
        "volume": np.rint(s.v).astype(np.int64),
        "minute": s.minute.astype(np.int16),
        "session_id": s.day.astype(np.int32),
    })
    return df, info


def compute_gap_before(session_dates: np.ndarray, calendar: np.ndarray) -> np.ndarray:
    """gap_before per session. ``session_dates`` sorted datetime64[D] of the kept
    days; ``calendar`` sorted datetime64[D] of all trading dates. True where a
    calendar date lies strictly between the previous kept day and this one."""
    session_dates = np.asarray(session_dates, dtype="datetime64[D]")
    calendar = np.asarray(calendar, dtype="datetime64[D]")
    idx = np.searchsorted(calendar, session_dates)
    if len(idx) and (idx.max() >= len(calendar) or not np.all(calendar[idx] == session_dates)):
        raise ValueError("a session date is missing from the trading calendar")
    gap = np.zeros(len(idx), dtype=bool)
    gap[1:] = np.diff(idx) > 1
    return gap


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_symbols(src: str) -> List[str]:
    return sorted(f[: -len("_minute.csv")] for f in os.listdir(src) if f.endswith("_minute.csv"))


def _session_dates(df: pd.DataFrame) -> np.ndarray:
    return df["datetime"].dt.normalize().values[df["minute"].values == 0].astype("datetime64[D]")


def build(src: str = DEFAULT_SRC, out: str = DEFAULT_OUT,
          symbols: Optional[List[str]] = None, sectors_csv: str = DEFAULT_SECTORS_CSV) -> dict:
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    symbols = [s.strip().upper() for s in (symbols or discover_symbols(src)) if s.strip()]
    if not symbols:
        raise SystemExit(f"no <SYMBOL>_minute.csv files found in {src}")
    sector_map = load_sector_map(sectors_csv)

    # ---- pass 1: clean each stock, write a temporary parquet ------------- #
    stocks: Dict[str, dict] = {}
    all_dates = set()
    for sym in symbols:
        path = os.path.join(src, f"{sym}_minute.csv")
        t1 = time.time()
        df, info = clean_file(path, sym)
        rows_kept = len(df)
        rows_after_session = (info["rows_raw"] - info["dropped_nonpositive_price"]
                              - info["dropped_duplicate_minutes"] - info["dropped_out_of_session"])
        sdates = _session_dates(df)
        assert len(sdates) == info["days_kept"] == int(df["session_id"].max()) + 1
        all_dates.update(sdates.tolist())
        df["sector"] = sector_map.get(sym, UNKNOWN)
        tmp = os.path.join(out, f"{sym}.pass1.parquet")
        df.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
        stocks[sym] = {
            "rows_read": info["rows_raw"],
            "rows_dropped_nonpositive_price": info["dropped_nonpositive_price"],
            "rows_dropped_duplicate_minutes": info["dropped_duplicate_minutes"],
            "rows_dropped_out_of_session": info["dropped_out_of_session"],
            "rows_dropped_incomplete_days": rows_after_session - rows_kept,
            "rows_kept": rows_kept,
            "timestamps_floored_from_seconds": info["floored_seconds"],
            "days_total_in_session": info["days_total"],
            "days_dropped_incomplete": info["days_dropped_incomplete"],
            "ohlc_inconsistent_rows": info["bad_ohlc_bars_kept"],
            "zero_volume_bars": info["zero_volume_bars"],
            "first_date": str(sdates[0]),
            "last_date": str(sdates[-1]),
            "sessions_kept": int(info["days_kept"]),
            "sector": df["sector"].iloc[0],
            "_tmp": tmp,
            "_secs": round(time.time() - t1, 1),
        }
        print(f"[pass1] {sym}: {rows_kept} rows, {info['days_kept']} sessions "
              f"({stocks[sym]['_secs']}s)", flush=True)
        del df

    # Calendar = every date any stock kept. That is the weekdays that traded plus
    # the rare special weekend sessions (e.g. budget-day Saturdays) which have a
    # full 375-bar session; ordinary weekends and holidays are absent.
    calendar = np.array(sorted(all_dates), dtype="datetime64[D]")

    # ---- pass 2: add gap_before, write the final parquet ------------------ #
    for sym, m in stocks.items():
        tmp = m.pop("_tmp")
        m.pop("_secs")
        df = pd.read_parquet(tmp)
        gap = compute_gap_before(_session_dates(df), calendar)
        df["gap_before"] = gap[df["session_id"].values]
        df = df[["datetime", "open", "high", "low", "close", "volume",
                 "minute", "session_id", "gap_before", "sector"]]
        final = os.path.join(out, f"{sym}.parquet")
        df.to_parquet(final, engine="pyarrow", compression="zstd", index=False)
        os.remove(tmp)
        m["gap_before_days"] = int(gap.sum())
        m["parquet_bytes"] = os.path.getsize(final)
        m["sha256"] = _sha256(final)
        print(f"[pass2] {sym}: gap_before days = {m['gap_before_days']}", flush=True)
        del df

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_seconds": round(time.time() - t0, 1),
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "python": sys.version.split()[0],
        "session_bars": SESSION_BARS,
        "source_dir": os.path.abspath(src),
        "calendar_symbols": symbols,
        "calendar_days": int(len(calendar)),
        "cleaning": "validation.data.load_real",
        "stocks": stocks,
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build per-stock parquet files from NSE 1-minute CSVs")
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--symbols", default=None, help="comma separated, default: every file in --src")
    args = ap.parse_args(argv)
    syms = args.symbols.split(",") if args.symbols else None
    man = build(args.src, args.out, syms)
    print(f"done in {man['build_seconds']}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
