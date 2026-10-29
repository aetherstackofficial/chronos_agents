"""Bar series, real-data loading/cleaning and shared preprocessing (spec §2).

Real and simulated data go through the SAME code path so that every metric
is computed identically on both (spec §15 step 2).

Symmetry decisions forced by the reference data (Kaggle NSE 1-minute OHLCV,
trade prices only, no quotes, no sub-bar data):

* Price series. The spec's tests 1-4 use the mid-quote. Real bars have no
  mid, so real-vs-sim comparisons use the trade-price close on both sides;
  the sim's mid-quote variant is computed too and is what the spec-default
  criteria are judged on.
* Realised volatility sigma_t. The spec wants sub-bar returns; neither
  source has them (the engine has no sub-minute events). Both sides use the
  Parkinson range estimator sigma_t = ln(H/L) / sqrt(4 ln 2).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd

SESSION_BARS = 375                     # NSE 09:15-15:29 one-minute bars
_PARKINSON = 1.0 / np.sqrt(4.0 * np.log(2.0))


@dataclass
class BarSeries:
    """1-minute bars with session structure. ``minute`` is 0..374."""
    day: np.ndarray
    minute: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    mid: Optional[np.ndarray] = None
    label: str = ""
    info: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.c)

    @property
    def n_days(self) -> int:
        return int(len(np.unique(self.day)))

    def subset(self, mask: np.ndarray) -> "BarSeries":
        return BarSeries(self.day[mask], self.minute[mask], self.o[mask],
                         self.h[mask], self.l[mask], self.c[mask], self.v[mask],
                         None if self.mid is None else self.mid[mask],
                         self.label, dict(self.info))


# --------------------------------------------------------------------------- #
# real data
# --------------------------------------------------------------------------- #
def load_real(path: str, label: Optional[str] = None) -> BarSeries:
    """Load and clean one Kaggle ``<SYM>_minute.csv``.

    Cleaning rules (each count is recorded in ``info['cleaning']``):
    1. drop rows with any non-positive price (bad 2015 rows in some files);
    2. floor timestamps to the minute (some days start at 09:15:01);
    3. drop duplicate minutes (keep first);
    4. keep only 09:15-15:29 bars (drops special/Muhurat evening sessions);
    5. keep only days with exactly 375 bars, so every day has the same
       time-of-day bins and no return spans a missing minute.
    Prices are as shipped (corporate-action adjusted by the vendor); that is
    fine for returns because overnight returns are excluded (spec §2.2).
    """
    df = pd.read_csv(path, usecols=["date", "open", "high", "low", "close", "volume"])
    info: Dict[str, int] = {"rows_raw": len(df)}
    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    info["dropped_nonpositive_price"] = int(bad.sum())
    df = df[~bad].copy()
    ts = pd.to_datetime(df["date"])
    info["floored_seconds"] = int((ts.dt.second != 0).sum())
    df["ts"] = ts.dt.floor("min")
    dup = df["ts"].duplicated()
    info["dropped_duplicate_minutes"] = int(dup.sum())
    df = df[~dup]
    mins = df["ts"].dt.hour * 60 + df["ts"].dt.minute - (9 * 60 + 15)
    in_sess = (mins >= 0) & (mins < SESSION_BARS)
    info["dropped_out_of_session"] = int((~in_sess).sum())
    df = df[in_sess].assign(minute=mins[in_sess])
    df["d"] = df["ts"].dt.normalize()
    counts = df.groupby("d")["minute"].transform("size")
    full = counts == SESSION_BARS
    info["days_total"] = int(df["d"].nunique())
    info["days_dropped_incomplete"] = int(df.loc[~full, "d"].nunique())
    df = df[full].sort_values("ts")
    bad_ohlc = (df.high < df[["open", "close", "low"]].max(axis=1)) | \
               (df.low > df[["open", "close", "high"]].min(axis=1))
    info["bad_ohlc_bars_kept"] = int(bad_ohlc.sum())
    info["zero_volume_bars"] = int((df.volume <= 0).sum())
    days = df["d"].values
    _, day_idx = np.unique(days, return_inverse=True)
    sym = label or os.path.basename(path).split("_")[0]
    s = BarSeries(day_idx.astype(int), df["minute"].values.astype(int),
                  df.open.values.astype(float), df.high.values.astype(float),
                  df.low.values.astype(float), df.close.values.astype(float),
                  df.volume.values.astype(float), None, sym,
                  {"cleaning": info,
                   "dates": pd.DatetimeIndex(np.unique(days)).strftime("%Y-%m-%d").tolist()})
    info["days_kept"] = s.n_days
    return s


def real_windows(s: BarSeries, n_days: int, step_days: int
                 ) -> Iterator[Tuple[int, int, BarSeries]]:
    """Rolling windows of ``n_days`` sessions stepped by ``step_days`` (§15.3)."""
    nd = s.n_days
    for start in range(0, nd - n_days + 1, step_days):
        mask = (s.day >= start) & (s.day < start + n_days)
        yield start, start + n_days, s.subset(mask)


# --------------------------------------------------------------------------- #
# simulator output
# --------------------------------------------------------------------------- #
def from_sim(bars: Dict[str, np.ndarray], warmup_days: int,
             label: str = "SIM") -> BarSeries:
    """Convert sim_driver bars to a BarSeries, dropping warm-up days (§2.2)."""
    keep = bars["day"] > warmup_days
    mid = bars["mid"][keep].copy()
    n_nan = int(np.isnan(mid).sum())
    if n_nan:
        # One-sided book at bar close: carry the last valid mid forward.
        mid = pd.Series(mid).ffill().bfill().values
    day = bars["day"][keep] - warmup_days - 1
    return BarSeries(day.astype(int), (bars["minute"][keep] - 1).astype(int),
                     bars["o"][keep], bars["h"][keep], bars["l"][keep],
                     bars["c"][keep], bars["v"][keep].astype(float), mid, label,
                     {"mid_nan_filled": n_nan})


# --------------------------------------------------------------------------- #
# preprocessing (§2.1-2.3)
# --------------------------------------------------------------------------- #
def intraday_returns(s: BarSeries, price: str = "c") -> Tuple[np.ndarray, np.ndarray]:
    """Log returns r_t = ln(P_t/P_{t-1}) excluding the first bar of each day.

    Returns (r, idx) where idx[k] is the bar index of r[k], so volume and
    sigma of the same bar can be aligned with it.
    """
    p = s.mid if price == "mid" else getattr(s, price)
    r = np.diff(np.log(p))
    same_day = s.day[1:] == s.day[:-1]
    idx = np.arange(1, len(p))[same_day]
    return r[same_day], idx


def parkinson_sigma(s: BarSeries) -> np.ndarray:
    return _PARKINSON * np.log(s.h / s.l)


def deseasonalize(x: np.ndarray, minute: np.ndarray, bins: int = SESSION_BARS
                  ) -> np.ndarray:
    """x_t / mean(x over all days in the same time-of-day bin) (§2.3)."""
    sums = np.bincount(minute, weights=x, minlength=bins)
    cnt = np.bincount(minute, minlength=bins)
    mean = np.divide(sums, cnt, out=np.zeros(bins), where=cnt > 0)
    denom = mean[minute]
    return np.divide(x, denom, out=np.zeros_like(x, dtype=float), where=denom > 0)


def deseasonalized_returns(r: np.ndarray, minute: np.ndarray) -> np.ndarray:
    """r_t / mean(|r| in its time-of-day bin): |r~| equals deseasonalized |r|."""
    sums = np.bincount(minute, weights=np.abs(r), minlength=SESSION_BARS)
    cnt = np.bincount(minute, minlength=SESSION_BARS)
    mean = np.divide(sums, cnt, out=np.zeros(SESSION_BARS), where=cnt > 0)
    denom = mean[minute]
    return np.divide(r, denom, out=np.zeros_like(r), where=denom > 0)
