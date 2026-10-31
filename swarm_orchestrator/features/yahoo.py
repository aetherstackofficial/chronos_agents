"""Yahoo Finance bar downloads for replay (via the unofficial ``yfinance`` lib).

Fetches 15-minute OHLCV bars and turns them into rows in the replay CSV format
(``date,time,open,high,low,close,volume``). The caller saves every download to
disk and replays from that file -- Yahoo is never in the replay path, so a run
can be repeated even after Yahoo blocks us or the data ages out of its window.

Yahoo's limits (checked live, Oct 2026 -- they change without notice):

* ``15m`` -- only the last ~60 days, in one request. This is the only interval
  offered: 1-minute Yahoo data (last ~30 days, 8 days per request) was removed
  from the product on purpose. Your own CSV can still hold 1-minute bars.
* Indian stocks need an exchange suffix: ``RELIANCE.NS`` (NSE) or
  ``RELIANCE.BO`` (BSE). Plain ``RELIANCE`` returns nothing.

yfinance does NOT raise on these errors: it logs a warning and returns an
empty table. So the limits are enforced here, before calling it, and an empty
result is turned into an error that says what to check.

Downloaded times stay in the exchange's own time zone (IST for .NS/.BO), so a
RELIANCE.NS bar at 09:15 reads 09:15 in the CSV.

Note on 15m data: the engine replays one bar per physics tick (= one sim
minute), so a 15-minute bar plays as one minute of sim time. The prices and
their order are real; the session is compressed 15x.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import time
from typing import Dict, List

log = logging.getLogger("chronos.yahoo")

# Per-interval limits. Kept one day inside Yahoo's real edges: a request that
# starts exactly 30 days back is rejected as "not within the last 30 days".
INTERVAL_LIMITS: Dict[str, dict] = {
    "15m": {"max_days": 59, "days_per_request": 59},
}
# Pause between chunks if a range is ever split into several requests, to stay
# gentle with an unofficial API that throttles heavy use (a 15m download is one request).
CHUNK_PAUSE_S = 0.5

# Letters, digits and the punctuation real tickers use: RELIANCE.NS, ^NSEI,
# BRK-B, EURUSD=X. Anything else is rejected before it reaches Yahoo.
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9.^=&-]{1,20}$")


class YahooError(Exception):
    """A download failure with the HTTP status the API should answer with."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _suffix_hint(symbol: str) -> str:
    """Hint for the most common mistake: an Indian stock without .NS/.BO."""
    if "." in symbol or symbol.startswith("^"):
        return ""
    return (f" Indian stocks need an exchange suffix: {symbol}.NS (NSE) "
            f"or {symbol}.BO (BSE).")


def validate_request(symbol: str, interval: str, days: int) -> str:
    """Check the request against Yahoo's limits; return the cleaned symbol."""
    symbol = (symbol or "").strip().upper()
    if not _SYMBOL_RE.match(symbol):
        raise YahooError(400, f"invalid symbol {symbol!r}; example: RELIANCE.NS")
    if interval not in INTERVAL_LIMITS:
        raise YahooError(400, f"only 15-minute bars are supported (interval '15m'); got {interval!r}")
    max_days = INTERVAL_LIMITS[interval]["max_days"]
    if not isinstance(days, int) or isinstance(days, bool) or not 1 <= days <= max_days:
        raise YahooError(
            400, f"days must be 1-{max_days} for {interval} data "
                 f"(Yahoo keeps {interval} bars for about {max_days + 1} days)")
    return symbol


def _windows(start: dt.datetime, end: dt.datetime,
             step_days: int) -> List[tuple]:
    """Split [start, end) into consecutive windows of at most step_days."""
    out, cur, step = [], start, dt.timedelta(days=step_days)
    while cur < end:
        out.append((cur, min(cur + step, end)))
        cur += step
    return out


def fetch_bars(symbol: str, interval: str, days: int,
               now: dt.datetime | None = None) -> dict:
    """Download bars from Yahoo and return them as replay CSV rows.

    Returns ``{"symbol", "rows", "repaired", "requests"}`` where each row is
    ``{"date","time","open","high","low","close","volume"}``.
    Raises YahooError (400 bad request, 404 no data, 502 Yahoo failed,
    503 yfinance not installed).
    """
    symbol = validate_request(symbol, interval, days)
    try:
        import yfinance as yf          # imported here: optional dependency
    except ImportError:
        raise YahooError(503, "Yahoo downloads are unavailable: the yfinance "
                              "package is not installed on the server")

    end = now or dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=days)
    step = INTERVAL_LIMITS[interval]["days_per_request"]
    windows = _windows(start, end, step)

    frames = []
    ticker = yf.Ticker(symbol)
    for i, (w_start, w_end) in enumerate(windows):
        if i:
            time.sleep(CHUNK_PAUSE_S)
        try:
            df = ticker.history(start=w_start, end=w_end, interval=interval,
                                auto_adjust=False, actions=False)
        except Exception as exc:       # network, rate limit, Yahoo format change
            raise YahooError(
                502, f"Yahoo Finance request failed ({type(exc).__name__}: {exc}). "
                     "It is an unofficial API and may be rate-limiting; "
                     "try again in a few minutes.")
        if df is not None and len(df):
            frames.append(df)

    if not frames:
        raise YahooError(
            404, f"Yahoo returned no {interval} data for {symbol} in the last "
                 f"{days} day(s). Check the symbol, and that the market was open "
                 f"in that period.{_suffix_hint(symbol)}")

    rows, repaired = [], 0
    seen = set()
    for df in frames:
        for ts, r in df.iterrows():
            if ts in seen:             # chunk edges can overlap by one bar
                continue
            seen.add(ts)
            try:
                o, h, l, c = (float(r["Open"]), float(r["High"]),
                              float(r["Low"]), float(r["Close"]))
                v = r["Volume"]
                v = 0 if v != v else int(v)          # NaN volume -> 0
            except (KeyError, TypeError, ValueError):
                continue
            # Skip empty/placeholder bars (NaN or non-positive prices).
            if not all(x == x and x > 0 for x in (o, h, l, c)):
                continue
            # Yahoo occasionally reports a high a hair below the close (or a
            # low above the open). The replayer rejects such bars, so widen
            # high/low to cover the bar and count how many were fixed.
            hi, lo = max(o, h, l, c), min(o, h, l, c)
            if hi != h or lo != l:
                repaired += 1
            rows.append({"ts": ts, "open": round(o, 4), "high": round(hi, 4),
                         "low": round(lo, 4), "close": round(c, 4),
                         "volume": max(v, 0)})

    if not rows:
        raise YahooError(404, f"Yahoo returned only empty bars for {symbol}.")

    rows.sort(key=lambda r: r["ts"])
    for r in rows:
        ts = r.pop("ts")
        r["date"] = ts.strftime("%Y-%m-%d")   # exchange-local (IST for .NS)
        r["time"] = ts.strftime("%H:%M")
    log.info("yahoo: %s %s %dd -> %d bars in %d request(s), %d repaired",
             symbol, interval, days, len(rows), len(windows), repaired)
    return {"symbol": symbol, "rows": rows, "repaired": repaired,
            "requests": len(windows)}
