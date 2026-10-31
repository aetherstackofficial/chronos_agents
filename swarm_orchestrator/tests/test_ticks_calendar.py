"""NSE tick table, monthly tick review and the session calendar (engine)."""
import datetime as dt
import glob
import os

import numpy as np
import pytest

from engine.book import TICK_BANDS, tick_for_price
from engine.matching_engine import MatchingEngine

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = glob.glob(os.path.join(HERE, "data", "real", "nse_kaggle", "*_minute.csv"))


def _on_grid(prices, tick):
    k = np.asarray(prices, float) / tick
    return np.abs(k - np.round(k)) < 1e-6


def _stamp(unix):
    return dt.datetime.fromtimestamp(unix, dt.timezone.utc)


@pytest.mark.parametrize("price,tick", [
    (3.0, 0.01), (249.99, 0.01), (250.0, 0.05), (999.95, 0.05), (1000.0, 0.10),
    (4999.9, 0.10), (5000.0, 0.50), (9999.5, 0.50), (10000.0, 1.00),
    (19999.0, 1.00), (20000.0, 5.00), (150000.0, 5.00)])
def test_nse_tick_bands(price, tick):
    assert tick_for_price(price) == tick


@pytest.mark.skipif(not REAL, reason="real NSE data not downloaded (python -m validation.fetch_data)")
def test_real_closes_follow_table_and_monthly_review():
    """Every real close from 2025-06 sits on the grid of the tick implied by
    the previous month's last close, and is not on a coarser grid overall
    (the table is not finer than the market)."""
    pd = pytest.importorskip("pandas")
    coarser = {t: c for (_, t), (_, c) in zip(TICK_BANDS, TICK_BANDS[1:])}
    for path in REAL:
        d = pd.read_csv(path, usecols=["date", "close"])
        d["date"] = pd.to_datetime(d["date"])
        d = d[d.date >= "2025-05-01"]
        d["ym"] = d.date.dt.to_period("M")
        month_last = d.groupby("ym").close.last()
        used = {}
        for ym, g in d[d.date >= "2025-06-01"].groupby("ym"):
            tick = tick_for_price(month_last[ym - 1])
            assert _on_grid(g.close, tick).all(), (path, str(ym), tick)
            used.setdefault(tick, []).extend(g.close.tolist())
        for tick, px in used.items():
            if tick in coarser:
                assert not _on_grid(px, coarser[tick]).all(), (path, tick)


def _engine(price=1350.0):
    e = MatchingEngine(db=None)
    e.init_sim("RELIANCE", "SIM", price, seed=1)
    return e


def _run_days(e, days):
    """Physics only (no agents, so no trades); returns (day, minute, unix) per tick."""
    out = []
    for _ in range(days * e.SESSION_MINUTES):
        p = [x for t, x in e.physics_tick() if t == "tick"][-1]
        out.append((p["day_count"], p["market_minute"], p["unix_time"]))
    return out


def test_sessions_are_weekdays_0915_to_1530_without_drift():
    rows = _run_days(_engine(), 45)
    by_day = {}
    for d, m, u in rows:
        by_day.setdefault(d, []).append((m, u))
    dates = []
    for d, ms in by_day.items():
        first, last = _stamp(ms[0][1]), _stamp(ms[-1][1])
        assert ms[0][0] == 1 and ms[-1][0] == 375
        assert (first.hour, first.minute) == (9, 16), d     # bar 1 closes 09:16
        assert (last.hour, last.minute) == (15, 30), d      # bar 375 closes 15:30
        assert first.date() == last.date()
        assert first.weekday() < 5, d
        dates.append(first.date())
    # Consecutive weekdays: Thu 1 Jan 2026, Fri 2 Jan, Mon 5 Jan, ...
    assert dates[:3] == [dt.date(2026, 1, 1), dt.date(2026, 1, 2), dt.date(2026, 1, 5)]
    for a, b in zip(dates, dates[1:]):
        assert (b - a).days == (3 if a.weekday() == 4 else 1)


def test_tick_changes_only_at_month_start_from_previous_month_close():
    e = _engine(990.0)                       # ₹250-1000 band: tick 0.05
    assert e.tick == 0.05
    _run_days(e, 4)                          # through Tue 6 Jan
    e.last_price = 1005.0                    # stands in for trades across ₹1000 mid-month
    for _ in range(18):                      # rest of January (Wed 7 .. Fri 30 Jan)
        _run_days(e, 1)
        assert e.tick == 0.05, _stamp(e.sim_unix_time)
    assert _stamp(e.sim_unix_time).month == 1
    _run_days(e, 1)                          # Mon 2 Feb: review on January's last close
    assert _stamp(e.sim_unix_time).date() == dt.date(2026, 2, 2)
    assert e.tick == 0.10


def test_band_crossing_reverted_before_month_end_keeps_tick():
    e = _engine(990.0)
    _run_days(e, 4)
    e.last_price = 1005.0
    _run_days(e, 10)
    e.last_price = 995.0                     # back in band by January's last close
    _run_days(e, 9)                          # into the first February session
    assert _stamp(e.sim_unix_time).month == 2
    assert e.tick == 0.05
