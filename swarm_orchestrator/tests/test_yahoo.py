"""Yahoo Finance replay download tests (features/yahoo.py, POST /api/replay/yahoo).

A fake ``yfinance`` module is injected, so these tests never touch the network
and do not need yfinance installed. The endpoint tests need fastapi +
python-multipart and are skipped without them.
"""
import datetime as dt
import os
import sys
import types

import pandas as pd
import pytest

from engine.matching_engine import MatchingEngine
import features.yahoo as yahoo

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NOW = dt.datetime(2026, 10, 24, 12, 0, tzinfo=dt.timezone.utc)


def _frame(start, n, high_bump=0.5):
    """n consecutive 1-minute bars starting at ``start`` (IST)."""
    idx = pd.DatetimeIndex([start + dt.timedelta(minutes=i) for i in range(n)])
    return pd.DataFrame({"Open": [100.0] * n, "High": [100.0 + high_bump] * n,
                         "Low": [99.5] * n, "Close": [100.2] * n,
                         "Volume": [1000] * n}, index=idx)


@pytest.fixture
def fake_yf(monkeypatch):
    """Install a fake yfinance; record every history() call."""
    calls = []
    state = {"frame": lambda s, e: _frame(dt.datetime(2026, 10, 23, 9, 15, tzinfo=IST), 3),
             "error": None}

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start, end, interval, **kw):
            calls.append((self.symbol, start, end, interval))
            if state["error"]:
                raise state["error"]
            return state["frame"](start, end)

    mod = types.ModuleType("yfinance")
    mod.Ticker = Ticker
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    monkeypatch.setattr(yahoo, "CHUNK_PAUSE_S", 0)
    return {"calls": calls, "state": state}


# --- limits are enforced before Yahoo is called ------------------------------
@pytest.mark.parametrize("symbol,interval,days,needle", [
    ("RELIANCE.NS", "1m", 3, "15-minute"),        # 1-minute Yahoo data is no longer offered
    ("RELIANCE.NS", "15m", 60, "1-59"),
    ("RELIANCE.NS", "5m", 3, "interval"),
    ("RELIANCE.NS", "15m", 0, "1-59"),
    ("../etc", "15m", 3, "invalid symbol"),
])
def test_limits_rejected_without_calling_yahoo(fake_yf, symbol, interval, days, needle):
    with pytest.raises(yahoo.YahooError) as exc:
        yahoo.fetch_bars(symbol, interval, days, now=NOW)
    assert exc.value.status == 400 and needle in exc.value.message
    assert fake_yf["calls"] == []


def test_windows_helper_still_splits_long_ranges():
    start = NOW - dt.timedelta(days=20)
    windows = yahoo._windows(start, NOW, 7)
    assert len(windows) == 3 and windows[0][0] == start and windows[-1][1] == NOW


def test_15m_is_one_request(fake_yf):
    yahoo.fetch_bars("TCS.NS", "15m", 45, now=NOW)
    assert len(fake_yf["calls"]) == 1 and fake_yf["calls"][0][3] == "15m"


def test_rows_keep_exchange_time_and_drop_duplicates(fake_yf):
    out = yahoo.fetch_bars("RELIANCE.NS", "15m", 20, now=NOW)
    assert len(out["rows"]) == 3
    assert (out["rows"][0]["date"], out["rows"][0]["time"]) == ("2026-10-23", "09:15")


def test_empty_result_gives_ns_hint(fake_yf):
    fake_yf["state"]["frame"] = lambda s, e: pd.DataFrame()
    with pytest.raises(yahoo.YahooError) as exc:
        yahoo.fetch_bars("RELIANCE", "15m", 3, now=NOW)
    assert exc.value.status == 404 and "RELIANCE.NS" in exc.value.message


def test_yahoo_exception_becomes_502(fake_yf):
    fake_yf["state"]["error"] = RuntimeError("Too Many Requests")
    with pytest.raises(yahoo.YahooError) as exc:
        yahoo.fetch_bars("RELIANCE.NS", "15m", 3, now=NOW)
    assert exc.value.status == 502 and "rate-limiting" in exc.value.message


def test_inconsistent_bar_is_repaired(fake_yf):
    # High below Close: the replayer would reject this bar, so it is widened.
    fake_yf["state"]["frame"] = lambda s, e: _frame(
        dt.datetime(2026, 10, 23, 9, 15, tzinfo=IST), 2, high_bump=0.1)
    out = yahoo.fetch_bars("RELIANCE.NS", "15m", 3, now=NOW)
    assert out["repaired"] == 2
    assert all(r["high"] >= r["close"] for r in out["rows"])


# --- endpoint: download is saved as a CSV the engine can replay --------------
@pytest.fixture
def client(fake_yf, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from features.routes_uploads import router
    monkeypatch.setenv("CHRONOS_UPLOAD_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_endpoint_saves_csv_and_engine_replays_it(client, tmp_path):
    r = client.post("/api/replay/yahoo",
                    json={"symbol": "RELIANCE.NS", "interval": "15m", "days": 3})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["source"] == "yahoo" and body["bars"] == 3
    assert os.path.dirname(body["csv_path"]) == str(tmp_path)
    assert os.path.basename(body["csv_path"]).startswith("yahoo_RELIANCE_NS_15m_2026-10-23")
    _, n = MatchingEngine().start_replay(body["csv_path"], "RELIANCE.NS", "ENERGY")
    assert n == 3


def test_endpoint_reports_errors(client, fake_yf, tmp_path):
    r = client.post("/api/replay/yahoo", json={"symbol": "RELIANCE.NS", "interval": "15m", "days": 60})
    assert r.status_code == 400
    fake_yf["state"]["frame"] = lambda s, e: pd.DataFrame()
    r = client.post("/api/replay/yahoo", json={"symbol": "RELIANCE", "interval": "15m", "days": 3})
    assert r.status_code == 404 and ".NS" in r.json()["detail"]
    assert os.listdir(tmp_path) == []           # failed downloads leave no file
