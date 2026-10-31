"""Replay CSV upload tests (POST /api/replay/upload, features/routes_uploads.py).

The router is mounted on a bare FastAPI app, so no ZeroMQ sockets or engine
process are needed. Skipped when fastapi / python-multipart are not installed.
"""
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from engine.matching_engine import MatchingEngine  # noqa: E402
from features.routes_uploads import router  # noqa: E402

GOOD_CSV = (
    "date,time,open,high,low,close,volume\n"
    "2026-01-02,09:15,100.00,100.50,99.80,100.20,800\n"
    "2026-01-02,09:16,100.20,100.30,100.00,100.10,400\n"
)


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    """Point uploads at a temp dir with a small (1 MB) size limit."""
    monkeypatch.setenv("CHRONOS_UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("CHRONOS_UPLOAD_MAX_MB", "1")
    return tmp_path


@pytest.fixture
def client(upload_dir):
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _post(client, name, body):
    data = body.encode() if isinstance(body, str) else body
    return client.post("/api/replay/upload", files={"file": (name, data, "text/csv")})


def test_valid_csv_is_saved_and_loads_in_engine(client, upload_dir):
    r = _post(client, "prices.csv", GOOD_CSV)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["bars"] == 2 and body["filename"] == "prices.csv"
    assert os.path.dirname(body["csv_path"]) == str(upload_dir)
    # The returned path is exactly what REPLAY_START needs.
    _, n_bars = MatchingEngine().start_replay(body["csv_path"], "TCS", "TECH")
    assert n_bars == 2


def test_datetime_column_instead_of_date_time(client):
    csv_text = ("Timestamp,Open,High,Low,Close,Volume\n"
                "2026-01-02 09:15,100,101,99,100.5,10\n")
    assert _post(client, "dt.csv", csv_text).status_code == 201


def test_missing_column_reports_which(client, upload_dir):
    r = _post(client, "bad.csv", "date,time,open,high,low,close\n2026-01-02,09:15,1,1,1,1\n")
    assert r.status_code == 400
    assert "missing required columns ['volume']" in r.json()["detail"]
    assert r.json()["detail"].startswith("bad.csv:")      # user's name, not server path
    assert os.listdir(upload_dir) == []                    # nothing left behind


def test_bad_row_reports_line_number(client):
    csv_text = GOOD_CSV + "2026-01-02,09:17,100,99,101,100,5\n"   # high < low
    r = _post(client, "bad.csv", csv_text)
    assert r.status_code == 400
    assert "line 4" in r.json()["detail"] and "inconsistent OHLC" in r.json()["detail"]


def test_wrong_extension_rejected(client):
    r = _post(client, "prices.xlsx", GOOD_CSV)
    assert r.status_code == 400 and ".csv" in r.json()["detail"]


def test_empty_file_rejected(client):
    r = _post(client, "empty.csv", "")
    assert r.status_code == 400 and "empty" in r.json()["detail"]


def test_binary_file_rejected(client):
    r = _post(client, "photo.csv", b"\xff\xd8\xff\xe0" + bytes(range(256)))
    assert r.status_code == 400 and "UTF-8" in r.json()["detail"]


def test_oversized_file_rejected(client, upload_dir):
    row = "2026-01-02,09:15,100,101,99,100,10\n"
    big = "date,time,open,high,low,close,volume\n" + row * 40_000   # ~1.4 MB > 1 MB
    r = _post(client, "big.csv", big)
    assert r.status_code == 413 and "1 MB" in r.json()["detail"]
    assert os.listdir(upload_dir) == []


def test_path_traversal_name_stays_in_upload_dir(client, upload_dir):
    r = _post(client, "../../evil name.csv", GOOD_CSV)
    assert r.status_code == 201
    saved = r.json()["csv_path"]
    assert os.path.dirname(saved) == str(upload_dir)
    assert os.path.basename(saved).startswith("evil_name_")


def test_two_uploads_with_same_name_do_not_overwrite(client):
    a = _post(client, "prices.csv", GOOD_CSV).json()["csv_path"]
    b = _post(client, "prices.csv", GOOD_CSV).json()["csv_path"]
    assert a != b and os.path.exists(a) and os.path.exists(b)
