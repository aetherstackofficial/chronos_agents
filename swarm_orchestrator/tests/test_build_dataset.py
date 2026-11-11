"""Training-data builder: cleaning counts, schema, session ids, gap_before, manifest."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from training.data.build_dataset import build, compute_gap_before

REAL_RELIANCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "real", "nse_kaggle", "RELIANCE_minute.csv")
COLUMNS = ["datetime", "open", "high", "low", "close", "volume",
           "minute", "session_id", "gap_before", "sector"]


def _day_rows(date: str, n_bars: int = 375, price: float = 100.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows, p = [], price
    for i in range(n_bars):
        mins = 9 * 60 + 15 + i
        o = p
        c = round(max(1.0, p + rng.normal(0, 0.05)), 2)
        h = round(max(o, c) + 0.02, 2)
        l = round(min(o, c) - 0.02, 2)
        rows.append((f"{date} {mins // 60:02d}:{mins % 60:02d}:00", o, h, l, c, 1000 + i))
        p = c
    return rows


def _write_csv(path, rows):
    pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"]).to_csv(path, index=False)


def test_compute_gap_before_missing_weekday():
    cal = np.array(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                    "2024-01-08"], dtype="datetime64[D]")
    mine = np.array(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05", "2024-01-08"],
                    dtype="datetime64[D]")           # Jan 3 missing
    gap = compute_gap_before(mine, cal)
    assert gap.tolist() == [False, False, True, False, False]   # weekend alone is not a gap


def test_compute_gap_before_rejects_date_outside_calendar():
    with pytest.raises(ValueError):
        compute_gap_before(np.array(["2024-01-02"], dtype="datetime64[D]"),
                           np.array(["2024-01-01"], dtype="datetime64[D]"))


def test_builder_synthetic_two_stocks(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    dates_a = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    rows_a = [r for i, d in enumerate(dates_a) for r in _day_rows(d, seed=i)]
    _write_csv(src / "AAA_minute.csv", rows_a)

    # BBB: misses 2024-01-03 entirely; has an incomplete day (2024-01-05, 100 bars),
    # an out-of-session bar, a duplicate minute and a non-positive price row.
    rows_b = []
    for i, d in enumerate(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-08"]):
        rows_b += _day_rows(d, seed=10 + i)
    rows_b += _day_rows("2024-01-05", n_bars=100, seed=99)
    rows_b.append(("2024-01-02 15:30:00", 100, 101, 99, 100, 5))          # out of session
    rows_b.append(("2024-01-02 09:15:00", 100, 101, 99, 100, 5))          # duplicate minute
    rows_b.append(("2024-01-04 10:00:00", 0, 1, 0, 0, 5))                 # non-positive
    rows_b.sort(key=lambda r: r[0])
    _write_csv(src / "BBB_minute.csv", rows_b)

    man = build(str(src), str(out), ["AAA", "BBB"])

    a = pd.read_parquet(out / "AAA.parquet")
    b = pd.read_parquet(out / "BBB.parquet")
    for df, n in ((a, 6), (b, 4)):
        assert list(df.columns) == COLUMNS
        assert len(df) == 375 * n
        assert (df.groupby("session_id").size() == 375).all()
        assert df["session_id"].drop_duplicates().tolist() == list(range(n))
        assert df["minute"].min() == 0 and df["minute"].max() == 374
        assert str(df["datetime"].dtype).startswith("datetime64")
        assert df["datetime"].dt.second.eq(0).all()
    assert a["gap_before"].sum() == 0
    # BBB kept Jan 1, 2, 4, 8. Calendar (union with AAA) also has Jan 3 and Jan 5, so
    # Jan 4 (after missing Jan 3) and Jan 8 (after missing Jan 5, dropped as incomplete) have gaps.
    first = b[b["minute"] == 0].reset_index(drop=True)
    assert first["gap_before"].tolist() == [False, False, True, True]
    assert first["datetime"].dt.strftime("%Y-%m-%d").tolist() == \
        ["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-08"]

    m = man["stocks"]["BBB"]
    assert m["rows_read"] == len(rows_b)
    assert m["rows_dropped_nonpositive_price"] == 1
    assert m["rows_dropped_duplicate_minutes"] == 1
    assert m["rows_dropped_out_of_session"] == 1
    assert m["days_dropped_incomplete"] == 1
    assert m["rows_dropped_incomplete_days"] == 100
    assert m["sessions_kept"] == 4 and m["rows_kept"] == 4 * 375
    assert m["first_date"] == "2024-01-01" and m["last_date"] == "2024-01-08"
    assert m["ohlc_inconsistent_rows"] == 0
    assert m["gap_before_days"] == 2


def test_manifest_content_and_sector(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    _write_csv(src / "RELIANCE_minute.csv",
               [r for i, d in enumerate(["2024-01-01", "2024-01-02"]) for r in _day_rows(d, seed=i)])
    build(str(src), str(out))
    man = json.load(open(out / "manifest.json"))
    import hashlib
    assert man["pandas"] == pd.__version__ and "pyarrow" in man and "build_seconds" in man
    s = man["stocks"]["RELIANCE"]
    assert s["sector"] == "Energy"
    assert s["sha256"] == hashlib.sha256(open(out / "RELIANCE.parquet", "rb").read()).hexdigest()
    assert pd.read_parquet(out / "RELIANCE.parquet")["sector"].eq("Energy").all()
    assert not list(out.glob("*.pass1.parquet"))


@pytest.mark.skipif(not os.path.exists(REAL_RELIANCE), reason="real RELIANCE csv not present")
def test_builder_real_slice(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    pd.read_csv(REAL_RELIANCE, nrows=12000).to_csv(src / "RELIANCE_minute.csv", index=False)
    man = build(str(src), str(out), ["RELIANCE"])
    df = pd.read_parquet(out / "RELIANCE.parquet")
    n = man["stocks"]["RELIANCE"]["sessions_kept"]
    assert 25 <= n <= 33
    assert len(df) == 375 * n
    assert (df.groupby("session_id").size() == 375).all()
    assert df["session_id"].drop_duplicates().tolist() == list(range(n))
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert df["datetime"].is_monotonic_increasing
    assert df.loc[df["minute"] == 0, "datetime"].dt.strftime("%H:%M").eq("09:15").all()
