"""start_replay_from_bars must be indistinguishable from start_replay(csv)."""
import os

import pandas as pd
import pytest

from engine.matching_engine import MatchingEngine
from engine.replay import HistoricalReplayer
from training.data.build_dataset import build
from training.data.episode_bars import bars_for_sessions
from training.envs.replay_from_bars import start_replay_from_bars

REAL_RELIANCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "real", "nse_kaggle", "RELIANCE_minute.csv")
SEED = 7


@pytest.fixture(scope="module")
def reliance_df(tmp_path_factory):
    if not os.path.exists(REAL_RELIANCE):
        pytest.skip("real RELIANCE csv not present")
    base = tmp_path_factory.mktemp("rb")
    src, out = base / "src", base / "out"
    src.mkdir()
    pd.read_csv(REAL_RELIANCE, nrows=3000).to_csv(src / "RELIANCE_minute.csv", index=False)
    build(str(src), str(out), ["RELIANCE"])
    return pd.read_parquet(out / "RELIANCE.parquet")


def _write_csv(path, bars):
    ts = pd.to_datetime([b["ts"] for b in bars], unit="s")
    pd.DataFrame({"datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                  "open": [b["o"] for b in bars], "high": [b["h"] for b in bars],
                  "low": [b["l"] for b in bars], "close": [b["c"] for b in bars],
                  "volume": [b["v"] for b in bars]}).to_csv(path, index=False)


def _strip_run_id(pubs):
    out = []
    for topic, p in pubs:
        p = dict(p)
        p.pop("run_id", None)
        out.append((topic, p))
    return out


def test_bars_match_load_csv_format(reliance_df, tmp_path):
    bars = bars_for_sessions(reliance_df, 0, 1)
    assert len(bars) == 375 and bars[0]["day_count"] == 1 and bars[0]["minute"] == 0
    assert {"day_count", "minute", "o", "h", "l", "c", "v", "ts"} == set(bars[0])
    path = tmp_path / "t.csv"
    _write_csv(path, bars)
    loaded = HistoricalReplayer.load_csv(str(path))
    assert len(loaded) == len(bars)
    for lb, b in zip(loaded, bars):
        assert lb == {"o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"], "ts": b["ts"]}


def test_parity_with_start_replay_csv(reliance_df, tmp_path):
    bars = bars_for_sessions(reliance_df, 0, 3)
    assert len(bars) == 3 * 375
    path = tmp_path / "RELIANCE_3d.csv"
    _write_csv(path, bars)

    a = MatchingEngine(db=None)
    run_a, n_a = a.start_replay(str(path), "RELIANCE", "Energy", seed=SEED)
    b = MatchingEngine(db=None)
    run_b, n_b = start_replay_from_bars(b, bars, "RELIANCE", "Energy", seed=SEED)
    assert n_a == n_b == len(bars)
    assert a.mode == b.mode == "REPLAY"
    assert a.paused is False and b.paused is False
    assert a.sim_unix_time == b.sim_unix_time == bars[0]["ts"]

    n_steps = len(bars)
    ticks_a, ticks_b, pubs_a, pubs_b = [], [], [], []
    for _ in range(n_steps):
        pa, pb = a.physics_tick(), b.physics_tick()
        pubs_a.append(_strip_run_id(pa))
        pubs_b.append(_strip_run_id(pb))
        ticks_a.append([p for t, p in pa if t == "tick"][0])
        ticks_b.append([p for t, p in pb if t == "tick"][0])

    assert all(ta["run_id"] == run_a and tb["run_id"] == run_b for ta, tb in zip(ticks_a, ticks_b))
    for i, (ta, tb) in enumerate(zip(ticks_a, ticks_b)):
        ta, tb = dict(ta), dict(tb)
        ta.pop("run_id"), tb.pop("run_id")
        assert ta == tb, f"tick {i} differs"
    # every publication (trades, events, ...) matches too, not just the tick
    assert pubs_a == pubs_b
    assert ticks_a[-1]["day_count"] == 3          # engine crossed two session boundaries
    assert a.replayer.cursor == b.replayer.cursor == len(bars)


def test_symbol_none_keeps_engine_symbol_and_requires_bars():
    e = MatchingEngine(db=None)
    bars = [{"o": 100.0, "h": 100.5, "l": 99.5, "c": 100.2, "v": 100, "ts": 1767258900}]
    start_replay_from_bars(e, bars, None, "Energy", seed=1)
    assert e.symbol == "TCS" and e.sector == "Energy" and e.mode == "REPLAY"
    with pytest.raises(ValueError):
        start_replay_from_bars(e, [], "X", "Energy")
