"""Replay fidelity: the replayed tape reproduces every OHLCV bar exactly."""
import csv
import os

import pytest

from engine.matching_engine import MatchingEngine

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(HERE, "data", "real", "nse_kaggle", "RELIANCE_minute.csv")

# (o, h, l, c, v) on the ₹0.10 grid (₹1,000-5,000 band).
TAPE = [
    (1350.0, 1351.0, 1349.5, 1350.5, 120_000),   # opening bar, far above any old cap
    (1352.0, 1352.4, 1351.8, 1352.0, 9_000),     # gap open: o != previous close
    (1352.0, 1352.0, 1350.0, 1350.3, 4_001),     # o == h (legs merge), odd volume
    (1350.3, 1350.3, 1350.3, 1350.3, 0),         # no trades: flat at previous close
    (1349.0, 1349.0, 1349.0, 1349.0, 0),         # zero volume yet moved: impossible
    (1349.0, 1349.6, 1348.9, 1349.2, 2),         # needs 4 prints either way: impossible
    (1349.2, 1349.3, 1349.1, 1349.3, 3),         # 3 prints only low-first (o, l, h=c)
    (1349.3, 1349.4, 1349.0, 1349.4, 3),         # low-first again: o, l, h=c
]


def _write(tmp_path, bars):
    path = os.path.join(tmp_path, "tape.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["datetime", "open", "high", "low", "close", "volume"])
        for i, (o, h, l, c, v) in enumerate(bars):
            w.writerow([f"2026-03-02 09:{15 + i:02d}:00", o, h, l, c, v])
    return path


def _replay(path, n):
    e = MatchingEngine(db=None)
    e.start_replay(path, "RELIANCE", "ENERGY", seed=1)
    out = []
    for _ in range(n):
        p = [x for t, x in e.physics_tick() if t == "tick"][-1]
        b = p["bar"]
        out.append((b["o"], b["h"], b["l"], b["c"], b["v"]))
    return e, out


def test_every_bar_reproduced_and_impossible_bars_keep_prices(tmp_path):
    e, got = _replay(_write(str(tmp_path), TAPE), len(TAPE))
    for i, (want, have) in enumerate(zip(TAPE, got)):
        assert have[:4] == pytest.approx(want[:4], abs=1e-9), (i, want, have)
    exact = [0, 1, 2, 3, 6, 7]
    for i in exact:
        assert got[i][4] == TAPE[i][4], (i, TAPE[i], got[i])
    # Bar 5 (index 4): 1 share prints the moved price; bar 6: 4 shares for 4 prices.
    assert got[4][4] == 1 and got[5][4] == 4
    assert e.replayer.inexact_bars == [(5, 0, 1), (6, 2, 4)]


def test_open_is_first_trade_in_live_mode_too():
    e = MatchingEngine(db=None)
    e.init_sim("RELIANCE", "SIM", 1350.0, seed=1)
    e.physics_tick()
    # Cross SIM_LP's ask: the bar's first print, not the previous close, is the open.
    ack, _ = e.handle_message({"msg": "ORDER", "agent_id": "STRAT_T", "action": "BUY",
                               "type": "MARKET", "qty": 50})
    px = ack["average_price"]
    bar = [x for t, x in e.physics_tick() if t == "tick"][-1]["bar"]
    assert bar["o"] == pytest.approx(px) and bar["l"] >= bar["o"] - 1e-9
    assert bar["o"] != 1350.0


@pytest.mark.skipif(not os.path.exists(REAL), reason="real NSE data not downloaded")
def test_real_reliance_week_replays_exactly(tmp_path):
    pd = pytest.importorskip("pandas")
    d = pd.read_csv(REAL)
    d = d[(d.date >= "2026-03-02") & (d.date < "2026-03-07")]
    d = d[["date", "open", "high", "low", "close", "volume"]].rename(columns={"date": "datetime"})
    path = os.path.join(str(tmp_path), "week.csv")
    d.to_csv(path, index=False)
    want = list(d[["open", "high", "low", "close", "volume"]].itertuples(index=False, name=None))
    e, got = _replay(path, len(want))
    bad = [(i, w, g) for i, (w, g) in enumerate(zip(want, got))
           if round(w[0] * 100) != round(g[0] * 100) or round(w[1] * 100) != round(g[1] * 100)
           or round(w[2] * 100) != round(g[2] * 100) or round(w[3] * 100) != round(g[3] * 100)
           or int(w[4]) != int(g[4])]
    assert len(want) == 4 * 375 and not bad, bad[:3]    # 3 Mar 2026 was a holiday
