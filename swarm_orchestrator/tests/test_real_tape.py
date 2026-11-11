"""tools/real_tape.py: the real candles behind the chart, paired with the minutes that ran."""
import math

import numpy as np
import pytest

from agents import backtest_mode as btm
from engine.matching_engine import MatchingEngine
from engine.replay import HistoricalReplayer
from tools import real_tape
from training.envs.replay_from_bars import start_replay_from_bars

NAMES = ("physics_tick", "handle_message", "_refresh_sim_lp")


@pytest.fixture
def rt(monkeypatch):
    saved = {n: MatchingEngine.__dict__[n] for n in NAMES}
    saved_step, saved_inst = HistoricalReplayer.step, btm._INSTALLED
    monkeypatch.delenv("CHRONOS_MODE", raising=False)
    monkeypatch.delenv("CHRONOS_TAPE_END", raising=False)
    btm._INSTALLED = False
    btm.install_switchable()
    real_tape.install()
    yield
    for n, f in saved.items():
        setattr(MatchingEngine, n, f)
    HistoricalReplayer.step, btm._INSTALLED = saved_step, saved_inst


def tape(n, seed=1):
    rng = np.random.default_rng(seed)
    bars, o = [], 100.0
    for k in range(n):
        c = round(o * math.exp(rng.normal(0, 6e-4)), 2)
        bars.append({"o": o, "h": max(o, c) + 0.02, "l": min(o, c) - 0.02, "c": c, "v": 3000,
                     "ts": 1_759_999_980 + k * 60})   # whole minutes, like real data
        o = c
    return bars


def engine(bars):
    e = MatchingEngine(db=None)
    e.SESSION_MINUTES = 375
    start_replay_from_bars(e, bars, "SYN", "S", seed=5)
    return e


def real(e, since=0):
    ack, pubs = e.handle_message({"msg": "REAL_TAPE", "since": since})
    assert pubs == [] and ack["status"] == "OK"
    return ack


def test_only_minutes_that_ran_and_paused_ticks_do_not_count(rt):
    bars = tape(30)
    e = engine(bars)
    e.paused = True                            # a replay starts paused on the live server: nothing runs
    for _ in range(3):
        e.physics_tick()
    assert real(e)["count"] == 0
    e.paused = False
    for _ in range(10):
        e.physics_tick()
    ack = real(e)
    assert ack["tape"] and ack["count"] == 10
    assert [b["c"] for b in ack["bars"]] == [b["c"] for b in bars[:10]]          # never the future
    assert [b["time"] for b in ack["bars"]] == [b["ts"] for b in bars[:10]]      # in REPLAY: the bars' own times
    assert [b["c"] for b in real(e, since=7)["bars"]] == [b["c"] for b in bars[7:10]]


def test_after_a_free_market_hand_over_the_real_data_carries_on(rt):
    bars = tape(30)
    e = engine(bars)
    e.paused = False
    for _ in range(5):
        e.physics_tick()
    e.handle_message({"msg": "ORDER", "agent_id": "STRAT_T", "action": "BUY", "type": "MARKET", "qty": 500})
    assert e.mode == "REACTIVE" and e.replayer.cursor == 5          # free market: the tape stopped here
    for _ in range(8):
        e.physics_tick()
    ack = real(e)
    assert ack["count"] == 13 and [b["c"] for b in ack["bars"]] == [b["c"] for b in bars[:13]]
    times = [b["time"] for b in ack["bars"]]
    assert times == sorted(set(times))                                 # one real candle per simulated minute


def test_no_replay_no_real_data_and_a_new_replay_starts_afresh(rt):
    e = MatchingEngine(db=None)
    e.init_sim("TCS", "TECH", 1000.0, seed=3)
    e.paused = False
    e.physics_tick()
    ack = real(e)
    assert ack["tape"] is False and ack["count"] == 0 and ack["bars"] == []
    e2 = engine(tape(20))
    e2.paused = False
    for _ in range(4):
        e2.physics_tick()
    start_replay_from_bars(e2, tape(20, seed=2), "SYN", "S", seed=6)
    assert real(e2)["count"] == 0


def test_the_simulated_candles_are_kept_for_the_chart(rt):
    e = engine(tape(30))
    e.paused = False
    for _ in range(12):
        e.physics_tick()
    ack = real(e, since=10 ** 9)
    assert ack["bars"] == [] and ack["sim_count"] == 12
    times = [b["time"] for b in ack["sim"]]
    assert times == sorted(set(times)) and all(t % 60 == 0 for t in times)
    assert ack["sim"][-1]["c"] == e.last_price
    assert len(e.handle_message({"msg": "REAL_TAPE", "since": 10 ** 9, "sim_since": 10})[0]["sim"]) == 2
