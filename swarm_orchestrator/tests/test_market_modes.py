"""Run-time market-mode switches of agents/backtest_mode.py (the dashboard's "Market mode" card).

install_switchable() patches three class attributes (MatchingEngine._refresh_sim_lp, MatchingEngine.handle_message,
HistoricalReplayer.step); the ``sw`` fixture saves and restores them around every test. Synthetic dated tapes only.
"""
import math
import os

import numpy as np
import pytest

from agents import backtest_mode as btm
from engine.matching_engine import MatchingEngine
from engine.replay import HistoricalReplayer
from training.envs.replay_from_bars import start_replay_from_bars

M = 40
P0 = 100.0


@pytest.fixture
def sw(monkeypatch):
    saved = (MatchingEngine._refresh_sim_lp, MatchingEngine.handle_message, HistoricalReplayer.step,
             btm._INSTALLED)
    for k in list(os.environ):
        if k.startswith("CHRONOS_BT_") or k in ("CHRONOS_MODE", "CHRONOS_TAPE_END"):
            monkeypatch.delenv(k)
    btm._INSTALLED = False
    btm.install_switchable()
    yield monkeypatch
    (MatchingEngine._refresh_sim_lp, MatchingEngine.handle_message, HistoricalReplayer.step,
     btm._INSTALLED) = saved


def tape(n, seed=1, sigma=6e-4):
    rng = np.random.default_rng(seed)
    bars, o = [], P0
    for k in range(n):
        c = round(o * math.exp(rng.normal(0, sigma)), 2)
        h = round(max(o, c) + abs(rng.normal(0, 0.02)), 2)
        l = round(min(o, c) - abs(rng.normal(0, 0.02)), 2)
        day, minute = divmod(k, M)
        bars.append({"o": o, "h": h, "l": l, "c": c, "v": int(rng.integers(1500, 4000)),
                     "ts": 1_760_000_000 + day * 86400 + minute * 60})
        o = c
    return bars


def replay(bars, seed=5):
    e = MatchingEngine(db=None)
    e.SESSION_MINUTES = M
    start_replay_from_bars(e, bars, "SYN", "S", seed=seed)
    e.paused = False
    return e


def mode(e, **kw):
    ack, pubs = e.handle_message({"msg": "BT_MODE", **kw})
    assert pubs == []
    return ack


def buy(e, qty):
    ack, _ = e.handle_message({"msg": "ORDER", "agent_id": "STRAT_T", "action": "BUY", "type": "MARKET",
                               "qty": int(qty)})
    return ack


def test_defaults_are_the_normal_engine(sw):
    e = replay(tape(30))
    ack = mode(e)
    assert ack["status"] == "OK" and ack["mode"] == "free" and ack["on_tape_end"] == "pause"
    assert ack["engine_mode"] == "REPLAY" and not ack["locked"] and ack["tape_bars"] == 30
    for _ in range(10):
        e.physics_tick()
    buy(e, 500)
    assert e.mode == "REACTIVE"
    e.physics_tick()
    assert btm.get_state(e) is None                     # free market: backtest mode never ran
    assert e.replayer.cursor == 10                      # the tape stopped at the hand-over


def test_backtest_chosen_in_the_dashboard_keeps_the_tape_running(sw):
    bars = tape(30)
    e = replay(bars)
    assert mode(e, mode="backtest")["mode"] == "backtest"
    for _ in range(10):
        e.physics_tick()
    buy(e, 500)
    for _ in range(5):
        e.physics_tick()
    st = btm.get_state(e)
    assert st is not None and st["n_bars"] == 5 and e.replayer.cursor == 15


def test_mode_is_locked_after_the_hand_over_but_tape_end_policy_is_not(sw):
    e = replay(tape(30))
    for _ in range(5):
        e.physics_tick()
    buy(e, 500)
    ack = mode(e, mode="backtest")
    assert ack["status"] == "REJECTED" and "new replay" in ack["reason"]
    assert mode(e)["mode"] == "free" and mode(e)["locked"]
    assert mode(e, mode="free")["status"] == "OK"        # no change requested: fine
    assert mode(e, on_tape_end="continue")["on_tape_end"] == "continue"


@pytest.mark.parametrize("kw", [{"mode": "fast"}, {"on_tape_end": "stop"}])
def test_bad_values_are_rejected(sw, kw):
    e = replay(tape(10))
    ack = mode(e, **kw)
    assert ack["status"] == "REJECTED"
    assert mode(e)["mode"] == "free" and mode(e)["on_tape_end"] == "pause"


def test_plain_replay_pauses_at_the_end_by_default(sw):
    e = replay(tape(12))
    for _ in range(14):
        e.physics_tick()
    assert e.paused and e.mode == "REPLAY"


def test_plain_replay_hands_over_to_the_agents_at_the_end(sw):
    bars = tape(12)
    e = replay(bars)
    mode(e, on_tape_end="continue")
    events = []
    for _ in range(12):
        e.physics_tick()
    for _ in range(5):
        events += [p[1] for p in e.physics_tick() if p[0] == "event"]
    assert not e.paused and e.mode == "REACTIVE"
    assert any(ev.get("kind") == "mode_change" for ev in events)
    assert e.sim_unix_time == bars[-1]["ts"] + 5 * 60    # the file's clock carries on, minute by minute
    bids, asks = e.book.depth(5)
    assert bids and asks                                 # the liquidity provider keeps quoting
    assert mode(e)["tape_exhausted"] and mode(e)["locked"]


def test_backtest_replay_continues_with_the_agents_after_the_data(sw):
    bars = tape(20)
    e = replay(bars)
    mode(e, mode="backtest", on_tape_end="continue")
    for _ in range(5):
        e.physics_tick()
    buy(e, 500)
    for _ in range(15 + 6):
        e.physics_tick()
    assert not e.paused and e.mode == "REACTIVE"
    st = btm.get_state(e)
    assert st["exhausted"] and st["n_bars"] == 15
    bids, asks = e.book.depth(5)
    assert bids and asks
    assert mode(e)["tape_exhausted"]


def test_a_new_replay_does_not_report_the_old_runs_tape_end(sw):
    e = replay(tape(10))
    mode(e, mode="backtest", on_tape_end="continue")
    for _ in range(3):
        e.physics_tick()
    buy(e, 500)
    for _ in range(12):
        e.physics_tick()
    assert mode(e)["tape_exhausted"]
    start_replay_from_bars(e, tape(10, seed=2), "SYN", "S", seed=6)   # same engine, a new replay
    st = mode(e)
    assert st["engine_mode"] == "REPLAY" and not st["locked"] and st["bars_done"] == 0
    assert not st["tape_exhausted"]


def test_backtest_replay_pauses_after_the_data_when_asked(sw):
    e = replay(tape(20))
    mode(e, mode="backtest", on_tape_end="pause")
    for _ in range(5):
        e.physics_tick()
    buy(e, 500)
    for _ in range(18):
        e.physics_tick()
    assert e.paused and e.replayer.finished


def test_install_switchable_is_idempotent(sw):
    wrapped = (MatchingEngine._refresh_sim_lp, MatchingEngine.handle_message, HistoricalReplayer.step)
    btm.install_switchable()
    btm.install()
    assert (MatchingEngine._refresh_sim_lp, MatchingEngine.handle_message, HistoricalReplayer.step) == wrapped
