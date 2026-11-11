"""tools/sim_speed.py: the chart's fast-forward button.

install() wraps MatchingEngine.handle_message, _apply_fill, start_replay and init_sim; the ``ff`` fixture saves
and restores them, and passes a stand-in server module whose TICK_INTERVAL_S is checked.
"""
import math
import types

import numpy as np

import pytest

from engine.matching_engine import MatchingEngine
from tools import sim_speed
from training.envs.replay_from_bars import start_replay_from_bars

NAMES = ("handle_message", "_apply_fill", "physics_tick", "start_replay", "init_sim")


@pytest.fixture
def ff():
    saved = {n: MatchingEngine.__dict__[n] for n in NAMES}
    saved_server = (sim_speed._server, sim_speed._normal_poll_ms)
    server = types.SimpleNamespace(TICK_INTERVAL_S=1.0, POLL_TIMEOUT_MS=50)
    sim_speed.install(server)
    yield server
    for n, f in saved.items():
        setattr(MatchingEngine, n, f)
    sim_speed._server, sim_speed._normal_poll_ms = saved_server


def live():
    e = MatchingEngine(db=None)
    e.init_sim("TCS", "TECH", 1000.0, seed=3)
    e.paused = False
    for _ in range(3):
        e.physics_tick()
    return e


def speed(e, **kw):
    ack, pubs = e.handle_message({"msg": "SPEED", **kw})
    return ack, pubs


def order(e, agent, action, qty):
    ack, pubs = e.handle_message({"msg": "ORDER", "agent_id": agent, "action": action, "type": "MARKET",
                                  "qty": qty})
    return ack, pubs


def test_status_is_normal_speed_by_default(ff):
    ack, pubs = speed(live())
    assert ack["status"] == "OK" and not ack["fast_forward"] and ack["factor"] == 1 and pubs == []
    assert ff.TICK_INTERVAL_S == 1.0


def test_fast_forward_sets_the_tick_interval_and_tells_the_dashboards(ff):
    e = live()
    ack, pubs = speed(e, action="fast_forward", factor=20)
    assert ack["fast_forward"] and ack["factor"] == 20
    assert ff.TICK_INTERVAL_S == pytest.approx(0.05) and ff.POLL_TIMEOUT_MS == 5
    assert [p[1]["kind"] for p in pubs] == ["speed"] and pubs[0][1]["action"] == "fast_forward"


def test_other_agents_trades_do_not_stop_it(ff):
    e = live()
    speed(e, action="fast_forward")
    order(e, "MM_TEST", "BUY", 5)
    for _ in range(5):
        e.physics_tick()
    assert speed(e)[0]["fast_forward"] and ff.TICK_INTERVAL_S == pytest.approx(1 / 20)


def test_the_users_trade_brings_it_back_to_normal(ff):
    e = live()
    speed(e, action="fast_forward", factor=20)
    ack, pubs = order(e, "STRAT_abcd1234", "BUY", 5)
    assert ack["status"] in ("FILLED", "PARTIAL")
    st = speed(e)[0]
    assert not st["fast_forward"] and "STRAT_abcd1234" in st["last_stop"]
    assert ff.TICK_INTERVAL_S == 1.0 and ff.POLL_TIMEOUT_MS == 50
    ev = [p[1] for p in pubs if p[0] == "event" and p[1].get("kind") == "speed"]
    assert len(ev) == 1 and ev[0]["action"] == "normal" and ev[0]["agent_id"] == "STRAT_abcd1234"
    trades = [p for p in pubs if p[0] == "trade"]
    assert trades and pubs.index(trades[0]) < pubs.index(next(p for p in pubs if p[1] is ev[0]))


def test_stop_button_and_new_run(ff):
    e = live()
    speed(e, action="fast_forward")
    ack, pubs = speed(e, action="normal")
    assert not ack["fast_forward"] and ff.TICK_INTERVAL_S == 1.0 and pubs[0][1]["action"] == "normal"
    assert speed(e, action="normal")[1] == []                     # already normal: nothing to tell
    speed(e, action="fast_forward", factor=10)
    e.init_sim("TCS", "TECH", 1000.0, seed=4)                       # a new run
    assert not speed(e)[0]["fast_forward"] and ff.TICK_INTERVAL_S == 1.0


@pytest.mark.parametrize("kw", [{"action": "rewind"}, {"action": "fast_forward", "factor": 1},
                                {"action": "fast_forward", "factor": 500}, {"action": "fast_forward", "factor": "x"}])
def test_bad_requests_are_rejected(ff, kw):
    e = live()
    ack, pubs = speed(e, **kw)
    assert ack["status"] == "REJECTED" and pubs == [] and ff.TICK_INTERVAL_S == 1.0


def test_install_is_idempotent(ff):
    wrapped = {n: MatchingEngine.__dict__[n] for n in NAMES}
    sim_speed.install()
    assert {n: MatchingEngine.__dict__[n] for n in NAMES} == wrapped


def replay_engine(n=30, seed=1):
    rng = np.random.default_rng(seed)
    bars, o = [], 100.0
    for k in range(n):
        c = round(o * math.exp(rng.normal(0, 6e-4)), 2)
        bars.append({"o": o, "h": max(o, c) + 0.02, "l": min(o, c) - 0.02, "c": c, "v": 3000,
                     "ts": 1_760_000_000 + k * 60})
        o = c
    e = MatchingEngine(db=None)
    e.SESSION_MINUTES = 375
    start_replay_from_bars(e, bars, "SYN", "S", seed=5)
    e.paused = False
    return e


def test_skip_all_runs_to_the_end_of_the_data_then_pauses(ff):
    e = replay_engine(30)
    for _ in range(5):
        e.physics_tick()
    ack, pubs = speed(e, action="fast_forward", until="data_end")
    assert ack["status"] == "OK" and ack["until"] == "data_end" and ack["factor"] == sim_speed.SKIP_FACTOR
    assert ack["minutes_left"] == 25 and ff.TICK_INTERVAL_S == pytest.approx(1 / sim_speed.SKIP_FACTOR)
    assert ff.POLL_TIMEOUT_MS == 0                            # no waiting at Skip all speed
    order(e, "STRAT_abcd1234", "BUY", 5)                      # the user's trade does not stop it
    assert speed(e)[0]["fast_forward"]
    events = []
    for _ in range(24):
        events += e.physics_tick()
    assert not e.paused and speed(e)[0]["minutes_left"] == 1
    events += e.physics_tick()
    assert e.paused and not speed(e)[0]["fast_forward"] and ff.TICK_INTERVAL_S == 1.0
    ev = [p[1] for p in events if p[0] == "event" and p[1].get("kind") == "speed"]
    assert ev[-1]["action"] == "normal" and ev[-1]["reason"] == "the data ended" and ev[-1]["paused"]


def test_skip_all_does_not_count_paused_minutes(ff):
    e = replay_engine(20)
    speed(e, action="fast_forward", until="data_end")
    e.paused = True
    for _ in range(30):
        e.physics_tick()
    assert speed(e)[0]["minutes_left"] == 20
    e.paused = False
    for _ in range(20):
        e.physics_tick()
    assert e.paused and not speed(e)[0]["fast_forward"]


def test_skip_all_needs_real_data(ff):
    ack, pubs = speed(live(), action="fast_forward", until="data_end")   # a simulated stock: no data
    assert ack["status"] == "REJECTED" and "no real data" in ack["reason"] and pubs == []
    ack, _ = speed(live(), action="fast_forward", until="tomorrow")
    assert ack["status"] == "REJECTED"


def test_the_server_clock_gets_full_resolution():
    import time
    saved = (sim_speed._server, sim_speed._normal_poll_ms)
    methods = {n: MatchingEngine.__dict__[n] for n in NAMES}
    server = types.SimpleNamespace(TICK_INTERVAL_S=1.0, POLL_TIMEOUT_MS=50, time=time)
    try:
        sim_speed.install(server)
        assert server.time.monotonic is time.perf_counter      # Windows' monotonic() steps 15.6 ms
        assert server.time.sleep is time.sleep                 # everything else is the real module
    finally:
        sim_speed._server, sim_speed._normal_poll_ms = saved
        for n, f in methods.items():
            setattr(MatchingEngine, n, f)


def test_the_server_sockets_get_deep_queues():
    import zmq
    saved = (sim_speed._server, sim_speed._normal_poll_ms)
    methods = {n: MatchingEngine.__dict__[n] for n in NAMES}
    server = types.SimpleNamespace(TICK_INTERVAL_S=1.0, POLL_TIMEOUT_MS=50, zmq=zmq)
    try:
        sim_speed.install(server)
        ctx = server.zmq.Context()
        sock = ctx.socket(server.zmq.PUB)                      # everything else is the real zmq module
        assert sock.getsockopt(zmq.SNDHWM) == sim_speed.QUEUE_HWM
        sock.close(0)
        ctx.term()
    finally:
        sim_speed._server, sim_speed._normal_poll_ms = saved
        for n, f in methods.items():
            setattr(MatchingEngine, n, f)
