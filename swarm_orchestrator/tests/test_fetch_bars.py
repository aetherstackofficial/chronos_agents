"""Late joiners: engine FETCH_BARS and MarketContext.seed (PROTOCOL §4)."""
import asyncio

import pytest

from core.loop_manager import SwarmOrchestrator
from core.market_context import MarketContext
from engine.matching_engine import MatchingEngine


def running_engine(ticks=20, seed=3):
    e = MatchingEngine()
    e.init_sim("TCS", "TECH", 190.0, seed=seed)
    e.paused = False
    payloads = []
    for _ in range(ticks):
        for topic, p in e.physics_tick():
            if topic == "tick":
                payloads.append(p)
    return e, payloads


def fetch(e, n=None):
    msg = {"msg": "FETCH_BARS"}
    if n is not None:
        msg["n"] = n
    return e.handle_message(msg)[0]


# --- engine ----------------------------------------------------------------------
def test_fetch_bars_returns_last_n_closed_bars_oldest_first():
    e, payloads = running_engine(20)
    ack = fetch(e, 5)
    assert ack["msg"] == "ACK" and ack["status"] == "BARS" and ack["n"] == 5
    assert ack["run_id"] == e.run_id
    bars = ack["bars"]
    assert [b["minute"] for b in bars] == [p["bar"]["minute"] for p in payloads[-5:]]
    assert {"day_count", "minute", "o", "h", "l", "c", "v"} <= set(bars[0])
    assert bars[-1]["c"] == payloads[-1]["bar"]["c"]


def test_fetch_bars_default_and_fewer_than_n_available():
    e, _ = running_engine(7)
    assert fetch(e)["n"] == 7                 # default 375, only 7 exist
    assert fetch(e, 1000)["n"] == 7


def test_fetch_bars_rejects_bad_n():
    e, _ = running_engine(3)
    for bad in (0, -1, 1001, "5", 2.5, True, None):
        msg = {"msg": "FETCH_BARS", "n": bad}
        assert e.handle_message(msg)[0]["status"] == "REJECTED"


def test_fetch_bars_works_while_paused_and_is_read_only():
    e, _ = running_engine(5)
    e.paused = True
    before = (e.tick_count, e.market_minute, len(e._closed_bars))
    assert fetch(e, 3)["status"] == "BARS"
    assert (e.tick_count, e.market_minute, len(e._closed_bars)) == before


def test_closed_bars_are_bounded():
    e = MatchingEngine()
    e.init_sim("TCS", "TECH", 190.0, seed=1)
    e.paused = False
    for _ in range(e.CLOSED_BARS_KEPT + 50):
        e.physics_tick()
    assert len(e._closed_bars) == e.CLOSED_BARS_KEPT
    assert fetch(e, 1000)["n"] == 1000


def test_new_run_clears_the_history():
    e, _ = running_engine(10)
    e.init_sim("TCS", "TECH", 190.0, seed=4)
    assert fetch(e, 5)["n"] == 0


# --- MarketContext.seed ------------------------------------------------------------
def test_seed_equals_having_watched_live():
    e, payloads = running_engine(150)
    live = MarketContext(min_bars=100)
    for p in payloads:
        live.update(p)
    late = MarketContext(min_bars=100)
    assert late.seed(fetch(e, 1000)["bars"], e.run_id) == 150
    for attr in ("bars_seen", "ready", "day_open", "day_high", "day_low", "day_close",
                 "prev_day", "days"):
        assert getattr(late, attr) == getattr(live, attr), attr
    assert late.swing_highs() == live.swing_highs() and late.swing_lows() == live.swing_lows()
    assert late.ret(30) == live.ret(30)


def test_seeded_context_continues_with_live_ticks_without_double_counting():
    e, payloads = running_engine(40)
    late = MarketContext()
    late.seed(fetch(e, 1000)["bars"], e.run_id)
    assert late.update(payloads[-1]) is False          # the bar it was seeded with
    assert late.bars_seen == 40
    for topic, p in e.physics_tick():
        if topic == "tick":
            assert late.update(p) is True              # same run_id: continues, no reset
    assert late.bars_seen == 41


def test_seed_replaces_old_state_and_skips_bad_bars():
    ctx = MarketContext()
    ctx.update({"run_id": "x", "paused": False, "day_count": 1,
                "bar": {"minute": 1, "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}})
    good = {"day_count": 1, "minute": 1, "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 5}
    n = ctx.seed([good, {"minute": 2}, {"day_count": 1, "minute": 3, "o": "x", "h": 1, "l": 1, "c": 1}, None])
    assert n == 1 and ctx.bars_seen == 1 and ctx.day_open == 10
    assert ctx.seed([]) == 0 and ctx.bars_seen == 0 and ctx.day_open is None


# --- swarm worker -------------------------------------------------------------------
class Client:
    def __init__(self, reply=None, exc=None, delay=0):
        self.reply, self.exc, self.delay, self.sent = reply, exc, delay, []

    async def control(self, msg):
        self.sent.append(msg)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.reply


def test_orchestrator_seeds_context_from_the_engine():
    e, _ = running_engine(30)
    client = Client(reply=fetch(e, 400))
    orch = SwarmOrchestrator(client, seed=1)
    assert asyncio.run(orch.seed_context()) == 30
    assert client.sent == [{"msg": "FETCH_BARS", "n": 400}]
    assert orch.context.bars_seen == 30


@pytest.mark.parametrize("client", [
    Client(exc=RuntimeError("engine down")),
    Client(reply={"status": "REJECTED"}),
    Client(reply=None),
    Client(reply={"status": "BARS", "bars": []}),
])
def test_orchestrator_seed_failure_leaves_an_empty_context(client):
    orch = SwarmOrchestrator(client, seed=1)
    assert asyncio.run(orch.seed_context()) == 0
    assert orch.context.bars_seen == 0


def test_orchestrator_seed_never_waits_longer_than_the_timeout():
    orch = SwarmOrchestrator(Client(reply={"status": "BARS", "bars": []}, delay=5), seed=1)
    assert asyncio.run(orch.seed_context(timeout=0.05)) == 0


# --- strategy SDK --------------------------------------------------------------------
def _runner():
    from runner.sdk import Strategy, _StrategyRunner
    return _StrategyRunner(Strategy(), "t", "STRAT_x", "127.0.0.1", 59993, 59994)


def test_sdk_seed_context_over_a_real_socket():
    """A tiny ROUTER stands in for the engine and answers FETCH_BARS."""
    import json
    import threading

    import zmq

    e, _ = running_engine(25)
    router_ctx = zmq.Context()
    router = router_ctx.socket(zmq.ROUTER)
    port = router.bind_to_random_port("tcp://127.0.0.1")

    def serve():
        if router.poll(3000):
            ident, payload = router.recv_multipart()
            assert json.loads(payload)["msg"] == "FETCH_BARS"
            router.send_multipart([ident, json.dumps(fetch(e, 400)).encode()])

    t = threading.Thread(target=serve)
    t.start()
    from runner.sdk import Strategy, _StrategyRunner
    runner = _StrategyRunner(Strategy(), "t", "STRAT_x", "127.0.0.1", port, 59994)
    try:
        assert runner._seed_context() == 25
        assert runner.context.bars_seen == 25
    finally:
        t.join()
        runner.sub.close(0)
        runner.dealer.close(0)
        router.close(0)
        router_ctx.term()


def test_sdk_seed_context_times_out_quietly():
    runner = _runner()                       # nothing is listening on the order port
    try:
        assert runner._seed_context(timeout_s=0.1) == 0
        assert runner.context.bars_seen == 0
    finally:
        runner.sub.close(0)
        runner.dealer.close(0)
