"""Phase 5: deterministic REPLAY -> REACTIVE hand-over (HANDOVER, PROTOCOL §9)."""
import csv
import os

import pytest

from engine.matching_engine import MatchingEngine

N_BARS, AT = 40, 25


def _tape(tmp_path):
    path = os.path.join(str(tmp_path), "tape.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["datetime", "open", "high", "low", "close", "volume"])
        px = 1350.0
        for i in range(N_BARS):
            o, c = px, round(px + (0.3 if i % 3 else -0.2), 1)
            w.writerow([f"2026-03-02 {9 + (15 + i) // 60:02d}:{(15 + i) % 60:02d}:00",
                        o, max(o, c) + 0.2, min(o, c) - 0.1, c, 1000 + 37 * i])
            px = c
    return path


def _engine(path):
    e = MatchingEngine(db=None)
    e.start_replay(path, "RELIANCE", "ENERGY", seed=5)
    return e


def _tick(e):
    pubs = e.physics_tick()
    return pubs, [p for t, p in pubs if t == "tick"][-1]


def _state(e):
    return (e.rng.getstate(), e.regime, e.last_price, e.book.depth(50),
            {k: (v["cash"], v["pos"]) for k, v in e.agent_ledger.items()})


def test_switches_after_at_bar_with_no_trade_and_history_is_untouched(tmp_path):
    path = _tape(tmp_path)
    plain, ctrl = _engine(path), _engine(path)
    ack, _ = ctrl.handle_message({"msg": "HANDOVER", "at_bar": AT})
    assert ack["status"] == "HANDOVER_ARMED" and ack["at_bar"] == AT
    for i in range(AT):
        _, a = _tick(plain)
        pubs, b = _tick(ctrl)
        assert a["bar"] == b["bar"] and a["mode"] == b["mode"] == "REPLAY", i
    # Same state as a plain replay of AT bars at the moment of the switch.
    assert _state(plain) == _state(ctrl)
    pubs, p = _tick(ctrl)
    assert p["mode"] == "REACTIVE" and ctrl.replayer.cursor == AT
    ev = [x for t, x in pubs if t == "event" and x["kind"] == "mode_change"]
    assert ev == [{"seq": ev[0]["seq"], "kind": "mode_change", "mode": "REACTIVE",
                   "trigger_agent": "HANDOVER", "bars_replayed": AT,
                   "bars_remaining": N_BARS - AT}]
    for _ in range(5):                          # the tape no longer drives the price
        _tick(ctrl)
    assert ctrl.replayer.cursor == AT and not ctrl.paused


def test_control_run_matches_user_trade_run_except_for_the_user(tmp_path):
    """Treatment: a user trade after bar AT latches. Control: HANDOVER at AT.
    Both switch before bar AT+1; the RNG stream and regime stay in step, so
    the only difference left is the user's own order."""
    path = _tape(tmp_path)
    treat, ctrl = _engine(path), _engine(path)
    ctrl.handle_message({"msg": "HANDOVER", "at_bar": AT})
    for _ in range(AT):
        _tick(treat), _tick(ctrl)
    ack, pubs = treat.handle_message({"msg": "ORDER", "agent_id": "USER_1", "action": "BUY",
                                      "type": "MARKET", "qty": 1})
    assert treat.mode == "REACTIVE" and ack["executed_qty"] == 1
    for _ in range(10):
        _, a = _tick(treat)
        _, b = _tick(ctrl)
        assert a["mode"] == b["mode"] == "REACTIVE"
        assert treat.rng.getstate() == ctrl.rng.getstate()
        assert treat.regime == ctrl.regime
        assert treat.replayer.cursor == ctrl.replayer.cursor == AT


def test_handover_at_tape_end_switches_instead_of_pausing(tmp_path):
    e = _engine(_tape(tmp_path))
    e.handle_message({"msg": "HANDOVER", "at_bar": N_BARS})
    for _ in range(N_BARS + 3):
        _, p = _tick(e)
    assert p["mode"] == "REACTIVE" and not e.paused


def test_handover_now_and_rearm(tmp_path):
    e = _engine(_tape(tmp_path))
    for _ in range(10):
        _tick(e)
    assert e.handle_message({"msg": "HANDOVER", "at_bar": 30})[0]["status"] == "HANDOVER_ARMED"
    ack, _ = e.handle_message({"msg": "HANDOVER"})          # re-arm: "now" = bars so far
    assert ack["at_bar"] == 10
    _, p = _tick(e)
    assert p["mode"] == "REACTIVE" and e.replayer.cursor == 10


@pytest.mark.parametrize("msg,why", [
    ({"msg": "HANDOVER", "at_bar": 3}, "already replayed"),
    ({"msg": "HANDOVER", "at_bar": N_BARS + 1}, "beyond the tape"),
    ({"msg": "HANDOVER", "at_bar": "7"}, "must be an int"),
    ({"msg": "HANDOVER", "at_bar": True}, "must be an int"),
])
def test_rejections(tmp_path, msg, why):
    e = _engine(_tape(tmp_path))
    for _ in range(5):
        _tick(e)
    ack, _ = e.handle_message(msg)
    assert ack["status"] == "REJECTED" and why in ack["reason"]


def test_rejected_outside_replay():
    e = MatchingEngine(db=None)
    e.init_sim("RELIANCE", "SIM", 1350.0, seed=1)
    ack, _ = e.handle_message({"msg": "HANDOVER", "at_bar": 0})
    assert ack["status"] == "REJECTED" and "REPLAY mode" in ack["reason"]


def test_same_seed_same_handover_identical_with_swarm(tmp_path):
    from validation.sim_driver import Simulation
    path = _tape(tmp_path)

    def run():
        sim = Simulation(9, 1.0, symbol="RELIANCE", hash_events=True, replay_csv=path)
        sim.engine.handle_message({"msg": "HANDOVER", "at_bar": AT})
        for _ in range(AT + 30):
            sim.step()
        return sim.event_hash(), sim.engine.mode
    (h1, m1), (h2, m2) = run(), run()
    assert h1 == h2 and m1 == m2 == "REACTIVE"


def test_bridge_endpoint_forwards_and_maps_rejection(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import bridge
    sent = []

    async def fake(msg, timeout=None):
        sent.append(msg)
        if msg.get("at_bar") == 99:
            return {"msg": "ACK", "status": "REJECTED", "reason": "at_bar 99 beyond the tape (40 bars)"}
        return {"msg": "ACK", "status": "HANDOVER_ARMED", "at_bar": msg.get("at_bar", 4), "bars_replayed": 4}
    monkeypatch.setattr(bridge, "_control_request", fake)
    client = TestClient(bridge.app)                      # no lifespan: no sockets opened
    r = client.post("/api/replay/handover", json={"at_bar": 25})
    assert r.status_code == 200 and r.json() == {"status": "HANDOVER_ARMED", "at_bar": 25, "bars_replayed": 4}
    assert client.post("/api/replay/handover", json={}).json()["at_bar"] == 4
    assert sent[1] == {"msg": "HANDOVER"}
    r = client.post("/api/replay/handover", json={"at_bar": 99})
    assert r.status_code == 409 and "beyond the tape" in r.json()["detail"]
