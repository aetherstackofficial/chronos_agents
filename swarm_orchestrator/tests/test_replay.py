"""Pillar-1 history-repeater tests — deterministic replay and the reactive latch (PROTOCOL §9)."""
import csv

import pytest

from engine.matching_engine import MatchingEngine


@pytest.fixture
def replay_csv(tmp_path):
    """A 30-bar synthetic 1-minute tape."""
    path = tmp_path / "TCS_1min.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "time", "open", "high", "low", "close", "volume"])
        price = 190.0
        for i in range(30):
            o = price
            c = round(price + (0.1 if i % 2 == 0 else -0.05), 2)
            h = max(o, c) + 0.05
            low = min(o, c) - 0.05
            w.writerow(["2026-01-02", f"09:{15 + i:02d}:00", o, h, low, c, 5000])
            price = c
    return str(path)


def test_replay_starts_in_replay_mode(replay_csv):
    e = MatchingEngine()
    run_id, n_bars = e.start_replay(replay_csv, "TCS", "TECH", seed=7)
    assert run_id
    assert n_bars == 30
    assert e.mode == "REPLAY"


def test_replay_is_deterministic_without_user_trade(replay_csv):
    """§9: same seed + same CSV + no user trade ⇒ identical price path."""
    def run():
        e = MatchingEngine()
        e.start_replay(replay_csv, "TCS", "TECH", seed=7)
        prices = []
        for _ in range(25):
            pubs = e.physics_tick()
            tick = [p for t, p in pubs if t == "tick"][0]
            prices.append(tick["last_price"])
        return prices

    assert run() == run()


def test_first_strat_fill_latches_to_reactive(replay_csv):
    """§9: the first fill involving a STRAT_ agent flips mode to REACTIVE permanently."""
    e = MatchingEngine()
    e.start_replay(replay_csv, "TCS", "TECH", seed=7)
    e.physics_tick()  # build some book depth from the replayer
    assert e.mode == "REPLAY"

    # A strategy crosses the spread — this must trigger the latch.
    saw_mode_change = False
    for _ in range(5):
        ack, pubs = e.handle_message(
            {"msg": "ORDER", "agent_id": "STRAT_test01", "action": "BUY",
             "type": "MARKET", "qty": 100})
        for topic, payload in pubs:
            if topic == "event" and payload.get("kind") == "mode_change":
                saw_mode_change = True
                assert payload.get("mode") == "REACTIVE"
                assert payload.get("trigger_agent") == "STRAT_test01"
        if e.mode == "REACTIVE":
            break
        e.physics_tick()

    assert e.mode == "REACTIVE"
    assert saw_mode_change


def test_no_price_gap_across_latch(replay_csv):
    """The replay→reactive hand-off must be seamless (no discontinuity in last_price)."""
    e = MatchingEngine()
    e.start_replay(replay_csv, "TCS", "TECH", seed=7)
    for _ in range(3):
        e.physics_tick()
    price_before = e.last_price
    e.handle_message({"msg": "ORDER", "agent_id": "STRAT_x", "action": "BUY",
                      "type": "MARKET", "qty": 50})
    # The latch itself must not teleport the price beyond a real fill move.
    assert abs(e.last_price - price_before) < price_before * 0.05
