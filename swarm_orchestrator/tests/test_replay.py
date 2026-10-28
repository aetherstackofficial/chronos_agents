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


# --- replay shows the CSV's own dates (not the engine's 1 Jan 2026 clock) ----
def _epoch(y, mo, d, h, mi):
    import calendar
    return calendar.timegm((y, mo, d, h, mi, 0))


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return str(path)


def _tick(pubs):
    return next(p for topic, p in pubs if topic == "tick")


def test_replay_uses_csv_dates(tmp_path):
    path = _write_csv(tmp_path / "d.csv",
                      ["date", "time", "open", "high", "low", "close", "volume"],
                      [["2026-10-19", "09:15", 100, 100.5, 99.8, 100.2, 800],
                       ["2026-10-19", "09:16:00", 100.2, 100.3, 100, 100.1, 400],
                       ["2026-10-20", "09:15", 100.1, 100.4, 100, 100.3, 400]])
    e = MatchingEngine()
    e.start_replay(path, "TCS", "TECH", seed=1)
    assert e.sim_unix_time == _epoch(2026, 10, 19, 9, 15)       # before tick 1
    t1, t2, t3 = (_tick(e.physics_tick()) for _ in range(3))
    assert t1["unix_time"] == _epoch(2026, 10, 19, 9, 15)
    assert t2["unix_time"] == _epoch(2026, 10, 19, 9, 16)
    assert t3["unix_time"] == _epoch(2026, 10, 20, 9, 15)       # next trading day


def test_replay_datetime_column_with_offset_keeps_wall_clock(tmp_path):
    path = _write_csv(tmp_path / "dt.csv",
                      ["Datetime", "Open", "High", "Low", "Close", "Volume"],
                      [["2026-10-19 09:15:00+05:30", 100, 101, 99, 100.5, 10]])
    e = MatchingEngine()
    e.start_replay(path, "TCS", "TECH", seed=1)
    assert _tick(e.physics_tick())["unix_time"] == _epoch(2026, 10, 19, 9, 15)


def test_replay_with_unreadable_dates_still_loads_on_engine_clock(tmp_path):
    path = _write_csv(tmp_path / "bad.csv",
                      ["date", "time", "open", "high", "low", "close", "volume"],
                      [["yesterday", "morning", 100, 101, 99, 100.5, 10]])
    e = MatchingEngine()
    _, n = e.start_replay(path, "TCS", "TECH", seed=1)
    assert n == 1
    assert _tick(e.physics_tick())["unix_time"] == MatchingEngine.SIM_EPOCH + 60


def test_clock_continues_from_file_time_after_latch(replay_csv):
    e = MatchingEngine()
    e.start_replay(replay_csv, "TCS", "TECH", seed=7)
    for _ in range(3):
        e.physics_tick()
    last = e.sim_unix_time
    e.handle_message({"msg": "ORDER", "agent_id": "STRAT_me", "action": "BUY",
                      "type": "MARKET", "qty": 5})
    assert e.mode == "REACTIVE"
    assert _tick(e.physics_tick())["unix_time"] == last + 60    # no jump back to 2026-01-01
