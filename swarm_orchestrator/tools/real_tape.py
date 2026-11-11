"""The real market behind the chart: the chart's eye button (faded real candles, for comparison).

Once the agents (or the user's orders) move the price, the simulated chart drifts from the real data. To draw the
real candles in the same places as the simulated ones, the n-th market minute that runs is paired with the n-th bar
of the replayed data, at the simulated minute it ran: in REPLAY that is the bar's own time; after a hand-over the
engine's clock goes its own way (nights, weekends), so pairing by the clock's own minutes keeps the two charts
side by side. In Free market mode the real data stops at the hand-over, but its later bars are still paired with
the minutes the agents run, so the comparison carries on. Only minutes that have run are given out: never the
future.

It also keeps the SIMULATED one-minute candles of the run, as the ticks published them: at Skip all speed a
dashboard can miss ticks (and a reloaded page has none of the run so far), so it fills its chart from these.

Control message (bridge: GET /api/replay/real?since=N&sim_since=M):
    {"msg": "REAL_TAPE", "since": N, "sim_since": M} -> {"msg": "ACK", "status": "OK", "tape": bool, "count": int,
        "run_id", "bars": [{"time", "o", "h", "l", "c", "v"}, ...] (real bars N.. that have run),
        "sim_count": int, "sim": [{"time", "o", "h", "l", "c", "v"}, ...] (simulated candles M..)}

engine/matching_engine.py is not modified: MatchingEngine.physics_tick and handle_message are wrapped (idempotent).
"""
from __future__ import annotations

import functools

from engine.matching_engine import MatchingEngine


def _state(engine):
    """The pairing for the engine's current replay (a new replay starts a new one); None without a replay."""
    rp = engine.replayer
    if rp is None or not getattr(rp, "bars", None):
        return None
    st = engine.__dict__.get("_rt")
    if st is None or st["rp"] is not rp:
        st = engine.__dict__["_rt"] = {"rp": rp, "times": [], "last": None}
    return st


SIM_KEEP = 20_000                      # simulated candles kept per run (about 53 trading days)


def _record_sim(engine, pubs) -> None:
    """The run's simulated one-minute candles, exactly as the last tick of this physics tick published them."""
    tick = next((p for t, p in reversed(pubs or []) if t == "tick"), None)
    if not tick or not tick.get("bar") or not tick.get("run_id") or not isinstance(tick.get("unix_time"), (int, float)):
        return
    sim = engine.__dict__.get("_rt_sim")
    if sim is None or sim["run"] != tick["run_id"]:
        sim = engine.__dict__["_rt_sim"] = {"run": tick["run_id"], "bars": []}
    t = int(tick["unix_time"])
    b = tick["bar"]
    row = {"time": t - t % 60, "o": b.get("o"), "h": b.get("h"), "l": b.get("l"), "c": b.get("c"), "v": b.get("v", 0)}
    bars = sim["bars"]
    if bars and bars[-1]["time"] == row["time"]:
        bars[-1] = row
    elif not bars or row["time"] > bars[-1]["time"]:
        bars.append(row)
        if len(bars) > SIM_KEEP:
            del bars[: len(bars) - SIM_KEEP]


def _record(engine) -> None:
    st = _state(engine)
    if st is None or engine.paused:
        return
    t = engine.sim_unix_time
    if not isinstance(t, (int, float)):
        return
    minute = int(t) - int(t) % 60
    if minute != st["last"] and len(st["times"]) < len(st["rp"].bars):
        st["times"].append(minute)
        st["last"] = minute


def _handle(engine, msg: dict) -> dict:
    st = _state(engine)
    out = {"msg": "ACK", "status": "OK", "tape": st is not None, "count": 0, "bars": [],
           "run_id": getattr(engine, "run_id", None)}
    if st is not None:
        try:
            since = max(0, int(msg.get("since") or 0))
        except (TypeError, ValueError):
            since = 0
        times, bars = st["times"], st["rp"].bars
        out["count"] = len(times)
        out["bars"] = [{"time": times[i], "o": bars[i]["o"], "h": bars[i]["h"], "l": bars[i]["l"],
                        "c": bars[i]["c"], "v": bars[i].get("v", 0)} for i in range(since, len(times))]
    sim = engine.__dict__.get("_rt_sim")
    if sim is not None and sim["run"] == getattr(engine, "run_id", None):
        try:
            sim_since = max(0, int(msg.get("sim_since") or 0))
        except (TypeError, ValueError):
            sim_since = 0
        out["sim_count"] = len(sim["bars"])
        out["sim"] = sim["bars"][sim_since:]
    else:
        out["sim_count"], out["sim"] = 0, []
    out["seq"] = engine._next_seq()
    return out


def install() -> None:
    pt = MatchingEngine.physics_tick
    if not getattr(pt, "_rt_wrapped", False):
        @functools.wraps(pt)
        def physics_tick(self):
            pubs = pt(self)
            _record(self)
            _record_sim(self, pubs)
            return pubs
        physics_tick._rt_wrapped = True
        MatchingEngine.physics_tick = physics_tick

    hm = MatchingEngine.handle_message
    if not getattr(hm, "_rt_wrapped", False):
        @functools.wraps(hm)
        def handle_message(self, msg):
            if isinstance(msg, dict) and msg.get("msg") == "REAL_TAPE":
                return _handle(self, msg), []
            return hm(self, msg)
        handle_message._rt_wrapped = True
        MatchingEngine.handle_message = handle_message
