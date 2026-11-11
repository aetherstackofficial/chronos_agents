"""Fast-forward for the live engine: the chart's "20x" button.

engine/server.py ticks every ``TICK_INTERVAL_S`` seconds (1.0) and reads that module constant again on every
loop, so the speed can change without editing the server: ``install(server)`` keeps a reference to the server
module and the SPEED control message sets ``server.TICK_INTERVAL_S = 1 / factor`` (and, while fast, a shorter
``server.POLL_TIMEOUT_MS`` so the loop does not overshoot each tick).

Two kinds, chosen by ``until``:
  * "trade" (the 20x button): stops by itself, back to 1x, at the first fill in which a user strategy takes part
    (an agent id starting STRAT_ or USER_), so the user sees their trade at normal speed;
  * "data_end" (the Skip all button): does NOT stop at the user's trades; it runs as many more market minutes as
    the replayed data has left (len(bars) - cursor, counted only while the market runs), then PAUSES the market
    and goes back to 1x, so the user can review every trade at once. In Free market mode the real data stops at
    the hand-over, so this is "as long as the data would have lasted".
Both stop when a new run starts.

Control message (sent by the bridge, POST/GET /api/speed):
    {"msg": "SPEED"}                                          -> status only
    {"msg": "SPEED", "action": "fast_forward", "factor": 20}  -> run `factor` times faster until a user fill
    {"msg": "SPEED", "action": "fast_forward", "until": "data_end"}  -> top speed to the end of the data, then pause
    {"msg": "SPEED", "action": "normal"}                      -> back to 1x
ACK: {"msg": "ACK", "status": "OK"|"REJECTED", "fast_forward": bool, "factor": int, "until": str,
      "minutes_left": int, "last_stop": str|None, "seq"}.
Every change is also published as an event {"kind": "speed", ...} so every open dashboard updates.

engine/matching_engine.py is not modified: MatchingEngine.handle_message, _apply_fill, physics_tick,
start_replay and init_sim are wrapped (idempotent; tests restore them).
"""
from __future__ import annotations

import functools
import logging
import time as _time
from types import ModuleType
from typing import Optional

from engine.matching_engine import MatchingEngine

logger = logging.getLogger("tools.sim_speed")

NORMAL_INTERVAL_S = 1.0
DEFAULT_FACTOR = 20
MAX_FACTOR = 50                         # the 20x button's range (2..50)
SKIP_FACTOR = 1000                      # Skip all: as fast as the engine can tick (a minute every 1 ms at most)
USER_PREFIXES = ("STRAT_", "USER_")
ACTIONS = ("fast_forward", "normal")
UNTIL = ("trade", "data_end")

_server: Optional[ModuleType] = None    # the engine.server module whose TICK_INTERVAL_S is set
_normal_poll_ms: Optional[int] = None   # the server's own POLL_TIMEOUT_MS (50), restored at 1x
FAST_POLL_MS = 5                        # while fast: wait at most 5 ms for messages (1 ms at Skip all speed)


class _HiResTime:
    """The time module, with monotonic() at full resolution. On Windows time.monotonic() moves in ~15.6 ms steps,
    so the server loop (which schedules ticks with it) could not tick more than ~64 times a second however short
    TICK_INTERVAL_S was: that capped Skip all at ~60x. perf_counter is monotonic too, at microsecond resolution."""
    monotonic = staticmethod(_time.perf_counter)

    def __getattr__(self, name):
        return getattr(_time, name)


QUEUE_HWM = 200_000                     # messages a ZMQ socket may queue (the default is 1000)


def _big_queues(server: ModuleType) -> None:
    """Skip all sends thousands of prints a second; with ZMQ's default 1000-message queues the PUB socket dropped
    ticks and trades whenever a subscriber (the bridge) fell behind for a moment: gaps in the chart, and fills
    missing from Results. Give the server's sockets deep queues instead (engine/server.py is not modified: its
    zmq.Context is swapped for one whose new sockets get QUEUE_HWM)."""
    import zmq
    if getattr(getattr(server, "zmq", None), "_big_queues", False):
        return

    class _Context(zmq.Context):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.setsockopt(zmq.SNDHWM, QUEUE_HWM)
            self.setsockopt(zmq.RCVHWM, QUEUE_HWM)

    class _Zmq:
        _big_queues = True
        Context = _Context

        def __getattr__(self, name):
            return getattr(zmq, name)

    if getattr(server, "zmq", None) is zmq:
        server.zmq = _Zmq()


def _state(engine) -> dict:
    return engine.__dict__.setdefault("_ff", {"on": False, "factor": 1, "until": "trade", "left": 0,
                                              "last_stop": None})


def _set_factor(factor: int) -> None:
    """The server loop waits up to POLL_TIMEOUT_MS (50 ms) for messages before it checks the tick clock; at 20x
    a tick is due every 50 ms, so that wait alone made it ~16x. Shorten it while fast, restore it at 1x."""
    if _server is None:
        return
    _server.TICK_INTERVAL_S = NORMAL_INTERVAL_S / max(1, factor)
    if _normal_poll_ms is not None:
        tick_ms = 1000.0 / max(1, factor)
        if factor <= 1:
            _server.POLL_TIMEOUT_MS = _normal_poll_ms
        elif tick_ms < 4:
            # Windows sleeps at least ~15 ms for any wait > 0 (that capped Skip all at ~60x): don't wait at all;
            # the loop spins one core for the few seconds a skip takes
            _server.POLL_TIMEOUT_MS = 0
        else:
            _server.POLL_TIMEOUT_MS = max(1, min(_normal_poll_ms, FAST_POLL_MS, int(tick_ms / 4)))


def status(engine) -> dict:
    st = _state(engine)
    return {"fast_forward": st["on"], "factor": st["factor"] if st["on"] else 1,
            "until": st["until"] if st["on"] else None,
            "minutes_left": st["left"] if st["on"] and st["until"] == "data_end" else None,
            "last_stop": st["last_stop"]}


def _event(engine, action: str, **extra) -> tuple:
    return ("event", {"seq": engine._next_seq(), "kind": "speed", "action": action,
                      **status(engine), **extra})


def _normal(engine, reason: str) -> bool:
    """Back to 1x. Returns True if fast-forward was on."""
    st = _state(engine)
    if not st["on"]:
        return False
    st.update(on=False, factor=1, until="trade", left=0, last_stop=reason)
    _set_factor(1)
    return True


def _reject(engine, reason: str):
    return ({"msg": "ACK", "status": "REJECTED", "reason": reason, **status(engine), "seq": engine._next_seq()}, [])


def _handle_speed(engine, msg: dict):
    action = msg.get("action")
    if action is not None and action not in ACTIONS:
        return ({"msg": "ACK", "status": "REJECTED", "reason": f"action must be one of {ACTIONS}",
                 **status(engine), "seq": engine._next_seq()}, [])
    pubs = []
    if action == "fast_forward":
        until = msg.get("until") or "trade"
        if until not in UNTIL:
            return _reject(engine, f"until must be one of {UNTIL}")
        try:
            factor = int(msg.get("factor", DEFAULT_FACTOR if until == "trade" else SKIP_FACTOR))
        except (TypeError, ValueError):
            factor = 0
        top = MAX_FACTOR if until == "trade" else SKIP_FACTOR
        if not 2 <= factor <= top:
            return _reject(engine, f"factor must be 2..{top}")
        left = 0
        if until == "data_end":
            rp = engine.replayer
            if rp is None or not getattr(rp, "bars", None):
                return _reject(engine, "this run has no real data to skip to the end of")
            left = len(rp.bars) - int(rp.cursor)
            if left <= 0:
                return _reject(engine, "the data has already run out")
        st = _state(engine)
        st.update(on=True, factor=factor, until=until, left=left, last_stop=None)
        _set_factor(factor)
        logger.info("fast-forward %dx %s", factor,
                    "until the next user trade" if until == "trade" else f"for the last {left} minutes of data")
        pubs.append(_event(engine, "fast_forward"))
    elif action == "normal":
        if _normal(engine, "stopped from the dashboard"):
            logger.info("fast-forward stopped from the dashboard")
            pubs.append(_event(engine, "normal", reason="stopped from the dashboard"))
    return {"msg": "ACK", "status": "OK", **status(engine), "seq": engine._next_seq()}, pubs


def install(server: Optional[ModuleType] = None) -> None:
    """Wrap the engine (once) and remember the server module whose tick interval is changed."""
    global _server, _normal_poll_ms
    if server is not None:
        _server = server
        poll = getattr(server, "POLL_TIMEOUT_MS", None)
        _normal_poll_ms = poll if isinstance(poll, int) and poll > 0 else None
        if getattr(server, "time", None) is _time:
            server.time = _HiResTime()           # the server's tick clock at full resolution (see _HiResTime)
        _big_queues(server)                      # deep message queues (see _big_queues)

    hm = MatchingEngine.handle_message
    if not getattr(hm, "_ff_wrapped", False):
        @functools.wraps(hm)          # keeps the inner wrappers' markers (backtest_mode's _bt_wrapped)
        def handle_message(self, msg):
            if isinstance(msg, dict) and msg.get("msg") == "SPEED":
                return _handle_speed(self, msg)
            return hm(self, msg)
        handle_message._ff_wrapped = True
        MatchingEngine.handle_message = handle_message

    af = MatchingEngine._apply_fill
    if not getattr(af, "_ff_wrapped", False):
        @functools.wraps(af)
        def _apply_fill(self, fill, taker_side, pubs):
            out = af(self, fill, taker_side, pubs)
            ff = self.__dict__.get("_ff", {})
            if ff.get("on") and ff.get("until") == "trade":
                who = next((a for a in (fill.get("taker_id", ""), fill.get("maker_id", ""))
                            if str(a).startswith(USER_PREFIXES)), None)
                if who is not None:
                    reason = f"your trade ({who} {taker_side if who == fill.get('taker_id') else _other(taker_side)} " \
                             f"{fill.get('qty')} @ {fill.get('price')})"
                    _normal(self, reason)
                    logger.info("fast-forward stopped: %s", reason)
                    pubs.append(_event(self, "normal", reason=reason, agent_id=who))
            return out
        _apply_fill._ff_wrapped = True
        MatchingEngine._apply_fill = _apply_fill

    pt = MatchingEngine.physics_tick
    if not getattr(pt, "_ff_wrapped", False):
        @functools.wraps(pt)
        def physics_tick(self):
            ff = self.__dict__.get("_ff")
            counting = bool(ff and ff["on"] and ff["until"] == "data_end" and not self.paused)
            pubs = pt(self)
            if counting:
                ff["left"] -= 1
                if ff["left"] <= 0:
                    self.paused = True
                    _normal(self, "the data ended")
                    logger.info("skip all: the data ended; market paused for review")
                    pubs.append(_event(self, "normal", reason="the data ended", paused=True))
            return pubs
        physics_tick._ff_wrapped = True
        MatchingEngine.physics_tick = physics_tick

    for name in ("start_replay", "init_sim"):
        orig = getattr(MatchingEngine, name, None)
        if orig is None or getattr(orig, "_ff_wrapped", False):
            continue

        def make(orig):
            @functools.wraps(orig)
            def wrapped(self, *a, **kw):
                _normal(self, "a new run started")
                return orig(self, *a, **kw)
            wrapped._ff_wrapped = True
            return wrapped
        setattr(MatchingEngine, name, make(orig))


def _other(side: str) -> str:
    return "SELL" if side == "BUY" else "BUY"
