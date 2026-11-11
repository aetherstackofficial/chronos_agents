"""Fast-forward for the live engine: the chart's "20x" button.

engine/server.py ticks every ``TICK_INTERVAL_S`` seconds (1.0) and reads that module constant again on every
loop, so the speed can change without editing the server: ``install(server)`` keeps a reference to the server
module and the SPEED control message sets ``server.TICK_INTERVAL_S = 1 / factor`` (and, while fast, a shorter
``server.POLL_TIMEOUT_MS`` so the loop does not overshoot each tick).

Fast-forward stops by itself, back to 1x, at the first fill in which a user strategy takes part (an agent id
starting STRAT_ or USER_), so the user sees their trade at normal speed. It also stops when a new run starts.

Control message (sent by the bridge, POST/GET /api/speed):
    {"msg": "SPEED"}                                          -> status only
    {"msg": "SPEED", "action": "fast_forward", "factor": 20}  -> run `factor` times faster until a user fill
    {"msg": "SPEED", "action": "normal"}                      -> back to 1x
ACK: {"msg": "ACK", "status": "OK"|"REJECTED", "fast_forward": bool, "factor": int, "last_stop": str|None, "seq"}.
Every change is also published as an event {"kind": "speed", ...} so every open dashboard updates.

engine/matching_engine.py is not modified: MatchingEngine.handle_message, _apply_fill, start_replay and
init_sim are wrapped (idempotent; tests restore them).
"""
from __future__ import annotations

import functools
import logging
from types import ModuleType
from typing import Optional

from engine.matching_engine import MatchingEngine

logger = logging.getLogger("tools.sim_speed")

NORMAL_INTERVAL_S = 1.0
DEFAULT_FACTOR = 20
MAX_FACTOR = 50
USER_PREFIXES = ("STRAT_", "USER_")
ACTIONS = ("fast_forward", "normal")

_server: Optional[ModuleType] = None    # the engine.server module whose TICK_INTERVAL_S is set
_normal_poll_ms: Optional[int] = None   # the server's own POLL_TIMEOUT_MS (50), restored at 1x
FAST_POLL_MS = 5                        # while fast: wait at most 5 ms for messages, so ticks are not late


def _state(engine) -> dict:
    return engine.__dict__.setdefault("_ff", {"on": False, "factor": 1, "last_stop": None})


def _set_factor(factor: int) -> None:
    """The server loop waits up to POLL_TIMEOUT_MS (50 ms) for messages before it checks the tick clock; at 20x
    a tick is due every 50 ms, so that wait alone made it ~16x. Shorten it while fast, restore it at 1x."""
    if _server is None:
        return
    _server.TICK_INTERVAL_S = NORMAL_INTERVAL_S / max(1, factor)
    if _normal_poll_ms is not None:
        _server.POLL_TIMEOUT_MS = _normal_poll_ms if factor <= 1 else min(_normal_poll_ms, FAST_POLL_MS)


def status(engine) -> dict:
    st = _state(engine)
    return {"fast_forward": st["on"], "factor": st["factor"] if st["on"] else 1, "last_stop": st["last_stop"]}


def _event(engine, action: str, **extra) -> tuple:
    return ("event", {"seq": engine._next_seq(), "kind": "speed", "action": action,
                      **status(engine), **extra})


def _normal(engine, reason: str) -> bool:
    """Back to 1x. Returns True if fast-forward was on."""
    st = _state(engine)
    if not st["on"]:
        return False
    st.update(on=False, factor=1, last_stop=reason)
    _set_factor(1)
    return True


def _handle_speed(engine, msg: dict):
    action = msg.get("action")
    if action is not None and action not in ACTIONS:
        return ({"msg": "ACK", "status": "REJECTED", "reason": f"action must be one of {ACTIONS}",
                 **status(engine), "seq": engine._next_seq()}, [])
    pubs = []
    if action == "fast_forward":
        try:
            factor = int(msg.get("factor", DEFAULT_FACTOR))
        except (TypeError, ValueError):
            factor = 0
        if not 2 <= factor <= MAX_FACTOR:
            return ({"msg": "ACK", "status": "REJECTED", "reason": f"factor must be 2..{MAX_FACTOR}",
                     **status(engine), "seq": engine._next_seq()}, [])
        st = _state(engine)
        st.update(on=True, factor=factor, last_stop=None)
        _set_factor(factor)
        logger.info("fast-forward %dx until the next user trade", factor)
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
            if self.__dict__.get("_ff", {}).get("on"):
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
