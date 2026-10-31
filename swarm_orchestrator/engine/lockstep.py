"""Opt-in lockstep between the engine's physics clock and swarm workers.

Live, the engine ticks on a 1 Hz wall-clock timer whatever the swarm is
doing, so a worker that stalls past the next tick (seen: a one-off 1.1 s
first-tick stall on Windows) trades its orders a minute late and the run
diverges from an identical-seed run. In lockstep the server holds the next
physics tick until every expected worker has sent ``TICK_DONE`` for the tick
it last published -- never sooner than the normal 1 s cadence, so pacing is
unchanged unless the swarm is slow. A worker that stays silent past
``timeout_s`` releases the tick anyway and the server publishes a
``lockstep`` timeout event: that run is no longer reproducible, visibly.

Off (``workers=0``) by default. Pure logic, no sockets: engine/server.py
feeds it publications, TICK_DONE messages and the monotonic time.
"""
from __future__ import annotations

from typing import Optional, Set


class Lockstep:
    def __init__(self, workers: int = 0, timeout_s: float = 10.0) -> None:
        self.workers = max(0, int(workers))
        self.timeout_s = float(timeout_s)
        self._waiting_seq: Optional[int] = None
        self._since = 0.0
        self._done: Set[str] = set()

    @property
    def enabled(self) -> bool:
        return self.workers > 0

    def published(self, tick_seq: int, paused: bool, now: float) -> None:
        """A tick payload went out. Paused ticks are never waited for: the
        swarm does not act on them (core/loop_manager.py)."""
        if not self.enabled or paused:
            self._waiting_seq = None
            return
        self._waiting_seq, self._since, self._done = int(tick_seq), now, set()

    def tick_done(self, worker: str, tick_seq) -> bool:
        """Record a worker's TICK_DONE; False if it is for an older tick."""
        if self._waiting_seq is None or tick_seq != self._waiting_seq:
            return False
        self._done.add(str(worker))
        return True

    def missing(self) -> int:
        return 0 if self._waiting_seq is None else max(0, self.workers - len(self._done))

    def ready(self, now: float) -> bool:
        """May the next physics tick fire (as far as lockstep is concerned)?"""
        return self.missing() == 0 or self.timed_out(now)

    def timed_out(self, now: float) -> bool:
        return self.missing() > 0 and now - self._since >= self.timeout_s

    def release(self) -> None:
        self._waiting_seq = None
