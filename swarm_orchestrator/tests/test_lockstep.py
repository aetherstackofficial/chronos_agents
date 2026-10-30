"""Opt-in engine/swarm lockstep (engine/lockstep.py)."""
from engine.lockstep import Lockstep


def test_off_by_default_never_blocks():
    lock = Lockstep()
    lock.published(10, paused=False, now=0.0)
    assert not lock.enabled and lock.ready(0.0)


def test_waits_for_every_worker_of_the_published_tick():
    lock = Lockstep(workers=2, timeout_s=10)
    lock.published(10, paused=False, now=0.0)
    assert not lock.ready(1.0)
    assert lock.tick_done("SWARM_A", 10)
    assert not lock.ready(1.0)
    assert lock.tick_done("SWARM_A", 10)        # duplicates do not count twice
    assert not lock.ready(1.0) and lock.missing() == 1
    assert lock.tick_done("SWARM_B", 10)
    assert lock.ready(1.0) and not lock.timed_out(1.0)


def test_stale_tick_done_is_ignored():
    lock = Lockstep(workers=1)
    lock.published(11, paused=False, now=0.0)
    assert not lock.tick_done("SWARM_A", 10)
    assert not lock.ready(0.5)


def test_paused_ticks_are_not_waited_for():
    lock = Lockstep(workers=1)
    lock.published(12, paused=True, now=0.0)
    assert lock.ready(0.0)


def test_timeout_releases_and_is_reported():
    lock = Lockstep(workers=1, timeout_s=10)
    lock.published(13, paused=False, now=0.0)
    assert not lock.ready(9.9)
    assert lock.ready(10.0) and lock.timed_out(10.0) and lock.missing() == 1
    lock.release()
    assert lock.ready(10.0) and not lock.timed_out(10.0)
