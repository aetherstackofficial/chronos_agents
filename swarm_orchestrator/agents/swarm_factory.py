"""One way to build the swarm, shared by worker.py and the validation harness.

Determinism comes from two things here (PROTOCOL §8.7):

* ``TickClock`` -- agent cooldowns count physics ticks, read from the tick
  payload, instead of wall time. Live runs one tick per wall second
  (engine/server.py), so cooldowns keep their meaning in seconds, but a run
  no longer depends on how fast the swarm process happens to be.
* Per-agent RNGs seeded from ``f"{seed}:{agent_id}"`` -- independent of
  construction order and of any global RNG state (model loading touches the
  global ones). ``seed=None`` keeps the old unseeded behaviour.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

SESSION_MINUTES = 375

# Construction order is part of the run definition: the orchestrator draws
# each agent's awake/asleep state in list order.
ROLE_ORDER = ("mm", "whale", "retail", "redteam")
DEFAULT_COUNTS: Dict[str, int] = {"mm": 15, "whale": 4, "retail": 80, "redteam": 3}


class TickClock:
    """Seconds as counted by physics ticks: 1 tick = 1 s (the live cadence).

    The orchestrator calls ``observe`` with each tick payload before any
    agent acts, so every agent sees the same time for the same tick. A
    paused engine republishes the same minute, so the clock stands still.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def observe(self, tick: dict) -> None:
        day, minute = tick.get("day_count"), tick.get("market_minute")
        if day is not None and minute is not None:
            self.now = float((int(day) - 1) * SESSION_MINUTES + int(minute))

    def __call__(self) -> float:
        return self.now


def agent_rng(seed: Optional[int], agent_id: str) -> random.Random:
    return random.Random(None if seed is None else f"{seed}:{agent_id}")


def make_agent(role: str, index: int, seed: Optional[int], clock: TickClock):
    """One agent; imports are lazy because live_agents loads the PPO models."""
    if role == "redteam":
        from agents.red_team import RedTeamAgent
        aid = f"REDTEAM_{index}"
        return RedTeamAgent(aid, clock=clock, rng=agent_rng(seed, aid))
    from agents.live_agents import LiveMarketMaker, LiveRetailAgent, LiveWhaleAgent
    cls, prefix = {"mm": (LiveMarketMaker, "MM"), "whale": (LiveWhaleAgent, "WHALE"),
                   "retail": (LiveRetailAgent, "RETAIL")}[role]
    aid = f"{prefix}_{index}"
    if role == "retail":
        # A retail agent's personality (agents/heuristic_retail.py) is drawn from the run seed.
        return cls(aid, clock=clock, rng=agent_rng(seed, aid), seed=seed)
    return cls(aid, clock=clock, rng=agent_rng(seed, aid))


def build_swarm(counts: Dict[str, int], seed: Optional[int], clock: TickClock) -> List:
    return [make_agent(role, i, seed, clock)
            for role in ROLE_ORDER for i in range(counts.get(role, 0))]
