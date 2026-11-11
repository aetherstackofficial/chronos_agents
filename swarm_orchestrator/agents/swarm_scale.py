"""Opt-in "size" switches for the simulated agents, for experiments (nothing else is modified).

Today the agents (market makers, whales, retail) are about 5 % of the traded volume; on the real NSE
individuals, prop traders and institutions are ~95 % (2025: individuals 33.5 %, prop 29.8 %, DII 14.2 %,
FPI ~17 %). These switches scale the agents up, within the ranges the v2 models were trained on
(whale capacity 0.05 %..3 % of ADV, market-maker capacity 0.02 %..0.5 % of ADV).

Variables (all unset = no patch at all; a run is then bit-identical to one without this module):
  CHRONOS_SCALE_COUNTS="mm=45,whale=16,retail=300"   any subset of roles. Wraps
        validation.sim_driver.Simulation.__init__: every role listed whose incoming count is NONZERO gets
        the listed count; zero counts (ablations: whale=0, the no-agent runs) and roles not in the
        incoming dict stay as they are, unknown roles are ignored. counts=None means a copy of
        LIVE_COUNTS (never mutated). ``sim.counts`` shows the counts actually used.
  CHRONOS_SCALE_RETAIL_SIZE=k   (float > 0, default 1 = off). One retail agent stands for a group of k
        investors: start_cash x k (clipped to MatchingEngine.RETAIL_CASH_MIN..RETAIL_CASH_MAX) and
        max_position_cap x k (int, at least 1), via dataclasses.replace on the frozen personality;
        agent.cash is reset. Every other personality field (type, thresholds, risk appetite, cooldown,
        leverage, exits) is unchanged, so order size grows like k until a cap or the cash clip is met.
  The whale / market-maker size ranges are CHRONOS_V2_WHALE_CAP and CHRONOS_V2_MM_CAP
  (agents/live_agents_v2.py).

Order: ``install()`` wraps whatever swarm_factory.make_agent is at that moment, so call it AFTER
agents.live_agents_v2.install() (which calls it itself, last); in the other order the v2 patch sits
outside the scale patch and still delegates retail to it, so both orders give scaled retail.
Scripts that set the variables after the modules are installed call ``install()`` again: it is
idempotent per switch, and installs a switch only once its variable is set.

Limits: only the simulation driver (validation/sim_driver.py) is scaled; the live worker
(worker.py / DEFAULT_COUNTS) is NOT patched. If ``validation`` cannot be imported the counts switch is
skipped silently. Retail cash is clipped at the engine's 1e9 (above it the engine would silently fall
back to 50,000, so the clip also protects against that); k below 1 is clipped at the 10,000 floor.
"""
from __future__ import annotations

import dataclasses
import inspect
import os
from typing import Dict, Optional

COUNTS_VAR = "CHRONOS_SCALE_COUNTS"
RETAIL_VAR = "CHRONOS_SCALE_RETAIL_SIZE"
ROLES = ("mm", "whale", "retail", "redteam")

_ORIG: Dict[str, object] = {}        # "sim_init", "make_agent" -> the function that was wrapped


def parse_counts(text: str) -> Dict[str, int]:
    """"mm=45,whale=16" -> {"mm": 45, "whale": 16}. Raises ValueError on a malformed pair."""
    out: Dict[str, int] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        role, sep, val = part.partition("=")
        if not sep:
            raise ValueError(f"{COUNTS_VAR}: expected role=count, got {part!r}")
        n = int(val)
        if n < 0:
            raise ValueError(f"{COUNTS_VAR}: negative count in {part!r}")
        out[role.strip()] = n
    return out


def retail_size() -> float:
    """CHRONOS_SCALE_RETAIL_SIZE as a float (1.0 when unset or empty). Raises ValueError if <= 0."""
    raw = os.environ.get(RETAIL_VAR, "").strip()
    if not raw:
        return 1.0
    k = float(raw)
    if not k > 0:
        raise ValueError(f"{RETAIL_VAR} must be > 0, got {raw!r}")
    return k


def scale_counts(counts: Optional[Dict[str, int]], base: Dict[str, int],
                 scaled: Dict[str, int]) -> Dict[str, int]:
    """The counts a Simulation should use: ``counts`` (None: a copy of ``base``) with every role of
    ``scaled`` that has a nonzero count replaced by the scaled count. Pure."""
    out = dict(base if counts is None else counts)
    for role, n in scaled.items():
        if out.get(role, 0) != 0:
            out[role] = n
    return out


def scale_retail(agent, k: float) -> None:
    """Scale one retail agent in place (see the module docstring)."""
    from engine.matching_engine import MatchingEngine
    p = agent.personality
    cash = min(max(float(p.start_cash) * k, MatchingEngine.RETAIL_CASH_MIN), MatchingEngine.RETAIL_CASH_MAX)
    cap = max(1, int(p.max_position_cap * k))
    agent.personality = dataclasses.replace(p, start_cash=cash, max_position_cap=cap)
    agent.cash = cash


def _install_counts() -> None:
    try:
        from validation import sim_driver
    except Exception:                        # validation not importable here: nothing to scale
        return
    orig = sim_driver.Simulation.__init__
    sig = inspect.signature(orig)

    def __init__(self, *args, **kwargs):
        scaled = parse_counts(os.environ.get(COUNTS_VAR, ""))
        if scaled:
            bound = sig.bind(self, *args, **kwargs)
            bound.arguments["counts"] = scale_counts(bound.arguments.get("counts"), sim_driver.LIVE_COUNTS,
                                                     scaled)
            args, kwargs = bound.args[1:], bound.kwargs
        orig(self, *args, **kwargs)

    __init__.__wrapped__ = orig
    sim_driver.Simulation.__init__ = __init__
    _ORIG["sim_init"] = orig


def _install_retail() -> None:
    import agents.swarm_factory as sf
    orig = sf.make_agent

    def make_agent(role, index, seed, clock):
        agent = orig(role, index, seed, clock)
        if role == "retail" and hasattr(agent, "personality"):
            k = retail_size()
            if k != 1.0:
                scale_retail(agent, k)
        return agent

    sf.make_agent = make_agent
    _ORIG["make_agent"] = orig


def install() -> None:
    """Install the switches whose variables are set; idempotent per switch, no-op when none is set."""
    if "sim_init" not in _ORIG and os.environ.get(COUNTS_VAR, "").strip():
        parse_counts(os.environ[COUNTS_VAR])         # fail early on a malformed value
        _install_counts()
    if "make_agent" not in _ORIG and retail_size() != 1.0:
        _install_retail()


def uninstall() -> None:
    """Undo ``install()`` (tests)."""
    if "sim_init" in _ORIG:
        from validation import sim_driver
        sim_driver.Simulation.__init__ = _ORIG.pop("sim_init")
    if "make_agent" in _ORIG:
        import agents.swarm_factory as sf
        sf.make_agent = _ORIG.pop("make_agent")
