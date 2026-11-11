"""agents/swarm_scale.py (opt-in size switches) and CHRONOS_V2_WHALE_CAP of agents/live_agents_v2.py.

Hermetic and fast: Simulation.__init__ and swarm_factory.make_agent are restored after every test by
monkeypatch, no PPO model is loaded (the v2 whale gets a dummy model).
"""
import dataclasses
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agents.swarm_factory as sf  # noqa: E402
from agents import live_agents_v2 as v2  # noqa: E402
from agents import swarm_scale as ss  # noqa: E402
from engine.matching_engine import MatchingEngine  # noqa: E402
from validation import sim_driver  # noqa: E402

ENV_VARS = (ss.COUNTS_VAR, ss.RETAIL_VAR, "CHRONOS_V2_WHALE_CAP", "CHRONOS_V2_MM_CAP")


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    """No scale variables; the patched attributes and the module's state are restored afterwards."""
    for k in ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(sim_driver.Simulation, "__init__", sim_driver.Simulation.__init__)
    monkeypatch.setattr(sf, "make_agent", sf.make_agent)
    monkeypatch.setattr(ss, "_ORIG", {})
    yield


class _FakeModel:
    decision_minutes = 1


@pytest.fixture
def no_models(monkeypatch):
    monkeypatch.setattr(v2, "get_model", lambda name: _FakeModel())


def record_counts(monkeypatch):
    """Replace Simulation.__init__ by a stub that stores the counts it was given."""
    seen = {}

    def fake(self, seed, start_price, symbol="SIM", counts=None, **kw):
        seen["counts"] = counts
        self.counts = dict(sim_driver.LIVE_COUNTS if counts is None else counts)

    monkeypatch.setattr(sim_driver.Simulation, "__init__", fake)
    return seen


def retail(index=0, seed=7):
    return sf.make_agent("retail", index, seed, sf.TickClock())


# --------------------------------------------------------------------------- no-op
def test_noop_without_variables():
    init, make = sim_driver.Simulation.__init__, sf.make_agent
    ss.install()
    assert sim_driver.Simulation.__init__ is init
    assert sf.make_agent is make
    assert ss._ORIG == {}


def test_noop_with_default_retail_size(monkeypatch):
    make = sf.make_agent
    monkeypatch.setenv(ss.RETAIL_VAR, "1")
    ss.install()
    assert sf.make_agent is make


def test_install_is_idempotent(monkeypatch):
    monkeypatch.setenv(ss.COUNTS_VAR, "retail=5")
    monkeypatch.setenv(ss.RETAIL_VAR, "3")
    ss.install()
    init, make = sim_driver.Simulation.__init__, sf.make_agent
    ss.install()
    assert sim_driver.Simulation.__init__ is init and sf.make_agent is make


def test_late_variable_is_picked_up_by_second_install(monkeypatch):
    ss.install()                                   # nothing set: nothing installed
    monkeypatch.setenv(ss.RETAIL_VAR, "2")
    ss.install()
    assert "make_agent" in ss._ORIG


def test_bad_values_raise(monkeypatch):
    monkeypatch.setenv(ss.RETAIL_VAR, "0")
    with pytest.raises(ValueError):
        ss.install()
    monkeypatch.setenv(ss.RETAIL_VAR, "1")
    monkeypatch.setenv(ss.COUNTS_VAR, "mm45")
    with pytest.raises(ValueError):
        ss.install()


# --------------------------------------------------------------------------- counts
def test_counts_none_means_scaled_copy_of_live_counts(monkeypatch):
    seen = record_counts(monkeypatch)
    monkeypatch.setenv(ss.COUNTS_VAR, "mm=45,whale=16,retail=300")
    before = dict(sim_driver.LIVE_COUNTS)
    ss.install()
    sim = sim_driver.Simulation(1, 100.0)
    assert seen["counts"] == {"mm": 45, "whale": 16, "retail": 300, "redteam": 0}
    assert sim.counts == seen["counts"]
    assert sim_driver.LIVE_COUNTS == before                   # not mutated


def test_counts_subset_of_roles(monkeypatch):
    seen = record_counts(monkeypatch)
    monkeypatch.setenv(ss.COUNTS_VAR, "retail=300")
    ss.install()
    sim_driver.Simulation(1, 100.0)
    assert seen["counts"] == {"mm": 15, "whale": 4, "retail": 300, "redteam": 0}


def test_counts_explicit_dict_zero_roles_kept_unknown_ignored(monkeypatch):
    seen = record_counts(monkeypatch)
    monkeypatch.setenv(ss.COUNTS_VAR, "mm=45,whale=16,retail=300,redteam=2,bogus=9")
    ss.install()
    given = {"mm": 15, "whale": 0, "retail": 80, "redteam": 0}      # the realism suite's ablation
    sim_driver.Simulation(1, 100.0, counts=given)
    assert seen["counts"] == {"mm": 45, "whale": 0, "retail": 300, "redteam": 0}
    assert given == {"mm": 15, "whale": 0, "retail": 80, "redteam": 0}   # caller's dict untouched
    assert "bogus" not in seen["counts"]
    sim_driver.Simulation(1, 100.0, counts={"mm": 0, "whale": 0, "retail": 0, "redteam": 0})
    assert seen["counts"] == {"mm": 0, "whale": 0, "retail": 0, "redteam": 0}


def test_counts_role_absent_from_dict_stays_absent(monkeypatch):
    seen = record_counts(monkeypatch)
    monkeypatch.setenv(ss.COUNTS_VAR, "mm=45,retail=300")
    ss.install()
    sim_driver.Simulation(1, 100.0, counts={"retail": 80})
    assert seen["counts"] == {"retail": 300}


def test_counts_positional_and_keyword_arguments(monkeypatch):
    seen = record_counts(monkeypatch)
    monkeypatch.setenv(ss.COUNTS_VAR, "retail=9")
    ss.install()
    sim_driver.Simulation(1, 100.0, "SIM", {"retail": 80, "mm": 0})
    assert seen["counts"] == {"retail": 9, "mm": 0}
    sim_driver.Simulation(1, 100.0, counts={"retail": 80})
    assert seen["counts"] == {"retail": 9}


def test_real_simulation_builds_the_scaled_swarm(monkeypatch):
    monkeypatch.setenv(ss.COUNTS_VAR, "retail=6")
    ss.install()
    sim = sim_driver.Simulation(3, 100.0, counts={"mm": 0, "whale": 0, "retail": 2, "redteam": 0})
    assert sim.counts["retail"] == 6
    assert [a.agent_id for a in sim.orch.agents] == [f"RETAIL_{i}" for i in range(6)]


def test_scale_counts_is_pure():
    base = {"mm": 15, "whale": 4}
    assert ss.scale_counts(None, base, {"mm": 2}) == {"mm": 2, "whale": 4}
    assert base == {"mm": 15, "whale": 4}


# --------------------------------------------------------------------------- retail size
def test_retail_scaling(monkeypatch):
    plain = [retail(i) for i in range(12)]
    monkeypatch.setenv(ss.RETAIL_VAR, "20")
    ss.install()
    scaled = [retail(i) for i in range(12)]
    for a, b in zip(plain, scaled):
        p, q = a.personality, b.personality
        want_cash = min(max(p.start_cash * 20, MatchingEngine.RETAIL_CASH_MIN), MatchingEngine.RETAIL_CASH_MAX)
        assert q.start_cash == want_cash and b.cash == want_cash
        assert q.max_position_cap == int(p.max_position_cap * 20)
        assert dataclasses.replace(q, start_cash=p.start_cash, max_position_cap=p.max_position_cap) == p
        assert type(b.personality.max_position_cap) is int


def test_retail_cash_clipped_to_engine_band(monkeypatch):
    monkeypatch.setenv(ss.RETAIL_VAR, "1e6")
    ss.install()
    a = retail(0)
    assert a.personality.start_cash == MatchingEngine.RETAIL_CASH_MAX == a.cash   # not 50,000 fallback
    monkeypatch.setenv(ss.RETAIL_VAR, "1e-9")
    b = retail(0)
    assert b.personality.start_cash == MatchingEngine.RETAIL_CASH_MIN == b.cash
    assert b.personality.max_position_cap >= 1


def test_retail_scaled_cash_accepted_by_engine(monkeypatch):
    monkeypatch.setenv(ss.RETAIL_VAR, "30")
    ss.install()
    a = retail(1)
    e = MatchingEngine()
    e.init_sim("TCS", "TECH", 190.0, seed=1)
    ack, _ = e.handle_message({"msg": "CANCEL_ALL", "agent_id": a.agent_id, "start_cash": a.cash})
    assert ack["cash"] == a.cash


def test_retail_scaling_leaves_other_roles_alone(monkeypatch):
    plain = sf.make_agent("mm", 0, 7, sf.TickClock())
    monkeypatch.setenv(ss.RETAIL_VAR, "50")
    ss.install()
    mm = sf.make_agent("mm", 0, 7, sf.TickClock())
    assert type(mm) is type(plain) and mm.cash == plain.cash


@pytest.mark.parametrize("scale_first", [True, False])
def test_composes_with_v2_patch_in_both_orders(monkeypatch, no_models, scale_first):
    monkeypatch.setenv(ss.RETAIL_VAR, "10")
    monkeypatch.setattr(v2, "_INSTALLED", False)
    from core.loop_manager import SwarmOrchestrator
    monkeypatch.setattr(SwarmOrchestrator, "_on_tick", SwarmOrchestrator._on_tick)
    base_cash = retail(2).personality.start_cash
    if scale_first:
        ss.install()
    v2.install()                    # calls swarm_scale.install() last (no-op if already installed)
    ss.install()
    r = retail(2)
    assert r.personality.start_cash == min(base_cash * 10, MatchingEngine.RETAIL_CASH_MAX)
    assert type(sf.make_agent("whale", 0, 7, sf.TickClock())).__name__ == "LiveWhaleAgentV2"


# --------------------------------------------------------------------------- whale cap
def _whale(seed=5, index=0):
    aid = f"WHALE_{index}"
    return v2.LiveWhaleAgentV2(aid, "whale_swing", rng=sf.agent_rng(seed, aid))


def test_whale_cap_default_sequence_unchanged(no_models):
    for i in range(6):
        w = _whale(5, i)
        r = sf.agent_rng(5, f"WHALE_{i}")
        assert w.capacity == w.max_position == r.randint(50_000, 200_000)
        assert w._risk == r.uniform(*v2.RISK_TOLERANCE_RANGE)


def test_whale_cap_explicit_default_equals_unset(no_models, monkeypatch):
    unset = [_whale(5, i).capacity for i in range(5)]
    monkeypatch.setenv("CHRONOS_V2_WHALE_CAP", "50000,200000")
    assert [_whale(5, i).capacity for i in range(5)] == unset


def test_whale_cap_range_respected(no_models, monkeypatch):
    monkeypatch.setenv("CHRONOS_V2_WHALE_CAP", "55000,330000")
    caps = [_whale(9, i).capacity for i in range(60)]
    assert all(55_000 <= c <= 330_000 for c in caps)
    assert max(caps) > 200_000                      # the range really is wider than the default
    w = _whale(9, 3)
    r = sf.agent_rng(9, "WHALE_3")
    assert w.capacity == r.randint(55_000, 330_000)


def test_mm_cap_still_works(no_models, monkeypatch):
    monkeypatch.setenv("CHRONOS_V2_MM_CAP", "11000,55000")
    mm = v2.LiveMarketMakerV2("MM_0", "mm", rng=sf.agent_rng(5, "MM_0"))
    assert 11_000 <= mm.capacity <= 55_000


# --------------------------------------------------------------------------- determinism
def _fingerprint(agents):
    out = []
    for a in agents:
        p = getattr(a, "personality", None)
        out.append((a.agent_id, a.cash, p, a._rng.random()))
    return out


def test_same_seed_same_swarm(monkeypatch):
    monkeypatch.setenv(ss.COUNTS_VAR, "retail=12")
    monkeypatch.setenv(ss.RETAIL_VAR, "25")
    ss.install()
    runs = []
    for _ in range(2):
        sim = sim_driver.Simulation(4, 100.0, counts={"mm": 0, "whale": 0, "retail": 3, "redteam": 0})
        runs.append(_fingerprint(sim.orch.agents))
    assert runs[0] == runs[1]
    assert len(runs[0]) == 12


def test_unset_variables_swarm_identical_to_factory(monkeypatch):
    ss.install()
    a = _fingerprint(sf.build_swarm({"retail": 5}, 11, sf.TickClock()))
    monkeypatch.setenv(ss.RETAIL_VAR, "1")
    b = _fingerprint(sf.build_swarm({"retail": 5}, 11, sf.TickClock()))
    assert a == b
