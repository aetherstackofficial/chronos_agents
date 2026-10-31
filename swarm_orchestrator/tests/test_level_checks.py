"""The round-number clustering check (validation/checks/level_tests.py). numpy only."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.checks import level_tests as lt  # noqa: E402

TICK = 0.1


def _grid_prices(n, rng, a=0.0, lo=1300.0, hi=1400.0):
    """Prices on the 0.1 grid; a share `a` is moved to a whole rupee (a level)."""
    p = np.round(rng.uniform(lo, hi, n) / TICK) * TICK
    pick = rng.random(n) < a
    p[pick] = np.round(p[pick])
    return p


def test_a_uniform_walk_over_the_grid_shows_no_clustering():
    p = _grid_prices(400_000, np.random.default_rng(1))
    assert lt.level_excess(p, 1.0, TICK) == pytest.approx(1.0, abs=0.05)
    assert lt.level_excess(p, 10.0, TICK) == pytest.approx(1.0, abs=0.12)


def test_injected_clustering_is_measured_and_inverted_back_to_the_share():
    a, g = 0.10, 10                                           # 10 grid points per rupee at tick 0.1
    p = _grid_prices(600_000, np.random.default_rng(2), a=a)
    e = lt.level_excess(p, 1.0, TICK)
    assert e == pytest.approx(1 + a * g / (1 - a), rel=0.08)  # 2.1 (the model of the module docstring)
    assert lt.implied_anchor_prob(e, g) == pytest.approx(a, abs=0.01)


def test_the_implied_share_formula_round_trips():
    for a in (0.02, 0.09, 0.3):
        for g in (10, 20):
            assert lt.implied_anchor_prob(1 + a * g / (1 - a), g) == pytest.approx(a)
    assert lt.implied_anchor_prob(1.0, 10) == 0.0 and lt.implied_anchor_prob(0.8, 10) == 0.0
    assert lt.implied_anchor_prob(float("nan"), 10) == 0.0


def test_steps_not_wider_than_two_ticks_are_not_measurable():
    p = _grid_prices(1000, np.random.default_rng(3))
    assert np.isnan(lt.level_excess(p, 0.2, TICK))            # two ticks: neighbours are levels themselves
    assert np.isnan(lt.level_excess(np.array([]), 1.0, TICK))


def test_base_unit_is_the_agents_rule():
    u = lt.base_unit(np.array([1354.0, 190.0, 14_000.0, 3.0]), 0.001)
    assert u.tolist() == pytest.approx([1.0, 0.1, 10.0, 0.001])
    assert lt.base_unit(np.array([1354.0]), 5.0).tolist() == [5.0]        # never below a tick


def test_infer_tick_picks_the_coarsest_grid_the_prices_sit_on():
    rng = np.random.default_rng(4)
    assert lt.infer_tick(np.round(rng.uniform(100, 200, 5000) / 0.05) * 0.05) == 0.05
    assert lt.infer_tick(np.round(rng.uniform(1000, 2000, 5000) / 0.1) * 0.1) == 0.1
    assert lt.infer_tick(np.round(rng.uniform(10000, 20000, 5000) / 0.5) * 0.5) == 0.5


def _series(a, seed):
    rng = np.random.default_rng(seed)
    return {c: _grid_prices(200_000, rng, a=a) for c in ("c", "h", "l")}


def test_the_gate_fails_a_market_without_clustering_and_passes_one_with_it():
    flat = lt.t9_round_number_clustering(_series(0.0, 5), TICK, None)
    assert flat["pass"] is False and flat["metrics"]["excess_base_unit_mean"] == pytest.approx(1.0, abs=0.06)
    rich = lt.t9_round_number_clustering(_series(0.09, 6), TICK, None)
    assert rich["pass"] is True and rich["target_pass"] is None            # no reference given


def test_target_compares_with_the_real_band():
    ref = {"A": {"excess_base_unit_mean": 3.0}, "B": {"excess_base_unit_mean": 3.5},
           "C": {"excess_base_unit_mean": float("nan")}}
    near = lt.t9_round_number_clustering(_series(0.17, 7), TICK, ref)       # excess ~ 2.9 -> inside 2.4 .. 4.2
    far = lt.t9_round_number_clustering(_series(0.02, 8), TICK, ref)        # ~1.2 -> below the band
    assert near["target_pass"] is True and far["target_pass"] is False
    assert near["metrics"]["real_band_lo"] == pytest.approx(2.4) and near["metrics"]["real_band_hi"] == pytest.approx(4.2)


def test_real_reference_helper_measures_each_stock_with_its_own_tick():
    rng = np.random.default_rng(9)
    series = {"X": {c: _grid_prices(150_000, rng, a=0.1, lo=1100, hi=1600) for c in ("c", "h", "l")}}
    ref = lt.real_reference(series)
    assert ref["X"]["tick"] == 0.1 and ref["X"]["excess_base_unit_mean"] > 1.8
    assert set(ref["X"]["excess"]) == {"c", "h", "l"}


REF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "validation", "config", "round_number_reference.json")


@pytest.mark.skipif(not os.path.exists(REF), reason="reference not built (python -m validation.level_report --reference)")
def test_the_committed_real_reference_supports_the_frozen_anchor_probability():
    from agents import heuristic_retail as hr
    ref = json.load(open(REF))
    assert {"RELIANCE", "MARUTI"} <= set(ref)
    shares = []
    for sym in ("RELIANCE", "MARUTI", "SBIN"):
        e = ref[sym]["excess_base_unit_mean"]
        assert e > 2.0, f"{sym}: real round-number clustering should be clearly present"
        shares.append(lt.implied_anchor_prob(e, 20))                        # 20 grid points per unit for these three
    anchor = hr.CONFIG["levels"]["anchor_prob"]
    assert min(shares) - 0.03 <= anchor <= max(shares) + 0.03
