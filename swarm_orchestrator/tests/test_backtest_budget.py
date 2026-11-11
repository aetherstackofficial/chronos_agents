"""Backtest mode, "replace, don't add" rules (agents/backtest_mode.py): AGENT_FP, VOLBUDGET, DEPTHBUDGET.

Same synthetic instrument and fixtures as tests/test_backtest_mode.py (price 100, tick 0.01, adv 1e6,
sigma_d 1 %, so F(d) = 1e6 (d/100)^2 = 100 d^2 shares within d ticks; a flat tape of 3000 shares a bar).
"""
import copy
import math

import pytest

from agents import backtest_mode as btm
from test_backtest_mode import (ADV, M, P0, SIGMA, STRAT, TICK, bt, closes, flat_tape,  # noqa: F401
                                make as _make, order, walk_tape)

MM = "MM_1"
USER = "STRAT_X"
FP_DEFAULTS = dict(AGENT_FP="all", VOLBUDGET=0, VOLBUDGET_MAXCUT=0.9, DEPTHBUDGET=0)


def make(bt, bars, **kw):
    """test_backtest_mode.make, but every CHRONOS_BT_* parameter of an earlier engine of the test is reset."""
    import os
    for k in list(os.environ):
        if k.startswith("CHRONOS_BT_") and k != "CHRONOS_BT_TAPE":
            bt.delenv(k)
    return _make(bt, bars, **kw)


def scene(bt, bars, script, n, m=100, **env):
    """Engine on ``bars``; ``script``: tick index -> list of (agent, side, qty). Runs ``n`` ticks."""
    e = make(bt, bars, m=m, **env)
    run(e, script, 0, n)
    return e


def run(e, script, lo, hi):
    for k in range(lo, hi):
        for agent, side, qty in script.get(k, []):
            order(e, agent, side, qty)
        e.physics_tick()


def u_after(bt, who, side="BUY", qty=5_000, **env):
    """u of the step that follows one market order of ``who`` on a flat tape."""
    e = scene(bt, flat_tape(300), {1: [(who, side, qty)]}, 2, **env)
    return btm.get_state(e)


# --------------------------------------------------------------------------- #
# helpers of the module
# --------------------------------------------------------------------------- #
def test_is_user():
    assert btm._is_user("STRAT_X") and btm._is_user("USER_1") and btm._is_user("STRAT_PROBE")
    for aid in ("MM_1", "WHALE_3", "RETAIL_9", "REDTEAM_1", "SIM_LP", "GHOST_HIST", "", None):
        assert not btm._is_user(aid)


def test_live_flow_split_groups_user_and_agents(bt):
    e = make(bt, flat_tape(300), m=100)
    e.physics_tick()
    order(e, MM, "BUY", 700)
    order(e, "RETAIL_1", "SELL", 300)
    order(e, USER, "BUY", 400)
    order(e, "USER_2", "SELL", 150)
    user, agent = btm._live_flow_split(e)
    assert (user[0], user[1]) == (400, 150) and (agent[0], agent[1]) == (700, 300)
    assert user[2] > 0 and user[3] > 0 and agent[2] > 0 and agent[3] > 0
    assert user[4] >= user[2] and agent[4] >= agent[2] and user[5] <= user[3] and agent[5] <= agent[3]
    B, S, *_ = btm._live_flow(e)                      # the old combined numbers stay available
    assert (B, S) == (1100, 450)


# --------------------------------------------------------------------------- #
# (a) defaults are bit-identical to explicit defaults
# --------------------------------------------------------------------------- #
def test_default_env_equals_explicit_defaults(bt):
    bars = walk_tape(sessions=3)
    script = {4: [(MM, "BUY", 800)], 9: [(USER, "BUY", 6_000)], 15: [("WHALE_1", "SELL", 3_000)],
              20: [("RETAIL_1", "BUY", 200)], 30: [(USER, "SELL", 2_000), (MM, "SELL", 500)],
              45: [(USER, "BUY", 9_000)]}
    runs = []
    for env in ({}, FP_DEFAULTS):
        e = scene(bt, bars, script, 90, m=M, **env)
        runs.append((list(e._closed_bars), btm.get_state(e), btm.agent_flow_log(e), e.book.depth(60),
                     e.last_price))
    assert runs[0] == runs[1]
    st = runs[0][1]
    assert st["u"] != 0.0 and st["agent_fp"] == "all" and st["volbudget"] == 0 and st["depthbudget"] == 0
    assert st["volbudget_maxcut"] == 0.9
    assert st["vol_debt"] == 0.0 and st["vol_cut_total"] == 0.0 and st["depth_displaced_total"] == 0
    assert st["agent_aggr_total"] == 800 + 3_000 + 200 + 500 and st["latent_posted_total"] > 0


def test_all_mode_footprint_is_the_combined_flow(bt):
    """AGENT_FP=all: an agent's buy moves the level exactly as a user's buy of the same size does."""
    s_agent = u_after(bt, MM)
    s_user = u_after(bt, USER)
    assert s_agent["u"] > 0 and s_agent["u"] == s_user["u"]
    assert s_agent["n_agent"] == 5_000 and s_agent["n_user"] == 0
    assert s_user["n_user"] == 5_000 and s_user["n_agent"] == 0


def test_invalid_agent_fp_falls_back_to_all(bt):
    assert u_after(bt, MM, AGENT_FP="bogus")["agent_fp"] == "all"
    assert u_after(bt, MM, AGENT_FP="OFF")["agent_fp"] == "off"


# --------------------------------------------------------------------------- #
# (b) off
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("agent", [MM, "RETAIL_1", "WHALE_2", "REDTEAM_1"])
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_off_agents_never_move_the_level(bt, agent, side):
    st = u_after(bt, agent, side, AGENT_FP="off")
    assert st["u"] == 0.0 and st["xp"] == 0.0 and st["xt"] == 0.0
    assert st["n_agent"] == (5_000 if side == "BUY" else -5_000) and st["n_user"] == 0
    assert st["agent_fp"] == "off"


@pytest.mark.parametrize("name", [USER, "USER_7", "STRAT_PROBE"])
def test_off_the_user_still_moves_the_level(bt, name):
    st = u_after(bt, name, AGENT_FP="off")
    assert st["u"] > 0 and st["n_user"] == 5_000 and st["n_agent"] == 0
    assert st["u"] == u_after(bt, name)["u"]                    # same as the default for user-only flow


def test_off_user_and_agent_together_move_by_the_user_only(bt):
    both = scene(bt, flat_tape(300), {1: [(USER, "BUY", 3_000), (MM, "BUY", 5_000)]}, 2, AGENT_FP="off")
    st = btm.get_state(both)
    assert st["n_user"] == 3_000 and st["n_agent"] == 5_000
    assert st["u"] > 0 and st["xp"] + st["xt"] == pytest.approx(3_000, rel=0.02)    # the user's 3 000 only


def test_off_with_footprint_disabled_nothing_moves(bt):
    st = u_after(bt, USER, AGENT_FP="off", FOOTPRINT=0)
    assert st["u"] == 0.0 and st["n_user"] == 5_000


# --------------------------------------------------------------------------- #
# (c) diff
# --------------------------------------------------------------------------- #
def _baseline_run(bt, side="BUY", qty=5_000):
    """No-user run: the agent trades, the default mode records its flow."""
    e = scene(bt, flat_tape(300), {1: [(MM, side, qty)]}, 3)
    return btm.agent_flow_log(e)


def test_agent_flow_log_records_every_footprint_step(bt):
    log = _baseline_run(bt)
    assert len(log) == 3 and sorted(log) == sorted(set(log))
    assert sorted(log.values()) == [0.0, 0.0, 5_000.0]
    assert btm.agent_flow_log(make(bt, flat_tape(10))) == {}     # no state yet


def test_diff_baseline_equal_to_the_agents_flow_leaves_u_unchanged(bt):
    log = _baseline_run(bt)
    e = make(bt, flat_tape(300), m=100, AGENT_FP="diff")
    btm.set_agent_baseline(e, log)                              # before the first tick (adopted at creation)
    run(e, {1: [(MM, "BUY", 5_000)]}, 0, 3)
    st = btm.get_state(e)
    assert st["u"] == 0.0 and st["xp"] == 0.0 and st["xt"] == 0.0
    assert st["agent_base_len"] == 3
    # the same through a state that already exists
    e2 = make(bt, flat_tape(300), m=100, AGENT_FP="diff")
    e2.physics_tick()
    btm.set_agent_baseline(e2, log)
    assert btm.get_state(e2)["agent_base_len"] == 3
    assert btm.agent_flow_log(e2) != log


def test_diff_baseline_zero_equals_all_for_agent_only_flow(bt):
    s_all = u_after(bt, MM)
    log = {k: 0.0 for k in _baseline_run(bt)}
    e = make(bt, flat_tape(300), m=100, AGENT_FP="diff")
    btm.set_agent_baseline(e, log)
    run(e, {1: [(MM, "BUY", 5_000)]}, 0, 2)
    assert btm.get_state(e)["u"] == s_all["u"] > 0
    assert btm.get_state(e)["xp"] == s_all["xp"]
    assert list(e._closed_bars) == list(scene(bt, flat_tape(300), {1: [(MM, "BUY", 5_000)]}, 2)._closed_bars)


def test_diff_baseline_is_a_differential(bt):
    """The user's order changes the agents' flow from 5 000 to 3 000: the level moves by the -2 000 only
    (CAP=0: the cap needs a trade on the pushed side, see the next tests)."""
    log = {k: v for k, v in _baseline_run(bt).items()}
    e = make(bt, flat_tape(300), m=100, AGENT_FP="diff", CAP=0)
    btm.set_agent_baseline(e, log)
    run(e, {1: [(MM, "BUY", 3_000)]}, 0, 2)
    st = btm.get_state(e)
    assert st["n_agent"] == 3_000 and st["u"] < 0 and st["xp"] + st["xt"] == pytest.approx(-2_000, rel=0.05)
    assert st["u"] == pytest.approx(btm.curve_u(st["xp"] + st["xt"], st), rel=1e-9)


def test_diff_without_a_baseline_equals_off(bt):
    for who, side in ((MM, "BUY"), (MM, "SELL"), (USER, "BUY"), (USER, "SELL")):
        d = u_after(bt, who, side, AGENT_FP="diff")
        o = u_after(bt, who, side, AGENT_FP="off")
        for k in ("u", "xp", "xt", "n_user", "n_agent"):
            assert d[k] == o[k], (who, side, k)
    assert u_after(bt, MM, AGENT_FP="diff")["u"] == 0.0


def test_diff_no_user_run_is_only_moved_by_the_bars(bt):
    bars = walk_tape(sessions=2)
    script = {5: [(MM, "BUY", 4_000)], 12: [("WHALE_1", "SELL", 6_000)], 20: [("RETAIL_1", "BUY", 900)]}
    e = scene(bt, bars, script, 2 * M, m=M, AGENT_FP="diff")
    assert btm.get_state(e)["u"] == 0.0
    ref = scene(bt, bars, {}, 2 * M, m=M, AGENT_FP="diff")
    # the agents' orders perturb the book a little but the level is the tape's (+/- the bars the print rounds)
    assert abs(math.log(e.last_price / ref.last_price)) < 4 * TICK / P0 * 10


def test_diff_user_flow_enters_in_full(bt):
    log = {k: 0.0 for k in _baseline_run(bt)}
    e = make(bt, flat_tape(300), m=100, AGENT_FP="diff")
    btm.set_agent_baseline(e, log)
    run(e, {1: [(USER, "BUY", 5_000)]}, 0, 2)
    st = btm.get_state(e)
    assert st["u"] > 0 and st["n_user"] == 5_000 and st["u"] == u_after(bt, USER)["u"]


def test_diff_differential_without_a_trade_on_its_side_is_safe(bt):
    """Baseline says the agents sold 4 000 here; they only sold 1 000: +3 000 differential with no buy of the
    minute (capped at the sellers' worst price) and, in another minute, a differential with no trade at all."""
    e = make(bt, flat_tape(300), m=100, AGENT_FP="diff")
    e.physics_tick()
    tc = e.tick_count
    btm.set_agent_baseline(e, {tc: -4_000.0, tc + 1: 2_000.0})
    order(e, MM, "SELL", 1_000)
    e.physics_tick()                                            # key tc: differential +3 000, only a sell traded
    st = btm.get_state(e)
    assert math.isfinite(st["u"]) and st["u"] <= 0.0 + 1e-12    # cannot go above the sell price paid
    e.physics_tick()                                            # key tc+1: differential -2 000, no trade at all
    st = btm.get_state(e)
    assert math.isfinite(st["u"])
    assert st["n_agent"] == 0.0


def test_diff_baseline_survives_deepcopy_and_is_a_copy(bt):
    log = _baseline_run(bt)
    for before_first_tick in (True, False):
        e = make(bt, flat_tape(300), m=100, AGENT_FP="diff")
        if not before_first_tick:
            e.physics_tick()
            order(e, MM, "BUY", 5_000)
            e.physics_tick()
        mapping = dict(log)
        btm.set_agent_baseline(e, mapping)
        mapping.clear()                                         # the engine holds its own copy
        twin = copy.deepcopy(e)
        for eng in (e, twin):
            if before_first_tick:
                run(eng, {1: [(MM, "BUY", 5_000)]}, 0, 3)
            else:
                run(eng, {}, 0, 1)
        assert btm.get_state(twin) == btm.get_state(e)
        assert list(twin._closed_bars) == list(e._closed_bars)
        if before_first_tick:
            assert btm.get_state(twin)["u"] == 0.0 and btm.get_state(twin)["agent_base_len"] == 3
        assert btm.get_state(twin)["agent_base_len"] == 3


def test_diff_snapshot_determinism_with_a_user(bt):
    log = _baseline_run(bt)
    e = make(bt, walk_tape(sessions=2), m=M, AGENT_FP="diff")
    btm.set_agent_baseline(e, log)
    script = {1: [(MM, "BUY", 5_000)], 6: [(USER, "BUY", 4_000)], 20: [(USER, "SELL", 2_000), (MM, "SELL", 300)]}
    run(e, script, 0, 10)
    twin = copy.deepcopy(e)
    run(e, script, 10, 50)
    run(twin, script, 10, 50)
    assert list(e._closed_bars) == list(twin._closed_bars)
    assert btm.get_state(e) == btm.get_state(twin) and btm.agent_flow_log(e) == btm.agent_flow_log(twin)


# --------------------------------------------------------------------------- #
# (f) the flow log
# --------------------------------------------------------------------------- #
def test_flow_log_is_bounded_at_5000(bt):
    n = 5_150
    e = make(bt, flat_tape(n + 10), m=n + 10)
    for k in range(n):
        if k == 3:
            order(e, MM, "BUY", 100)                           # early entry: must be dropped
        if k == n - 2:
            order(e, MM, "SELL", 40)                           # late entry: must stay
        e.physics_tick()
    log = btm.agent_flow_log(e)
    assert btm._FLOW_KEEP == 5_000 and len(log) == 5_000
    ticks = sorted(log)
    assert ticks[-1] == e.tick_count and ticks[0] == e.tick_count - 4_999          # the most recent ones
    assert 100.0 not in log.values() and -40.0 in log.values()
    assert btm.get_state(e)["agent_flow_len"] == 5_000
    log[ticks[0]] = 12345.0                                    # a copy: the state is not touched
    assert btm.agent_flow_log(e)[ticks[0]] != 12345.0


def test_flow_log_is_recorded_in_every_mode(bt):
    for env in (dict(AGENT_FP="off"), dict(AGENT_FP="diff"), dict(FOOTPRINT=0), dict(VOLBUDGET=1)):
        e = scene(bt, flat_tape(300), {1: [(MM, "BUY", 500)], 2: [(USER, "BUY", 700)]}, 4, **env)
        assert sorted(btm.agent_flow_log(e).values()) == [0.0, 0.0, 0.0, 500.0], env


# --------------------------------------------------------------------------- #
# (d) volume budget
# --------------------------------------------------------------------------- #
V = 3_000


def _bar_volumes(e):
    return [b["v"] for b in e._closed_bars]


def test_volbudget_cuts_the_printed_volume_by_the_agents_volume(bt):
    """flat tape, 3 000 shares a bar. The agent buys 1 000: the bar's printed volume is 3 000 - 1 000, so
    the bar (printed + the agent's trade) holds the real 3 000 again."""
    off = scene(bt, flat_tape(300, v=V), {1: [(MM, "BUY", 1_000)]}, 4)
    on = scene(bt, flat_tape(300, v=V), {1: [(MM, "BUY", 1_000)]}, 4, VOLBUDGET=1)
    assert _bar_volumes(off) == [V, V + 1_000, V, V]           # today: volume is added on top
    assert _bar_volumes(on) == [V, V, V, V]                    # replaced
    st = btm.get_state(on)
    assert st["agent_aggr_total"] == 1_000 and st["vol_cut_total"] == 1_000 and st["vol_debt"] == 0.0


def test_volbudget_debt_carries_and_cut_never_exceeds_maxcut(bt):
    e = make(bt, flat_tape(300, v=V), m=100, VOLBUDGET=1)
    e.physics_tick()
    order(e, MM, "BUY", 5_000)                                  # debt 5 000 against 3 000-share bars
    e.physics_tick()
    st = btm.get_state(e)
    assert st["vol_cut_total"] == 2_700 and st["vol_debt"] == 2_300                  # 0.9 * 3 000
    assert e._closed_bars[-1]["v"] == (V - 2_700) + 5_000
    e.physics_tick()                                            # the rest of the debt: 2 300 < 2 700
    st = btm.get_state(e)
    assert st["vol_cut_total"] == 5_000 and st["vol_debt"] == 0.0
    assert e._closed_bars[-1]["v"] == V - 2_300
    e.physics_tick()
    assert e._closed_bars[-1]["v"] == V and btm.get_state(e)["vol_cut_total"] == 5_000


def test_volbudget_never_cuts_more_than_maxcut_of_any_bar(bt):
    bars = walk_tape(sessions=2)
    e = make(bt, bars, m=M, VOLBUDGET=1, VOLBUDGET_MAXCUT=0.5)
    cuts, prev_total = [], 0.0
    for k in range(2 * M):
        if k % 5 == 2:
            order(e, MM if k % 2 else "WHALE_1", "BUY" if k % 3 else "SELL", 4_000)
        e.physics_tick()
        cut = btm.get_state(e)["vol_cut_total"] - prev_total
        prev_total += cut
        assert cut <= 0.5 * bars[k]["v"] + 1e-9, k
        cuts.append(cut)
    st = btm.get_state(e)
    assert sum(cuts) == st["vol_cut_total"] > 0
    assert st["vol_cut_total"] + st["vol_debt"] == pytest.approx(st["agent_aggr_total"])


def test_volbudget_never_subtracts_the_users_volume(bt):
    on = scene(bt, flat_tape(300, v=V), {1: [(USER, "BUY", 2_000), ("STRAT_PROBE", "SELL", 500)]}, 4, VOLBUDGET=1)
    st = btm.get_state(on)
    assert st["vol_cut_total"] == 0.0 and st["vol_debt"] == 0.0 and st["agent_aggr_total"] == 0.0
    assert _bar_volumes(on)[1] == V + 2_500                     # the user's volume is extra
    both = scene(bt, flat_tape(300, v=V), {1: [(USER, "BUY", 2_000), (MM, "SELL", 700)]}, 4, VOLBUDGET=1)
    assert btm.get_state(both)["vol_cut_total"] == 700
    assert _bar_volumes(both)[1] == V + 2_000


def test_volbudget_counts_both_directions_and_skips_internal(bt):
    e = scene(bt, flat_tape(300, v=V), {1: [(MM, "BUY", 400), ("RETAIL_1", "SELL", 300)]}, 3, VOLBUDGET=1)
    st = btm.get_state(e)
    assert st["agent_aggr_total"] == 700 and st["vol_cut_total"] == 700    # SIM_LP / GHOST_HIST not counted


def test_volbudget_keeps_the_floor_for_tiny_bars(bt):
    """A bar with fewer shares than path targets still prints one share per target."""
    bars = [{"o": P0, "h": P0 + 0.05, "l": P0 - 0.05, "c": P0, "v": 6} for _ in range(60)]
    e = make(bt, bars, m=60, VOLBUDGET=1)
    e.physics_tick()
    order(e, MM, "BUY", 20)
    for _ in range(6):
        e.physics_tick()
    st = btm.get_state(e)
    assert st["n_bars"] == 7 and st["vol_cut_total"] <= 0.9 * 6 * 7 and st["n_inexact"] >= 1
    assert all(b["v"] >= 1 for b in e._closed_bars)


def test_volbudget_off_has_no_debt(bt):
    e = scene(bt, flat_tape(300, v=V), {1: [(MM, "BUY", 5_000)]}, 4)
    st = btm.get_state(e)
    assert st["vol_debt"] == 0.0 and st["vol_cut_total"] == 0.0 and st["agent_aggr_total"] == 5_000


# --------------------------------------------------------------------------- #
# (e) depth budget
# --------------------------------------------------------------------------- #
def lp_side(e, bid):
    """{distance in ticks: qty} of the SIM_LP orders on one side."""
    book = e.book._bids if bid else e.book._asks
    out = {}
    P = round(e.last_price / TICK) * TICK
    for key in book.keys():
        for o in book[key]:
            if o["agent_id"] == e.LP_AGENT:
                px = -key if bid else key
                d = round((P - px) / TICK) if bid else round((px - P) / TICK)
                out[d] = out.get(d, 0) + o["qty"]
    return out


def cum_within(side, d):
    return sum(q for k, q in side.items() if k <= d)


def F(d):
    return round(ADV * (d * TICK / (P0 * SIGMA)) ** 2)


def depth_engine(bt, **env):
    """Engine at 100.00 with an empty book; returns (engine, state)."""
    e = make(bt, flat_tape(300), m=100, **env)
    st = btm._new_state("env", "t", SIGMA, ADV, 0.0, P0, 0)
    return e, st


def test_depthbudget_without_agents_is_identical_to_today(bt):
    bars = walk_tape(sessions=2)
    a = scene(bt, bars, {5: [(USER, "BUY", 9_000)], 12: [(USER, "SELL", 4_000)]}, 2 * M, m=M)
    b = scene(bt, bars, {5: [(USER, "BUY", 9_000)], 12: [(USER, "SELL", 4_000)]}, 2 * M, m=M, DEPTHBUDGET=1)
    assert list(a._closed_bars) == list(b._closed_bars) and a.book.depth(80) == b.book.depth(80)
    sa, sb = btm.get_state(a), btm.get_state(b)
    assert sb["depth_displaced_total"] == 0 and sa["latent_posted_total"] == sb["latent_posted_total"] > 0
    sb.pop("depthbudget"), sa.pop("depthbudget")
    assert sa == sb


def test_depthbudget_agent_depth_replaces_latent_depth(bt):
    e, st = depth_engine(bt, DEPTHBUDGET=1)
    P = e.last_price
    assert P == P0
    base_st = btm._new_state("env", "t", SIGMA, ADV, 0.0, P, 0)
    base_st["depthbudget"] = 0
    btm._post_book(e, [], base_st, 1.0)
    base_b, base_a = lp_side(e, True), lp_side(e, False)
    assert cum_within(base_b, 2) == F(2) == 400 and base_b[1] == 100

    order(e, MM, "BUY", 60, "LIMIT", round(P - TICK, 2))                 # 60 shares at distance 1
    order(e, "WHALE_1", "SELL", 90, "LIMIT", round(P + 2 * TICK, 2))     # 90 shares at distance 2
    order(e, USER, "BUY", 70, "LIMIT", round(P - 2 * TICK, 2))           # the user's resting order: no effect
    btm._post_book(e, [], st, 1.0)
    lb, la = lp_side(e, True), lp_side(e, False)
    assert cum_within(lb, 2) == F(2) - 60 and lb[1] == 40
    assert cum_within(la, 2) == F(2) - 90 and la[1] == 100 and la[2] == 400 - 90 - 100
    for d in (1, 2, 3, 5, 9, 12, 20, 50):                                # total depth within d stays F(d)
        if d in lb:
            assert cum_within(lb, d) + 60 == F(d), d
            assert cum_within(la, d) + 90 == F(d) or d < 2, d
    assert st["depth_displaced_total"] == 60 + 90
    assert st["latent_posted_total"] == sum(lb.values()) + sum(la.values())
    # the user's resting bid is untouched by the budget and the agents' orders are still there
    assert e.book.best_bid() == pytest.approx(P - TICK)


def test_depthbudget_total_depth_follows_F_everywhere(bt):
    e, st = depth_engine(bt, DEPTHBUDGET=1)
    P = e.last_price
    order(e, MM, "BUY", 60, "LIMIT", round(P - TICK, 2))
    order(e, MM, "BUY", 500, "LIMIT", round(P - 6 * TICK, 2))
    btm._post_book(e, [], st, 1.0)
    lb = lp_side(e, True)
    agents = {1: 60, 6: 500}
    for d in sorted(lb):
        assert cum_within(lb, d) + sum(q for k, q in agents.items() if k <= d) == pytest.approx(F(d), abs=3), d


def test_depthbudget_skips_levels_that_agents_exceed(bt):
    e, st = depth_engine(bt, DEPTHBUDGET=1)
    P = e.last_price
    order(e, MM, "BUY", 500, "LIMIT", round(P - TICK, 2))                # 500 > F(2) = 400
    btm._post_book(e, [], st, 1.0)
    lb = lp_side(e, True)
    assert 1 not in lb and 2 not in lb                                   # a level can be skipped
    assert cum_within(lb, 3) == F(3) - 500                               # and the total is max(F, agents)
    assert min(lb) == 3


def test_depthbudget_order_beyond_the_reference_counts_everywhere(bt):
    e, st = depth_engine(bt, DEPTHBUDGET=1)
    P = e.last_price
    order(e, MM, "BUY", 50, "LIMIT", round(P + 3 * TICK, 2))             # a bid above the reference
    btm._post_book(e, [], st, 1.0)
    lb = lp_side(e, True)
    assert lb[1] == 100 - 50 and cum_within(lb, 2) == F(2) - 50 and cum_within(lb, 9) == F(9) - 50
    assert st["depth_displaced_total"] == 50


def test_depthbudget_latent_never_crosses(bt):
    e, st = depth_engine(bt, DEPTHBUDGET=1)
    P = e.last_price
    order(e, MM, "BUY", 500, "LIMIT", round(P + 0.03, 2))
    order(e, "MM_2", "SELL", 500, "LIMIT", round(P + 0.20, 2))
    order(e, "MM_3", "BUY", 300, "LIMIT", round(P - 0.04, 2))
    btm._post_book(e, [], st, 1.0)
    bids, asks = e.book.depth(200)
    assert e.book.best_bid() == pytest.approx(P + 0.03) and e.book.best_bid() < e.book.best_ask()
    assert all(px < P + 0.20 - 1e-9 for px, _ in bids)
    lp_asks = [px for px, _ in asks if px < P + 0.20 - 1e-9]
    assert lp_asks and min(lp_asks) > P + 0.03 + 1e-9
    assert all(q > 0 for _, q in bids + asks)


def test_depthbudget_in_a_run_never_crosses_and_posts_no_empty_orders(bt):
    bars = walk_tape(sessions=2)
    e = make(bt, bars, m=M, DEPTHBUDGET=1)
    for k in range(2 * M):
        if k % 4 == 1:
            order(e, MM, "BUY" if k % 8 == 1 else "SELL", 300, "LIMIT",
                  round(e.last_price + (-1 if k % 8 == 1 else 1) * 2 * TICK, 2))
        e.physics_tick()
        bb, ba = e.book.best_bid(), e.book.best_ask()
        assert bb is None or ba is None or bb < ba, k
    assert btm.get_state(e)["latent_posted_total"] > 0


def test_depthbudget_reads_the_book_without_changing_it(bt):
    e, st = depth_engine(bt, DEPTHBUDGET=1)
    P = e.last_price
    order(e, MM, "BUY", 60, "LIMIT", round(P - TICK, 2))
    order(e, "WHALE_1", "SELL", 90, "LIMIT", round(P + 2 * TICK, 2))
    ref_i = int(round(P / TICK))
    ds = btm._distances(50, 40)
    before = (e.book.depth(200), len(e.book._index))
    cb = btm._agent_cum(e, ref_i, ds, True)
    ca = btm._agent_cum(e, ref_i, ds, False)
    assert (e.book.depth(200), len(e.book._index)) == before
    assert cb[0] == 60 and cb[-1] == 60 and ca[0] == 0 and ca[1] == 90 and ca[-1] == 90
    assert cb == sorted(cb) and ca == sorted(ca)


# --------------------------------------------------------------------------- #
# volume budget 2: passive sweeps of the leg's ghost
# --------------------------------------------------------------------------- #
def _rest_then_print(bt, who, qty, vol_mode, ticks=3, v=V):
    """Flat tape (one flat leg per bar, target 100.00). ``who`` rests ``qty`` shares of ask at 100.00 after
    the first bar; the next bar's ghost sweeps it."""
    e = make(bt, flat_tape(300, v=v), m=100, VOLBUDGET=vol_mode)
    e.physics_tick()
    ack = order(e, who, "SELL", qty, "LIMIT", 100.0)
    assert ack["status"] == "RESTING"
    for _ in range(ticks - 1):
        e.physics_tick()
    return e


def test_volbudget2_agent_resting_below_leg_qty_replaces_ladder(bt):
    X = 1_000
    off = _rest_then_print(bt, MM, X, 0, ticks=2)
    on = _rest_then_print(bt, MM, X, 2, ticks=2)
    assert _bar_volumes(off) == [V, V + X]                      # today: the swept order is on top
    assert _bar_volumes(on) == [V, V]                           # X from the agent + (V - X) from the ladder
    st = btm.get_state(on)
    assert st["vol_cut_passive_total"] == X and st["passive_excess_total"] == 0 and st["vol_debt"] == 0.0
    assert st["vol_cut_total"] == 0 and on.agent_ledger[MM]["pos"] == -X     # the agent really sold
    assert closes(on) == closes(off)


def test_volbudget2_agent_at_least_leg_qty_keeps_one_ladder_share(bt):
    X = 4_000
    e = _rest_then_print(bt, MM, X, 2, ticks=2)
    assert e._closed_bars[-1]["v"] == X + 1
    st = btm.get_state(e)
    assert st["vol_cut_passive_total"] == V - 1 and st["passive_excess_total"] == X - (V - 1)
    assert st["vol_debt"] == X - (V - 1) == 1_001
    e.physics_tick()                                            # the next bar is cut by the debt
    assert e._closed_bars[-1]["v"] == V - 1_001
    st = btm.get_state(e)
    assert st["vol_debt"] == 0.0 and st["vol_cut_total"] == 1_001


def test_volbudget2_debt_cut_is_limited_by_maxcut(bt):
    e = _rest_then_print(bt, MM, 20_000, 2, ticks=2)
    st = btm.get_state(e)
    assert e._closed_bars[-1]["v"] == 20_001 and st["vol_debt"] == 20_000 - (V - 1)
    debt = st["vol_debt"]
    e.physics_tick()
    assert e._closed_bars[-1]["v"] == V - 2_700                # 0.9 * 3 000
    assert btm.get_state(e)["vol_debt"] == debt - 2_700


def test_volbudget2_users_resting_order_is_not_subtracted(bt):
    e = _rest_then_print(bt, USER, 1_000, 2, ticks=2)
    assert _bar_volumes(e) == [V, V + 1_000]
    st = btm.get_state(e)
    assert st["vol_cut_passive_total"] == 0 and st["passive_excess_total"] == 0 and st["vol_debt"] == 0.0
    ref = _rest_then_print(bt, USER, 1_000, 0, ticks=2)
    assert _bar_volumes(ref) == _bar_volumes(e) and closes(ref) == closes(e)


def test_volbudget2_agent_and_user_resting_together(bt):
    e = make(bt, flat_tape(300, v=V), m=100, VOLBUDGET=2)
    e.physics_tick()
    order(e, MM, "SELL", 500, "LIMIT", 100.0)
    order(e, USER, "SELL", 700, "LIMIT", 100.0)
    e.physics_tick()
    assert e._closed_bars[-1]["v"] == V + 700                   # only the agent's 500 replaced ladder volume
    assert btm.get_state(e)["vol_cut_passive_total"] == 500


def test_volbudget2_mode1_is_unchanged_by_the_passive_code(bt):
    X = 1_000
    m1 = _rest_then_print(bt, MM, X, 1, ticks=3)
    assert _bar_volumes(m1) == [V, V + X, V]                    # passive volume is not touched in mode 1
    st = btm.get_state(m1)
    assert st["vol_cut_passive_total"] == 0 and st["passive_excess_total"] == 0 and st["vol_debt"] == 0.0
    bars = walk_tape(sessions=2)
    script = {4: [(MM, "BUY", 800)], 9: [(USER, "BUY", 6_000)], 15: [("WHALE_1", "SELL", 3_000)],
              20: [("RETAIL_1", "BUY", 200)], 30: [(USER, "SELL", 2_000), (MM, "SELL", 500)]}
    runs = []
    for env in (dict(VOLBUDGET=1), dict(VOLBUDGET=1, VOLBUDGET_MAXCUT=0.9, AGENT_FP="all", DEPTHBUDGET=0)):
        e = scene(bt, bars, script, 2 * M, m=M, **env)
        runs.append((list(e._closed_bars), btm.get_state(e), e.book.depth(60)))
    assert runs[0] == runs[1]
    s1 = runs[0][1]
    assert s1["vol_cut_total"] > 0 and s1["vol_cut_passive_total"] == 0 and s1["passive_excess_total"] == 0


def test_volbudget2_without_agents_equals_volbudget0(bt):
    bars = walk_tape(sessions=2, zero_vol_at=(7,))
    script = {5: [(USER, "BUY", 9_000)], 12: [(USER, "SELL", 4_000)], 25: [("STRAT_PROBE", "BUY", 2_500)]}
    a = scene(bt, bars, script, 2 * M, m=M)
    b = scene(bt, bars, script, 2 * M, m=M, VOLBUDGET=2)
    assert list(a._closed_bars) == list(b._closed_bars) and a.book.depth(80) == b.book.depth(80)
    sa, sb = btm.get_state(a), btm.get_state(b)
    assert sb["vol_cut_passive_total"] == 0 and sb["passive_excess_total"] == 0 and sb["vol_cut_total"] == 0
    sa.pop("volbudget"), sb.pop("volbudget")
    assert sa == sb


def test_volbudget2_volume_is_conserved_over_time(bt):
    """An agent re-rests 1 500 shares at the price every bar: with mode 2 the bars add up to the tape."""
    def total(mode):
        e = make(bt, flat_tape(300, v=V), m=100, VOLBUDGET=mode)
        e.physics_tick()
        for _ in range(20):
            order(e, MM, "SELL", 1_500, "LIMIT", 100.0)
            e.physics_tick()
        return sum(_bar_volumes(e)), e
    t0, _ = total(0)
    t2, e2 = total(2)
    assert t0 == 21 * V + 20 * 1_500 and t2 == 21 * V
    assert btm.get_state(e2)["vol_cut_passive_total"] == 20 * 1_500
