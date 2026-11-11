"""Backtest mode (agents/backtest_mode.py): the real tape keeps running with the participants' footprint.

Hermetic: synthetic tapes registered with register_tape(); the real RELIANCE file is not read. ``install()``
patches a class attribute, so the ``bt`` fixture saves and restores MatchingEngine._refresh_sim_lp and the
module's own flags/caches around every test.

Synthetic instrument: price 100 (tick 0.01), adv = 1e6 shares, sigma_d = 1 %, Y = 1, so the displayed book
holds F(d) = 1e6 (d / 100)^2 shares within d ticks (100 at the first tick, 250 000 at d_max = 50) and a
market order of 20 000 shares walks to about 14 ticks (u(Q) = 1.4e-3).
"""
import copy
import importlib
import logging
import math
import os
import runpy
import sys

import numpy as np
import pytest

from agents import backtest_mode as btm
from engine.matching_engine import MatchingEngine
from training.envs.replay_from_bars import start_replay_from_bars

M = 40                      # bars per (short) session; engines get SESSION_MINUTES = M
P0 = 100.0
TICK = 0.01
ADV = 1e6
SIGMA = 0.01
STRAT = "STRAT_T"


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def bt(monkeypatch):
    """Environment for one test: mode on, tape selected by key 't', all BT parameters at defaults."""
    from engine.replay import HistoricalReplayer
    saved = (MatchingEngine._refresh_sim_lp, btm._INSTALLED, dict(btm._TAPES), dict(btm._DIST_CACHE),
             MatchingEngine.handle_message, HistoricalReplayer.step)
    for k in list(os.environ):
        if k.startswith("CHRONOS_BT_") or k in ("CHRONOS_V2_SEED_SKIP", "CHRONOS_MODE", "CHRONOS_TAPE_END"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("CHRONOS_MODE", "backtest")
    monkeypatch.setenv("CHRONOS_BT_TAPE", "t")
    btm._INSTALLED = False
    yield monkeypatch
    MatchingEngine._refresh_sim_lp = saved[0]
    MatchingEngine.handle_message = saved[4]
    HistoricalReplayer.step = saved[5]
    btm._INSTALLED = saved[1]
    btm._TAPES.clear()
    btm._TAPES.update(saved[2])
    btm._DIST_CACHE.clear()
    btm._DIST_CACHE.update(saved[3])


def walk_tape(sessions=3, m=M, seed=1, start=P0, sigma=6e-4, gap=30e-4, zero_vol_at=()):
    """Random-walk tape on the 0.01 grid with overnight gaps; returns bars (dicts o,h,l,c,v)."""
    rng = np.random.default_rng(seed)
    bars, prev = [], start
    for s in range(sessions):
        o = round(prev * math.exp(rng.normal(0, gap)), 2) if s else prev
        for k in range(m):
            c = round(o * math.exp(rng.normal(0, sigma)), 2)
            h = round(max(o, c) + abs(rng.normal(0, 0.02)), 2)
            l = round(min(o, c) - abs(rng.normal(0, 0.02)), 2)
            v = int(rng.integers(1500, 4000))
            if len(bars) in zero_vol_at:
                c = h = l = o
                v = 0
            bars.append({"o": o, "h": h, "l": l, "c": c, "v": v})
            o = c
        prev = o
    return bars


def flat_tape(n, price=P0, v=3000):
    return [{"o": price, "h": price, "l": price, "c": price, "v": v} for _ in range(n)]


def make(bt, bars, start=None, prev_close=P0, sigma=SIGMA, adv=ADV, tod=None, m=M, seed=3, **env):
    """Register the tape, install, return a running LIVE engine started at ``start``."""
    for k, v in env.items():
        bt.setenv(f"CHRONOS_BT_{k}", str(v))
    btm.register_tape("t", bars, prev_close, sigma, adv, tod)
    btm.install()
    e = MatchingEngine(db=None)
    e.SESSION_MINUTES = m
    e.init_sim("SYN", "S", P0 if start is None else start, seed=seed)
    e.paused = False
    return e


def order(e, agent, side, qty, typ="MARKET", price=None):
    msg = {"msg": "ORDER", "agent_id": agent, "action": side, "type": typ, "qty": int(qty)}
    if price is not None:
        msg["price"] = price
    ack, _ = e.handle_message(msg)
    return ack


def closes(e):
    return [b["c"] for b in e._closed_bars]


def equity(e, agent=STRAT):
    led = e.agent_ledger[agent]
    return led["cash"] + led["pos"] * e.last_price


# --------------------------------------------------------------------------- #
# 1. no-op without CHRONOS_MODE=backtest
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", [None, "", "replay", "Backtest"])
def test_install_is_noop_without_backtest_mode(bt, mode):
    if mode is None:
        bt.delenv("CHRONOS_MODE")
    else:
        bt.setenv("CHRONOS_MODE", mode)
    before = MatchingEngine._refresh_sim_lp
    btm.install()
    assert MatchingEngine._refresh_sim_lp is before
    assert not btm.active()
    # an engine run is untouched: no state is created
    e = MatchingEngine(db=None)
    e.init_sim("SYN", "S", P0, seed=3)
    e.paused = False
    e.physics_tick()
    assert btm.get_state(e) is None


def test_install_is_idempotent(bt):
    btm.register_tape("t", flat_tape(10), P0, SIGMA, ADV)
    btm.install()
    wrapped = MatchingEngine._refresh_sim_lp
    btm.install()
    assert MatchingEngine._refresh_sim_lp is wrapped
    assert btm.active()


# --------------------------------------------------------------------------- #
# 2. zero flow: the tape is printed exactly
# --------------------------------------------------------------------------- #
def test_zero_flow_prints_the_tape_exactly(bt):
    bars = walk_tape(sessions=3, zero_vol_at=(7, M + 3))
    e = make(bt, bars)
    for _ in range(len(bars)):
        e.physics_tick()
    got = list(e._closed_bars)
    assert len(got) == len(bars)
    gaps = 0
    for k, (b, r) in enumerate(zip(got, bars)):
        assert b["c"] == pytest.approx(r["c"], abs=1e-9), k
        if r["v"] >= 4:
            assert b["v"] == r["v"], k
        if r["v"] == 0:
            assert b["v"] == 0 and b["o"] == b["c"] == b["h"] == b["l"], k
        if k % M == 0 and k and abs(r["o"] - bars[k - 1]["c"]) > 1e-9:
            gaps += 1
            assert b["o"] == pytest.approx(r["o"], abs=1e-9)      # the overnight gap is printed
    assert gaps >= 1
    st = btm.get_state(e)
    assert st["u"] == 0.0 and st["xp"] == 0.0 and st["xt"] == 0.0 and st["n_bars"] == len(bars)


# --------------------------------------------------------------------------- #
# 3. level scaling
# --------------------------------------------------------------------------- #
def test_level_is_the_simulators_own(bt):
    bars = walk_tape(sessions=2)
    e = make(bt, bars, start=1.37 * P0)
    for _ in range(len(bars)):
        e.physics_tick()
    st = btm.get_state(e)
    assert st["base"] == pytest.approx(math.log(1.37))
    for b, r in zip(e._closed_bars, bars):
        assert abs(b["c"] - 1.37 * r["c"]) <= e.tick + 1e-9


# --------------------------------------------------------------------------- #
# 4. footprint
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cap", [1, 2])
def test_footprint_of_a_walking_buy(bt, cap):
    m = 100
    bars = flat_tape(3 * m)
    e = make(bt, bars, m=m, CAP=cap)
    e.physics_tick()
    ack = order(e, STRAT, "BUY", 20_000)
    assert ack["executed_qty"] == 20_000                       # walks the displayed book
    vwap = ack["average_price"]
    worst = max(t["price"] for t in e._pending_trades if t["buyer"] == STRAT)
    assert vwap > P0 + 5 * TICK
    e.physics_tick()
    st = btm.get_state(e)
    x1 = st["xp"] + st["xt"]
    bound = vwap if cap == 1 else worst                        # CAP=1: average price paid; CAP=2: highest
    assert 0 < st["u"] <= math.log(bound / P0) + 1e-9          # never above the price the buyer paid
    if cap == 2:
        assert x1 == pytest.approx(20_000, rel=0.05)           # a block that walks the book is not capped
    us, cl = [st["u"]], [e._closed_bars[-1]["c"]]
    for _ in range(250):
        e.physics_tick()
        s = btm.get_state(e)
        us.append(s["u"])
        cl.append(e._closed_bars[-1]["c"])
    assert min(us) >= 0 and min(cl) >= P0                      # never changes sign
    assert all(b <= a + 1e-12 for a, b in zip(us, us[1:]))     # and only fades
    assert cl[0] > P0 + 3 * TICK and cl[60] > P0               # later closes sit above the tape
    for u, c in zip(us, cl):                                   # printed level = tape level * exp(u)
        assert abs(math.log(c / P0) - u) <= 0.6 * TICK / P0 + 1e-12
    s = btm.get_state(e)
    u_inf = btm.curve_u(s["perm"] * x1, s)
    assert s["u"] == pytest.approx(u_inf, rel=0.02)
    assert s["xp"] == pytest.approx(s["perm"] * x1)


# --------------------------------------------------------------------------- #
# 5. one block vs slices
# --------------------------------------------------------------------------- #
def _consume(bt, slices, Q=20_000, ticks=10, **env):
    m = 100
    e = make(bt, flat_tape(3 * m), m=m, **env)
    e.physics_tick()
    for k in range(ticks):
        if k < slices:
            assert order(e, STRAT, "BUY", Q // slices)["executed_qty"] == Q // slices
        e.physics_tick()
    s = btm.get_state(e)
    return s["xp"] + s["xt"], s


@pytest.mark.parametrize("cap", [0, 2])
def test_block_and_slices_consume_the_same_volume(bt, cap):
    """Single Q vs 10 slices of Q/10 (one per minute). The temporary part has decayed for a different time
    in the two cases (block: 9 minutes, slice j: 10 - j minutes), so each result is divided by the decay
    factor its own schedule implies before comparing. Tolerance 20 %: the book is discretised to ticks and
    the slices walk a book that is re-posted around a level they have moved themselves."""
    x1, s = _consume(bt, 1, CAP=cap)
    x10, _ = _consume(bt, 10, CAP=cap)
    rho, perm = s["rho"], s["perm"]
    f1 = perm + (1 - perm) * rho ** 9
    f10 = perm + (1 - perm) * np.mean([rho ** k for k in range(10)])
    assert (x1 / f1) / (x10 / f10) == pytest.approx(1.0, abs=0.20)
    assert (x1 / f1) == pytest.approx(20_000, rel=0.1)


def test_block_and_slices_without_decay(bt):
    x1, _ = _consume(bt, 1, CAP=2, HALFLIFE=1e9)
    x10, _ = _consume(bt, 10, CAP=2, HALFLIFE=1e9)
    assert x1 == pytest.approx(20_000, rel=0.01) and x10 == pytest.approx(20_000, rel=0.01)


def test_average_price_cap_consumes_less_for_a_block_than_for_slices(bt):
    """Documents why the default is CAP=2: with CAP=1 (cap at the AVERAGE price paid) a block that walks
    the book moves the level to (2/3) u(Q) only, i.e. consumes about 4/9 of Q, while the slices consume all
    of it. CAP=2 (cap at the highest price paid) removes the gap, see the tests above."""
    x1, _ = _consume(bt, 1, CAP=1, HALFLIFE=1e9)
    x10, _ = _consume(bt, 10, CAP=1, HALFLIFE=1e9)
    assert 0.4 < x1 / x10 < 0.75
    assert x10 == pytest.approx(20_000, rel=0.1)


# --------------------------------------------------------------------------- #
# 6. no profit from own impact (book only, flat tape)
# --------------------------------------------------------------------------- #
Q = 8_000


def _schedule(name):
    """minute -> list of (leg, fraction of Q or None = close the whole position) ; leg 'in'/'out'."""
    if name == "block_in_block_out_next":
        return {0: [("in", 1.0)], 1: [("out", None)]}
    if name == "block_in_block_out_30":
        return {0: [("in", 1.0)], 30: [("out", None)]}
    if name == "block_in_20_slices_out":
        s = {0: [("in", 1.0)]}
        s.update({m: [("out", 1 / 20)] for m in range(1, 20)})
        s[20] = [("out", None)]
        return s
    if name == "20_slices_in_block_out":
        s = {m: [("in", 1 / 20)] for m in range(20)}
        s[20] = [("out", None)]
        return s
    if name == "ping_pong":
        return {0: [("in", 0.125), ("out", None)] * 20}
    raise KeyError(name)


PATTERNS = ["block_in_block_out_next", "block_in_block_out_30", "block_in_20_slices_out",
            "20_slices_in_block_out", "ping_pong"]


def _run_pattern(bt, name, first, **env):
    m = 100
    e = make(bt, flat_tape(3 * m), m=m, **env)
    e.physics_tick()
    sched = _schedule(name)
    sign_in = "BUY" if first == "buy" else "SELL"
    sign_out = "SELL" if first == "buy" else "BUY"
    for minute in range(max(sched) + 3):
        for leg, frac in sched.get(minute, []):
            pos = e.agent_ledger[STRAT]["pos"] if STRAT in e.agent_ledger else 0
            if leg == "in":
                order(e, STRAT, sign_in, int(Q * frac))
            elif pos != 0:
                qty = abs(pos) if frac is None else int(Q * frac)
                order(e, STRAT, sign_out, qty)
        e.physics_tick()
    led = e.agent_ledger[STRAT]
    assert led["pos"] == 0, "pattern did not close its position"
    return equity(e) - led["start_cash"]


@pytest.mark.parametrize("first", ["buy", "sell"])
@pytest.mark.parametrize("name", PATTERNS)
def test_no_profit_from_own_impact(bt, name, first):
    assert _run_pattern(bt, name, first) <= 0.0                # default cap (CAP=2)


@pytest.mark.parametrize("name", PATTERNS)
def test_no_profit_from_own_impact_cap1(bt, name):
    assert _run_pattern(bt, name, "buy", CAP=1) <= 0.0
    assert _run_pattern(bt, name, "sell", CAP=1) <= 0.0


# --------------------------------------------------------------------------- #
# 7. the cap, with flat third-party liquidity
# --------------------------------------------------------------------------- #
def _flat_liquidity_roundtrip(bt, **env):
    m = 100
    e = make(bt, flat_tape(3 * m), m=m, **env)
    e.physics_tick()
    px = e.last_price
    ask = round(px + TICK, 2)
    order(e, "MM_T", "SELL", 20_000, "LIMIT", ask)
    ack = order(e, STRAT, "BUY", 3_000)
    assert ack["executed_qty"] == 3_000 and ack["average_price"] == pytest.approx(ask)   # no walking
    e.physics_tick()
    new_px = e.last_price
    bid = round(new_px - TICK, 2)
    order(e, "MM_T", "BUY", 20_000, "LIMIT", bid)
    sold = order(e, STRAT, "SELL", 3_000)
    assert sold["executed_qty"] == 3_000 and sold["average_price"] == pytest.approx(bid)
    return e, ask, new_px, equity(e) - e.agent_ledger[STRAT]["start_cash"]


def test_cap_stops_the_premium_round_trip(bt):
    e, ask, new_px, pnl = _flat_liquidity_roundtrip(bt)
    st = btm.get_state(e)
    assert st["u"] <= math.log(ask / P0) + 1e-9                # level rises by at most the premium paid
    assert new_px <= ask + 1e-9
    assert pnl <= 0


def test_without_the_cap_the_same_pattern_profits(bt):
    e, ask, new_px, pnl = _flat_liquidity_roundtrip(bt, CAP=0)
    assert new_px > ask + 2 * TICK                              # the level ran away from the price paid
    assert pnl > 0                                              # why the cap exists


def test_average_price_cap_also_stops_the_premium_round_trip(bt):
    e, ask, new_px, pnl = _flat_liquidity_roundtrip(bt, CAP=1)
    assert new_px <= ask + 1e-9 and pnl <= 0


# --------------------------------------------------------------------------- #
# 8. snapshot determinism
# --------------------------------------------------------------------------- #
def test_snapshot_determinism(bt):
    bars = walk_tape(sessions=3)
    e = make(bt, bars)
    script = {3: ("BUY", 6_000), 9: ("SELL", 2_500), 15: ("BUY", 12_000), 22: ("SELL", 4_000), 31: ("SELL", 5_000)}

    def go(eng, lo, hi):
        for k in range(lo, hi):
            if k in script:
                order(eng, STRAT, *script[k])
            eng.physics_tick()

    go(e, 0, 20)
    twin = copy.deepcopy(e)
    go(e, 20, 70)
    go(twin, 20, 70)
    assert list(e._closed_bars) == list(twin._closed_bars)
    assert btm.get_state(e) == btm.get_state(twin)
    assert btm.get_state(e)["u"] != 0.0


# --------------------------------------------------------------------------- #
# 9. replayer continuation
# --------------------------------------------------------------------------- #
def _dated(bars, m=M):
    out = []
    for k, b in enumerate(bars):
        day, minute = divmod(k, m)
        out.append(dict(b, ts=1_760_000_000 + day * 86400 + minute * 60))
    return out


def _replay_engine(bt, bars, m=M, seed=5):
    bt.delenv("CHRONOS_BT_TAPE", raising=False)
    btm.install()
    e = MatchingEngine(db=None)
    e.SESSION_MINUTES = m
    start_replay_from_bars(e, bars, "SYN", "S", seed=seed)
    e.paused = False
    return e


def test_replayer_continuation_keeps_printing_the_tape(bt):
    bars = _dated(walk_tape(sessions=3, m=M)[: 3 * M - 7])        # ends mid-session on purpose
    e = _replay_engine(bt, bars)
    for _ in range(10):
        e.physics_tick()
    assert e.mode == "REPLAY"
    assert btm.get_state(e) is None
    assert closes(e) == pytest.approx([b["c"] for b in bars[:10]])   # plain replay, untouched
    ack = order(e, STRAT, "BUY", 600)                                # fills the bracket -> latch
    assert ack["executed_qty"] == 600 and e.mode == "REACTIVE"
    vwap = ack["average_price"]
    cur0 = e.replayer.cursor
    assert cur0 == 10

    rows = []
    n_left = len(bars) - cur0
    for _ in range(n_left):
        e.physics_tick()
        rp = e.replayer
        st = btm.get_state(e)
        rows.append((rp.cursor, e._closed_bars[-1]["c"], bars[rp.cursor - 1]["c"], st["u"], e.sim_unix_time,
                     bars[rp.cursor - 1]["ts"]))
    assert not e.paused and not e.replayer.finished
    assert [r[0] for r in rows] == list(range(cur0 + 1, len(bars) + 1))
    for cursor, sim_c, real_c, u, unix, ts in rows:
        assert abs(math.log(sim_c / real_c) - u) <= 1.2 * e.tick / real_c      # bars follow the tape + footprint
        assert unix == ts                                                      # the file's own times
    assert rows[0][3] > 0 and rows[0][3] <= math.log(vwap / bars[9]["c"]) + 1e-9
    # relative moves follow the tape: the sim path = tape path x exp(u), u smooth
    d_sim = np.diff(np.log([r[1] for r in rows]))
    d_real = np.diff(np.log([r[2] for r in rows]))
    assert np.max(np.abs(d_sim - d_real)) < 0.002
    # tape exhausted: the engine pauses once
    events = []
    for _ in range(3):
        events.append([p for p in e.physics_tick() if p[0] == "event"])
    assert e.paused and e.replayer.finished
    assert [len(x) for x in events] == [1, 0, 0]
    assert events[0][0][1]["action"] == "day_close"
    assert e.replayer.cursor == len(bars)


def test_replayer_continuation_mid_step_latch(bt):
    """A resting STRAT_T bid that the replayer's own print walks through latches inside HistoricalReplayer.step:
    that tick must not consume a second bar."""
    base = walk_tape(sessions=1, m=M)
    drop = dict(o=base[4]["c"], h=base[4]["c"], l=round(base[4]["c"] - 0.30, 2), c=round(base[4]["c"] - 0.25, 2), v=5000)
    bars = _dated(base[:5] + [drop] + base[5:20])
    e = _replay_engine(bt, bars)
    for _ in range(5):
        e.physics_tick()
    px = e.last_price
    ack = order(e, STRAT, "BUY", 100, "LIMIT", round(px - 0.10, 2), )
    assert ack["status"] == "RESTING"
    e.physics_tick()                                              # bar 6 (the drop) fills the bid -> latch
    assert e.mode == "REACTIVE" and e.replayer.cursor == 6
    st = btm.get_state(e)
    assert st is not None and st["n_bars"] == 0                   # no second bar in this tick
    assert st["c_prev"] == pytest.approx(bars[5]["c"])
    e.physics_tick()
    assert e.replayer.cursor == 7 and btm.get_state(e)["n_bars"] == 1


# --------------------------------------------------------------------------- #
# 10. env tape exhausted -> previous liquidity refresh, no exception
# --------------------------------------------------------------------------- #
def test_env_tape_exhaustion_falls_back(bt, caplog):
    calls = []
    orig = MatchingEngine._refresh_sim_lp

    def spy(self, pubs, reference=None):
        calls.append(reference)
        return orig(self, pubs, reference)

    MatchingEngine._refresh_sim_lp = spy
    bars = walk_tape(sessions=2)
    e = make(bt, bars)
    with caplog.at_level(logging.WARNING, logger="agents.backtest_mode"):
        for _ in range(2 * M + 6):
            e.physics_tick()
    assert len(calls) == 6                                         # spy ran only after the tape ended
    st = btm.get_state(e)
    assert st["exhausted"] and st["n_bars"] == 2 * M
    assert sum("exhausted" in r.message for r in caplog.records) == 1
    bids, asks = e.book.depth(20)
    assert bids and asks and e._closed_bars[-1]["c"] > 0


# --------------------------------------------------------------------------- #
# the displayed book
# --------------------------------------------------------------------------- #
def test_book_follows_the_single_curve(bt):
    e = make(bt, flat_tape(3 * M))
    e.physics_tick()
    bids, asks = e.book.depth(100)
    P = e.last_price
    assert bids[0][0] == pytest.approx(P - TICK) and asks[0][0] == pytest.approx(P + TICK)
    assert len(bids) == len(asks) <= 40
    d_max = math.ceil(P * SIGMA * 0.5 / TICK)
    assert round((asks[-1][0] - P) / TICK) == d_max and round((P - bids[-1][0]) / TICK) == d_max
    for side in (asks, bids):
        ticks = [round(abs(p - P) / TICK) for p, _ in side]
        assert ticks[:12] == list(range(1, 13)) and all(b > a for a, b in zip(ticks, ticks[1:]))
        cum = 0
        for d, (_, q) in zip(ticks, side):
            cum += q
            F = ADV * (d * TICK / (P * SIGMA)) ** 2
            assert cum == pytest.approx(F, rel=0.01, abs=len(side))


def test_market_order_walks_to_the_curve(bt):
    e = make(bt, flat_tape(3 * M))
    e.physics_tick()
    P = e.last_price
    ack = order(e, STRAT, "BUY", 20_000)
    assert ack["executed_qty"] == 20_000
    u = btm.curve_u(20_000, btm.get_state(e))
    assert e.last_price == pytest.approx(P * math.exp(u), abs=2 * TICK)


def test_book_never_crosses_other_participants(bt):
    """Other participants' quotes sit where the new book would cross them (the print itself normally sweeps
    such quotes, so build the situation directly): LP bids stay strictly below the best other ask, LP asks
    strictly above the best other bid."""
    e = make(bt, flat_tape(3 * M))
    P = e.last_price
    order(e, "MM_T", "BUY", 500, "LIMIT", round(P + 0.03, 2))      # rests: the book is still empty
    order(e, "MM_T2", "SELL", 500, "LIMIT", round(P + 0.20, 2))
    assert e.book.best_bid() == pytest.approx(P + 0.03)
    st = btm._new_state("env", "t", SIGMA, ADV, 0.0, P, 0)
    btm._post_book(e, [], st, 1.0)
    bids, asks = e.book.depth(200)
    assert e.book.best_bid() == pytest.approx(P + 0.03)            # nothing traded
    lp_asks = [px for px, q in asks if px < P + 0.20 - 1e-9]
    assert lp_asks and min(lp_asks) > P + 0.03 + 1e-9
    assert max(px for px, _ in bids) == pytest.approx(P + 0.03)
    assert all(px < P + 0.20 - 1e-9 for px, _ in bids)
    assert e.book.best_bid() < e.book.best_ask()


def test_footprint_off_is_plain_replay(bt):
    bars = walk_tape(sessions=2)
    e = make(bt, bars, FOOTPRINT=0)
    for k in range(len(bars)):
        if k in (5, 12, 40):
            order(e, STRAT, "BUY", 15_000)
        e.physics_tick()
    for b, r in zip(e._closed_bars, bars):
        assert b["c"] == pytest.approx(r["c"], abs=1e-9)
    assert btm.get_state(e)["u"] == 0.0


def test_scale_from_replayed_bars():
    """sigma_d, adv from complete replayed days; 1-minute fallback with fewer than three."""
    m = 20
    rng = np.random.default_rng(0)
    bars, px = [], 100.0
    for d in range(5):
        for k in range(m):
            px *= math.exp(rng.normal(0, 0.001))
            bars.append({"o": px, "h": px, "l": px, "c": px, "v": 1000})
    sig, adv = btm._scale_from_bars(bars, len(bars), m)
    closes_d = [bars[(d + 1) * m - 1]["c"] for d in range(5)]
    assert sig == pytest.approx(np.std(np.diff(np.log(closes_d)), ddof=1))
    assert adv == pytest.approx(m * 1000)
    sig2, adv2 = btm._scale_from_bars(bars, 30, m)               # one full day: 1-minute fallback
    r = np.diff(np.log([b["c"] for b in bars[:30]]))
    assert sig2 == pytest.approx(np.std(r, ddof=1) * math.sqrt(m)) and adv2 == pytest.approx(1000 * m)
    sig3, adv3 = btm._scale_from_bars(bars, 1, m)                # nothing to estimate: defaults
    assert sig3 > 0 and adv3 > 0


def test_env_tape_scale_is_causal_and_cached(bt):
    """env_tape() of a registered tape exposes the contract keys; real-data scale uses sessions before the tape."""
    btm.register_tape("t", walk_tape(sessions=2), P0, SIGMA, ADV)
    t = btm.env_tape()
    for k in ("prev_close", "adv", "sigma_d", "dates", "o", "h", "l", "c", "v", "symbol", "skip"):
        assert k in t
    assert len(t["c"]) == 2 * M and t["prev_close"] == P0
    c2 = np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.01, 12 * 6))).reshape(12, 6) * 100
    v2 = np.full((12, 6), 100.0)
    prof = btm._depth_profile(c2, v2, window=3)
    assert prof.shape == (6,) and prof.mean() == pytest.approx(1.0, abs=0.3) and prof.min() >= 0.25 and prof.max() <= 4


# --------------------------------------------------------------------------- #
# launcher
# --------------------------------------------------------------------------- #
def test_launcher_import_does_not_start_the_server(monkeypatch):
    from engine.replay import HistoricalReplayer
    server = importlib.import_module("engine.server")
    called = []
    monkeypatch.setattr(server, "main", lambda: called.append(1))
    monkeypatch.delenv("CHRONOS_MODE", raising=False)
    had_end = os.environ.pop("CHRONOS_TAPE_END", None)
    saved = (MatchingEngine._refresh_sim_lp, MatchingEngine.handle_message, HistoricalReplayer.step, btm._INSTALLED)
    saved_ff = {n: MatchingEngine.__dict__[n] for n in ("_apply_fill", "start_replay", "init_sim")}
    try:
        sys.modules.pop("tools.run_engine_backtest", None)
        mod = importlib.import_module("tools.run_engine_backtest")
        assert called == [] and callable(mod.main)
        sys.modules.pop("tools.run_engine_backtest", None)
        runpy.run_module("tools.run_engine_backtest", run_name="__main__")   # python -m tools.run_engine_backtest
        assert called == [1]
        # the switches are installed, the mode is off (no CHRONOS_MODE): a normal engine until the dashboard says so
        assert getattr(MatchingEngine.handle_message, "_bt_wrapped", False)
        assert os.environ["CHRONOS_TAPE_END"] == "continue"
        e = MatchingEngine(db=None)
        assert not btm.backtest_enabled(e) and btm.tape_end_policy(e) == "continue"
        assert getattr(MatchingEngine._apply_fill, "_ff_wrapped", False)        # the fast-forward button too
    finally:
        (MatchingEngine._refresh_sim_lp, MatchingEngine.handle_message, HistoricalReplayer.step,
         btm._INSTALLED) = saved
        for n, f in saved_ff.items():
            setattr(MatchingEngine, n, f)
        os.environ.pop("CHRONOS_TAPE_END", None)
        if had_end is not None:
            os.environ["CHRONOS_TAPE_END"] = had_end
