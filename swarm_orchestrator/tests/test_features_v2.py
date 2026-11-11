"""Observation builder v2 (agents/features_v2.py): spec, parity, real engine run, AgentBook."""
import csv
import math
import os
import random
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import features_v2 as F
from agents.features_v2 import (AgentBook, OBS_SPEC, build_observation_v2, obs_dict,
                                sector_index, spec_hash)
from core.market_memory import MarketMemory
from training.data.sectors import SECTORS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_CSV = os.path.join(ROOT, "data", "real", "nse_kaggle", "RELIANCE_minute.csv")
SESSION = 375
ROLES = ("whale", "mm")


# --------------------------------------------------------------------------- helpers
def synth_bars(sessions, seed=7, start_price=100.0):
    rng = random.Random(seed)
    price = start_price
    out = []
    for d in range(sessions):
        vol = 0.0004 + 0.0004 * rng.random()
        scale = 0.5 + rng.random()
        for m in range(1, SESSION + 1):
            o = price
            c = max(o * math.exp(rng.gauss(0.00001, vol)), 1.0)
            h = max(o, c) * (1.0 + abs(rng.gauss(0, vol / 2)))
            l = min(o, c) * (1.0 - abs(rng.gauss(0, vol / 2)))
            v = round(1000 * scale * (0.5 + rng.random()))
            out.append({"day_count": d + 1, "minute": m, "o": o, "h": h, "l": l, "c": c, "v": v})
            price = c
    return out


def as_tick(b, run_id="r1", sector="Energy", **extra):
    t = {"run_id": run_id, "paused": False, "day_count": b["day_count"],
         "market_minute": b["minute"], "unix_time": 1767258900 + 86400 * (b["day_count"] - 1)
         + 60 * b["minute"], "sector": sector, "symbol": "X",
         "last_price": b["c"], "vwap": b["c"] * 0.999, "volume_ma": 1000.0,
         "step_volume": b["v"], "rsi": 55.0, "ofi": 0.1, "volatility": 0.001,
         "bar": {"minute": b["minute"], "o": b["o"], "h": b["h"], "l": b["l"],
                 "c": b["c"], "v": b["v"]}}
    t.update(extra)
    return t


def day_tick(day, price, minute=100, **extra):
    t = {"day_count": day, "market_minute": minute, "last_price": price}
    t.update(extra)
    return t


def vec_dict(role, vec):
    return obs_dict(role, vec)


class StubMemory:
    """Minimal MarketMemory stand-in with fully controlled features."""

    def __init__(self, overrides=None, adv=None):
        self._f = dict.fromkeys(MarketMemory.FEATURE_NAMES, 0.0)
        self._f.update(overrides or {})
        self._adv = adv

    def features(self):
        return dict(self._f)

    def avg_daily_volume(self, n=20):
        return self._adv


BASE_TICK = as_tick(synth_bars(1)[10])


# --------------------------------------------------------------------------- 1. spec
def test_lengths_and_names():
    mem = MarketMemory()
    book = AgentBook(1e6, 1e4, n_outputs=3)
    book4 = AgentBook(1e6, 1e4, n_outputs=4)
    for role, b in (("whale", book), ("mm", book4)):
        names = OBS_SPEC[role]
        assert len(names) == len(set(names))
        v = build_observation_v2(role, BASE_TICK, mem, b)
        assert v.shape == (len(names),) and v.dtype == np.float32
        assert np.isfinite(v).all()
        assert list(obs_dict(role, v)) == list(names)
    assert len(MarketMemory.FEATURE_NAMES) == 58
    assert sum(n.startswith("mem_") for n in OBS_SPEC["whale"]) == 58
    assert sum(n.startswith("last_action_") for n in OBS_SPEC["whale"]) == 3
    assert sum(n.startswith("last_action_") for n in OBS_SPEC["mm"]) == 4


def test_whale_superset_of_mm_memory_and_spec_contents():
    w = set(OBS_SPEC["whale"])
    m = {n for n in OBS_SPEC["mm"] if n.startswith("mem_")}
    assert m < {n for n in w if n.startswith("mem_")}
    assert {"cap_vs_adv", "traded_vs_adv", "risk_appetite"} <= w
    assert not {"cap_vs_adv", "traded_vs_adv", "risk_appetite"} & set(OBS_SPEC["mm"])
    for role in ROLES:
        for n in ("vwap_dist", "bar_range", "volume_ratio", "rsi", "order_flow", "tod_sin",
                  "tod_cos", "to_close", "dow_sin", "dow_cos", "inventory", "open_pnl",
                  "drawdown", "holding", "pnl_today", "exposure"):
            assert n in OBS_SPEC[role]
        assert [n for n in OBS_SPEC[role] if n.startswith("sector_")] == \
            ["sector_" + s for s in SECTORS]
    # the memory inputs the market maker uses
    assert "mem_lvl_up1_age" in OBS_SPEC["mm"] and "mem_lvl_up2_dist" not in OBS_SPEC["mm"]
    assert "mem_ret_5d" not in OBS_SPEC["mm"] and "mem_atr_regime" in OBS_SPEC["mm"]


def test_spec_hash_stable_and_role_specific():
    assert spec_hash("whale") == spec_hash("whale")
    assert spec_hash("whale") != spec_hash("mm")
    assert len(spec_hash("mm")) == 64


# --------------------------------------------------------------------------- 2. market / time / sector values
def test_market_time_values():
    mem = StubMemory()
    book = AgentBook(1e6, 1e4)
    t = dict(BASE_TICK, last_price=101.0, vwap=100.0, volume_ma=50.0, step_volume=100,
             rsi=70.0, ofi=-0.25, market_minute=SESSION // 3 * 1, unix_time=1767258900)
    t["bar"] = {"minute": 1, "o": 100, "h": 102, "l": 99, "c": 100.0, "v": 1}
    d = vec_dict("whale", build_observation_v2("whale", t, mem, book))
    assert d["vwap_dist"] == pytest.approx(0.01)
    assert d["bar_range"] == pytest.approx(0.03)
    assert d["volume_ratio"] == pytest.approx(2.0)
    assert d["rsi"] == pytest.approx(0.2)
    assert d["order_flow"] == pytest.approx(-0.25)
    assert d["tod_sin"] == pytest.approx(math.sin(2 * math.pi * 125 / 375), abs=1e-6)
    assert d["to_close"] == pytest.approx(250 / 375)
    # 2026-01-01 is a Thursday -> weekday 3 of 5
    assert d["dow_sin"] == pytest.approx(math.sin(2 * math.pi * 3 / 5), abs=1e-6)
    assert d["dow_cos"] == pytest.approx(math.cos(2 * math.pi * 3 / 5), abs=1e-6)
    # guards
    t2 = dict(t, vwap=0.0, volume_ma=0.0)
    d2 = vec_dict("mm", build_observation_v2("mm", t2, mem, book))
    assert d2["vwap_dist"] == 0.0 and d2["volume_ratio"] == 1.0
    # no sim time -> weekday inputs 0, 0
    t3 = dict(t)
    del t3["unix_time"]
    d3 = vec_dict("mm", build_observation_v2("mm", t3, mem, book))
    assert d3["dow_sin"] == 0.0 and d3["dow_cos"] == 0.0


def test_unknown_sector_is_unknown_one_hot():
    mem = StubMemory()
    book = AgentBook(1e6, 1e4)
    for sec in ("Atlantis", None, 17, ""):
        d = vec_dict("mm", build_observation_v2("mm", dict(BASE_TICK, sector=sec), mem, book))
        hot = [k for k, v in d.items() if k.startswith("sector_") and v == 1.0]
        assert hot == ["sector_UNKNOWN"], sec
    t = {k: v for k, v in BASE_TICK.items() if k != "sector"}      # absent field
    d = vec_dict("mm", build_observation_v2("mm", t, mem, book))
    assert d["sector_UNKNOWN"] == 1.0


def test_live_sector_aliases():
    assert SECTORS[sector_index("TECH")] == "Information Technology"
    assert SECTORS[sector_index("bank")] == "Financial Services"
    assert SECTORS[sector_index("OIL")] == "Energy"
    assert SECTORS[sector_index("DEFENSE")] == "Industrials"
    assert SECTORS[sector_index("Energy")] == "Energy"
    assert SECTORS[sector_index("healthcare")] == "Healthcare"
    mem = StubMemory()
    d = vec_dict("whale", build_observation_v2("whale", dict(BASE_TICK, sector="PHARMA"),
                                               mem, AgentBook(1e6, 1e4)))
    assert [k for k, v in d.items() if k.startswith("sector_") and v] == ["sector_Healthcare"]


# --------------------------------------------------------------------------- 3. parity
def test_update_vs_seed_parity():
    bars = synth_bars(6)[:5 * SESSION + 140]
    a, b = MarketMemory(), MarketMemory()
    book = AgentBook(1e6, 1e4)
    book.on_fill("BUY", 100, 100.0)
    last_tick = None
    for bar in bars:
        last_tick = as_tick(bar)
        a.update(last_tick)
    assert b.seed(bars, "r1") == len(bars)
    book.on_tick(last_tick)
    for role in ROLES:
        va = build_observation_v2(role, last_tick, a, book)
        vb = build_observation_v2(role, last_tick, b, book)
        assert np.array_equal(va, vb)
    assert a.avg_daily_volume() == b.avg_daily_volume() is not None


# --------------------------------------------------------------------------- 4. clipping, ADV
def test_unbounded_ratios_clipped():
    big = {"rv30_ratio": 55.0, "rv_today_ratio": 1e6, "jump_10m": -300.0}
    for role in ROLES:
        d = vec_dict(role, build_observation_v2(role, BASE_TICK, StubMemory(big),
                                                AgentBook(1e6, 1e4)))
        assert d["mem_rv30_ratio"] == 10.0 and d["mem_rv_today_ratio"] == 10.0
        assert d["mem_jump_10m"] == -10.0
    d = vec_dict("mm", build_observation_v2("mm", BASE_TICK, StubMemory({"jump_10m": 99.0,
                 "rv30_ratio": -1.0}), AgentBook(1e6, 1e4)))
    assert d["mem_jump_10m"] == 10.0 and d["mem_rv30_ratio"] == 0.0
    d = vec_dict("whale", build_observation_v2("whale", BASE_TICK, StubMemory({"rv30_ratio": 4.2}),
                                               AgentBook(1e6, 1e4)))
    assert d["mem_rv30_ratio"] == pytest.approx(4.2)        # in range: untouched


def test_adv_cold_start_and_warm():
    book = AgentBook(1e6, capacity=5_000.0)
    book.on_fill("BUY", 1_000, 100.0)
    t = dict(BASE_TICK, volume_ma=40.0)
    # cold start: no completed session -> ADV = 375 * volume_ma = 15000
    d = vec_dict("whale", build_observation_v2("whale", t, StubMemory(adv=None), book))
    assert d["cap_vs_adv"] == pytest.approx(5_000 / 15_000)
    assert d["traded_vs_adv"] == pytest.approx(1_000 / 15_000)
    # warm: the memory's ADV is used
    d = vec_dict("whale", build_observation_v2("whale", t, StubMemory(adv=50_000.0), book))
    assert d["cap_vs_adv"] == pytest.approx(0.1) and d["traded_vs_adv"] == pytest.approx(0.02)
    # real memory: cold at the start, warm after a completed session
    mem = MarketMemory()
    bars = synth_bars(3)
    for bar in bars[:100]:
        mem.update(as_tick(bar))
    assert mem.sessions == 0 and mem.avg_daily_volume() is None
    for bar in bars[100:]:
        mem.update(as_tick(bar))
    assert mem.sessions == 2
    adv = mem.avg_daily_volume(20)
    per_day = [sum(b["v"] for b in bars if b["day_count"] == d) for d in (1, 2)]
    assert adv == pytest.approx(sum(per_day) / 2)
    d = vec_dict("whale", build_observation_v2("whale", t, mem, book))
    assert d["cap_vs_adv"] == pytest.approx(5_000 / adv)
    # zero volume everywhere: no division error
    d = vec_dict("whale", build_observation_v2("whale", dict(t, volume_ma=0.0),
                                               StubMemory(adv=None), book))
    assert d["cap_vs_adv"] == 0.0 and d["traded_vs_adv"] == 0.0


def test_risk_appetite_range():
    for tol, want in ((0.03, 0.0), (0.09, 0.5), (0.15, 1.0)):
        b = AgentBook(1e6, 1e4, risk_tolerance=tol)
        d = vec_dict("whale", build_observation_v2("whale", BASE_TICK, StubMemory(), b))
        assert d["risk_appetite"] == pytest.approx(want)


def test_last_action_slots():
    b = AgentBook(1e6, 1e4, n_outputs=3)
    b.set_last_action([0.5, -1.0, 0.25])
    d = vec_dict("whale", build_observation_v2("whale", BASE_TICK, StubMemory(), b))
    assert [d["last_action_0"], d["last_action_1"], d["last_action_2"]] == [0.5, -1.0, 0.25]
    b4 = AgentBook(1e6, 1e4, n_outputs=4)
    b4.set_last_action([1, 2])                                  # padded
    assert b4.last_action == [1.0, 2.0, 0.0, 0.0]
    b4.set_last_action([float("nan"), 1, 2, 3, 4])              # scrubbed / truncated
    assert b4.last_action == [0.0, 1.0, 2.0, 3.0]
    d = vec_dict("mm", build_observation_v2("mm", BASE_TICK, StubMemory(), b4))
    assert d["last_action_3"] == 3.0


# --------------------------------------------------------------------------- 5. AgentBook
def test_book_buy_then_sell_pnl_and_fees():
    b = AgentBook(capital=1_000_000, capacity=10_000)
    b.on_tick(day_tick(1, 100.0))
    b.on_fill("BUY", 100, 100.0, fee=1.0)
    assert b.position == 100 and b.avg_price == 100.0
    assert b.realised_pnl == -1.0                       # the opening fee is already a loss
    b.on_tick(day_tick(1, 105.0))
    assert b.open_pnl == pytest.approx(500 / 1e6)
    assert b.exposure == pytest.approx(100 * 105 / 1e6)
    assert b.inventory == pytest.approx(0.01)
    b.on_fill("SELL", 100, 110.0, fee=1.5)
    assert b.position == 0 and b.avg_price == 0.0
    assert b.realised_pnl == pytest.approx(1000 - 2.5)
    assert b.fees_paid == pytest.approx(2.5)
    assert b.pnl_today == pytest.approx(997.5 / 1e6)
    assert b.open_pnl == 0.0 and b.shares_traded_today == 200
    assert b.equity == pytest.approx(1_000_000 + 997.5)


def test_book_average_short_partial_and_flip():
    b = AgentBook(1e6, 1e4)
    b.on_tick(day_tick(1, 100.0))
    b.on_fill("BUY", 100, 100.0)
    b.on_fill("BUY", 100, 110.0)
    assert b.avg_price == pytest.approx(105.0) and b.position == 200
    b.on_fill("SELL", 50, 115.0)                        # partial close
    assert b.realised_pnl == pytest.approx(50 * 10) and b.position == 150
    assert b.avg_price == pytest.approx(105.0)
    b.on_fill("SELL", 200, 100.0)                       # closes 150 (-5 each), shorts 50
    assert b.realised_pnl == pytest.approx(500 - 750)
    assert b.position == -50 and b.avg_price == 100.0
    b.on_tick(day_tick(1, 90.0))
    assert b.unrealised_pnl == pytest.approx(500.0)     # short gains when price falls
    b.on_fill("BUY", 50, 90.0)
    assert b.realised_pnl == pytest.approx(500 - 750 + 500) and b.position == 0
    # short from flat
    b2 = AgentBook(1e6, 1e4)
    b2.on_fill("sell", 10, 100.0)
    b2.on_fill("buy", 10, 90.0, fee=2.0)
    assert b2.realised_pnl == pytest.approx(100 - 2)
    b2.on_fill("BUY", 0, 90.0)                          # ignored
    b2.on_fill("BUY", 5, -1.0)                          # ignored
    assert b2.position == 0


def test_book_holding_counts_sessions():
    b = AgentBook(1e6, 1e4)
    b.on_tick(day_tick(1, 100.0, minute=1))
    b.on_fill("BUY", 10, 100.0)
    b.on_tick(day_tick(1, 100.0, minute=2))
    assert b.holding_sessions == 0 and b.holding == 0.0
    for d in (2, 3, 4):
        b.on_tick(day_tick(d, 100.0, minute=1))
        b.on_tick(day_tick(d, 100.0, minute=2))        # same session: no extra count
        assert b.holding_sessions == d - 1
    assert b.holding == pytest.approx(3 / 20)
    b.on_fill("SELL", 10, 100.0)                       # flat -> reset
    assert b.holding_sessions == 0
    b.on_tick(day_tick(5, 100.0))
    assert b.holding_sessions == 0                     # not in a position: no count
    b.on_fill("BUY", 10, 100.0)
    for d in range(6, 40):
        b.on_tick(day_tick(d, 100.0))
    assert b.holding_sessions == 34 and b.holding == 1.0     # capped at 20 sessions
    b.on_fill("SELL", 25, 100.0)                       # flips short -> a fresh position
    assert b.position == -15 and b.holding_sessions == 0


def test_book_drawdown_nonneg_and_window():
    b = AgentBook(1_000_000, 10_000)
    b.on_tick(day_tick(1, 100.0))
    assert b.drawdown == 0.0
    b.on_fill("BUY", 1000, 100.0)
    b.on_tick(day_tick(1, 120.0))                      # equity 1,020,000 (peak)
    assert b.drawdown == 0.0
    b.on_tick(day_tick(1, 110.0))                      # 1,010,000
    assert b.drawdown == pytest.approx(10_000 / 1_020_000)
    assert b.open_pnl == pytest.approx(10_000 / 1e6)
    b.on_tick(day_tick(1, 90.0))                       # below the start: still >= 0
    assert b.drawdown == pytest.approx(30_000 / 1_020_000)
    # random walk: never negative
    rng = random.Random(3)
    p = 100.0
    for d in range(1, 30):
        for m in range(1, 5):
            p *= math.exp(rng.gauss(0, 0.02))
            b.on_tick(day_tick(d + 1, p, minute=m))
            assert b.drawdown >= 0.0
    # the peak of the first session falls out of the 5-session window
    b2 = AgentBook(1_000_000, 10_000)
    b2.on_fill("BUY", 1000, 100.0)
    b2.on_tick(day_tick(1, 200.0))                     # peak equity 1,100,000 in session 1
    for d in range(2, 7):                              # five later sessions at a lower level
        b2.on_tick(day_tick(d, 100.0))
    assert b2.drawdown == 0.0                          # the 200.0 peak left the window
    b2.on_tick(day_tick(6, 90.0))
    assert b2.drawdown == pytest.approx(10_000 / 1_000_000)


def test_book_new_session_resets_today():
    b = AgentBook(1e6, 1e4)
    b.on_tick(day_tick(1, 100.0))
    b.on_fill("BUY", 100, 100.0)
    b.on_fill("SELL", 100, 102.0)
    assert b.pnl_today == pytest.approx(200 / 1e6) and b.shares_traded_today == 200
    b.on_tick(day_tick(1, 101.0, minute=200))          # same session: kept
    assert b.shares_traded_today == 200
    b.on_tick(day_tick(2, 101.0, minute=1))            # new session
    assert b.pnl_today == 0.0 and b.shares_traded_today == 0.0
    assert b.realised_pnl == pytest.approx(200)        # cumulative P&L survives
    b.on_fill("BUY", 30, 101.0)
    assert b.shares_traded_today == 30


def test_book_before_first_tick_and_validation():
    b = AgentBook(1e6, 1e4)
    assert (b.inventory, b.open_pnl, b.drawdown, b.holding, b.pnl_today, b.exposure) == \
        (0.0,) * 6
    b.on_fill("BUY", 10, 100.0)                         # fill before any tick
    assert b.unrealised_pnl == 0.0 and b.exposure == 0.0
    with pytest.raises(ValueError):
        AgentBook(0, 1)
    with pytest.raises(ValueError):
        AgentBook(1, 0)


def test_book_deterministic():
    def run():
        b = AgentBook(1e6, 1e4)
        rng = random.Random(5)
        out = []
        p = 100.0
        for d in range(1, 8):
            for m in range(1, 50):
                p *= math.exp(rng.gauss(0, 0.003))
                b.on_tick(day_tick(d, p, minute=m))
                if rng.random() < 0.1:
                    b.on_fill(rng.choice(("BUY", "SELL")), rng.randint(1, 80), p, 0.5)
                out.append((b.inventory, b.open_pnl, b.drawdown, b.holding, b.pnl_today,
                            b.exposure))
        return out
    assert run() == run()


# --------------------------------------------------------------------------- 6. speed
def test_speed_overhead_over_memory_features():
    bars = synth_bars(25)
    mem = MarketMemory()
    for bar in bars[:-10]:
        mem.update(as_tick(bar))
    tick = as_tick(bars[-10])
    mem.update(tick)
    book = AgentBook(1e6, 1e4)
    book.on_tick(tick)
    book.on_fill("BUY", 100, tick["last_price"])
    feats = mem.features()
    for role in ROLES:
        best = 1e9
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(500):
                build_observation_v2(role, tick, mem, book, feats=feats)
            best = min(best, (time.perf_counter() - t0) / 500)
        print(f"build_observation_v2[{role}] overhead: {best * 1e6:.1f} us")
        assert best < 0.2e-3, f"{role}: {best * 1e6:.0f} us"
    # passing feats= is only a cache: same vector as computing it inside
    assert np.array_equal(build_observation_v2("whale", tick, mem, book),
                          build_observation_v2("whale", tick, mem, book, feats=feats))


# --------------------------------------------------------------------------- 7. real engine run
def _complete_days(n=3):
    out = []
    cur_date, cur = None, []

    def flush():
        if cur_date and len(cur) == SESSION and cur[0]["date"].endswith("09:15:00") \
                and cur[-1]["date"].endswith("15:29:00"):
            out.append((cur_date, list(cur)))

    with open(REAL_CSV, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            date = row["date"][:10]
            if date != cur_date:
                flush()
                if len(out) >= n:
                    return out
                cur_date, cur = date, []
            cur.append(row)
        flush()
    return out[:n]


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="real NSE data not present")
def test_real_engine_replay_three_sessions(tmp_path):
    from engine.matching_engine import MatchingEngine

    days = _complete_days(3)
    if len(days) < 3:
        pytest.skip("fewer than 3 complete sessions in the RELIANCE file")
    path = tmp_path / "reliance3.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "time", "open", "high", "low", "close", "volume"])
        for _, rows in days:
            for r in rows:
                w.writerow([r["date"][:10], r["date"][11:16], r["open"], r["high"], r["low"], r["close"], r["volume"]])

    eng = MatchingEngine(db=None)
    _, n_bars = eng.start_replay(str(path), "RELIANCE", "Energy", seed=1)
    assert n_bars == 3 * SESSION
    eng.paused = False
    mem = MarketMemory()
    books = {"whale": AgentBook(1e7, 5_000, n_outputs=3), "mm": AgentBook(1e6, 2_000, n_outputs=4)}
    rows = {r: [] for r in ROLES}
    for i in range(n_bars):
        pubs = eng.physics_tick()
        tick = next(p for topic, p in pubs if topic == "tick")
        mem.update(tick)
        for role in ROLES:
            book = books[role]
            book.on_tick(tick)
            vec = build_observation_v2(role, tick, mem, book)
            assert vec.shape == (len(OBS_SPEC[role]),)
            assert np.isfinite(vec).all(), (role, i, [n for n, v in zip(OBS_SPEC[role], vec)
                                                       if not np.isfinite(v)])
            rows[role].append(vec)
            if i % 50 == 7:                              # trade a little so own-state moves
                book.on_fill("BUY" if (i // 50) % 2 == 0 else "SELL", 100, tick["last_price"], 0.5)
                book.set_last_action(np.full(book.n_outputs, 0.1 * ((i // 50) % 5)))
    assert mem.sessions == 2
    for role in ROLES:
        arr = np.stack(rows[role])
        assert arr.shape == (n_bars, len(OBS_SPEC[role]))
        const = [n for n, c in zip(OBS_SPEC[role], arr.std(axis=0) == 0.0) if c]
        live = len(OBS_SPEC[role]) - len(const)
        print(f"\n[{role}] {len(OBS_SPEC[role])} inputs, {live} vary, constant: {const}")
        # sector is fixed in a run; exactly one sector slot is hot and it is Energy
        assert arr[0, OBS_SPEC[role].index("sector_Energy")] == 1.0
        # the core market inputs must actually move
        for n in ("vwap_dist", "bar_range", "order_flow", "tod_sin", "mem_ret_1m",
                  "inventory", "exposure"):
            assert n not in const, n
