"""MarketMemory (core/market_memory.py): the long (one-year) market memory."""
import csv
import json
import math
import os
import random

import pytest

from core.market_memory import MarketMemory, _bin

SESSION = 375
REAL_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "real", "nse_kaggle", "RELIANCE_minute.csv")


# --------------------------------------------------------------------------- helpers
def synth_bars(sessions, seed=7, start_price=100.0, first_day=1, start_minute=0):
    """Deterministic random-walk bars: dicts {day_count, minute, o, h, l, c, v}."""
    rng = random.Random(seed)
    price = start_price
    out = []
    for d in range(sessions):
        # a slow regime per day so volatility / volume vary between sessions
        day_vol = 0.0004 + 0.0004 * rng.random()
        day_scale = 0.5 + rng.random()
        for m in range(SESSION):
            o = price
            c = max(o * math.exp(rng.gauss(0.00001, day_vol)), 1.0)
            h = max(o, c) * (1.0 + abs(rng.gauss(0, day_vol / 2)))
            l = min(o, c) * (1.0 - abs(rng.gauss(0, day_vol / 2)))
            u = 1.0 + 1.5 * (abs(m - SESSION / 2) / (SESSION / 2)) ** 2   # U-shaped volume
            v = round(1000 * u * day_scale * (0.5 + rng.random()))
            out.append({"day_count": first_day + d, "minute": start_minute + m,
                        "o": o, "h": h, "l": l, "c": c, "v": v})
            price = c
    return out


def as_tick(b, run_id="r1", paused=False):
    return {"run_id": run_id, "paused": paused, "day_count": b["day_count"],
            "market_minute": b["minute"],
            "bar": {"minute": b["minute"], "o": b["o"], "h": b["h"], "l": b["l"],
                    "c": b["c"], "v": b["v"]}}


def add(mm, b):
    return mm.add_bar(b["day_count"], b["minute"], b["o"], b["h"], b["l"], b["c"], b["v"])


def snap(mm):
    return mm.features(), mm.levels(), mm.levels(10, 10)


def assert_same(a, b):
    fa, la, wa = a
    fb, lb, wb = b
    assert list(fa) == list(fb) == list(MarketMemory.FEATURE_NAMES)
    for k in fa:
        assert fa[k] == pytest.approx(fb[k], rel=1e-9, abs=1e-12), k
    assert la == lb
    assert wa == wb


@pytest.fixture(scope="module")
def bars40():
    return synth_bars(40)


# --------------------------------------------------------------------------- 1. feeding paths agree
def test_update_seed_add_bar_agree(bars40):
    a, b, c = MarketMemory(), MarketMemory(), MarketMemory()
    for i, bar in enumerate(bars40):
        assert a.update(as_tick(bar)) is True
        assert add(c, bar) is True
        if i in (500, 1000, 8000, 14999):
            sa = snap(a)
            sc = snap(c)
            assert_same(sa, sc)
    assert b.seed(bars40, "r1") == len(bars40)
    assert_same(snap(a), snap(b))
    assert_same(snap(a), snap(c))
    assert a.sessions == b.sessions == c.sessions == 39
    assert len(a.FEATURE_NAMES) == len(set(a.FEATURE_NAMES))


def test_seed_then_live_tick_repeat_ignored(bars40):
    mm = MarketMemory()
    mm.seed(bars40[:1000], "r1")
    n = mm.bars_seen
    assert mm.update(as_tick(bars40[999])) is False        # repeated last bar
    assert mm.update(as_tick(bars40[1000])) is True        # continues the run
    assert mm.bars_seen == n + 1


# --------------------------------------------------------------------------- 2. no look-ahead
@pytest.mark.parametrize("t", [100, 374, 375, 376, 3000, 3750 * 2 + 200, 9 * 375 + 5])
def test_no_look_ahead(bars40, t):
    full = MarketMemory()
    for bar in bars40[:t + 1]:
        add(full, bar)
    prefix_snap = snap(full)

    # same prefix, then a different (wild) future: the state at t must not change
    alt = []
    rng = random.Random(99)
    for bar in bars40[t + 1:t + 2000]:
        bar = dict(bar)
        bar["c"] = bar["c"] * (1.0 + rng.uniform(-0.1, 0.1))
        bar["h"] = max(bar["o"], bar["c"]) * 1.05
        bar["l"] = min(bar["o"], bar["c"]) * 0.95
        alt.append(bar)
    other = MarketMemory()
    snaps = {}
    for i, bar in enumerate(bars40[:t + 1]):
        add(other, bar)
    snaps["at_t"] = snap(other)
    for bar in alt:
        add(other, bar)
    assert_same(prefix_snap, snaps["at_t"])


def test_swing_pivot_needs_three_later_sessions():
    # 12 flat-ish sessions; session 4 gets one spike high -> a pivot only after 3 later sessions
    bars = synth_bars(12, seed=3)
    spike = 100 * 1.6
    for b in bars:
        if b["day_count"] == 4 and b["minute"] == 200:
            b["h"] = spike
    def has_spike_level(mm):
        lv = mm.levels(60, 60)
        return any(abs(L["price"] - spike) < 1e-9 and "swing_high" in L["kind"]
                   for L in lv["above"] + lv["below"])
    mm = MarketMemory()
    seen = {}
    for b in bars:
        add(mm, b)
        seen[(b["day_count"], b["minute"])] = None
        # sessions == completed days. Pivot (index 3) needs sessions >= 7
        if b["minute"] == 5:
            if mm.sessions < 7:
                assert not has_spike_level(mm), mm.sessions
    # after day 8 started, 7 sessions are complete -> the pivot is confirmed
    mm2 = MarketMemory()
    for b in bars:
        if b["day_count"] <= 8:
            add(mm2, b)
    assert mm2.sessions == 7
    assert has_spike_level(mm2)


# --------------------------------------------------------------------------- 3. bounded
def test_bounded_and_profile_rolls():
    bars = synth_bars(400, seed=11)
    mm = MarketMemory()
    for b in bars:
        add(mm, b)
    assert mm.sessions == 250
    assert len(mm._days) <= 250
    assert len(mm._bars) <= 20 * SESSION
    assert len(mm._day_profs) <= 250
    assert len(mm._trs) <= 250
    assert len(mm._imin) <= 20
    assert len(mm._rets) <= 30

    # rebuild from the last 250 completed sessions (150..399) + today (400)
    expect = {}
    for b in bars:
        if b["day_count"] >= 150:
            tp = (b["h"] + b["l"] + b["c"]) / 3.0
            k = _bin(tp)
            expect[k] = expect.get(k, 0.0) + b["v"]
    got = mm.volume_profile()
    assert set(got) == set(expect)
    for k, v in expect.items():
        assert got[k] == pytest.approx(v, rel=1e-9, abs=1e-6)
    assert math.isclose(sum(got.values()), sum(expect.values()), rel_tol=1e-9)


def test_small_memory_parameters():
    mm = MarketMemory(max_days=5, intraday_days=2)
    for b in synth_bars(12, seed=5):
        add(mm, b)
    assert mm.sessions == 5
    assert len(mm._bars) <= 2 * SESSION
    f = mm.features()
    assert all(math.isfinite(x) for x in f.values())


# --------------------------------------------------------------------------- 4. MarketContext semantics
def test_paused_repeated_and_run_id(bars40):
    mm = MarketMemory()
    b0, b1, b2 = bars40[0], bars40[1], bars40[2]
    assert mm.update(as_tick(b0)) is True
    assert mm.update(as_tick(b0)) is False                       # repeated (day, minute)
    assert mm.update(as_tick(b1, paused=True)) is False          # paused
    assert mm.bars_seen == 1
    assert mm.update(as_tick(b1)) is True
    assert mm.update({"paused": False, "bar": None}) is False
    assert mm.update("junk") is False
    bad = as_tick(b2)
    bad["bar"]["c"] = float("nan")
    assert mm.update(bad) is False
    bad2 = as_tick(b2)
    del bad2["bar"]["h"]
    assert mm.update(bad2) is False
    assert mm.bars_seen == 2

    # a changed run_id starts fresh; the same (day, minute) is new again
    assert mm.update(as_tick(b1, run_id="r2")) is True
    assert mm.bars_seen == 1
    assert mm.last_price == b1["c"]
    # run_id None neither resets nor sets
    t = as_tick(b2)
    t["run_id"] = None
    assert mm.update(t) is True
    assert mm.bars_seen == 2


def test_session_closes_when_day_changes():
    mm = MarketMemory()
    bars = synth_bars(3, seed=2)
    for b in bars[:SESSION]:
        add(mm, b)
    assert mm.sessions == 0                      # day 1 is still "today"
    add(mm, bars[SESSION])
    assert mm.sessions == 1                      # first bar of day 2 closed day 1
    d1 = bars[:SESSION]
    assert mm._days[-1][1] == max(b["h"] for b in d1)
    assert mm._days[-1][2] == min(b["l"] for b in d1)
    assert mm._days[-1][3] == d1[-1]["c"]
    assert mm._days[-1][4] == sum(b["v"] for b in d1)


# --------------------------------------------------------------------------- 5. availability
ZERO_AT_0 = ["gap", "range_pos_5d", "prev_high_dist", "prev_low_dist", "prev_close_dist",
             "rv30_ratio", "rv_today_ratio", "jump_10m", "vol_same_minute",
             "vol_today_vs_expected", "ret_5d", "ret_20d", "ret_60d", "ret_120d",
             "year_range_pos", "year_high_dist_atr", "year_low_dist_atr", "ema50d_dist_atr",
             "ema200d_dist_atr", "ema50_vs_200_atr", "stock_drawdown", "atr_regime",
             "volume_regime", "ret_1d"]


def test_availability_grows_with_sessions():
    bars = synth_bars(7, seed=21)
    hist = []
    for n_sessions in (0, 1, 5):
        mm = MarketMemory()
        for b in bars[:n_sessions * SESSION + 30]:           # +30 bars of "today"
            add(mm, b)
        assert mm.sessions == n_sessions
        f = mm.features()
        assert list(f) == list(MarketMemory.FEATURE_NAMES)
        assert all(isinstance(x, float) and math.isfinite(x) for x in f.values())
        hist.append(f["hist_avail"])
        assert f["ret_120d"] == f["ret_60d"] == f["ret_20d"] == 0.0
        assert f["ema200d_dist_atr"] == f["ema50d_dist_atr"] == f["ema50_vs_200_atr"] == 0.0
        assert f["atr_regime"] == 0.0                         # needs 14 sessions
        if n_sessions == 0:
            for k in ZERO_AT_0:
                assert f[k] == 0.0, k
            assert mm.levels() == {"above": [], "below": []}
            assert f["lvl_up1_dist"] == 10.0 and f["lvl_dn1_dist"] == -10.0
            assert f["lvl_up1_age"] == 1.0 and f["lvl_up1_strength"] == 0.0
        elif n_sessions == 1:
            assert f["ret_5d"] == 0.0 and f["range_pos_5d"] == 0.0
            assert f["prev_close_dist"] != 0.0 and f["vol_same_minute"] != 0.0
            assert mm.levels(10, 10)["above"] or mm.levels(10, 10)["below"]
        else:
            assert f["ret_5d"] != 0.0 and f["range_pos_5d"] != 0.0
        assert math.isclose(f["hist_avail"], math.log1p(n_sessions) / math.log1p(250))
    assert hist[0] == 0.0 < hist[1] < hist[2]


def test_empty_and_single_bar_are_finite():
    mm = MarketMemory()
    f = mm.features()
    assert len(f) == len(MarketMemory.FEATURE_NAMES) and all(math.isfinite(x) for x in f.values())
    assert mm.last_price is None and mm.sessions == 0
    json.dumps(mm.summary())
    mm.add_bar(1, 0, 10.0, 10.0, 10.0, 10.0, 0)           # flat, zero volume
    f = mm.features()
    assert all(math.isfinite(x) for x in f.values())
    assert f["range_pos_today"] == 0.5
    json.dumps(mm.summary())


def test_flat_market_does_not_crash():
    mm = MarketMemory()
    for d in range(1, 12):
        for m in range(SESSION):
            mm.add_bar(d, m, 50.0, 50.0, 50.0, 50.0, 100)
    f = mm.features()
    assert all(math.isfinite(x) for x in f.values())
    assert mm.levels() == {"above": [], "below": []}       # ATR == 0 -> no levels


def test_level_fields_and_feature_encoding(bars40):
    mm = MarketMemory()
    for b in bars40[:23 * SESSION + 200]:       # price sits inside the range here
        add(mm, b)
    lv = mm.levels(3, 3)
    price = mm.last_price
    assert lv["above"] and lv["below"]
    for side in ("above", "below"):
        for L in lv[side]:
            assert set(L) == {"price", "kind", "strength", "touches",
                              "sessions_since_touch", "dist_atr"}
            assert 0.0 <= L["strength"] <= 1.0
            assert math.isclose(L["dist_atr"], (L["price"] - price) / mm.atr)
    assert [L["price"] for L in lv["above"]] == sorted(L["price"] for L in lv["above"])
    assert [L["price"] for L in lv["below"]] == sorted((L["price"] for L in lv["below"]), reverse=True)
    assert all(L["dist_atr"] > 0 for L in lv["above"]) and all(L["dist_atr"] <= 0 for L in lv["below"])
    f = mm.features()
    L = lv["above"][0]
    assert f["lvl_up1_dist"] == pytest.approx(max(-10, min(10, L["dist_atr"])))
    assert f["lvl_up1_touches"] == pytest.approx(math.log1p(L["touches"]) / math.log1p(250))
    # merged levels are further apart than 0.5 ATR
    allv = sorted(x["price"] for x in mm.levels(200, 200)["above"] + mm.levels(200, 200)["below"])
    gaps = [b - a for a, b in zip(allv, allv[1:])]
    assert all(g >= 0.5 * mm.atr - 1e-9 for g in gaps)
    s = json.dumps(mm.summary())
    assert "levels" in s


def test_atr_wilder_and_emas():
    mm = MarketMemory()
    # day i: range 2, flat closes -> TR constant 2 after day 1 (gap 0)
    for d in range(1, 21):
        for m in range(3):
            mm.add_bar(d, m, 100.0, 101.0, 99.0, 100.0, 10)
    assert mm.sessions == 19
    assert mm.atr == pytest.approx(2.0)
    mm2 = MarketMemory()
    for d, rng_ in enumerate([2.0, 4.0, 6.0], 1):          # 1..13 days: simple mean of TRs
        mm2.add_bar(d, 0, 100.0, 100.0 + rng_ / 2, 100.0 - rng_ / 2, 100.0, 10)
    mm2.add_bar(4, 0, 100.0, 100.0, 100.0, 100.0, 10)
    assert mm2.sessions == 3
    assert mm2.atr == pytest.approx(4.0)


# --------------------------------------------------------------------------- 6. real data
def load_real(sessions=60):
    bars, day, cur = [], 0, None
    with open(REAL_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            date, hhmm = row["date"][:10], row["date"][11:16]
            hh, mm_ = int(hhmm[:2]), int(hhmm[3:5])
            minute = hh * 60 + mm_ - (9 * 60 + 15)
            if not 0 <= minute < SESSION:
                continue
            if date != cur:
                if day >= sessions:
                    break
                cur, day = date, day + 1
            bars.append({"day_count": day, "minute": minute, "o": float(row["open"]),
                         "h": float(row["high"]), "l": float(row["low"]),
                         "c": float(row["close"]), "v": float(row["volume"])})
    return bars


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="real NSE data not present")
def test_real_data_sanity():
    bars = load_real(60)
    mm = MarketMemory()
    for b in bars:
        add(mm, b)
    assert mm.sessions >= 55
    f = mm.features()
    assert all(isinstance(x, float) and math.isfinite(x) for x in f.values())
    lv = mm.levels(5, 5)
    assert lv["above"] + lv["below"]
    assert f["hist_avail"] > 0 and f["ret_20d"] != 0.0
    json.dumps(mm.summary())


# --------------------------------------------------------------------------- 7. determinism
def test_determinism(bars40):
    a, b = MarketMemory(), MarketMemory()
    for bar in bars40:
        add(a, bar)
        add(b, bar)
    assert a.features() == b.features()
    assert a.levels(10, 10) == b.levels(10, 10)
    assert json.dumps(a.summary()) == json.dumps(b.summary())
    # reset gives a clean slate
    a.reset()
    c = MarketMemory()
    assert a.features() == c.features() and a.sessions == 0


# --------------------------------------------------------------------------- 8. avg_daily_volume accessor
def test_avg_daily_volume(bars40):
    mm = MarketMemory()
    assert mm.avg_daily_volume() is None            # cold start
    for bar in bars40[:SESSION // 2]:
        add(mm, bar)
    assert mm.avg_daily_volume() is None            # today is not completed
    mm = MarketMemory()
    for bar in bars40:
        add(mm, bar)
    assert mm.sessions == 39
    per_day = {}
    for b in bars40:
        per_day[b["day_count"]] = per_day.get(b["day_count"], 0.0) + b["v"]
    done = [per_day[d] for d in sorted(per_day)[:39]]   # last day (40) is still open
    assert mm.avg_daily_volume(20) == pytest.approx(sum(done[-20:]) / 20)
    assert mm.avg_daily_volume(1) == pytest.approx(done[-1])
    assert mm.avg_daily_volume(500) == pytest.approx(sum(done) / 39)   # fewer sessions than n
