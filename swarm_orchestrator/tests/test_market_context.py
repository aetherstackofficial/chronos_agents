"""MarketContext (core/market_context.py): the shared, read-only market history."""
import asyncio
import copy

import pytest

from core.loop_manager import SwarmOrchestrator
from core.market_context import MarketContext


def tick(day, minute, o, h=None, l=None, c=None, v=100, **extra):
    c = o if c is None else c
    h = max(o, c) if h is None else h
    l = min(o, c) if l is None else l
    t = {"run_id": "r1", "paused": False, "day_count": day, "market_minute": minute,
         "bar": {"minute": minute, "o": o, "h": h, "l": l, "c": c, "v": v}}
    t.update(extra)
    return t


def feed(ctx, ticks):
    return [ctx.update(t) for t in ticks]


def flat_series(n, day=1, start=1, price=100.0):
    return [tick(day, start + i, price) for i in range(n)]


# --- paused / repeated ticks ---------------------------------------------------
def test_paused_and_repeated_ticks_add_no_bars():
    ctx = MarketContext()
    t = tick(1, 1, 100.0)
    assert ctx.update(t) is True
    assert ctx.update(dict(t)) is False                      # same (day, minute)
    assert ctx.update(dict(tick(1, 2, 101.0), paused=True)) is False
    assert ctx.update({"day_count": 1}) is False             # no bar at all
    assert ctx.bars_seen == 1
    assert ctx.update(tick(1, 2, 101.0)) is True
    assert ctx.bars_seen == 2


def test_same_minute_on_a_new_day_is_a_new_bar():
    ctx = MarketContext()
    assert ctx.update(tick(1, 5, 100.0)) and ctx.update(tick(2, 5, 100.0))


def test_malformed_bars_are_ignored():
    ctx = MarketContext()
    bad = tick(1, 1, 100.0)
    bad["bar"]["c"] = "x"
    assert ctx.update(bad) is False
    bad["bar"]["c"] = float("nan")
    assert ctx.update(bad) is False
    assert ctx.bars_seen == 0


# --- ready -----------------------------------------------------------------------
def test_ready_flips_at_exactly_min_bars():
    ctx = MarketContext(min_bars=10)
    for i, t in enumerate(flat_series(10), start=1):
        assert ctx.ready is False
        ctx.update(t)
        assert ctx.ready is (i >= 10)
    assert ctx.bars_seen == 10


def test_default_min_bars_is_120():
    ctx = MarketContext()
    feed(ctx, flat_series(119))
    assert not ctx.ready
    ctx.update(tick(1, 120, 100.0))
    assert ctx.ready


# --- today / history --------------------------------------------------------------
def test_today_values_before_and_after_first_bar():
    ctx = MarketContext()
    assert (ctx.day_open, ctx.day_high, ctx.day_low, ctx.day_close) == (None,) * 4
    assert ctx.prev_day is None and ctx.days == []
    feed(ctx, [tick(1, 1, 100, h=101, l=99.5, c=100.5),
               tick(1, 2, 100.5, h=103, l=100, c=102),
               tick(1, 3, 102, h=102.5, l=98, c=99)])
    assert (ctx.day_open, ctx.day_high, ctx.day_low, ctx.day_close) == (100, 103, 98, 99)


def test_day_rollover_moves_today_into_prev_day():
    ctx = MarketContext()
    feed(ctx, [tick(1, 1, 100, h=101, l=99, c=100.5, v=10),
               tick(1, 2, 100.5, h=104, l=100, c=103, v=20)])
    assert ctx.prev_day is None
    ctx.update(tick(2, 1, 105, h=106, l=104.5, c=105.5, v=7))
    assert ctx.prev_day == {"o": 100, "h": 104, "l": 99, "c": 103, "v": 30}
    assert (ctx.day_open, ctx.day_high, ctx.day_low, ctx.day_close) == (105, 106, 104.5, 105.5)
    assert ctx.days == [ctx.prev_day]


def test_days_keeps_last_five_newest_last():
    ctx = MarketContext()
    for d in range(1, 9):
        ctx.update(tick(d, 1, 100.0 + d))
    days = ctx.days
    assert len(days) == 5 and [x["o"] for x in days] == [103.0, 104.0, 105.0, 106.0, 107.0]
    assert ctx.prev_day["o"] == 107.0 and ctx.day_open == 108.0


def test_returned_dicts_are_copies():
    ctx = MarketContext()
    feed(ctx, [tick(1, 1, 100.0), tick(2, 1, 101.0)])
    ctx.prev_day["h"] = 9999
    ctx.days[0]["h"] = 9999
    assert ctx.prev_day["h"] == 100.0


# --- swing points -----------------------------------------------------------------
def peak_series(peak_at=10, total=30):
    """Flat at 100 with one spike high at ``peak_at`` and one dip low 12 bars later."""
    out = []
    for i in range(1, total + 1):
        h = 110.0 if i == peak_at else 100.5
        l = 90.0 if i == peak_at + 12 else 99.5
        out.append(tick(1, i, 100.0, h=h, l=l, c=100.0))
    return out


def test_swing_not_reported_until_k_later_bars_exist():
    k = 5
    ctx = MarketContext(swing_k=k)
    series = peak_series()
    feed(ctx, series[:9 + k])                  # peak is bar 10: only k-1 later bars so far
    assert ctx.swing_highs() == []
    ctx.update(series[9 + k])                  # the k-th later bar arrives
    assert ctx.swing_highs() == [110.0]


def test_swing_low_and_ordering_newest_last():
    ctx = MarketContext(swing_k=3)
    feed(ctx, peak_series(peak_at=8, total=40))
    assert ctx.swing_highs() == [110.0] and ctx.swing_lows() == [90.0]
    ctx2 = MarketContext(swing_k=2)
    seq = []
    for i in range(1, 40):
        h = {8: 105.0, 20: 108.0}.get(i, 100.5)
        seq.append(tick(1, i, 100.0, h=h, l=99.5, c=100.0))
    feed(ctx2, seq)
    assert ctx2.swing_highs() == [105.0, 108.0]


def test_equal_highs_are_not_a_swing():
    ctx = MarketContext(swing_k=2)
    feed(ctx, [tick(1, i, 100.0, h=101.0, l=99.5, c=100.0) for i in range(1, 30)])
    assert ctx.swing_highs() == [] and ctx.swing_lows() == []


# --- no look-ahead -------------------------------------------------------------------
def snapshot(ctx):
    return (ctx.bars_seen, ctx.ready, ctx.day_open, ctx.day_high, ctx.day_low, ctx.day_close,
            ctx.prev_day, ctx.days, ctx.swing_highs(), ctx.swing_lows(),
            ctx.ret(5), ctx.ret(30), ctx.vol_ratio())


def wavy(n):
    out = []
    for i in range(1, n + 1):
        base = 100 + 3 * ((i * 7) % 11 - 5) / 5
        out.append(tick(1 + (i - 1) // 60, 1 + (i - 1) % 60, base,
                        h=base + 0.4 + (i % 5) * 0.1, l=base - 0.4 - (i % 3) * 0.1,
                        c=base + 0.2 * ((-1) ** i), v=100 + i))
    return out


def test_no_lookahead_changing_later_bars_does_not_change_state_at_t():
    series = wavy(200)
    t = 150
    a, b = MarketContext(min_bars=50), MarketContext(min_bars=50)
    feed(a, series[:t])
    mutated = copy.deepcopy(series)
    for tk in mutated[t:]:                      # rewrite EVERY bar after t
        for key in ("o", "h", "l", "c"):
            tk["bar"][key] = tk["bar"][key] * 3 + 50
    feed(b, mutated[:t])
    assert snapshot(a) == snapshot(b)
    feed(a, series[t:])                         # and the futures really do differ
    feed(b, mutated[t:])
    assert snapshot(a) != snapshot(b)


def test_same_ticks_twice_give_the_same_state():
    series = wavy(300)
    a, b = MarketContext(min_bars=50), MarketContext(min_bars=50)
    feed(a, series)
    feed(b, series)
    assert snapshot(a) == snapshot(b)


# --- returns and volatility ----------------------------------------------------------
def test_ret_needs_enough_bars():
    ctx = MarketContext()
    feed(ctx, [tick(1, i, 100.0 + i) for i in range(1, 7)])   # closes 101..106
    assert ctx.ret(5) == pytest.approx(106 / 101 - 1)
    assert ctx.ret(6) is None and ctx.ret(0) is None
    assert ctx.ret(375) is None


def test_ret_375_with_enough_history():
    ctx = MarketContext()
    feed(ctx, [tick(1 + i // 375, 1 + i % 375, 100.0 + i * 0.01) for i in range(400)])
    assert ctx.ret(375) == pytest.approx(ctx._bars[-1]["c"] / ctx._bars[-376]["c"] - 1)


def test_vol_ratio_compares_today_with_recent_days():
    ctx = MarketContext()
    assert ctx.vol_ratio() is None
    calm = [tick(d, m, 100.0 + (0.1 if m % 2 else 0.0)) for d in (1, 2) for m in range(1, 31)]
    feed(ctx, calm)
    wild = [tick(3, m, 100.0 + (1.0 if m % 2 else 0.0)) for m in range(1, 31)]
    feed(ctx, wild)
    assert ctx.vol_ratio() > 5                  # today is ~10x as volatile as the norm


# --- bounded memory / new run ---------------------------------------------------------
def test_memory_is_bounded():
    ctx = MarketContext(max_bars=500)
    feed(ctx, [tick(1 + i // 375, 1 + i % 375, 100.0 + (i % 7)) for i in range(5000)])
    assert len(ctx._bars) <= 500 and len(ctx.days) <= 5
    assert len(ctx.swing_highs()) <= 64 and len(ctx.swing_lows()) <= 64
    assert ctx.bars_seen == 5000


def test_new_run_id_starts_a_fresh_history():
    ctx = MarketContext(min_bars=2)
    feed(ctx, [tick(1, 1, 100.0), tick(1, 2, 101.0)])
    assert ctx.ready
    assert ctx.update(dict(tick(1, 1, 50.0), run_id="r2")) is True
    assert ctx.bars_seen == 1 and not ctx.ready and ctx.day_open == 50.0


def test_invalid_settings_rejected():
    for kw in ({"min_bars": 0}, {"swing_k": 0}, {"days": 0}):
        with pytest.raises(ValueError):
            MarketContext(**kw)


# --- wiring: orchestrator ------------------------------------------------------------
class FakeAgent:
    agent_id = "MM_1"
    is_asleep = False

    def __init__(self):
        self.calls, self.seen_bars = 0, []

    def on_tick(self, tick):
        self.calls += 1
        self.seen_bars.append(self.context.bars_seen)    # context is current when the agent acts
        return [{"action": "BUY", "qty": 1}]

    def on_ack(self, ack):
        pass


class FakeClient:
    def __init__(self):
        self.sent = []

    async def send_order(self, msg):
        self.sent.append(msg)
        return {"status": "FILLED"}


def _orch():
    agent, client = FakeAgent(), FakeClient()
    orch = SwarmOrchestrator(client, seed=1)
    orch.QUIET_AWAKE_RATIO = 1.0
    orch.load_agents([agent])
    return orch, agent


def test_orchestrator_attaches_one_shared_context_to_every_agent():
    orch = SwarmOrchestrator(FakeClient(), seed=1)
    a, b = FakeAgent(), FakeAgent()
    orch.load_agents([a, b])
    assert a.context is b.context is orch.context


def test_context_is_updated_before_agents_act():
    orch, agent = _orch()
    asyncio.run(orch._on_tick(dict(tick(1, 1, 100.0), mode="LIVE")))
    asyncio.run(orch._on_tick(dict(tick(1, 2, 101.0), mode="LIVE")))
    assert agent.seen_bars == [1, 2]


def test_agents_are_never_gated_even_in_replay():
    for mode in ("LIVE", "REACTIVE", "REPLAY"):
        orch, agent = _orch()
        assert orch.context.ready is False                # not warmed up, still trades
        asyncio.run(orch._on_tick(dict(tick(1, 1, 100.0), mode=mode)))
        assert agent.calls == 1 and len(orch.client.sent) == 1


# --- wiring: strategy SDK ---------------------------------------------------------------
def test_sdk_strategy_gets_context_updated_before_on_tick():
    from runner.sdk import Strategy, _StrategyRunner

    seen = []

    class S(Strategy):
        def on_tick(self, state):
            seen.append((self.context.bars_seen, self.context.day_close))
            return []

    strat = S()
    assert S.context is None                                # class default, until attached
    runner = _StrategyRunner(strat, "t", "STRAT_x", "127.0.0.1", 59991, 59992)
    try:
        assert strat.context is runner.context
        runner._handle_tick(tick(1, 1, 100.0))
        runner._handle_tick(tick(1, 1, 100.0))              # repeat: no new bar
        runner._handle_tick(tick(1, 2, 101.0))
    finally:
        runner.sub.close(0)
        runner.dealer.close(0)
    assert seen == [(1, 100.0), (1, 100.0), (2, 101.0)]


def test_dry_run_gives_strategy_a_context(tmp_path):
    from runner.check import build_sample_ticks
    from runner.dryrun import run_dry

    code = (
        "from runner.sdk import Strategy\n"
        "class UserStrategy(Strategy):\n"
        "    def on_tick(self, state):\n"
        "        assert self.context is not None and self.context.bars_seen >= 1\n"
        "        return []\n")
    path = tmp_path / "s.py"
    path.write_text(code)
    report = run_dry(str(path), build_sample_ticks())
    assert report["ok"], report["errors"]
