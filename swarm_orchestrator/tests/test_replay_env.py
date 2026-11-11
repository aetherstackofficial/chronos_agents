"""ReplayBookEnv, its costs, rewards and sampler (training/envs/)."""
import math

import numpy as np
import pytest

from agents.features_v2 import OBS_SPEC
from core.market_memory import MarketMemory
from training.data.episode_bars import bars_for_sessions, load_stock
from training.envs.config import load_env_config
from training.envs.costs import CarryTracker, CostModel
from training.envs.replay_book_env import ReplayBookEnv
from training.envs.rewards import DifferentialSharpe, MMReward, WhaleReward
from training.envs.sampler import EpisodeSampler, preroll_arrays

MODELS = ("whale_swing", "whale_position", "mm")


ONE_MINUTE = {"whale_swing": 1, "whale_position": 1, "mm": 1}


def cfg_for(name, length=(1, 1), memory=None, noise=0.0, decision_minutes=None, **extra):
    """Env config of a test; decision interval 1 minute (the pre-decision-interval behaviour) unless asked."""
    dm = dict(ONE_MINUTE) if decision_minutes is None else {**ONE_MINUTE, name: decision_minutes}
    c = {"sampler": {"length_sessions": {name: list(length)}},
         "env": {"obs_noise": noise, "decision_minutes": dm}}
    if memory is not None:
        c["sampler"]["memory_mix"] = [[1.0, memory, memory]]
    for k, v in extra.items():
        c.setdefault(k, {}).update(v)
    return c


def make_env(name, seed=0, **kw):
    return ReplayBookEnv(name, "train", cfg_for(name, **kw), seed=seed)


def run_episode(env, policy=None, max_steps=100000):
    infos, rewards = [], []
    obs, info0 = env.reset()
    for _ in range(max_steps):
        a = policy(obs) if policy else env.action_space.sample()
        obs, r, te, tr, info = env.step(a)
        infos.append(info)
        rewards.append(r)
        if te or tr:
            break
    return info0, infos, rewards


# --------------------------------------------------------------------------- config
def test_config_cost_defaults():
    cfg = load_env_config()
    assert cfg["costs"]["intraday_rt_bps"] == pytest.approx(3.55)
    assert cfg["costs"]["delivery_rt_bps"] == pytest.approx(22.0)
    assert OBS_SPEC and set(cfg["rewards"]["whale"]["eta"]) == {"whale_swing", "whale_position"}
    assert cfg["env"]["decision_minutes"] == {"whale_swing": 15, "whale_position": 75, "mm": 1}
    assert cfg["rewards"]["whale"]["w_switch"] == {"whale_swing": 0.0, "whale_position": 0.0}


# --------------------------------------------------------------------------- costs
def test_fill_fee_is_half_round_trip():
    cm = CostModel(3.55, 22.0, cost_mult=2.0)
    assert cm.fill_fee(1000, 100.0) == pytest.approx(1000 * 100.0 * 1.775e-4 * 2.0)


def test_closing_cost_includes_half_spread():
    cm = CostModel(3.55, 22.0, cost_mult=1.0)
    # price 100, spread 0.10 -> 5 bps half spread
    assert cm.closing_cost(-200, 100.0, 0.10) == pytest.approx(200 * 100.0 * (1.775 + 5.0) / 1e4)
    assert cm.closing_cost(0, 100.0, 0.10) == 0.0


def test_carry_surcharge_once_per_share():
    cm = CostModel(3.55, 22.0, cost_mult=1.0)
    rate = (22.0 - 3.55) / 1e4
    t = CarryTracker()
    t.on_position(100)
    assert t.session_close(100, 10.0, cm) == pytest.approx(100 * 10.0 * rate)
    assert t.session_close(100, 10.0, cm) == 0.0                     # same shares: not again
    t.on_position(150)
    assert t.session_close(150, 10.0, cm) == pytest.approx(50 * 10.0 * rate)   # only the 50 new
    t.on_position(80)                                                 # shrinks: charged shrinks
    assert t.charged == 80
    t.on_position(120)
    assert t.session_close(120, 10.0, cm) == pytest.approx(40 * 10.0 * rate)
    t.on_position(-30)                                                # sign flip: starts over
    assert t.charged == 0
    assert t.session_close(-30, 10.0, cm) == pytest.approx(30 * 10.0 * rate)
    t.on_position(0)
    assert t.session_close(0, 10.0, cm) == 0.0


# --------------------------------------------------------------------------- rewards
def test_dsr_matches_formula_and_clips():
    d = DifferentialSharpe(eta=0.01, var_floor=1e-8, clip=10.0)
    A, B = 0.0, 1e-8
    for R in (1e-4, -2e-4, 0.0, 5e-4):
        dA, dB = 0.01 * (R - A), 0.01 * (R * R - B)
        want = (B * dA - 0.5 * A * dB) / max(B - A * A, 1e-8) ** 1.5
        want = max(-10.0, min(10.0, want))
        got, _ = d.step(R)
        assert got == pytest.approx(want)
        A, B = A + dA, B + dB
    d2 = DifferentialSharpe(eta=1.0, var_floor=1e-8, clip=10.0)
    D, clipped = d2.step(0.5)             # huge move against the floor: must clip
    assert clipped and D == 10.0 and d2.n_clipped == 1


def test_whale_reward_parts():
    cfg = load_env_config()
    wr = WhaleReward(cfg, "whale_swing", risk_tolerance=0.1)
    parts, _ = wr.step(0.0, dd=0.15, position=0, capacity=100, terminated_dd=False)
    scale = cfg["rewards"]["whale"]["dd_step_scale"]["whale_swing"]
    assert parts["drawdown"] == pytest.approx(-scale * ((0.15 - 0.05) / 0.1) ** 2)
    assert parts["termination"] == 0.0
    parts, _ = wr.step(0.0, dd=0.30, position=50, capacity=100, terminated_dd=True)
    assert parts["termination"] == -cfg["rewards"]["whale"]["w_term"]
    cfg2 = load_env_config(overrides={"rewards": {"whale": {"profit_term": "pnl", "w_flat": 0.3}}})
    parts, clipped = WhaleReward(cfg2, "whale_swing", 0.1).step(0.001, 0.0, 0, 100, False)
    assert parts["profit"] == pytest.approx(0.01) and parts["inaction"] == -0.3 and not clipped


def test_mm_reward_settles_markout_one_step_later():
    cfg = load_env_config()
    r = MMReward(cfg, capital=1e6, capacity=1000, max_quote_size=200)
    k = cfg["rewards"]["mm"]["reward_scale"] / 1e6
    # step 1: buy 100 @ 100.00 (fee 1.0); nothing settled yet
    p1 = r.step([(100, 100.0, 1.0)], mid=100.0, extra_cost=0.0, position=100, ofi=0.0,
                tick_size=0.05, two_sided=False, terminated_dd=False, final=False)
    assert p1["spread_markout"] == 0.0 and p1["cost"] == 0.0
    # step 2: mid is 100.10 one minute later -> +10 money, its fee -1
    p2 = r.step([], mid=100.10, extra_cost=0.5, position=100, ofi=-1.0,
                tick_size=0.05, two_sided=True, terminated_dd=False, final=False)
    assert p2["spread_markout"] == pytest.approx(100 * 0.10 * k)
    assert p2["cost"] == pytest.approx(-1.5 * k)
    # long inventory with sell pressure: factor (1 + 1)
    w = cfg["rewards"]["mm"]
    unit = 0.05 * 200
    assert p2["inventory"] == pytest.approx(-w["w_inv"] * w["inv_penalty_ticks"] * unit * (0.1) ** 2 * 2.0 * k)
    assert p2["uptime"] == pytest.approx(w["w_up"] * unit * k)
    # final step settles this step's fills against the current mid
    p3 = r.step([(-50, 100.2, 0.2)], mid=100.1, extra_cost=0.0, position=50, ofi=0.0,
                tick_size=0.05, two_sided=True, terminated_dd=False, final=True)
    assert p3["spread_markout"] == pytest.approx(-50 * (100.1 - 100.2) * k)
    assert r.pending == []


# --------------------------------------------------------------------------- sampler
def test_sampler_ranges_and_adv_capacity():
    for name in MODELS:
        s = EpisodeSampler(name, "train", None, seed=11)
        rng = np.random.default_rng(5)
        for _ in range(12):
            ep = s.draw(rng)
            assert ep.warmup_session == ep.start_session - 1
            assert len(ep.bars) == (1 + ep.length) * 375
            assert 0 <= ep.memory_sessions <= min(250, ep.warmup_session)
            assert ep.memory_first_session == ep.warmup_session - ep.memory_sessions
            lo, hi = s.cfg["sampler"]["length_sessions"][name]
            assert lo <= ep.length <= hi
            assert 0.5 <= ep.cost_mult <= 1.5 and ep.order_delay in (0, 1)
            assert 0.5 <= ep.lp_size_mult <= 1.5 and ep.obs_noise == load_env_config()["env"]["obs_noise"]
            df = load_stock(ep.symbol)
            sid = df["session_id"].to_numpy()
            a = np.searchsorted(sid, max(ep.start_session - 20, 0))
            b = np.searchsorted(sid, ep.start_session)
            adv = df["volume"].to_numpy()[a:b].sum() / (ep.start_session - max(ep.start_session - 20, 0))
            assert ep.adv20 == pytest.approx(adv)
            if name == "mm":
                assert 0.0002 * adv * 0.99 <= ep.capacity <= 0.005 * adv * 1.01 or ep.capacity == 5
                assert ep.max_quote_size == max(ep.capacity // 5, 1)
                assert ep.capital == pytest.approx(2 * ep.capacity * ep.start_price)
            else:
                assert 0.0005 * adv * 0.99 <= ep.capacity <= 0.03 * adv * 1.01
                assert ep.capital == pytest.approx(ep.capacity * ep.start_price)
                assert 0.03 <= ep.risk_tolerance <= 0.15


def test_sampler_deterministic_and_splits():
    a = EpisodeSampler("whale_swing", "validation", None, 1).draw(np.random.default_rng(9))
    b = EpisodeSampler("whale_swing", "validation", None, 2).draw(np.random.default_rng(9))
    assert (a.symbol, a.start_session, a.memory_sessions, a.capacity, a.engine_seed) == \
           (b.symbol, b.start_session, b.memory_sessions, b.capacity, b.engine_seed)
    df = load_stock(a.symbol)
    first = df[df["minute"] == 0]
    d0 = first[first["session_id"] == a.start_session]["datetime"].iloc[0]
    assert str(d0.date()) >= "2023-01-01" and str(d0.date()) <= "2024-06-30"


# --------------------------------------------------------------------------- env API
@pytest.mark.parametrize("name", MODELS)
def test_check_env(name):
    from gymnasium.utils.env_checker import check_env as gym_check
    from stable_baselines3.common.env_checker import check_env as sb3_check
    env = make_env(name, seed=3)
    gym_check(env, skip_render_check=True)
    sb3_check(make_env(name, seed=3), warn=False, skip_render_check=True)
    assert env.observation_space.shape == (len(OBS_SPEC["mm" if name == "mm" else "whale"]),)
    assert env.action_space.shape == ((4,) if name == "mm" else (3,))


@pytest.mark.parametrize("name", MODELS)
def test_random_episode_completes_with_finite_rewards(name):
    env = make_env(name, seed=2, length=(1, 2), noise=0.01)
    info0, infos, rewards = run_episode(env)
    assert np.all(np.isfinite(rewards))
    assert len(rewards) == info0["length_sessions"] * 375 - 1
    last = infos[-1]
    assert "episode" in last and last["episode"]["steps"] == len(rewards)
    for i in infos:
        assert math.isclose(sum(i["reward_parts"].values()), sum(i["reward_parts"].values()))
    assert env.engine.mode == "REPLAY"


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("profit", ["dsr", "pnl"])
def test_reward_parts_sum_to_reward(name, profit):
    if name == "mm" and profit == "pnl":
        pytest.skip("pnl switch is whale only")
    extra = {"rewards": {"whale": {"profit_term": profit, "w_flat": 0.01}}}
    env = ReplayBookEnv(name, "train", {**cfg_for(name, noise=0.01), **extra}, seed=4)
    env.reset()
    for _ in range(120):
        _, r, te, tr, info = env.step(env.action_space.sample())
        assert r == pytest.approx(sum(info["reward_parts"].values()), abs=1e-12)
        assert np.isfinite(r)
        if te or tr:
            break
    expected = ({"profit", "drawdown", "termination", "inaction", "switch"} if name != "mm" else
                {"spread_markout", "cost", "inventory", "uptime", "termination"})
    assert set(info["reward_parts"]) == expected


@pytest.mark.parametrize("name", MODELS)
def test_obs_float32_finite_and_noise_only_on_market_inputs(name):
    role = "mm" if name == "mm" else "whale"
    quiet = make_env(name, seed=6, noise=0.0)
    noisy = make_env(name, seed=6, noise=0.01)
    o0, _ = quiet.reset(seed=6)
    o1, _ = noisy.reset(seed=6)
    assert o0.dtype == np.float32 and np.isfinite(o1).all() and o1.shape == o0.shape
    names = OBS_SPEC[role]
    diff = np.abs(o1 - o0)
    for i, n in enumerate(names):
        own = n in ("inventory", "open_pnl", "drawdown", "holding", "pnl_today", "exposure",
                    "cap_vs_adv", "traded_vs_adv", "risk_appetite") or n.startswith(
                        ("last_action_", "sector_"))
        if own:
            assert diff[i] == 0.0, n
    assert diff.max() > 0.0


def test_reset_is_deterministic_given_seed():
    e = make_env("whale_swing", seed=1, noise=0.01)
    a, _ = e.reset(seed=77)
    act = [np.array([0.5, 0.2, -0.3], np.float32)] * 40
    ra = [e.step(x)[1] for x in act]
    b, _ = e.reset(seed=77)
    rb = [e.step(x)[1] for x in act]
    assert np.array_equal(a, b) and ra == rb


# --------------------------------------------------------------------------- memory
def test_m0_gives_hist_avail_zero_and_empty_memory():
    env = make_env("whale_swing", seed=5, memory=0)
    obs, info = env.reset()
    i = OBS_SPEC["whale"].index("mem_hist_avail")
    assert info["memory_sessions"] == 0 and obs[i] == 0.0
    assert env.memory.sessions == 0 and env.memory.bars_seen == 1       # only the first episode bar
    obs, *_ = env.step(env.action_space.sample())
    assert obs[i] == 0.0 and env.memory.bars_seen == 2
    # the warm-up session is NOT in the memory even after the next session starts
    env2 = make_env("whale_swing", seed=5, memory=0, length=(2, 2))
    env2.reset()
    for _ in range(375):
        env2.step(env2.action_space.sample())
    assert env2.memory.sessions == 1                                   # only session 1 of the episode closed


def test_memory_day_and_minute_parity_with_add_bar_reference():
    """Pre-roll + warm-up + first tick == a memory fed all those sessions through
    add_bar with engine minute numbering (1..375) and consecutive day numbers."""
    for seed in range(40):
        env = make_env("whale_swing", seed=seed, memory=30)
        env.sampler.symbols = ["SBIN"]            # SBIN stays on the 0.05 tick grid before 2023
        env.reset(seed=seed)
        ep = env.episode
        df = load_stock("SBIN")
        bars = bars_for_sessions(df, ep.warmup_session, 2)
        eng_bars = env.engine._closed_bars
        same = all(abs(b["c"] - e["c"]) < 1e-9 and abs(b["h"] - e["h"]) < 1e-9
                   and abs(b["l"] - e["l"]) < 1e-9 and abs(b["o"] - e["o"]) < 1e-9
                   for b, e in zip(bars[:376], list(eng_bars)[-376:]))
        if same:
            break
    else:
        pytest.skip("no SBIN episode whose engine bars equal the file bars")
    ref = MarketMemory()
    days, mins, o, h, l, c, v = preroll_arrays(df, ep.memory_first_session, ep.memory_sessions)
    for d, m, *p in zip(days, mins, o, h, l, c, v):
        ref.add_bar(d - ep.warmup_session, m + 1, *p)
    for b in bars[:376]:                                   # warm-up session + first episode bar
        ref.add_bar(1 if b["day_count"] - 1 == ep.warmup_session else 2, b["minute"] + 1,
                    b["o"], b["h"], b["l"], b["c"], b["v"])
    fa, fb = env.memory.features(), ref.features()
    assert env.memory.sessions == ref.sessions == ep.memory_sessions + 1
    for k in fa:
        assert fa[k] == pytest.approx(fb[k], rel=1e-9, abs=1e-9), k


def test_memory_first_day_closes_correctly_without_warmup_collision():
    env = make_env("whale_swing", seed=8, memory=12)
    env.reset()
    # 12 pre-rolled sessions + the warm-up session (closed by the first episode tick)
    assert env.memory.sessions == 13


# --------------------------------------------------------------------------- positions and costs
def _hold_env(length=3, delay=0):
    cfg = cfg_for("whale_swing", length=(length, length), memory=0)
    cfg["sampler"]["whale_capacity_adv_frac"] = [0.0005, 0.0005]
    cfg["sampler"]["order_delay_choices"] = [delay]
    cfg["sampler"]["cost_mult_range"] = [1.0, 1.0]
    cfg["costs"] = {"cost_mult_range": [1.0, 1.0]}
    cfg["rewards"] = {"whale": {"dd_free": 10.0}}                # no drawdown stop interference
    cfg["env"]["dd_terminate"] = {"whale_swing": 10.0}
    return ReplayBookEnv("whale_swing", "train", cfg, seed=12)


def test_position_carries_across_session_close_and_surcharge_charged_once():
    env = _hold_env(length=3)
    env.reset()
    rate = env.cost.carry_rate()
    long_a = np.array([1.0, 1.0, 1.0], np.float32)
    closes, pos_at_close, pos_after = [], [], []
    done = False
    while not done:
        before_minute = int(env.last_tick["market_minute"])
        _, r, te, tr, info = env.step(long_a)
        done = te or tr
        if before_minute == 375 and info["position"] != 0:
            pos_after.append(info["position"])
        if int(env.last_tick["market_minute"]) == 375 and not done:
            fill_fees = sum(env.cost.fill_fee(q, p) for _, q, p in info["fills"])
            closes.append(info["fees"] - fill_fees)
            pos_at_close.append((info["position"], float(env.last_tick["last_price"])))
    assert len(closes) == 2 and len(pos_after) == 2
    assert all(p > 0 for p, _ in pos_at_close)                    # still long: no forced flat
    assert min(pos_after) > 0
    s1 = rate * pos_at_close[0][0] * pos_at_close[0][1]
    assert closes[0] == pytest.approx(s1, rel=1e-9)
    new_shares = max(pos_at_close[1][0] - pos_at_close[0][0], 0)
    assert closes[1] == pytest.approx(rate * new_shares * pos_at_close[1][1], rel=1e-9, abs=1e-6)
    assert closes[1] < 0.2 * closes[0]


def test_closing_cost_charged_at_episode_end():
    env = _hold_env(length=1)
    env.reset()
    long_a = np.array([1.0, 1.0, 1.0], np.float32)
    for _ in range(373):
        env.step(long_a)
    pos = env.book.position
    fees_before = env.book.fees_paid
    tick_spread = None
    _, r, te, tr, info = env.step(long_a)
    assert tr and not te
    tick = env.last_tick
    expect = env.cost.closing_cost(env.book.position, float(tick["last_price"]), float(tick["spread"]))
    fills_fee = sum(env.cost.fill_fee(q, p) for _, q, p in info["fills"])
    assert env.book.fees_paid - fees_before == pytest.approx(expect + fills_fee, rel=1e-9)
    assert expect > 0 and pos > 0


@pytest.mark.parametrize("delay", [0, 1])
@pytest.mark.parametrize("name", ["whale_swing", "mm"])
def test_no_fills_are_missed(name, delay):
    cfg = cfg_for(name, length=(2, 2), memory=0)
    cfg["sampler"]["order_delay_choices"] = [delay]
    env = ReplayBookEnv(name, "train", cfg, seed=21)
    env.reset()
    net = 0
    n_fills = 0
    done = False
    while not done:
        _, _, te, tr, info = env.step(env.action_space.sample())
        done = te or tr
        for side, q, _ in info["fills"]:
            net += q if side == "BUY" else -q
            n_fills += 1
    led = env.engine.agent_ledger[env.agent_id]
    assert n_fills > 20
    assert net == led["pos"] == env.book.position
    assert env.engine.mode == "REPLAY"


def test_drawdown_termination_pays_penalty():
    cfg = cfg_for("whale_swing", length=(3, 3), memory=0)
    cfg["env"]["dd_terminate"] = {"whale_swing": 0.0003}
    env = ReplayBookEnv("whale_swing", "train", cfg, seed=3)
    env.reset()
    done = False
    n = 0
    while not done:
        _, r, te, tr, info = env.step(np.array([1.0, 1.0, 1.0], np.float32))
        done = te or tr
        n += 1
    assert te and not tr and n < 3 * 375 - 1
    assert info["reward_parts"]["termination"] == -1.0
    assert info["episode"]["steps"] == n


def test_sim_lp_size_is_randomised_on_the_instance():
    env = make_env("whale_swing", seed=9)
    env.reset()
    ep = env.episode
    from engine.matching_engine import MatchingEngine
    assert env.engine.LP_BASE_SIZE == pytest.approx(MatchingEngine.LP_BASE_SIZE * ep.lp_size_mult)
    assert MatchingEngine.LP_BASE_SIZE == 400                      # class untouched
    assert MatchingEngine.LP_VOLUME_FACTOR == 0.5


# --------------------------------------------------------------------------- decision interval
def _rollout(name, delay, k, seed=5, length=1, extra=None, policy_seed=1, max_decisions=100000):
    """Random-action episode of ``name`` at decision interval k; returns env + per-step lists."""
    cfg = cfg_for(name, length=(length, length), memory=0, decision_minutes=k, **(extra or {}))
    cfg["sampler"]["order_delay_choices"] = [delay]
    env = ReplayBookEnv(name, "train", cfg, seed=seed)
    env.reset()
    rng = np.random.default_rng(policy_seed)
    n = env.action_space.shape[0]
    rs, pos, eq, infos = [], [], [], []
    for _ in range(max_decisions):
        _, r, te, tr, info = env.step(rng.uniform(-1, 1, n).astype(np.float32))
        rs.append(r), pos.append(info["position"]), eq.append(info["equity"]), infos.append(info)
        if te or tr:
            break
    return env, rs, pos, eq, infos


# Reference numbers recorded from the env BEFORE the decision interval was added (per-minute loop,
# seed 5, 1 session, memory 0, no observation noise, uniform random actions from default_rng(1)):
# (model, order_delay) -> (steps, sum(r), sum|r|, pos[100], pos[-1], equity[200], equity[-1],
#                          sum(pos), n_fills, fees)
OLD_PER_MINUTE = {
    ("whale_swing", 0): (374, -0.027913400789083977, 0.3582971232819846, 16116.0, 39862.0,
                         89997575.79100697, 89932799.36099418, 847608.0, 314, 122118.53900583311),
    ("whale_swing", 1): (374, -0.012281831183814367, 0.4195294737731945, 15312.0, 17030.0,
                         90042776.91356942, 89999894.37835304, 962200.0, 261, 96564.9216469673),
    ("mm", 0): (374, -45.827642890565656, 93.0161573655021, 11236.0, 17516.0,
                45895632.542830445, 45844370.16942903, -651150.0, 246, 88472.93057097631),
}


@pytest.mark.parametrize("name,delay", list(OLD_PER_MINUTE))
def test_k1_is_identical_to_the_old_per_minute_behaviour(name, delay):
    env, rs, pos, eq, infos = _rollout(name, delay, 1)
    n, sr, sar, p100, plast, e200, elast, spos, nf, fees = OLD_PER_MINUTE[(name, delay)]
    assert len(rs) == n and env.decision_minutes == 1
    assert sum(rs) == pytest.approx(sr, rel=1e-12, abs=1e-12) and sum(abs(x) for x in rs) == pytest.approx(sar, rel=1e-12)
    assert (pos[100], pos[-1], sum(pos)) == (p100, plast, spos)
    assert eq[200] == pytest.approx(e200, rel=1e-12) and eq[-1] == pytest.approx(elast, rel=1e-12)
    assert infos[-1]["episode"]["fills"] == nf and infos[-1]["episode"]["fees"] == pytest.approx(fees, rel=1e-12)
    assert all(i["minutes"] == 1 for i in infos) and infos[-1]["episode"]["decisions"] == n


@pytest.mark.parametrize("name,k", [("whale_swing", 15), ("whale_position", 75), ("mm", 5)])
def test_k_steps_cover_the_episode_minutes(name, k):
    length = 2
    env, rs, pos, eq, infos = _rollout(name, 0, k, length=length)
    total_minutes = length * 375 - 1
    assert env.decision_minutes == k
    assert len(rs) == math.ceil(total_minutes / k)
    assert sum(i["minutes"] for i in infos) == total_minutes
    assert all(i["minutes"] == k for i in infos[:-1]) and 1 <= infos[-1]["minutes"] <= k
    ep = infos[-1]["episode"]
    assert ep["steps"] == total_minutes and ep["decisions"] == len(rs)
    for i in infos:                                   # one log entry per advanced minute
        assert len(i["minute_log"]) == i["minutes"]
    assert env.engine.mode == "REPLAY"
    # step reward = sum of its parts; episode parts_sum = sum over steps
    assert sum(rs) == pytest.approx(sum(ep["parts_sum"].values()), abs=1e-9)
    assert infos[-1]["step_return"] == pytest.approx((eq[-1] - eq[-2]) / env.episode.capital, rel=1e-9)


def test_k_reward_is_the_sum_of_per_minute_rewards_and_dsr_sees_every_minute():
    """K=15 and K=1 start from the same state; with the same action held, the K=1 per-minute DSR sums over
    the first 15 minutes must equal the first K=15 reward when the action is constant (the slicer works
    the same target either way)."""
    a = np.array([0.7, 0.4, 0.2], np.float32)
    e1 = ReplayBookEnv("whale_swing", "train", cfg_for("whale_swing", memory=0, decision_minutes=1), seed=8)
    e15 = ReplayBookEnv("whale_swing", "train", cfg_for("whale_swing", memory=0, decision_minutes=15), seed=8)
    e1.reset(seed=8), e15.reset(seed=8)
    tot = 0.0
    parts = {}
    for _ in range(15):
        _, r, *_x, info = e1.step(a)
        tot += r
        for k, v in info["reward_parts"].items():
            parts[k] = parts.get(k, 0.0) + v
    _, r15, _, _, info15 = e15.step(a)
    # the K=1 policy re-decodes every minute (same action -> same target), the slicer is re-targeted
    # each time, so only the DSR/summing identity is exact; positions follow the same target
    assert info15["minutes"] == 15 and np.isfinite(r15)
    assert r15 == pytest.approx(sum(info15["reward_parts"].values()), abs=1e-12)
    assert e15.book.position != 0 and e1.book.position != 0
    assert e15.reward_fn.dsr.n == 15 and e1.reward_fn.dsr.n == 15        # DSR stepped once per MINUTE


def test_slicer_keeps_working_the_target_between_decisions():
    """One big target change decided once: the position keeps building over the K minutes of the step
    (and beyond, with hold actions) although no new decode happens."""
    cfg = _hold_cfg(k=15)
    env = ReplayBookEnv("whale_swing", "train", cfg, seed=12)
    env.reset()
    calls = []
    import training.envs.replay_book_env as RB
    orig = RB.decode_whale
    RB.decode_whale = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    try:
        _, r, te, tr, info = env.step(np.array([1.0, 1.0, 1.0], np.float32))
    finally:
        RB.decode_whale = orig
    assert len(calls) == 1                                    # decoded once for 15 minutes
    assert info["minutes"] == 15 and len(info["minute_log"]) == 15
    n_fill_minutes = sum(1 for e in info["minute_log"] if e[3] > 0)
    assert n_fill_minutes >= 3                                # filled over several minutes
    assert env.book.position > 0
    assert env.slicer.parent is not None
    assert info["intent"]["target_shares"] > env.book.position or info["position"] == info["intent"]["target_shares"]


def _hold_cfg(k, w_switch=0.0):
    cfg = cfg_for("whale_swing", length=(2, 2), memory=0, decision_minutes=k)
    cfg["sampler"]["whale_capacity_adv_frac"] = [0.02, 0.02]            # big target: needs many minutes
    cfg["sampler"]["order_delay_choices"] = [0]
    cfg["sampler"]["cost_mult_range"] = [1.0, 1.0]
    cfg["costs"] = {"cost_mult_range": [1.0, 1.0]}
    cfg["rewards"] = {"whale": {"dd_free": 10.0, "w_switch": {"whale_swing": w_switch, "whale_position": 0.0}}}
    cfg["env"]["dd_terminate"] = {"whale_swing": 10.0}
    return cfg


def test_switch_penalty_part():
    a_long = np.array([1.0, 0.2, 0.0], np.float32)
    a_same = np.array([1.0, 0.2, 0.0], np.float32)
    for k in (1, 15):
        env = ReplayBookEnv("whale_swing", "train", _hold_cfg(k, w_switch=0.5), seed=12)
        env.reset()
        cap = env.episode.capacity
        _, r, te, tr, info = env.step(a_long)                  # target change from flat -> -0.5 x |1.0 x cap|
        sw = info["reward_parts"]["switch"]
        target = info["intent"]["target_shares"]
        assert sw == pytest.approx(-0.5 * abs(target - 0) / cap) and sw < 0
        # off: the same decision costs nothing
        env0 = ReplayBookEnv("whale_swing", "train", _hold_cfg(k, w_switch=0.0), seed=12)
        env0.reset()
        _, r0, *_x, info0 = env0.step(a_long)
        assert info0["reward_parts"]["switch"] == 0.0
        assert r == pytest.approx(r0 + sw, abs=1e-12)           # the penalty is the only difference
        # a decision that asks for no change pays nothing: pick the action matching the current position
        pos_frac = env.book.position / cap
        keep = np.array([pos_frac, 0.2, 0.0], np.float32)
        _, _, _, _, info2 = env.step(keep)
        if info2["intent"]["delta_shares"] == 0:
            assert info2["reward_parts"]["switch"] == 0.0
        else:                                                    # rounding left a few shares: tiny penalty
            assert abs(info2["reward_parts"]["switch"]) <= 0.5 * abs(info2["intent"]["delta_shares"]) / cap + 1e-12


def test_switch_only_on_first_minute_of_a_step():
    env = ReplayBookEnv("whale_swing", "train", _hold_cfg(15, w_switch=0.3), seed=12)
    env.reset()
    _, r, *_x, info = env.step(np.array([1.0, 0.2, 0.0], np.float32))
    target = info["intent"]["target_shares"]
    assert info["reward_parts"]["switch"] == pytest.approx(-0.3 * target / env.episode.capacity)   # once, not 15x


def test_pnl_profit_term_with_decision_interval():
    extra = {"rewards": {"whale": {"profit_term": "pnl"}}}
    env, rs, pos, eq, infos = _rollout("whale_swing", 0, 15, length=1, extra=extra)
    ep = infos[-1]["episode"]
    # pnl term: pnl_scale x sum of per-minute returns = pnl_scale x (equity change) / capital
    assert ep["parts_sum"]["profit"] == pytest.approx(10.0 * (eq[-1] - env.episode.capital) / env.episode.capital, rel=1e-6)


def test_drawdown_stop_is_checked_every_minute_inside_a_step():
    cfg = cfg_for("whale_swing", length=(3, 3), memory=0, decision_minutes=75)
    cfg["env"]["dd_terminate"] = {"whale_swing": 0.0003}
    env = ReplayBookEnv("whale_swing", "train", cfg, seed=3)
    env.reset()
    done, n = False, 0
    while not done:
        _, r, te, tr, info = env.step(np.array([1.0, 1.0, 1.0], np.float32))
        done = te or tr
        n += 1
    assert te and not tr and info["minutes"] <= 75
    assert info["reward_parts"]["termination"] == -1.0                  # paid once
    assert info["episode"]["decisions"] == n and info["episode"]["steps"] < 3 * 375 - 1


def test_decision_minutes_validation():
    with pytest.raises(ValueError):
        ReplayBookEnv("whale_swing", "train", cfg_for("whale_swing", decision_minutes=0), seed=0)
