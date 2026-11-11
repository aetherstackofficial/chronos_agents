"""ReplayBookEnv: a gymnasium environment where a whale or a market maker trades
INSIDE the real matching engine while it replays real 1-minute bars of one stock.

One episode = one stock, a few sessions. Real prints (GHOST_HIST) and a SIM_LP
ladder come from the replayer; the agent's orders go through ``handle_message``
exactly like a live agent's, so it has real impact on the book. The agent ids are
``TRAIN_WHALE`` / ``TRAIN_MM`` (not STRAT_/USER_), so the engine stays in REPLAY.

Observation = ``agents.features_v2.build_observation_v2`` (whale 93, mm 64) and the
action decoders are the live ones (``agents.action_decode`` + ``PovSlicerV2``).

Timeline of one episode (T = length x 375 engine ticks after the warm-up session):

    reset():  pre-roll memory -> warm-up session (agent idle) -> episode tick 1
              -> observation 1
    step():   decide on the last tick -> send orders -> physics_tick -> fills,
              memory, book, costs -> reward (per minute) -> observation.
              One step spans ``decision_minutes`` (K) 1-minute bars: the action is decoded
              on the first one, the PovSlicer keeps working the target on the other K-1;
              the step reward is the sum of the K per-minute rewards.
    the episode has T - 1 minutes (ceil((T - 1) / K) steps); the last step is
    truncated (or terminated by the drawdown stop, checked every minute) and the open
    position is charged its closing cost.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agents.action_decode import AgentState, decode_mm, decode_whale
from agents.execution_v2 import PovSlicerV2
from agents.features_v2 import OBS_SPEC, AgentBook, build_observation_v2
from core.market_memory import MarketMemory
from engine.matching_engine import MatchingEngine
from training.data.episode_bars import SESSION_BARS, load_stock
from training.envs.config import resolve_config
from training.envs.costs import CarryTracker, CostModel
from training.envs.replay_from_bars import start_replay_from_bars
from training.envs.rewards import MMReward, WhaleReward
from training.envs.sampler import MODEL_NAMES, Episode, EpisodeSampler, preroll_arrays

LAST_MINUTE = 375          # the engine stamps session bars 1..375


def charge_fee(book: AgentBook, fee: float) -> None:
    """Book a fee that belongs to no fill (delivery surcharge, closing cost)."""
    book.charge(fee)


def _noise_mask(role: str) -> np.ndarray:
    """1.0 for the market / memory / time inputs (noised), 0.0 for the rest."""
    m = np.zeros(len(OBS_SPEC[role]), dtype=np.float32)
    market = {"vwap_dist", "bar_range", "volume_ratio", "rsi", "order_flow",
              "tod_sin", "tod_cos", "to_close", "dow_sin", "dow_cos"}
    for i, n in enumerate(OBS_SPEC[role]):
        if n in market or n.startswith("mem_"):
            m[i] = 1.0
    return m


class ReplayBookEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, model_name: str, split: str = "train", cfg=None, seed: int = 0) -> None:
        super().__init__()
        if model_name not in MODEL_NAMES:
            raise ValueError(f"model_name must be one of {MODEL_NAMES}")
        self.model_name = model_name
        self.role = "mm" if model_name == "mm" else "whale"
        self.agent_id = "TRAIN_MM" if self.role == "mm" else "TRAIN_WHALE"
        self.split = split
        self.cfg = resolve_config(cfg)
        self.sampler = EpisodeSampler(model_name, split, self.cfg, seed)
        self._init_seed = seed
        self._seeded = False
        n_obs = len(OBS_SPEC[self.role])
        n_act = 4 if self.role == "mm" else 3
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (n_act,), np.float32)
        self._noise_mask = _noise_mask(self.role)
        self._dd_limit = float(self.cfg["env"]["dd_terminate"][model_name])
        self._noise_mode = self.cfg["env"]["obs_noise_mode"]
        # minutes between two decisions (one env step = this many 1-minute bars)
        self.decision_minutes = int(self.cfg["env"].get("decision_minutes", {}).get(model_name, 1))
        if self.decision_minutes < 1:
            raise ValueError("env.decision_minutes must be >= 1")
        # evaluator-visible state
        self.engine: Optional[MatchingEngine] = None
        self.memory: Optional[MarketMemory] = None
        self.book: Optional[AgentBook] = None
        self.last_tick: Optional[dict] = None
        self.episode: Optional[Episode] = None
        self.slicer: Optional[PovSlicerV2] = None
        self._done = True

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _trade_fills(self, pubs) -> List[Tuple[str, int, float]]:
        aid = self.agent_id
        out = []
        for topic, p in pubs:
            if topic == "trade":
                if p["buyer"] == aid:
                    out.append(("BUY", int(p["qty"]), float(p["price"])))
                if p["seller"] == aid:
                    out.append(("SELL", int(p["qty"]), float(p["price"])))
        return out

    @staticmethod
    def _tick_of(pubs) -> dict:
        for topic, p in reversed(pubs):
            if topic == "tick":
                return p
        raise RuntimeError("physics_tick published no tick")

    def _send(self, orders: List[dict]) -> List[Tuple[str, int, float]]:
        """Send orders through the engine's message plane; return this agent's fills."""
        fills: List[Tuple[str, int, float]] = []
        for o in orders:
            if o.get("cancel_all"):
                msg = {"msg": "CANCEL_ALL", "agent_id": self.agent_id}
            else:
                msg = {"msg": "ORDER", "agent_id": self.agent_id, **o}
            ack, pubs = self.engine.handle_message(msg)
            if ack.get("status") == "REJECTED":
                self._rejects += 1
            if self.slicer is not None:
                self.slicer.on_ack(ack)
            fills += self._trade_fills(pubs)
        return fills

    def _ledger(self) -> dict:
        return self.engine.agent_ledger[self.agent_id]

    def _apply_fills(self, fills, log: list) -> float:
        """Record fills in the book; return the fees (money) paid."""
        fees = 0.0
        for side, qty, price in fills:
            fee = self.cost.fill_fee(qty, price)
            self.book.on_fill(side, qty, price, fee)
            self.carry.on_position(self.book.position)
            fees += fee
            log.append((side, qty, price))
        return fees

    def _observe_tick(self, tick: dict) -> None:
        self.memory.update(tick)
        self.book.on_tick(tick)
        self.last_tick = tick

    def _obs(self) -> np.ndarray:
        obs = build_observation_v2(self.role, self.last_tick, self.memory, self.book)
        std = self.episode.obs_noise
        if std > 0.0:
            z = self.np_random.standard_normal(obs.shape[0]).astype(np.float32) * np.float32(std)
            if self._noise_mode == "relative":
                obs = obs * (1.0 + z * self._noise_mask)
            else:
                obs = obs + z * self._noise_mask
        return np.nan_to_num(obs.astype(np.float32, copy=False), copy=False)

    # ------------------------------------------------------------------ #
    # gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None and not self._seeded:
            seed = self._init_seed
        super().reset(seed=seed)
        self._seeded = True
        ep = self.episode = self.sampler.draw(self.np_random)
        df = load_stock(ep.symbol, self.sampler.parquet_dir)

        # engine + replay tape (warm-up session + episode sessions)
        eng = self.engine = MatchingEngine(db=None)
        start_replay_from_bars(eng, ep.bars, ep.symbol, ep.sector, ep.engine_seed)
        eng.paused = False
        # SIM_LP quote size: instance attributes shadow the class constants
        eng.LP_BASE_SIZE = type(eng).LP_BASE_SIZE * ep.lp_size_mult
        eng.LP_VOLUME_FACTOR = type(eng).LP_VOLUME_FACTOR * ep.lp_size_mult

        # memory pre-roll: parquet minute 0..374 -> engine minute 1..375; day numbers
        # -M..-1 (engine days are 1, 2, ...: no collision, and the warm-up session,
        # day 1, always differs from the last pre-rolled day so it closes it).
        mem = self.memory = MarketMemory()
        M = ep.memory_sessions
        if M > 0:
            warm = ep.warmup_session
            days, mins, o, h, l, c, v = preroll_arrays(df, ep.memory_first_session, M)
            add = mem.add_bar
            for d, m, o_, h_, l_, c_, v_ in zip(days, mins, o, h, l, c, v):
                add(d - warm, m + 1, o_, h_, l_, c_, v_)

        self.book = AgentBook(ep.capital, ep.capacity, ep.risk_tolerance,
                              n_outputs=4 if self.role == "mm" else 3)
        self.cost = CostModel.from_config(self.cfg, ep.cost_mult)
        self.carry = CarryTracker()
        self.slicer = PovSlicerV2(self.agent_id, volume_source="tick") if self.role == "whale" else None
        self._rejects = 0
        # create the agent's engine ledger (and sync the slicer to it)
        ack, _ = eng.handle_message({"msg": "CANCEL_ALL", "agent_id": self.agent_id})
        if self.slicer is not None:
            self.slicer.on_ack(ack)

        # warm-up session: the agent does not act, there is no reward. With M = 0 the
        # memory must stay empty, so the warm-up ticks only feed the book.
        if ep.warmup_session is not None:
            for _ in range(SESSION_BARS):
                tick = self._tick_of(eng.physics_tick())
                if M > 0:
                    mem.update(tick)
                self.book.on_tick(tick)
                self.last_tick = tick
        # first tick of the episode: the first observation shows the opening bar
        self._observe_tick(self._tick_of(eng.physics_tick()))

        self._total_ticks = ep.length * SESSION_BARS
        self._ep_ticks = 1
        self._done = False
        if self.role == "whale":
            self.reward_fn = WhaleReward(self.cfg, self.model_name, ep.risk_tolerance)
        else:
            self.reward_fn = MMReward(self.cfg, ep.capital, ep.capacity, ep.max_quote_size)
        self._parts_sum: Dict[str, float] = {}
        self._n_fills = 0
        self._n_decisions = 0
        self._fees_total = 0.0
        info = {"symbol": ep.symbol, "split": ep.split, "start_session": ep.start_session,
                "length_sessions": ep.length, "memory_sessions": M, "capacity": ep.capacity,
                "capital": ep.capital, "risk_tolerance": ep.risk_tolerance,
                "cost_mult": ep.cost_mult, "order_delay": ep.order_delay,
                "lp_size_mult": ep.lp_size_mult, "adv20": ep.adv20,
                "max_quote_size": ep.max_quote_size}
        return self._obs(), info

    def _decide(self, a: np.ndarray, tick: dict):
        book, ep = self.book, self.episode
        led = self._ledger()
        cash = ep.capital + (led["cash"] - led["start_cash"])
        if self.role == "whale":
            self.slicer.on_ack({"pos": led["pos"], "cash": led["cash"]})   # ledger sync
            state = AgentState(position=int(book.position), cash=float(cash), capacity=ep.capacity)
            res = decode_whale(a, state, tick)
            it = res.intent
            day, bar = int(tick["day_count"]), int(tick["market_minute"])
            orders: List[dict] = []
            if it["delta_shares"] != 0:
                orders += self.slicer.retarget(int(it["target_shares"]), it["side"], day, bar,
                                               it["participation"], it["aggressiveness"])
            elif self.slicer.parent is not None:
                # the policy now wants no change: stop the parent where it is (as live)
                self.slicer.parent.target_pos = int(book.position)
                self.slicer.parent.revisions += 1
            orders += self.slicer.tick_orders(tick)
            return orders, res
        state = AgentState(position=int(book.position), cash=float(cash), capacity=ep.capacity,
                           max_quote_size=ep.max_quote_size)
        res = decode_mm(a, state, tick)
        return list(res.orders), res

    def _hold_orders(self, tick: dict) -> List[dict]:
        """Minutes 2..K of a step: no new decision, the slicer keeps working the last target
        (whale), the market maker's quotes simply keep resting (mm: no orders)."""
        if self.role != "whale":
            return []
        led = self._ledger()
        self.slicer.on_ack({"pos": led["pos"], "cash": led["cash"]})       # ledger sync, as _decide
        return self.slicer.tick_orders(tick)

    def _minute(self, orders: List[dict], res, switch: float) -> dict:
        """Advance the engine ONE 1-minute bar with ``orders`` (decided on the last tick)."""
        ep, book, eng = self.episode, self.book, self.engine
        eq0 = book.equity
        fills_raw: List[Tuple[str, int, float]] = []
        if ep.order_delay == 0:
            fills_raw += self._send(orders)
            pubs = eng.physics_tick()
            fills_raw += self._trade_fills(pubs)
        else:
            pubs = eng.physics_tick()
            fills_raw += self._trade_fills(pubs)
            fills_raw += self._send(orders)         # decided on the old tick, sent after this one
        if eng.mode != "REPLAY":
            raise RuntimeError("engine left REPLAY mode (agent id latched the engine)")
        new_tick = self._tick_of(pubs)

        fills: List[Tuple[str, int, float]] = []
        fees = self._apply_fills(fills_raw, fills)
        self._observe_tick(new_tick)
        self._ep_ticks += 1
        final = self._ep_ticks >= self._total_ticks
        extra = 0.0                                   # carry surcharge / closing cost (money)
        price = float(new_tick["last_price"])
        if not final and int(new_tick["market_minute"]) == LAST_MINUTE and book.position != 0:
            sur = self.carry.session_close(book.position, price, self.cost)
            charge_fee(book, sur)
            extra += sur
        dd = book.drawdown
        terminated = dd >= self._dd_limit
        if terminated or (final and self.cfg["env"].get("close_at_end", True)):
            spread = float(new_tick.get("spread") or 0.0)
            cc = self.cost.closing_cost(book.position, price, spread)
            charge_fee(book, cc)
            extra += cc
        done = final or terminated
        eq1 = book.equity
        R = (eq1 - eq0) / ep.capital

        if self.role == "whale":
            parts, clipped = self.reward_fn.step(R, dd, book.position, ep.capacity, terminated, switch)
        else:
            signed = [((q if s == "BUY" else -q), p, self.cost.fill_fee(q, p)) for s, q, p in fills]
            two_sided = bool(res.intent.get("bid_qty", 0) > 0 and res.intent.get("ask_qty", 0) > 0)
            parts = self.reward_fn.step(
                signed, float(new_tick.get("mid_price") or price), extra, book.position,
                float(new_tick.get("ofi") or 0.0), float(new_tick.get("tick_size") or 0.0),
                two_sided, terminated, done)
            clipped = False
        return {"parts": parts, "clipped": clipped, "fills": fills, "fees": fees + extra,
                "eq1": eq1, "dd": dd, "terminated": terminated, "final": final, "done": done,
                "log": (int(new_tick["day_count"]), int(new_tick["market_minute"]), eq1, len(fills))}

    def step(self, action):
        """One decision = ``self.decision_minutes`` 1-minute bars (fewer at the end of the
        episode).  The action is decoded once, on the first minute; on the following minutes
        the slicer keeps working that target.  The reward is the sum of the per-minute rewards
        (the differential Sharpe ratio still sees every minute)."""
        if self._done:
            raise RuntimeError("step() called after the episode ended; call reset()")
        ep, book = self.episode, self.book
        a = np.nan_to_num(np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0))
        eq_start = book.equity
        pos_before = int(book.position)

        orders, res = self._decide(a, self.last_tick)
        switch = 0.0
        if self.role == "whale" and res.intent["delta_shares"] != 0:
            switch = abs(int(res.intent["target_shares"]) - pos_before) / float(ep.capacity)

        parts: Dict[str, float] = {}
        fills: List[Tuple[str, int, float]] = []
        fees = 0.0
        clipped = False
        minutes = 0
        minute_log: List[tuple] = []        # (day, market_minute, equity, n_fills) of every minute
        m = None
        for i in range(self.decision_minutes):
            if i > 0:
                orders = self._hold_orders(self.last_tick)
            m = self._minute(orders, res, switch if i == 0 else 0.0)
            minutes += 1
            minute_log.append(m["log"])
            for k, v in m["parts"].items():
                parts[k] = parts.get(k, 0.0) + v
            fills += m["fills"]
            fees += m["fees"]
            clipped = clipped or m["clipped"]
            if m["done"]:
                break
        terminated, final, done = m["terminated"], m["final"], m["done"]
        eq1, dd = m["eq1"], m["dd"]
        reward = float(sum(parts.values()))
        for k, v in parts.items():
            self._parts_sum[k] = self._parts_sum.get(k, 0.0) + v
        self._n_fills += len(fills)
        self._fees_total += fees
        self._n_decisions += 1
        book.set_last_action(a)
        self._done = done
        obs = self._obs()
        info = {"reward_parts": parts, "equity": eq1, "position": book.position,
                "step_return": (eq1 - eq_start) / ep.capital, "drawdown": dd, "fills": fills,
                "fees": fees, "notes": res.notes, "intent": res.intent, "dsr_clipped": clipped,
                "rejects": self._rejects, "minutes": minutes, "minute_log": minute_log}
        if done:
            info["episode"] = {"parts_sum": dict(self._parts_sum), "fills": self._n_fills,
                               "fees": self._fees_total, "steps": self._ep_ticks - 1,
                               "decisions": self._n_decisions,
                               "dsr_clips": getattr(getattr(self.reward_fn, "dsr", None), "n_clipped", 0),
                               "symbol": ep.symbol, "start_session": ep.start_session,
                               "memory_sessions": ep.memory_sessions, "length_sessions": ep.length,
                               "final_equity": eq1, "capital": ep.capital}
        return obs, reward, bool(terminated), bool(final and not terminated), info
