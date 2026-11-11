"""Live whale and market maker driven by the v2 models (agents/ml_models/v2/<name>/).

``agents/live_agents.py`` (the old 6-input models) is left untouched. These classes reproduce, in
the live swarm, the decision path of ``training/envs/replay_book_env.py``:

    every tick:     memory.update(tick); book.on_tick(tick)          (``observe``)
    on a decision:  obs = build_observation_v2(role, tick, memory, book)
                    a   = policy(ObsScaler.transform(obs))           (deterministic)
                    decode_whale / decode_mm  (agents/action_decode.py, shared with training)
                    whale: PovSlicerV2(volume_source="tick") works the target every minute
                    book.set_last_action(a)

Differences from training that the live swarm imposes:
* the orchestrator wakes an agent on a share of ticks only; a whale decides on the first tick it is
  awake once ``decision_minutes`` (K) minutes have passed since its last decision, and its slicer
  sends children only on awake ticks. ``observe`` must still see every tick: ``install()`` patches
  ``SwarmOrchestrator._on_tick`` to feed one shared ``MarketMemory`` and every v2 agent's book
  before the agents act (paused ticks excluded, as the orchestrator's MarketContext);
* fills are not pushed by the engine: the agent's book is synced from the authoritative cash and
  position in its acks (a position change is booked as a fill at |change in cash| / |change in
  position|, which includes any engine fee);
* capacities follow the old live agents (whale 50k-200k shares, market maker 5k-25k) so that a
  comparison of old and new swarms changes the policy only. Capital is capacity x first price
  (whale) or ``mm_capital_multiple`` x capacity x price (market maker), as in the training sampler,
  and the cash the decoder sees is that capital plus the change in the engine cash.

Selection: ``install(whales=[...])`` with model names under agents/ml_models/v2/ (cycled over the
whale slots), or the environment variable CHRONOS_V2_WHALES="whale_swing,whale_position".
``CHRONOS_V2_ROLES`` (default "whale,mm") limits which roles use v2 models.

Opt-in options (environment variables; every default reproduces the plain adapter above):
* CHRONOS_V2_VOLUME=curve        whale children sized from the NSE volume curve (old live whale) instead
                                 of the tick's volume_ma (training);
* CHRONOS_V2_SEED_SYMBOL=<SYM>   memory starts from the last CHRONOS_V2_SEED_SESSIONS (150) real sessions;
                                 CHRONOS_V2_SEED_SKIP=<n> ends that history n sessions before the file end
                                 (out-of-sample comparison with the n real sessions that follow);
* CHRONOS_V2_MANDATES=1          client order flow of the institutions (MandateBook; NOT part of the
                                 trained policy -- departs from the 6 Nov rule "agents decide on their
                                 own"; CHRONOS_V2_MANDATE_* parameters);
* CHRONOS_V2_REGIME_SIGMA / _PHI day-level activity regime of that flow; CHRONOS_V2_HAWKES_* self-excitation;
* CHRONOS_V2_MM_TIF=<n>          market-maker quotes rest up to n minutes (training: 1);
* CHRONOS_V2_MM_CAP=lo,hi        market-maker capacity range in shares;
* CHRONOS_V2_WHALE_CAP=lo,hi     whale capacity range in shares (default 50000,200000);
* CHRONOS_LP=latent ...          designated-liquidity replacement, see agents/latent_liquidity.py;
* CHRONOS_MODE=backtest         the real tape keeps running with the participants' footprint on it, see
                                 agents/backtest_mode.py (CHRONOS_BT_* parameters).
Each realism run records the variables it used (tools/run_realism_v2.py, ``swarm_env``).
"""
from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional

import numpy as np
import yaml

from agents.action_decode import AgentState, decode_mm, decode_whale
from agents.execution_v2 import PovSlicerV2
from agents.features_v2 import AgentBook, build_observation_v2
from agents.obs_scaler import ObsScaler
from core.market_memory import MarketMemory

V2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models", "v2")
SESSION_MINUTES = 375
CASH_INSTITUTION = 5_000_000.0     # engine/matching_engine.py ledger seeding (WHALE*, MM*)
CASH_MARKET_MAKER = 250_000.0
MM_CAPITAL_MULTIPLE = 2.0          # training sampler: mm capital = 2 x capacity x price
MM_QUOTE_DIVISOR = 5               # training sampler: max quote = capacity // 5
RISK_TOLERANCE_RANGE = (0.03, 0.15)

_MODELS: Dict[str, "V2Model"] = {}
_SEED_BARS: Dict[tuple, tuple] = {}


def seed_bars(symbol: str, sessions: int):
    """(day, minute, o, h, l, c, v) arrays of the last ``sessions`` real sessions of ``symbol``
    (data/real/nse_kaggle/<symbol>_minute.csv, cleaned by validation.data.load_real), days numbered
    -sessions..-1 and minutes 1..375 so that the first engine day (1) continues them. This is what
    the live system does when a worker joins after the replay (FETCH_BARS + memory seed): the agents
    start with the real history that precedes the simulated market."""
    key = (symbol, sessions, os.environ.get("CHRONOS_V2_SEED_SKIP", "0"))
    if key not in _SEED_BARS:
        from validation.data import load_real
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        s = load_real(os.path.join(root, "data", "real", "nse_kaggle", f"{symbol}_minute.csv"))
        all_days = np.unique(s.day)
        skip = int(os.environ.get("CHRONOS_V2_SEED_SKIP", "0"))   # drop the last `skip` real sessions
        if skip > 0:
            all_days = all_days[:-skip]
        days = all_days[-sessions:]
        keep = np.isin(s.day, days)
        d = np.searchsorted(days, s.day[keep]) - len(days)          # -sessions .. -1
        _SEED_BARS[key] = (d.astype(int), (s.minute[keep] + 1).astype(int), s.o[keep], s.h[keep],
                           s.l[keep], s.c[keep], s.v[keep])
    return _SEED_BARS[key]


class V2Model:
    """model.zip + fixed ObsScaler + decision interval of one exported run; deterministic forward
    pass (SB3 deterministic predict = clip(action_net(actor(features))) for a Box action space)."""

    def __init__(self, name: str) -> None:
        import torch
        from stable_baselines3 import PPO
        d = os.path.join(V2_DIR, name)
        with open(os.path.join(d, "config.yaml"), encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self.name, self.model_name = name, cfg["model"]
        self.role = "mm" if self.model_name == "mm" else "whale"
        self.decision_minutes = int(cfg.get("decision_minutes") or 1)
        if int(cfg.get("frame_stack", 1)) != 1:
            raise ValueError(f"{name}: frame_stack > 1 is not supported live")
        self.scaler = ObsScaler.load(os.path.join(d, "obs_scaler.json"), self.role)
        self.model = PPO.load(os.path.join(d, "model.zip"), device="cpu")
        pol = self.model.policy
        pol.set_training_mode(False)
        if pol.squash_output:
            raise RuntimeError("v2 forward pass assumes squash_output=False")
        self._torch, self._pol = torch, pol
        self._lo, self._hi = self.model.action_space.low, self.model.action_space.high
        # self-check against SB3 on random inputs
        probe = np.random.default_rng(12345).normal(0, 2, (50, self.scaler.n_inputs)).astype(np.float32)
        for x in probe:
            if not np.allclose(self.model.predict(x, deterministic=True)[0], self._forward(x), atol=1e-6):
                raise RuntimeError(f"{name}: forward pass differs from SB3 predict")

    def __deepcopy__(self, memo):
        # stateless and shared: a simulator snapshot (validation/sim_driver.py) keeps the same model
        return self

    def _forward(self, x: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            t = self._torch.as_tensor(np.asarray(x, np.float32).reshape(1, -1))
            f = self._pol.extract_features(t, self._pol.pi_features_extractor)
            a = self._pol.action_net(self._pol.mlp_extractor.forward_actor(f)).numpy()[0]
        return np.clip(a, self._lo, self._hi)

    def act(self, raw_obs: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(np.asarray(self.scaler.transform(np.asarray(raw_obs, np.float32)), np.float32))
        return self._forward(x)


def get_model(name: str) -> V2Model:
    m = _MODELS.get(name)
    if m is None:
        m = _MODELS[name] = V2Model(name)
    return m


def _t(tick: dict) -> int:
    return (int(tick.get("day_count") or 1) - 1) * SESSION_MINUTES + int(tick.get("market_minute") or 0)


class _V2Base:
    """Same surface as agents.live_agents._LiveAgentBase (agent_id, cash, inventory, is_asleep,
    on_tick, on_ack), plus ``observe`` (every tick) and a shared ``memory`` set by the patch."""
    role = ""
    engine_start_cash = 0.0

    def __init__(self, agent_id: str, model: str, clock=None, rng=None) -> None:
        self.agent_id = agent_id
        self.m = get_model(model)
        self._rng = rng or random.Random()
        self.cash, self.inventory, self.is_asleep = self.engine_start_cash, 0, False
        self.last_trade = float("-inf")
        self._clock = clock
        self.memory: Optional[MarketMemory] = None       # shared, set by the orchestrator patch
        self.book: Optional[AgentBook] = None
        self.capacity = 0
        self.capital = 0.0
        self._synced_pos, self._synced_cash = 0, self.engine_start_cash
        self.context = None
        self.decisions = 0

    # ---- state
    def _capital_for(self, price: float) -> float:
        raise NotImplementedError

    def _ensure_book(self, tick: dict) -> bool:
        if self.book is not None:
            return True
        price = float(tick.get("last_price") or 0.0)
        if price <= 0:
            return False
        self.capital = self._capital_for(price)
        self.book = AgentBook(self.capital, self.capacity, self._risk, n_outputs=4 if self.role == "mm" else 3)
        return True

    def _sync(self) -> None:
        """Book the change in the engine position/cash since the last sync as a fill."""
        if self.book is None:
            return
        dpos, dcash = int(self.inventory) - self._synced_pos, float(self.cash) - self._synced_cash
        if dpos != 0:
            self.book.on_fill("BUY" if dpos > 0 else "SELL", abs(dpos), abs(dcash) / abs(dpos))
        elif abs(dcash) > 1e-9:
            self.book.charge(-dcash)
        self._synced_pos, self._synced_cash = int(self.inventory), float(self.cash)

    def observe(self, tick: dict) -> None:
        """Every unpaused tick, before any agent acts (memory is updated by the patch)."""
        if not self._ensure_book(tick):
            return
        self._sync()
        self.book.on_tick(tick)

    def _cash_for_decoder(self) -> float:
        return self.capital + (float(self.cash) - self.engine_start_cash)

    def on_ack(self, ack: dict) -> None:
        if (cash := ack.get("cash")) is not None:
            self.cash = float(cash)
        if (pos := ack.get("pos")) is not None:
            self.inventory = int(pos)

    def _obs(self, tick: dict) -> Optional[np.ndarray]:
        if self.book is None or self.memory is None:
            return None
        self._sync()
        return build_observation_v2(self.role, tick, self.memory, self.book)


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


class MarketRegime:
    """Market-wide intensity of client order flow, shared by all institutions of one simulation.

    * day level: activity multiplier exp(s z_d - s^2/2), z_d an AR(1) over sessions (coefficient
      ``phi``): some days are busy (news, expiries, index rebalancing), busy days cluster;
    * intraday: self-excitation (herding): every new client order raises the arrival rate of all
      institutions by ``eta``, decaying with time constant ``tau`` minutes (Hawkes process);
    * intraday activity (CHRONOS_V2_INTRA_SIGMA, _HALFLIFE minutes): a market-wide factor
      exp(si u_t - si^2/2), u_t an AR(1) over minutes, multiplies the arrival rate AND the execution
      speed of every working client order: busy and quiet spells of minutes to hours.
    Off unless CHRONOS_V2_REGIME_SIGMA, CHRONOS_V2_HAWKES_ETA or CHRONOS_V2_INTRA_SIGMA > 0.
    """

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.s = _envf("CHRONOS_V2_REGIME_SIGMA", 0.0)
        self.phi = _envf("CHRONOS_V2_REGIME_PHI", 0.8)
        self.eta = _envf("CHRONOS_V2_HAWKES_ETA", 0.0)
        self.decay = math.exp(-1.0 / max(_envf("CHRONOS_V2_HAWKES_TAU", 30.0), 1e-6))
        self.z = self.rng.gauss(0.0, 1.0)
        self.day = None
        self.excite = 0.0
        self.si = _envf("CHRONOS_V2_INTRA_SIGMA", 0.0)
        self.ia = 0.5 ** (1.0 / max(_envf("CHRONOS_V2_INTRA_HALFLIFE", 30.0), 1e-6))
        self.u = self.rng.gauss(0.0, 1.0) if self.si > 0 else 0.0   # no draw when off: default runs unchanged
        self.intra = 1.0
        self.mult = 1.0

    def on_tick(self, tick: dict) -> None:
        day = tick.get("day_count")
        if day != self.day:
            if self.day is not None:
                self.z = self.phi * self.z + math.sqrt(max(1.0 - self.phi ** 2, 0.0)) * self.rng.gauss(0.0, 1.0)
            self.day = day
        self.excite *= self.decay
        if self.si > 0:
            self.u = self.ia * self.u + math.sqrt(1.0 - self.ia ** 2) * self.rng.gauss(0.0, 1.0)
            self.intra = math.exp(self.si * self.u - 0.5 * self.si ** 2)
        self.mult = math.exp(self.s * self.z - 0.5 * self.s ** 2) * (1.0 + self.excite) * self.intra

    def on_arrival(self) -> None:
        self.excite += self.eta


class MandateBook:
    """Client mandates of one institution (opt-in: CHRONOS_V2_MANDATES=1).

    Real big players trade mostly on behalf of clients (fund flows, rebalancing, index changes):
    an order of Q shares arrives and is worked over hours or days by an execution algorithm that
    trades a fixed share p of the market's volume (POV/VWAP). Model (Lillo-Mike-Farmer 2005):
    * arrivals: Poisson, ``rate`` per session, spread over the session like the NSE volume curve
      (more orders start near the open and the close);
    * size: Pareto with tail exponent ``alpha`` (~1.5 for real metaorders), scale ``q0`` x ADV,
      capped at ``qmax`` x ADV;
    * sign: buy with probability 0.5 + ``bias`` x (the RL policy's target in [-1, 1]);
    * execution: each minute a mandate becomes due p x (expected NSE volume of that minute),
      p ~ U[pmin, pmax]; an agent that was asleep catches up on what fell due meanwhile;
      buy and sell amounts due at the same minute are crossed internally, the net is sent as
      a marketable limit order at the opposite touch (``through`` ticks beyond it);
    * a mandate stays open across sessions until it is filled.
    """

    def __init__(self, rng: random.Random, curve, symbol: str = "") -> None:
        self.rng, self.curve, self.symbol = rng, curve, symbol
        self.rate = _envf("CHRONOS_V2_MANDATE_RATE", 6.0)
        self.alpha = _envf("CHRONOS_V2_MANDATE_ALPHA", 1.5)
        self.q0 = _envf("CHRONOS_V2_MANDATE_Q0", 0.001)
        self.qmax = _envf("CHRONOS_V2_MANDATE_QMAX", 0.05)
        self.pmin = _envf("CHRONOS_V2_MANDATE_PMIN", 0.03)
        self.pmax = _envf("CHRONOS_V2_MANDATE_PMAX", 0.15)
        self.bias = _envf("CHRONOS_V2_MANDATE_BIAS", 0.3)
        self.through = int(_envf("CHRONOS_V2_MANDATE_THROUGH", 0))
        self.kappa = _envf("CHRONOS_V2_MANDATE_KAPPA", 0.0)
        c = curve.curve_for(symbol)
        w = np.asarray(c["minute_weights"], float)
        self.w = w / w.sum()                       # arrival intensity profile over the session
        self.adv = float(c["adv_shares"])
        self.active: List[dict] = []
        self.filled_total = 0
        self.crossed_total = 0
        self.n_started = 0

    def advance(self, minute_idx: int, rl_target: float, inventory: float = 0.0,
                regime: Optional["MarketRegime"] = None) -> None:
        """One market minute passed (called every tick, awake or not). ``inventory``: the
        institution's position; with ``kappa`` > 0 new client orders lean against it (fund flows
        mean-revert over long horizons), keeping the book of client positions bounded."""
        m = min(max(minute_idx, 0), len(self.w) - 1)
        mult = regime.mult if regime is not None else 1.0
        if self.rng.random() < self.rate * self.w[m] * mult:
            if regime is not None:
                regime.on_arrival()
            u = self.rng.random()
            q = min(self.q0 * self.adv * (1.0 - u) ** (-1.0 / self.alpha), self.qmax * self.adv)
            lean = self.kappa * inventory / max(self.qmax * self.adv, 1.0)
            p_buy = min(max(0.5 + self.bias * rl_target - lean, 0.05), 0.95)
            self.active.append({"sign": 1 if self.rng.random() < p_buy else -1, "remaining": int(q),
                                "p": self.rng.uniform(self.pmin, self.pmax), "due": 0.0})
            self.n_started += 1
        ev = self.curve.expected(self.symbol, m) * (regime.intra if regime is not None else 1.0)
        for md in self.active:
            md["due"] = min(md["due"] + md["p"] * ev, float(md["remaining"]))

    def child(self) -> int:
        """Signed net quantity to send now (buy > 0); crossed amounts are booked as done."""
        buy = sum(md["due"] for md in self.active if md["sign"] > 0)
        sell = sum(md["due"] for md in self.active if md["sign"] < 0)
        cross = min(buy, sell)
        net = buy - sell
        if cross > 0:            # internal cross: `cross` shares of each side are done at no impact
            for side, tot in ((1, buy), (-1, sell)):
                for md in self.active:
                    if md["sign"] == side:
                        x = md["due"] * cross / tot
                        md["due"] -= x
                        md["remaining"] -= int(round(x))
            self.active = [md for md in self.active if md["remaining"] > 0]
        self.crossed_total += int(cross)
        return int(round(net))

    def on_filled(self, signed_qty: int) -> None:
        """Book an executed child against the mandates of its side (oldest first)."""
        left = abs(int(signed_qty))
        side = 1 if signed_qty > 0 else -1
        self.filled_total += left
        for md in self.active:
            if md["sign"] != side or left <= 0:
                continue
            x = min(left, md["remaining"])
            md["remaining"] -= x
            md["due"] = max(md["due"] - x, 0.0)
            left -= x
        self.active = [md for md in self.active if md["remaining"] > 0]


class LiveWhaleAgentV2(_V2Base):
    role = "whale"
    engine_start_cash = CASH_INSTITUTION

    def __init__(self, agent_id: str, model: str = "whale_swing", clock=None, rng=None) -> None:
        super().__init__(agent_id, model, clock, rng)
        lo, hi = (int(x) for x in os.environ.get("CHRONOS_V2_WHALE_CAP", "50000,200000").split(","))
        self.max_position = self.capacity = self._rng.randint(lo, hi)
        self._risk = self._rng.uniform(*RISK_TOLERANCE_RANGE)
        # "curve": children sized from the real NSE intraday volume curve, as the old live whale
        # (agents/execution.py); "tick": from the tick's volume_ma, as in the replay training env.
        self.slicer = PovSlicerV2(agent_id, volume_source=os.environ.get("CHRONOS_V2_VOLUME", "tick"))
        self._last_decision_t: Optional[int] = None
        self._rl_target = 0.0
        self.mandates: Optional[MandateBook] = None
        if os.environ.get("CHRONOS_V2_MANDATES") == "1":
            from agents.execution import VolumeCurve
            self.mandates = MandateBook(random.Random(self._rng.getrandbits(64)), VolumeCurve(),
                                        os.environ.get("CHRONOS_V2_SEED_SYMBOL", "RELIANCE"))
        self._pending_child: Dict[str, int] = {}
        self._n_child = 0

    def _capital_for(self, price: float) -> float:
        return float(self.capacity) * price

    def observe(self, tick: dict) -> None:
        super().observe(tick)
        if self.mandates is not None and tick.get("market_minute") is not None:
            self.mandates.advance(int(tick["market_minute"]) - 1, self._rl_target, float(self.inventory),
                                  getattr(self, "regime", None))

    def on_ack(self, ack: dict) -> None:
        super().on_ack(ack)
        self.slicer.on_ack(ack)
        cid = ack.get("client_order_id")
        if cid is not None and cid in self._pending_child:
            q = self._pending_child.pop(cid)
            done = int(ack.get("executed_qty") or 0)
            if done > 0 and self.mandates is not None:
                self.mandates.on_filled(done if q > 0 else -done)

    def _mandate_orders(self, tick: dict) -> List[dict]:
        if self.mandates is None:
            return []
        q = self.mandates.child()
        bid, ask = tick.get("best_bid"), tick.get("best_ask")
        if q == 0 or bid is None or ask is None:
            return []
        ts = float(tick.get("tick_size") or 0.05)
        th = self.mandates.through * ts
        price = round((float(ask) + th) if q > 0 else (float(bid) - th), 6)
        self._n_child += 1
        cid = f"{self.agent_id}:M{self._n_child}"
        self._pending_child = {cid: q}           # only the newest child is tracked
        return [{"action": "BUY" if q > 0 else "SELL", "type": "LIMIT", "price": price, "qty": abs(q),
                 "tif_ticks": 1, "client_order_id": cid}]

    def on_tick(self, tick: dict) -> List[dict]:
        if self.is_asleep:
            return []
        if self.book is None:
            self.observe(tick)
        obs = self._obs(tick)
        if obs is None:
            return []
        orders: List[dict] = []
        day, bar, t = int(tick["day_count"]), int(tick["market_minute"]), _t(tick)
        due = self._last_decision_t is None or t - self._last_decision_t >= self.m.decision_minutes
        if due and bar < self.slicer.last_child_bar:
            a = self.m.act(obs)
            state = AgentState(position=int(self.inventory), cash=self._cash_for_decoder(), capacity=self.capacity)
            res = decode_whale(a, state, tick)
            it = res.intent
            if it["delta_shares"] != 0:
                orders += self.slicer.retarget(int(it["target_shares"]), it["side"], day, bar,
                                               it["participation"], it["aggressiveness"])
                self.last_trade = t
            elif self.slicer.parent is not None:
                self.slicer.parent.target_pos = int(self.inventory)
                self.slicer.parent.revisions += 1
            self.book.set_last_action(a)
            self._last_decision_t = t
            self._rl_target = float(a[0])
            self.decisions += 1
        mandate = self._mandate_orders(tick)
        if mandate:
            # One working order per minute: the engine skips an agent's own resting orders when
            # matching (self-trade prevention), so a client child sent next to a resting slicer
            # child of the other side could rest through it and cross the book. The client order
            # replaces this minute's slicer work; the slicer resumes from the ledger next minute.
            return [{"cancel_all": True}] + mandate
        return orders + self.slicer.tick_orders(tick)


class LiveMarketMakerV2(_V2Base):
    role = "mm"
    engine_start_cash = CASH_MARKET_MAKER

    def __init__(self, agent_id: str, model: str = "mm", clock=None, rng=None) -> None:
        super().__init__(agent_id, model, clock, rng)
        # capacity range in shares (default: the old live market maker's 5k-25k); training drew it
        # log-uniform in 0.02%-0.5% of the stock's ADV (2.4k-60k shares for RELIANCE)
        lo, hi = (int(x) for x in os.environ.get("CHRONOS_V2_MM_CAP", "5000,25000").split(","))
        self.max_position = self.capacity = self._rng.randint(lo, hi)
        self.max_quote_size = max(self.capacity // MM_QUOTE_DIVISOR, 1)
        self._risk = 0.08
        self._last_decision_t: Optional[int] = None

    def _capital_for(self, price: float) -> float:
        return MM_CAPITAL_MULTIPLE * float(self.capacity) * price

    def on_tick(self, tick: dict) -> List[dict]:
        if self.is_asleep:
            return []
        if self.book is None:
            self.observe(tick)
        obs = self._obs(tick)
        if obs is None:
            return []
        t = _t(tick)
        if self._last_decision_t is not None and t - self._last_decision_t < self.m.decision_minutes:
            return []
        a = self.m.act(obs)
        state = AgentState(position=int(self.inventory), cash=self._cash_for_decoder(), capacity=self.capacity,
                           max_quote_size=self.max_quote_size)
        # CHRONOS_V2_MM_TIF: quotes rest up to this many minutes (default 1, as in training). In
        # training the market maker re-quotes every minute; live it acts only on the minutes the
        # orchestrator wakes it, and 1-minute quotes leave it out of the book in between.
        tif = int(_envf("CHRONOS_V2_MM_TIF", 1))
        res = decode_mm(a, state, tick, {"mm_tif_ticks": tif} if tif != 1 else None)
        self.book.set_last_action(a)
        self._last_decision_t = t
        self.decisions += 1
        return list(res.orders)


# --------------------------------------------------------------------------- swarm hook
_INSTALLED = False


def install(whales: Optional[List[str]] = None, mm: str = "mm", roles: Optional[List[str]] = None) -> None:
    """Make agents.swarm_factory build v2 whales / market makers and feed their memory every tick.
    Idempotent; call it at import time of a runner script so spawned worker processes do it too."""
    global _INSTALLED
    if _INSTALLED:
        return
    import agents.swarm_factory as sf
    from core.loop_manager import SwarmOrchestrator

    whales = whales or [w for w in os.environ.get("CHRONOS_V2_WHALES", "whale_swing").split(",") if w]
    roles = roles or [r for r in os.environ.get("CHRONOS_V2_ROLES", "whale,mm").split(",") if r]
    orig_make = sf.make_agent

    def make_agent(role, index, seed, clock):
        if role == "whale" and "whale" in roles:
            aid = f"WHALE_{index}"
            return LiveWhaleAgentV2(aid, whales[index % len(whales)], clock=clock, rng=sf.agent_rng(seed, aid))
        if role == "mm" and "mm" in roles:
            aid = f"MM_{index}"
            return LiveMarketMakerV2(aid, mm, clock=clock, rng=sf.agent_rng(seed, aid))
        return orig_make(role, index, seed, clock)

    sf.make_agent = make_agent
    from agents import latent_liquidity
    latent_liquidity.install()            # no-op unless CHRONOS_LP=latent
    from agents import backtest_mode
    backtest_mode.install()               # no-op unless CHRONOS_MODE=backtest
    from agents import swarm_scale
    swarm_scale.install()                 # no-op unless CHRONOS_SCALE_* is set (after the make_agent patch)
    orig_on_tick = SwarmOrchestrator._on_tick

    async def _on_tick(self, tick: dict) -> None:
        if not tick.get("paused"):
            if getattr(self, "_v2_src", None) is not self.agents:
                # first tick, or the agent list was replaced (sim_driver.restore of a snapshot):
                # the agents' own (copied) memory is the shared one from now on
                self._v2_src = self.agents
                self._v2_agents = [a for a in self.agents if isinstance(a, _V2Base)]
                mem = next((a.memory for a in self._v2_agents if a.memory is not None), None)
                if mem is None:
                    mem = MarketMemory()
                    sym, n = os.environ.get("CHRONOS_V2_SEED_SYMBOL"), int(os.environ.get("CHRONOS_V2_SEED_SESSIONS", "150"))
                    if sym and n > 0:
                        for row in zip(*seed_bars(sym, n)):
                            mem.add_bar(*(int(x) if i < 2 else float(x) for i, x in enumerate(row)))
                for a in self._v2_agents:
                    a.memory = mem
                self._v2_memory = mem
                reg = next((a.regime for a in self._v2_agents if getattr(a, "regime", None) is not None), None)
                if reg is None and self._v2_agents:
                    reg = MarketRegime(random.Random(self._v2_agents[0]._rng.getrandbits(64)))
                for a in self._v2_agents:
                    a.regime = reg
                self._v2_regime = reg
            mem = self._v2_memory
            if self._v2_regime is not None:
                self._v2_regime.on_tick(tick)
            mem.update(tick)
            for a in self._v2_agents:
                a.observe(tick)
        await orig_on_tick(self, tick)

    SwarmOrchestrator._on_tick = _on_tick
    _INSTALLED = True
