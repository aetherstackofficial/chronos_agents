import logging
import os
import random
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.environments import MarketMakerEnv, SmartMoneyEnv
from agents.execution import PovSlicer, VolumeCurve, skew_ticks, snap
from agents.features import build_observation
from agents.heuristic_retail import CONFIG as RETAIL_CONFIG
from agents.heuristic_retail import (Position, conviction, draw, draw_exit_plan, exit_reason,
                                     limit_price, order_qty, sample_personality)

logger = logging.getLogger(__name__)
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models")

_dummy_df = pd.DataFrame({k: v * 10 for k, v in {
    "date": ["2026-01-01"], "time": ["09:30:00"], "Open": [100.0], "High": [100.0],
    "Low": [100.0], "Close": [100.0], "Volume": [1000], "VWAP": [100.0],
    "Volume_MA": [1000.0], "Order_Flow_Imbalance": [0.0], "RSI_14": [50.0]
}.items()})

mm_norm_env = VecNormalize.load(os.path.join(_MODELS_DIR, "mm_vec_normalize_stats.pkl"), DummyVecEnv([lambda: MarketMakerEnv(_dummy_df, max_capacity=5000)]))
mm_norm_env.training = mm_norm_env.norm_reward = False
mm_model = PPO.load(os.path.join(_MODELS_DIR, "market_maker_us_base.zip"))

whale_norm_env = VecNormalize.load(os.path.join(_MODELS_DIR, "sm_vec_normalize_stats.pkl"), DummyVecEnv([lambda: SmartMoneyEnv(_dummy_df, max_capacity=10000)]))
whale_norm_env.training = whale_norm_env.norm_reward = False
whale_model = PPO.load(os.path.join(_MODELS_DIR, "smart_money_us_base.zip"))

# Expected NSE volume per minute, shared by every retail agent's participation cap.
_RETAIL_CURVE = VolumeCurve()

def _mid_price(tick: dict) -> float:
    return float(tick.get("mid_price") or tick["last_price"])


class _LiveAgentBase:
    max_position: int = 0

    def __init__(self, agent_id: str, clock=None) -> None:
        self.agent_id = agent_id
        self.cash, self.inventory, self.is_asleep, self.last_trade = 0.0, 0, False, 0.0
        self.start_equity: float | None = None
        # Injectable clock for tests; None reads the time module at call time
        # (the validation harness swaps that module for a virtual clock).
        self._clock = clock

    def _now(self) -> float:
        return self._clock() if self._clock is not None else time.time()

    def on_tick(self, tick: dict) -> list[dict]:
        raise NotImplementedError

    def on_ack(self, ack: dict) -> None:
        if (cash := ack.get("cash")) is not None: self.cash = float(cash)
        if (pos := ack.get("pos")) is not None: self.inventory = int(pos)

    def _port_return(self, tick: dict) -> float:
        equity = self.cash + self.inventory * float(tick["last_price"])
        if self.start_equity is None:
            if self.cash <= 0: return 0.0
            self.start_equity = equity
        return (equity - self.start_equity) / self.start_equity if self.start_equity else 0.0

    def _cap_qty(self, side: str, qty: int) -> int:
        if self.max_position <= 0 or qty <= 0: return max(0, qty)
        return max(0, min(qty, int(self.max_position - self.inventory if side == "BUY" else self.max_position + self.inventory)))


class LiveMarketMaker(_LiveAgentBase):
    COOLDOWN_S, MAX_BPS, MIN_BPS, QUOTE_TIF_TICKS = 1.0, 50.0, 2.0, 5

    def __init__(self, agent_id: str, clock=None, rng=None) -> None:
        super().__init__(agent_id, clock)
        self.max_position = self.max_capacity = (rng or random).randint(5_000, 25_000)

    def on_tick(self, tick: dict) -> list[dict]:
        if self.is_asleep or (now := self._now()) - self.last_trade < self.COOLDOWN_S: return []

        obs = mm_norm_env.normalize_obs(build_observation(tick, self._port_return(tick)))
        a = mm_model.predict(obs, deterministic=True)[0][0]
        
        bid_bps = max(self.MIN_BPS, ((float(a[0]) + 1.0) / 2.0) * self.MAX_BPS)
        ask_bps = max(self.MIN_BPS, ((float(a[1]) + 1.0) / 2.0) * self.MAX_BPS)
        quote_qty = int(max(10.0, min(((float(a[2]) + 1.0) / 2.0) * self.max_capacity, 3000.0)))

        mid = _mid_price(tick)
        bid_qty, ask_qty = self._cap_qty("BUY", quote_qty), self._cap_qty("SELL", quote_qty)

        # Phase 4 inventory skew (uncalibrated fixed rule): both quotes move
        # against inventory by up to 3 ticks, then snap to the engine tick.
        tick_size = float(tick.get("tick_size") or 0.0)
        self.last_skew_ticks = skew_ticks(self.inventory, self.max_position)
        shift = self.last_skew_ticks * tick_size

        self.last_trade = now
        orders: list[dict] = [{"cancel_all": True}]
        if bid_qty > 0: orders.append({"action": "BUY", "type": "LIMIT", "price": snap(mid * (1.0 - bid_bps / 10_000.0) + shift, tick_size), "qty": bid_qty, "tif_ticks": self.QUOTE_TIF_TICKS})
        if ask_qty > 0: orders.append({"action": "SELL", "type": "LIMIT", "price": snap(mid * (1.0 + ask_bps / 10_000.0) + shift, tick_size), "qty": ask_qty, "tif_ticks": self.QUOTE_TIF_TICKS})
        return orders


class LiveWhaleAgent(_LiveAgentBase):
    COOLDOWN_S, MAX_CLIP = 2.0, 10_000

    def __init__(self, agent_id: str, clock=None, rng=None, slicer: PovSlicer | None = None) -> None:
        super().__init__(agent_id, clock)
        self.max_position = self.max_capacity = (rng or random).randint(50_000, 200_000)
        # Phase 4: the policy sets a target; the slicer works it as POV
        # children (agents/execution.py, agents/config/phase4_execution.yaml).
        self.slicer = slicer or PovSlicer(agent_id)

    def on_ack(self, ack: dict) -> None:
        super().on_ack(ack)
        self.slicer.on_ack(ack)

    def on_tick(self, tick: dict) -> list[dict]:
        if self.is_asleep: return []
        orders: list[dict] = []
        day, bar = int(tick["day_count"]), int(tick["market_minute"])

        if (now := self._now()) - self.last_trade >= self.COOLDOWN_S and bar < self.slicer.last_child_bar:
            obs = whale_norm_env.normalize_obs(build_observation(tick, self._port_return(tick)))
            action = whale_model.predict(obs, deterministic=True)[0]

            delta = int(float(np.clip(action[0][0], -1.0, 1.0)) * self.max_capacity) - self.inventory
            delta = max(-self.MAX_CLIP, min(delta, self.MAX_CLIP))
            qty = self._cap_qty(side := "BUY" if delta > 0 else "SELL", abs(delta)) if delta else 0
            if qty:
                self.last_trade = now
                orders += self.slicer.retarget(self.inventory + (qty if side == "BUY" else -qty), side, day, bar)
            elif self.slicer.parent is not None:
                # The policy now wants no change: stop the parent where it is.
                self.slicer.parent.target_pos = self.inventory
                self.slicer.parent.revisions += 1

        return orders + self.slicer.tick_orders(tick)


class LiveRetailAgent(_LiveAgentBase):
    """One retail trader with its own personality (agents/heuristic_retail.py).

    Holds at most one position and has at most one order outstanding:
    flat -> entry order -> position with its own exit plan -> exit order ->
    flat. Every order is a MARKET or a passive LIMIT order. A resting limit
    order is resolved with CANCEL_ALL once its time-to-live has passed: the
    ack carries the engine's authoritative cash and position, from which the
    fill is worked out (the engine does not push fills of resting orders).

    Emotions (agents/config/retail_personalities.yaml `emotions`): after a
    realised loss the agent may trade harder; some trades are closed early
    out of fear of losing the profit; three realised losses in a row end its
    day. Every rule is symmetric and none sets a price direction.
    """
    SESSION_MINUTES = int(RETAIL_CONFIG["meta"]["session_minutes"])
    # An order whose ack never arrived (orchestrator logs a timeout) is
    # forgotten after this many minutes so the agent cannot stay blocked.
    PENDING_TIMEOUT_TICKS = 3

    def __init__(self, agent_id: str, clock=None, rng=None, seed=None, cfg=None) -> None:
        super().__init__(agent_id, clock)
        self._cfg = RETAIL_CONFIG if cfg is None else cfg
        # A run seed fixes the whole crowd. Without one (tests, ad-hoc use) it
        # is drawn from the global generator, which the validation harness
        # seeds, so seeded harness runs stay reproducible.
        self.seed = random.randrange(2 ** 31) if seed is None else seed
        self.personality = sample_personality(agent_id, self.seed, self._cfg)
        # Runtime decisions use their own stream, separate from the
        # personality stream, so personality changes never shift them.
        self._rng = rng or random.Random(f"{self.seed}:{agent_id}:run")
        self.cash = self.personality.start_cash
        self._last_trade_t: int | None = None
        self.position: Position | None = None
        self.closed_trades: list[dict] = []   # realised results, newest last
        self._pending: dict | None = None     # order sent, submit-ack not back yet
        self._working: dict | None = None     # limit order resting in the book
        self._order_n = 0
        # emotions
        self._day = -1
        self.loss_streak = 0
        self.halted_day: int | None = None    # day of the last capitulation
        self._aggr_left = 0                   # entries still sent in "overtrading" mode
        self._aggr_size = 1.0

    def on_liquidation(self) -> None:
        # Mirrors the engine's reset: the agent's own starting cash, no shares.
        self.cash, self.inventory = self.personality.start_cash, 0
        self.position, self._pending, self._working = None, None, None

    def _tick_index(self, tick: dict) -> int:
        """Engine minutes since the run started. Cooldowns and holding times
        count these, never wall-clock seconds, so a run does not depend on how
        fast it executes."""
        return int(tick["day_count"]) * self.SESSION_MINUTES + int(tick["market_minute"])

    # -- acks ----------------------------------------------------------------
    def on_ack(self, ack: dict) -> None:
        super().on_ack(ack)  # cash and inventory
        if self._working is not None and "cancelled" in ack:
            self._resolve_working()
            return
        pend = self._pending
        if pend is None or ack.get("client_order_id") != pend["cid"]:
            return
        self._pending = None
        if ack.get("status") == "REJECTED":
            return
        executed, avg = int(ack.get("executed_qty") or 0), float(ack.get("average_price") or 0.0)
        if executed > 0 and avg > 0:
            self._book_fill(pend, executed, avg)
        if pend["otype"] == "LIMIT" and int(ack.get("resting_qty") or 0) > 0:
            # cash0/pos0 already include any immediate fill of this order.
            self._working = {**pend, "cash0": self.cash, "pos0": self.inventory,
                             "resolve_t": pend["t"] + pend["tif"]}

    def _resolve_working(self) -> None:
        """The CANCEL_ALL ack is back: whatever the resting order traded shows
        up as the change in cash and position since it was placed."""
        w, self._working = self._working, None
        dpos, dcash = self.inventory - w["pos0"], self.cash - w["cash0"]
        if dpos != 0 and w["side"] * dpos > 0:
            self._book_fill(w, abs(dpos), abs(dcash) / abs(dpos))

    def _book_fill(self, order: dict, qty: int, avg: float) -> None:
        """Record `qty` shares of an entry or exit order filled at `avg`."""
        if order["kind"] == "entry":
            pos = self.position
            if pos is None:
                self.position = Position(side=order["side"], qty=qty, entry_price=avg,
                                         entry_t=order["t"], plan=order["plan"], best_price=avg)
            else:  # a resting entry order filled in more than one piece
                total = pos.qty + qty
                pos.entry_price = (pos.entry_price * pos.qty + avg * qty) / total
                pos.qty = total
            return
        pos = self.position
        if pos is None:
            return
        pos.exited_qty += qty
        pos.exit_notional += avg * qty
        if self.inventory == 0:  # fully closed: book the realised result
            exit_price = pos.exit_notional / pos.exited_qty
            pnl = pos.side * (exit_price - pos.entry_price) * pos.exited_qty
            trade = {"side": pos.side, "qty": pos.exited_qty, "entry_price": pos.entry_price,
                     "exit_price": exit_price, "pnl": pnl,
                     "pnl_pct": pos.side * (exit_price - pos.entry_price) / pos.entry_price * 100.0,
                     "held_ticks": order["t"] - pos.entry_t, "reason": order["reason"],
                     "entry_t": pos.entry_t, "exit_t": order["t"], "plan": pos.plan.style}
            self.closed_trades.append(trade)
            self.position = None
            self._feel(trade)

    def _feel(self, trade: dict) -> None:
        """Emotional reaction to a realised result."""
        emo = self._cfg["emotions"]
        if trade["pnl"] > 0:
            self.loss_streak = 0
            return
        if trade["pnl"] == 0:
            return
        self.loss_streak += 1
        if self.loss_streak >= emo["capitulation"]["consecutive_losses"]:
            self.halted_day, self._aggr_left = self._day, 0   # done for the day
            return
        over = emo["loss_overtrade"]
        if self._rng.random() < over["probability"]:
            self._aggr_left = int(over["duration_trades"])
            self._aggr_size = draw(over["size_multiplier"], self._rng)

    # -- decisions -----------------------------------------------------------
    def on_tick(self, tick: dict) -> list[dict]:
        t, price, day = self._tick_index(tick), float(tick["last_price"]), int(tick["day_count"])
        if day != self._day:
            self._day = day
            if self._cfg["emotions"]["capitulation"]["streak_resets_on_new_day"]:
                self.loss_streak = 0
        if self._pending is not None and t - self._pending["t"] > self.PENDING_TIMEOUT_TICKS:
            self._pending = None
        if self.is_asleep or price <= 0 or self._pending is not None:
            return []
        if self._working is not None:
            # Wait for the resting order's time-to-live, then resolve it.
            return [{"cancel_all": True}] if t >= self._working["resolve_t"] else []
        self._reconcile(price, t)
        if self.halted_day == day:  # capitulated today: only dump what is left
            return self._exit_orders(tick, price, t, force="capitulation") if self.position else []
        if self.position is not None:
            return self._exit_orders(tick, price, t)
        return self._entry_orders(tick, price, t)

    def _reconcile(self, price: float, t: int) -> None:
        """Keep `position` consistent with the shares the engine says we hold
        (a liquidation or a lost ack can change them behind our back)."""
        if self.position is not None and self.inventory == 0:
            self.position = None
        elif self.position is None and self.inventory != 0:
            side = 1 if self.inventory > 0 else -1
            self.position = Position(side=side, qty=abs(self.inventory), entry_price=price, entry_t=t,
                                     plan=draw_exit_plan(self.personality, self._rng, self._cfg), best_price=price)

    def _participation_cap(self, tick: dict) -> int:
        """Largest order: a share of the shares NSE would expect this minute."""
        expected = _RETAIL_CURVE.expected(str(tick.get("symbol", "")), int(tick["market_minute"]) - 1)
        return max(1, int(self._cfg["orders"]["participation_cap"] * expected))

    def _entry_orders(self, tick: dict, price: float, t: int) -> list[dict]:
        p, cfg = self.personality, self._cfg
        pers, over = cfg["personality"], cfg["emotions"]["loss_overtrade"]
        aggressive = self._aggr_left > 0
        cooldown = p.cooldown_ticks
        threshold = p.conviction_threshold
        if aggressive:
            cooldown = max(1, round(cooldown * over["cooldown_multiplier"]))
            threshold *= over["threshold_multiplier"]
        if self._last_trade_t is not None and t - self._last_trade_t < cooldown:
            return []
        if p.kind == "noise":
            noise = pers["noise_agent"]
            if self._rng.random() >= noise["trade_prob_per_tick"]:
                return []
            side = 1 if self._rng.random() < 0.5 else -1
            strength = draw(noise["conviction"], self._rng)
        else:
            sig = conviction(float(tick["rsi"]), p, pers["logistic_slope"])
            if abs(sig) < threshold:
                return []
            if self._rng.random() >= pers["base_trade_prob"] * abs(sig):
                return []
            side, strength = (1 if sig > 0 else -1), abs(sig)
        if side < 0 and not cfg["wealth"]["allow_short"]:
            return []
        otype, limit_px, tif = self._order_type(tick, price, side, "entry")
        qty = order_qty(p, strength, limit_px or price, self.cash, side, cfg,
                        size_multiplier=self._aggr_size if aggressive else 1.0,
                        max_qty=self._participation_cap(tick))
        if qty <= 0:
            return []
        plan = draw_exit_plan(p, self._rng, cfg)
        self._last_trade_t = t
        if aggressive:
            self._aggr_left -= 1
        return [self._order(side, qty, "entry", t, otype, limit_px, tif, plan=plan)]

    def _exit_orders(self, tick: dict, price: float, t: int, force: str | None = None) -> list[dict]:
        p, pos = self.personality, self.position
        pos.mark(price)
        sig = conviction(float(tick["rsi"]), p, self._cfg["personality"]["logistic_slope"])
        reason = force or exit_reason(pos, price, t, sig, p.conviction_threshold)
        if reason is None:
            return []
        otype, limit_px, tif = self._order_type(tick, price, -pos.side, "exit", reason)
        qty = min(abs(self.inventory), self._participation_cap(tick))
        self._last_trade_t = t
        return [self._order(-pos.side, qty, "exit", t, otype, limit_px, tif, reason=reason)]

    def _order_type(self, tick: dict, price: float, side: int, kind: str, reason: str | None = None):
        """(type, limit price, time-to-live) for one order. Urgent exits are
        always MARKET; everything else rolls the market/limit dice."""
        orders = self._cfg["orders"]
        urgent = kind == "exit" and reason in orders["limit"]["urgent_exit_reasons"]
        if urgent or self._rng.random() < orders["type_mix"]["market"]:
            return "MARKET", None, 0
        lim = orders["limit"]
        offset = self._rng.choices(lim["offset_ticks_choices"], weights=lim["offset_ticks_weights"])[0]
        px = limit_price(side, tick.get("best_bid"), tick.get("best_ask"), price,
                         float(tick["tick_size"]), offset)
        return "LIMIT", px, int(draw(lim["tif_ticks"], self._rng))

    def _order(self, side: int, qty: int, kind: str, t: int, otype: str, limit_px, tif: int,
               plan=None, reason=None) -> dict:
        self._order_n += 1
        cid = f"{self.agent_id}:{'E' if kind == 'entry' else 'X'}{self._order_n}"
        self._pending = {"cid": cid, "kind": kind, "otype": otype, "tif": tif, "t": t, "plan": plan,
                         "reason": reason, "side": side}
        # start_cash is read by the engine only on this agent's first message.
        order = {"action": "BUY" if side > 0 else "SELL", "type": otype, "qty": qty,
                 "client_order_id": cid, "start_cash": self.personality.start_cash}
        if otype == "LIMIT":
            order.update(price=limit_px, tif_ticks=tif)
        return order
