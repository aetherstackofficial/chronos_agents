"""Headless, seeded, in-process driver for the CURRENT simulator (spec §14).

The validation suite treats the simulator as a black box that produces bars,
an event log and snapshot/restore. Nothing here changes market mechanics: the
engine (`MatchingEngine`), the orchestrator (`SwarmOrchestrator`) and the live
agents run unmodified. The driver only replaces the parts of the deployment
that are not simulation logic:

* ZMQ transport -> an in-process client whose ``send_order`` calls
  ``engine.handle_message`` directly (same handler, same agent order).

The swarm itself is built exactly as worker.py builds it
(``agents.swarm_factory``): agents count cooldowns on a ``TickClock`` read
from the tick payload and draw from per-agent RNGs seeded from the run seed,
and the orchestrator submits agents' orders one agent at a time. A seeded
worker therefore runs the same swarm as this harness. ``seed_globals=False``
reproduces a worker started without ``--seed`` (unseeded swarm).

Known, reported gaps that the driver cannot remove without changing the
simulator: one tick is one sim-minute, so there are no sub-minute events,
no 10-second book snapshots and no seconds-spaced TWAP children.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from engine.matching_engine import MatchingEngine  # noqa: E402

# Live swarm composition from worker.py DEFAULT_COUNTS with ``--role all``
# (red team is not part of "all", so it is off by default here too).
LIVE_COUNTS = {"mm": 15, "whale": 4, "retail": 80, "redteam": 0}


class _InProcClient:
    """EngineClient replacement: same send_order contract, no sockets."""

    def __init__(self, sim: "Simulation") -> None:
        self._sim = sim

    async def send_order(self, msg: dict) -> dict:
        ack, pubs = self._sim.engine.handle_message(msg)
        self._sim._absorb(pubs, between_ticks=True)
        return ack

    def close(self) -> None:
        pass


@dataclass
class Fill:
    qty: int
    avg_price: float
    available: int


@dataclass
class RunOutput:
    """Everything the tests consume from one seeded run."""
    seed: int
    bars: Dict[str, np.ndarray]          # one entry per sim-minute
    trades: Dict[str, np.ndarray]        # one entry per print
    book_snaps: Optional[np.ndarray]     # (S, 2, L) volume at d ticks from touch
    book_spreads_ticks: Optional[np.ndarray]
    invariant_violations: Dict[str, int]
    event_hash: Optional[str]
    meta: dict = field(default_factory=dict)


def _ticks(price: Optional[float], tick: float) -> Optional[int]:
    return None if price is None else int(round(price / tick))


class Simulation:
    """One seeded simulator instance driven tick by tick."""

    def __init__(self, seed: int, start_price: float, symbol: str = "SIM",
                 counts: Optional[Dict[str, int]] = None,
                 hash_events: bool = False,
                 null_strategy: bool = False,
                 seed_globals: bool = True,
                 replay_csv: Optional[str] = None) -> None:
        self.seed = seed
        self.counts = dict(LIVE_COUNTS if counts is None else counts)
        # Imports first: loading the PPO models touches global RNG state, so
        # seeding before them would make run 1 of a process differ from run 2.
        from agents.swarm_factory import TickClock, build_swarm
        self.clock = TickClock()
        self._install_fast_predict()
        if seed_globals:
            # The swarm no longer draws from these; seeded so that any
            # remaining global-RNG use still cannot make runs differ.
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
        swarm_seed = seed if seed_globals else None

        self.engine = MatchingEngine(db=None)
        if replay_csv is None:
            self.engine.init_sim(symbol, "SIM", float(start_price), seed=seed)
        else:
            self.engine.start_replay(replay_csv, symbol, "SIM", seed=seed)
        self.engine.paused = False

        self._hasher = hashlib.sha256() if hash_events else None
        if hash_events:
            self._wrap_submit_for_logging()

        from core.loop_manager import SwarmOrchestrator
        self.orch = SwarmOrchestrator(_InProcClient(self), seed=swarm_seed,
                                      clock=self.clock)
        self.orch.load_agents(build_swarm(self.counts, swarm_seed, self.clock))
        # A null strategy is a tick subscriber that never sends anything
        # (runner/sdk.py only talks to the engine when on_tick returns
        # orders), so it lives outside the orchestrator and its RNG.
        self.null_strategy = null_strategy
        self._null_ticks_seen = 0
        self._loop = asyncio.new_event_loop()

        self._in_tick = False
        self.trade_rows: List[tuple] = []
        self.bar_rows: List[tuple] = []
        self.last_payload: Optional[dict] = None

    # -- setup ------------------------------------------------------------ #
    @staticmethod
    def _install_fast_predict() -> None:
        """Swap SB3 ``predict`` for a direct deterministic forward pass.

        Performance only: for a Box action space without squashing,
        SB3's deterministic predict is clip(action_net(actor(features))).
        A startup self-check asserts bit-identical actions against SB3 on
        random observations and refuses to run if they ever differ.
        """
        import agents.live_agents as la
        import torch
        for model in (la.mm_model, la.whale_model):
            if getattr(model, "_fast_predict_ok", False):
                continue
            pol = model.policy
            pol.set_training_mode(False)
            lo, hi = model.action_space.low, model.action_space.high
            obs_shape, act_shape = model.observation_space.shape, model.action_space.shape
            if pol.squash_output:
                raise RuntimeError("fast predict assumes squash_output=False")

            def fast(obs, deterministic=True, _pol=pol, _lo=lo, _hi=hi):
                with torch.no_grad():
                    t = torch.as_tensor(np.asarray(obs, np.float32).reshape(-1, *obs_shape))
                    f = _pol.extract_features(t, _pol.pi_features_extractor)
                    a = _pol.action_net(_pol.mlp_extractor.forward_actor(f)).numpy()
                return np.clip(a.reshape((-1, *act_shape)), _lo, _hi), None

            probe = np.random.default_rng(12345).normal(0, 3, (200, 1, obs_shape[0]))
            for o in probe.astype(np.float32):
                if not np.array_equal(model.predict(o, deterministic=True)[0], fast(o)[0]):
                    raise RuntimeError("fast predict differs from SB3 predict")
            model.predict = fast
            model._fast_predict_ok = True

    def _wrap_submit_for_logging(self) -> None:
        eng = self.engine
        orig = eng._submit_order

        def logged(agent_id, action, order_type, qty, price, tif_ticks, pubs,
                   client_order_id=None):
            self._log(("O", agent_id, action, order_type, int(qty),
                       _ticks(price, eng.tick), int(tif_ticks)))
            ack = orig(agent_id, action, order_type, qty, price, tif_ticks,
                       pubs, client_order_id)
            self._log(("A", ack["order_id"], ack["status"],
                       ack["executed_qty"], ack["resting_qty"]))
            return ack

        eng._submit_order = logged

    def _log(self, event: tuple) -> None:
        if self._hasher is not None:
            self._hasher.update(repr(event).encode())

    # -- event plumbing --------------------------------------------------- #
    def _absorb(self, pubs, between_ticks: bool) -> None:
        eng = self.engine
        # A print belongs to the bar that is open when it happens: during a
        # physics tick that is bar tick_count, between ticks it is the next.
        bar_idx = eng.tick_count + (1 if between_ticks else 0)
        for topic, payload in pubs:
            if topic == "trade":
                self.trade_rows.append((
                    bar_idx, _ticks(payload["price"], eng.tick), payload["qty"],
                    1 if payload["aggressor"] == "BUY" else -1,
                    payload["buyer"], payload["seller"]))
                self._log(("T", payload["seq"], _ticks(payload["price"], eng.tick),
                           payload["qty"], payload["buyer"], payload["seller"],
                           payload["aggressor"]))
            elif topic == "event":
                self.orch._on_event(payload)
            elif topic == "tick":
                self.last_payload = payload

    # -- stepping --------------------------------------------------------- #
    def physics(self) -> dict:
        """Advance one physics tick (= one sim-minute) and record the bar."""
        eng = self.engine
        pubs = eng.physics_tick()
        self._absorb(pubs, between_ticks=False)
        p = self.last_payload
        bar = p["bar"]
        bb, ba = p["best_bid"], p["best_ask"]
        self.bar_rows.append((
            p["day_count"], p["market_minute"], eng.tick,
            bar["o"], bar["h"], bar["l"], bar["c"], int(bar["v"]),
            np.nan if bb is None else bb, np.nan if ba is None else ba))
        self._log(("B", p["day_count"], p["market_minute"],
                   _ticks(bar["o"], eng.tick), _ticks(bar["c"], eng.tick),
                   int(bar["v"]), _ticks(bb, eng.tick), _ticks(ba, eng.tick)))
        return p

    def agents_act(self, tick: dict) -> None:
        if self.null_strategy:
            self._null_ticks_seen += 1   # receives the tick, sends nothing
        self._loop.run_until_complete(self.orch._on_tick(tick))

    def step(self) -> dict:
        p = self.physics()
        self.agents_act(p)
        return p

    # -- book access for tests -------------------------------------------- #
    def mid(self) -> Optional[float]:
        bb, ba = self.engine.book.best_bid(), self.engine.book.best_ask()
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    def visible_qty(self, side: str) -> int:
        bids, asks = self.engine.book.depth(10 ** 6)
        return int(sum(q for _, q in (asks if side == "BUY" else bids)))

    def market_order(self, agent_id: str, side: str, qty: int) -> Fill:
        available = self.visible_qty(side)
        ack, pubs = self.engine.handle_message(
            {"msg": "ORDER", "agent_id": agent_id, "action": side,
             "type": "MARKET", "qty": int(qty)})
        self._absorb(pubs, between_ticks=True)
        return Fill(int(ack.get("executed_qty", 0)),
                    float(ack.get("average_price", 0.0)), available)

    def depth_by_tick_distance(self, L: int) -> Tuple[np.ndarray, Optional[int]]:
        """Volume at d = 1..L ticks from the best quote on each side.

        d = 1 is the best quote itself (spec T5: "distance d = 1..L ticks
        from the best quote"; we index the touch as d = 1).
        """
        eng = self.engine
        bids, asks = eng.book.depth(10 ** 6)
        out = np.zeros((2, L))
        for s, levels in ((0, bids), (1, asks)):
            if not levels:
                continue
            best = _ticks(levels[0][0], eng.tick)
            for px, q in levels:
                d = abs(_ticks(px, eng.tick) - best) + 1
                if d > L:
                    break
                out[s, d - 1] += q
        spread = None
        if bids and asks:
            spread = _ticks(asks[0][0], eng.tick) - _ticks(bids[0][0], eng.tick)
        return out, spread

    def check_invariants(self) -> Dict[str, int]:
        """Book invariants of spec T5 step 5 / §14.2 step 2 on the live book."""
        eng = self.engine
        v = {"crossed": 0, "nonpositive_size": 0, "off_grid": 0,
             "duplicate_ids": 0}
        bb, ba = eng.book.best_bid(), eng.book.best_ask()
        if bb is not None and ba is not None and not bb < ba:
            v["crossed"] += 1
        seen = set()
        for book in (eng.book._bids, eng.book._asks):
            for key, level in book.items():
                px = abs(key)            # bids are keyed by -price
                for o in level:
                    if o["qty"] <= 0:
                        v["nonpositive_size"] += 1
                    if abs(px / eng.tick - round(px / eng.tick)) > 1e-6:
                        v["off_grid"] += 1
                    if o["order_id"] in seen:
                        v["duplicate_ids"] += 1
                    seen.add(o["order_id"])
        return v

    # -- snapshot / restore (harness-level; spec T7 setup step 2) --------- #
    def snapshot(self) -> dict:
        # Agents hold the shared TickClock: the memo keeps that reference
        # pointing at the live clock instead of a frozen copy.
        return copy.deepcopy({
            "engine": self.engine,
            "agents": self.orch.agents,
            "orch": (self.orch._rng.getstate(), self.orch._tick_count,
                     self.orch._last_news_tick),
            "py_rng": random.getstate(),
            "np_rng": np.random.get_state(),
            "clock": self.clock.now,
            "last_payload": self.last_payload,
        }, {id(self.clock): self.clock})

    def restore(self, snap: dict) -> None:
        s = copy.deepcopy(snap, {id(self.clock): self.clock})
        self.engine = s["engine"]
        self.orch.agents = s["agents"]
        rng_state, self.orch._tick_count, self.orch._last_news_tick = s["orch"]
        self.orch._rng.setstate(rng_state)
        random.setstate(s["py_rng"])
        np.random.set_state(s["np_rng"])
        self.clock.now = s["clock"]
        self.last_payload = s["last_payload"]

    # -- output ----------------------------------------------------------- #
    def bars_array(self) -> Dict[str, np.ndarray]:
        cols = ("day", "minute", "tick", "o", "h", "l", "c", "v",
                "best_bid", "best_ask")
        a = np.array(self.bar_rows, dtype=float) if self.bar_rows else \
            np.zeros((0, len(cols)))
        out = {c: a[:, i] for i, c in enumerate(cols)}
        out["day"] = out["day"].astype(int)
        out["minute"] = out["minute"].astype(int)
        out["v"] = out["v"].astype(np.int64)
        out["mid"] = (out["best_bid"] + out["best_ask"]) / 2.0
        return out

    def trades_array(self) -> Dict[str, np.ndarray]:
        if not self.trade_rows:
            return {k: np.zeros(0) for k in ("bar", "price_ticks", "qty", "sign")}
        bar, px, qty, sign, buyer, seller = zip(*self.trade_rows)
        return {"bar": np.asarray(bar), "price_ticks": np.asarray(px),
                "qty": np.asarray(qty), "sign": np.asarray(sign),
                "buyer": np.asarray(buyer), "seller": np.asarray(seller)}

    def event_hash(self) -> Optional[str]:
        return None if self._hasher is None else self._hasher.hexdigest()


def simulate(seed: int, n_days: int, start_price: float,
             counts: Optional[Dict[str, int]] = None,
             book_snap_prob: float = 0.0, book_L: int = 20,
             hash_events: bool = False, null_strategy: bool = False,
             invariant_every: int = 1) -> RunOutput:
    """Run ``n_days`` x 375 sim-minutes and return bars, trades and book data.

    Book snapshots are taken after the agents have acted (the state an order
    arriving mid-minute would meet), at random ticks with probability
    ``book_snap_prob``, using a dedicated RNG so sampling never perturbs the
    simulation streams.
    """
    sim = Simulation(seed, start_price, counts=counts, hash_events=hash_events,
                     null_strategy=null_strategy)
    snap_rng = np.random.default_rng([seed, 7])
    snaps, spreads = [], []
    inv = {"crossed": 0, "nonpositive_size": 0, "off_grid": 0,
           "duplicate_ids": 0, "checked": 0}
    n_ticks = n_days * MatchingEngine.SESSION_MINUTES
    for t in range(n_ticks):
        sim.step()
        if invariant_every and t % invariant_every == 0:
            for k, val in sim.check_invariants().items():
                inv[k] += val
            inv["checked"] += 1
        if book_snap_prob and snap_rng.random() < book_snap_prob:
            q, spr = sim.depth_by_tick_distance(book_L)
            snaps.append(q)
            spreads.append(np.nan if spr is None else spr)
    return RunOutput(
        seed=seed, bars=sim.bars_array(), trades=sim.trades_array(),
        book_snaps=np.array(snaps) if snaps else None,
        book_spreads_ticks=np.array(spreads) if spreads else None,
        invariant_violations=inv, event_hash=sim.event_hash(),
        meta={"n_days": n_days, "start_price": start_price,
              "counts": sim.counts,
              "ledger_types": sorted({l["type"] for l in
                                      sim.engine.agent_ledger.values()})})
