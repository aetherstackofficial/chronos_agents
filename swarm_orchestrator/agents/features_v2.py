"""Observation builder for the v2 RL agents (whale and market maker).

``agents/features.py`` (6 inputs) stays untouched for the old models. This module
builds the richer observation of the new agents from three sources:

* the engine ``tick`` (PROTOCOL.md 5.1),
* a ``MarketMemory`` (long market memory, ``core/market_memory.py``),
* an ``AgentBook`` (the agent's OWN state: position, P&L, drawdown ...).

Training and live must use exactly this code path, with ``AgentBook`` as the
single tracker of the agent's own state, so every input means the same thing in
both. ``spec_hash(role)`` fingerprints the input names plus the source of this
file and of ``core/market_memory.py``; a live model can refuse to load when the
hash it was trained with no longer matches.

Call order per tick (training and live)::

    memory.update(tick); book.on_tick(tick)
    obs = build_observation_v2(role, tick, memory, book)
    ... act ...; book.on_fill(...) for each fill; book.set_last_action(raw)

Tick fields used: last_price, vwap, volume_ma, step_volume, rsi, ofi,
bar{h,l,c}, market_minute, day_count, sector, unix_time (sim time, IST wall
clock stored as UTC; used for the weekday; if absent the weekday inputs are 0,0).
"""
from __future__ import annotations

import hashlib
import math
import os
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from core.market_memory import MarketMemory
from agents.sectors import LIVE_ALIASES, SECTORS

SESSION_MINUTES = 375
ADV_SESSIONS = 20
HOLD_CAP = 20
DD_WINDOW = 5
RISK_LO, RISK_HI = 0.03, 0.15
TWO_PI = 2.0 * math.pi

# --------------------------------------------------------------------------- spec
_MARKET = ("vwap_dist", "bar_range", "volume_ratio", "rsi", "order_flow")
_TIME = ("tod_sin", "tod_cos", "to_close", "dow_sin", "dow_cos")
_SECTOR = tuple(f"sector_{s}" for s in SECTORS)
_STATE = ("inventory", "open_pnl", "drawdown", "holding", "pnl_today", "exposure")
_N_ACTIONS = {"whale": 3, "mm": 4}
_WHALE_ONLY = ("cap_vs_adv", "traded_vs_adv", "risk_appetite")

_MM_MEM = ("hist_avail",
           "ret_1m", "ret_5m", "ret_15m", "ret_30m", "ret_60m", "ret_1d",
           "ema20_dist", "ema60_dist", "gap", "from_open",
           "range_pos_today", "range_pos_5d",
           "prev_high_dist", "prev_low_dist", "prev_close_dist",
           "rv30_ratio", "rv_today_ratio", "jump_10m",
           "vol_same_minute", "vol_today_vs_expected",
           "atr_regime", "volume_regime",
           "lvl_up1_dist", "lvl_up1_strength", "lvl_up1_touches", "lvl_up1_age",
           "lvl_dn1_dist", "lvl_dn1_strength", "lvl_dn1_touches", "lvl_dn1_age")

_MEM_KEYS: Dict[str, Tuple[str, ...]] = {
    "whale": tuple(MarketMemory.FEATURE_NAMES),
    "mm": _MM_MEM,
}


def _actions(role: str) -> Tuple[str, ...]:
    return tuple(f"last_action_{i}" for i in range(_N_ACTIONS[role]))


OBS_SPEC: Dict[str, Tuple[str, ...]] = {
    "whale": (_MARKET + tuple("mem_" + k for k in _MEM_KEYS["whale"]) + _TIME + _SECTOR
              + _STATE + _actions("whale") + _WHALE_ONLY),
    "mm": (_MARKET + tuple("mem_" + k for k in _MEM_KEYS["mm"]) + _TIME + _SECTOR
           + _STATE + _actions("mm")),
}

for _r, _names in OBS_SPEC.items():
    assert len(_names) == len(set(_names)), f"duplicate input names in {_r}"
_missing = set(_MM_MEM) - set(MarketMemory.FEATURE_NAMES)
assert not _missing, f"MarketMemory no longer has {sorted(_missing)}"

# Unbounded MarketMemory ratios, clipped for the policy.
_CLIPS = {"rv30_ratio": (0.0, 10.0), "rv_today_ratio": (0.0, 10.0),
          "jump_10m": (-10.0, 10.0)}
_MEM_CLIP = {k: _CLIPS.get(k) for k in MarketMemory.FEATURE_NAMES}


# --------------------------------------------------------------------------- sector
_SECTOR_VOCAB = {s.lower(): s for s in SECTORS}
_SECTOR_IDX = {s: i for i, s in enumerate(SECTORS)}
_UNKNOWN_IDX = _SECTOR_IDX["UNKNOWN"]
_sector_cache: Dict[object, int] = {}


def sector_index(sector) -> int:
    """Index into ``SECTORS`` for a live or vocabulary sector string (else UNKNOWN)."""
    try:
        return _sector_cache[sector]
    except (KeyError, TypeError):
        pass
    idx = _UNKNOWN_IDX
    if isinstance(sector, str):
        key = sector.strip()
        name = LIVE_ALIASES.get(key.upper()) or LIVE_ALIASES.get(key)
        if name is None:
            name = _SECTOR_VOCAB.get(key.lower(), "UNKNOWN")
        idx = _SECTOR_IDX.get(name, _UNKNOWN_IDX)
    try:
        if len(_sector_cache) < 4096:
            _sector_cache[sector] = idx
    except TypeError:
        pass
    return idx


# --------------------------------------------------------------------------- AgentBook
class AgentBook:
    """The agent's own state, tracked identically in training and live.

    Positions are signed share counts (long > 0). Realised P&L includes fees.
    Equity = capital + realised P&L (since the start) + unrealised P&L.
    """

    def __init__(self, capital: float, capacity: float, risk_tolerance: float = 0.08,
                 n_outputs: int = 3) -> None:
        if capital <= 0 or capacity <= 0:
            raise ValueError("capital and capacity must be > 0")
        self.capital = float(capital)
        self.capacity = float(capacity)
        self.risk_tolerance = float(risk_tolerance)
        self.n_outputs = int(n_outputs)
        self._pos = 0.0
        self._avg = 0.0
        self._realised = 0.0
        self._realised_today = 0.0
        self._fees = 0.0
        self._traded_today = 0.0
        self._holding = 0
        self._price: Optional[float] = None
        self._day: Optional[int] = None
        # equity peak of each of the last DD_WINDOW sessions; the last entry is today's
        self._peaks: Deque[float] = deque([self.capital], maxlen=DD_WINDOW)
        self._last_action: List[float] = [0.0] * self.n_outputs

    # ---- feeding
    def on_fill(self, side: str, qty: float, price: float, fee: float = 0.0) -> None:
        """Record an executed fill. ``side`` is BUY/SELL (any case)."""
        qty = float(qty)
        price = float(price)
        if not qty > 0.0 or not price > 0.0:
            return
        buy = str(side).strip().upper().startswith("B")
        signed = qty if buy else -qty
        pos = self._pos
        pnl = 0.0
        if pos == 0.0 or (pos > 0.0) == (signed > 0.0):
            new_pos = pos + signed
            self._avg = (abs(pos) * self._avg + qty * price) / abs(new_pos)
        else:
            closing = min(qty, abs(pos))
            pnl = closing * (price - self._avg) * (1.0 if pos > 0.0 else -1.0)
            new_pos = pos + signed
            if abs(new_pos) < 1e-9:
                new_pos = 0.0
                self._avg = 0.0
            elif (new_pos > 0.0) != (pos > 0.0):          # flipped through zero
                self._avg = price
        if new_pos == 0.0 or pos == 0.0 or (new_pos > 0.0) != (pos > 0.0):
            self._holding = 0                              # flat, or a fresh position
        self._pos = new_pos
        fee = float(fee)
        self._realised += pnl - fee
        self._realised_today += pnl - fee
        self._fees += fee
        self._traded_today += qty

    def charge(self, fee: float) -> None:
        """Book a cost that belongs to no fill (overnight delivery surcharge,
        closing cost at the end of an episode). Counts as realised P&L today."""
        fee = float(fee)
        if fee:
            self._realised -= fee
            self._realised_today -= fee
            self._fees += fee

    def on_tick(self, tick: dict) -> None:
        """Mark to market and do the new-session bookkeeping (call once per tick)."""
        price = float(tick["last_price"])
        if price > 0.0:
            self._price = price
        day = tick.get("day_count")
        if day is not None and day != self._day:
            if self._day is not None:                      # a new session starts
                self._realised_today = 0.0
                self._traded_today = 0.0
                if self._pos != 0.0:
                    self._holding += 1
                self._peaks.append(self.equity)            # evicts the oldest session
            self._day = day
        eq = self.equity
        if eq > self._peaks[-1]:
            self._peaks[-1] = eq

    def set_last_action(self, action) -> None:
        """Store the previous RAW policy output (padded / truncated to n_outputs)."""
        a = [float(x) for x in np.asarray(action, dtype=np.float64).ravel()[:self.n_outputs]]
        a += [0.0] * (self.n_outputs - len(a))
        self._last_action = [x if math.isfinite(x) else 0.0 for x in a]

    # ---- raw state
    @property
    def position(self) -> float:
        return self._pos

    @property
    def avg_price(self) -> float:
        return self._avg

    @property
    def realised_pnl(self) -> float:
        return self._realised

    @property
    def fees_paid(self) -> float:
        return self._fees

    @property
    def unrealised_pnl(self) -> float:
        if self._pos == 0.0 or self._price is None:
            return 0.0
        return self._pos * (self._price - self._avg)

    @property
    def equity(self) -> float:
        return self.capital + self._realised + self.unrealised_pnl

    @property
    def shares_traded_today(self) -> float:
        return self._traded_today

    @property
    def holding_sessions(self) -> int:
        return self._holding

    @property
    def last_action(self) -> List[float]:
        return list(self._last_action)

    # ---- model inputs
    @property
    def inventory(self) -> float:
        return self._pos / self.capacity

    @property
    def open_pnl(self) -> float:
        return self.unrealised_pnl / self.capital

    @property
    def drawdown(self) -> float:
        """Equity drop from the highest equity of the last 5 sessions, as a fraction (>= 0)."""
        peak = max(self._peaks)
        if peak <= 0.0:
            return 0.0
        return max((peak - self.equity) / peak, 0.0)

    @property
    def holding(self) -> float:
        return min(self._holding, HOLD_CAP) / HOLD_CAP

    @property
    def pnl_today(self) -> float:
        return self._realised_today / self.capital

    @property
    def exposure(self) -> float:
        if self._price is None:
            return 0.0
        return self._pos * self._price / self.capital

    @property
    def risk_appetite(self) -> float:
        return (self.risk_tolerance - RISK_LO) / (RISK_HI - RISK_LO)


# --------------------------------------------------------------------------- builder
def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def build_observation_v2(role: str, tick: dict, memory: MarketMemory, book: AgentBook,
                         feats: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Observation of ``role`` ("whale" or "mm") as float32 of shape (len(OBS_SPEC[role]),).

    ``feats`` may carry an already computed ``memory.features()`` (the expensive
    part) so a caller building both roles on one tick computes it only once.
    Market fields of the tick are required (KeyError if missing, like
    ``agents/features.py``); ``unix_time`` and ``sector`` degrade gracefully.
    """
    mem_keys = _MEM_KEYS[role]          # KeyError for an unknown role
    last = float(tick["last_price"])
    vwap = float(tick["vwap"])
    vol_ma = float(tick["volume_ma"])
    bar = tick["bar"]
    bc = float(bar["c"])

    out: List[float] = [
        (last - vwap) / vwap if vwap > 0.0 else 0.0,                      # vwap_dist
        (float(bar["h"]) - float(bar["l"])) / bc if bc > 0.0 else 0.0,    # bar_range
        float(tick["step_volume"]) / vol_ma if vol_ma > 0.0 else 1.0,     # volume_ratio
        float(tick["rsi"]) / 100.0 - 0.5,                                 # rsi
        float(tick["ofi"]),                                               # order_flow
    ]

    # ---- memory
    if feats is None:
        feats = memory.features()
    clips = _MEM_CLIP
    for k in mem_keys:
        x = feats[k]
        c = clips[k]
        out.append(_clip(x, c[0], c[1]) if c is not None else x)

    # ---- time
    minute = float(tick.get("market_minute", 0) or 0)
    ang = TWO_PI * minute / SESSION_MINUTES
    out.append(math.sin(ang))
    out.append(math.cos(ang))
    out.append((SESSION_MINUTES - minute) / SESSION_MINUTES)
    unix = tick.get("unix_time")
    if unix is None:
        out.append(0.0)
        out.append(0.0)
    else:
        wd = (int(unix) // 86400 + 3) % 7        # 1970-01-01 was a Thursday (3); Monday = 0
        a = TWO_PI * wd / 5.0                    # 5 trading days
        out.append(math.sin(a))
        out.append(math.cos(a))

    # ---- sector one-hot
    idx = sector_index(tick.get("sector"))
    oh = [0.0] * len(SECTORS)
    oh[idx] = 1.0
    out.extend(oh)

    # ---- own state
    out.append(book.inventory)
    out.append(book.open_pnl)
    out.append(book.drawdown)
    out.append(book.holding)
    out.append(book.pnl_today)
    out.append(book.exposure)
    n_act = _N_ACTIONS[role]
    la = book._last_action[:n_act]
    out.extend(la)
    if len(la) < n_act:                  # a book made with fewer outputs: pad with zeros
        out.extend([0.0] * (n_act - len(la)))

    if role == "whale":
        adv = memory.avg_daily_volume(ADV_SESSIONS)
        if adv is None:
            adv = SESSION_MINUTES * vol_ma       # cold start: no completed session yet
        if adv > 0.0:
            out.append(book.capacity / adv)
            out.append(book.shares_traded_today / adv)
        else:
            out.append(0.0)
            out.append(0.0)
        out.append(book.risk_appetite)

    return np.nan_to_num(np.asarray(out, dtype=np.float32), copy=False)


def obs_dict(role: str, vec) -> Dict[str, float]:
    """Name -> value view of an observation, for the decision log."""
    names = OBS_SPEC[role]
    if len(vec) != len(names):
        raise ValueError(f"{role}: observation has {len(vec)} values, spec has {len(names)}")
    return {n: float(v) for n, v in zip(names, vec)}


# --------------------------------------------------------------------------- hash
_hash_cache: Dict[str, str] = {}


def _source(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read().replace(b"\r\n", b"\n")      # same hash on Windows and Linux


def spec_hash(role: str) -> str:
    """sha256 over the input names of ``role`` + the source of this file and of
    ``core/market_memory.py``. A trained model stores it; live refuses to load
    the model when it differs."""
    h = _hash_cache.get(role)
    if h is None:
        import core.market_memory as _mm
        m = hashlib.sha256()
        m.update(role.encode())
        for name in OBS_SPEC[role]:
            m.update(b"\x00" + name.encode())
        m.update(b"\x01" + _source(os.path.abspath(__file__)))
        m.update(b"\x02" + _source(os.path.abspath(_mm.__file__)))
        h = _hash_cache[role] = m.hexdigest()
    return h
