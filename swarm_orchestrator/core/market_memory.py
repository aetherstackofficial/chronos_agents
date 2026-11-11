"""MarketMemory: a LONG, bounded memory of the market (up to one year of sessions).

``MarketContext`` remembers only ~1000 minute bars / 5 days. RL agents also need
"where are we in the year, which price levels matter, is volume/volatility
unusual". This class keeps that, and is fed exactly like ``MarketContext``:
once per engine tick via ``update(tick)``, or rebuilt from closed bars via
``seed(bars, run_id)`` / ``add_bar(...)``.

Rules (copied from MarketContext):

* **Pure Python stdlib.** No numpy, zmq or engine imports, no randomness.
* **Only reports the past.** State at bar *t* depends only on bars up to *t*.
  A daily swing pivot is used only after 3 later completed sessions exist.
* **Bounded.** 250 daily bars, 20 sessions of minute bars (7500), one volume
  profile dict per completed day, 20 sessions of minute-of-day volume.
* **Deterministic.** The same bars in give the same ``features()`` out.
* **update() semantics:** paused ticks are skipped, a bar is added once per
  ``(day_count, bar["minute"])``, a changed ``run_id`` starts a fresh memory,
  a session closes when ``day_count`` changes.

Cost model: ``add_bar`` is O(1) amortised. Everything heavy (price levels,
medians, year high/low) is computed lazily, at most once per session close, the
first time ``levels()`` / ``features()`` / ``summary()`` is called afterwards.
Because those results depend only on completed sessions they are identical to
recomputing at every close.

Minute numbering: the engine stamps bars 1..375 (bar-close minute) while
historical files use 0..374 (minutes since 09:15). Minute-of-day arrays are
indexed directly by ``minute`` (size 376) so both conventions work.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from collections import deque
from itertools import accumulate
from typing import Deque, Dict, List, Optional, Tuple

_REQUIRED = ("o", "h", "l", "c")
_LOG_BIN = math.log(1.002)          # volume-profile bin width: 0.2 %
_ATR_N = 14
_EMA50_A = 2.0 / 51.0
_EMA200_A = 2.0 / 201.0
_EMA20_A = 2.0 / 21.0
_EMA60_A = 2.0 / 61.0
_LOG1P_250 = math.log1p(250)
_PIVOT_K = 3                        # days on each side of a daily swing pivot
_MAX_NODES = 8                      # volume-profile peaks kept
_MERGE_ATR = 0.5                    # merge levels closer than this many ATR
_STRENGTH_ATR = 0.25                # volume window around a level (+-ATR)
_TOUCH_ATR = 0.1                    # touch tolerance (ATR)
_DIST_CLIP = 10.0


def _bin(price: float) -> int:
    """Volume-profile bin of a price (log-spaced, 0.2 % wide)."""
    return math.floor(math.log(price) / _LOG_BIN)


def _median(xs) -> float:
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _feature_names() -> Tuple[str, ...]:
    names = ["hist_avail",
             "ret_1m", "ret_5m", "ret_15m", "ret_30m", "ret_60m", "ret_1d",
             "ema20_dist", "ema60_dist", "gap", "from_open",
             "range_pos_today", "range_pos_5d",
             "prev_high_dist", "prev_low_dist", "prev_close_dist",
             "rv30_ratio", "rv_today_ratio", "jump_10m",
             "vol_same_minute", "vol_today_vs_expected",
             "ret_5d", "ret_20d", "ret_60d", "ret_120d",
             "year_range_pos", "year_high_dist_atr", "year_low_dist_atr",
             "ema50d_dist_atr", "ema200d_dist_atr", "ema50_vs_200_atr",
             "stock_drawdown", "atr_regime", "volume_regime"]
    for side in ("up", "dn"):
        for i in (1, 2, 3):
            for part in ("dist", "strength", "touches", "age"):
                names.append(f"lvl_{side}{i}_{part}")
    return tuple(names)


class MarketMemory:
    """Long rolling market memory fed one tick at a time. See the module docstring."""

    SESSION_BARS = 375
    FEATURE_NAMES: Tuple[str, ...] = _feature_names()

    def __init__(self, max_days: int = 250, intraday_days: int = 20) -> None:
        if max_days < 1 or intraday_days < 1:
            raise ValueError("max_days and intraday_days must be >= 1")
        self._max_days = int(max_days)
        self._intraday_days = int(intraday_days)
        self._max_minute_bars = self._intraday_days * self.SESSION_BARS
        self.reset()

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Forget everything (a new run starts over from bar 1)."""
        self._run_id: Optional[str] = None
        self.bars_seen = 0
        self._last_day: Optional[int] = None
        self._last_minute: Optional[int] = None
        self._last_v = 0.0
        # minute bars (o, h, l, c, v) of the last intraday_days sessions
        self._bars: Deque[Tuple[float, float, float, float, float]] = deque(
            maxlen=self._max_minute_bars)
        self._rets: Deque[float] = deque(maxlen=30)   # intraday 1-bar log returns
        self._e20 = 0.0
        self._e60 = 0.0
        # today (partial session)
        self._day: Optional[int] = None
        self._t_o = self._t_h = self._t_l = self._t_c = 0.0
        self._t_v = 0.0
        self._t_n = 0
        self._tprof: Dict[int, float] = {}
        self._tvol: List[float] = [0.0] * (self.SESSION_BARS + 1)
        self._r_n = 0
        self._r_s = 0.0
        self._r_q = 0.0
        # completed sessions
        self._days: Deque[Tuple[float, float, float, float, float]] = deque(
            maxlen=self._max_days)
        self._days_total = 0
        self._trs: Deque[float] = deque(maxlen=self._max_days)
        self._tr_n = 0
        self._tr_sum = 0.0
        self._atr: Optional[float] = None
        self._e50 = 0.0
        self._e200 = 0.0
        # rolling volume profile over completed sessions
        self._prof: Dict[int, float] = {}
        self._bin_days: Dict[int, int] = {}
        self._day_profs: Deque[Dict[int, float]] = deque()
        # last intraday_days sessions
        self._ivols: Deque[float] = deque(maxlen=self._intraday_days)
        self._imin: Deque[Tuple[List[float], List[float]]] = deque()
        self._sum_vol = [0.0] * (self.SESSION_BARS + 1)
        self._sum_cum = [0.0] * (self.SESSION_BARS + 1)
        # confirmed daily swing pivots: (absolute day index, kind, price)
        self._pivots: Deque[Tuple[int, str, float]] = deque()
        # lazily computed per-session-close aggregates
        self._dirty = True
        self._agg: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # feeding
    # ------------------------------------------------------------------ #
    def update(self, tick: dict) -> bool:
        """Add the tick's closed bar; return True if it was new.

        Same rules as ``MarketContext.update``: paused ticks are skipped, a bar
        is added once per ``(day_count, bar["minute"])`` and a changed
        ``run_id`` starts a fresh memory.
        """
        if not isinstance(tick, dict) or tick.get("paused"):
            return False
        bar = tick.get("bar")
        if not isinstance(bar, dict):
            return False
        run_id = tick.get("run_id")
        if run_id is not None and run_id != self._run_id:
            if self._run_id is not None:
                self.reset()
            self._run_id = run_id
        try:
            o, h, l, c = (float(bar[k]) for k in _REQUIRED)
            v = float(bar.get("v") or 0)
            minute = int(bar.get("minute", tick.get("market_minute")))
            day = int(tick.get("day_count", 1))
        except (KeyError, TypeError, ValueError):
            return False
        return self._add(day, minute, o, h, l, c, v)

    def seed(self, bars: List[dict], run_id: Optional[str] = None) -> int:
        """Rebuild the memory from closed bars (oldest first); return how many were added.

        Each bar is ``{day_count, minute, o, h, l, c, v}``. Replaces whatever was
        seen before. Pass the engine's ``run_id`` so the next live ticks continue
        this memory; a bar that the first live tick repeats is ignored.
        """
        self.reset()
        self._run_id = run_id
        added = 0
        for b in bars or []:
            try:
                o, h, l, c = (float(b[k]) for k in _REQUIRED)
                v = float(b.get("v") or 0)
                minute, day = int(b["minute"]), int(b["day_count"])
            except (KeyError, TypeError, ValueError):
                continue
            if self._add(day, minute, o, h, l, c, v):
                added += 1
        return added

    def add_bar(self, day: int, minute: int, o: float, h: float, l: float,
                c: float, v: float) -> bool:
        """Fast path for training pre-roll; same as one ``update`` of that bar.

        Expects plain numbers (no conversion); returns False for a repeated
        ``(day, minute)`` or a non-finite / non-positive price.
        """
        return self._add(day, minute, o, h, l, c, v)

    def _add(self, day, minute, o, h, l, c, v) -> bool:
        if day == self._last_day and minute == self._last_minute:
            return False
        s = o + h + l + c + v
        if s - s != 0.0 or o <= 0.0 or h <= 0.0 or l <= 0.0 or c <= 0.0:
            return False                       # NaN / inf / non-positive price
        if v < 0.0:
            v = 0.0
        self._last_day = day
        self._last_minute = minute

        if day != self._day:
            if self._day is not None:
                self._close_day()
            self._day = day
            self._t_o = o
            self._t_h = h
            self._t_l = l
            self._t_v = 0.0
            self._t_n = 0
            self._tprof = {}
            self._tvol = [0.0] * (self.SESSION_BARS + 1)
            self._r_n = 0
            self._r_s = 0.0
            self._r_q = 0.0
        else:
            if h > self._t_h:
                self._t_h = h
            if l < self._t_l:
                self._t_l = l
            r = math.log(c / self._t_c)
            self._rets.append(r)
            self._r_n += 1
            self._r_s += r
            self._r_q += r * r
        self._t_c = c
        self._t_v += v
        self._t_n += 1
        if v > 0.0:
            b = math.floor(math.log((h + l + c) / 3.0) / _LOG_BIN)
            tp = self._tprof
            tp[b] = tp.get(b, 0.0) + v
            if 0 <= minute <= self.SESSION_BARS:
                self._tvol[minute] += v
        self._bars.append((o, h, l, c, v))
        self._last_v = v
        if self.bars_seen:
            self._e20 += _EMA20_A * (c - self._e20)
            self._e60 += _EMA60_A * (c - self._e60)
        else:
            self._e20 = self._e60 = c
        self.bars_seen += 1
        return True

    def _close_day(self) -> None:
        """Move today into the history (called when day_count changes)."""
        o, h, l, c, v = self._t_o, self._t_h, self._t_l, self._t_c, self._t_v
        prev_c = self._days[-1][3] if self._days else None
        self._days.append((o, h, l, c, v))
        self._days_total += 1
        n_total = self._days_total

        # true range / Wilder ATR
        tr = h - l if prev_c is None else max(h - l, abs(h - prev_c), abs(l - prev_c))
        self._trs.append(tr)
        self._tr_n += 1
        if self._tr_n <= _ATR_N:
            self._tr_sum += tr
            self._atr = self._tr_sum / self._tr_n
        else:
            self._atr = (self._atr * (_ATR_N - 1) + tr) / _ATR_N

        # daily EMAs (seeded with the first close)
        if n_total == 1:
            self._e50 = self._e200 = c
        else:
            self._e50 += _EMA50_A * (c - self._e50)
            self._e200 += _EMA200_A * (c - self._e200)

        # rolling volume profile: subtract the evicted day exactly
        if len(self._day_profs) >= self._max_days:
            old = self._day_profs.popleft()
            prof, bd = self._prof, self._bin_days
            for b, vol in old.items():
                left = bd[b] - 1
                if left <= 0:
                    del bd[b]
                    del prof[b]
                else:
                    bd[b] = left
                    prof[b] -= vol
        day_prof = self._tprof
        prof, bd = self._prof, self._bin_days
        for b, vol in day_prof.items():
            prof[b] = prof.get(b, 0.0) + vol
            bd[b] = bd.get(b, 0) + 1
        self._day_profs.append(day_prof)

        # intraday volatility of the session
        if self._r_n >= 2:
            mean = self._r_s / self._r_n
            self._ivols.append(math.sqrt(max(self._r_q / self._r_n - mean * mean, 0.0)))

        # minute-of-day volume (per minute and cumulative) of the last N sessions
        vol_arr = self._tvol
        cum_arr = list(accumulate(vol_arr))
        if len(self._imin) >= self._intraday_days:
            ov, oc = self._imin.popleft()
            self._sum_vol = [a - b for a, b in zip(self._sum_vol, ov)]
            self._sum_cum = [a - b for a, b in zip(self._sum_cum, oc)]
        self._sum_vol = [a + b for a, b in zip(self._sum_vol, vol_arr)]
        self._sum_cum = [a + b for a, b in zip(self._sum_cum, cum_arr)]
        self._imin.append((vol_arr, cum_arr))

        # daily swing pivot, confirmed once _PIVOT_K later sessions exist
        days = self._days
        k = _PIVOT_K
        if len(days) >= 2 * k + 1:
            i = len(days) - 1 - k
            ph, pl = days[i][1], days[i][2]
            is_hi = is_lo = True
            for j in range(i - k, i + k + 1):
                if j == i:
                    continue
                if days[j][1] >= ph:
                    is_hi = False
                if days[j][2] <= pl:
                    is_lo = False
            abs_i = n_total - 1 - k
            if is_hi:
                self._pivots.append((abs_i, "swing_high", ph))
            if is_lo:
                self._pivots.append((abs_i, "swing_low", pl))
        first_abs = n_total - len(days)
        while self._pivots and self._pivots[0][0] < first_abs:
            self._pivots.popleft()

        self._dirty = True

    # ------------------------------------------------------------------ #
    # simple accessors
    # ------------------------------------------------------------------ #
    @property
    def sessions(self) -> int:
        """Completed sessions in memory (0..max_days)."""
        return len(self._days)

    @property
    def last_price(self) -> Optional[float]:
        return self._t_c if self._t_n else None

    @property
    def atr(self) -> Optional[float]:
        """Wilder ATR(14) over completed days; None with no completed day."""
        return self._atr

    def avg_daily_volume(self, n: int = 20) -> Optional[float]:
        """Mean total volume of the last ``n`` COMPLETED sessions (today excluded).

        None while no session has completed (cold start). Read-only.
        """
        days = self._days
        k = min(max(int(n), 1), len(days))
        if not k:
            return None
        total = 0.0
        for i in range(-k, 0):
            total += days[i][4]
        return total / k

    def volume_profile(self, include_today: bool = True) -> Dict[int, float]:
        """Rolling volume profile {bin: volume}: last max_days sessions (+ today)."""
        out = dict(self._prof)
        if include_today:
            for b, v in self._tprof.items():
                out[b] = out.get(b, 0.0) + v
        return out

    # ------------------------------------------------------------------ #
    # lazily computed aggregates (once per session close)
    # ------------------------------------------------------------------ #
    def _ensure(self) -> dict:
        if not self._dirty and self._agg is not None:
            return self._agg
        days = self._days
        agg: dict = {"year_hi": None, "year_lo": None, "max_close": None,
                     "med_tr": 0.0, "med_vol": 0.0, "rv_norm": 0.0,
                     "lv_prices": [], "lv": []}
        if days:
            agg["year_hi"] = max(d[1] for d in days)
            agg["year_lo"] = min(d[2] for d in days)
            agg["max_close"] = max(d[3] for d in days)
            agg["med_tr"] = _median(self._trs)
            agg["med_vol"] = _median(d[4] for d in days)
        norm = [x for x in self._ivols if x > 0.0]
        if norm:
            agg["rv_norm"] = sum(norm) / len(norm)
        atr = self._atr
        if atr is not None and atr > 0.0 and self._prof:
            agg["lv_prices"], agg["lv"] = self._build_levels(atr)
        self._agg = agg
        self._dirty = False
        return agg

    def _profile_volume(self, lo_price: float, hi_price: float) -> float:
        prof = self._prof
        lo_b = _bin(lo_price) if lo_price > 0.0 else min(prof)
        hi_b = _bin(hi_price)
        if hi_b - lo_b > len(prof):          # sparse profile: iterate keys instead
            return sum(v for b, v in prof.items() if lo_b <= b <= hi_b)
        get = prof.get
        return sum(get(b, 0.0) for b in range(lo_b, hi_b + 1))

    def _volume_nodes(self) -> List[Tuple[float, float]]:
        """Top peaks of the +-2-bin smoothed profile: [(price, smoothed volume)]."""
        prof = self._prof
        lo, hi = min(prof) - 4, max(prof) + 4
        arr = [prof.get(b, 0.0) for b in range(lo, hi + 1)]
        n = len(arr)
        sm = [0.0] * n
        for i in range(2, n - 2):
            sm[i] = arr[i - 2] + arr[i - 1] + arr[i] + arr[i + 1] + arr[i + 2]
        peaks = [(sm[i], i) for i in range(2, n - 2)
                 if sm[i] > 0.0 and sm[i] > sm[i - 1] and sm[i] >= sm[i + 1]]
        peaks.sort(key=lambda p: (-p[0], p[1]))
        return [(math.exp((lo + i + 0.5) * _LOG_BIN), s) for s, i in peaks[:_MAX_NODES]]

    def _build_levels(self, atr: float):
        """Candidate levels -> merge -> strength / touches. Completed sessions only."""
        cands: List[Tuple[float, str]] = [(p, kind) for _, kind, p in self._pivots]
        cands += [(p, "volume_node") for p, _ in self._volume_nodes()]
        if not cands:
            return [], []
        w = _STRENGTH_ATR * atr
        evid = [self._profile_volume(p - w, p + w) for p, _ in cands]
        order = sorted(range(len(cands)), key=lambda i: (cands[i][0], cands[i][1]))
        clusters = []        # [price, anchor_evidence, total_evidence, {kinds}]
        for i in order:
            p, kind = cands[i]
            e = evid[i]
            if clusters and p - clusters[-1][0] < _MERGE_ATR * atr:
                cl = clusters[-1]
                if e > cl[1]:
                    cl[0], cl[1] = p, e
                cl[2] += e
                cl[3].add(kind)
            else:
                clusters.append([p, e, e, {kind}])
        top = max(cl[2] for cl in clusters)
        dl = [d[2] for d in self._days]
        dh = [d[1] for d in self._days]
        n = len(dl)
        tol = _TOUCH_ATR * atr
        prices: List[float] = []
        levels: List[tuple] = []
        for price, _, ev, kinds in clusters:
            touches = 0
            last = None
            for j in range(n):
                if dl[j] - tol <= price <= dh[j] + tol:
                    touches += 1
                    last = j
            age = None if last is None else n - last
            strength = ev / top if top > 0.0 else 0.0
            prices.append(price)
            levels.append((price, "+".join(sorted(kinds)), strength, touches, age))
        return prices, levels

    # ------------------------------------------------------------------ #
    # levels
    # ------------------------------------------------------------------ #
    def _level_dict(self, lv: tuple, price: float, atr: float) -> dict:
        p, kind, strength, touches, age = lv
        if self._t_n and self._t_l - _TOUCH_ATR * atr <= p <= self._t_h + _TOUCH_ATR * atr:
            touches += 1
            age = 0
        return {"price": p, "kind": kind, "strength": strength, "touches": touches,
                "sessions_since_touch": age, "dist_atr": (p - price) / atr}

    def levels(self, n_above: int = 3, n_below: int = 3) -> dict:
        """Nearest price levels around the last close, nearest first.

        ``{"above": [...], "below": [...]}``; each level has ``price, kind,
        strength, touches, sessions_since_touch, dist_atr`` (signed
        ``(level - price) / ATR``). Empty until one session has completed.
        """
        price = self.last_price
        agg = self._ensure()
        atr = self._atr
        out: dict = {"above": [], "below": []}
        if price is None or not agg["lv"] or not atr or atr <= 0.0:
            return out
        prices, lv = agg["lv_prices"], agg["lv"]
        idx = bisect_right(prices, price)
        for i in range(idx, min(idx + max(int(n_above), 0), len(lv))):
            out["above"].append(self._level_dict(lv[i], price, atr))
        for i in range(idx - 1, max(idx - 1 - max(int(n_below), 0), -1), -1):
            out["below"].append(self._level_dict(lv[i], price, atr))
        return out

    # ------------------------------------------------------------------ #
    # features
    # ------------------------------------------------------------------ #
    def features(self) -> Dict[str, float]:
        """Fixed-order dict of unitless features (see ``FEATURE_NAMES``).

        All values are finite floats. A value whose window is not yet filled is
        0.0 (a missing price level uses the explicit defaults of the spec).
        """
        f = dict.fromkeys(self.FEATURE_NAMES, 0.0)
        for side, sign in (("up", _DIST_CLIP), ("dn", -_DIST_CLIP)):
            for i in (1, 2, 3):
                f[f"lvl_{side}{i}_dist"] = sign
                f[f"lvl_{side}{i}_age"] = 1.0
        price = self.last_price
        if price is None:
            return f
        agg = self._ensure()
        days = self._days
        n_days = len(days)
        atr = self._atr if (self._atr and self._atr > 0.0) else None
        c = price
        f["hist_avail"] = math.log1p(n_days) / math.log1p(self._max_days)

        # ---- short-term (minute bars)
        bars = self._bars
        nb = len(bars)
        for name, k in (("ret_1m", 1), ("ret_5m", 5), ("ret_15m", 15),
                        ("ret_30m", 30), ("ret_60m", 60), ("ret_1d", self.SESSION_BARS)):
            if nb > k:
                f[name] = math.log(c / bars[-1 - k][3])
        if self.bars_seen >= 20:
            f["ema20_dist"] = (c - self._e20) / c
        if self.bars_seen >= 60:
            f["ema60_dist"] = (c - self._e60) / c
        t_o, t_h, t_l = self._t_o, self._t_h, self._t_l
        f["from_open"] = math.log(c / t_o)
        f["range_pos_today"] = (c - t_l) / (t_h - t_l) if t_h > t_l else 0.5
        if n_days:
            pd = days[-1]
            f["gap"] = math.log(t_o / pd[3])
            f["prev_high_dist"] = (pd[1] - c) / c
            f["prev_low_dist"] = (pd[2] - c) / c
            f["prev_close_dist"] = (pd[3] - c) / c
        if n_days >= 5:
            hi, lo = t_h, t_l
            for i in range(-5, 0):
                d = days[i]
                if d[1] > hi:
                    hi = d[1]
                if d[2] < lo:
                    lo = d[2]
            f["range_pos_5d"] = (c - lo) / (hi - lo) if hi > lo else 0.5

        norm = agg["rv_norm"]
        if norm > 0.0:
            rets = self._rets
            if len(rets) >= 30:
                m = sum(rets) / 30.0
                var = sum((x - m) * (x - m) for x in rets) / 30.0
                f["rv30_ratio"] = math.sqrt(var) / norm
            if len(rets) >= 10:
                last10 = list(rets)[-10:]
                f["jump_10m"] = max(last10, key=abs) / norm
            if self._r_n >= 2:
                mean = self._r_s / self._r_n
                f["rv_today_ratio"] = math.sqrt(max(self._r_q / self._r_n - mean * mean, 0.0)) / norm

        nd = len(self._imin)
        mi = self._last_minute
        have_min = nd > 0 and mi is not None and 0 <= mi <= self.SESSION_BARS
        share = 0.0
        if have_min:
            mean_v = self._sum_vol[mi] / nd
            if mean_v > 0.0:
                f["vol_same_minute"] = _clip(self._last_v / mean_v - 1.0, -1.0, 9.0)
            mean_c = self._sum_cum[mi] / nd
            if mean_c > 0.0:
                f["vol_today_vs_expected"] = _clip(self._t_v / mean_c - 1.0, -1.0, 9.0)
            full = self._sum_cum[self.SESSION_BARS]
            if full > 0.0:
                share = self._sum_cum[mi] / full

        # ---- long memory
        if n_days:
            for name, k in (("ret_5d", 5), ("ret_20d", 20), ("ret_60d", 60), ("ret_120d", 120)):
                if n_days >= k:
                    f[name] = math.log(c / days[-k][3])
            yh = max(agg["year_hi"], t_h)
            yl = min(agg["year_lo"], t_l)
            f["year_range_pos"] = (c - yl) / (yh - yl) if yh > yl else 0.5
            f["stock_drawdown"] = c / max(agg["max_close"], c) - 1.0
            med_vol = agg["med_vol"]
            if med_vol > 0.0:
                exp_full = self._t_v / share if share > 0.0 else self._t_v
                f["volume_regime"] = _clip(exp_full / med_vol - 1.0, -1.0, 9.0)
            if atr is not None:
                f["year_high_dist_atr"] = (yh - c) / atr
                f["year_low_dist_atr"] = (yl - c) / atr
                if n_days >= 50:
                    f["ema50d_dist_atr"] = (c - self._e50) / atr
                if n_days >= 200:
                    f["ema200d_dist_atr"] = (c - self._e200) / atr
                    f["ema50_vs_200_atr"] = (self._e50 - self._e200) / atr
                if n_days >= _ATR_N and agg["med_tr"] > 0.0:
                    f["atr_regime"] = atr / agg["med_tr"] - 1.0

        # ---- levels
        if atr is not None and agg["lv"]:
            lv = self.levels(3, 3)
            for side, key, sign in (("up", "above", _DIST_CLIP), ("dn", "below", -_DIST_CLIP)):
                for i, L in enumerate(lv[key][:3], 1):
                    f[f"lvl_{side}{i}_dist"] = _clip(L["dist_atr"], -_DIST_CLIP, _DIST_CLIP)
                    f[f"lvl_{side}{i}_strength"] = L["strength"]
                    f[f"lvl_{side}{i}_touches"] = math.log1p(L["touches"]) / _LOG1P_250
                    age = L["sessions_since_touch"]
                    f[f"lvl_{side}{i}_age"] = 1.0 if age is None else math.log1p(age) / _LOG1P_250
        return f

    # ------------------------------------------------------------------ #
    # human-readable snapshot
    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        """JSON-serialisable snapshot for a decision log."""
        agg = self._ensure()
        n_days = len(self._days)
        price = self.last_price
        feats = self.features()
        yh = yl = None
        if n_days and price is not None:
            yh = max(agg["year_hi"], self._t_h)
            yl = min(agg["year_lo"], self._t_l)
        return {
            "sessions": n_days,
            "bars_seen": self.bars_seen,
            "last_price": price,
            "atr": self._atr,
            "year_high": yh,
            "year_low": yl,
            "ema50": self._e50 if n_days >= 50 else None,
            "ema200": self._e200 if n_days >= 200 else None,
            "levels": self.levels(),
            "regimes": {
                "atr_regime": feats["atr_regime"],
                "volume_regime": feats["volume_regime"],
                "rv_today_ratio": feats["rv_today_ratio"],
                "year_range_pos": feats["year_range_pos"],
                "stock_drawdown": feats["stock_drawdown"],
            },
        }
