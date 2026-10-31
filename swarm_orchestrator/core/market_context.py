"""MarketContext: a shared, read-only memory of what the market has done.

Agents and strategies are handed one tick at a time, so on their own they cannot
see yesterday's range, today's high/low or recent swing points. This object
remembers that, built only from the tick payload (PROTOCOL section 5.1):
``tick["bar"]`` (the just-closed minute bar: minute, o, h, l, c, v), plus
``day_count`` and ``paused``.

Rules it keeps:

* **Pure Python.** No model, zmq or engine imports and no randomness, so the
  swarm, the strategy SDK (which runs inside the sandbox image) and the training
  repo can all import this one file.
* **Never in the way.** It only reports what has already happened. It has no
  forecast or "direction" signal, and nothing in it can block or delay an order.
  Until ``ready`` is True, callers fall back to their old behaviour.
* **No look-ahead.** State at bar *t* depends only on bars up to *t*. A swing
  point is reported only after ``swing_k`` later bars exist.
* **Bounded memory** (deques) and **deterministic**: the same ticks in give the
  same state out.

Typical use::

    ctx = MarketContext()
    ctx.update(tick)                      # once per tick, before anyone acts
    if ctx.ready and ctx.prev_day:
        level = ctx.prev_day["h"]         # yesterday's high
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

DEFAULT_MIN_BARS = 120        # bars seen before ``ready`` flips to True
DEFAULT_SWING_K = 5           # bars on each side a pivot must out-reach
DEFAULT_DAYS = 5              # completed days kept in ``days``
DEFAULT_MAX_BARS = 1000       # intraday bars kept (ret(375) needs 376)
MAX_SWINGS = 64               # pivots kept per side

_REQUIRED = ("o", "h", "l", "c")


class MarketContext:
    """Rolling market history fed one tick at a time. See the module docstring."""

    def __init__(self, min_bars: int = DEFAULT_MIN_BARS, swing_k: int = DEFAULT_SWING_K,
                 days: int = DEFAULT_DAYS, max_bars: int = DEFAULT_MAX_BARS) -> None:
        if min_bars < 1 or swing_k < 1 or days < 1:
            raise ValueError("min_bars, swing_k and days must be >= 1")
        # Keep enough bars for the swing window and for 375-minute returns.
        self.min_bars = int(min_bars)
        self.swing_k = int(swing_k)
        self._max_days = int(days)
        self._max_bars = max(int(max_bars), 2 * self.swing_k + 2)
        self.reset()

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Forget everything (a new run starts over from bar 1)."""
        self.bars_seen = 0
        self._run_id: Optional[str] = None
        self._last_key: Optional[Tuple[int, int]] = None
        self._bars: Deque[Dict[str, float]] = deque(maxlen=self._max_bars)
        self._day_count: Optional[int] = None
        self._today: Optional[Dict[str, float]] = None
        self._days: Deque[Dict[str, float]] = deque(maxlen=self._max_days)
        self._swing_highs: Deque[float] = deque(maxlen=MAX_SWINGS)
        self._swing_lows: Deque[float] = deque(maxlen=MAX_SWINGS)
        # Intra-day 1-bar log-return moments (n, sum, sum of squares) for vol_ratio.
        self._ret_n = 0
        self._ret_sum = 0.0
        self._ret_sq = 0.0
        self._day_vols: Deque[float] = deque(maxlen=self._max_days)

    @property
    def ready(self) -> bool:
        """True once at least ``min_bars`` bars have been seen."""
        return self.bars_seen >= self.min_bars

    # ------------------------------------------------------------------ #
    # feeding
    # ------------------------------------------------------------------ #
    def update(self, tick: dict) -> bool:
        """Add the tick's closed bar; return True if it was new.

        Paused ticks are skipped (a paused engine re-sends its last bar every
        tick), and a bar is added only once per ``(day_count, bar["minute"])``.
        A changed ``run_id`` starts a fresh history, because a new run restarts
        its day and minute counters.
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
        if not all(math.isfinite(x) for x in (o, h, l, c, v)):
            return False
        return self._add(day, minute, o, h, l, c, v)

    def seed(self, bars: List[dict], run_id: Optional[str] = None) -> int:
        """Rebuild the history from closed bars (oldest first); return how many were added.

        For a worker or strategy that starts after the engine is already running
        (engine message FETCH_BARS). Each bar is ``{day_count, minute, o, h, l, c, v}``.
        Replaces whatever was seen before. Pass the engine's ``run_id`` so the next
        live ticks of that run continue this history instead of restarting it; a bar
        that the first live tick repeats is ignored, never counted twice.
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
            if all(math.isfinite(x) for x in (o, h, l, c, v)) and self._add(day, minute, o, h, l, c, v):
                added += 1
        return added

    def _add(self, day: int, minute: int, o: float, h: float, l: float,
             c: float, v: float) -> bool:
        key = (day, minute)
        if key == self._last_key:
            return False
        self._last_key = key

        if self._day_count is not None and day != self._day_count:
            self._close_day()
        if self._today is None:
            self._day_count = day
            self._today = {"o": o, "h": h, "l": l, "c": c, "v": 0.0}
        t = self._today
        t["h"], t["l"], t["c"] = max(t["h"], h), min(t["l"], l), c
        t["v"] += v

        if self._bars and self._bars[-1]["day"] == day and self._bars[-1]["c"] > 0 and c > 0:
            r = math.log(c / self._bars[-1]["c"])
            self._ret_n += 1
            self._ret_sum += r
            self._ret_sq += r * r
        self._bars.append({"day": day, "minute": minute, "o": o, "h": h, "l": l, "c": c, "v": v})
        self.bars_seen += 1
        self._detect_swings()
        return True

    def _close_day(self) -> None:
        """Move today into the history and start a new day."""
        if self._today is not None:
            self._days.append(self._today)
            vol = self._ret_vol()
            if vol is not None:
                self._day_vols.append(vol)
        self._today = None
        self._ret_n, self._ret_sum, self._ret_sq = 0, 0.0, 0.0

    def _detect_swings(self) -> None:
        """Confirm the pivot (if any) that now has ``swing_k`` bars after it."""
        k, bars = self.swing_k, self._bars
        i = len(bars) - 1 - k                     # the newest bar that can be confirmed
        if i < k:
            return                                # not enough bars before it
        window = [bars[j] for j in range(i - k, i + k + 1) if j != i]
        pivot = bars[i]
        if all(pivot["h"] > b["h"] for b in window):
            self._swing_highs.append(pivot["h"])
        if all(pivot["l"] < b["l"] for b in window):
            self._swing_lows.append(pivot["l"])

    # ------------------------------------------------------------------ #
    # today
    # ------------------------------------------------------------------ #
    @property
    def day_open(self) -> Optional[float]:
        return self._today["o"] if self._today else None

    @property
    def day_high(self) -> Optional[float]:
        return self._today["h"] if self._today else None

    @property
    def day_low(self) -> Optional[float]:
        return self._today["l"] if self._today else None

    @property
    def day_close(self) -> Optional[float]:
        """The latest close of the current day."""
        return self._today["c"] if self._today else None

    # ------------------------------------------------------------------ #
    # history
    # ------------------------------------------------------------------ #
    @property
    def prev_day(self) -> Optional[Dict[str, float]]:
        """Yesterday as ``{o,h,l,c,v}``, or None on the first day."""
        return dict(self._days[-1]) if self._days else None

    @property
    def days(self) -> List[Dict[str, float]]:
        """The last completed days (default 5), newest last. Today is not included."""
        return [dict(d) for d in self._days]

    # ------------------------------------------------------------------ #
    # structure
    # ------------------------------------------------------------------ #
    def swing_highs(self) -> List[float]:
        """Confirmed swing-high prices, newest last.

        A bar counts when its high is above the ``swing_k`` bars before and the
        ``swing_k`` bars after it, so it appears only once those later bars exist.
        """
        return list(self._swing_highs)

    def swing_lows(self) -> List[float]:
        """Confirmed swing-low prices, newest last (mirror of swing_highs)."""
        return list(self._swing_lows)

    # ------------------------------------------------------------------ #
    # nice to have (used by the later PPO retraining)
    # ------------------------------------------------------------------ #
    def ret(self, minutes: int) -> Optional[float]:
        """Return over the last ``minutes`` bars (close vs the close that many bars ago).

        None if there are not enough bars. Bars are one sim-minute each.
        """
        n = int(minutes)
        if n < 1 or len(self._bars) <= n:
            return None
        past = self._bars[-1 - n]["c"]
        return (self._bars[-1]["c"] / past - 1.0) if past else None

    def _ret_vol(self) -> Optional[float]:
        if self._ret_n < 2:
            return None
        mean = self._ret_sum / self._ret_n
        var = max(self._ret_sq / self._ret_n - mean * mean, 0.0)
        return math.sqrt(var)

    def vol_ratio(self) -> Optional[float]:
        """Today's realised volatility divided by the recent daily norm.

        Volatility is the standard deviation of intra-day 1-bar log returns. The
        norm is the mean of the completed days kept. None until today has at least
        two returns and one earlier day has a non-zero volatility.
        """
        today = self._ret_vol()
        norm = [x for x in self._day_vols if x > 0]
        if today is None or not norm:
            return None
        return today / (sum(norm) / len(norm))
