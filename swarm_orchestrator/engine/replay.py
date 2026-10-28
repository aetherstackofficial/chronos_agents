"""Pillar 1 — history repeater (PROTOCOL §9).

Drives historical 1-minute bars THROUGH the real matching engine: each
physics tick the replayer refreshes bracketing SIM_LP depth and fires ghost
child orders so real prints walk last_price approximately o -> h -> l -> c
of the current bar, with total quantity derived from the bar volume
(capped). Volume, RSI, VWAP and OFI therefore emerge from real trades.

The latch (first fill touching a ``STRAT_``/``USER_`` agent flips the engine
to REACTIVE permanently) lives in ``MatchingEngine._apply_fill``; this
module simply stops driving once ``engine.mode != "REPLAY"``.

Determinism: uses ONLY the engine's seeded rng.
"""
from __future__ import annotations

import calendar
import csv
import datetime as dt
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from .book import round_to_tick

if TYPE_CHECKING:  # avoid a runtime circular import
    from .matching_engine import MatchingEngine

logger = logging.getLogger("engine.replay")

_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

# Non-ISO date layouts accepted besides ISO ("2026-10-19 09:15[:SS]").
_TS_FORMATS = ("%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
               "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
               "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M")


def _parse_ts(text: str) -> Optional[int]:
    """Bar timestamp text -> epoch seconds, or None if it can't be read.

    The clock time is read as written and encoded as if it were UTC. That is
    the engine's convention (SIM_EPOCH is "09:15 IST-as-UTC") and what the
    dashboard's UTC formatters expect, so an NSE bar at 09:15 shows 09:15. Any
    UTC offset in the text ("+05:30") is dropped, not converted, for the same
    reason.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        t = dt.datetime.fromisoformat(text)
    except ValueError:
        t = None
        for fmt in _TS_FORMATS:
            try:
                t = dt.datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if t is None:
            return None
    return calendar.timegm(t.replace(tzinfo=None).timetuple())


class HistoricalReplayer:
    """Owns the historical bar cursor for one REPLAY run."""

    GHOST_AGENT = "GHOST_HIST"
    LP_AGENT = "SIM_LP"
    # Child prints per bar: open -> high -> low -> close.
    CHILD_TARGETS = 4
    # Cap on ladder levels per child so a large gap cannot bloat the book.
    MAX_WALK_LEVELS = 12
    # Bar volume is used as total child quantity, capped for book stability.
    MAX_BAR_QTY = 20_000
    MIN_BAR_QTY = CHILD_TARGETS
    # Ghost/ladder orders live one tick only; the resting bracket after the
    # bar is re-quoted by the normal SIM_LP refresh cadence.
    DRIVE_TIF_TICKS = 1

    def __init__(self, engine: "MatchingEngine", bars: List[dict]) -> None:
        self.engine = engine
        self.bars = bars
        self.cursor = 0
        self.finished = False

    # ------------------------------------------------------------------ #
    # CSV loading
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_csv(csv_path: str) -> List[dict]:
        """Load 1-minute bars from a CSV.

        Accepts columns date,time,open,high,low,close,volume
        (case-insensitive) or a single datetime/timestamp column instead of
        date+time. Raises ValueError with a clear message on bad input.

        Each bar also carries ``"ts"``: its date/time as epoch seconds (see
        _parse_ts), so the replay shows the file's real dates on the chart
        instead of the engine's own clock. If ANY row's date/time can't be
        read, ``ts`` is None for every bar and the engine falls back to its
        own clock (the file still loads; an unreadable date is not an error).
        """
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise ValueError(f"{csv_path}: empty CSV (no header row)")
            col_map = {name.strip().lower(): name for name in reader.fieldnames}
            missing = [c for c in _REQUIRED_COLUMNS if c not in col_map]
            if missing:
                raise ValueError(
                    f"{csv_path}: missing required columns {missing}; "
                    f"found {sorted(col_map)}")
            has_datetime = ("datetime" in col_map or "timestamp" in col_map
                            or ("date" in col_map and "time" in col_map))
            if not has_datetime:
                raise ValueError(
                    f"{csv_path}: need date+time columns or a single "
                    "datetime/timestamp column")
            bars: List[dict] = []
            all_dated = True
            for line_no, row in enumerate(reader, start=2):
                try:
                    o = float(row[col_map["open"]])
                    h = float(row[col_map["high"]])
                    l = float(row[col_map["low"]])
                    c = float(row[col_map["close"]])
                    v = int(float(row[col_map["volume"]]))
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(
                        f"{csv_path}: line {line_no}: unparseable bar "
                        f"({exc})") from exc
                if min(o, h, l, c) <= 0 or v < 0:
                    raise ValueError(
                        f"{csv_path}: line {line_no}: non-positive price "
                        "or negative volume")
                if h < max(o, c, l) or l > min(o, c, h):
                    raise ValueError(
                        f"{csv_path}: line {line_no}: inconsistent OHLC "
                        f"(o={o} h={h} l={l} c={c})")
                if "datetime" in col_map:
                    stamp = row.get(col_map["datetime"])
                elif "timestamp" in col_map:
                    stamp = row.get(col_map["timestamp"])
                else:
                    stamp = (f"{row.get(col_map['date'], '')} "
                             f"{row.get(col_map['time'], '')}")
                ts = _parse_ts(stamp)
                all_dated = all_dated and ts is not None
                bars.append({"o": o, "h": h, "l": l, "c": c, "v": v, "ts": ts})
        if not bars:
            raise ValueError(f"{csv_path}: CSV contains no bars")
        if not all_dated:
            logger.warning("%s: some date/time values are unreadable; replay "
                           "will use the engine clock instead of file dates",
                           csv_path)
            for b in bars:
                b["ts"] = None
        return bars

    # ------------------------------------------------------------------ #
    # per-tick drive
    # ------------------------------------------------------------------ #
    def step(self, pubs: List[Tuple[str, dict]]) -> None:
        """Consume one historical bar: print ~4 ghost child trades along
        o -> h -> l -> c, then leave a resting SIM_LP bracket around the
        close so swarm agents can trade during the live second."""
        engine = self.engine
        if self.cursor >= len(self.bars):
            if not self.finished:
                self.finished = True
                engine.paused = True
                pubs.append(("event", {
                    "seq": engine._next_seq(), "kind": "session",
                    "action": "day_close", "day_count": engine.day_count}))
                logger.info("replay tape exhausted after %d bars; paused",
                            len(self.bars))
            return

        bar = self.bars[self.cursor]
        self.cursor += 1
        # Show the file's own date/time: trades and the tick published this
        # second both read engine.sim_unix_time, and the dashboard keys its
        # candles on it. (Without this the chart ran on the engine's own clock
        # from 1 Jan 2026, whatever dates the CSV held.)
        if bar.get("ts") is not None:
            engine.sim_unix_time = bar["ts"]
        total_qty = max(self.MIN_BAR_QTY, min(bar["v"], self.MAX_BAR_QTY))
        child_qty = max(1, total_qty // self.CHILD_TARGETS)

        for target in (bar["o"], bar["h"], bar["l"], bar["c"]):
            if engine.mode != "REPLAY":
                return  # the latch fired; the swarm owns the price now
            self._drive_to(target, child_qty, pubs)

        if engine.mode == "REPLAY":
            engine._refresh_sim_lp(pubs, reference=bar["c"])

    def _drive_to(self, target: float, child_qty: int,
                  pubs: List[Tuple[str, dict]]) -> None:
        """Print real trades that walk last_price to the target price.

        Posts a SIM_LP limit ladder from last_price to the target, then
        fires a marketable ghost LIMIT at the target sized to the visible
        opposite quantity — the limit price guarantees no overshoot, and
        the ladder guarantees the final print lands on the target. Swarm
        orders resting inside the walked range fill for real on the way.
        """
        engine = self.engine
        target = round_to_tick(target, engine.tick)
        last = round_to_tick(engine.last_price, engine.tick)
        engine.flush_agent_orders(self.LP_AGENT)

        if abs(target - last) < engine.tick / 2:
            # Flat leg: print volume at the current price.
            engine.submit_flow_order(self.LP_AGENT, "SELL", "LIMIT",
                                     child_qty, target, pubs,
                                     tif_ticks=self.DRIVE_TIF_TICKS)
            side = "BUY"
        else:
            side = "BUY" if target > last else "SELL"
            maker_side = "SELL" if side == "BUY" else "BUY"
            per_level = max(1, child_qty // self.MAX_WALK_LEVELS)
            for price in self._ladder_prices(last, target):
                engine.submit_flow_order(self.LP_AGENT, maker_side, "LIMIT",
                                         per_level, price, pubs,
                                         tif_ticks=self.DRIVE_TIF_TICKS)

        available = engine.book.cum_qty_within(side, target,
                                               exclude_agent=self.GHOST_AGENT)
        if available > 0:
            engine.submit_flow_order(self.GHOST_AGENT, side, "LIMIT",
                                     available, target, pubs,
                                     tif_ticks=self.DRIVE_TIF_TICKS)

    def _ladder_prices(self, last: float, target: float) -> List[float]:
        """Tick prices from just past last_price to the target (inclusive),
        evenly thinned to at most MAX_WALK_LEVELS so big gaps stay cheap."""
        tick = self.engine.tick
        n_ticks = int(round(abs(target - last) / tick))
        n_levels = min(max(n_ticks, 1), self.MAX_WALK_LEVELS)
        step = (target - last) / n_levels
        prices: List[float] = []
        seen: set = set()
        for i in range(1, n_levels + 1):
            px = round_to_tick(last + step * i, tick)
            if px not in seen and px > 0:
                seen.add(px)
                prices.append(px)
        if not prices or prices[-1] != target:
            if target in seen:
                prices.remove(target)
            prices.append(target)
        return prices
