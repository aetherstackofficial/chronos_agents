"""A10 round-number clustering of trade prices (Phase 3; numpy only).

Real traders put orders, and so get trades, at round prices more often than a
uniform walk over the tick grid would. The statistic compares, on the SAME tick
grid, how often a price sits exactly on a multiple of a step with how often it
sits 1-2 ticks next to one:

    excess(step) = P(price is a multiple of step) / mean P(price is k ticks away), k = -2, -1, +1, +2

1.0 means no clustering. The grid control removes the effect of the tick size
itself (a coarser grid puts more prices on every multiple).

Why it exists: agents place a share ``a`` of their limit orders at price
levels (agents/config/retail_personalities.yaml ``levels.anchor_prob``). If a
share ``a`` of prices sits on one level per step and the rest are spread over
``g`` grid points per step, then excess = 1 + a*g/(1-a), i.e.
``a = (excess - 1) / (g + excess - 1)``. ``anchor_prob`` was set from the real
excess of four NSE stocks (validation/config/round_number_reference.json). It
can only be checked in the simulator once retail trades a real share of the
volume; until then the simulated excess stays close to 1.0, and that is the
honest result this check reports.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

NEIGHBOURS = (-2, -1, 1, 2)
COLUMNS = ("c", "h", "l")


def infer_tick(prices: np.ndarray) -> float:
    """Coarsest of the NSE tick sizes that >= 99% of the prices sit on."""
    p = np.asarray(prices, float)
    for t in (0.5, 0.1, 0.05, 0.01, 0.001):
        if np.mean(np.abs(p / t - np.round(p / t)) < 1e-6) > 0.99:
            return t
    return 0.001


def base_unit(prices: np.ndarray, tick: float) -> np.ndarray:
    """Per-price base round-number unit, the same rule the agents use
    (agents/levels.py level_unit): 1/1000 of the price's decade, >= one tick."""
    p = np.asarray(prices, float)
    return np.maximum(10.0 ** (np.floor(np.log10(p)) - 3), tick)


def level_excess(prices: np.ndarray, step, tick: float) -> float:
    """Excess of prices exactly on multiples of `step` (scalar or per-price
    array) over their on-grid neighbours. NaN when the step is not wider than
    two ticks (neighbours would be levels themselves) or there is no data."""
    p = np.asarray(prices, float)
    step = np.broadcast_to(np.asarray(step, float), p.shape)
    ok = step > 2.0 * tick + 1e-12
    if not ok.any():
        return float("nan")
    p, step = p[ok], step[ok]
    res = p - np.round(p / step) * step
    tol = tick * 1e-3
    at = np.mean(np.abs(res) < tol)
    ctl = np.mean([np.mean(np.abs(res - k * tick) < tol) for k in NEIGHBOURS])
    return float(at / ctl) if ctl > 0 else float("nan")


def implied_anchor_prob(excess: float, grid_points_per_step: float) -> float:
    """Share of prices that must sit on levels to produce `excess` (see module doc)."""
    if not np.isfinite(excess) or excess <= 1.0:
        return 0.0
    return float((excess - 1.0) / (grid_points_per_step + excess - 1.0))


def excess_by_tier(series: Dict[str, np.ndarray], tick: float,
                   multiples: Sequence[int] = (1, 5, 10, 50, 100)) -> dict:
    """{column: {multiple: excess}} with step = multiple * base unit of each price."""
    out = {}
    for col, prices in series.items():
        prices = np.asarray(prices, float)
        u = base_unit(prices, tick)
        out[col] = {int(m): level_excess(prices, u * m, tick) for m in multiples}
    return out


def t9_round_number_clustering(series: Dict[str, np.ndarray], tick: float,
                               real_reference: Optional[dict], cfg: Optional[dict] = None) -> dict:
    """Simulated round-number clustering against the real NSE reference.

    ``series``: {"c": closes, "h": highs, "l": lows} of the simulated bars.
    Gate (pre-registered, deliberately low): some clustering at all, i.e. the
    mean excess over close/high/low at the base unit is at least ``min_excess``.
    Target (reported, not gated): the simulated excess sits inside the real
    band, widened by 20%.
    """
    c = {"min_excess": 1.15, "band_margin": 0.20, **(cfg or {})}
    table = excess_by_tier(series, tick)
    base = [table[col][1] for col in COLUMNS if col in table and np.isfinite(table[col][1])]
    mean_base = float(np.mean(base)) if base else float("nan")
    metrics = {"excess_base_unit_mean": mean_base, "tick": float(tick),
               "n_prices": float(sum(len(v) for v in series.values()))}
    for col in COLUMNS:
        if col in table:
            for m, v in table[col].items():
                metrics[f"excess_{col}_x{m}"] = v
    crit = {"has_clustering": bool(np.isfinite(mean_base) and mean_base >= c["min_excess"])}
    target = None
    if real_reference:
        band = [v["excess_base_unit_mean"] for v in real_reference.values()
                if np.isfinite(v.get("excess_base_unit_mean", float("nan")))]
        if band:
            lo, hi = min(band) * (1 - c["band_margin"]), max(band) * (1 + c["band_margin"])
            metrics["real_band_lo"], metrics["real_band_hi"] = float(lo), float(hi)
            target = bool(np.isfinite(mean_base) and lo <= mean_base <= hi)
    return {"metrics": metrics, "criteria": crit, "pass": all(crit.values()),
            "target_pass": target, "extra": {"table": {k: {int(m): v for m, v in d.items()}
                                                       for k, d in table.items()}}}


def real_reference(series_by_stock: Dict[str, Dict[str, np.ndarray]]) -> dict:
    """Reference numbers for the check, measured with the SAME function on real
    bars: {stock: {tick, excess by tier, mean excess at the base unit, ...}}."""
    ref = {}
    for sym, series in series_by_stock.items():
        tick = infer_tick(np.concatenate([np.asarray(v, float) for v in series.values()]))
        table = excess_by_tier(series, tick)
        base = [table[c][1] for c in COLUMNS if c in table and np.isfinite(table[c][1])]
        ref[sym] = {"tick": tick,
                    "excess_base_unit_mean": float(np.mean(base)) if base else float("nan"),
                    "excess": {c: {str(m): v for m, v in d.items()} for c, d in table.items()}}
    return ref
