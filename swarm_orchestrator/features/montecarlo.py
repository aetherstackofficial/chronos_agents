"""Monte-Carlo counterfactual fan (PROTOCOL_V2.md §3) — real, in-process.

Every path is a full transport-free :class:`engine.matching_engine.MatchingEngine`
run: seeded ``seed_base + i``, driven by ``horizon`` real ``physics_tick()``
calls, with prices produced ONLY by real fills in the book (regime ghost flow +
``SIM_LP`` designated liquidity, exactly as in a live run). An optional ``shock``
injects a real MARKET order at a chosen tick — a "what if I trade this size"
counterfactual whose impact is emergent book depletion, not gap math.

Paths are fanned out over a :class:`concurrent.futures.ProcessPoolExecutor`
(module-level picklable worker) and fall back to a sequential loop if the pool
cannot start (e.g. restricted spawn environments). NO sockets, NO torch.
"""
from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("chronos.features")

# ---------------------------------------------------------------------------
# Limits & policy constants
# ---------------------------------------------------------------------------
DEFAULT_N_PATHS = 60
DEFAULT_HORIZON = 90
MIN_PATHS = 2
MIN_HORIZON = 10
MAX_SAMPLE_PATHS = 30
MAX_WORKERS = min(6, (os.cpu_count() or 2))
SHOCK_AGENT_ID = "MC_USER"
PERCENTILE_LEVELS: Tuple[int, ...] = (5, 25, 50, 75, 95)

# One (symbol, price, seed, horizon, shock) tuple per path — picklable.
PathArgs = Tuple[str, float, int, int, Optional[Dict[str, Any]]]


def _simulate_path(args: PathArgs) -> List[float]:
    """Run ONE full engine path and return its per-tick ``last_price`` series.

    Module-level so it is picklable by ``ProcessPoolExecutor`` (Windows spawn).
    The engine is constructed without persistence (``db=None``) — pure in-memory
    matching, deterministic for the given seed.
    """
    from engine.matching_engine import MatchingEngine  # local: cheap in workers

    symbol, price, seed, horizon, shock = args
    eng = MatchingEngine()
    eng.init_sim(symbol, "TECH", price, seed=seed)

    shock_minute = -1
    if shock is not None:
        shock_minute = int(shock["minute"])

    path: List[float] = []
    for tick in range(horizon):
        eng.physics_tick()
        if tick == shock_minute:
            eng.handle_message({
                "msg": "ORDER",
                "agent_id": SHOCK_AGENT_ID,
                "action": shock["side"],          # type: ignore[index]
                "type": "MARKET",
                "qty": int(shock["qty"]),         # type: ignore[index]
            })
        path.append(float(eng.last_price))
    return path


def _validate_shock(shock: Optional[Dict[str, Any]], horizon: int) -> Optional[Dict[str, Any]]:
    """Normalize/validate the optional shock spec; raise ``ValueError`` if bad."""
    if shock is None:
        return None
    if not isinstance(shock, dict):
        raise ValueError("shock must be an object {minute, side, qty}")
    try:
        minute = int(shock["minute"])
        side = str(shock["side"]).upper()
        qty = int(shock["qty"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"shock requires integer 'minute', 'qty' and 'side': {exc}") from exc
    if side not in ("BUY", "SELL"):
        raise ValueError("shock side must be 'BUY' or 'SELL'")
    if qty <= 0:
        raise ValueError("shock qty must be a positive integer")
    minute = max(0, min(minute, horizon - 1))
    return {"minute": minute, "side": side, "qty": qty}


def _round_series(values: np.ndarray, ndigits: int = 4) -> List[float]:
    """Round a 1-D array to a plain float list (thin JSON payload)."""
    return [round(float(v), ndigits) for v in values]


def run_fan(symbol: str, price: float, n_paths: int = DEFAULT_N_PATHS,
            horizon: int = DEFAULT_HORIZON, seed_base: int = 1000,
            shock: Optional[Dict[str, Any]] = None,
            max_paths: int = 400, max_horizon: int = 750) -> dict:
    """Run the Monte-Carlo fan and return the PROTOCOL_V2 §3 result dict.

    :param symbol: instrument symbol for every simulated engine.
    :param price: starting price of every path.
    :param n_paths: number of independent engine paths (clamped to
        ``[MIN_PATHS, max_paths]``).
    :param horizon: physics ticks (sim-minutes) per path (clamped to
        ``[MIN_HORIZON, max_horizon]``).
    :param seed_base: path ``i`` uses seed ``seed_base + i`` — the whole fan is
        deterministic for a given ``seed_base``.
    :param shock: optional ``{"minute": int, "side": "BUY"|"SELL", "qty": int}``
        counterfactual market order injected on every path.
    :returns: ``{symbol, n_paths, horizon, percentiles{p5..p95}, mean,
        sample_paths, terminal{p5..p95,mean,min,max}, start_price}``.
    :raises ValueError: on non-positive price, empty symbol, or malformed shock.
    """
    if not symbol or not str(symbol).strip():
        raise ValueError("symbol must be a non-empty string")
    price = float(price)
    if not math.isfinite(price) or price <= 0.0:
        raise ValueError("price must be a positive finite number")

    n_paths = max(MIN_PATHS, min(int(n_paths), int(max_paths)))
    horizon = max(MIN_HORIZON, min(int(horizon), int(max_horizon)))
    shock = _validate_shock(shock, horizon)

    tasks: List[PathArgs] = [
        (str(symbol), price, int(seed_base) + i, horizon, shock)
        for i in range(n_paths)
    ]

    paths = _run_pool(tasks)

    arr = np.asarray(paths, dtype=np.float64)          # shape (n_paths, horizon)
    pct = np.percentile(arr, PERCENTILE_LEVELS, axis=0)  # (5, horizon)
    percentiles = {
        f"p{level}": _round_series(pct[k])
        for k, level in enumerate(PERCENTILE_LEVELS)
    }
    mean_series = _round_series(arr.mean(axis=0))

    if n_paths <= MAX_SAMPLE_PATHS:
        sample_idx = np.arange(n_paths)
    else:
        sample_idx = np.linspace(0, n_paths - 1, MAX_SAMPLE_PATHS).astype(int)
    sample_paths = [_round_series(arr[i], 2) for i in sample_idx]

    final = arr[:, -1]
    terminal = {
        f"p{level}": round(float(np.percentile(final, level)), 4)
        for level in PERCENTILE_LEVELS
    }
    terminal["mean"] = round(float(final.mean()), 4)
    terminal["min"] = round(float(final.min()), 4)
    terminal["max"] = round(float(final.max()), 4)

    return {
        "symbol": str(symbol),
        "n_paths": n_paths,
        "horizon": horizon,
        "percentiles": percentiles,
        "mean": mean_series,
        "sample_paths": sample_paths,
        "terminal": terminal,
        "start_price": price,
    }


def _run_pool(tasks: List[PathArgs]) -> List[List[float]]:
    """Fan the path tasks over a process pool; sequential fallback on failure."""
    n = len(tasks)
    try:
        chunksize = max(1, math.ceil(n / (MAX_WORKERS * 2)))
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            paths = list(pool.map(_simulate_path, tasks, chunksize=chunksize))
        log.info("montecarlo: %d paths x %d ticks via %d workers",
                 n, tasks[0][3], MAX_WORKERS)
        return paths
    except Exception as exc:  # BrokenProcessPool, PicklingError, spawn issues
        log.warning("montecarlo: process pool failed (%s); "
                    "falling back to sequential loop", exc)
        return [_simulate_path(t) for t in tasks]
