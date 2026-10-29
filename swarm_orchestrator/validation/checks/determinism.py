"""T6 replay fidelity (Part A) and seeded determinism (Part B), spec §9."""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np
import pandas as pd

from validation.data import BarSeries
from validation.sim_driver import Simulation

NSE_TICKS = ((250.0, 0.01), (1000.0, 0.05), (5000.0, 0.10), (10000.0, 0.50),
             (20000.0, 1.0), (float("inf"), 5.0))   # CM tick table, rev. 2025-04-15


def _nse_tick(p: float) -> float:
    for upper, t in NSE_TICKS:
        if p < upper:
            return t
    return NSE_TICKS[-1][1]


def _off_grid(prices: np.ndarray, tick) -> float:
    t = np.asarray(tick, float)
    return float(np.mean(np.abs(prices / t - np.round(prices / t)) > 1e-6))


def write_replay_csv(s: BarSeries, month: str, out_dir: str) -> tuple:
    """Write the month's full sessions in the engine's replay CSV format."""
    dates = np.asarray(s.info["dates"])
    days = np.flatnonzero(np.char.startswith(dates.astype(str), month))
    if len(days) == 0:
        raise ValueError(f"no full sessions in {month}")
    mask = np.isin(s.day, days)
    sub = s.subset(mask)
    ts = pd.to_datetime(dates[sub.day]) + pd.to_timedelta(9 * 60 + 15 + sub.minute, "min")
    df = pd.DataFrame({"datetime": ts.strftime("%Y-%m-%d %H:%M:%S"), "open": sub.o,
                       "high": sub.h, "low": sub.l, "close": sub.c,
                       "volume": sub.v.astype(np.int64)})
    path = os.path.join(out_dir, f"replay_{s.label}_{month}.csv")
    df.to_csv(path, index=False)
    return path, sub


def _replay(path: str, n_bars: int, seed: int, swarm: bool, symbol: str) -> dict:
    sim = Simulation(seed, 1.0, symbol=symbol, counts=None if swarm else {},
                     replay_csv=path)
    eng = sim.engine
    rows = []
    for _ in range(n_bars):
        p = sim.step() if swarm else sim.physics()
        b = p["bar"]
        rows.append((p["day_count"], p["market_minute"], b["o"], b["h"], b["l"], b["c"], b["v"]))
    a = np.asarray(rows, float)
    return {"day": a[:, 0], "minute": a[:, 1], "o": a[:, 2], "h": a[:, 3],
            "l": a[:, 4], "c": a[:, 5], "v": a[:, 6], "engine_tick": eng.tick,
            "mode_at_end": eng.mode}


def part_a(s: BarSeries, cfg: dict, seed: int = 1, swarm: bool = False) -> dict:
    month = cfg["t6_determinism"]["replay_month"]
    with tempfile.TemporaryDirectory() as tmp:
        path, sub = write_replay_csv(s, month, tmp)
        sim = _replay(path, len(sub), seed, swarm, s.label)
    paise = lambda x: np.round(np.asarray(x) * 100).astype(np.int64)
    bad = np.zeros(len(sub), bool)
    per_field = {}
    for f in ("o", "h", "l", "c"):
        mis = paise(getattr(sub, f)) != paise(sim[f])
        per_field[f] = int(mis.sum())
        bad |= mis
    vmis = sub.v.astype(np.int64) != sim["v"].astype(np.int64)
    per_field["v"] = int(vmis.sum())
    bad |= vmis
    ts_ok = bool(np.array_equal(sim["day"] - 1, sub.day - sub.day.min()) and
                 np.array_equal(sim["minute"] - 1, sub.minute))
    first = np.flatnonzero(bad)[:5]
    nse_ticks = np.vectorize(_nse_tick)(sub.c)
    m = {"bars": float(len(sub)), "mismatching_bars": float(bad.sum()),
         **{f"mismatch_{k}": float(v) for k, v in per_field.items()},
         "volume_over_engine_cap": float(np.mean(sub.v > 20000)),
         "real_close_off_engine_grid": _off_grid(sub.c, sim["engine_tick"]),
         "real_close_off_nse_grid": _off_grid(sub.c, nse_ticks),
         "engine_tick": float(sim["engine_tick"])}
    crit = {"zero_mismatching_bars": int(bad.sum()) == 0,
            "bar_count_and_timestamps_identical": ts_ok}
    examples = [{"bar": int(i), "real": [float(getattr(sub, f)[i]) for f in "ohlc"] + [int(sub.v[i])],
                 "sim": [float(sim[f][i]) for f in "ohlc"] + [int(sim["v"][i])]} for i in first]
    return {"metrics": m, "criteria": crit, "pass": all(crit.values()),
            "extra": {"month": month, "swarm": swarm, "first_mismatches": examples,
                      "mode_at_end": sim["mode_at_end"]}}


def _hash_run(seed: int, days: int, price: float, null_strategy=False,
              seed_globals=True) -> str:
    sim = Simulation(seed, price, hash_events=True, null_strategy=null_strategy,
                     seed_globals=seed_globals)
    for _ in range(days * 375):
        sim.step()
    return sim.event_hash()


def part_b(cfg: dict, price: float, seed: int = 1) -> dict:
    days = cfg["t6_determinism"]["part_b_days"]
    h1 = _hash_run(seed, days, price)
    h2 = _hash_run(seed, days, price)
    hn = _hash_run(seed, days, price, null_strategy=True)
    hd = _hash_run(seed + 1, days, price)
    d1 = _hash_run(seed, days, price, seed_globals=False)
    d2 = _hash_run(seed, days, price, seed_globals=False)
    crit = {"same_seed_identical": h1 == h2,
            "null_strategy_identical": h1 == hn,
            "different_seed_differs": h1 != hd}
    m = {"days": float(days)}
    return {"metrics": m, "criteria": crit, "pass": all(crit.values()),
            "extra": {"hash_seed": h1, "hash_seed_repeat": h2, "hash_null_strategy": hn,
                      "hash_other_seed": hd,
                      "deployed_rng_identical": d1 == d2,
                      "deployed_note": ("worker.py seeds neither the orchestrator nor "
                                        "random/numpy; two deployed-config runs with the "
                                        "same engine seed give hashes "
                                        + ("EQUAL" if d1 == d2 else "DIFFERENT"))}}
