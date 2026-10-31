"""Round-number clustering: build the real reference, or score a simulated run.

    python -m validation.level_report --reference            # real NSE reference -> config/round_number_reference.json
    python -m validation.level_report --seed 1 --days 3      # simulate and compare with the reference

The reference needs the Kaggle files (python -m validation.fetch_data) and uses
only the period after the last corporate action of each stock: the vendor
adjusts older prices for bonus issues and splits, which moves round numbers
off their real prices (RELIANCE: 1:1 bonus on 2024-10-28).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data", "real", "nse_kaggle")
REF_PATH = os.path.join(HERE, "config", "round_number_reference.json")
# First session whose prices are NOT back-adjusted for a later corporate action.
CLEAN_FROM = {"RELIANCE": "2024-11-01", "SBIN": "2025-07-01", "ITC": "2025-07-01",
              "MARUTI": "2025-07-01"}


def build_reference() -> dict:
    import pandas as pd
    from validation.checks.level_tests import real_reference
    series = {}
    for sym, start in CLEAN_FROM.items():
        path = os.path.join(DATA_DIR, f"{sym}_minute.csv")
        if not os.path.exists(path):
            print(f"{sym}: {path} missing (python -m validation.fetch_data --instruments {sym})")
            continue
        df = pd.read_csv(path, usecols=["date", "high", "low", "close"])
        df = df[(df[["high", "low", "close"]] > 0).all(axis=1)]
        df = df[pd.to_datetime(df["date"]).dt.normalize() >= start]
        series[sym] = {"c": df["close"].values, "h": df["high"].values, "l": df["low"].values}
    ref = real_reference(series)
    for sym in ref:
        ref[sym]["clean_from"] = CLEAN_FROM[sym]
        ref[sym]["n_bars"] = int(len(series[sym]["c"]))
    return ref


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", action="store_true", help="measure the real data and write the reference file")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--start-price", type=float, default=1354.0)
    a = ap.parse_args(argv)

    if a.reference:
        ref = build_reference()
        os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
        with open(REF_PATH, "w") as fh:
            json.dump(ref, fh, indent=1)
        for sym, v in ref.items():
            print(f"{sym}: tick {v['tick']}, mean excess at the base unit {v['excess_base_unit_mean']:.2f}")
        print("wrote", REF_PATH)
        return

    from validation.checks.level_tests import infer_tick, t9_round_number_clustering
    from validation.sim_driver import simulate
    out = simulate(a.seed, a.days, a.start_price, counts=None)
    bars = out.bars
    series = {"c": np.asarray(bars["c"], float), "h": np.asarray(bars["h"], float), "l": np.asarray(bars["l"], float)}
    tick = infer_tick(series["c"])
    ref = json.load(open(REF_PATH)) if os.path.exists(REF_PATH) else None
    res = t9_round_number_clustering(series, tick, ref)
    print(json.dumps({k: res[k] for k in ("metrics", "criteria", "pass", "target_pass")}, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1:])
