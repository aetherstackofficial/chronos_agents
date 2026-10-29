"""Real-data reference statistics and acceptance bands (spec §15).

For each instrument: rolling windows of the same length as a simulator run
(``n_days`` sessions, stepped by 20), every bar test computed exactly as on
the simulator (trade-price closes, no bootstrap), then

* band = 5th-95th percentile over CALIBRATION windows (first 70% of dates),
  widened by 10% of the value (§15.4);
* holdout = the same percentiles over windows wholly in the last 30%, and
  the share of holdout windows that fall inside the calibration band (§15.5);
* spec-default pass rate of the real windows, i.e. how often the real NSE
  market itself passes each spec threshold.

Usage: python -m validation.calibrate --n-days 268 [--instruments RELIANCE,...]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

from validation.checks.bar_tests import BAR_TESTS, intraday_profiles
from validation.checks.flow_tests import a8_no_free_lunch
from validation.data import load_real, real_windows
from validation.stats_utils import band_from_real, round_sig, summarize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data", "real", "nse_kaggle")
CFG_PATH = os.path.join(HERE, "config", "thresholds.yaml")
STEP_DAYS = 20
CALIB_FRACTION = 0.70


def load_cfg() -> dict:
    with open(CFG_PATH) as fh:
        return yaml.safe_load(fh)


def _score_window(w, cfg, seed):
    out = {}
    for name, fn in BAR_TESTS.items():
        res = fn(w, cfg, price="c", bootstrap=False, seed=seed)
        out[name] = {"metrics": res["metrics"], "pass": res["pass"] if res["criteria"] else None}
    res = a8_no_free_lunch(w, cfg, seed=seed, price="c")
    out["A8_no_free_lunch"] = {"metrics": res["metrics"], "pass": res["pass"]}
    return out


def calibrate_instrument(symbol: str, n_days: int) -> dict:
    cfg = load_cfg()
    t0 = time.time()
    s = load_real(os.path.join(DATA_DIR, f"{symbol}_minute.csv"))
    dates = s.info["dates"]
    split = int(round(CALIB_FRACTION * s.n_days))
    windows = {"calibration": [], "holdout": []}
    for start, end, w in real_windows(s, n_days, STEP_DAYS):
        group = "calibration" if end <= split else "holdout" if start >= split else None
        if group:
            windows[group].append((start, end, _score_window(w, cfg, seed=start)))
    full_prof = intraday_profiles(s, "c")
    tests = {}
    for name in windows["calibration"][0][2]:
        entry = {"metrics": {}}
        for group in ("calibration", "holdout"):
            p = [x[2][name]["pass"] for x in windows[group]]
            p = [v for v in p if v is not None]
            entry[f"spec_default_pass_rate_{group}"] = float(np.mean(p)) if p else None
        metric_names = windows["calibration"][0][2][name]["metrics"].keys()
        for mn in metric_names:
            cal = [x[2][name]["metrics"][mn] for x in windows["calibration"]]
            hol = [x[2][name]["metrics"][mn] for x in windows["holdout"]]
            band = band_from_real(cal)
            hv = np.asarray([v for v in hol if np.isfinite(v)])
            entry["metrics"][mn] = {
                "band": band, "calibration": summarize(cal), "holdout": summarize(hol),
                "holdout_in_band": (float(np.mean((hv >= band[0]) & (hv <= band[1])))
                                    if band and len(hv) else None)}
        tests[name] = entry
    # A5: correlation of each window's profile with the full-sample profile,
    # so a sim profile can be judged against the real spread of that number.
    corr = {"abs_ret": [], "volume": []}
    for _, _, w in real_windows(s, n_days, STEP_DAYS):
        wp = intraday_profiles(w, "c")
        for k in corr:
            corr[k].append(float(np.corrcoef(wp[k][1:], full_prof[k][1:])[0, 1]))
    for k, v in corr.items():
        tests["A5_intraday_seasonality"]["metrics"][f"{k}_profile_corr_with_real"] = {
            "band": band_from_real(v), "calibration": summarize(v)}
    return round_sig({
        "instrument": symbol, "source": "kaggle debashis74017 NIFTY-100 1-minute",
        "generated_on": time.strftime("%Y-%m-%d"),
        "cleaning": s.info["cleaning"], "first_date": dates[0], "last_date": dates[-1],
        "split_date": dates[split], "window_days": n_days, "step_days": STEP_DAYS,
        "n_windows": {k: len(v) for k, v in windows.items()},
        "price_series": "trade-price close (no quotes in the reference data)",
        "sigma_estimator": "Parkinson range",
        "tests": tests,
        "real_profiles": {k: v.tolist() for k, v in full_prof.items()},
        "seconds": time.time() - t0})


def calibration_path(symbol: str, n_days: int) -> str:
    return os.path.join(HERE, "config", f"calibration_{symbol}_{n_days}d.json")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-days", type=int, required=True)
    ap.add_argument("--instruments", default="RELIANCE,HDFCBANK,INFY,SBIN,ITC,MARUTI")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    syms = a.instruments.split(",")
    with ProcessPoolExecutor(a.workers) as ex:
        for sym, res in zip(syms, ex.map(calibrate_instrument, syms, [a.n_days] * len(syms))):
            with open(calibration_path(sym, a.n_days), "w") as fh:
                json.dump(res, fh, indent=1)
            print(f"{sym}: {res['n_windows']} windows, {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
