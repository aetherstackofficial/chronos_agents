"""Run the realism suite against the current simulator (spec §14).

    python -m validation.runner --mode smoke|full [--instrument RELIANCE]

Per seed (one process each): a warm-up + N-return run of the live swarm,
bar/flow/book tests on its output, and on "impact seeds" the snapshot
harness (T7a, T7b, T8, A6, A7). T6 and the T4 ablation run as separate jobs.
Results are aggregated with the §2.6 rule and written as a §13 report card.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from validation.calibrate import calibration_path, load_cfg, CFG_PATH, DATA_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(HERE, "reports")

MODES = {
    # Smoke: §14.5 sizes. Impact harness heavily reduced; never a realism claim.
    "smoke": {"seeds": 5, "n_returns": 20000, "impact_seeds": 2,
              "n_snap_single": 20, "n_snap_meta": 4, "ablation_seeds": 3,
              "book_snap_prob": 0.25},
    # Full: §2.2 sizes for bar tests. Impact harness reduced from the spec's
    # 20 seeds x 200 snapshots to fit the measured ~160 ticks/s/core.
    "full": {"seeds": 20, "n_returns": 100000, "impact_seeds": 8,
             "n_snap_single": 200, "n_snap_meta": 50, "ablation_seeds": 5,
             "book_snap_prob": 0.05},
}
RETURNS_PER_DAY = 374          # 375 bars, first bar of each day excluded

# Metrics compared against the real-data band (all computed on trade closes).
KEY_METRICS = {
    "T1_fat_tails": ["excess_kurtosis", "zero_return_frac"],
    "T2_vol_clustering": ["rho_1", "beta"],
    "T3_no_return_autocorr": ["rho_1", "max_abs_rho"],
    "T4_vol_volume": ["spearman", "pearson"],
    "A1_aggregational_gaussianity": ["kurt_1m", "kurt_60m", "daily_over_1min"],
    "A3_volume_clustering": ["rho_1", "beta"],
    "A4_extreme_intermittency": ["interarrival_cv"],
    "A5_intraday_seasonality": ["abs_ret_open_ratio", "volume_open_ratio",
                                "volume_close_ratio", "abs_ret_profile_corr_with_real",
                                "volume_profile_corr_with_real"],
    "A8_no_free_lunch": ["max_rule_t", "random_mean_net_bps"],
    "A9_leverage_asymmetry": ["leverage_mean_1_10", "skew_multi_day"],
}
# Tests whose spec row is truncated in the PDF: judged on the real band only.
BAND_ONLY = {"A4_extreme_intermittency", "A5_intraday_seasonality", "A9_leverage_asymmetry"}
UNCALIBRATED = {"T5_book_shape", "T7_sqrt_impact", "T7a_book_walk", "T8_monotonic_impact",
                "A2_order_flow_sign_memory", "A6_impact_decay", "A7_spread_depth_recovery"}


def _quiet_worker():
    logging.disable(logging.CRITICAL)
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)


# --------------------------------------------------------------------------- #
# per-seed job
# --------------------------------------------------------------------------- #
def seed_job(seed: int, mode: dict, cfg: dict, price: float, impact: bool,
             real_profiles: dict) -> dict:
    _quiet_worker()
    from validation.checks import bar_tests, book_tests, flow_tests, impact_tests
    from validation.data import from_sim
    from validation.sim_driver import Simulation
    t0 = time.time()
    warm = cfg["protocol"]["warmup_days"]
    n_days = math.ceil(mode["n_returns"] / RETURNS_PER_DAY)
    sim = Simulation(seed, price)
    n_ticks = (warm + n_days) * 375
    rng = np.random.default_rng([seed, 11])
    n_snap = mode["n_snap_single"] if impact else 0
    snap_at = set(rng.choice(np.arange(warm * 375, n_ticks), n_snap, replace=False).tolist()) \
        if n_snap else set()
    snaps, book, spreads = [], [], []
    inv = {"crossed": 0, "nonpositive_size": 0, "off_grid": 0, "duplicate_ids": 0, "checked": 0}
    L = cfg["t5_book_shape"]["L"]
    for t in range(n_ticks):
        p = sim.physics()
        if t in snap_at:
            snaps.append(sim.snapshot())
        sim.agents_act(p)
        for k, v in sim.check_invariants().items():
            inv[k] += v
        inv["checked"] += 1
        if t >= warm * 375 and rng.random() < mode["book_snap_prob"]:
            q, sp = sim.depth_by_tick_distance(L)
            book.append(q)
            spreads.append(np.nan if sp is None else sp)
    t_run = time.time() - t0
    bars = sim.bars_array()
    s = from_sim(bars, warm, f"SIM_seed{seed}")
    res = {"spec": {}, "real_sym": {}}
    for name, fn in bar_tests.BAR_TESTS.items():
        res["spec"][name] = fn(s, cfg, price="mid", bootstrap=True, seed=seed)
        res["real_sym"][name] = fn(s, cfg, price="c", bootstrap=False, seed=seed)
    prof = bar_tests.intraday_profiles(s, "c")
    for k in ("abs_ret", "volume"):
        corr = float(np.corrcoef(prof[k][1:], np.asarray(real_profiles[k])[1:])[0, 1])
        res["real_sym"]["A5_intraday_seasonality"]["metrics"][f"{k}_profile_corr_with_real"] = corr
    res["spec"]["A8_no_free_lunch"] = flow_tests.a8_no_free_lunch(s, cfg, seed=seed, price="mid")
    res["real_sym"]["A8_no_free_lunch"] = flow_tests.a8_no_free_lunch(s, cfg, seed=seed, price="c")
    min_bar = warm * 375 + 1
    res["spec"]["A2_order_flow_sign_memory"] = flow_tests.a2_sign_memory(sim.trades_array(), cfg, min_bar)
    res["spec"]["T5_book_shape"] = book_tests.t5_book_shape(np.asarray(book), np.asarray(spreads), cfg, seed)
    res["spec"]["T0_invariants"] = book_tests.invariants_result(inv)

    # Daily scale for impact normalisation (§10 setup step 1): close-to-close.
    last = np.r_[s.day[1:] != s.day[:-1], True]
    dclose = s.c[last]
    sigma_D = float(np.std(np.diff(np.log(dclose)), ddof=1))
    V_D = float(np.bincount(s.day, weights=s.v).mean())
    meta = {"seed": seed, "N_returns": int(len(s) - s.n_days), "n_days": s.n_days,
            "sigma_D": sigma_D, "V_D": V_D, "price_first": float(s.c[0]),
            "price_last": float(s.c[-1]), "price_min": float(s.c.min()),
            "price_max": float(s.c.max()), "engine_tick_last": float(bars["tick"][-1]),
            "mid_nan_filled": s.info.get("mid_nan_filled", 0),
            "zero_volume_bar_frac": float(np.mean(s.v == 0)),
            "median_bar_volume": float(np.median(s.v)), "run_seconds": t_run}
    if impact and snaps:
        t1 = time.time()
        sizes = sorted(set(cfg["t8_monotonic"]["sizes"]) |
                       {int(round(q)) for q in np.geomspace(*cfg["t7_sqrt_impact"]["book_walk_Q"][:2],
                                                            cfg["t7_sqrt_impact"]["book_walk_Q"][2])})
        tr = impact_tests.single_order_trials(sim, snaps, sizes)
        res["spec"]["T8_monotonic_impact"] = impact_tests.t8_monotonic(tr, cfg)
        res["spec"]["T7a_book_walk"] = impact_tests.t7a_book_walk(tr, cfg, seed)
        res["spec"]["A7_spread_depth_recovery"] = impact_tests.a7_recovery(sim, snaps, cfg)
        mt = impact_tests.metaorder_trials(sim, snaps[:mode["n_snap_meta"]], V_D, cfg)
        res["spec"]["T7_sqrt_impact"] = impact_tests.t7b_metaorder(mt, sigma_D, cfg, seed)
        res["spec"]["A6_impact_decay"] = impact_tests.a6_decay(mt, cfg)
        meta["impact_seconds"] = time.time() - t1
        meta["n_snapshots"] = len(snaps)
    meta["total_seconds"] = time.time() - t0
    return {"kind": "seed", "seed": seed, "res": res, "meta": meta,
            "bars_close": s.c[::15].tolist()}


def ablation_job(seed: int, n_days: int, cfg: dict, price: float, counts: dict) -> dict:
    _quiet_worker()
    from validation.checks.bar_tests import t4_vol_volume
    from validation.data import from_sim
    from validation.sim_driver import simulate
    warm = cfg["protocol"]["warmup_days"]
    out = {}
    for label, cnt in (("baseline", None), ("ablated", counts)):
        o = simulate(seed, warm + n_days, price, counts=cnt, invariant_every=0)
        r = t4_vol_volume(from_sim(o.bars, warm), cfg, bootstrap=False)
        out[label] = r["metrics"].get("spearman", float("nan"))
    return {"kind": "ablation", "seed": seed, "res": out}


def t6_job(cfg: dict, symbol: str, price: float) -> dict:
    _quiet_worker()
    from validation.checks import determinism
    from validation.data import load_real
    s = load_real(os.path.join(DATA_DIR, f"{symbol}_minute.csv"))
    return {"kind": "t6",
            "part_a_swarm_off": determinism.part_a(s, cfg, swarm=False),
            "part_a_swarm_on": determinism.part_a(s, cfg, swarm=True),
            "part_b": determinism.part_b(cfg, price)}


# --------------------------------------------------------------------------- #
# aggregation (§2.6, §13)
# --------------------------------------------------------------------------- #
def _scalar_metrics(results: list) -> dict:
    keys = [k for k, v in results[0]["metrics"].items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    from validation.stats_utils import summarize
    return {k: summarize([r["metrics"].get(k, np.nan) for r in results]) for k in keys}


def aggregate(name: str, spec_results: list, real_results: list, calib: dict,
              pass_rate_req: float, cfg: dict) -> dict:
    entry = {}
    if spec_results:
        passes = np.array([r["pass"] for r in spec_results], bool)
        crit_names = spec_results[0]["criteria"].keys()
        entry["pass_rate"] = float(passes.mean()) if crit_names else None
        entry["criteria_pass_rate"] = {c: float(np.mean([bool(r["criteria"].get(c)) for r in spec_results]))
                                       for c in crit_names}
        entry["metrics"] = _scalar_metrics(spec_results)
        entry["n_seeds"] = len(spec_results)
        status = "pass" if crit_names and passes.mean() >= pass_rate_req else "fail"
        if name == "T1_fat_tails":
            med = entry["metrics"]["excess_kurtosis"]["median"]
            gate = np.mean([bool(r.get("gate_pass")) for r in spec_results])
            entry["gate"] = {"criterion": "ci95_lower(excess_kurtosis) > 0", "pass_rate": float(gate)}
            entry["target"] = {"criterion": f"median excess_kurtosis >= {cfg['t1_fat_tails']['target_excess_kurtosis']}",
                               "passed": bool(med >= cfg["t1_fat_tails"]["target_excess_kurtosis"])}
            if not entry["target"]["passed"]:
                status = "fail"
            zf_real = calib.get("tests", {}).get(name, {}).get("metrics", {}).get("zero_return_frac")
            if zf_real and real_results:
                zr = zf_real["calibration"]["median"]
                zs = np.median([r["metrics"]["zero_return_frac"] for r in real_results])
                ok = abs(zs - zr) * 100 <= cfg["t1_fat_tails"]["zero_frac_band_pp"]
                entry["sanity"] = {"criterion": "zero-return fraction within +/-10 pp of real",
                                   "sim_trade_price": float(zs), "real_median": zr, "passed": bool(ok)}
        if name == "T7a_book_walk":
            entry["gate"] = {"criterion": "ci95_upper(delta) < 0.9",
                             "pass_rate": float(np.mean([bool(r["gate_pass"]) for r in spec_results]))}
            entry["target"] = {"criterion": "delta in [0.4, 0.6]",
                               "pass_rate": float(np.mean([bool(r["target_pass"]) for r in spec_results]))}
        if name in BAND_ONLY:
            status = None   # decided by the real band below
        entry["status"] = status
    # Real-data comparison on trade closes.
    tcal = calib.get("tests", {}).get(name)
    if tcal and real_results:
        comp = {}
        for mn in KEY_METRICS.get(name, []):
            c = tcal["metrics"].get(mn)
            if not c or not c.get("band"):
                continue
            lo, hi = c["band"]
            vals = np.array([r["metrics"].get(mn, np.nan) for r in real_results], float)
            inb = (vals >= lo) & (vals <= hi)
            comp[mn] = {"band": c["band"], "real_median": c["calibration"]["median"],
                        "real_holdout_median": c.get("holdout", {}).get("median"),
                        "holdout_in_band": c.get("holdout_in_band"),
                        "sim_median": float(np.nanmedian(vals)), "sim_in_band_rate": float(inb.mean())}
        entry["real_band"] = comp
        if comp:
            ok = all(v["sim_in_band_rate"] >= pass_rate_req for v in comp.values())
            entry["real_band_status"] = "pass" if ok else "fail"
            entry["real_spec_default_pass_rate"] = tcal.get("spec_default_pass_rate_calibration")
            if name in BAND_ONLY:
                entry["status"] = entry["real_band_status"]
                entry["provisional"] = "spec criterion truncated in PDF; judged on real band"
    if name in UNCALIBRATED:
        entry["calibration"] = "uncalibrated: reference data has no order book / trades"
    return entry


CODE_DIRS = ("engine", "agents", "core", "validation")
CODE_FILES = ("worker.py",)


def _code_hash() -> str:
    """SHA-256 over the simulator's source, configs and model files, so a
    report card names the exact code it measured, uncommitted edits
    included. Version control is deliberately not consulted."""
    root = os.path.dirname(HERE)
    paths = [os.path.join(root, f) for f in CODE_FILES]
    for d in CODE_DIRS:
        for dirpath, dirnames, files in os.walk(os.path.join(root, d)):
            dirnames[:] = [x for x in dirnames if x not in ("__pycache__", "reports")]
            paths += [os.path.join(dirpath, f) for f in files
                      if f.endswith((".py", ".yaml", ".json", ".zip", ".pkl"))]
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.relpath(p, root).replace(os.sep, "/").encode())
        with open(p, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODES), default="smoke")
    ap.add_argument("--instrument", default="RELIANCE")
    # Each worker loads torch + both PPO models; 11 workers hit MemoryError
    # on a 5.8 GB machine, 5 is the measured safe default there.
    ap.add_argument("--workers", type=int, default=min(5, max(1, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--seed0", type=int, default=1)
    a = ap.parse_args(argv)
    mode = MODES[a.mode]
    cfg = load_cfg()
    n_days = math.ceil(mode["n_returns"] / RETURNS_PER_DAY)
    cpath = calibration_path(a.instrument, n_days)
    if not os.path.exists(cpath):
        raise SystemExit(f"missing {cpath}: run python -m validation.calibrate --n-days {n_days}")
    with open(cpath) as fh:
        calib = json.load(fh)
    # Start the sim at the instrument's last real close (same NSE tick band).
    from validation.data import load_real
    real = load_real(os.path.join(DATA_DIR, f"{a.instrument}_minute.csv"))
    price = float(real.c[-1])
    seeds = list(range(a.seed0, a.seed0 + mode["seeds"]))
    impact_seeds = set(seeds[:mode["impact_seeds"]])
    t0 = time.time()
    outputs, errors = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {}
        # Longest jobs first: impact seeds, then T6, then the rest.
        for sd in sorted(seeds, key=lambda x: x not in impact_seeds):
            futs[ex.submit(seed_job, sd, mode, cfg, price, sd in impact_seeds,
                           calib["real_profiles"])] = f"seed {sd}"
        futs[ex.submit(t6_job, cfg, a.instrument, price)] = "T6"
        abl_days = math.ceil(MODES["smoke"]["n_returns"] / RETURNS_PER_DAY)
        for sd in seeds[:mode["ablation_seeds"]]:
            futs[ex.submit(ablation_job, sd, abl_days, cfg, price,
                           {"mm": 15, "whale": 0, "retail": 80, "redteam": 0})] = f"ablation {sd}"
        for f in as_completed(futs):
            try:
                outputs.append(f.result())
                print(f"done {futs[f]} at {time.time() - t0:.0f}s", flush=True)
            except Exception:
                errors.append({"job": futs[f], "traceback": traceback.format_exc()})
                print(f"ERROR {futs[f]}\n{errors[-1]['traceback']}", flush=True)
    from validation.report import build_report
    report = build_report(outputs, errors, calib, cfg, mode, a, price, n_days,
                          _code_hash(), time.time() - t0)
    os.makedirs(REPORT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = os.path.join(REPORT_DIR, f"report_card_{a.mode}_{a.instrument}_{stamp}.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1)
    raw = os.path.join(REPORT_DIR, f"raw_{a.mode}_{a.instrument}_{stamp}.json")
    with open(raw, "w") as fh:
        from validation.stats_utils import round_sig
        json.dump(round_sig([o for o in outputs]), fh)
    print(f"report: {path}\nraw: {raw}\noverall: {report['overall']}")


if __name__ == "__main__":
    main()
