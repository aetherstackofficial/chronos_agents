"""JSON report card (spec §13)."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from validation.calibrate import CFG_PATH
from validation.stats_utils import round_sig

ORDER = ["T6_determinism", "T0_invariants", "T1_fat_tails", "T2_vol_clustering",
         "T3_no_return_autocorr", "T4_vol_volume", "A1_aggregational_gaussianity",
         "A3_volume_clustering", "A4_extreme_intermittency", "A5_intraday_seasonality",
         "T5_book_shape", "T7_sqrt_impact", "T7a_book_walk", "T8_monotonic_impact",
         "A6_impact_decay", "A7_spread_depth_recovery", "A2_order_flow_sign_memory",
         "A8_no_free_lunch", "A9_leverage_asymmetry"]


def build_report(outputs, errors, calib, cfg, mode, args, price, n_days, code_hash,
                 seconds) -> dict:
    from validation.runner import aggregate
    seeds = sorted([o for o in outputs if o["kind"] == "seed"], key=lambda o: o["seed"])
    pr = cfg["pass_rate"]
    tests = {}
    names = set()
    for o in seeds:
        names |= set(o["res"]["spec"])
    for name in names:
        spec_r = [o["res"]["spec"][name] for o in seeds if name in o["res"]["spec"]]
        real_r = [o["res"]["real_sym"][name] for o in seeds if name in o["res"]["real_sym"]]
        tests[name] = aggregate(name, spec_r, real_r, calib, pr, cfg)
        tests[name]["seeds"] = [o["seed"] for o in seeds if name in o["res"]["spec"]]

    # T4 circularity (ablation: whales off).
    abl = [o for o in outputs if o["kind"] == "ablation"]
    if "T4_vol_volume" in tests and abl:
        b = np.median([o["res"]["baseline"] for o in abl])
        a = np.median([o["res"]["ablated"] for o in abl])
        drop = 1 - a / b if b else float("nan")
        ok = bool(drop >= cfg["t4_vol_volume"]["ablation_drop_min"])
        tests["T4_vol_volume"]["circularity"] = {
            "ablation": "whales off (mm 15, retail 80)", "seeds": [o["seed"] for o in abl],
            "spearman_baseline_median": b, "spearman_ablated_median": a,
            "relative_drop": drop, "criterion": "drop >= 0.5", "passed": ok}
        if not ok:
            tests["T4_vol_volume"]["status"] = "fail"

    t6 = [o for o in outputs if o["kind"] == "t6"]
    if t6:
        t = t6[0]
        ok = t["part_a_swarm_off"]["pass"] and t["part_b"]["pass"]
        tests["T6_determinism"] = {
            "status": "pass" if ok else "fail",
            "pass_rate": 1.0 if ok else 0.0,
            "part_a_replay_fidelity_swarm_off": t["part_a_swarm_off"],
            "part_a_replay_fidelity_swarm_on_informational": t["part_a_swarm_on"],
            "part_b_seeded_determinism": t["part_b"],
            "notes": ("Part A gates on swarm-off replay (spec: no strategy trading). "
                      "Per spec §14.2 a T6 failure means all later results are "
                      "conditional; the suite still runs so the baseline is complete.")}
    for e in errors:
        key = "T6_determinism" if e["job"] == "T6" else e["job"]
        tests.setdefault(key, {})
        tests[key].update({"status": "error", "error": e["traceback"][-2000:]})
    for name in ORDER:
        tests.setdefault(name, {"status": "error", "reason": "no result produced"})

    statuses = [v.get("status") for v in tests.values()]
    overall = {"status": "pass" if all(s == "pass" for s in statuses) else "fail",
               "passed": statuses.count("pass"), "failed": statuses.count("fail"),
               "skipped": statuses.count("skipped"), "errors": statuses.count("error")}
    with open(CFG_PATH, "rb") as fh:
        cfg_bytes = fh.read()
    config_hash = hashlib.sha256(cfg_bytes + json.dumps(mode, sort_keys=True).encode()
                                 + repr(price).encode()).hexdigest()[:16]
    real_ref = {k: {"spec_default_pass_rate_calibration": v.get("spec_default_pass_rate_calibration"),
                    "spec_default_pass_rate_holdout": v.get("spec_default_pass_rate_holdout")}
                for k, v in calib["tests"].items()}
    meta = [o["meta"] for o in seeds]
    report = {
        "spec_version": cfg["spec_version"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "simulator": {"code_hash": code_hash,
                      "config_hash": config_hash, "instrument": f"SIM~{args.instrument}",
                      "start_price": price, "bar_interval_s": 60,
                      "swarm": "live default: 15 MM (PPO), 4 whale (PPO), 80 retail (heuristic), SIM_LP; no news/oracle",
                      "harness": ("validation/sim_driver.py: in-process client; swarm built as worker.py "
                                  "--seed builds it (agents/swarm_factory.py: tick clock, per-agent seeded RNGs)")},
        "run": {"mode": args.mode, "n_seeds": len(seeds), "seeds": [o["seed"] for o in seeds],
                "n_returns_per_seed": int(np.median([m["N_returns"] for m in meta])) if meta else 0,
                "sessions_per_seed": n_days, "warmup_days": cfg["protocol"]["warmup_days"],
                "impact_seeds": [m["seed"] for m in meta if "impact_seconds" in m],
                "impact_snapshots_single_order": mode["n_snap_single"],
                "impact_snapshots_metaorder": mode["n_snap_meta"], "wall_seconds": seconds},
        "calibration_data": {k: calib[k] for k in ("instrument", "source", "first_date", "last_date",
                                                   "split_date", "window_days", "step_days",
                                                   "n_windows", "price_series", "sigma_estimator",
                                                   "cleaning")},
        "overall": overall,
        "tests": {k: tests[k] for k in ORDER if k in tests},
        "real_nse_reference": real_ref,
        "seed_meta": meta,
    }
    return round_sig(report)
