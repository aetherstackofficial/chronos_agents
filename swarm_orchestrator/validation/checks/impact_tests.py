"""Impact harness: T7a book walk, T7b metaorder, T8 monotonic, A6 decay,
A7 spread/depth recovery (spec §10-12).

All trials start from a snapshot taken right after a physics tick and before
the agents act (the moment the swarm receives the tick). Every trial
restores the snapshot first (spec T7 setup step 3), including RNG states,
so trials never contaminate each other.

Engine-forced deviations (reported, not hidden):
* TWAP children are one per sim-minute, not "every few seconds": the engine
  has no sub-minute clock.
* A market order larger than the visible opposite depth is partially filled
  and the remainder discarded; if it empties a side the mid is undefined, so
  that trial is counted as ``exhausted`` and excluded from impact means.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

PROBE = "STRAT_PROBE"
SIDES = (("BUY", 1), ("SELL", -1))


def _signed_log(mid1, mid0, sign) -> float:
    if mid1 is None or mid0 is None:
        return float("inf")
    return sign * float(np.log(mid1 / mid0))


# --------------------------------------------------------------------------- #
# T8 monotonic impact + T7a book walk (same single-order measurements)
# --------------------------------------------------------------------------- #
def single_order_trials(sim, snaps: List[dict], sizes: List[int]) -> dict:
    """impact[snap, side, q], px[...] and conservation checks."""
    S, Q = len(snaps), len(sizes)
    imp = np.full((S, 2, Q), np.nan)
    px = np.full((S, 2, Q), np.nan)
    cons_fail = book_fail = skipped = 0
    for i, snap in enumerate(snaps):
        for j, (side, sign) in enumerate(SIDES):
            for k, q in enumerate(sizes):
                sim.restore(snap)
                mid0 = sim.mid()
                if mid0 is None:
                    skipped += 1
                    break
                before = sim.visible_qty(side)
                f = sim.market_order(PROBE, side, q)
                if f.qty != min(q, f.available):
                    cons_fail += 1
                if sim.visible_qty(side) != before - f.qty:
                    book_fail += 1
                imp[i, j, k] = _signed_log(sim.mid(), mid0, sign)
                px[i, j, k] = sign * f.avg_price if f.qty else np.nan
    return {"impact": imp, "exec_px": px, "conservation_failures": cons_fail,
            "book_change_failures": book_fail, "skipped_one_sided": skipped,
            "sizes": list(sizes)}


def t8_monotonic(tr: dict, cfg: dict) -> dict:
    c = cfg["t8_monotonic"]
    sizes = tr["sizes"]
    sel = [sizes.index(q) for q in c["sizes"]]
    imp, px = tr["impact"][:, :, sel], tr["exec_px"][:, :, sel]
    valid = np.isfinite(px[:, :, 0]) | np.isinf(imp[:, :, 0])
    d_imp = imp[:, :, 1:] - imp[:, :, :-1]
    d_px = px[:, :, 1:] - px[:, :, :-1]
    with np.errstate(invalid="ignore"):
        v_imp = int(np.sum((d_imp < -1e-12) & ~np.isnan(d_imp)))
        v_px = int(np.sum((d_px < -1e-9) & ~np.isnan(d_px)))
    checks = int(np.sum(~np.isnan(d_imp)))
    fin = np.where(np.isfinite(imp), imp, np.nan)
    buy, sell = np.nanmean(fin[:, 0, :], axis=0), np.nanmean(fin[:, 1, :], axis=0)
    rel = np.abs(buy - sell) / np.maximum(np.abs(buy + sell) / 2, 1e-12)
    m = {"checks": float(checks), "mid_violations": float(v_imp),
         "exec_px_violations": float(v_px),
         "conservation_failures": float(tr["conservation_failures"]),
         "book_change_failures": float(tr["book_change_failures"]),
         "frac_exhausted_at_max_Q": float(np.mean(np.isinf(imp[:, :, -1]))),
         "max_buy_sell_rel_diff": float(np.nanmax(rel)),
         "n_valid_snapshot_sides": float(valid.sum())}
    crit = {"zero_mid_violations": v_imp == 0, "zero_exec_px_violations": v_px == 0,
            "conservation": tr["conservation_failures"] == 0 and tr["book_change_failures"] == 0,
            "symmetry_lt_20pct": bool(np.nanmax(rel) < c["symmetry_max_rel_diff"])}
    return {"metrics": m, "criteria": crit, "pass": all(crit.values()),
            "extra": {"buy_mean_impact": buy.tolist(), "sell_mean_impact": sell.tolist(),
                      "sizes": c["sizes"]}}


def _fit_exponent(x: np.ndarray, trial_imp: np.ndarray, n_boot: int, seed: int) -> dict:
    """Fit log mean(I) = a + delta log x over x with positive finite means.

    trial_imp: (n_trials, len(x)) signed impacts, inf = exhausted (dropped).
    """
    fin = np.where(np.isfinite(trial_imp), trial_imp, np.nan)

    def fit(rows):
        mean = np.nanmean(rows, axis=0)
        ok = np.isfinite(mean) & (mean > 0)
        if ok.sum() < 3:
            return np.nan, np.nan, int(ok.sum())
        lx, ly = np.log(x[ok]), np.log(mean[ok])
        slope, icpt = np.polyfit(lx, ly, 1)
        pred = icpt + slope * lx
        r2 = 1 - np.sum((ly - pred) ** 2) / max(np.sum((ly - ly.mean()) ** 2), 1e-300)
        return float(slope), float(r2), int(ok.sum())

    delta, r2, n_used = fit(fin)
    rng = np.random.default_rng(seed)
    boots = [fit(fin[rng.integers(0, len(fin), len(fin))])[0] for _ in range(n_boot)]
    boots = np.asarray([b for b in boots if np.isfinite(b)])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan))
    return {"delta": delta, "r2": r2, "delta_ci95_lo": float(lo),
            "delta_ci95_hi": float(hi), "n_x_used": float(n_used),
            "n_x_dropped": float(len(x) - n_used),
            "mean_impact": np.nanmean(fin, axis=0).tolist()}


def t7a_book_walk(tr: dict, cfg: dict, seed: int = 0) -> dict:
    c = cfg["t7_sqrt_impact"]
    lo, hi, n = c["book_walk_Q"]
    grid = [int(round(q)) for q in np.geomspace(lo, hi, n)]
    sel = [tr["sizes"].index(q) for q in grid]
    imp = tr["impact"][:, :, sel].reshape(-1, len(grid))   # both sides as trials
    f = _fit_exponent(np.asarray(grid, float), imp, 1000, seed)
    m = {k: v for k, v in f.items() if k != "mean_impact"}
    m["frac_exhausted"] = float(np.mean(np.isinf(imp)))
    d0, d1 = c["delta_range"]
    crit = {"gate_ci_upper_lt_0.9": bool(f["delta_ci95_hi"] < c["book_walk_gate_ci_upper"])}
    target = bool(np.isfinite(f["delta"]) and d0 <= f["delta"] <= d1)
    return {"metrics": m, "criteria": crit, "pass": all(crit.values()),
            "gate_pass": crit["gate_ci_upper_lt_0.9"], "target_pass": target,
            "diagnostic_only": True,
            "extra": {"Q": grid, "mean_impact": f["mean_impact"]}}


# --------------------------------------------------------------------------- #
# T7b metaorders + A6 decay
# --------------------------------------------------------------------------- #
def metaorder_trials(sim, snaps: List[dict], V_D: float, cfg: dict) -> dict:
    c = cfg["t7_sqrt_impact"]
    qv_lo, qv_hi, nq = c["metaorder_QV"]
    qv = np.geomspace(qv_lo, qv_hi, nq)
    Ts = c["metaorder_T"]
    hs = cfg["a6_impact_decay"]["horizons_after"]
    S = len(snaps)
    out = {}
    for T in Ts:
        # Decay horizons: A6 uses T=30 (all horizons); T=120 only needs +30.
        horizons = hs if T == min(Ts) else [h for h in hs if h <= 30]
        I = np.full((S, 2, nq), np.nan)
        D = np.full((S, 2, nq, len(horizons)), np.nan)
        for i, snap in enumerate(snaps):
            for j, (side, sign) in enumerate(SIDES):
                for k, x in enumerate(qv):
                    Q = max(T, int(round(x * V_D)))
                    child = [Q // T + (1 if n < Q % T else 0) for n in range(T)]
                    sim.restore(snap)
                    mid0 = sim.mid()
                    if mid0 is None:
                        continue
                    mid_end = None
                    for n, q in enumerate(child):
                        sim.market_order(PROBE, side, q)
                        if n == T - 1:
                            mid_end = sim.mid()
                        sim.agents_act(sim.last_payload)
                        sim.physics()
                    I[i, j, k] = _signed_log(mid_end, mid0, sign)
                    done = 1
                    for hi_, h in enumerate(horizons):
                        while done < h:
                            sim.agents_act(sim.last_payload)
                            sim.physics()
                            done += 1
                        D[i, j, k, hi_] = _signed_log(sim.mid(), mid0, sign)
        out[T] = {"I": I, "D": D, "horizons": horizons}
    return {"QV": qv.tolist(), "by_T": out}


def t7b_metaorder(mt: dict, sigma_D: float, cfg: dict, seed: int = 0) -> dict:
    c = cfg["t7_sqrt_impact"]
    qv = np.asarray(mt["QV"])
    m, crit, extra = {}, {}, {}
    Ys = {}
    for T, d in mt["by_T"].items():
        I = d["I"]
        trials = I.reshape(-1, len(qv))
        f = _fit_exponent(qv, trials, 1000, seed)
        fin = np.where(np.isfinite(I), I, np.nan)
        mean_I = np.nanmean(fin.reshape(-1, len(qv)), axis=0)
        Y = float(np.nanmedian(mean_I / (sigma_D * np.sqrt(qv))))
        Ys[T] = Y
        buy = np.nanmean(fin[:, 0, :], axis=0)
        sell = np.nanmean(fin[:, 1, :], axis=0)
        for key in ("delta", "delta_ci95_hi", "r2", "n_x_dropped"):
            m[f"{key}_T{T}"] = f[key]
        m[f"Y_T{T}"] = Y
        m[f"frac_exhausted_T{T}"] = float(np.mean(np.isinf(I)))
        d0, d1 = c["delta_range"]
        crit[f"delta_in_range_T{T}"] = bool(np.isfinite(f["delta"]) and d0 <= f["delta"] <= d1)
        crit[f"delta_ci_upper_lt_0.8_T{T}"] = bool(f["delta_ci95_hi"] < c["delta_ci_upper_max"])
        crit[f"r2_ge_0.85_T{T}"] = bool(np.isfinite(f["r2"]) and f["r2"] >= c["r2_min"])
        crit[f"Y_in_band_T{T}"] = bool(c["Y_range"][0] <= Y <= c["Y_range"][1])
        crit[f"sign_every_Q_T{T}"] = bool(np.all(buy > 0) and np.all(sell > 0))
        extra[f"mean_impact_T{T}"] = mean_I.tolist()
    T0, T1 = sorted(Ys)
    ratio = Ys[T0] / Ys[T1] if Ys[T1] else np.inf
    m["Y_ratio_T30_T120"] = float(ratio)
    crit["horizon_Y_within_1.5x"] = bool(np.isfinite(ratio) and ratio > 0 and
                                        1 / c["horizon_Y_ratio_max"] <= ratio <= c["horizon_Y_ratio_max"])
    extra["QV"] = mt["QV"]
    return {"metrics": m, "criteria": crit, "pass": all(crit.values()), "extra": extra}


def a6_decay(mt: dict, cfg: dict) -> dict:
    T = min(mt["by_T"])
    d = mt["by_T"][T]
    fin = lambda a: np.where(np.isfinite(a), a, np.nan)
    I = np.nanmean(fin(d["I"]).reshape(-1, d["I"].shape[-1]), axis=0)
    m, ratios = {}, []
    for hi_, h in enumerate(d["horizons"]):
        Dh = np.nanmean(fin(d["D"][..., hi_]).reshape(-1, d["I"].shape[-1]), axis=0)
        ok = I > 0
        r = float(np.nanmedian(Dh[ok] / I[ok])) if ok.any() else np.nan
        m[f"ratio_T_plus_{h}"] = r
        ratios.append(r)
    ratios = np.asarray(ratios)
    crit = {"between_0_and_1": bool(np.all((ratios > 0.0) & (ratios < 1.0))),
            "decreasing": bool(np.all(np.diff(ratios) <= 0)),
            "not_exactly_0_or_1": bool(np.all((np.abs(ratios) > 0.02) & (np.abs(ratios - 1) > 0.02)))}
    return {"metrics": m, "criteria": crit, "pass": all(crit.values())}


# --------------------------------------------------------------------------- #
# A7 spread and depth recovery
# --------------------------------------------------------------------------- #
def _depth_near(sim, side: str, ticks: int = 10) -> int:
    eng = sim.engine
    bids, asks = eng.book.depth(10 ** 6)
    levels = asks if side == "BUY" else bids
    if not levels:
        return 0
    best = levels[0][0]
    return int(sum(q for p, q in levels if abs(p - best) <= ticks * eng.tick + 1e-9))


def _spread_ticks(sim):
    eng = sim.engine
    bb, ba = eng.book.best_bid(), eng.book.best_ask()
    return None if bb is None or ba is None else round((ba - bb) / eng.tick)


def a7_recovery(sim, snaps: List[dict], cfg: dict) -> dict:
    c = cfg["a7_recovery"]
    rec, never = [], 0
    for snap in snaps:
        sim.restore(snap)
        sp0, dp0 = _spread_ticks(sim), _depth_near(sim, "BUY")
        _, asks = sim.engine.book.depth(c["sweep_levels"])
        if sp0 is None or len(asks) < c["sweep_levels"]:
            continue
        sim.market_order(PROBE, "BUY", int(sum(q for _, q in asks)))
        t = None
        for minute in range(1, c["max_wait_minutes"] + 1):
            sim.agents_act(sim.last_payload)
            sim.physics()
            sp, dp = _spread_ticks(sim), _depth_near(sim, "BUY")
            if sp is not None and sp <= sp0 / c["recovery_fraction"] and \
                    dp >= c["recovery_fraction"] * dp0:
                t = minute
                break
        if t is None:
            never += 1
        else:
            rec.append(t)
    n = len(rec) + never
    m = {"n_trials": float(n),
         "recovery_minutes_median": float(np.median(rec)) if rec else np.nan,
         "recovery_minutes_p95": float(np.percentile(rec, 95)) if rec else np.nan,
         "frac_never_recovered": float(never / n) if n else np.nan}
    # "Within the real instrument's range" needs order-book data: not gated.
    crit = {"no_permanent_hole": n > 0 and never == 0}
    return {"metrics": m, "criteria": crit, "pass": all(crit.values())}
