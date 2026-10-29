"""T5 order-book shape and the book-invariant gate (spec §8, §14.2 step 2)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def t5_book_shape(snaps: np.ndarray, spreads_ticks: np.ndarray, cfg: dict,
                  seed: int = 0) -> dict:
    """``snaps``: (S, 2, L) volume at d = 1..L ticks from the touch, per side.

    Distance-from-mid profiles are not computed: with a 1-minute engine the
    snapshot times are tick boundaries, and the spread is reported instead.
    """
    c = cfg["t5_book_shape"]
    L, fr = c["L"], c["fit_range"]
    m = {"n_snapshots": float(len(snaps))}
    if snaps is None or len(snaps) == 0:
        return {"metrics": m, "criteria": {"has_snapshots": False}, "pass": False}
    q = snaps.mean(axis=0).mean(axis=0)
    d = np.arange(1, L + 1)
    qmin = q.min()
    m["max_min_ratio"] = float(q.max() / qmin) if qmin > 0 else float("inf")
    m["frac_levels_empty"] = float(np.mean(snaps == 0))
    m["near_spearman_d1_5"] = float(stats.spearmanr(q[:5], d[:5])[0])
    m["peak_distance_ticks"] = float(np.argmax(q) + 1)

    def gamma_of(prof):
        C = np.cumsum(prof)
        return np.polyfit(np.log(d[:fr]), np.log(C[:fr]), 1)[0]

    m["gamma"] = float(gamma_of(q))
    rng = np.random.default_rng(seed)
    gs = [gamma_of(snaps[rng.integers(0, len(snaps), len(snaps))].mean(axis=0).mean(axis=0))
          for _ in range(500)]
    m["gamma_ci95_lo"], m["gamma_ci95_hi"] = (float(x) for x in np.percentile(gs, [2.5, 97.5]))
    bid, ask = snaps[:, 0, :].sum(axis=1), snaps[:, 1, :].sum(axis=1)
    m["bid_ask_depth_ratio"] = float(bid.mean() / ask.mean()) if ask.mean() else float("inf")
    sp = spreads_ticks[np.isfinite(spreads_ticks)]
    m["spread_mean_ticks"] = float(sp.mean())
    m["spread_median_ticks"] = float(np.median(sp))
    m["spread_frac_one_tick"] = float(np.mean(sp == 1))
    crit = {"min_snapshots": len(snaps) >= c["min_snapshots"],
            "non_flat_max_min_ge_1.5": m["max_min_ratio"] >= c["max_min_ratio"],
            "near_mid_rising_d1_5": m["near_spearman_d1_5"] > 0,
            "convex_gamma_ci_lo_gt_1": m["gamma_ci95_lo"] > c["gamma_min"],
            "bid_ask_ratio_0.5_2": 0.5 <= m["bid_ask_depth_ratio"] <= 2.0}
    # Spread criterion ("within 1 tick of the real instrument") needs real
    # quotes, which the reference data lacks: reported, not gated.
    return {"metrics": m, "criteria": crit, "pass": all(crit.values()),
            "gate_pass": None, "target_pass": None,
            "extra": {"q": q.tolist()}}


def invariants_result(inv: dict) -> dict:
    viol = {k: v for k, v in inv.items() if k != "checked"}
    crit = {f"zero_{k}": v == 0 for k, v in viol.items()}
    crit["checked_some"] = inv.get("checked", 0) > 0
    return {"metrics": {k: float(v) for k, v in inv.items()}, "criteria": crit,
            "pass": all(crit.values())}
