"""Return-process tests on 1-minute bars: T1-T4, A1, A3, A4, A5, A9.

Every function takes a BarSeries and returns
``{"metrics": {...scalars...}, "criteria": {name: bool}, "pass": bool}``
(plus ``extra`` for non-scalar diagnostics). They are pure functions of the
bars, so the same code scores real NSE windows and simulator seeds.

``price`` selects the series: "mid" (spec default for the sim) or "c"
(trade-price close; the only option for real bars).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats

from validation.data import (SESSION_BARS, BarSeries, deseasonalize,
                             deseasonalized_returns, intraday_returns,
                             parkinson_sigma)
from validation.stats_utils import (acf, arch_lm_p, block_bootstrap,
                                    excess_kurtosis, first_lag_below,
                                    ljung_box_p, power_law_fit)


def _result(metrics: dict, criteria: dict, extra: dict = None, **flags) -> dict:
    out = {"metrics": metrics, "criteria": criteria,
           "pass": bool(all(criteria.values())) if criteria else False}
    out.update(flags)
    if extra:
        out["extra"] = extra
    return out


# --------------------------------------------------------------------------- #
# T1 fat tails
# --------------------------------------------------------------------------- #
def t1_fat_tails(s: BarSeries, cfg: dict, price: str = "mid",
                 bootstrap: bool = True, seed: int = 0) -> dict:
    c = cfg["t1_fat_tails"]
    r, idx = intraday_returns(s, price)
    k = excess_kurtosis(r)
    m = {"excess_kurtosis": k, "N": float(len(r)),
         "zero_return_frac": float(np.mean(r == 0)),
         "jb_p": float(stats.jarque_bera(r).pvalue)}
    rt = deseasonalized_returns(r, s.minute[idx])
    m["excess_kurtosis_deseasonalized"] = excess_kurtosis(rt)
    z = np.abs(r - r.mean()) / r.std()
    m["q99_abs_z"] = float(np.quantile(z, 0.99))     # Gaussian 2.58
    m["q999_abs_z"] = float(np.quantile(z, 0.999))   # Gaussian 3.29
    crit = {}
    if bootstrap:
        p = cfg["protocol"]
        ks = block_bootstrap(lambda ix: excess_kurtosis(r[ix]), len(r),
                             p["bootstrap_block"], p["bootstrap_resamples"], seed)
        m["ci95_lo"], m["ci95_hi"] = (float(x) for x in np.percentile(ks, [2.5, 97.5]))
        crit["gate_ci95_lower_gt_0"] = m["ci95_lo"] > 0
    crit["target_excess_kurtosis"] = k >= c["target_excess_kurtosis"]
    return _result(m, crit, gate_pass=crit.get("gate_ci95_lower_gt_0"),
                   target_pass=crit["target_excess_kurtosis"])


# --------------------------------------------------------------------------- #
# T2 volatility clustering
# --------------------------------------------------------------------------- #
def _clustering(a: np.ndarray, max_lag: int, cfg2: dict) -> tuple:
    rho = acf(a, max_lag)
    fit = power_law_fit(rho)
    noise = 1.96 / np.sqrt(len(a))
    m = {"rho_1": float(rho[0]), "rho_10": float(rho[9]), "rho_30": float(rho[29]),
         "rho_100": float(rho[max_lag - 1]), "beta": fit["beta"],
         "beta_se": fit["beta_se"], "ljung_box_p": ljung_box_p(a, 10),
         "noise_level": float(noise)}
    lag_noise = first_lag_below(rho, noise)
    m["first_lag_below_noise"] = float(lag_noise) if lag_noise else float(max_lag + 1)
    lo, hi = cfg2["beta_range"]
    crit = {"ljung_box_p_lt_0.01": m["ljung_box_p"] < 0.01,
            "rho_1_10_all_positive": bool(np.all(rho[:10] > 0)),
            "rho_1_ge_min": m["rho_1"] >= cfg2.get("rho1_min", 0.05),
            "beta_in_range": bool(np.isfinite(fit["beta"]) and lo <= fit["beta"] <= hi),
            "rho_30_above_noise": m["rho_30"] > noise}
    return m, crit, rho


def t2_vol_clustering(s: BarSeries, cfg: dict, price: str = "mid", **_) -> dict:
    c = cfg["t2_vol_clustering"]
    r, idx = intraday_returns(s, price)
    rt = deseasonalized_returns(r, s.minute[idx])
    m, crit, rho = _clustering(np.abs(rt), c["max_lag"], c)
    m["arch_lm_p"] = arch_lm_p(rt, 10)
    return _result(m, crit, extra={"rho": rho.tolist()})


# --------------------------------------------------------------------------- #
# T3 no return autocorrelation
# --------------------------------------------------------------------------- #
def t3_no_autocorr(s: BarSeries, cfg: dict, price: str = "mid", **_) -> dict:
    c = cfg["t3_no_autocorr"]
    r, _ = intraday_returns(s, price)
    rho = acf(r, c["lags"])
    n = len(r)
    m = {f"rho_{k + 1}": float(rho[k]) for k in range(c["lags"])}
    m.update({"ljung_box_p": ljung_box_p(r, c["lags"]),
              "max_abs_rho": float(np.max(np.abs(rho))),
              "bonferroni_bound": float(2.576 / np.sqrt(n)),
              "lags_outside_bonferroni": float(np.sum(np.abs(rho) > 2.576 / np.sqrt(n)))})
    crit = {"ljung_box_p_gt_0.01": m["ljung_box_p"] > c["ljung_box_p_min"],
            "max_abs_rho_le_0.03": m["max_abs_rho"] <= c["max_abs_rho"]}
    return _result(m, crit)


# --------------------------------------------------------------------------- #
# T4 volume-volatility coupling
# --------------------------------------------------------------------------- #
def _vol_volume_series(s: BarSeries):
    # Same bar set as the return tests: first bar of each day excluded.
    first = np.r_[True, s.day[1:] != s.day[:-1]]
    keep = ~first
    v = deseasonalize(s.v[keep], s.minute[keep])
    sg = deseasonalize(parkinson_sigma(s)[keep], s.minute[keep])
    return v, sg


def t4_vol_volume(s: BarSeries, cfg: dict, bootstrap: bool = True, seed: int = 0,
                  **_) -> dict:
    c = cfg["t4_vol_volume"]
    v_all, s_all = _vol_volume_series(s)
    ok = (v_all > 0) & (s_all > 0)
    v, sg = v_all[ok], s_all[ok]
    m = {"frac_bars_used": float(ok.mean())}
    if len(v) < 1000:
        return _result({**m, "pearson": np.nan, "spearman": np.nan},
                       {"enough_bars_with_v_and_sigma_gt_0": False})
    m["pearson"] = float(np.corrcoef(v, sg)[0, 1])
    rv, rs = stats.rankdata(v), stats.rankdata(sg)
    m["spearman"] = float(np.corrcoef(rv, rs)[0, 1])
    lv, ls = np.log(v), np.log(sg)
    m["log_pearson"] = float(np.corrcoef(lv, ls)[0, 1])
    # Lead-lag on the full aligned series (zeros kept so lags stay aligned).
    ll = {}
    for j in range(-5, 6):
        a = v_all[max(0, -j):len(v_all) - max(0, j)]
        b = s_all[max(0, j):len(s_all) - max(0, -j)]
        ll[j] = float(np.corrcoef(a, b)[0, 1])
    m["leadlag_0"] = ll[0]
    m["leadlag_m1"], m["leadlag_p1"] = ll[-1], ll[1]
    crit = {}
    if bootstrap:
        p = cfg["protocol"]
        n = len(v)
        # Spearman CI resamples the precomputed ranks (no re-ranking): an
        # approximation that keeps 1,000 resamples affordable at N=1e5.
        sp = block_bootstrap(lambda ix: np.corrcoef(rv[ix], rs[ix])[0, 1], n,
                             p["bootstrap_block"], p["bootstrap_resamples"], seed)
        lg = block_bootstrap(lambda ix: np.corrcoef(lv[ix], ls[ix])[0, 1], n,
                             p["bootstrap_block"], p["bootstrap_resamples"], seed + 1)
        m["spearman_ci95_lo"] = float(np.percentile(sp, 2.5))
        m["log_pearson_ci95_lo"] = float(np.percentile(lg, 2.5))
        crit["spearman_ci_lower_gt_0"] = m["spearman_ci95_lo"] > 0
        crit["log_pearson_ci_lower_gt_0"] = m["log_pearson_ci95_lo"] > 0
    crit.update({
        "spearman_ge_min": m["spearman"] >= c["rho_min"],
        "pearson_ge_min": m["pearson"] >= c["rho_min"],
        "log_pearson_gt_0": m["log_pearson"] > 0,
        "leadlag_peak_at_0": max(ll, key=ll.get) == 0,
        "leadlag_pm1_positive": ll[-1] > 0 and ll[1] > 0})
    return _result(m, crit, extra={"leadlag": ll})


# --------------------------------------------------------------------------- #
# A1 aggregational Gaussianity
# --------------------------------------------------------------------------- #
def a1_aggregation(s: BarSeries, cfg: dict, price: str = "mid", **_) -> dict:
    c = cfg["a1_aggregation"]
    r, idx = intraday_returns(s, price)
    day = s.day[idx]
    pos = s.minute[idx] - 1          # 0..373 within the day's return series
    m = {}
    ks = []
    for tau in c["scales_min"]:
        if tau == 1:
            agg = r
        else:
            # Non-overlapping within-day blocks; overnight never included.
            key = day * 1000 + pos // tau
            agg = np.bincount(np.unique(key, return_inverse=True)[1], weights=r)
        kv = excess_kurtosis(agg) if len(agg) > 20 else np.nan
        m[f"kurt_{tau}m"] = kv
        ks.append(kv)
    ks = np.asarray(ks)
    k1, k60, kday = ks[0], m.get("kurt_60m"), ks[-1]
    m["daily_over_1min"] = float(kday / k1) if k1 else np.nan
    m["n_days"] = float(len(np.unique(day)))
    rho_scale = stats.spearmanr(c["scales_min"], ks)[0]
    m["spearman_scale_kurtosis"] = float(rho_scale)
    crit = {"falls_with_scale": bool(np.all(ks[1:] < k1) and rho_scale < 0),
            "daily_le_half_1min": bool(kday <= c["daily_max_fraction_of_1min"] * k1),
            "kurt_60m_gt_0": bool(k60 is not None and k60 > 0)}
    return _result(m, crit)


# --------------------------------------------------------------------------- #
# A3 volume clustering
# --------------------------------------------------------------------------- #
def a3_volume_clustering(s: BarSeries, cfg: dict, **_) -> dict:
    c = cfg["a3_volume_clustering"]
    first = np.r_[True, s.day[1:] != s.day[:-1]]
    v = deseasonalize(s.v[~first], s.minute[~first])
    cfg2 = {"beta_range": c["beta_range"], "rho1_min": cfg["t2_vol_clustering"]["rho1_min"]}
    m, crit, rho = _clustering(v, c["max_lag"], cfg2)
    return _result(m, crit, extra={"rho": rho.tolist()})


# --------------------------------------------------------------------------- #
# A4 extreme-event intermittency  (criterion PROVISIONAL: PDF row truncated)
# --------------------------------------------------------------------------- #
def a4_intermittency(s: BarSeries, cfg: dict, price: str = "mid", **_) -> dict:
    c = cfg["a4_intermittency"]
    r, idx = intraday_returns(s, price)
    a = np.abs(deseasonalized_returns(r, s.minute[idx]))
    thr = np.quantile(a, 1 - c["top_fraction"])
    t = np.flatnonzero(a > thr)
    if len(t) < 10:
        return _result({"n_events": float(len(t))}, {"enough_events": False})
    gaps = np.diff(t).astype(float)
    mean = gaps.mean()
    m = {"n_events": float(len(t)), "interarrival_mean": float(mean),
         "interarrival_cv": float(gaps.std() / mean),
         "frac_gap_le_5": float(np.mean(gaps <= 5)),
         "frac_gap_le_5_poisson": float(1 - np.exp(-5 / mean))}
    crit = {"cv_gt_min_PROVISIONAL": m["interarrival_cv"] > c["cv_min"]}
    return _result(m, crit)


# --------------------------------------------------------------------------- #
# A5 intraday seasonality  (criterion PROVISIONAL: judged on the real band)
# --------------------------------------------------------------------------- #
def intraday_profiles(s: BarSeries, price: str = "c") -> Dict[str, np.ndarray]:
    r, idx = intraday_returns(s, price)
    mins = s.minute[idx]
    cnt = np.bincount(mins, minlength=SESSION_BARS)
    absr = np.bincount(mins, weights=np.abs(r), minlength=SESSION_BARS)
    absr = np.divide(absr, cnt, out=np.zeros(SESSION_BARS), where=cnt > 0)
    vc = np.bincount(s.minute, minlength=SESSION_BARS)
    vol = np.bincount(s.minute, weights=s.v, minlength=SESSION_BARS) / np.maximum(vc, 1)
    return {"abs_ret": absr / absr[1:].mean(), "volume": vol / vol.mean()}


def a5_seasonality(s: BarSeries, cfg: dict, price: str = "mid", **_) -> dict:
    prof = intraday_profiles(s, price)
    m = {}
    for name, p in prof.items():
        mid = p[150:225].mean()                       # ~11:45-13:00 lull
        m[f"{name}_open_ratio"] = float(p[1:16].mean() / mid) if mid else np.nan
        m[f"{name}_close_ratio"] = float(p[-15:].mean() / mid) if mid else np.nan
        m[f"{name}_profile_cv"] = float(p[1:].std() / p[1:].mean())
    # Pass/fail is assigned by the runner against the real-data band.
    return _result(m, {}, extra={k: v.tolist() for k, v in prof.items()})


# --------------------------------------------------------------------------- #
# A9 leverage and asymmetry  (OPTIONAL, criterion PROVISIONAL)
# --------------------------------------------------------------------------- #
def a9_leverage(s: BarSeries, cfg: dict, price: str = "mid", **_) -> dict:
    c = cfg["a9_leverage"]
    r, idx = intraday_returns(s, price)
    rt = deseasonalized_returns(r, s.minute[idx])
    a = np.abs(rt)
    lev = [float(np.corrcoef(rt[:-k], a[k:])[0, 1]) for k in range(1, c["lags"] + 1)]
    day = s.day[idx]
    daily = np.bincount(np.unique(day, return_inverse=True)[1], weights=r)
    nb = len(daily) // c["multi_day"]
    md = daily[:nb * c["multi_day"]].reshape(nb, c["multi_day"]).sum(axis=1)
    m = {"leverage_mean_1_10": float(np.mean(lev)), "leverage_1": lev[0],
         "skew_multi_day": float(stats.skew(md)) if nb > 10 else np.nan,
         "skew_daily": float(stats.skew(daily))}
    return _result(m, {}, extra={"leverage": lev})


BAR_TESTS = {
    "T1_fat_tails": t1_fat_tails,
    "T2_vol_clustering": t2_vol_clustering,
    "T3_no_return_autocorr": t3_no_autocorr,
    "T4_vol_volume": t4_vol_volume,
    "A1_aggregational_gaussianity": a1_aggregation,
    "A3_volume_clustering": a3_volume_clustering,
    "A4_extreme_intermittency": a4_intermittency,
    "A5_intraday_seasonality": a5_seasonality,
    "A9_leverage_asymmetry": a9_leverage,
}
