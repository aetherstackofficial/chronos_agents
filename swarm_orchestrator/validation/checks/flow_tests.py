"""Order-flow and strategy tests: A2 (sign memory, sim only) and A8 (no free
lunch, runs on any BarSeries so real NSE data gives the reference)."""
from __future__ import annotations

from typing import Dict

import numpy as np

from validation.data import BarSeries
from validation.stats_utils import acf, power_law_fit


# --------------------------------------------------------------------------- #
# A2 order-flow sign memory
# --------------------------------------------------------------------------- #
def taker_order_signs(trades: Dict[str, np.ndarray], min_bar: int) -> np.ndarray:
    """One sign per aggressive order, not per print.

    A market order that walks several levels produces several prints with
    the same sign; counting each print would inflate sign autocorrelation.
    Consecutive prints with the same bar, taker and sign are merged (the
    trade log has no taker order id, so this is the closest proxy).
    """
    keep = trades["bar"] >= min_bar
    bar, sign = trades["bar"][keep], trades["sign"][keep]
    taker = np.where(sign > 0, trades["buyer"][keep], trades["seller"][keep])
    new = np.r_[True, (bar[1:] != bar[:-1]) | (sign[1:] != sign[:-1])
                | (taker[1:] != taker[:-1])]
    return sign[new].astype(float)


def a2_sign_memory(trades: Dict[str, np.ndarray], cfg: dict, min_bar: int) -> dict:
    c = cfg["a2_sign_memory"]
    s = taker_order_signs(trades, min_bar)
    if len(s) < 10 * c["max_lag"]:
        return {"metrics": {"n_orders": float(len(s))},
                "criteria": {"enough_orders": False}, "pass": False}
    rho = acf(s, c["max_lag"])
    fit = power_law_fit(rho)
    m = {"n_orders": float(len(s)), "rho_1": float(rho[0]),
         "rho_20": float(rho[19]), "beta": fit["beta"],
         "buy_fraction": float(np.mean(s > 0))}
    crit = {"rho_1_20_all_positive": bool(np.all(rho[:c["positive_lags"]] > 0)),
            "beta_lt_1": bool(np.isfinite(fit["beta"]) and fit["beta"] < c["beta_max"])}
    return {"metrics": m, "criteria": crit, "pass": all(crit.values()),
            "extra": {"rho": rho.tolist()}}


# --------------------------------------------------------------------------- #
# A8 no free lunch
# --------------------------------------------------------------------------- #
def _tstat(x: np.ndarray) -> float:
    if len(x) < 3 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def a8_no_free_lunch(s: BarSeries, cfg: dict, seed: int = 0, price: str = "c",
                     **_) -> dict:
    """Simple rules and random strategies on the bar closes, net of costs.

    Execution at the bar close of the signal bar and of the exit bar, all
    positions flat by the session end (intraday, as MIS). Cost is the
    intraday NSE round-trip with zero brokerage (the cost most favourable
    to the strategies, i.e. the strictest test); the discount-broker cost is
    reported too. No spread/slippage on top: on real data the close is a
    trade price, so half the bounce is already in it.
    """
    c = cfg["a8_no_free_lunch"]
    cost = c["cost_bps_zero_brokerage"] * 1e-4
    p = getattr(s, price) if price != "mid" else s.mid
    lp = np.log(p)
    day, minute = s.day, s.minute
    days = np.unique(day)
    starts = np.searchsorted(day, days)
    ends = np.r_[starts[1:], len(day)] - 1
    rules: Dict[str, np.ndarray] = {}

    # Buy and hold within the session (open-bar close -> last-bar close).
    rules["buy_hold_intraday"] = lp[ends] - lp[starts] - cost

    def lagged_rule(k: int, direction: int) -> np.ndarray:
        # Signal at bars with minute % k == 0 from the past k-bar return;
        # hold k bars; skip if the exit would cross the session end.
        t = np.flatnonzero((minute % k == 0) & (minute >= k) & (minute + k < 375))
        past = lp[t] - lp[t - k]
        fwd = lp[t + k] - lp[t]
        sig = np.sign(past) * direction
        live = sig != 0
        return sig[live] * fwd[live] - cost

    for k in (5, 15, 30):
        rules[f"momentum_{k}m"] = lagged_rule(k, +1)
        rules[f"mean_reversion_{k}m"] = lagged_rule(k, -1)

    # Volatility timing: long for 30 bars when trailing-30 |r| sum is below
    # the median of the previous 60 such marks (~5 sessions); no lookahead.
    t = np.flatnonzero((minute % 30 == 0) & (minute >= 30) & (minute + 30 < 375))
    absr = np.abs(np.diff(lp, prepend=lp[0]))
    absr[np.r_[True, day[1:] != day[:-1]]] = 0.0
    csum = np.cumsum(absr)
    tv = csum[t] - csum[t - 30]
    fwd = lp[t + 30] - lp[t]
    vt = []
    for i in range(60, len(t)):
        if tv[i] < np.median(tv[i - 60:i]):
            vt.append(fwd[i] - cost)
    rules["vol_timing_30m"] = np.asarray(vt)

    rule_t = {k: _tstat(v) for k, v in rules.items()}
    rule_mean_bps = {k: float(v.mean() * 1e4) if len(v) else 0.0
                     for k, v in rules.items()}

    # Random strategies: 100 trades each, random bar/side/hold (1..60 bars).
    rng = np.random.default_rng([seed, 8])
    n = len(p)
    means, ts = [], []
    for _ in range(c["n_random"]):
        t0 = rng.integers(0, n, 100)
        hold = rng.integers(1, 61, 100)
        t1 = np.minimum(t0 + hold, ends[np.searchsorted(days, day[t0])])
        side = rng.choice([-1.0, 1.0], 100)
        ok = t1 > t0
        pnl = side[ok] * (lp[t1[ok]] - lp[t0[ok]]) - cost
        means.append(pnl.mean())
        ts.append(_tstat(pnl))
    means, ts = np.asarray(means), np.asarray(ts)
    best = max(rule_t, key=lambda k: abs(rule_t[k]))
    m = {"random_mean_net_bps": float(means.mean() * 1e4),
         "random_frac_t_gt_3": float(np.mean(ts > c["t_stat_max"])),
         "max_rule_t": float(max(rule_t.values())),
         "max_abs_rule_t": float(abs(rule_t[best])),
         **{f"t_{k}": v for k, v in rule_t.items()}}
    # Gross (pre-cost) mean of mean-reversion 5m: a direct read of bounce/
    # predictable direction, independent of the cost assumption.
    m["mean_reversion_5m_gross_bps"] = rule_mean_bps["mean_reversion_5m"] + \
        c["cost_bps_zero_brokerage"]
    crit = {"random_mean_net_le_0": m["random_mean_net_bps"] <= 0,
            "no_rule_t_gt_3": m["max_rule_t"] <= c["t_stat_max"]}
    return {"metrics": m, "criteria": crit, "pass": all(crit.values()),
            "extra": {"rule_mean_net_bps": rule_mean_bps, "best_abs_t_rule": best}}
