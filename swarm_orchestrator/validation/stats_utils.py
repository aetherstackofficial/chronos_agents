"""Statistical helpers shared by every test (spec §2.4-2.5, §14.1)."""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Optional, Sequence

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf as _acf


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelation at lags 1..nlags."""
    return _acf(np.asarray(x, float), nlags=nlags, fft=True)[1:]


def ljung_box_p(x: np.ndarray, lags: int) -> float:
    """Ljung-Box p-value, Q = n(n+2) sum rho_k^2 / (n-k), chi2(lags).

    Same statistic as statsmodels' acorr_ljungbox (checked equal in
    tests/test_validation_stats.py) but from the FFT ACF, which is ~100x
    faster at N = 1e5.
    """
    x = np.asarray(x, float)
    n = len(x)
    rho = acf(x, lags)
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
    return float(stats.chi2.sf(q, lags))


def ljung_box_p_statsmodels(x: np.ndarray, lags: int) -> float:
    return float(acorr_ljungbox(np.asarray(x, float), lags=[lags],
                                return_df=True)["lb_pvalue"].iloc[0])


def arch_lm_p(x: np.ndarray, lags: int = 10) -> float:
    return float(het_arch(np.asarray(x, float), nlags=lags)[1])


def excess_kurtosis(x: np.ndarray) -> float:
    return float(stats.kurtosis(x, fisher=True, bias=False))


def block_bootstrap(stat: Callable[[np.ndarray], float], n: int, block: int,
                    n_boot: int, seed: int) -> np.ndarray:
    """Moving-block bootstrap of ``stat(index_array)`` (spec T1 step 4).

    ``stat`` receives resampled indices so paired series (V_t, sigma_t) are
    resampled together.
    """
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offs = np.arange(block)
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        idx = (starts[:, None] + offs[None, :]).ravel()[:n]
        out[b] = stat(idx)
    return out


def power_law_fit(rho: np.ndarray, lags: Optional[np.ndarray] = None
                  ) -> Dict[str, float]:
    """Fit rho_k ~ c k^-beta by OLS in log-log space on lags with rho_k > 0."""
    lags = np.arange(1, len(rho) + 1) if lags is None else lags
    pos = rho > 0
    if pos.sum() < 3:
        return {"beta": float("nan"), "beta_se": float("nan"), "n_pos": int(pos.sum())}
    res = stats.linregress(np.log(lags[pos]), np.log(rho[pos]))
    return {"beta": float(-res.slope), "beta_se": float(res.stderr),
            "n_pos": int(pos.sum())}


def first_lag_below(rho: np.ndarray, level: float) -> Optional[int]:
    below = np.flatnonzero(rho < level)
    return int(below[0] + 1) if len(below) else None


def summarize(values: Iterable[float]) -> Dict[str, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return {"median": None, "p05": None, "p95": None, "n": 0}
    return {"median": float(np.median(v)), "p05": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)), "n": int(len(v))}


def band_from_real(values: Sequence[float], widen: float = 0.10
                   ) -> Optional[list]:
    """5th-95th percentile of real-data windows widened by 10% of value (§15.4)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    lo, hi = np.percentile(v, [5, 95])
    return [float(lo - widen * abs(lo)), float(hi + widen * abs(hi))]


def round_sig(x, sig: int = 6):
    """Round floats (recursively) to 6 significant digits (§13 rules)."""
    if isinstance(x, dict):
        return {k: round_sig(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [round_sig(v, sig) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        if not math.isfinite(x) or x == 0.0:
            return None if not math.isfinite(x) else 0.0
        return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return round_sig(x.tolist(), sig)
    return x
