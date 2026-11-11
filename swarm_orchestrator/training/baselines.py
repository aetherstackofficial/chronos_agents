"""Rule-based baselines for the v2 agents.  Every policy has ``predict(obs, env) -> action in [-1, 1]``
(and ``reset()``), reading ``env.memory`` / ``env.book`` / ``env.last_tick`` and the raw observation.

* ``AvellanedaStoikovMM``  - market maker.  Reservation price r = mid - q*gamma*sigma^2*(T-t),
  half-spread delta = gamma*sigma^2*(T-t)/2 + (1/gamma)*ln(1 + gamma/k), q = inventory/capacity,
  sigma from the memory's per-minute return volatility.  The quotes are mapped onto the decoder
  by inverting ``agents.action_decode.decode_mm`` (tick offsets from the touch -> output in [-1, 1],
  size fraction of ``max_quote_size`` -> output in [-1, 1]).
* ``BuyAndHold``  - whale: target +1 from the first step, participation 15%, marketable.
* ``TrendRule``  - whale: long if minute EMA20 > EMA60 else short (memory features
  ``ema20_dist`` / ``ema60_dist``), full size, participation 15%, marketable.

    python -m training.baselines --model mm --split validation --episodes 20 [--policy as]
    python -m training.baselines --model whale_swing --split validation --episodes 20

writes the same JSON as ``training.evaluate`` to ``training/runs/baselines/<model>_<policy>_<stamp>/eval_<split>.json``.

Units of the Avellaneda-Stoikov config (all in TICKS and MINUTES, so one setting suits every
stock): ``gamma`` = risk aversion per tick of price variance, ``k`` = order-arrival decay per
tick, ``horizon_minutes`` = cap of the remaining-time factor T-t (the AS "end of day" becomes a
rolling horizon so the quotes do not blow up at the open), ``size_fraction`` = quote size as a
fraction of max_quote_size when inventory is zero.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from agents.action_decode import DEFAULTS
from training import evaluate as E
from training import train as T

SESSION_MINUTES = 375
AS_DEFAULTS = {
    "gamma": 0.1,               # risk aversion, 1/tick
    "k": 1.5,                   # order-flow decay, 1/tick
    "horizon_minutes": 30.0,    # cap on T - t
    "size_fraction": 0.5,       # of max_quote_size at zero inventory
    "sigma_floor_ticks": 0.2,   # per-sqrt-minute sigma floor, in ticks
}


def _clip1(x: float) -> float:
    return float(min(1.0, max(-1.0, x)))


def _tick_size(tick: dict) -> float:
    return float(tick.get("tick_size") or DEFAULTS["fallback_tick_size"])


def minute_sigma(memory, price: float) -> Optional[float]:
    """Per-minute volatility of log returns from the memory (std of the last <= 30 intraday
    1-bar returns, ``memory._rets``); None when fewer than 5 are available."""
    rets = list(getattr(memory, "_rets", []) or [])
    if len(rets) < 5:
        return None
    m = sum(rets) / len(rets)
    var = sum((x - m) ** 2 for x in rets) / len(rets)
    return math.sqrt(max(var, 0.0))


class AvellanedaStoikovMM:
    name = "avellaneda_stoikov"

    def __init__(self, cfg: Optional[dict] = None, decoder_cfg: Optional[dict] = None) -> None:
        self.cfg = {**AS_DEFAULTS, **(cfg or {})}
        self.dec = {**DEFAULTS, **(decoder_cfg or {})}

    def reset(self) -> None:
        pass

    # offsets (ticks from the own touch) -> output in [-1, 1]; inverse of decode_mm's rounding
    def _offset_to_output(self, off_ticks: float) -> float:
        omin, omax = self.dec["mm_offset_min"], self.dec["mm_offset_max"]
        off = min(max(round(off_ticks), omin), omax)
        return _clip1(2.0 * (off - omin) / (omax - omin) - 1.0)

    @staticmethod
    def _size_to_output(frac: float) -> float:
        return _clip1(2.0 * frac - 1.0)

    def quotes(self, env) -> dict:
        """Intermediate numbers (in ticks) - exposed for tests and the decision log."""
        tick = getattr(env, "last_tick", None) or {}
        book = env.book
        ts = _tick_size(tick)
        bb, ba = tick.get("best_bid"), tick.get("best_ask")
        last = float(tick.get("last_price") or 0.0)
        if bb is None or ba is None:
            half = self.dec["mm_fallback_half_spread_ticks"] * ts
            bb, ba = last - half, last + half
        bb, ba = float(bb), float(ba)
        mid = 0.5 * (bb + ba)
        sig = minute_sigma(env.memory, mid)
        sigma_t = max((sig * mid / ts) if sig is not None else 0.0, self.cfg["sigma_floor_ticks"])
        minute = float(tick.get("market_minute", 0) or 0)
        tau = min(max(SESSION_MINUTES - minute, 0.0), self.cfg["horizon_minutes"])
        q = float(book.position) / float(book.capacity)
        g, k = self.cfg["gamma"], self.cfg["k"]
        r = -q * g * sigma_t ** 2 * tau                                   # reservation price - mid, ticks
        delta = g * sigma_t ** 2 * tau / 2.0 + math.log(1.0 + g / k) / g  # half-spread, ticks
        bid_t, ask_t = r - delta, r + delta                              # quotes relative to mid, ticks
        # decoder: bid = best_bid - off_b*ts, ask = best_ask + off_a*ts
        half_mkt = 0.5 * (ba - bb) / ts
        off_b = (-half_mkt) - bid_t                                       # ticks below the best bid
        off_a = ask_t - half_mkt                                          # ticks above the best ask
        return {"mid": mid, "sigma_ticks": sigma_t, "tau": tau, "q": q, "reservation_ticks": r,
                "half_spread_ticks": delta, "bid_offset": off_b, "ask_offset": off_a, "tick_size": ts}

    def predict(self, obs, env) -> np.ndarray:
        d = self.quotes(env)
        q = d["q"]
        sf = self.cfg["size_fraction"]
        # quote less on the side that grows the inventory; stop at the capacity edge
        bid_frac = sf * max(0.0, 1.0 - max(q, 0.0))
        ask_frac = sf * max(0.0, 1.0 + min(q, 0.0))
        floor = self.dec["mm_size_floor"]
        bid_out = self._size_to_output(bid_frac) if bid_frac >= floor else -1.0
        ask_out = self._size_to_output(ask_frac) if ask_frac >= floor else -1.0
        return np.array([self._offset_to_output(d["bid_offset"]), self._offset_to_output(d["ask_offset"]),
                         bid_out, ask_out], dtype=np.float32)


class _WhaleRule:
    participation_output = 1.0          # +1 -> participation_max (15%)
    aggressiveness_output = 1.0         # +1 -> aggr 1.0: priced through the opposite touch (marketable)

    def reset(self) -> None:
        pass

    def target(self, obs, env) -> float:                               # pragma: no cover
        raise NotImplementedError

    def predict(self, obs, env) -> np.ndarray:
        return np.array([_clip1(self.target(obs, env)), self.participation_output,
                         self.aggressiveness_output], dtype=np.float32)


class BuyAndHold(_WhaleRule):
    name = "buy_and_hold"

    def target(self, obs, env) -> float:
        return 1.0


class TrendRule(_WhaleRule):
    """Long if the minute EMA20 is above the EMA60, else short.  ``ema*_dist`` = (price - ema)/price,
    so EMA20 > EMA60  <=>  ema20_dist < ema60_dist.  Before 60 bars exist the memory reports 0 for
    both, which counts as 'not above' -> short; the env pre-rolls memory so this is only the cold start."""
    name = "trend_ema20_60"

    def target(self, obs, env) -> float:
        m = env.memory
        if getattr(m, "bars_seen", 0) < 60:
            return 0.0
        f = m.features()
        return 1.0 if f["ema20_dist"] < f["ema60_dist"] else -1.0


class TrendBandRule(_WhaleRule):
    """Minute EMA20/60 trend with an entry threshold and hysteresis, traded passively.

    gap = (EMA20 - EMA60) / price.  Flat -> long when gap > theta, short when gap < -theta.
    A long is closed when the averages cross back (gap < 0) and only reversed if gap < -theta
    (mirror for a short).  ``long_only`` never goes short.  Orders: 5% participation, one tick
    inside the own side (participation output -3/7, aggressiveness output -1/3).
    Name ``band<bp>`` or ``band<bp>L`` (long only), e.g. band20 = 20 bp threshold."""
    participation_output = -3 / 7
    aggressiveness_output = -1 / 3

    def __init__(self, theta_bp: float = 20.0, long_only: bool = False) -> None:
        self.theta = float(theta_bp) / 1e4
        self.long_only = bool(long_only)
        self.name = f"band{theta_bp:g}{'L' if long_only else ''}"
        self.state = 0.0

    def reset(self) -> None:
        self.state = 0.0

    def target(self, obs, env) -> float:
        m = env.memory
        if getattr(m, "bars_seen", 0) < 60:
            return 0.0
        f = m.features()
        gap = f["ema60_dist"] - f["ema20_dist"]
        s, th = self.state, self.theta
        if s == 0.0:
            s = 1.0 if gap > th else (-1.0 if gap < -th else 0.0)
        elif s > 0 and gap < 0:
            s = -1.0 if gap < -th else 0.0
        elif s < 0 and gap > 0:
            s = 1.0 if gap > th else 0.0
        if self.long_only and s < 0:
            s = 0.0
        self.state = s
        return s


class MomentumRule(_WhaleRule):
    """Time-series momentum on the daily memory, traded passively (5% participation, one tick
    inside).  Target = clip(R / scale): R = 60-day return, scale 10% (whale_position) or R =
    20-day return, scale 6% (other whales).  Before the window is filled the memory reports 0 -> flat."""
    name = "mom"
    participation_output = -3 / 7
    aggressiveness_output = -1 / 3

    def __init__(self, model_name: str = "whale_position") -> None:
        pos = model_name == "whale_position"
        self.key, self.scale = ("ret_60d", 0.10) if pos else ("ret_20d", 0.06)

    def target(self, obs, env) -> float:
        r = env.memory.features().get(self.key) or 0.0
        return float(r) / self.scale


class LongMomentumRule(MomentumRule):
    """MomentumRule that never goes short (target = max(0, R / scale))."""
    name = "momL"

    def target(self, obs, env) -> float:
        return max(0.0, super().target(obs, env))


class DailyTrendRule(_WhaleRule):
    """Daily trend on the memory, traded passively. d = distance of the price to its 50-day EMA in ATR
    units, x = 50-day EMA vs 200-day EMA in ATR units.
    dtrend: d / 2 | longtrend: max(0, d / 2) | dual: long (0.4 + d/3) when d > 0 and x > 0, short
    (-0.4 + d/3) when both < 0, flat otherwise."""
    participation_output = -3 / 7
    aggressiveness_output = -1 / 3

    def __init__(self, variant: str = "dtrend") -> None:
        if variant not in ("dtrend", "longtrend", "dual"):
            raise ValueError(variant)
        self.variant = self.name = variant

    def target(self, obs, env) -> float:
        f = env.memory.features()
        d, x = f.get("ema50d_dist_atr") or 0.0, f.get("ema50_vs_200_atr") or 0.0
        if self.variant == "dtrend":
            return d / 2.0
        if self.variant == "longtrend":
            return max(0.0, d / 2.0)
        if d > 0 and x > 0:
            return 0.4 + d / 3.0
        if d < 0 and x < 0:
            return -0.4 + d / 3.0
        return 0.0


def make_baseline(model_name: str, which: Optional[str] = None, cfg: Optional[dict] = None):
    role = T.role_of(model_name)
    which = which or ("as" if role == "mm" else "trend")
    if role == "mm":
        if which not in ("as", "avellaneda_stoikov"):
            raise ValueError("mm baseline is 'as'")
        return AvellanedaStoikovMM(cfg)
    if which in ("buyhold", "buy_and_hold"):
        return BuyAndHold()
    if which in ("trend", "trend_ema20_60"):
        return TrendRule()
    if which in ("mom", "momentum"):
        return MomentumRule(model_name)
    if which == "momL":
        return LongMomentumRule(model_name)
    if which in ("dtrend", "longtrend", "dual"):
        return DailyTrendRule(which)
    if which.startswith("band"):
        spec = which[4:]
        long_only = spec.endswith("L")
        return TrendBandRule(float(spec.rstrip("L")), long_only)
    raise ValueError("whale baselines: 'buyhold', 'trend' or 'band<bp>[L]'")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="training.baselines", description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=T.MODELS)
    ap.add_argument("--split", default="validation", choices=("validation", "test", "train"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--policy", help="mm: as | whale: buyhold, trend (default: all for the role)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cost-mult", type=float)
    ap.add_argument("--stocks")
    ap.add_argument("--episode-sessions", type=int, default=5, help="sessions per episode (default 5, like the eval)")
    ap.add_argument("--decision-minutes", type=int,
                    help="minutes per decision (default: env.yaml, whale 15 / 75 and mm 1); the rule acts once per decision")
    ap.add_argument("--out-dir")
    a = ap.parse_args(argv)
    if str(T.ROOT) not in sys.path:
        sys.path.insert(0, str(T.ROOT))
    role = T.role_of(a.model)
    which_list = [a.policy] if a.policy else (["as"] if role == "mm" else ["buyhold", "trend"])
    stocks = [s.strip() for s in a.stocks.split(",")] if a.stocks else None
    for which in which_list:
        pol = make_baseline(a.model, which)
        cfg = {"model": a.model, "env_overrides": {}, "eval_env_overrides": {}}
        env_cfg = E.eval_env_cfg(cfg, a.cost_mult, stocks, a.episode_sessions, model_name=a.model,
                               decision_minutes=a.decision_minutes)
        res = E.evaluate_policy(a.model, pol, a.split, env_cfg, a.episodes, a.seed, algo="baseline",
                                run_id=f"baseline_{pol.name}", stocks=stocks)
        res.update({"policy": f"baseline:{pol.name}", "cost_mult": a.cost_mult, "stocks": stocks,
                    "timestamp": dt.datetime.now().isoformat(timespec="seconds")})
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(a.out_dir) if a.out_dir else T.RUNS_DIR / "baselines" / f"{a.model}_{pol.name}_{stamp}"
        out = out_dir / f"eval_{a.split}.json"
        E.write_eval_json(out, res)
        m = res["metrics"]
        print(f"{a.model} {pol.name}: wrote {out}")
        print(f"  pnl mean {m['profit_after_costs_mean']:.2f}  sharpe {m['sharpe_per_session_annualised']}  "
              f"worst dd {m['worst_drawdown']:.4f}  turnover/session {m['turnover_per_session']:.3f}  "
              f"steps {res['steps']} (x{res['decision_minutes']} min)  wall {res['wall_seconds']}s")
        print("  checks failed:", res["checks_failed"] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
