"""Opt-in replacement of the engine's designated liquidity (SIM_LP) refresh, for experiments.

engine/matching_engine.py is NOT modified: ``install()`` swaps ``MatchingEngine._refresh_sim_lp`` in the
current process only (the validation workers call it through agents.live_agents_v2.install). Replay
mode is unaffected (the engine refreshes SIM_LP only outside REPLAY).

Engine default: 8 levels per side, spaced 1-3 ticks around the last price, each of size
400 + 0.5 x (20-minute mean volume) -- the book thickens as soon as volume rises, so trading volume
hardly moves the price, and every move is permanent (the ladder re-centres on the last trade).

Latent-liquidity book (CHRONOS_LP=latent), after the latent order book picture of Toth et al. (2011)
and Donier et al. (2015): slow participants' limit orders are spread around the price with a density
that grows linearly with distance, which makes impact concave (square-root) and lets part of it fade:
* one level per tick at distance d = 1..D from the reference, size  lam x d^a  (a = 1: linear
  density), lam = CHRONOS_LP_LAMBDA x (expected minute volume of the instrument's average day);
  the depth does not follow the last 20 minutes' volume;
* reference = last price pulled toward a slow anchor (EMA of the last price, half-life
  CHRONOS_LP_HALFLIFE minutes) by a fraction CHRONOS_LP_THETA: part of a price move made by a burst
  of orders fades as the anchor has not followed it yet (CHRONOS_LP_THETA = 0: no fading);
* when the last minute's range was large against its slow average, the book is thinner by the factor
  1 / (1 + CHRONOS_LP_VOLFB x excess): liquidity providers step back when the market moves.
"""
from __future__ import annotations

import os

_INSTALLED = False


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def real_depth_profile(symbol: str, window: int = 15):
    """Time-of-day liquidity of the real instrument, from its NSE minute bars (all sessions):
    depth(m) ~ mean volume(m) / mean |return|(m)^2 -- the depth at which the observed volume moves
    the price by the observed amount under the square-root (linear-density) book -- smoothed over
    ``window`` minutes, normalised to mean 1, clipped to [0.25, 4]. Real NSE books are thin right
    after the open and deep into the close (volume x2.3 at the close, |return| only x1.1)."""
    import numpy as np
    from validation.data import load_real
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s = load_real(os.path.join(root, "data", "real", "nse_kaggle", f"{symbol}_minute.csv"))
    n = 375
    lc = np.log(s.c)
    r = np.abs(np.diff(lc))
    same = s.day[1:] == s.day[:-1]
    mins = s.minute[1:][same]
    absr = np.bincount(mins, weights=r[same], minlength=n) / np.maximum(np.bincount(mins, minlength=n), 1)
    vol = np.bincount(s.minute, weights=s.v, minlength=n) / np.maximum(np.bincount(s.minute, minlength=n), 1)
    absr[0] = absr[1]
    k = np.ones(window) / window
    pad = window // 2
    sm = lambda x: np.convolve(np.r_[np.full(pad, x[0]), x, np.full(pad, x[-1])], k, "valid")[:n]
    prof = sm(vol) / np.maximum(sm(absr), 1e-12) ** 2
    prof = prof / prof.mean()
    return np.clip(prof, 0.25, 4.0)


def install() -> None:
    global _INSTALLED
    if _INSTALLED or os.environ.get("CHRONOS_LP") != "latent":
        return
    from engine.matching_engine import MatchingEngine, round_to_tick   # noqa: F401
    from agents.execution import VolumeCurve

    lam_rel = _f("CHRONOS_LP_LAMBDA", 0.04)
    depth = int(_f("CHRONOS_LP_DEPTH", 20))
    expo = _f("CHRONOS_LP_EXPONENT", 1.0)
    theta = _f("CHRONOS_LP_THETA", 0.0)
    half = max(_f("CHRONOS_LP_HALFLIFE", 60.0), 1.0)
    volfb = _f("CHRONOS_LP_VOLFB", 0.0)
    step = max(int(_f("CHRONOS_LP_STEP", 1)), 1)
    spread1 = os.environ.get("CHRONOS_LP_SPREAD1") == "1"
    alpha = 1.0 - 0.5 ** (1.0 / half)
    sym = os.environ.get("CHRONOS_V2_SEED_SYMBOL", "RELIANCE")
    c = VolumeCurve().curve_for(sym)
    per_min = float(c["adv_shares"]) / len(c["minute_weights"])
    lam = lam_rel * per_min
    tod = None
    if os.environ.get("CHRONOS_LP_TOD") == "1":
        tod = real_depth_profile(sym) ** _f("CHRONOS_LP_TOD_POWER", 1.0)
        tod = tod / tod.mean()

    def _refresh_sim_lp(self, pubs, reference=None):
        self.flush_agent_orders(self.LP_AGENT)
        last = reference if reference is not None else self.last_price
        st = self.__dict__.setdefault("_llp", {"anchor": None, "rng_ema": None})
        st["anchor"] = last if st["anchor"] is None else st["anchor"] + alpha * (last - st["anchor"])
        ref = round_to_tick(last + theta * (st["anchor"] - last), self.tick)
        thin = 1.0
        if volfb > 0 and self._last_bar is not None and self._last_bar.get("c"):
            r = (self._last_bar["h"] - self._last_bar["l"]) / self._last_bar["c"]
            st["rng_ema"] = r if st["rng_ema"] is None else st["rng_ema"] + 0.01 * (r - st["rng_ema"])
            if st["rng_ema"] and st["rng_ema"] > 0:
                thin = 1.0 / (1.0 + volfb * max(r / st["rng_ema"] - 1.0, 0.0))
        if tod is not None:
            thin *= float(tod[min(max(int(self.market_minute) - 1, 0), len(tod) - 1)])
        # touch: ref -/+ 1 tick (2-tick spread), or with CHRONOS_LP_SPREAD1=1 a 1-tick spread whose
        # touch on the side that traded more in the last minute is the last price itself
        if spread1:
            # orientation follows the last minute's net aggressor side; on a tie (incl. no trades)
            # it keeps the previous one, so that a quiet market has no direction of its own
            if self._last_min_buy != self._last_min_sell:
                st["up"] = self._last_min_buy > self._last_min_sell
            elif "up" not in st:
                st["up"] = self.rng.random() < 0.5
            up = st["up"]
            bid0 = round_to_tick(ref - self.tick, self.tick) if up else ref
            ask0 = round_to_tick(bid0 + self.tick, self.tick)
            # never take liquidity: stay behind the other participants' resting quotes
            bb_o, ba_o = self.book.best_bid(), self.book.best_ask()
            if ba_o is not None and bid0 >= ba_o - 1e-9:
                bid0 = round_to_tick(ba_o - self.tick, self.tick)
            if bb_o is not None and ask0 <= bb_o + 1e-9:
                ask0 = round_to_tick(bb_o + self.tick, self.tick)
        else:
            bid0 = round_to_tick(ref - self.tick, self.tick)
            ask0 = round_to_tick(ref + self.tick, self.tick)
        # levels at distances d = 1, 1 + U{1..step}, ... ticks (step 1: every tick); a level of a
        # sparser book carries the depth of the ticks it stands for, so the density stays ~ d^a
        for side in ("BUY", "SELL"):
            d, prev = 1, 0
            while d <= depth:
                width = d - prev
                base = lam * thin * width * d ** expo
                sz = max(1, int(base * self.rng.uniform(1 - self.LP_SIZE_JITTER, 1 + self.LP_SIZE_JITTER)))
                px = round_to_tick(bid0 - (d - 1) * self.tick if side == "BUY" else ask0 + (d - 1) * self.tick,
                                   self.tick)
                if px > 0:
                    self.submit_flow_order(self.LP_AGENT, side, "LIMIT", sz, px, pubs, tif_ticks=self.LP_TIF_TICKS)
                prev = d
                d += self.rng.randint(1, step) if step > 1 else 1

    MatchingEngine._refresh_sim_lp = _refresh_sim_lp
    _INSTALLED = True
