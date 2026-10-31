"""Build agents/config/volume_curves.json from the real NSE reference data.

Per symbol: mean daily volume (ADV) over the latest 268 full sessions and the
intraday shape over the same sessions (mean volume per minute-of-day bin,
normalised so the 375 weights average 1). Expected volume of minute m is then
ADV / 375 * weight[m]. Used by the whale POV slicer (Phase 4).

Usage: python -m validation.build_volume_curves
"""
from __future__ import annotations

import json
import os

import numpy as np

from validation.calibrate import DATA_DIR
from validation.data import SESSION_BARS, load_real

SYMBOLS = ("RELIANCE", "HDFCBANK", "INFY", "SBIN", "ITC", "MARUTI")
WINDOW_DAYS = 268
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "agents", "config", "volume_curves.json")


def curve(symbol: str) -> dict:
    s = load_real(os.path.join(DATA_DIR, f"{symbol}_minute.csv"))
    first = s.n_days - WINDOW_DAYS
    w = s.subset(s.day >= first)
    daily = np.bincount(w.day - first, weights=w.v)
    per_min = np.bincount(w.minute, weights=w.v, minlength=SESSION_BARS) / WINDOW_DAYS
    # NSE's 09:15 bar carries the pre-open call-auction print. The engine has
    # no auction (its first minute is continuous trading) and POV algorithms
    # exclude auction volume, so minute 0 takes minute 1's expected volume.
    # Other minutes are left as measured (no renormalisation).
    auction_excess = float(per_min[0] - per_min[1])
    per_min[0] = per_min[1]
    adv_cont = float(per_min.sum())
    dates = s.info["dates"]
    return {"adv_shares": round(adv_cont),
            "adv_shares_including_open_auction": round(float(daily.mean())),
            "open_auction_excess_shares": round(auction_excess),
            "window": {"first": dates[first], "last": dates[-1], "sessions": WINDOW_DAYS},
            "minute_weights": [round(float(x), 5) for x in per_min / per_min.mean()]}


def main() -> None:
    out = {"_about": ("Expected NSE volume per minute = adv_shares / 375 * minute_weights[m]. "
                      "adv_shares is the continuous-session volume: the 09:15 bar's open-auction "
                      "excess is removed (minute 0 set to minute 1). Source: Kaggle "
                      "debashis74017 NIFTY-100 1-minute bars, latest 268 full sessions per "
                      "symbol. Built by validation/build_volume_curves.py."),
           "fallback_symbol": "RELIANCE",
           "symbols": {sym: curve(sym) for sym in SYMBOLS}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    for sym, c in out["symbols"].items():
        wts = np.asarray(c["minute_weights"])
        print(f"{sym}: ADV {c['adv_shares']:,}  {c['window']['first']}..{c['window']['last']}  "
              f"open/midday {wts[:15].mean() / wts[150:225].mean():.2f}  "
              f"close/midday {wts[-15:].mean() / wts[150:225].mean():.2f}")


if __name__ == "__main__":
    main()
