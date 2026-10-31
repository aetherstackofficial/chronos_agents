"""Download the reference NSE files from Kaggle (no login needed per file).

Dataset: debashis74017/algo-trading-data-nifty-100-data-with-indicators
(CC0, "1 minute data 2015-2026", ~55 MB per stock). Only the instruments
used for calibration are fetched, not the 21 GB dataset.
"""
from __future__ import annotations

import argparse
import os
import urllib.request

from validation.calibrate import DATA_DIR

URL = ("https://www.kaggle.com/api/v1/datasets/download/"
       "debashis74017/algo-trading-data-nifty-100-data-with-indicators/{sym}_minute.csv")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruments", default="RELIANCE,HDFCBANK,INFY,SBIN,ITC,MARUTI")
    a = ap.parse_args(argv)
    os.makedirs(DATA_DIR, exist_ok=True)
    for sym in a.instruments.split(","):
        dest = os.path.join(DATA_DIR, f"{sym}_minute.csv")
        if os.path.exists(dest):
            print(f"{sym}: present")
            continue
        urllib.request.urlretrieve(URL.format(sym=sym), dest)
        print(f"{sym}: {os.path.getsize(dest) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
