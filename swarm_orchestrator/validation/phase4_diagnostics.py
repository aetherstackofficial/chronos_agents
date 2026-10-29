"""Phase 4 mechanism diagnostics (not part of the realism suite).

    python -m validation.phase4_diagnostics [--seeds 5] [--days 20]

Per seed, two paired runs of the live swarm with identical seeds:
  * skew ON  (the frozen rule, k = 3 ticks)
  * skew OFF (k = 0; everything else identical)
Both runs inject a synthetic "news" event once per session at a seeded
random minute. That only switches the orchestrator to full participation for
its 60-tick news window (worker.py behaviour); it carries no price signal.

Reported:
  * whale effective participation = whale fills / expected NSE volume of the
    same bars, per whale, split into quiet bars and news-window bars;
  * parents: count, close status, fill ratio, shortfall, child fill rate;
  * scale mismatch: whale share of simulated volume, simulated vs expected
    NSE bar volume;
  * market makers: inventory level, time pinned at the limit, inventory
    mean reversion (AR(1) on per-minute inventory), skew ON vs OFF.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WARMUP_DAYS = 2


def _run(seed: int, days: int, price: float, k_ticks: int) -> dict:
    import logging
    import warnings
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    import agents.execution as ex
    ex.CONFIG["mm_inventory_skew"]["k_ticks"] = k_ticks      # diagnostic-only override
    from validation.sim_driver import Simulation

    sim = Simulation(seed, price, symbol="RELIANCE")
    curve = ex.VolumeCurve()
    # Count what the whales actually send (observation only).
    whale_msgs = {"MARKET": 0, "LIMIT": 0, "CANCEL_ALL": 0}
    orig_handle = sim.engine.handle_message

    def counted(msg, _orig=orig_handle):
        if str(msg.get("agent_id", "")).startswith("WHALE"):
            key = "CANCEL_ALL" if msg.get("msg") == "CANCEL_ALL" else msg.get("type")
            if key in whale_msgs:
                whale_msgs[key] += 1
        return _orig(msg)

    sim.engine.handle_message = counted
    rng = np.random.default_rng([seed, 41])
    news_at = {d: int(rng.integers(1, 375)) for d in range(1, WARMUP_DAYS + days + 1)}
    window = sim.orch.NEWS_WINDOW_TICKS
    whales = [a for a in sim.orch.agents if a.agent_id.startswith("WHALE")]
    mms = [a for a in sim.orch.agents if a.agent_id.startswith("MM")]
    n = (WARMUP_DAYS + days) * 375
    news_flag = np.zeros(n, bool)
    mm_inv = np.zeros((n, len(mms)))
    bar_day = np.zeros(n, int)
    bar_min = np.zeros(n, int)
    last_news = -10 ** 9
    for t in range(n):
        p = sim.physics()
        d, m = p["day_count"], p["market_minute"]
        if news_at.get(d) == m:
            sim.orch._on_event({"kind": "news", "score": 0.0})
            last_news = t
        # Orders sent after tick t trade in bar t+1 (index t+1 in bars).
        if t + 1 < n:
            news_flag[t + 1] = (t - last_news) <= window
        sim.agents_act(p)
        led = sim.engine.agent_ledger
        mm_inv[t] = [led.get(a.agent_id, {"pos": 0})["pos"] for a in mms]
        bar_day[t], bar_min[t] = d, m
    tr = sim.trades_array()
    bars = sim.bars_array()
    vol = bars["v"].astype(float)
    whale_fill = np.zeros(n)
    is_w_b = np.char.startswith(tr["buyer"].astype(str), "WHALE")
    is_w_s = np.char.startswith(tr["seller"].astype(str), "WHALE")
    # Per-whale participation counts each whale's own fills (a whale-to-whale
    # print is two fills); the volume share counts each print once.
    for side_mask in (is_w_b, is_w_s):
        np.add.at(whale_fill, tr["bar"][side_mask] - 1, tr["qty"][side_mask])
    whale_print = np.zeros(n)
    anyw = is_w_b | is_w_s
    np.add.at(whale_print, tr["bar"][anyw] - 1, tr["qty"][anyw])
    exp_vol = np.array([curve.expected("RELIANCE", m - 1) for m in bars["minute"]])
    keep = bars["day"] > WARMUP_DAYS
    q, nw = keep & ~news_flag, keep & news_flag
    nwh = len(whales)
    part = lambda mask: float(whale_fill[mask].sum() / exp_vol[mask].sum() / nwh) if mask.any() else None
    parents = [p for w in whales for p in w.slicer.log if p.opened_day > WARMUP_DAYS]
    status = {}
    for p in parents:
        status[p.status] = status.get(p.status, 0) + 1
    planned = np.array([abs(p.target_pos - p.pos0) for p in parents if p.pos0 is not None], float)
    filled = np.array([p.filled for p in parents if p.pos0 is not None], float)
    sent = np.array([p.child_qty_sent for p in parents], float)
    caps = [c for w in whales for c in w.slicer.children_log if c["day"] > WARMUP_DAYS]
    bind = float(np.mean([c["qty"] == c["cap"] for c in caps])) if caps else None
    inv = mm_inv[bar_day > WARMUP_DAYS]
    maxpos = np.array([a.max_position for a in mms], float)
    frac = np.abs(inv) / maxpos
    # AR(1) coefficient of per-minute inventory, pooled over MMs (1 = no reversion).
    x, y = inv[:-1].ravel(), inv[1:].ravel()
    x, y = x - x.mean(), y - y.mean()
    phi = float((x @ y) / (x @ x)) if (x @ x) > 0 else None
    return {
        "seed": seed, "k_ticks": k_ticks,
        "participation_quiet": part(q), "participation_news": part(nw),
        "news_bar_fraction": float(nw.sum() / keep.sum()),
        "parents": len(parents), "parent_status": status,
        "parent_fill_ratio": float(filled.sum() / planned.sum()) if planned.sum() else None,
        "parent_shortfall_share": float(np.sum([p.shortfall for p in parents]) / planned.sum()) if planned.sum() else None,
        "child_fill_rate": float(filled.sum() / sent.sum()) if sent.sum() else None,
        "cap_binding_fraction": bind, "children": len(caps),
        "whale_share_of_sim_volume": float(whale_print[keep].sum() / vol[keep].sum()) if vol[keep].sum() else None,
        "mm_mean_abs_skew_ticks": float(np.mean(np.abs([
            np.clip(-np.round(ex.CONFIG["mm_inventory_skew"]["k_ticks"] * inv / maxpos), -3, 3)]))),
        "sim_over_expected_nse_volume": float(vol[keep].sum() / exp_vol[keep].sum()),
        "whale_messages": whale_msgs,
        "mm_mean_abs_inventory_frac": float(frac.mean()),
        "mm_time_at_limit_frac": float(np.mean(frac >= 0.9)),
        "mm_inventory_ar1": phi,
        "price_change": float(np.log(bars["c"][-1] / bars["c"][keep][0])),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--price", type=float, default=1352.0)
    a = ap.parse_args(argv)
    jobs = [(s, a.days, a.price, k) for s in range(1, a.seeds + 1) for k in (3, 0)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_run, *zip(*jobs)))
    out = os.path.join(HERE, "reports", f"phase4_diagnostics_{time.strftime('%Y%m%dT%H%M%S')}.json")
    with open(out, "w") as fh:
        json.dump({"seeds": a.seeds, "days": a.days, "warmup_days": WARMUP_DAYS,
                   "seconds": time.time() - t0, "runs": res}, fh, indent=1)
    for k in (3, 0):
        rs = [r for r in res if r["k_ticks"] == k]
        med = lambda key: np.median([r[key] for r in rs if r[key] is not None])
        print(f"skew k={k}: part quiet {med('participation_quiet'):.4f} news {med('participation_news'):.4f} "
              f"| fill ratio {med('parent_fill_ratio'):.3f} child fill {med('child_fill_rate'):.3f} "
              f"cap binds {med('cap_binding_fraction'):.3f} | whale share {med('whale_share_of_sim_volume'):.3f} "
              f"sim/NSE vol {med('sim_over_expected_nse_volume'):.4f} | MM |inv| {med('mm_mean_abs_inventory_frac'):.3f} "
              f"at-limit {med('mm_time_at_limit_frac'):.3f} AR1 {med('mm_inventory_ar1'):.4f} skew {med('mm_mean_abs_skew_ticks'):.4f}")
    print("report:", out)


if __name__ == "__main__":
    main()
