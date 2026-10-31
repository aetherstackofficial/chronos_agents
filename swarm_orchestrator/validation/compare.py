"""Compare two report cards: python -m validation.compare <before.json> <after.json>

Prints status changes per test and the key real-band metrics side by side,
with the distance to the real median, so "moved the right way" is explicit.
"""
from __future__ import annotations

import json
import sys

from validation.runner import KEY_METRICS


def _f(x):
    return "-" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))


def compare(a: dict, b: dict) -> str:
    out = ["| Test | Before | After | Seed pass rate before -> after |", "|---|---|---|---|"]
    for name in b["tests"]:
        ta, tb = a["tests"].get(name, {}), b["tests"][name]
        out.append(f"| {name} | {ta.get('status')} | {tb.get('status')} | "
                   f"{_f(ta.get('pass_rate'))} -> {_f(tb.get('pass_rate'))} |")
    out += ["", "| Test | Metric | Real median | Before | After | Closer to real? |", "|---|---|---|---|---|---|"]
    for name, metrics in KEY_METRICS.items():
        ra = (a["tests"].get(name) or {}).get("real_band") or {}
        rb = (b["tests"].get(name) or {}).get("real_band") or {}
        for mn in metrics:
            if mn not in rb:
                continue
            real, va, vb = rb[mn]["real_median"], (ra.get(mn) or {}).get("sim_median"), rb[mn]["sim_median"]
            closer = None if va is None else abs(vb - real) < abs(va - real)
            out.append(f"| {name} | {mn} | {_f(real)} | {_f(va)} | {_f(vb)} | "
                       f"{'yes' if closer else ('no' if closer is not None else '-')} |")
    # Spec-only (uncalibrated) metrics of the impact/book tests.
    out += ["", "| Test | Metric (median over seeds) | Before | After |", "|---|---|---|---|"]
    for name in ("T5_book_shape", "T7_sqrt_impact", "T7a_book_walk", "T8_monotonic_impact",
                 "A6_impact_decay", "A7_spread_depth_recovery", "A2_order_flow_sign_memory"):
        ma = (a["tests"].get(name) or {}).get("metrics") or {}
        mb = (b["tests"].get(name) or {}).get("metrics") or {}
        for mn, s in mb.items():
            out.append(f"| {name} | {mn} | {_f((ma.get(mn) or {}).get('median'))} | {_f(s.get('median'))} |")
    return "\n".join(out)


if __name__ == "__main__":
    with open(sys.argv[1]) as fa, open(sys.argv[2]) as fb:
        print(compare(json.load(fa), json.load(fb)))
