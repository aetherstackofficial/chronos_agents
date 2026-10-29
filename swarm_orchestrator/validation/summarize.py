"""Render a report card as a markdown table: python -m validation.summarize <report.json>"""
from __future__ import annotations

import json
import sys


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3g}"
    return str(x)


def render(r: dict) -> str:
    lines = [f"# Baseline report card ({r['run']['mode']}, {r['simulator']['instrument']})", "",
             f"Commit {r['simulator']['git_commit']} (dirty={r['simulator']['working_tree_dirty']}), "
             f"seeds {r['run']['n_seeds']}, N/seed {r['run']['n_returns_per_seed']}, "
             f"overall **{r['overall']['status']}** {r['overall']}", "",
             "| Test | Spec status | Seed pass rate | Real-band status | Failing criteria (pass rate) |",
             "|---|---|---|---|---|"]
    for name, t in r["tests"].items():
        crit = t.get("criteria_pass_rate") or {}
        bad = ", ".join(f"{k} ({v:.2f})" for k, v in crit.items() if v < 0.9)
        lines.append(f"| {name} | {t.get('status')} | {_fmt(t.get('pass_rate'))} | "
                     f"{_fmt(t.get('real_band_status'))} | {bad or '-'} |")
    lines += ["", "## Simulator vs real NSE (trade-price closes)", "",
              "| Test | Metric | Sim median | Real median | Real band | Sim seeds in band |",
              "|---|---|---|---|---|---|"]
    for name, t in r["tests"].items():
        for mn, c in (t.get("real_band") or {}).items():
            lines.append(f"| {name} | {mn} | {_fmt(c['sim_median'])} | {_fmt(c['real_median'])} | "
                         f"[{_fmt(c['band'][0])}, {_fmt(c['band'][1])}] | {_fmt(c['sim_in_band_rate'])} |")
    return "\n".join(lines)


if __name__ == "__main__":
    with open(sys.argv[1]) as fh:
        print(render(json.load(fh)))
