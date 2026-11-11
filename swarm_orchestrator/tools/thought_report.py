"""Turn a decision log (thought log JSONL) into one self-contained HTML per agent.

    python -m tools.thought_report <file.jsonl | run_dir> [--out file.html]

No external files: the chart is inline SVG.  One HTML is written per agent; with
--out and several agents the agent id is appended to the file stem.

Memory levels are read tolerantly from ``saw.memory`` (its exact shape is owned by
MarketMemory.summary()): keys such as nearest_above / levels_above / above /
resistance, and nearest_below / levels_below / below / support (a number, a dict
with price/level, or a list of those), or a generic ``levels`` list that is split
around ``last_price``.  Anything not found is simply not drawn.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import sys

MPD = 390  # market minutes per day (same default as ThoughtLog)
_ABOVE = ("nearest_above", "level_above", "levels_above", "above", "nearest_resistance",
          "resistance", "resistances")
_BELOW = ("nearest_below", "level_below", "levels_below", "below", "nearest_support",
          "support", "supports")
_PRICE_OBS = ("last_price", "close", "price")


def load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prices(v) -> list[float]:
    out: list[float] = []
    if isinstance(v, bool):
        return out
    if isinstance(v, (int, float)):
        out.append(float(v))
    elif isinstance(v, dict):
        for k in ("price", "level", "p", "value"):
            if isinstance(v.get(k), (int, float)):
                out.append(float(v[k]))
                break
    elif isinstance(v, (list, tuple)):
        for x in v:
            out.extend(_prices(x))
    return out


def _abs(rec: dict, mpd: int = MPD) -> int:
    w = rec["when"]
    return int(w["day"]) * mpd + int(w["sim_minute"])


def _price(rec: dict):
    mem = (rec.get("saw") or {}).get("memory") or {}
    p = mem.get("last_price")
    if isinstance(p, (int, float)):
        return float(p)
    obs = (rec.get("saw") or {}).get("obs") or {}
    for k in _PRICE_OBS:
        if isinstance(obs.get(k), (int, float)):
            return float(obs[k])
    return None


def _levels(rec: dict, price):
    mem = (rec.get("saw") or {}).get("memory") or {}
    above = [x for k in _ABOVE for x in _prices(mem.get(k))]
    below = [x for k in _BELOW for x in _prices(mem.get(k))]
    lv = mem.get("levels")
    if isinstance(lv, dict):  # MarketMemory.summary(): {"above": [{price,...}], "below": [...]}
        above += _prices(lv.get("above"))
        below += _prices(lv.get("below"))
        lv = None
    gen = _prices(lv)
    if price is not None:
        above += [x for x in gen if x > price]
        below += [x for x in gen if x <= price]
        above = [x for x in above if x > price] or above
        below = [x for x in below if x < price] or below
        up = min(above, key=lambda x: abs(x - price)) if above else None
        dn = min(below, key=lambda x: abs(x - price)) if below else None
    else:
        up = min(above) if above else None
        dn = max(below) if below else None
    return up, dn


def _action(rec: dict) -> tuple[str, str]:
    """-> (marker kind: buy|sell|quote|hold, short text)."""
    orders = (rec.get("chose") or {}).get("orders") or []
    sides = {str(o.get("side", "")).upper() for o in orders}
    bits = []
    for o in orders[:4]:
        bits.append(f"{str(o.get('side', '?')).upper()} {o.get('qty', '')}"
                    + (f" @{o['price']:.2f}" if isinstance(o.get("price"), (int, float)) else ""))
    if not orders:
        kind, txt = "hold", "no order"
    elif sides == {"BUY"}:
        kind, txt = "buy", "; ".join(bits)
    elif sides == {"SELL"}:
        kind, txt = "sell", "; ".join(bits)
    else:
        kind, txt = "quote", "; ".join(bits)
    return kind, txt


def _num(x, nd=3):
    return "-" if x is None else f"{x:.{nd}g}"


def _sure_text(rec: dict) -> str:
    s = rec.get("sure")
    if not s:
        return "-"
    parts = []
    if s.get("value") is not None:
        parts.append(f"value {_num(s['value'])}")
    if s.get("q_chosen") is not None:
        parts.append(f"Q {_num(s['q_chosen'])}")
    if s.get("q_gap") is not None:
        parts.append(f"vs hold {s['q_gap']:+.3g}")
    if s.get("action_std"):
        parts.append(f"std {sum(s['action_std']) / len(s['action_std']):.2f}")
    return ", ".join(parts) or "-"


def _why_text(rec: dict) -> str:
    w = rec.get("why")
    if not w:
        return "-"
    out = []
    for e in w["combined"][:3]:
        arrow = "&#9650;" if e["sign"] > 0 else ("&#9660;" if e["sign"] < 0 else "&middot;")
        out.append(f"{html.escape(e['phrase'])} <span class='dim'>({arrow}{e['total_abs']:.2g})</span>")
    return "<br>".join(out)


def _then_text(rec: dict, key: str) -> str:
    o = (rec.get("then") or {}).get(key)
    if not o:
        return "-"
    pnl = "n/a" if o.get("pnl") is None else f"{o['pnl']:+.2f}"
    star = "" if o.get("complete") else "*"
    return f"{pnl}{star}<br><span class='dim'>{o.get('n_fills', 0)} fills</span>"


def _chart(rows: list[dict]) -> str:
    W, H, L, R, T, B = 980, 360, 62, 14, 14, 34
    xs = [_abs(r) for r in rows]
    pts = [(x, _price(r)) for x, r in zip(xs, rows)]
    lv = [(x, *_levels(r, p)) for (x, p), r in zip(pts, rows)]
    ys = [p for _, p in pts if p is not None]
    ys += [v for _, u, d in lv for v in (u, d) if v is not None]
    if not ys:
        return "<p class='dim'>No price or level data in these records (memory.last_price missing).</p>"
    y0, y1 = min(ys), max(ys)
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 1, y1 + 1
    pad = (y1 - y0) * 0.06
    y0, y1 = y0 - pad, y1 + pad
    xmin, xmax = min(xs), max(xs)
    if xmax == xmin:
        xmax = xmin + 1

    def X(x):
        return L + (x - xmin) / (xmax - xmin) * (W - L - R)

    def Y(y):
        return T + (y1 - y) / (y1 - y0) * (H - T - B)

    s = [f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' role='img' "
         f"aria-label='price, memory levels and decisions'>"]
    for i in range(5):  # y grid
        v = y0 + (y1 - y0) * i / 4
        s.append(f"<line x1='{L}' x2='{W - R}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' class='grid'/>"
                 f"<text x='{L - 6}' y='{Y(v) + 4:.1f}' class='ax' text-anchor='end'>{v:.2f}</text>")
    for i in range(6):  # x ticks
        x = xmin + (xmax - xmin) * i / 5
        d, m = divmod(int(x), MPD)
        s.append(f"<text x='{X(x):.1f}' y='{H - 12}' class='ax' text-anchor='middle'>d{d} m{m}</text>")
    # level marks: each record's nearest levels span until the next record
    for i, (x, u, d) in enumerate(lv):
        xe = lv[i + 1][0] if i + 1 < len(lv) else xmax
        xe = max(xe, x + (xmax - xmin) * 0.01)
        if u is not None:
            s.append(f"<line x1='{X(x):.1f}' x2='{X(xe):.1f}' y1='{Y(u):.1f}' y2='{Y(u):.1f}' class='lvu'/>")
        if d is not None:
            s.append(f"<line x1='{X(x):.1f}' x2='{X(xe):.1f}' y1='{Y(d):.1f}' y2='{Y(d):.1f}' class='lvd'/>")
    line = " ".join(f"{X(x):.1f},{Y(p):.1f}" for x, p in pts if p is not None)
    if line:
        s.append(f"<polyline points='{line}' class='px'/>")
    for (x, p), r in zip(pts, rows):
        if p is None:
            continue
        kind, txt = _action(r)
        cx, cy = X(x), Y(p)
        tip = html.escape(f"d{r['when']['day']} m{r['when']['sim_minute']}: {kind} {txt}")
        if r.get("kind") == "heartbeat" and kind == "hold":
            m = f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='2' class='mhb'/>"
        elif kind == "buy":
            m = f"<path d='M{cx:.1f} {cy - 8:.1f} l6 11 h-12z' class='mbuy'/>"
        elif kind == "sell":
            m = f"<path d='M{cx:.1f} {cy + 8:.1f} l6 -11 h-12z' class='msell'/>"
        elif kind == "quote":
            m = f"<path d='M{cx:.1f} {cy - 6:.1f} l6 6 l-6 6 l-6 -6z' class='mq'/>"
        else:
            m = f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='4' class='mhold'/>"
        s.append(f"<g>{m}<title>{tip}</title></g>")
    s.append("</svg>")
    return "".join(s)


_CSS = """
:root{--bg:#fff;--fg:#1d2433;--dim:#6b7385;--grid:#e3e6ee;--px:#2d5bd7;--up:#c2410c;--dn:#0f766e;
--buy:#15803d;--sell:#b91c1c;--q:#7c3aed;--card:#f6f7fb}
@media(prefers-color-scheme:dark){:root{--bg:#14171f;--fg:#e6e9f2;--dim:#9aa3b8;--grid:#2a3040;--px:#7aa2ff;
--up:#fb923c;--dn:#2dd4bf;--buy:#4ade80;--sell:#f87171;--q:#a78bfa;--card:#1c2130}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,sans-serif;margin:0;padding:16px}
h1{font-size:20px;margin:0 0 4px}.dim{color:var(--dim)}
.meta{margin-bottom:12px}svg{width:100%;height:auto;background:var(--card);border-radius:8px}
.grid{stroke:var(--grid)}.ax{fill:var(--dim);font-size:11px}.px{fill:none;stroke:var(--px);stroke-width:1.6}
.lvu{stroke:var(--up);stroke-width:1.6;stroke-dasharray:5 3}.lvd{stroke:var(--dn);stroke-width:1.6;stroke-dasharray:5 3}
.mbuy{fill:var(--buy)}.msell{fill:var(--sell)}.mq{fill:var(--q)}.mhold{fill:var(--dim)}.mhb{fill:var(--dim)}
.legend span{margin-right:16px;white-space:nowrap}
table{border-collapse:collapse;width:100%;margin-top:14px;font-size:13px}
th,td{border-bottom:1px solid var(--grid);padding:5px 8px;text-align:left;vertical-align:top}
th{position:sticky;top:0;background:var(--bg)}tr.hb td{color:var(--dim)}
.buy{color:var(--buy);font-weight:600}.sell{color:var(--sell);font-weight:600}.quote{color:var(--q);font-weight:600}
.wrap{overflow-x:auto}
"""


def build_report(path: str) -> str:
    rows = load(path)
    if not rows:
        return "<!doctype html><html><body><p>empty log</p></body></html>"
    w = rows[0]["when"]
    n_ch = sum(1 for r in rows if r.get("kind") == "change")
    head = (f"<h1>Decisions of {html.escape(str(w['agent_id']))}</h1>"
            f"<div class='meta dim'>run {html.escape(str(w['run_id']))} &middot; model "
            f"{html.escape(str(w['model']))} ({html.escape(str(w['algo']))}) &middot; "
            f"{len(rows)} records, {n_ch} changes &middot; mode "
            f"{html.escape(str(w['mode']))}</div>")
    legend = ("<div class='legend dim'>"
              "<span><b style='color:var(--px)'>&mdash;</b> price</span>"
              "<span><b style='color:var(--up)'>- - -</b> nearest level above</span>"
              "<span><b style='color:var(--dn)'>- - -</b> nearest level below</span>"
              "<span><b style='color:var(--buy)'>&#9650;</b> buy</span>"
              "<span><b style='color:var(--sell)'>&#9660;</b> sell</span>"
              "<span><b style='color:var(--q)'>&#9670;</b> quotes</span>"
              "<span>&#9679; other change / heartbeat</span></div>")
    trs = []
    for r in rows:
        kind, txt = _action(r)
        wr = r["when"]
        intent = json.dumps((r.get("chose") or {}).get("intent"), separators=(",", ":"))
        notes = "<br>".join(html.escape(str(n)) for n in (r.get("chose") or {}).get("notes") or [])
        trs.append(
            f"<tr class='{'hb' if r.get('kind') == 'heartbeat' else ''}'>"
            f"<td>d{wr['day']} m{wr['sim_minute']}<br><span class='dim'>{r.get('kind')}</span></td>"
            f"<td><span class='{kind}'>{kind}</span> {html.escape(txt)}"
            f"<br><span class='dim'>{html.escape(intent[:120])}</span></td>"
            f"<td>{notes or '-'}</td><td>{_why_text(r)}</td><td>{html.escape(_sure_text(r))}</td>"
            f"<td>{_then_text(r, 'm15')}</td><td>{_then_text(r, 'm60')}</td></tr>")
    table = ("<div class='wrap'><table><tr><th>time</th><th>action</th><th>decoder notes</th>"
             "<th>top reasons (what pushed it)</th><th>value / sure</th><th>after 15 min</th>"
             "<th>after 60 min</th></tr>" + "".join(trs) + "</table></div>"
             "<p class='dim'>* outcome incomplete when the log was closed. Heartbeat rows skip the "
             "reasons to save time.</p>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Decisions of {html.escape(str(w['agent_id']))}</title><style>{_CSS}</style></head>"
            f"<body>{head}{legend}{_chart(rows)}{table}</body></html>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools.thought_report", description=__doc__.split("\n")[0])
    ap.add_argument("path", help="a .jsonl file or a run directory of them")
    ap.add_argument("--out", help="output html file (agent id appended if several agents)")
    a = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.path, "*.jsonl"))) if os.path.isdir(a.path) else [a.path]
    if not files:
        print("no .jsonl files found", file=sys.stderr)
        return 1
    for f in files:
        agent = os.path.splitext(os.path.basename(f))[0]
        if a.out and len(files) == 1:
            out = a.out
        elif a.out:
            stem, ext = os.path.splitext(a.out)
            out = f"{stem}_{agent}{ext or '.html'}"
        else:
            out = os.path.splitext(f)[0] + ".html"
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(build_report(f))
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
