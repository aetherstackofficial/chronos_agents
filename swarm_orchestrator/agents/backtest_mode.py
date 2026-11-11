"""Backtest mode: the real tape keeps running, the participants leave a footprint on it (opt-in).

engine/matching_engine.py is NOT modified: ``install()`` wraps ``MatchingEngine._refresh_sim_lp`` in
the current process only (agents.live_agents_v2.install() calls it; so does tools/run_engine_backtest.py).
Without ``CHRONOS_MODE=backtest`` it does nothing at all.

What the engine does today: until a user strategy trades, the real bars are replayed; after its first
fill the real data is dropped and the agents alone make the price. Here the real market keeps running.

What the real data supplies: each minute's RELATIVE move (the bar's open, high, low, close divided by
the previous real close) and its volume. It never supplies the price LEVEL: the level is the
simulator's own and starts where the simulator starts (``base`` = log of simulator price over real
price at activation). The tape is the real sessions that follow the agents' seeded memory
(CHRONOS_V2_SEED_SKIP), or, after a user's fill latched a replay, the rest of the replayed file.

One minute, four steps (all inside the engine's ``_refresh_sim_lp`` call of the minute):

1. Live flow. The trades of the live second that just ended (the engine's ``_pending_trades``) whose
   aggressor is not an internal agent (SIM_LP, GHOST_HIST) give the net bought volume n = B - S and the
   buyers' / sellers' average price.
2. Footprint. One consumed-volume number X (shares, signed) is split into a permanent part (share
   CHRONOS_BT_PERM, never refilled) and a temporary part that refills with half-life
   CHRONOS_BT_HALFLIFE minutes. The live flow adds to X, but never further than the point where the
   price level would sit above the highest price the buyers paid in that minute (below the lowest
   price the sellers received, for a sell): you cannot push the level beyond the prices you traded
   at. The level offset is u(X) = sign(X) Y sigma_d sqrt(|X| / adv) (square-root law).
3. News. The real bar, multiplied by exp(base + u), is printed through the book exactly as the
   history replayer prints a bar (o -> h -> l -> c, exactly the bar's volume, first print at the
   open). Part of the footprint therefore carries into every following real bar and part fades.
4. Book. Liquidity is posted around the new price: SIM_LP quotes at tick distances 1..12, then
   growing by 25 % up to the distance where the displayed depth is CHRONOS_BT_DEPTH_ADV of the average
   daily volume. The cumulative displayed depth within d ticks is F(d) = adv (d tick / (P Y sigma_d))^2,
   the same curve that defines u(X): a market order of Q shares walks to a log distance of about
   u(Q). Sizes follow the time-of-day depth profile of the instrument. Quotes never cross other
   participants' resting orders.

Scale (sigma_d: daily volatility, adv: average daily volume in shares, tod: time-of-day depth) is taken
from data BEFORE the tape (CHRONOS_BT_SCALE_SESSIONS, default 250 sessions), or, after a latch, from
the bars replayed so far (tod flat). It is computed once per engine, never per tick.

Parameters (environment, read when an engine's state is created):
  CHRONOS_MODE=backtest      switch the mode on
  CHRONOS_BT_SYMBOL          instrument (default CHRONOS_V2_SEED_SYMBOL, default RELIANCE)
  CHRONOS_BT_SKIP            number of real sessions of the tape (default CHRONOS_V2_SEED_SKIP; > 0)
  CHRONOS_BT_SCALE_SESSIONS  sessions before the tape used for the scale (250)
  CHRONOS_BT_TAPE=<key>      use a tape given to register_tape() instead (tests, synthetic data)
  CHRONOS_BT_Y (1.0)         square-root-law coefficient
  CHRONOS_BT_PERM (0.25)     share of consumed volume that is never refilled
  CHRONOS_BT_HALFLIFE (10)   minutes, half-life of the refill of the rest
  CHRONOS_BT_DEPTH_ADV (.25) displayed book reaches the distance where its depth is this share of adv
  CHRONOS_BT_LEVELS (40)     most displayed levels per side
  CHRONOS_BT_TOD (1)         use the time-of-day depth profile for displayed sizes
  CHRONOS_BT_FOOTPRINT (1)   0: the level never shifts (plain replay, for ablation)
  CHRONOS_BT_CAP (2)         2: cap at the highest price paid (lowest received); 1: at the average price
                             paid (received); 0: no cap (tests / ablation only)
  CHRONOS_BT_AGENT_FP (all)  what enters the footprint: all | off | diff (see "Replace, don't add")
  CHRONOS_BT_VOLBUDGET (0)   1: the agents' aggressive volume is taken out of the printed tape volume;
                             2: also the agents' resting orders that a leg's ghost sweeps (see below)
  CHRONOS_BT_VOLBUDGET_MAXCUT (.9)  most of a bar's volume that may be taken out
  CHRONOS_BT_DEPTHBUDGET (0) 1: the latent book posts only what the agents' resting orders do not provide

Replace, don't add: simulated agents (market makers, whales, retail; every non-internal id that is not
STRAT_* / USER_*) stand for part of the real market, which the tape already contains. Three opt-in rules
make them REPLACE part of the tape instead of adding to it. The user (STRAT_*, USER_*) is always extra.
 - Volume: the real tape holds all real volume. AGENT_FP/VOLBUDGET=1: each minute the agents' aggressive
   volume (bought + sold, never the user's, never internal) goes to a debt; the printed bar volume is cut
   by min(debt, MAXCUT * v) and the debt shrinks by the cut (what is not cut carries to later bars).
   The user's volume is never subtracted: a user's order is volume the real market did not have.
   VOLBUDGET=2 adds passive sweeps: the ghost of a leg (target, qty) also fills every AGENT order resting
   inside the walked range, on top of qty. Before the leg, the swept quantity is taken out of qty (at least
   one ladder share stays, so the print still lands on the target); what does not fit goes to the debt.
 - Liquidity: the latent book already holds the calibrated depth F(d). DEPTHBUDGET=1: the cumulative
   resting quantity of AGENT orders within d ticks (bids >= ref - d ticks, asks <= ref + d ticks; orders
   at or beyond the reference count everywhere) is subtracted: latent size at level d =
   max(0, F(d) - agent_cum(d) - latent already posted on that side). Total depth follows F(d) unless the
   agents alone exceed it. A side/level with no agent order within d keeps today's sizing exactly.
 - Price: the real price path already contains the effect of the real participants' normal trading, so
   their normal trading must not move the level again; only what the user causes may. Footprint input
   per minute: n_user (user aggressors) and n_agent (agent aggressors), net shares bought.
   all  (default): n = n_user + n_agent, cap prices of all non-internal aggressors (today's behaviour).
   off:  n = n_user, cap prices of user trades only. The agents never move the level.
   diff: n = n_user + (n_agent - baseline[tick_count]); baseline = the agents' flow at the same engine
         tick_count in a no-user run from the same snapshot (agent_flow_log(engine) of that run, fed to
         set_agent_baseline(engine, mapping)). Without a baseline value for the tick, baseline = n_agent
         (zero differential): diff then acts like off, and a no-user run only moves with the real bars.
         Cap prices of all non-internal aggressors (a differential with no trade on its side is capped
         at the other side's worst price; with no trade at all it does not move the level).
 The agents' net flow of every footprint step is logged in the state by engine.tick_count (latest 5000).

Cap and slicing: capping at the AVERAGE price paid (CAP=1) lets a single block that walks the book move the
level only to (2/3) u(Q), i.e. consume 4/9 of Q, while the same volume in small slices consumes all of it
(each slice is bought near the level the previous one left). CAP=2 (the default since 2026-11-11) caps at
the highest price paid, which leaves a block that walks the book uncapped (consumed = Q, like the slices)
and still stops a buyer who pays a flat third-party ask from moving the level beyond that ask. Without any
cap such a buyer could buy from an agent at the touch and sell back above the price paid (unit test).

Known limits: passive fills are as generous as in the engine's REPLAY mode (an order is filled in full
when the printed tape trades through it); a market order larger than the displayed depth is partly
discarded by the engine (the remainder of a market order never rests); overnight decay of the
footprint is not modelled (the temporary part decays per minute, sessions are simply consecutive);
circuit limits are not modelled; if the tape is exhausted (environment tape) the previous liquidity
refresh takes over and the market free-runs.

State: ``engine.__dict__["_bt"]`` is a small dict of plain numbers and strings (the engine is deep-copied
by snapshot/restore); tapes live in a module-level cache.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional

logger = logging.getLogger("agents.backtest_mode")

_INSTALLED = False
_TAPES: Dict[str, dict] = {}          # tape key -> tape (numpy arrays + python lists)
_DIST_CACHE: Dict[tuple, tuple] = {}  # (d_max, levels) -> tick distances of the displayed levels


# --------------------------------------------------------------------------- #
# the curve
# --------------------------------------------------------------------------- #
def curve_u(x: float, st: dict) -> float:
    """Log-price offset of consumed volume ``x`` (signed shares)."""
    if x == 0:
        return 0.0
    return math.copysign(st["Y"] * st["sigma_d"] * math.sqrt(abs(x) / st["adv"]), x)


def curve_x(u: float, st: dict) -> float:
    """Inverse of curve_u: consumed volume (signed shares) that moves the level by ``u``."""
    if u == 0:
        return 0.0
    return math.copysign(st["adv"] * (u / (st["Y"] * st["sigma_d"])) ** 2, u)


# --------------------------------------------------------------------------- #
# tapes
# --------------------------------------------------------------------------- #
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def _make_tape(o, h, l, c, v, prev_close, sigma_d, adv, tod, dates, symbol, skip) -> dict:
    import numpy as np
    o, h, l, c, v = (np.asarray(a, dtype=float) for a in (o, h, l, c, v))
    return {"o": o, "h": h, "l": l, "c": c, "v": v, "prev_close": float(prev_close),
            "sigma_d": float(sigma_d), "adv": float(adv), "dates": dates,
            "symbol": symbol, "skip": skip, "n": len(c),
            "tod": None if tod is None else [float(x) for x in tod],
            "_o": o.tolist(), "_h": h.tolist(), "_l": l.tolist(), "_c": c.tolist(), "_v": v.tolist()}


def register_tape(key: str, bars, prev_close: float, sigma_d: float, adv: float, tod=None) -> None:
    """Register a tape under ``key`` (select it with CHRONOS_BT_TAPE=<key>). ``bars``: a list of dicts
    with o, h, l, c, v, or a dict of equally long arrays o, h, l, c, v. ``tod``: optional depth profile."""
    if isinstance(bars, dict):
        cols = [bars[k] for k in "ohlcv"]
    else:
        cols = [[b[k] for b in bars] for k in "ohlcv"]
    _TAPES[key] = _make_tape(*cols, prev_close, sigma_d, adv, tod, None, key, 0)


def _depth_profile(c2, v2, window: int = 15):
    """Time-of-day depth, as agents.latent_liquidity.real_depth_profile but on the given sessions only:
    mean volume(m) / mean |1-min return|(m)^2, smoothed over ``window`` minutes, mean 1, clipped."""
    import numpy as np
    n = c2.shape[1]
    r = np.abs(np.diff(np.log(c2), axis=1)).mean(axis=0)          # return into minute m = 1..n-1
    absr = np.r_[r[0], r]                                         # minute 0 has none: copy minute 1
    vol = v2.mean(axis=0)
    k = np.ones(window) / window
    pad = window // 2
    sm = lambda x: np.convolve(np.r_[np.full(pad, x[0]), x, np.full(pad, x[-1])], k, "valid")[:n]
    prof = sm(vol) / np.maximum(sm(absr), 1e-12) ** 2
    prof = prof / prof.mean()
    return np.clip(prof, 0.25, 4.0)


def _load_real_tape(symbol: str, skip: int, scale_sessions: int) -> dict:
    import numpy as np
    from validation.data import load_real
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "data", "real", "nse_kaggle", f"{symbol}_minute.csv")
    s = load_real(path)
    n_min = 375
    n_days = len(s.c) // n_min
    if n_days * n_min != len(s.c):
        raise RuntimeError(f"{symbol}: cleaned sessions are not all {n_min} bars")
    o2, h2, l2, c2, v2 = (a.reshape(n_days, n_min) for a in (s.o, s.h, s.l, s.c, s.v))
    if skip >= n_days - 2:
        raise ValueError(f"{symbol}: CHRONOS_BT_SKIP={skip} leaves no scale sessions ({n_days} sessions)")
    pre = np.arange(n_days - skip)                       # sessions before the tape
    sc = pre[-max(int(scale_sessions), 2):]              # scale sessions (the last of them is prev_close's)
    daily_close = c2[sc, -1]
    sigma_d = float(np.std(np.diff(np.log(daily_close)), ddof=1))
    adv = float(v2[sc].sum(axis=1).mean())
    tod = _depth_profile(c2[sc], v2[sc])
    tp = np.arange(n_days - skip, n_days)                # tape sessions
    dates = [s.info["dates"][d] for d in tp]
    return _make_tape(o2[tp].ravel(), h2[tp].ravel(), l2[tp].ravel(), c2[tp].ravel(), v2[tp].ravel(),
                      c2[pre[-1], -1], sigma_d, adv, tod, dates, symbol, skip)


def _env_tape_key() -> Optional[str]:
    """Key of the tape the environment asks for, or None (no tape: free-run)."""
    key = os.environ.get("CHRONOS_BT_TAPE")
    if key:
        return key
    try:
        skip = int(os.environ.get("CHRONOS_BT_SKIP") or os.environ.get("CHRONOS_V2_SEED_SKIP") or 0)
    except ValueError:
        skip = 0
    if skip <= 0:
        return None
    sym = os.environ.get("CHRONOS_BT_SYMBOL") or os.environ.get("CHRONOS_V2_SEED_SYMBOL") or "RELIANCE"
    scale = int(_env_float("CHRONOS_BT_SCALE_SESSIONS", 250))
    return f"real:{sym}:{skip}:{scale}"


def _tape_for_key(key: str) -> dict:
    tp = _TAPES.get(key)
    if tp is None:
        if not key.startswith("real:"):
            raise KeyError(f"CHRONOS_BT_TAPE={key!r}: no such tape (register_tape() it first)")
        _, sym, skip, scale = key.split(":")
        tp = _TAPES[key] = _load_real_tape(sym, int(skip), int(scale))
    return tp


def env_tape() -> dict:
    """The tape of the current environment: arrays o, h, l, c, v (all tape sessions, 375 bars each for
    real data), prev_close, dates (of the tape sessions), sigma_d, adv, symbol, skip."""
    key = _env_tape_key()
    if key is None:
        raise ValueError("no tape: set CHRONOS_BT_TAPE or CHRONOS_BT_SKIP / CHRONOS_V2_SEED_SKIP > 0")
    tp = _tape_for_key(key)
    out = {k: v for k, v in tp.items() if not k.startswith("_")}
    out["tod"] = None if tp["tod"] is None else list(tp["tod"])
    return out


MIN_SCALE_BARS = 30        # fewer replayed bars than this: size the market from the tape's first session instead


def _scale_from_bars(bars: List[dict], upto: int, session: int):
    """(sigma_d, adv) from the first ``upto`` replayed bars: daily log returns and mean daily volume
    over complete days (grouped by ``ts`` when present, else consecutive ``session``-bar chunks); with
    fewer than 3 complete days, from 1-minute returns scaled by sqrt(session) and mean bar volume x session.

    With fewer than MIN_SCALE_BARS replayed (a hand-over in the first minutes) one or two bars cannot size the
    market, so the tape's first session (at least MIN_SCALE_BARS bars) is used instead. This reads ahead only
    the SIZE of the market (how volatile, how much volume), never which way the price goes, and only at a hand-over
    in the first minutes. Bars with no volume (Yahoo has a few) are left out of the volume mean. Without both
    fixes a hand-over at the first bar, whose volume was 0, gave adv = 1 share: a 1-share latent book and a
    footprint that moved TCS ~5% off the real price."""
    import numpy as np
    n = upto if upto >= MIN_SCALE_BARS else min(len(bars), max(upto, session, MIN_SCALE_BARS))
    seen = bars[:n]
    c = np.array([b["c"] for b in seen], dtype=float)
    v = np.array([b["v"] for b in seen], dtype=float)
    vpos = v[v > 0]
    sigma_d = adv = None
    if len(c):
        if all(b.get("ts") is not None for b in seen):
            day = np.array([b["ts"] // 86400 for b in seen])
            _, idx, cnt = np.unique(day, return_index=True, return_counts=True)
            groups = [(i, i + k) for i, k in zip(idx, cnt) if k == session]
        else:
            groups = [(i, i + session) for i in range(0, len(c) - session + 1, session)]
        if len(groups) >= 3:
            dc = np.array([c[b - 1] for _, b in groups])
            sigma_d = float(np.std(np.diff(np.log(dc)), ddof=1))
            adv = float(np.mean([v[a:b].sum() for a, b in groups]))
        elif len(c) >= 3:
            sigma_d = float(np.std(np.diff(np.log(c)), ddof=1)) * math.sqrt(session)
            adv = float(vpos.mean()) * session if len(vpos) else None
    if sigma_d is None:
        sigma_d = 0.015
    if not adv:
        adv = float(vpos.mean()) * session if len(vpos) else 1e6     # no volume in the data at all: a deep market
    return max(sigma_d, 1e-4), max(adv, 1.0)


# --------------------------------------------------------------------------- #
# per-engine state
# --------------------------------------------------------------------------- #
_FLOW_KEEP = 5000                      # most entries of the agents' flow log (the state is deep-copied)
_AGENT_FP_MODES = ("all", "off", "diff")


def _agent_fp_mode() -> str:
    m = os.environ.get("CHRONOS_BT_AGENT_FP", "all").strip().lower()
    return m if m in _AGENT_FP_MODES else "all"


def _params() -> dict:
    halflife = max(_env_float("CHRONOS_BT_HALFLIFE", 10.0), 1e-6)
    return {"Y": _env_float("CHRONOS_BT_Y", 1.0),
            "perm": _env_float("CHRONOS_BT_PERM", 0.25),
            "halflife": halflife, "rho": 0.5 ** (1.0 / halflife),
            "depth_adv": _env_float("CHRONOS_BT_DEPTH_ADV", 0.25),
            "levels": max(int(_env_float("CHRONOS_BT_LEVELS", 40)), 1),
            "tod_on": int(_env_float("CHRONOS_BT_TOD", 1)),
            "footprint": int(_env_float("CHRONOS_BT_FOOTPRINT", 1)),
            "cap": int(_env_float("CHRONOS_BT_CAP", 2)),
            "agent_fp": _agent_fp_mode(),
            "volbudget": int(_env_float("CHRONOS_BT_VOLBUDGET", 0)),
            "volbudget_maxcut": min(max(_env_float("CHRONOS_BT_VOLBUDGET_MAXCUT", 0.9), 0.0), 1.0),
            "depthbudget": int(_env_float("CHRONOS_BT_DEPTHBUDGET", 0))}


def _new_state(src: str, tape_key: Optional[str], sigma_d: float, adv: float, base: float,
               c_prev: float, tick_count: int) -> dict:
    st = {"src": src, "tape": tape_key, "sigma_d": float(sigma_d), "adv": float(adv),
          "base": float(base), "c_prev": float(c_prev), "xp": 0.0, "xt": 0.0, "u": 0.0,
          "n_bars": 0, "n_inexact": 0, "last_tick": tick_count, "exhausted": False,
          "live_buy": 0.0, "live_sell": 0.0, "live_buy_total": 0.0, "live_sell_total": 0.0,
          "n_user": 0.0, "n_agent": 0.0, "agent_vol": 0.0, "agent_aggr_total": 0.0,
          "vol_debt": 0.0, "vol_cut_total": 0.0, "vol_cut_passive_total": 0, "passive_excess_total": 0,
          "latent_posted_total": 0, "depth_displaced_total": 0,
          "agent_flow": {}, "agent_base": {}}
    st.update(_params())
    return st


def get_state(engine) -> Optional[dict]:
    """A copy of the engine's backtest state (None if the mode never ran on it). The agents' flow log and
    baseline are not copied in here (see agent_flow_log); their sizes are ``agent_flow_len`` / ``agent_base_len``."""
    st = engine.__dict__.get("_bt")
    if st is None:
        return None
    out = {k: v for k, v in st.items() if k not in ("agent_flow", "agent_base")}
    out["agent_flow_len"] = len(st["agent_flow"])
    out["agent_base_len"] = len(st["agent_base"])
    return out


def agent_flow_log(engine) -> dict:
    """A copy of {engine.tick_count: net shares bought by AGENT aggressors in that footprint step} (the latest
    5000 steps; empty if the mode never ran on the engine)."""
    st = engine.__dict__.get("_bt")
    return {} if st is None else dict(st["agent_flow"])


def set_agent_baseline(engine, mapping) -> None:
    """Store a copy of ``mapping`` ({tick_count: agents' net flow}, from agent_flow_log of a no-user run
    started from the same snapshot) for AGENT_FP=diff. Kept in the engine, so deepcopy carries it; set
    before the mode's first tick, it is adopted when the state is created."""
    base = {int(k): float(v) for k, v in dict(mapping).items()}
    st = engine.__dict__.get("_bt")
    if st is None:
        engine.__dict__["_bt_baseline"] = base
    else:
        st["agent_base"] = base


def active() -> bool:
    """True once install() has switched the mode on in this process."""
    return _INSTALLED


# --------------------------------------------------------------------------- #
# one tick
# --------------------------------------------------------------------------- #
def _live_flow(engine):
    """(B, S, vwap_buy, vwap_sell, worst_buy, worst_sell) of the non-internal aggressors in the engine's
    pending trades (worst = highest price paid by buyers / lowest price received by sellers)."""
    ledger = engine.agent_ledger
    bq = sq = bn = sn = 0.0
    wb, ws = 0.0, float("inf")
    for t in engine._pending_trades:
        buy = t["aggressor"] == "BUY"
        led = ledger.get(t["buyer"] if buy else t["seller"])
        if led is not None and led["internal"]:
            continue
        q = t["qty"]
        px = t["price"]
        if buy:
            bq += q
            bn += q * px
            if px > wb:
                wb = px
        else:
            sq += q
            sn += q * px
            if px < ws:
                ws = px
    return bq, sq, (bn / bq if bq else 0.0), (sn / sq if sq else 0.0), wb, ws


def _is_user(aid) -> bool:
    """True for the user's strategy ids (STRAT_*, USER_*)."""
    return isinstance(aid, str) and aid.startswith(("STRAT_", "USER_"))


def _live_flow_split(engine):
    """(user, agent): for the user's aggressors (STRAT_*, USER_*) and for the AGENT aggressors (every other
    non-internal id), each (B, S, vwap_buy, vwap_sell, worst_buy, worst_sell) as in _live_flow."""
    ledger = engine.agent_ledger
    acc = {True: [0.0, 0.0, 0.0, 0.0, 0.0, float("inf")], False: [0.0, 0.0, 0.0, 0.0, 0.0, float("inf")]}
    for t in engine._pending_trades:
        buy = t["aggressor"] == "BUY"
        aid = t["buyer"] if buy else t["seller"]
        led = ledger.get(aid)
        if led is not None and led["internal"]:
            continue
        a = acc[_is_user(aid)]
        q = t["qty"]
        px = t["price"]
        if buy:
            a[0] += q
            a[2] += q * px
            if px > a[4]:
                a[4] = px
        else:
            a[1] += q
            a[3] += q * px
            if px < a[5]:
                a[5] = px
    out = []
    for is_user in (True, False):
        bq, sq, bn, sn, wb, ws = acc[is_user]
        out.append((bq, sq, (bn / bq if bq else 0.0), (sn / sq if sq else 0.0), wb, ws))
    return out[0], out[1]


def _footprint(engine, st: dict) -> float:
    """Steps 1-2: fold the live second's flow into the consumed volume; returns the new offset u."""
    B, S, vb, vs, wb, ws = _live_flow(engine)
    st["live_buy"], st["live_sell"] = B, S
    st["live_buy_total"] += B
    st["live_sell_total"] += S
    user, agent = _live_flow_split(engine)
    n_user = user[0] - user[1]
    n_agent = agent[0] - agent[1]
    st["n_user"], st["n_agent"] = n_user, n_agent
    st["agent_vol"] = agent[0] + agent[1]
    st["agent_aggr_total"] += st["agent_vol"]
    log = st["agent_flow"]
    log[engine.tick_count] = n_agent
    while len(log) > _FLOW_KEEP:
        del log[next(iter(log))]
    if not st["footprint"]:
        return 0.0
    mode = st["agent_fp"]
    if mode == "off":
        n = n_user
        vb, vs, wb, ws = user[2], user[3], user[4], user[5]
    elif mode == "diff":
        n = n_user + (n_agent - st["agent_base"].get(engine.tick_count, n_agent))
    else:
        n = B - S
    st["xt"] *= st["rho"]
    x_rel = st["xp"] + st["xt"]
    if n != 0:
        x_new = x_rel + n
        if st["cap"]:
            ref_px = st["c_prev"] * math.exp(st["base"])     # tape level in sim prices in the live second
            if st["cap"] == 2:                               # variant: cap at the worst price, not the average
                vb, vs = wb, ws
            if n > 0:
                px = vb if vb > 0 else (ws if 0 < ws < float("inf") else 0.0)   # diff: no buy -> sellers' worst
                if px > 0:
                    u_cap = max(curve_u(x_rel, st), math.log(px / ref_px))
                    x_new = min(x_new, max(x_rel, curve_x(u_cap, st)))
                else:
                    x_new = x_rel                            # no trade of the minute: no price to stand on
            else:
                px = vs if 0 < vs < float("inf") else (wb if wb > 0 else 0.0)
                if px > 0:
                    u_cap = min(curve_u(x_rel, st), math.log(px / ref_px))
                    x_new = max(x_new, min(x_rel, curve_x(u_cap, st)))
                else:
                    x_new = x_rel
        n_eff = x_new - x_rel
        st["xp"] += st["perm"] * n_eff
        st["xt"] += (1.0 - st["perm"]) * n_eff
    return curve_u(st["xp"] + st["xt"], st)


def _swept_agent_qty(engine, side: str, target: float, ghost: str) -> int:
    """Resting quantity of AGENT orders (not the user's, not internal, not the ghost) that a ``side`` order
    limited at ``target`` takes: asks <= target (BUY), bids >= target (SELL). Read-only."""
    book = engine.book._asks if side == "BUY" else engine.book._bids
    ledger = engine.agent_ledger
    eps = engine.tick * 1e-6
    q = 0
    skip: Dict[str, bool] = {}
    for key in book.keys():                                   # best price first
        price = key if side == "BUY" else -key
        if (price > target + eps) if side == "BUY" else (price < target - eps):
            break
        for o in book[key]:
            aid = o["agent_id"]
            sk = skip.get(aid)
            if sk is None:
                led = ledger.get(aid)
                sk = skip[aid] = (_is_user(aid) or aid == ghost or aid == engine.LP_AGENT
                                  or (led is not None and bool(led["internal"])))
            if not sk:
                q += o["qty"]
    return q


def _print_bar(engine, pubs, st, o, h, l, c, v, f, rp, number) -> None:
    """Step 3: print the real bar scaled by ``f`` exactly as HistoricalReplayer.step does (no mode check)."""
    from engine.book import round_to_tick
    from engine.replay import HistoricalReplayer
    tick = engine.tick
    targets = HistoricalReplayer.price_path({"o": o * f, "h": h * f, "l": l * f, "c": c * f}, tick)
    volume = int(v)
    if st["volbudget"] and volume > 0:                       # the agents' volume is already in the tape
        cut = int(min(st["vol_debt"], st["volbudget_maxcut"] * volume))
        if cut > 0:
            volume -= cut
            st["vol_debt"] -= cut
            st["vol_cut_total"] += cut
    last = round_to_tick(engine.last_price, tick)
    if volume == 0 and len(targets) == 1 and abs(targets[0] - last) < tick / 2:
        targets = []
    elif volume < len(targets):
        if rp is not None:
            rp.inexact_bars.append((number, volume, len(targets)))
        st["n_inexact"] += 1
        volume = len(targets)
    legs = HistoricalReplayer._split(volume, len(targets))
    driver = HistoricalReplayer(engine, [])
    for i, (target, qty) in enumerate(zip(targets, legs)):
        if st["volbudget"] >= 2:
            engine.flush_agent_orders(driver.LP_AGENT)             # what _drive_to does first
            flat = abs(target - round_to_tick(engine.last_price, tick)) < tick / 2
            side = "BUY" if (flat or target > round_to_tick(engine.last_price, tick)) else "SELL"
            swept = _swept_agent_qty(engine, side, target, driver.GHOST_AGENT)
            if swept > 0:                                          # these fill on top of the leg's qty
                cut_leg = min(swept, max(qty - 1, 0))
                qty -= cut_leg
                st["vol_cut_passive_total"] += cut_leg
                st["passive_excess_total"] += swept - cut_leg
                st["vol_debt"] += swept - cut_leg
        driver._drive_to(target, qty, pubs, walk=i > 0)


def _distances(d_max: int, levels: int) -> tuple:
    key = (d_max, levels)
    ds = _DIST_CACHE.get(key)
    if ds is None:
        out = list(range(1, min(12, d_max) + 1))[:levels]
        while out[-1] < d_max and len(out) < levels:
            out.append(min(max(out[-1] + 1, math.ceil(out[-1] * 1.25)), d_max))
        if out[-1] < d_max:
            out[-1] = d_max
        ds = _DIST_CACHE[key] = tuple(out)
    return ds


def _agent_cum(engine, ref_i: int, ds: tuple, bid_side: bool) -> List[int]:
    """Cumulative resting quantity of AGENT orders (not the user's, not internal) within d ticks of the
    reference, for every d in ``ds``. Read-only. A bid counts at d if its price >= (ref_i - d) tick, an ask
    if its price <= (ref_i + d) tick; an order at or beyond the reference counts for every d."""
    from bisect import bisect_left
    book = engine.book._bids if bid_side else engine.book._asks
    ledger = engine.agent_ledger
    tick = engine.tick
    d_last = ds[-1]
    add = [0] * len(ds)
    skip: Dict[str, bool] = {}
    for key in book.keys():                                   # best price first
        price = -key if bid_side else key
        dist = (ref_i - price / tick) if bid_side else (price / tick - ref_i)
        k = math.ceil(dist - 1e-6)
        if k > d_last:
            break
        q = 0
        for o in book[key]:
            aid = o["agent_id"]
            sk = skip.get(aid)
            if sk is None:
                led = ledger.get(aid)
                sk = skip[aid] = _is_user(aid) or (led is not None and bool(led["internal"]))
            if not sk:
                q += o["qty"]
        if q:
            add[bisect_left(ds, k)] += q
    cum, run = [], 0
    for x in add:
        run += x
        cum.append(run)
    return cum


def _post_book(engine, pubs, st, tod_value: float) -> None:
    """Step 4: displayed latent book around the price, never crossing other participants' orders."""
    engine.flush_agent_orders(engine.LP_AGENT)
    tick = engine.tick
    dp = engine.price_dp
    ref_i = int(round(engine.last_price / tick))
    P = ref_i * tick
    if P <= 0:
        return
    Ys = st["Y"] * st["sigma_d"]
    d_max = max(1, math.ceil(P * Ys * math.sqrt(st["depth_adv"]) / tick - 1e-9))
    ds = _distances(d_max, st["levels"])
    A = st["adv"] * (tick / (P * Ys)) ** 2           # F(d) = A d^2
    bb, ba = engine.book.best_bid(), engine.book.best_ask()   # SIM_LP flushed: other participants only
    eps = tick * 1e-6
    budget = st["depthbudget"]
    if budget:                                        # agents' resting depth replaces latent depth
        acb = _agent_cum(engine, ref_i, ds, True)
        acs = _agent_cum(engine, ref_i, ds, False)
        lat_b = lat_s = 0                             # latent depth of each side so far (posted or crossed)
    prev = 0
    t_prev = 0
    for i, d in enumerate(ds):
        size_b = size_s = max(1, int(round(A * (d * d - prev * prev) * tod_value)))
        prev = d
        if budget:
            t_d = int(round(A * d * d * tod_value))   # F(d)
            if acb[i] > 0:
                size_b = max(0, t_d - acb[i] - lat_b)
                st["depth_displaced_total"] += max(0, (t_d - t_prev) - size_b)
            if acs[i] > 0:
                size_s = max(0, t_d - acs[i] - lat_s)
                st["depth_displaced_total"] += max(0, (t_d - t_prev) - size_s)
            lat_b += size_b
            lat_s += size_s
            t_prev = t_d
        bid_px = round((ref_i - d) * tick, dp)
        ask_px = round((ref_i + d) * tick, dp)
        if size_b > 0 and bid_px > 0 and (ba is None or bid_px < ba - eps):
            engine.submit_flow_order(engine.LP_AGENT, "BUY", "LIMIT", size_b, bid_px, pubs,
                                     tif_ticks=engine.LP_TIF_TICKS)
            st["latent_posted_total"] += size_b
        if size_s > 0 and (bb is None or ask_px > bb + eps):
            engine.submit_flow_order(engine.LP_AGENT, "SELL", "LIMIT", size_s, ask_px, pubs,
                                     tif_ticks=engine.LP_TIF_TICKS)
            st["latent_posted_total"] += size_s


def _tod_value(st: dict, tape: Optional[dict], minute: int) -> float:
    if not st["tod_on"] or tape is None or tape["tod"] is None:
        return 1.0
    tod = tape["tod"]
    return tod[(minute - 1) % len(tod)]


def _exhausted_replayer(engine, pubs) -> None:
    """What HistoricalReplayer.step does when the tape is used up."""
    rp = engine.replayer
    if not rp.finished:
        rp.finished = True
        engine.paused = True
        pubs.append(("event", {"seq": engine._next_seq(), "kind": "session",
                               "action": "day_close", "day_count": engine.day_count}))
        logger.info("backtest: replay tape exhausted after %d bars; paused", len(rp.bars))


def _tick(engine, pubs, orig) -> None:
    d = engine.__dict__
    st = d.get("_bt")
    if st is not None and engine.tick_count <= st["last_tick"]:
        st = None                                   # the engine was reset (init_sim / start_replay)
    rp = engine.replayer
    mid_step = False

    if st is None:
        if rp is not None:
            seen = d.get("_bt_seen")
            prev_cursor = seen[1] if seen and seen[0] == engine.tick_count - 1 and seen[1] is not None else 0
            mid_step = rp.cursor > prev_cursor
            cur = rp.cursor
            sigma_d, adv = _scale_from_bars(rp.bars, cur, engine.SESSION_MINUTES)
            c_prev = rp.bars[cur - 1]["c"] if cur > 0 else rp.bars[0]["o"]
            st = _new_state("replayer", None, sigma_d, adv, 0.0, c_prev, engine.tick_count)
        else:
            key = _env_tape_key()
            if key is None:
                return orig(engine, pubs)
            tp = _tape_for_key(key)
            i = (engine.day_count - 1) * engine.SESSION_MINUTES + (engine.market_minute - 1)
            c_prev = tp["_c"][i - 1] if 0 < i <= tp["n"] else tp["prev_close"]
            base = math.log(engine.last_price / c_prev)
            st = _new_state("env", key, tp["sigma_d"], tp["adv"], base, c_prev, engine.tick_count)
        pending = d.pop("_bt_baseline", None)
        if pending is not None:
            st["agent_base"] = pending
        d["_bt"] = st
    st["last_tick"] = engine.tick_count

    if st["src"] == "replayer":
        if rp is None:
            return orig(engine, pubs)
        tape = None
        i = rp.cursor
        if not mid_step and i >= len(rp.bars):
            if tape_end_policy(engine) == "continue":
                if not st["exhausted"]:
                    st["exhausted"] = True
                    logger.warning("backtest: replay tape exhausted after %d bars; the market free-runs "
                                   "with the agents", len(rp.bars))
                return orig(engine, pubs)
            _exhausted_replayer(engine, pubs)
            return
    else:
        tape = _TAPES[st["tape"]]
        i = (engine.day_count - 1) * engine.SESSION_MINUTES + (engine.market_minute - 1)
        if i >= tape["n"]:
            if not st["exhausted"]:
                st["exhausted"] = True
                logger.warning("backtest: tape %s exhausted at tick %d; the market free-runs",
                               st["tape"], engine.tick_count)
            return orig(engine, pubs)

    if not mid_step:
        u = _footprint(engine, st)
        st["u"] = u
        if st["volbudget"]:
            st["vol_debt"] += st["agent_vol"]
        f = math.exp(st["base"] + u)
        if st["src"] == "replayer":
            bar = rp.bars[i]
            rp.cursor = i + 1
            if bar.get("ts") is not None:
                engine.sim_unix_time = bar["ts"]
            o, h, l, c, v = bar["o"], bar["h"], bar["l"], bar["c"], bar["v"]
            number = rp.cursor
        else:
            o, h, l, c, v = tape["_o"][i], tape["_h"][i], tape["_l"][i], tape["_c"][i], tape["_v"][i]
            rp = None
            number = i + 1
        _print_bar(engine, pubs, st, o, h, l, c, v, f, rp, number)
        st["c_prev"] = c
        st["n_bars"] += 1
    _post_book(engine, pubs, st, _tod_value(st, tape, engine.market_minute))


# --------------------------------------------------------------------------- #
# installation
# --------------------------------------------------------------------------- #
def install() -> None:
    """Wrap MatchingEngine._refresh_sim_lp (current process only). No-op unless CHRONOS_MODE=backtest;
    idempotent."""
    if _INSTALLED or os.environ.get("CHRONOS_MODE") != "backtest":
        return
    install_switchable()


def install_switchable() -> None:
    """Install the mode so that it can be switched per engine at run time (the dashboard's mode card).

    Wraps three things in this process only; the engine's source is not modified:
    * MatchingEngine._refresh_sim_lp: backtest mode when it is on for that engine (see backtest_enabled);
    * MatchingEngine.handle_message: one extra control message, BT_MODE (see _handle_bt_mode);
    * HistoricalReplayer.step: with tape_end_policy "continue", a replay whose tape runs out before any
      hand-over is handed to the agents (as an armed HANDOVER would) instead of pausing the market.
    Idempotent. Off by default for an engine unless CHRONOS_MODE=backtest; the tape-end policy defaults
    to CHRONOS_TAPE_END (else "pause", the engine's own behaviour).
    """
    global _INSTALLED
    from engine.matching_engine import MatchingEngine
    from engine.replay import HistoricalReplayer
    if not _INSTALLED:
        orig = MatchingEngine._refresh_sim_lp

        def _refresh_sim_lp(self, pubs, reference=None):
            if not backtest_enabled(self):
                return orig(self, pubs, reference)
            if reference is not None:                     # REPLAY mode's own bracket call: untouched
                rp = self.replayer
                self.__dict__["_bt_seen"] = (self.tick_count, rp.cursor if rp is not None else None)
                return orig(self, pubs, reference)
            return _tick(self, pubs, orig)

        MatchingEngine._refresh_sim_lp = _refresh_sim_lp
        _INSTALLED = True
    if not getattr(MatchingEngine.handle_message, "_bt_wrapped", False):
        orig_hm = MatchingEngine.handle_message

        def handle_message(self, msg):
            if isinstance(msg, dict) and msg.get("msg") == "BT_MODE":
                return _handle_bt_mode(self, msg), []
            return orig_hm(self, msg)

        handle_message._bt_wrapped = True
        MatchingEngine.handle_message = handle_message
    if not getattr(HistoricalReplayer.step, "_bt_wrapped", False):
        orig_step = HistoricalReplayer.step

        def step(self, pubs):
            eng = self.engine
            if (self.cursor >= len(self.bars) and not self.finished and eng.mode == "REPLAY"
                    and tape_end_policy(eng) == "continue"):
                self.finished = True
                logger.warning("replay tape exhausted after %d bars; handing the market to the agents",
                               len(self.bars))
                # same switch as an armed HANDOVER at the start of this tick (keeps the file's clock)
                eng._latch_reactive("HANDOVER", pubs)
                return
            return orig_step(self, pubs)

        step._bt_wrapped = True
        HistoricalReplayer.step = step


# --------------------------------------------------------------------------- #
# run-time switches (dashboard)
# --------------------------------------------------------------------------- #
MODES = ("free", "backtest")
TAPE_END_POLICIES = ("continue", "pause")


def backtest_enabled(engine) -> bool:
    """Backtest mode on for this engine: the dashboard's choice, else CHRONOS_MODE=backtest."""
    on = engine.__dict__.get("_bt_on")
    return (os.environ.get("CHRONOS_MODE") == "backtest") if on is None else bool(on)


def tape_end_policy(engine) -> str:
    """What happens when the replayed data runs out: "continue" (agents keep trading) or "pause"."""
    p = engine.__dict__.get("_bt_tape_end") or os.environ.get("CHRONOS_TAPE_END", "pause")
    return p if p in TAPE_END_POLICIES else "pause"


def mode_status(engine) -> dict:
    rp = engine.replayer
    st = engine.__dict__.get("_bt")
    if st is not None and (engine.mode != "REACTIVE" or engine.tick_count < st["last_tick"]):
        st = None                                   # left over from an earlier run (a new replay was started)
    return {"mode": "backtest" if backtest_enabled(engine) else "free",
            "on_tape_end": tape_end_policy(engine),
            "engine_mode": engine.mode,
            "locked": engine.mode == "REACTIVE",
            "tape_bars": len(rp.bars) if rp is not None else 0,
            "bars_done": rp.cursor if rp is not None else 0,
            "tape_exhausted": bool((rp is not None and rp.finished) or (st is not None and st.get("exhausted")))}


def _handle_bt_mode(engine, msg: dict) -> dict:
    """BT_MODE {mode?: free|backtest, on_tape_end?: continue|pause} -> ACK with the current settings.

    The mode decides what happens from the hand-over on (the user's first fill, or HANDOVER), so it can only
    change while the market has not handed over; start a new replay to change it again. The tape-end policy
    can change at any time."""
    mode, policy = msg.get("mode"), msg.get("on_tape_end")
    if mode is not None and mode not in MODES:
        return engine._reject(f"mode must be one of {MODES}")
    if policy is not None and policy not in TAPE_END_POLICIES:
        return engine._reject(f"on_tape_end must be one of {TAPE_END_POLICIES}")
    if mode is not None and (mode == "backtest") != backtest_enabled(engine):
        if engine.mode == "REACTIVE":
            return engine._reject("the market has already handed over to the agents; start a new replay "
                                  "to change the mode")
        engine.__dict__["_bt_on"] = mode == "backtest"
        logger.info("mode set to %s", mode)
    if policy is not None:
        engine.__dict__["_bt_tape_end"] = policy
    return {"msg": "ACK", "status": "OK", **mode_status(engine), "seq": engine._next_seq()}
