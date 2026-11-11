/* ============================================================================
   views/market.js — the hero view: candle+volume chart, order book,
   trade tape, agent leaderboard. All data from the live tick/trade stream.
   ============================================================================ */

import { runstate } from '../lib/runstate.js';
import { mountRunSetup } from './run_setup.js';
import { mountTradeReview, whoIs, placedBy, strategyNames } from './trade_review.js';

const TAPE_CAP = 60;
const BOOK_LEVELS = 10;
const FF_FACTOR = 20;          // the fast-forward button: this many times faster until the user's next trade
const EYE_ON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><circle cx="12" cy="12" r="3" fill="none" stroke="currentColor" stroke-width="1.8"/></svg>';
const EYE_OFF = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><circle cx="12" cy="12" r="3" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M4 4l16 16" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
const RUN_GAP_S = 120;         // trade minutes at most this far apart share one chart label (one sliced block)

/* Chart timeframes, in minutes. Candles are built here from the engine's 1-minute bars and aligned to the NSE
   open, 09:15 (the sim clock is IST written as UTC), like TradingView: 15m candles start 09:15, 09:30, ...;
   1h candles 09:15, 10:15, .... The newest candle keeps growing tick by tick until its period ends. */
const TIMEFRAMES = [1, 5, 15, 30, 60];
const SESSION_ORIGIN_S = (9 * 60 + 15) * 60;
const TF_KEY = 'chronos.market.tf';
const tfLabel = (m) => (m >= 60 ? `${m / 60}h` : `${m}m`);
function loadTf() {
  try { const v = Number(localStorage.getItem(TF_KEY)); return TIMEFRAMES.includes(v) ? v : 1; } catch { return 1; }
}
function saveTf(v) { try { localStorage.setItem(TF_KEY, String(v)); } catch { /* private window: not remembered */ } }
/** Start (unix s) of the `tf`-minute candle containing time t. */
function bucketOf(t, tf) {
  const p = tf * 60;
  return Math.floor((t - SESSION_ORIGIN_S) / p) * p + SESSION_ORIGIN_S;
}
/** 1-minute bars {time,open,high,low,close,volume} (ascending) -> tf-minute candles. */
function aggregate(bars, tf) {
  if (tf === 1) return bars.slice();
  const out = [];
  for (const b of bars) {
    const t = bucketOf(b.time, tf);
    const c = out[out.length - 1];
    if (c && c.time === t) {
      c.high = Math.max(c.high, b.high);
      c.low = Math.min(c.low, b.low);
      c.close = b.close;
      c.volume += b.volume || 0;
    } else {
      out.push({ time: t, open: b.open, high: b.high, low: b.low, close: b.close, volume: b.volume || 0 });
    }
  }
  return out;
}
const SHARES = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });
const shares = (n) => SHARES.format(n);          // 100000 -> "100K" (fmt.compact adds a ₹, wrong for shares)

/* Styles for the user's trade markers, their hover box, the fast-forward button and the YOU tag on the tape. */
const STYLE_ID = 'mk-mine-style';
const CSS = `
.mk-tip{position:absolute;z-index:4;pointer-events:none;min-width:210px;max-width:300px;padding:8px 10px;
  background:var(--bg-2);border:1px solid var(--border);border-radius:var(--r-md);box-shadow:0 8px 24px rgba(0,0,0,.45);
  font-size:var(--fs-xs);color:var(--text);line-height:1.5}
.mk-tip h4{margin:0 0 4px;font-size:var(--fs-sm);font-weight:600;color:var(--text-hi,#fff);display:flex;gap:6px;align-items:center}
.mk-tip .mk-side{display:grid;grid-template-columns:auto 1fr;gap:1px 10px}
.mk-tip .mk-side+.mk-side{margin-top:6px;padding-top:6px;border-top:1px solid var(--border)}
.mk-tip .k{color:var(--text-mut)} .mk-tip .v{text-align:right}
.mk-tip .buy{color:var(--up)} .mk-tip .sell{color:var(--down)}
.mk-tip.pinned{pointer-events:auto;max-width:330px;border-color:var(--border-strong)}
.mk-tip h4 .mk-x{all:unset;cursor:pointer;margin-left:auto;font-size:16px;line-height:1;color:var(--text-mut);padding:0 4px;border-radius:3px}
.mk-tip h4 .mk-x:hover{color:var(--text-hi);background:var(--bg-3)}
.mk-tip .mk-hint{margin-top:6px;color:var(--text-dim)}
.mk-tip .mk-ords-h{margin:8px 0 4px;padding-top:6px;border-top:1px solid var(--border);color:var(--text-mut)}
.mk-tip .mk-ords{display:flex;flex-direction:column;gap:2px;max-height:180px;overflow:auto}
.mk-tip .mk-ord{all:unset;cursor:pointer;display:flex;gap:6px;align-items:center;padding:4px 6px;border-radius:3px;
  border:1px solid var(--border)}
.mk-tip .mk-ord:hover,.mk-tip .mk-ord:focus-visible{background:var(--bg-3);border-color:var(--border-strong)}
.mk-tip .mk-ord .t{margin-left:auto;color:var(--text-mut)}
.mk-ff[aria-pressed="true"]{color:var(--warn);border-color:var(--warn);background:var(--warn-soft)}
.mk-ff-ico{width:12px;height:12px;vertical-align:-1px}
.mk-ff-note{max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mk-head-l{display:flex;align-items:center;gap:12px;min-width:0;flex:1 1 auto}
.mk-eye{all:unset;cursor:pointer;display:inline-flex;align-items:center;padding:3px 6px;border:1px solid var(--border);
  border-radius:var(--r-sm);color:var(--text-mut)}
.mk-eye:hover{color:var(--text-hi);background:var(--bg-3)}
.mk-eye[aria-pressed="true"]{color:var(--text-hi);background:var(--bg-3);border-color:var(--border-strong)}
.mk-eye:focus-visible{outline:2px solid var(--accent)}
.mk-eye svg{width:16px;height:16px}
.mk-foot-legend{color:var(--text-mut);font-size:12px}
.mk-foot-legend i{display:inline-block;width:8px;height:10px;border:1px solid var(--text-mut);background:transparent;margin-right:5px;vertical-align:-1px}
.mk-head-l [data-role="chart-title"]{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mk-tf{display:inline-flex;flex:none;border:1px solid var(--border);border-radius:var(--r-sm);overflow:hidden}
.panel-head .actions{flex:none}
.mk-ff-note{max-width:none;color:var(--text-hi)}
.mk-foot-note{display:flex;align-items:center;gap:14px;min-width:0;overflow:hidden;white-space:nowrap}
.mk-tf button{all:unset;cursor:pointer;padding:2px 8px;font:600 var(--fs-xs)/1.5 var(--mono);color:var(--text-mut);
  letter-spacing:0;text-transform:none}
.mk-tf button+button{border-left:1px solid var(--border)}
.mk-tf button:hover{color:var(--text);background:var(--bg-3)}
.mk-tf button[aria-pressed="true"]{color:var(--accent-2);background:var(--bg-3)}
.mk-tf button:focus-visible{outline:2px solid var(--accent-2);outline-offset:-2px}
.mk-right{display:flex;flex-direction:column;min-width:0;min-height:0}
.mk-live{flex:1 1 auto;min-height:0;display:flex}
.mk-setup{flex:1 1 auto;min-height:0}
.mk-right>[hidden]{display:none!important}
.mk-foot{flex:0 0 auto;display:flex;align-items:center;justify-content:space-between;gap:8px;
  padding:4px 8px 4px 12px;border-top:1px solid var(--border);background:var(--bg-1);font-size:12px}
.mk-foot-note{color:var(--text-mut)}
.mk-foot-btn{all:unset;display:inline-flex;align-items:center;gap:6px;cursor:pointer;padding:4px 8px;border-radius:4px;
  color:var(--text-mut);font-size:12px}
.mk-foot-btn:hover{color:var(--text-hi);background:var(--bg-3)}
.mk-foot-btn:focus-visible{outline:2px solid var(--accent)}
.mk-foot-btn svg{width:15px;height:15px}
.mk-foot-btn{white-space:nowrap;flex:none}
.mk-foot-note .mk-ff-note{overflow:hidden;text-overflow:ellipsis;min-width:0}
.mk-review{flex:1 1 auto;min-height:0}
/* full screen = the chart plus, when open, your trades; nothing else */
.view-cols.mk-fs{background:var(--bg-0);padding:6px}
.view-cols.mk-max{position:fixed;inset:0;z-index:var(--z-modal)}
.view-cols.mk-fs.mk-noside>.splitter,.view-cols.mk-fs.mk-noside>.mk-right{display:none!important}
.tape-row.mine{background:var(--bg-3)}
.tape-row .you{font-size:9px;font-weight:700;letter-spacing:.06em;color:var(--accent-2);margin-left:4px}
`;

export default {
  id: 'market',
  title: 'Market',
  icon: 'candles',
  group: '',

  _subs: [],
  _splits: [],
  _chart: null,
  _bids: null,
  _asks: null,
  _els: {},
  _latestBar: null,
  _hovering: false,

  async mount(el, ctx) {
    const { store, fmt, ui, charts, splitter } = ctx;
    this._subs = [];
    this._splits = [];

    el.innerHTML = `
      <div class="view-root">
        <div class="view-rows" data-role="rows">
          <div class="view-cols" data-role="cols">

            <section class="panel" aria-label="Price and volume">
              <div class="panel-head"><span class="mk-head-l"><span data-role="chart-title">Price &amp; Volume</span>
                  <span class="mk-tf" role="group" aria-label="Timeframe" data-role="tf">${TIMEFRAMES.map((m) =>
                    `<button type="button" data-tf="${m}" aria-pressed="false" title="${m >= 60 ? `${m / 60} hour` : `${m} minute`} candles">${tfLabel(m)}</button>`).join('')}</span>
                  <button type="button" class="mk-eye" data-role="eye" aria-pressed="false" hidden
                    title="Show the real market on this chart, in grey (to compare)">${EYE_OFF}</button></span>
                <span class="actions">
                  <button type="button" class="btn btn-ghost btn-sm" data-role="handover" hidden
                    title="Stop playing the real data here and let the simulated traders carry on">Hand over</button>
                  <button type="button" class="btn btn-ghost btn-sm mk-ff" data-role="ff" hidden
                    title="Run the market ${FF_FACTOR}× faster until one of your strategies trades, then back to normal speed"><svg class="mk-ff-ico" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6l7 6-7 6zM13 6l7 6-7 6z" fill="currentColor"/></svg> ${FF_FACTOR}×</button>
                  <button type="button" class="btn btn-ghost btn-sm mk-ff" data-role="skip" hidden
                    title="Run the rest of the data at top speed without stopping at your trades, then pause so you can review them all">Skip all</button>
                </span>
              </div>
              <div class="panel-body flush chart-host">
                <div data-role="chart"></div>
                <div class="ohlc-readout" data-role="readout" hidden></div>
                <div class="mk-tip" data-role="tip" hidden></div>
                <div class="empty" data-role="empty">
                  <div class="mk-empty-logo"><svg viewBox="320 575 1292 468" aria-hidden="true"><path d="M320 575 L822 575 L822 747 L693 747 L693 705 L449 705 L449 914 L822 914 L822 1043 L320 1043 Z" fill="currentColor"/><circle cx="957" cy="802" r="93" fill="currentColor"/><polygon points="1233,575 1377,575 1157,1043 1013,1043" fill="currentColor"/><polygon points="1467,575 1612,575 1392,1043 1249,1043" fill="currentColor"/></svg><b>CHRONOS</b><span>ALGORITHMIC MARKET SIMULATOR</span></div>
                  <div><span class="dim">Choose the data on the right and press Start market.</span></div>
                </div>
              </div>
              <div class="mk-foot">
                <span class="mk-foot-note"><span data-role="state-note"></span>
                  <span class="mk-ff-note" data-role="ff-note" hidden></span>
                  <span class="mk-foot-legend" data-role="real-legend" hidden><i></i>Grey candles: the real market</span></span>
                <button type="button" class="mk-foot-btn" data-role="fs" title="Show the chart on the whole screen (Esc to leave)">
                  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
                  <span data-role="fs-label">Full screen</span></button>
              </div>
            </section>

            <div class="mk-right" data-role="right">
              <div class="mk-live" data-role="live">
                <section class="panel" aria-label="Order book">
                  <div class="panel-head"><span>Order book</span></div>
                  <div class="panel-body flush" data-role="book">
                    <div data-role="asks"></div>
                    <div class="ob-spread" data-role="spread">spread —</div>
                    <div data-role="bids"></div>
                  </div>
                </section>

                <section class="panel" aria-label="Trades">
                  <div class="panel-head"><span>Trades</span>
                    <span class="actions"><button type="button" class="btn btn-ghost btn-icon btn-sm" data-role="pop-tape" aria-label="Open all trades in a new tab">${ui.icon('popout')}</button></span>
                  </div>
                  <div class="panel-body flush scroll-y"><div class="tape-list" data-role="tape"></div></div>
                </section>
              </div>
              <section class="panel mk-setup" data-role="setup" aria-label="New run" hidden></section>
              <section class="panel mk-review" data-role="review" aria-label="Your trades" hidden></section>
            </div>

          </div>

          <section class="panel" aria-label="Agent leaderboard">
            <div class="panel-head"><span>Agent Leaderboard</span>
              <span class="actions"><button type="button" class="btn btn-ghost btn-icon btn-sm" data-role="pop-agents" aria-label="Open all agents in a new tab">${ui.icon('popout')}</button></span>
            </div>
            <div class="panel-body flush scroll-y">
              <table class="grid" aria-label="Agent P&L leaderboard">
                <thead><tr><th>#</th><th>Agent</th><th>Type</th><th class="right">P&amp;L</th><th class="right">Pos</th><th class="right">Cash</th></tr></thead>
                <tbody data-role="lb"></tbody>
              </table>
            </div>
          </section>
        </div>
      </div>`;

    const q = (r) => el.querySelector(`[data-role="${r}"]`);
    this._els = {
      chartTitle: q('chart-title'), chart: q('chart'), readout: q('readout'),
      empty: q('empty'), asks: q('asks'), bids: q('bids'), spread: q('spread'),
      tape: q('tape'), lb: q('lb'),
    };

    q('pop-tape').addEventListener('click', () => window.open('pages/trades.html', '_blank', 'noopener'));
    q('pop-agents').addEventListener('click', () => window.open('pages/agents.html', '_blank', 'noopener'));

    /* ---- resizable layout ---- */
    this._splits.push(splitter.vSplit(el.querySelector('[data-role="rows"]'), {
      sizes: [72, 28], min: [220, 110], storageKey: 'chronos.split.market.rows',
    }));
    // chart | right side; the right side is the book + trades (live) or the New run panel (set-up)
    this._splits.push(splitter.hSplit(el.querySelector('[data-role="cols"]'), {
      sizes: [60, 40], min: [320, 300], storageKey: 'chronos.split.market.cols2',
    }));
    this._splits.push(splitter.hSplit(q('live'), {
      sizes: [48, 52], min: [150, 160], storageKey: 'chronos.split.market.live',
    }));

    /* ---- chart + crosshair readout ---- */
    this._chart = charts.candleChart(this._els.chart);
    const renderReadout = (bar) => {
      const b = bar || this._latestBar;
      if (!b) { this._els.readout.hidden = true; return; }
      const dirCls = b.close >= b.open ? 'up' : 'down';
      this._els.readout.hidden = false;
      this._els.readout.innerHTML =
        `<span>${fmt.time(b.time)}</span>`
        + `<span>O <b class="${dirCls}">${fmt.num(b.open)}</b></span>`
        + `<span>H <b class="${dirCls}">${fmt.num(b.high)}</b></span>`
        + `<span>L <b class="${dirCls}">${fmt.num(b.low)}</b></span>`
        + `<span>C <b class="${dirCls}">${fmt.num(b.close)}</b></span>`
        + `<span>V <b>${fmt.num(b.volume, 0)}</b></span>`;
    };
    this._chart.crosshairReadout((bar) => { this._hovering = !!bar; renderReadout(bar); });
    // Coming back from another page: redraw every candle of the running chart, not just the last one.
    this._tf = loadTf();
    const history = aggregate(store.bars(), this._tf);
    if (history.length) this._chart.setData(history);

    /* ---- the user's own trades: arrows on their candles + a hover box ---- */
    if (!document.getElementById(STYLE_ID)) {
      const st = document.createElement('style');
      st.id = STYLE_ID;
      st.textContent = CSS;
      document.head.appendChild(st);
    }
    // minute -> {BUY:agg, SELL:agg}; agg = {qty, notional, fills, lo, hi, first, last, agents:Set, cps:Map}
    this._mine = new Map();
    this._mineDirty = false;
    const minuteOf = (t) => t.unix_time - (t.unix_time % 60);
    const addMine = (t) => {
      const me = store.mySide(t);
      if (!me || !Number.isFinite(t.unix_time)) return;
      const time = minuteOf(t);
      if (!this._mine.has(time)) this._mine.set(time, {});
      const slot = this._mine.get(time);
      const a = slot[me.side] || (slot[me.side] = {
        qty: 0, notional: 0, fills: 0, lo: Infinity, hi: -Infinity, first: t.seq, last: t.seq,
        agents: new Set(), cps: new Map(), aggr: 0,
      });
      a.qty += t.qty;
      a.notional += t.qty * t.price;
      a.fills += 1;
      a.lo = Math.min(a.lo, t.price);
      a.hi = Math.max(a.hi, t.price);
      a.last = t.seq;
      a.agents.add(me.agent);
      if (t.aggressor === me.side) a.aggr += t.qty;          // the user crossed the spread (market order / marketable limit)
      a.cps.set(me.counterparty, (a.cps.get(me.counterparty) || 0) + t.qty);
      this._mineDirty = true;
    };
    const renderMarkers = () => {
      if (!this._mineDirty || !this._chart) return;
      this._mineDirty = false;
      const tf = this._tf;
      const lastBar = this._latestBar ? this._latestBar.time : Infinity;
      const gap = Math.max(RUN_GAP_S, 2 * tf * 60);
      const marks = [];
      for (const side of ['BUY', 'SELL']) {
        // One arrow per candle (of the chosen timeframe) in which the user traded; one label per run of such
        // candles (a block sent in slices) with the run's total, so ten slices read "BUY 100K" once.
        const perCandle = new Map();
        for (const [minute, slot] of this._mine) {
          if (!slot[side]) continue;
          const t = bucketOf(minute, tf);
          perCandle.set(t, (perCandle.get(t) || 0) + slot[side].qty);
        }
        let run = null;
        for (const time of [...perCandle.keys()].sort((x, y) => x - y)) {
          if (time > lastBar) { this._mineDirty = true; continue; }   // its candle is not drawn yet: next tick
          const m = { time, side, text: '' };
          if (!run || time - run.end > gap) { run = { head: m, qty: 0 }; }
          run.end = time;
          run.qty += perCandle.get(time);
          run.head.text = `${side} ${shares(run.qty)}`;
          marks.push(m);
        }
      }
      this._chart.setMarkers(marks);
    };
    /** Every trade aggregate of the user inside the candle starting at `time` (all its minutes merged). */
    const mineIn = (time) => {
      const end = time + this._tf * 60;
      const out = {};
      for (const [minute, slot] of this._mine) {
        if (minute < time || minute >= end) continue;
        for (const side of ['BUY', 'SELL']) {
          const a = slot[side];
          if (!a) continue;
          const o = out[side] || (out[side] = { qty: 0, notional: 0, fills: 0, lo: Infinity, hi: -Infinity,
            agents: new Set(), cps: new Map(), aggr: 0, minutes: 0 });
          o.qty += a.qty; o.notional += a.notional; o.fills += a.fills; o.aggr += a.aggr; o.minutes += 1;
          o.lo = Math.min(o.lo, a.lo); o.hi = Math.max(o.hi, a.hi);
          a.agents.forEach((x) => o.agents.add(x));
          a.cps.forEach((q, id) => o.cps.set(id, (o.cps.get(id) || 0) + q));
        }
      }
      return out.BUY || out.SELL ? out : null;
    };
    for (const t of store.myTrades()) addMine(t);
    if (history.length) this._latestBar = history[history.length - 1];
    renderMarkers();

    const tip = el.querySelector('[data-role="tip"]');
    const sideBlock = (side, a) => {
      const avg = a.notional / a.qty;
      const cps = [...a.cps.entries()].sort((x, y) => y[1] - x[1]);
      const cpText = cps.slice(0, 3).map(([id, q]) => `${id} (${whoIs(id)}) ${shares(q)}`).join(', ')
        + (cps.length > 3 ? ` +${cps.length - 3} more` : '');
      const cls = side === 'BUY' ? 'buy' : 'sell';
      return `<div class="mk-side">
          <span class="k">Side</span><span class="v ${cls}"><b>${side === 'BUY' ? 'BOUGHT' : 'SOLD'}</b></span>
          <span class="k">Quantity</span><span class="v">${fmt.num(a.qty, 0)}</span>
          <span class="k">Avg price</span><span class="v">₹${fmt.num(avg)}</span>
          ${a.lo !== a.hi ? `<span class="k">Price range</span><span class="v">₹${fmt.num(a.lo)} – ₹${fmt.num(a.hi)}</span>` : ''}
          <span class="k">Value</span><span class="v">${fmt.inr(a.notional)}</span>
          <span class="k">Fills</span><span class="v">${a.fills}${a.aggr ? ` (${Math.round((100 * a.aggr) / a.qty)}% crossed spread)` : ' (all passive)'}</span>
          ${this._tf > 1 ? `<span class="k">Minutes</span><span class="v">${a.minutes} of ${this._tf}</span>` : ''}
          <span class="k">Placed by</span><span class="v">${[...a.agents].map(placedBy).join(', ')}</span>
          <span class="k">Other side</span><span class="v">${cpText}</span>
        </div>`;
    };
    /** Per-minute orders of the user inside the candle starting at `time` (for the pinned box). */
    const ordersIn = (time) => {
      const end = time + this._tf * 60;
      const out = [];
      for (const [minute, slot] of this._mine) {
        if (minute < time || minute >= end) continue;
        for (const side of ['BUY', 'SELL']) if (slot[side]) out.push({ minute, side, a: slot[side] });
      }
      return out.sort((x, y) => x.minute - y.minute || (x.side < y.side ? -1 : 1));
    };
    const tipHtml = (time, pin) => {
      const slot = mineIn(time);
      if (!slot) return null;
      let html = `<h4>Your trade · ${fmt.dtime(time)}${this._tf > 1 ? ` · ${tfLabel(this._tf)} candle` : ''}`
        + `${pin ? '<button type="button" class="mk-x" data-x aria-label="Close">×</button>' : ''}</h4>`
        + (slot.BUY ? sideBlock('BUY', slot.BUY) : '')
        + (slot.SELL ? sideBlock('SELL', slot.SELL) : '');
      if (pin) {
        html += '<div class="mk-ords-h">Open a trade:</div><div class="mk-ords">'
          + ordersIn(time).map((o) => `<button type="button" class="mk-ord" data-key="${o.minute}|${o.side}">`
            + `<span class="${o.side === 'BUY' ? 'buy' : 'sell'}">${o.side === 'BUY' ? 'Bought' : 'Sold'}</span>`
            + ` ${fmt.num(o.a.qty, 0)} @ ₹${fmt.num(o.a.notional / o.a.qty)}<span class="t">${fmt.time(o.minute)} ›</span></button>`).join('')
          + '</div>';
      } else {
        html += '<div class="mk-hint">Click to keep this box open and see each trade</div>';
      }
      return html;
    };
    const placeTip = (px, py) => {
      // keep the box inside the chart: right of the pointer, or left of it near the right edge
      const host = tip.parentElement.getBoundingClientRect();
      const w = tip.offsetWidth;
      const hgt = tip.offsetHeight;
      const x = px + 16 + w > host.width ? px - 16 - w : px + 16;
      const y = Math.min(Math.max(8, py - hgt / 2), Math.max(8, host.height - hgt - 8));
      tip.style.left = `${Math.max(8, x)}px`;
      tip.style.top = `${y}px`;
    };
    let pinned = null;               // candle time of the pinned box
    const unpin = () => { pinned = null; tip.classList.remove('pinned'); tip.hidden = true; };
    const pinAt = (time, x, y) => {
      const html = tipHtml(time, true);
      if (!html) { unpin(); return; }
      pinned = time;
      tip.innerHTML = html;
      tip.classList.add('pinned');
      tip.hidden = false;
      placeTip(x, y);
    };
    this._chart.hover((h) => {
      if (pinned != null) return;                     // a pinned box stays put
      const html = h && tipHtml(h.time, false);
      if (!html) { tip.hidden = true; return; }
      tip.innerHTML = html;
      tip.hidden = false;
      placeTip(h.x, h.y);
    });
    this._chart.click((h) => { if (h) pinAt(h.time, h.x, h.y); else unpin(); });
    tip.addEventListener('click', (e) => {
      if (e.target.closest('[data-x]')) { unpin(); return; }
      const b = e.target.closest('.mk-ord');
      if (b) openTrade(b.dataset.key);
    });
    strategyNames(ctx.api);          // names for "Placed by"

    /* ---- timeframe switch (1m / 5m / 15m / 30m / 1h) ---- */
    const tfGroup = q('tf');
    const markTf = () => tfGroup.querySelectorAll('button').forEach((b) =>
      b.setAttribute('aria-pressed', Number(b.dataset.tf) === this._tf ? 'true' : 'false'));
    const setTf = (tf) => {
      if (tf === this._tf) return;
      this._tf = tf;
      saveTf(tf);
      markTf();
      const data = aggregate(store.bars(), tf);
      // Like TradingView: keep showing the newest candles, with a candle width that suits the count.
      const width = this._els.chart.clientWidth || 800;
      const spacing = Math.max(6, Math.min(24, (width * 0.75) / Math.max(1, data.length)));
      if (this._chart.chart) this._chart.chart.timeScale().applyOptions({ barSpacing: spacing });
      this._chart.setData(data, { fit: false });
      this._latestBar = data[data.length - 1] || null;
      this._mineDirty = true;
      renderMarkers();
      unpin();
      if (this._drawReal) this._drawReal();
      if (!this._hovering) renderReadout(null);
    };
    tfGroup.addEventListener('click', (e) => {
      const b = e.target.closest('button[data-tf]');
      if (b) setTf(Number(b.dataset.tf));
    });
    markTf();

    /* ---- order book ---- */
    this._asks = charts.depthBars(this._els.asks, {
      side: 'ask', onFormat: { price: (n) => fmt.num(n), qty: (n) => fmt.num(n, 0) },
    });
    this._bids = charts.depthBars(this._els.bids, {
      side: 'bid', onFormat: { price: (n) => fmt.num(n), qty: (n) => fmt.num(n, 0) },
    });

    const signedInr = (n) => (n < 0 ? '-' : '+') + fmt.inr(Math.abs(n));

    /* ---- tick pipeline ---- */
    const applyTick = (tick) => {
      if (!tick || !tick.run_id) return;       // an idle engine (nothing started yet): keep the empty chart
      this._els.empty.style.display = 'none';
      this._els.chartTitle.innerHTML = `${tick.symbol} · Price &amp; Volume`;

      // Candles: the store keeps one bar per sim-minute; the chart shows the chosen timeframe. Every candle
      // from the last one drawn up to now is (re)built from those bars, so the newest grows tick by tick and,
      // at Skip all speed (several minutes per screen frame), no candle in between is skipped.
      if (tick.bar && Number.isFinite(tick.unix_time)) {
        const from = this._latestBar ? this._latestBar.time : -Infinity;   // empty chart: every bar of the run
        for (const c of aggregate(store.barsSince(from), this._tf)) {
          if (this._latestBar && c.time < this._latestBar.time) continue;
          this._chart.update(c);
          this._latestBar = c;
        }
        if (!this._hovering) renderReadout(null);
        renderMarkers();
      }

      // Book: depth scaled to the max size in view across BOTH sides.
      const bids = (tick.lob_bids || []).slice(0, BOOK_LEVELS);
      const asks = (tick.lob_asks || []).slice(0, BOOK_LEVELS);
      const max = Math.max(1, ...bids.map((l) => l[1]), ...asks.map((l) => l[1]));
      this._asks.update([...asks].reverse(), max); // best ask sits next to the spread row
      this._bids.update(bids, max);
      this._els.spread.innerHTML = Number.isFinite(tick.spread)
        ? `spread <b>${fmt.num(tick.spread)}</b> · mid <b>${fmt.num(tick.mid_price)}</b>`
        : 'spread —';

      // Leaderboard (1 Hz).
      const rows = (tick.leaderboard || []).map((a, i) => `
        <tr class="row-link" tabindex="0" role="link" data-agent="${a.id}" aria-label="Open agent ${a.id}">
          <td class="dim">${i + 1}</td>
          <td class="mono">${a.id}</td>
          <td class="muted">${a.type || ''}</td>
          <td class="num right ${a.pnl >= 0 ? 'num-up' : 'num-down'}">${signedInr(a.pnl)}</td>
          <td class="num right">${fmt.num(a.pos, 0)}</td>
          <td class="num right">${fmt.compact(a.cash)}</td>
        </tr>`).join('');
      this._els.lb.innerHTML = rows || '<tr><td colspan="6" class="dim">no agents yet</td></tr>';
    };

    this._els.lb.addEventListener('click', (e) => {
      const tr = e.target.closest('tr[data-agent]');
      if (tr) window.open(`pages/agent.html?id=${encodeURIComponent(tr.dataset.agent)}`, '_blank', 'noopener');
    });
    this._els.lb.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter') return;
      const tr = e.target.closest('tr[data-agent]');
      if (tr) window.open(`pages/agent.html?id=${encodeURIComponent(tr.dataset.agent)}`, '_blank', 'noopener');
    });

    /* ---- tape ---- */
    const tapeRow = (t) => {
      const cls = t.aggressor === 'BUY' ? 'buy' : 'sell';
      const me = store.mySide(t);
      const div = document.createElement('div');
      div.className = `tape-row ${cls}${me ? ' mine' : ''}`;
      if (me) div.title = `Your strategy ${me.agent} ${me.side === 'BUY' ? 'bought' : 'sold'} ${t.qty} from ${me.counterparty}`;
      div.innerHTML = `<span class="tt">${fmt.time(t.unix_time)}</span>`
        + `<span class="tp">${fmt.num(t.price)}</span>`
        + `<span class="tq">${fmt.num(t.qty, 0)}</span>`
        + `<span class="ta">${t.aggressor}${me ? '<span class="you">YOU</span>' : ''}</span>`;
      return div;
    };
    const pushTrade = (t) => {
      this._els.tape.prepend(tapeRow(t));
      while (this._els.tape.children.length > TAPE_CAP) this._els.tape.lastElementChild.remove();
    };
    // Prints arrive in bursts (thousands a second when skipping): add them to the list once per frame, newest TAPE_CAP.
    let tradeBuf = [];
    let tradeRaf = 0;
    const flushTrades = () => {
      tradeRaf = 0;
      const batch = tradeBuf.slice(-TAPE_CAP);
      tradeBuf = [];
      if (!batch.length || !this._els.tape) return;
      if (this._tapeEmpty) { this._els.tape.innerHTML = ''; this._tapeEmpty = false; }
      const frag = document.createDocumentFragment();
      for (let i = batch.length - 1; i >= 0; i -= 1) frag.appendChild(tapeRow(batch[i]));
      this._els.tape.prepend(frag);
      while (this._els.tape.children.length > TAPE_CAP) this._els.tape.lastElementChild.remove();
    };

    // Seed from the ring buffer (oldest first → newest ends up on top).
    for (const t of store.recentTrades(TAPE_CAP)) pushTrade(t);
    if (!this._els.tape.children.length) {
      this._els.tape.innerHTML = '<div class="dim" style="padding:var(--s-3)">no prints yet</div>';
      this._tapeEmpty = true;
    }

    this._subs.push(store.onTrade((t) => {
      tradeBuf.push(t);
      if (tradeBuf.length > 4 * TAPE_CAP) tradeBuf = tradeBuf.slice(-TAPE_CAP);
      if (!tradeRaf) tradeRaf = requestAnimationFrame(flushTrades);
      if (store.mySide(t)) addMine(t);              // markers are drawn with the next frame's tick
    }));
    // Ticks: draw only the newest one per screen frame (the store has already kept every minute's bar).
    let pendingTick = null;
    let tickRaf = 0;
    this._subs.push(store.onTick((t) => {
      pendingTick = t;
      if (!tickRaf) {
        tickRaf = requestAnimationFrame(() => {
          tickRaf = 0;
          const t2 = pendingTick;
          pendingTick = null;
          applyTick(t2);
          renderMarkers();
        });
      }
    }));
    this._subs.push(() => { cancelAnimationFrame(tickRaf); cancelAnimationFrame(tradeRaf); });
    // A different run started (new stock / replay / START / reset): start the chart and tape afresh.
    this._subs.push(store.onReset(() => {
      if (this._chart) { this._chart.setData([]); this._chart.setMarkers([]); }   // timeframe stays as chosen
      this._mine.clear();
      this._mineDirty = false;
      unpin();
      this._latestBar = null;
      this._els.readout.hidden = true;
      this._els.tape.innerHTML = '<div class="dim" style="padding:var(--s-3)">no prints yet</div>';
      this._tapeEmpty = true;
      tradeBuf = [];
    }));

    /* ---- speed (tools/sim_speed.py): FF_FACTOR× until your next trade, or Skip all = top speed to the end of
            the data, then the market pauses so all your trades can be reviewed together ---- */
    const ffBtn = q('ff');
    const skipBtn = q('skip');
    const ffNote = q('ff-note');
    const FF_ICON = '<svg class="mk-ff-ico" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6l7 6-7 6zM13 6l7 6-7 6z" fill="currentColor"/></svg>';
    let noteTimer = null;
    let hasData = false;             // a replay with real data (Skip all needs an end to skip to)
    let lastSpeed = null;
    const note = (text, ms = 0, title = '') => {
      clearTimeout(noteTimer);
      ffNote.hidden = !text;
      ffNote.textContent = text || '';
      ffNote.title = title;
      if (text && ms) noteTimer = setTimeout(() => { ffNote.hidden = true; }, ms);
    };
    const showSpeed = (sp) => {
      if (!sp) return;
      lastSpeed = sp;
      const rs = runstate.get();
      const ffOn = !!sp.fast_forward && sp.until !== 'data_end';
      const skipOn = !!sp.fast_forward && sp.until === 'data_end';
      ffBtn.hidden = !rs.hasRun || rs.stopped;
      skipBtn.hidden = !rs.hasRun || rs.stopped || !hasData;
      if (this._eyeAvail) this._eyeAvail(rs.hasRun && hasData);
      ffBtn.setAttribute('aria-pressed', ffOn ? 'true' : 'false');
      ffBtn.innerHTML = ffOn ? `&#9632; Stop ${sp.factor}×` : `${FF_ICON} ${FF_FACTOR}×`;
      ffBtn.disabled = skipOn;
      skipBtn.setAttribute('aria-pressed', skipOn ? 'true' : 'false');
      skipBtn.textContent = skipOn ? 'Stop skipping' : 'Skip all';
      skipBtn.disabled = ffOn;
      if (ffOn) note(`${sp.factor}× until your next trade`);
      else if (skipOn) note(`Skipping to the end of the data (${sp.factor}×)`);
    };
    const refreshSpeed = async () => {
      try { hasData = ((await ctx.api.mode.get()).tape_bars || 0) > 0; } catch { hasData = false; }
      try {
        showSpeed(await ctx.api.speed.get());
      } catch (err) {
        ffBtn.hidden = true;               // 501: an engine without fast-forward (not started via run_local)
        skipBtn.hidden = true;
        if (err && err.status !== 501) console.warn('[market] speed', err);
      }
    };
    const setSpeed = async (btn, cfg) => {
      btn.disabled = true;
      try {
        const sp = await ctx.api.speed.set(cfg);
        if (sp.fast_forward && runstate.get().paused) await ctx.api.sim.resume();   // skipping a paused market runs it
        showSpeed(sp);
        if (!sp.fast_forward) note('');
      } catch (err) {
        btn.disabled = false;
        ui.toast(`Speed change failed: ${err.detail || err.message}`, 'bad');
      }
    };
    ffBtn.addEventListener('click', () => setSpeed(ffBtn, ffBtn.getAttribute('aria-pressed') === 'true'
      ? { action: 'normal' } : { action: 'fast_forward', factor: FF_FACTOR }));
    skipBtn.addEventListener('click', () => setSpeed(skipBtn, skipBtn.getAttribute('aria-pressed') === 'true'
      ? { action: 'normal' } : { action: 'fast_forward', until: 'data_end' }));
    this._subs.push(store.onEvent((ev) => {
      if (ev.kind !== 'speed') return;
      showSpeed(ev);
      if (ev.fast_forward) return;
      if (ev.agent_id) note('Back to 1× — your trade', 8000, ev.reason || '');
      else if (ev.reason === 'the data ended') note('The data ended — paused so you can review your trades', 12000);
      else note('');
    }));
    this._subs.push(store.onReset(() => { note(''); refreshSpeed(); }));
    // A new run (also the first one after a restart, which brings no "reset"): ask again whether it has real data,
    // which decides Skip all and the eye button.
    let speedRun = null;
    this._subs.push(runstate.onChange((st) => {
      if (st.runId && st.runId !== speedRun) { speedRun = st.runId; refreshSpeed(); } else showSpeed(lastSpeed);
    }));
    this._subs.push(() => clearTimeout(noteTimer));
    refreshSpeed();

    /* ---- right side: the book + trades (live), the New run panel (set-up), or your trades (review: shown by
            itself when the market is paused or stopped, or when a trade is opened from the chart; × closes it) ---- */
    const cols = q('cols');
    const live = q('live');
    const setupHost = q('setup');
    const reviewHost = q('review');
    const stateNote = q('state-note');
    const hoBtn = q('handover');
    this._setup = mountRunSetup(setupHost, ctx);
    const rv = { forced: false, dismissed: null };
    const rvKey = (st) => `${st.runId}|${st.stopped ? 'S' : st.paused ? 'P' : 'R'}`;
    let reviewShown = false;
    const showView = (st) => {
      const mode = st.view === 'setup' ? 'setup'
        : st.hasRun && (rv.forced || ((st.paused || st.stopped) && rv.dismissed !== rvKey(st))) ? 'review' : 'live';
      setupHost.hidden = mode !== 'setup';
      live.hidden = mode !== 'live';
      reviewHost.hidden = mode !== 'review';
      if (mode === 'review' && !reviewShown) this._review.show();
      reviewShown = mode === 'review';
      cols.classList.toggle('mk-noside', mode !== 'review');          // used in full screen only
      stateNote.textContent = !st.hasRun ? '' : st.stopped ? 'Market stopped' : st.paused ? 'Paused' : '';
      hoBtn.hidden = !(st.hasRun && !st.stopped && store.state.mode === 'REPLAY');
    };
    const closeReview = () => { rv.forced = false; rv.dismissed = rvKey(runstate.get()); showView(runstate.get()); };
    const openTrade = (key) => { rv.forced = true; showView(runstate.get()); this._review.openOrder(key); };
    const locate = (minute) => {
      const t = bucketOf(minute, this._tf);
      this._chart.showTime(t);
      requestAnimationFrame(() => {
        const x = this._chart.timeToX(t);
        if (x != null) pinAt(t, x, (this._els.chart.clientHeight || 300) / 3);
      });
    };
    this._review = mountTradeReview(reviewHost, ctx, { onClose: closeReview, onLocate: locate });
    this._subs.push(runstate.onChange(showView));
    this._subs.push(store.onEvent((ev) => { if (ev.kind === 'mode_change') showView(runstate.get()); }));
    this._subs.push(store.onReset(() => { rv.forced = false; rv.dismissed = null; }));
    showView(runstate.get());
    hoBtn.addEventListener('click', async () => {
      hoBtn.disabled = true;
      try {
        const r = await ctx.api.handover();
        ui.toast(`Handed over after ${r.bars_replayed ?? '?'} minutes of real data`, 'good');
      } catch (err) {
        ui.toast(`Hand over failed: ${err.detail || err.message}`, 'bad');
      } finally {
        hoBtn.disabled = false;
      }
    });

    /* ---- full screen: the chart, plus your trades when they are open (button under the chart; Esc leaves) ---- */
    const fsLabel = q('fs-label');
    const isFs = () => document.fullscreenElement === cols || cols.classList.contains('mk-max');
    const syncFs = () => {
      cols.classList.toggle('mk-fs', isFs());
      fsLabel.textContent = isFs() ? 'Exit full screen' : 'Full screen';
    };
    q('fs').addEventListener('click', async () => {
      if (isFs()) {
        cols.classList.remove('mk-max');
        if (document.fullscreenElement) { try { await document.exitFullscreen(); } catch { /* already left */ } }
      } else {
        try { await cols.requestFullscreen(); }
        catch { cols.classList.add('mk-max'); }   // no Fullscreen API here: fill the window instead
      }
      syncFs();
    });
    const onKey = (e) => { if (e.key === 'Escape' && cols.classList.contains('mk-max')) { cols.classList.remove('mk-max'); syncFs(); } };
    document.addEventListener('fullscreenchange', syncFs);
    document.addEventListener('keydown', onKey);
    this._subs.push(() => {
      document.removeEventListener('fullscreenchange', syncFs);
      document.removeEventListener('keydown', onKey);
      if (document.fullscreenElement === cols) document.exitFullscreen().catch(() => {});
    });

    /* ---- eye: the real market's candles, faded, behind the chart (tools/real_tape.py); off by default ---- */
    const eyeBtn = q('eye');
    const legend = q('real-legend');
    const real = { on: false, bars: [], count: 0, timer: 0, busy: false, sim: 0 };
    this._drawReal = () => { if (real.on) this._chart.setGhost(aggregate(real.bars, this._tf)); };
    /** Redraw every candle from the store (after filling minutes the page had missed). */
    const redrawAll = () => {
      const data = aggregate(store.bars(), this._tf);
      this._chart.setData(data, { fit: false });
      this._latestBar = data[data.length - 1] || null;
      this._mineDirty = true;
      renderMarkers();
    };
    // One sync with the engine (tools/real_tape.py): the simulated candles this page missed (Skip all can outrun
    // the socket; a reloaded page starts empty) and, with the eye on, the real candles.
    const pullReal = async () => {
      const rid = (store.latestTick() || {}).run_id;
      if (real.busy || !rid || document.hidden) return;
      real.busy = true;
      try {
        const wantReal = real.on;            // the eye as it was when asking (it may be switched while we wait)
        const r = await ctx.api.real(wantReal ? real.count : 1e9, real.sim);
        if (r.run_id !== rid) return;
        if (wantReal && real.on) {
          if (r.count < real.count) { real.bars = []; real.count = 0; return; }   // a new run: next pull starts over
          for (const b of r.bars || []) real.bars.push({ time: b.time, open: b.o, high: b.h, low: b.l, close: b.c, volume: b.v });
          real.count = r.count;
          if ((r.bars || []).length) this._drawReal();
        }
        if ((r.sim_count || 0) < real.sim) { real.sim = 0; return; }
        if (store.fillBars(r.sim || [], rid)) redrawAll();
        real.sim = Math.max(0, (r.sim_count || 0) - 1);       // ask again for the last (still forming) one
      } catch { /* engine without it, or a blip: try again next time */ } finally { real.busy = false; }
    };
    const syncTimer = setInterval(pullReal, 2000);
    this._subs.push(() => clearInterval(syncTimer));
    pullReal();
    const setReal = (on) => {
      real.on = on;
      eyeBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
      eyeBtn.innerHTML = on ? EYE_ON : EYE_OFF;
      eyeBtn.title = on ? 'Hide the real market' : 'Show the real market on this chart, in grey (to compare)';
      legend.hidden = !on;
      this._chart.showGhost(on);
      clearInterval(real.timer);
      if (on) { this._drawReal(); pullReal(); real.timer = setInterval(pullReal, 1000); }
    };
    eyeBtn.addEventListener('click', () => setReal(!real.on));
    this._eyeAvail = (ok) => { eyeBtn.hidden = !ok; if (!ok && real.on) setReal(false); };
    this._subs.push(store.onReset(() => {
      real.bars = [];
      real.count = 0;
      real.sim = 0;
      this._chart.setGhost([]);
      if (real.on) pullReal();
    }));
    this._subs.push(() => clearInterval(real.timer));

    // Seed everything from the cached tick (bridge replays it on WS connect).
    applyTick(store.latestTick());
  },

  unmount() {
    this._subs.forEach((u) => u());
    this._subs = [];
    this._splits.forEach((s) => s.destroy());
    this._splits = [];
    if (this._chart) { this._chart.dispose(); this._chart = null; }
    if (this._asks) { this._asks.dispose(); this._asks = null; }
    if (this._bids) { this._bids.dispose(); this._bids = null; }
    this._latestBar = null;
    this._tapeEmpty = false;
    this._mine = null;
    if (this._setup) { this._setup.destroy(); this._setup = null; }
    if (this._review) { this._review.destroy(); this._review = null; }
    this._drawReal = null;
    this._eyeAvail = null;
    this._els = {};
  },
};
