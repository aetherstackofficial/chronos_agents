/* ============================================================================
   views/market.js — the hero view: candle+volume chart, order book,
   trade tape, agent leaderboard. All data from the live tick/trade stream.
   ============================================================================ */

const TAPE_CAP = 60;
const BOOK_LEVELS = 10;
const FF_FACTOR = 20;          // the fast-forward button: this many times faster until the user's next trade
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
/** Who took the other side, in words (engine agent-id conventions, engine/matching_engine._classify_agent). */
function whoIs(id) {
  if (/SIM_LP/.test(id)) return ' (resting book)';
  if (/GHOST/.test(id)) return ' (background flow)';
  if (/WHALE/.test(id)) return ' (institution)';
  if (/GEMINI|ORACLE/.test(id)) return ' (news trader)';
  if (/MM/.test(id)) return ' (market maker)';
  if (/^(STRAT_|USER_)/.test(id)) return ' (a strategy)';
  return ' (retail)';
}

/* Styles for the user's trade markers, their hover box, the fast-forward button and the YOU tag on the tape. */
const STYLE_ID = 'mk-mine-style';
const CSS = `
.mk-tip{position:absolute;z-index:4;pointer-events:none;min-width:210px;max-width:300px;padding:8px 10px;
  background:var(--bg-2);border:1px solid var(--border);border-radius:var(--r-md);box-shadow:0 8px 24px rgba(0,0,0,.45);
  font-size:var(--fs-xs);color:var(--text);line-height:1.5}
.mk-tip h4{margin:0 0 4px;font-size:var(--fs-sm);font-weight:600;color:var(--text-hi,#fff);display:flex;gap:6px;align-items:center}
.mk-tip .mk-side{display:grid;grid-template-columns:auto 1fr;gap:1px 10px;font-family:var(--mono)}
.mk-tip .mk-side+.mk-side{margin-top:6px;padding-top:6px;border-top:1px solid var(--border)}
.mk-tip .k{color:var(--text-mut)} .mk-tip .v{text-align:right}
.mk-tip .buy{color:var(--up)} .mk-tip .sell{color:var(--down)}
.mk-ff[aria-pressed="true"]{color:var(--warn);border-color:var(--warn);background:var(--warn-soft)}
.mk-ff-note{max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mk-head-l{display:flex;align-items:center;gap:12px;min-width:0;flex:1 1 auto}
.mk-head-l [data-role="chart-title"]{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mk-tf{display:inline-flex;flex:none;border:1px solid var(--border);border-radius:var(--r-sm);overflow:hidden}
.panel-head .actions{flex:none}
.mk-ff-note{max-width:190px}
.mk-tf button{all:unset;cursor:pointer;padding:2px 8px;font:600 var(--fs-xs)/1.5 var(--mono);color:var(--text-mut);
  letter-spacing:0;text-transform:none}
.mk-tf button+button{border-left:1px solid var(--border)}
.mk-tf button:hover{color:var(--text);background:var(--bg-3)}
.mk-tf button[aria-pressed="true"]{color:var(--accent-2);background:var(--bg-3)}
.mk-tf button:focus-visible{outline:2px solid var(--accent-2);outline-offset:-2px}
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
                    `<button type="button" data-tf="${m}" aria-pressed="false" title="${m >= 60 ? `${m / 60} hour` : `${m} minute`} candles">${tfLabel(m)}</button>`).join('')}</span></span>
                <span class="actions">
                  <span class="chip mk-ff-note" data-role="ff-note" hidden></span>
                  <button type="button" class="btn btn-ghost btn-sm mk-ff" data-role="ff" hidden
                    title="Run the market ${FF_FACTOR}× faster until one of your strategies trades, then back to normal speed">&#9193; ${FF_FACTOR}×</button>
                  <span class="chip" data-role="vwap-chip" hidden>VWAP <b data-role="vwap"></b></span>
                </span>
              </div>
              <div class="panel-body flush chart-host">
                <div data-role="chart"></div>
                <div class="ohlc-readout" data-role="readout" hidden></div>
                <div class="mk-tip" data-role="tip" hidden></div>
                <div class="empty" data-role="empty">
                  ${ui.icon('candles')}
                  <div>No market yet.<br><span class="dim">Start a simulation to bring the venue to life.</span></div>
                  <button type="button" class="btn btn-primary" data-role="empty-start">START SIMULATION</button>
                </div>
              </div>
            </section>

            <section class="panel" aria-label="Order book">
              <div class="panel-head"><span>Order Book</span></div>
              <div class="panel-body flush" data-role="book">
                <div data-role="asks"></div>
                <div class="ob-spread" data-role="spread">spread —</div>
                <div data-role="bids"></div>
              </div>
            </section>

            <section class="panel" aria-label="Trade tape">
              <div class="panel-head"><span>Tape</span>
                <span class="actions"><button type="button" class="btn btn-ghost btn-icon btn-sm" data-role="pop-tape" aria-label="Open full tape in a new tab">${ui.icon('popout')}</button></span>
              </div>
              <div class="panel-body flush scroll-y"><div class="tape-list" data-role="tape"></div></div>
            </section>

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
      tape: q('tape'), lb: q('lb'), vwapChip: q('vwap-chip'), vwap: q('vwap'),
    };

    q('empty-start').addEventListener('click', () => document.getElementById('tb-start')?.click());
    q('pop-tape').addEventListener('click', () => window.open('pages/trades.html', '_blank', 'noopener'));
    q('pop-agents').addEventListener('click', () => window.open('pages/agents.html', '_blank', 'noopener'));

    /* ---- resizable layout ---- */
    this._splits.push(splitter.vSplit(el.querySelector('[data-role="rows"]'), {
      sizes: [72, 28], min: [220, 110], storageKey: 'chronos.split.market.rows',
    }));
    this._splits.push(splitter.hSplit(el.querySelector('[data-role="cols"]'), {
      sizes: [58, 20, 22], min: [320, 172, 180], storageKey: 'chronos.split.market.cols',
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
      const cpText = cps.slice(0, 3).map(([id, q]) => `${id}${whoIs(id)} ${shares(q)}`).join(', ')
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
          <span class="k">Strategy</span><span class="v">${[...a.agents].join(', ')}</span>
          <span class="k">Other side</span><span class="v">${cpText}</span>
        </div>`;
    };
    this._chart.hover((h) => {
      const slot = h && mineIn(h.time);
      if (!slot) { tip.hidden = true; return; }
      tip.innerHTML = `<h4>Your trade · ${fmt.dtime(h.time)}${this._tf > 1 ? ` · ${tfLabel(this._tf)} candle` : ''}</h4>`
        + (slot.BUY ? sideBlock('BUY', slot.BUY) : '')
        + (slot.SELL ? sideBlock('SELL', slot.SELL) : '');
      tip.hidden = false;
      // keep the box inside the chart: right of the pointer, or left of it near the right edge
      const host = tip.parentElement.getBoundingClientRect();
      const w = tip.offsetWidth;
      const hgt = tip.offsetHeight;
      const x = h.x + 16 + w > host.width ? h.x - 16 - w : h.x + 16;
      const y = Math.min(Math.max(8, h.y - hgt / 2), Math.max(8, host.height - hgt - 8));
      tip.style.left = `${Math.max(8, x)}px`;
      tip.style.top = `${y}px`;
    });

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
      tip.hidden = true;
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
      if (!tick) return;
      this._els.empty.style.display = 'none';
      this._els.chartTitle.innerHTML = `${tick.symbol} · Price &amp; Volume`;
      if (Number.isFinite(tick.vwap)) {
        this._els.vwapChip.hidden = false;
        this._els.vwap.textContent = fmt.num(tick.vwap);
      }

      // Candle: the store keeps one bar per sim-minute; the chart shows the chosen timeframe, so the newest
      // candle is rebuilt from the 1-minute bars of its period and grows tick by tick, like a live chart.
      if (tick.bar && Number.isFinite(tick.unix_time)) {
        const minute = tick.unix_time - (tick.unix_time % 60);
        const start = bucketOf(minute, this._tf);
        const part = store.barsTail(this._tf).filter((b) => b.time >= start && b.time <= minute);
        const c = aggregate(part, this._tf)[0];
        if (c && (!this._latestBar || c.time >= this._latestBar.time)) {
          this._chart.update(c);
          this._latestBar = c;
          if (!this._hovering) renderReadout(null);
          renderMarkers();
        }
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

    // Seed from the ring buffer (oldest first → newest ends up on top).
    for (const t of store.recentTrades(TAPE_CAP)) pushTrade(t);
    if (!this._els.tape.children.length) {
      this._els.tape.innerHTML = '<div class="dim" style="padding:var(--s-3)">no prints yet</div>';
      this._tapeEmpty = true;
    }

    this._subs.push(store.onTrade((t) => {
      if (this._tapeEmpty) { this._els.tape.innerHTML = ''; this._tapeEmpty = false; }
      pushTrade(t);
      if (store.mySide(t)) { addMine(t); renderMarkers(); }
    }));
    this._subs.push(store.onTick(applyTick));
    // A different run started (new stock / replay / START / reset): start the chart and tape afresh.
    this._subs.push(store.onReset(() => {
      if (this._chart) { this._chart.setData([]); this._chart.setMarkers([]); }   // timeframe stays as chosen
      this._mine.clear();
      this._mineDirty = false;
      tip.hidden = true;
      this._latestBar = null;
      this._els.readout.hidden = true;
      this._els.tape.innerHTML = '<div class="dim" style="padding:var(--s-3)">no prints yet</div>';
      this._tapeEmpty = true;
    }));

    /* ---- fast-forward: FF_FACTOR× until one of the user's strategies trades (tools/sim_speed.py) ---- */
    const ffBtn = q('ff');
    const ffNote = q('ff-note');
    let noteTimer = null;
    const showSpeed = (s) => {
      if (!s) return;
      ffBtn.hidden = false;
      const on = !!s.fast_forward;
      ffBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
      ffBtn.innerHTML = on ? `&#9632; Stop ${s.factor}×` : `&#9193; ${FF_FACTOR}×`;
      ffBtn.title = on
        ? `Running ${s.factor}× faster until one of your strategies trades. Click to go back to normal speed.`
        : `Run the market ${FF_FACTOR}× faster until one of your strategies trades, then back to normal speed`;
      if (on) { ffNote.hidden = false; ffNote.textContent = `${s.factor}× until your next trade`; }
    };
    const refreshSpeed = () => ctx.api.speed.get().then(showSpeed).catch((err) => {
      ffBtn.hidden = true;                       // 501: an engine without fast-forward (not started via run_local)
      if (err && err.status !== 501) console.warn('[market] speed', err);
    });
    ffBtn.addEventListener('click', async () => {
      const on = ffBtn.getAttribute('aria-pressed') === 'true';
      ffBtn.disabled = true;
      try {
        const s = await ctx.api.speed.set(on ? { action: 'normal' } : { action: 'fast_forward', factor: FF_FACTOR });
        showSpeed(s);
        if (!s.fast_forward) ffNote.hidden = true;
      } catch (err) {
        ui.toast(`Fast-forward failed: ${err.detail || err.message}`, 'bad');
      } finally {
        ffBtn.disabled = false;
      }
    });
    this._subs.push(store.onEvent((ev) => {
      if (ev.kind !== 'speed') return;
      showSpeed(ev);
      if (!ev.fast_forward) {
        clearTimeout(noteTimer);
        if (ev.agent_id) {                       // stopped by the user's own fill
          ffNote.hidden = false;
          ffNote.textContent = 'Back to 1× — your trade';
          ffNote.title = ev.reason || '';
          noteTimer = setTimeout(() => { ffNote.hidden = true; }, 8000);
        } else {
          ffNote.hidden = true;
        }
      }
    }));
    this._subs.push(store.onReset(() => { ffNote.hidden = true; refreshSpeed(); }));
    this._subs.push(() => clearTimeout(noteTimer));
    refreshSpeed();

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
    this._els = {};
  },
};
