/* ============================================================================
   views/market.js — the hero view: candle+volume chart, order book,
   trade tape, agent leaderboard. All data from the live tick/trade stream.
   ============================================================================ */

const TAPE_CAP = 60;
const BOOK_LEVELS = 10;

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
              <div class="panel-head"><span data-role="chart-title">Price &amp; Volume</span>
                <span class="actions"><span class="chip" data-role="vwap-chip" hidden>VWAP <b data-role="vwap"></b></span></span>
              </div>
              <div class="panel-body flush chart-host">
                <div data-role="chart"></div>
                <div class="ohlc-readout" data-role="readout" hidden></div>
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

      // Candle: one bar per sim-minute, keyed to the minute boundary.
      if (tick.bar && Number.isFinite(tick.unix_time)) {
        const barTime = tick.unix_time - (tick.unix_time % 60);
        const b = { time: barTime, o: tick.bar.o, h: tick.bar.h, l: tick.bar.l, c: tick.bar.c, v: tick.bar.v };
        this._chart.update(b);
        this._latestBar = { time: barTime, open: b.o, high: b.h, low: b.l, close: b.c, volume: b.v };
        if (!this._hovering) renderReadout(null);
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
      const div = document.createElement('div');
      div.className = `tape-row ${cls}`;
      div.innerHTML = `<span class="tt">${fmt.time(t.unix_time)}</span>`
        + `<span class="tp">${fmt.num(t.price)}</span>`
        + `<span class="tq">${fmt.num(t.qty, 0)}</span>`
        + `<span class="ta">${t.aggressor}</span>`;
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
    }));
    this._subs.push(store.onTick(applyTick));

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
    this._els = {};
  },
};
