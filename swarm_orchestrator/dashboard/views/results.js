/* ============================================================================
   views/results.js — master-detail strategy reports (PROTOCOL_V2 §2).
   Left: every strategy the session has run. Right: summary stats, equity
   curve from day snapshots, per-day outcomes, filtered trade log, downloads.
   ============================================================================ */

const LIST_POLL_MS = 5000;

export default {
  id: 'results',
  title: 'Results',
  icon: 'report',
  group: 'Trading',

  _splits: [],
  _timers: [],
  _chart: null,
  _els: {},
  _selected: null,
  _ctx: null,

  async mount(el, ctx) {
    const { api, ui, splitter } = ctx;
    this._ctx = ctx;
    this._selected = null;

    el.innerHTML = `
      <div class="view-root"><div class="view-cols" data-role="cols">
        <section class="panel" aria-label="Strategy list">
          <div class="panel-head"><span>Strategy Reports</span>
            <span class="actions"><button type="button" class="btn btn-ghost btn-icon btn-sm" data-role="refresh" aria-label="Refresh list">${ui.icon('refresh')}</button></span>
          </div>
          <div class="panel-body flush scroll-y" data-role="list">
            <div class="skeleton skel-line"></div><div class="skeleton skel-line"></div><div class="skeleton skel-line"></div>
          </div>
        </section>
        <div class="view-rows" data-role="detail" style="gap:var(--s-3);padding-left:var(--s-1)">
          <div class="empty" data-role="detail-empty">
            ${ui.icon('report')}
            <div>No report selected.<br><span class="dim">Run a strategy and its trades, P&amp;L and per-day outcomes are recorded here.</span></div>
            <a class="btn" href="#/strategies">Open Strategies</a>
          </div>
        </div>
      </div></div>`;

    const q = (r) => el.querySelector(`[data-role="${r}"]`);
    this._els = { list: q('list'), detail: q('detail'), detailEmpty: q('detail-empty') };

    this._splits.push(splitter.hSplit(q('cols'), {
      sizes: [26, 74], min: [220, 430], storageKey: 'chronos.split.results',
    }));

    q('refresh').addEventListener('click', () => this._renderList());
    this._els.list.addEventListener('click', (e) => {
      const item = e.target.closest('.res-item[data-id]');
      if (item) this._select(item.dataset.id);
    });
    this._els.list.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter') return;
      const item = e.target.closest('.res-item[data-id]');
      if (item) this._select(item.dataset.id);
    });

    await this._renderList(true);
    this._timers.push(setInterval(() => this._renderList(), LIST_POLL_MS));
  },

  /* ------------------------------ list ---------------------------------- */

  async _renderList(autoSelect = false) {
    const { api, ui, fmt } = this._ctx;
    let list;
    try { list = await api.results.list(); } catch (err) {
      if (this._els.list && this._els.list.isConnected) {
        this._els.list.innerHTML = `<div class="empty">${ui.icon('report')}<div>Results unavailable<br><span class="dim">${err.detail || err.message}</span></div></div>`;
      }
      return;
    }
    if (!this._els.list || !this._els.list.isConnected) return;
    if (!Array.isArray(list) || !list.length) {
      this._els.list.innerHTML = `<div class="empty">${ui.icon('report')}<div>No strategies have run yet.<br><span class="dim">Reports appear the moment one trades.</span></div></div>`;
      return;
    }
    this._els.list.innerHTML = list.map((s) => `
      <div class="res-item ${s.strategy_id === this._selected ? 'active' : ''}" data-id="${s.strategy_id}" tabindex="0" role="button" aria-label="Open report for ${esc(s.name)}">
        <div class="rrow">
          <span class="rname">${esc(s.name)}</span>
          <span class="chip ${s.status === 'running' ? 'up' : ''}">${esc(s.status || '')}</span>
        </div>
        <div class="rrow">
          <span class="ragent">${esc(s.agent_id || '')}</span>
          <span class="grow"></span>
          <span class="dim" style="font-size:var(--fs-xs)">${fmt.num(s.total_trades ?? 0, 0)} trades</span>
          <span class="rpnl ${((s.total_pnl ?? 0) >= 0) ? 'num-up' : 'num-down'}">${signedInr(fmt, s.total_pnl ?? 0)}</span>
        </div>
      </div>`).join('');
    if (autoSelect && !this._selected && list.length) this._select(list[0].strategy_id);
  },

  /* ----------------------------- detail --------------------------------- */

  async _select(id) {
    const { api, ui, fmt, charts, splitter } = this._ctx;
    this._selected = id;
    this._els.list.querySelectorAll('.res-item').forEach((n) => n.classList.toggle('active', n.dataset.id === id));

    if (this._chart) { this._chart.dispose(); this._chart = null; }
    const host = this._els.detail;
    host.innerHTML = '<div class="skeleton skel-block"></div>';

    let d;
    try { d = await api.results.detail(id); } catch (err) {
      host.innerHTML = `<div class="empty">${ui.icon('report')}<div>Report failed to load<br><span class="dim">${err.detail || err.message}</span></div></div>`;
      return;
    }
    if (!host.isConnected || this._selected !== id) return;

    const days = Array.isArray(d.days) ? d.days : [];
    const totalTrades = d.total_trades ?? days.reduce((a, x) => a + (x.n_trades || 0), 0);
    const realized = d.realized_pnl ?? d.total_pnl ?? 0;
    const nDays = d.n_days ?? days.length;
    const maxDD = days.length ? Math.max(...days.map((x) => x.max_drawdown ?? 0)) : (d.max_drawdown ?? 0);
    let winNum = 0; let winDen = 0;
    for (const day of days) {
      if (day.win_rate != null && day.n_trades) { winNum += day.win_rate * day.n_trades; winDen += day.n_trades; }
    }
    const winRate = winDen ? winNum / winDen : null;

    host.innerHTML = `
      <div style="display:flex;align-items:center;gap:var(--s-3);flex:0 0 auto;flex-wrap:wrap">
        <h2 style="font-size:var(--fs-lg)">${esc(d.name || id)}</h2>
        <span class="chip mono">${esc(d.agent_id || '')}</span>
        <span class="chip ${d.status === 'running' ? 'up' : ''}">${esc(d.status || '')}</span>
        <span class="grow"></span>
        <button type="button" class="btn btn-sm" data-role="dl-trades">${ui.icon('download')} Trades CSV</button>
        <button type="button" class="btn btn-sm" data-role="dl-outcomes">${ui.icon('download')} Outcomes CSV</button>
        <button type="button" class="btn btn-primary btn-sm" data-role="dl-pdf">${ui.icon('report')} PDF Report</button>
      </div>

      <div class="stat-row" role="group" aria-label="Summary statistics">
        <span class="stat"><span class="k">Total trades</span><span class="v">${fmt.num(totalTrades, 0)}</span></span>
        <span class="stat"><span class="k">Realized P&amp;L</span><span class="v ${realized >= 0 ? 'num-up' : 'num-down'}">${signedInr(fmt, realized)}</span></span>
        <span class="stat"><span class="k">Win rate</span><span class="v">${winRate != null ? fmt.pct(winRate) : '—'}</span></span>
        <span class="stat"><span class="k">Days</span><span class="v">${fmt.num(nDays, 0)}</span></span>
        <span class="stat"><span class="k">Max drawdown</span><span class="v ${maxDD > 0 ? 'num-down' : ''}">${fmt.inr(maxDD)}</span></span>
      </div>

      <div class="view-rows grow" data-role="stack">
        <section class="panel" aria-label="Equity curve">
          <div class="panel-head"><span>Equity Curve <span class="dim" style="text-transform:none;letter-spacing:0">— day snapshots</span></span></div>
          <div class="panel-body flush chart-host"><div data-role="equity"></div></div>
        </section>
        <div class="view-cols" data-role="tables">
          <section class="panel" aria-label="Per-day outcomes">
            <div class="panel-head"><span>Per-Day Outcomes</span></div>
            <div class="panel-body flush scroll-y" data-role="outcomes"></div>
          </section>
          <section class="panel" aria-label="Trade log">
            <div class="panel-head"><span>Trade Log</span>
              <span class="actions"><select class="select" data-role="dayfilter" aria-label="Filter trades by day" style="padding:4px 8px;font-size:var(--fs-xs)"></select></span>
            </div>
            <div class="panel-body flush scroll-y" data-role="trades"></div>
          </section>
        </div>
      </div>`;

    const q = (r) => host.querySelector(`[data-role="${r}"]`);
    q('dl-trades').addEventListener('click', () => window.open(api.results.csvUrl(id), '_blank', 'noopener'));
    q('dl-outcomes').addEventListener('click', () => window.open(api.results.outcomesCsvUrl(id), '_blank', 'noopener'));
    q('dl-pdf').addEventListener('click', () => window.open(api.results.pdfUrl(id), '_blank', 'noopener'));

    this._splits.push(splitter.vSplit(q('stack'), {
      sizes: [42, 58], min: [140, 160], storageKey: 'chronos.split.results.stack',
    }));
    this._splits.push(splitter.hSplit(q('tables'), {
      sizes: [50, 50], min: [220, 240], storageKey: 'chronos.split.results.tables',
    }));

    /* -- equity curve from day snapshots -- */
    const eqHost = q('equity');
    if (days.length) {
      const points = [];
      if (Number.isFinite(days[0].start_equity)) points.push({ time: 0, value: days[0].start_equity });
      days.forEach((day, i) => {
        if (Number.isFinite(day.end_equity)) points.push({ time: i + 1, value: day.end_equity });
      });
      const label = (t) => (t === 0 ? 'start' : `D${days[t - 1]?.day_count ?? t}`);
      this._chart = charts.lineChart(eqHost, {
        tickFormatter: label,
        priceFormatter: (v) => fmt.compact(v),
      });
      this._chart.setData(points);
    } else {
      eqHost.innerHTML = '<div class="chart-offline">no completed day snapshots yet</div>';
    }

    /* -- per-day outcomes -- */
    q('outcomes').innerHTML = days.length ? `
      <table class="grid" aria-label="Per-day outcomes">
        <thead><tr><th>Day</th><th>Run</th><th class="right">Trades</th><th class="right">Gross P&amp;L</th><th class="right">Equity</th><th class="right">Max DD</th><th class="right">Win</th></tr></thead>
        <tbody>${days.map((x) => `
          <tr>
            <td class="mono">D${x.day_count ?? '—'}</td>
            <td class="mono dim" title="${esc(x.run_id || '')}">${esc(shortRun(x.run_id))}</td>
            <td class="num right">${fmt.num(x.n_trades ?? 0, 0)}</td>
            <td class="num right ${((x.gross_pnl ?? 0) >= 0) ? 'num-up' : 'num-down'}">${signedInr(fmt, x.gross_pnl ?? 0)}</td>
            <td class="num right dim">${fmt.compact(x.start_equity)} → ${fmt.compact(x.end_equity)}</td>
            <td class="num right">${fmt.inr(x.max_drawdown ?? 0)}</td>
            <td class="num right">${x.win_rate != null ? fmt.pct(x.win_rate) : '—'}</td>
          </tr>`).join('')}</tbody></table>`
      : `<div class="empty">${ui.icon('clock')}<div><span class="dim">Day buckets close at each session end.</span></div></div>`;

    /* -- trade log + day filter -- */
    const filter = q('dayfilter');
    const dayValues = [...new Set(days.map((x) => x.day_count).filter((x) => x != null))];
    filter.innerHTML = '<option value="">All days</option>'
      + dayValues.map((x) => `<option value="${x}">Day ${x}</option>`).join('');
    const loadTrades = async () => {
      const day = filter.value === '' ? undefined : Number(filter.value);
      const tHost = q('trades');
      tHost.innerHTML = '<div class="skeleton skel-line"></div><div class="skeleton skel-line"></div>';
      let res;
      try { res = await api.results.trades(id, day); } catch (err) {
        tHost.innerHTML = `<div class="empty"><div class="dim">${err.detail || err.message}</div></div>`;
        return;
      }
      if (!tHost.isConnected) return;
      const trades = Array.isArray(res) ? res : (res && res.trades) || [];
      if (!trades.length) {
        tHost.innerHTML = `<div class="empty">${ui.icon('feed')}<div><span class="dim">No fills ${day != null ? `on day ${day}` : 'yet'}.</span></div></div>`;
        return;
      }
      tHost.innerHTML = `
        <table class="grid" aria-label="Trade log">
          <thead><tr><th>Time</th><th>Side</th><th class="right">Price</th><th class="right">Qty</th><th class="right">Cash Δ</th><th>Counterparty</th></tr></thead>
          <tbody>${trades.map((t) => `
            <tr>
              <td class="mono dim">${fmt.dtime(t.unix_time)}</td>
              <td><span class="chip ${t.side === 'BUY' ? 'up' : 'down'}">${esc(t.side || '')}</span></td>
              <td class="num right">${fmt.num(t.price)}</td>
              <td class="num right">${fmt.num(t.qty, 0)}</td>
              <td class="num right ${((t.cash_delta ?? 0) >= 0) ? 'num-up' : 'num-down'}">${signedInr(fmt, t.cash_delta ?? 0)}</td>
              <td class="mono dim">${esc(t.counterparty || '')}</td>
            </tr>`).join('')}</tbody></table>`;
    };
    filter.addEventListener('change', loadTrades);
    loadTrades();
  },

  unmount() {
    this._timers.forEach(clearInterval);
    this._timers = [];
    this._splits.forEach((s) => s.destroy());
    this._splits = [];
    if (this._chart) { this._chart.dispose(); this._chart = null; }
    this._els = {};
    this._selected = null;
    this._ctx = null;
  },
};

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function shortRun(runId) {
  return runId ? String(runId).slice(-8) : '—';
}
function signedInr(fmt, n) {
  return (n < 0 ? '-' : '+') + fmt.inr(Math.abs(n));
}
