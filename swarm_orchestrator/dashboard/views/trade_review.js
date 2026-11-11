/* ============================================================================
   views/trade_review.js — "Your trades": every trade of the user's strategies in
   this run, on the right of the main page (shown when the market is paused or
   stopped, or when a trade is clicked on the chart).

   Data: GET /api/results (strategy names) and /api/results/{sid}/trades (every fill,
   recorded by the bridge even while no page was open), filtered to the current run.
   Fills of one strategy in one market minute on one side are one "order" (a strategy
   sends at most a few orders a minute). Prices around an order come from the 1-minute
   candles this page has seen (store.bars()).
   ============================================================================ */

const STYLE_ID = 'tr-style';
const CSS = `
.tr{display:flex;flex-direction:column;height:100%;min-height:0;background:var(--bg-1)}
.tr-head{display:flex;align-items:center;gap:8px;padding:8px 8px 8px 12px;border-bottom:1px solid var(--border)}
.tr-head h3{flex:1;min-width:0;margin:0;font-size:13px;font-weight:600;color:var(--text-hi);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tr-head h3 small{font-weight:400;color:var(--text-mut);margin-left:6px}
.tr-x,.tr-backbtn{all:unset;cursor:pointer;color:var(--text-mut);border-radius:4px;padding:2px 6px;font-size:13px}
.tr-x{font-size:18px;line-height:1;padding:0 6px}
.tr-x:hover,.tr-backbtn:hover{color:var(--text-hi);background:var(--bg-3)}
.tr-x:focus-visible,.tr-backbtn:focus-visible{outline:2px solid var(--accent)}
.tr-body{flex:1 1 auto;min-height:0;overflow:auto}
.tr-msg{padding:16px 12px;color:var(--text-mut);font-size:12.5px;line-height:1.5}
.tr-strat{padding:10px 12px;border-bottom:1px solid var(--border);display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;font-size:12px}
.tr-strat .nm{grid-column:1/-1;font-size:13px;font-weight:600;color:var(--text-hi)}
.tr-strat .nm small{font-weight:400;color:var(--text-mut);margin-left:6px}
.tr-strat .k{color:var(--text-mut)}
.tr-strat .v{color:var(--text-hi);text-align:right}
.tr-list{width:100%;border-collapse:collapse;font-size:12px}
.tr-list th{position:sticky;top:0;background:var(--bg-1);text-align:left;font-weight:500;color:var(--text-mut);
  padding:6px 12px;border-bottom:1px solid var(--border)}
.tr-list td{padding:6px 12px;border-bottom:1px solid var(--border);white-space:nowrap}
.tr-list .r{text-align:right}
.tr-list tbody tr{cursor:pointer}
.tr-list tbody tr:hover,.tr-list tbody tr:focus-visible{background:var(--bg-2);outline:none}
.tr-buy{color:var(--up);font-weight:600}.tr-sell{color:var(--down);font-weight:600}
.tr-d{padding:12px}
.tr-d h4{margin:0 0 2px;font-size:15px;font-weight:600;color:var(--text-hi)}
.tr-d .sub{color:var(--text-mut);font-size:12px;margin-bottom:12px}
.tr-kv{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;font-size:12.5px;margin-bottom:14px}
.tr-kv .k{color:var(--text-mut)}
.tr-kv .v{color:var(--text-hi);text-align:right}
.tr-sec{font-size:12px;font-weight:600;color:var(--text-hi);margin:14px 0 6px}
.tr-good{color:var(--up)}.tr-bad{color:var(--down)}
.tr-note{color:var(--text-dim);font-size:11.5px;line-height:1.45;margin:6px 0 0}
.tr-d .tr-list td,.tr-d .tr-list th{padding:5px 6px}
.tr-d .btn{margin-top:12px}
`;

const SHARES = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });
const shares = (n) => SHARES.format(n);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/** Who took the other side, in words (engine agent-id conventions, engine/matching_engine._classify_agent). */
export function whoIs(id) {
  const s = String(id || '');
  if (/SIM_LP/.test(s)) return 'resting book';
  if (/GHOST/.test(s)) return 'background flow';
  if (/WHALE/.test(s)) return 'institution';
  if (/GEMINI|ORACLE/.test(s)) return 'news trader';
  if (/MM/.test(s)) return 'market maker';
  if (/^(STRAT_|USER_)/.test(s)) return 'a strategy';
  return 'retail';
}

/* ---- strategy names (agent_id -> name), shared with the chart's trade box ---- */
let namesAt = 0;
let names = new Map();
let namesBusy = null;
/** @returns {Promise<Map<string,{name:string, strategy_id:string}>>} cached for 10 s */
export function strategyNames(api) {
  if (Date.now() - namesAt < 10000) return Promise.resolve(names);
  if (!namesBusy) {
    namesBusy = api.results.list().then((list) => {
      names = new Map((list || []).map((s) => [s.agent_id, { name: s.name, strategy_id: s.strategy_id }]));
      namesAt = Date.now();
      return names;
    }).catch(() => names).finally(() => { namesBusy = null; });
  }
  return namesBusy;
}
/** "big_block (STRAT_1a2b3c4d)" or the bare id when the name is not known yet. */
export function placedBy(agent) {
  const n = names.get(agent);
  return n && n.name ? `${n.name} (${agent})` : agent;
}

/**
 * @param {HTMLElement} el
 * @param {object} ctx   the shell's {api, store, fmt, ui}
 * @param {{onClose:()=>void, onLocate:(minute:number)=>void}} hooks
 */
export function mountTradeReview(el, ctx, hooks) {
  if (!document.getElementById(STYLE_ID)) {
    const st = document.createElement('style');
    st.id = STYLE_ID;
    st.textContent = CSS;
    document.head.appendChild(st);
  }
  const { api, store, fmt } = ctx;
  let orders = [];            // [{key, agent, side, minute, qty, notional, fills:[...]}], oldest first
  let strategies = [];        // [{agent, name, ...summary}]
  let showing = null;         // order key on screen, or null for the list
  let loadSeq = 0;

  el.innerHTML = `
<div class="tr">
  <div class="tr-head">
    <button type="button" class="tr-backbtn" data-r="back" hidden>‹ All trades</button>
    <h3 data-r="title">Your trades</h3>
    <button type="button" class="tr-x" data-r="close" title="Back to the order book and trades" aria-label="Close your trades">×</button>
  </div>
  <div class="tr-body" data-r="body"><div class="tr-msg">Loading your trades…</div></div>
</div>`;
  const $ = (r) => el.querySelector(`[data-r="${r}"]`);
  const body = $('body');
  $('close').addEventListener('click', () => hooks.onClose());
  $('back').addEventListener('click', () => { showing = null; render(); });

  const minuteOf = (t) => t - (t % 60);
  const runId = () => (store.latestTick() || {}).run_id;
  const lastPrice = () => (store.latestTick() || {}).last_price;

  /** Close of the 1-minute candle at `minute` (or the nearest earlier one within 5 minutes). */
  function closeAt(minute) {
    const bars = store.bars();
    for (let i = bars.length - 1; i >= 0; i -= 1) {
      if (bars[i].time <= minute) return minute - bars[i].time <= 300 ? bars[i].close : null;
    }
    return null;
  }

  async function load() {
    const my = ++loadSeq;
    const rid = runId();
    let list;
    try {
      await strategyNames(api);
      list = await api.results.list();
    } catch (err) {
      if (my === loadSeq) body.innerHTML = `<div class="tr-msg">Could not load your trades: ${esc(err.detail || err.message)}</div>`;
      return;
    }
    // aggressor side is not in the server's rows: take it from the prints this page saw
    const seen = new Map();
    for (const t of store.myTrades()) seen.set(`${t.unix_time}|${t.price}|${t.qty}|${t.buyer}|${t.seller}`, t.aggressor);
    const rows = [];
    await Promise.all((list || []).map(async (s) => {
      let tr;
      try { tr = await api.results.trades(s.strategy_id); } catch { return; }
      for (const r of (tr && tr.trades) || tr || []) {
        if (rid && r.run_id !== rid) continue;
        rows.push({ ...r, agent: s.agent_id, name: s.name });
      }
    }));
    if (my !== loadSeq) return;
    rows.sort((a, b) => (a.unix_time - b.unix_time));
    const byKey = new Map();
    for (const r of rows) {
      const minute = minuteOf(r.unix_time);
      const key = `${minute}|${r.side}`;
      let o = byKey.get(key);
      if (!o) {
        o = { key, side: r.side, minute, qty: 0, notional: 0, fills: [], agents: new Set() };
        byKey.set(key, o);
      }
      const buyer = r.side === 'BUY' ? r.agent : r.counterparty;
      const seller = r.side === 'BUY' ? r.counterparty : r.agent;
      const aggr = seen.get(`${r.unix_time}|${r.price}|${r.qty}|${buyer}|${seller}`);
      o.qty += r.qty;
      o.notional += r.qty * r.price;
      o.agents.add(r.agent);
      o.fills.push({ time: r.unix_time, price: r.price, qty: r.qty, cp: r.counterparty, agent: r.agent,
                     crossed: aggr ? aggr === r.side : null });
    }
    orders = [...byKey.values()].sort((a, b) => a.minute - b.minute || (a.side < b.side ? -1 : 1));
    // running position per strategy, for "position after"
    const pos = new Map();
    for (const o of orders) {
      for (const f of o.fills) pos.set(f.agent, (pos.get(f.agent) || 0) + (o.side === 'BUY' ? f.qty : -f.qty));
      o.posAfter = [...o.agents].map((a) => [a, pos.get(a)]);
    }
    const sums = new Map();
    for (const r of rows) {
      const s = sums.get(r.agent) || { agent: r.agent, name: r.name, bq: 0, bv: 0, sq: 0, sv: 0, n: 0 };
      if (r.side === 'BUY') { s.bq += r.qty; s.bv += r.qty * r.price; } else { s.sq += r.qty; s.sv += r.qty * r.price; }
      s.n += 1;
      sums.set(r.agent, s);
    }
    strategies = [...sums.values()];
    render();
  }

  function render() {
    $('back').hidden = !showing;
    if (showing) { renderOrder(orders.find((o) => o.key === showing)); return; }
    $('title').innerHTML = `Your trades<small>${orders.length} order${orders.length === 1 ? '' : 's'}</small>`;
    if (!orders.length) {
      body.innerHTML = `<div class="tr-msg">No trades from your strategies in this run yet.<br>
        Run one from <a href="#/strategies">Strategies</a>, then come back here when the market is paused or stopped.</div>`;
      return;
    }
    const lp = lastPrice();
    const sumHtml = strategies.map((s) => {
      const posn = s.bq - s.sq;
      const pnl = Number.isFinite(lp) ? s.sv - s.bv + posn * lp : null;
      return `<div class="tr-strat">
        <div class="nm">${esc(s.name || s.agent)}<small>${esc(s.agent)}</small></div>
        <span class="k">Bought</span><span class="v">${fmt.num(s.bq, 0)}${s.bq ? ` @ ₹${fmt.num(s.bv / s.bq)}` : ''}</span>
        <span class="k">Sold</span><span class="v">${fmt.num(s.sq, 0)}${s.sq ? ` @ ₹${fmt.num(s.sv / s.sq)}` : ''}</span>
        <span class="k">Position now</span><span class="v">${fmt.num(posn, 0)}</span>
        <span class="k">P&amp;L at ₹${fmt.num(lp)}</span><span class="v ${pnl >= 0 ? 'tr-good' : 'tr-bad'}">${pnl == null ? '—' : `${pnl >= 0 ? '+' : '−'}${fmt.inr(Math.abs(pnl))}`}</span>
      </div>`;
    }).join('');
    const rowsHtml = orders.map((o) => `
      <tr tabindex="0" data-key="${o.key}">
        <td>${fmt.time(o.minute)}</td>
        <td class="${o.side === 'BUY' ? 'tr-buy' : 'tr-sell'}">${o.side === 'BUY' ? 'Buy' : 'Sell'}</td>
        <td class="r">${fmt.num(o.qty, 0)}</td>
        <td class="r">₹${fmt.num(o.notional / o.qty)}</td>
        <td class="r">${o.fills.length}</td>
      </tr>`).join('');
    body.innerHTML = `${sumHtml}
      <table class="tr-list"><thead><tr><th>Time</th><th>Side</th><th class="r">Qty</th><th class="r">Avg price</th><th class="r">Fills</th></tr></thead>
      <tbody>${rowsHtml}</tbody></table>`;
    const tb = body.querySelector('tbody');
    const open = (e) => { const tr = e.target.closest('tr[data-key]'); if (tr) openOrder(tr.dataset.key); };
    tb.addEventListener('click', open);
    tb.addEventListener('keydown', (e) => { if (e.key === 'Enter') open(e); });
  }

  /** How far `to` is from `from` in the order's favour (up after a buy, down after a sell): [₹ per share, %]. */
  const favour = (side, from, to) => {
    const d = (side === 'BUY' ? 1 : -1) * (to - from);
    return [d, (d / from) * 100];
  };
  /** "₹2.04/share (0.15%) better", green when in the order's favour; `qty` adds the rupees on the whole order. */
  const moveCell = ([rs, pct], good, bad, qty = 0) => (!Number.isFinite(rs) ? '—'
    : `<span class="${rs >= 0 ? 'tr-good' : 'tr-bad'}">₹${fmt.num(Math.abs(rs))}/share (${Math.abs(pct).toFixed(2)}%) `
      + `${rs >= 0 ? good : bad}${qty ? ` · ${fmt.inr(Math.abs(rs) * qty)} on the order` : ''}</span>`);

  function renderOrder(o) {
    if (!o) { showing = null; render(); return; }
    const avg = o.notional / o.qty;
    const prices = o.fills.map((f) => f.price);
    const lo = Math.min(...prices);
    const hi = Math.max(...prices);
    const before = closeAt(o.minute - 60);
    const later = [5, 15, 60].map((m) => [m, closeAt(o.minute + m * 60)]);
    const crossedKnown = o.fills.filter((f) => f.crossed != null);
    const crossedQty = crossedKnown.filter((f) => f.crossed).reduce((a, f) => a + f.qty, 0);
    const knownQty = crossedKnown.reduce((a, f) => a + f.qty, 0);
    const by = [...o.agents].map((a) => esc(placedBy(a))).join(', ');
    const posAfter = o.posAfter.map(([a, p]) => `${fmt.num(p, 0)}${o.agents.size > 1 ? ` (${esc(a)})` : ''}`).join(', ');
    const cps = new Map();
    for (const f of o.fills) cps.set(f.cp, (cps.get(f.cp) || 0) + f.qty);
    $('title').innerHTML = `${o.side === 'BUY' ? 'Buy' : 'Sell'} order<small>${fmt.dtime(o.minute)}</small>`;
    body.innerHTML = `<div class="tr-d">
      <h4 class="${o.side === 'BUY' ? 'tr-buy' : 'tr-sell'}">${o.side === 'BUY' ? 'Bought' : 'Sold'} ${fmt.num(o.qty, 0)} shares</h4>
      <div class="sub">${fmt.dtime(o.minute)} · ${esc((store.latestTick() || {}).symbol || '')}</div>
      <div class="tr-kv">
        <span class="k">Placed by</span><span class="v">${by}</span>
        <span class="k">Average price</span><span class="v">₹${fmt.num(avg)}</span>
        <span class="k">Price range</span><span class="v">${lo === hi ? `₹${fmt.num(lo)}` : `₹${fmt.num(lo)} – ₹${fmt.num(hi)}`}</span>
        <span class="k">Value</span><span class="v">${fmt.inr(o.notional)}</span>
        <span class="k">Fills</span><span class="v">${o.fills.length}${knownQty ? ` · ${Math.round((100 * crossedQty) / knownQty)}% crossed the spread` : ''}</span>
        <span class="k">Position after</span><span class="v">${posAfter}</span>
      </div>
      <div class="tr-sec">Price around the order</div>
      <div class="tr-kv">
        <span class="k">Last price before</span><span class="v">${before == null ? '—' : `₹${fmt.num(before)}`}</span>
        <span class="k">Vs last price</span><span class="v">${before == null ? '—' : moveCell(favour(o.side, avg, before), 'better', 'worse', o.qty)}</span>
        ${later.map(([m, c]) => `<span class="k">${m} min later</span><span class="v">${c == null ? '—' : `₹${fmt.num(c)} · ${moveCell(favour(o.side, avg, c), 'your way', 'against you')}`}</span>`).join('')}
      </div>
      <p class="tr-note">Vs last price: your average fill against the last price before the order (a buy below it or a
        sell above it is better). Later: where the price went from your average fill, per share. A dash means this
        page had not seen that minute.</p>
      <div class="tr-sec">Who was on the other side</div>
      <div class="tr-kv">${[...cps.entries()].sort((a, b) => b[1] - a[1]).map(([id, q]) =>
        `<span class="k">${esc(id)} · ${whoIs(id)}</span><span class="v">${fmt.num(q, 0)}</span>`).join('')}</div>
      <div class="tr-sec">Fills</div>
      <table class="tr-list"><thead><tr><th>Time</th><th class="r">Price</th><th class="r">Qty</th><th>Other side</th></tr></thead>
        <tbody>${o.fills.map((f) => `<tr><td>${fmt.time(f.time)}</td><td class="r">₹${fmt.num(f.price)}</td>
          <td class="r">${fmt.num(f.qty, 0)}</td><td>${esc(f.cp)}${f.crossed == null ? '' : f.crossed ? ' · you crossed' : ' · you rested'}</td></tr>`).join('')}</tbody></table>
      <button type="button" class="btn btn-sm" data-r="locate">Show on chart</button>
    </div>`;
    $('locate').addEventListener('click', () => hooks.onLocate(o.minute));
  }

  function openOrder(key) {
    showing = key;
    if (orders.some((o) => o.key === key)) render();
    else load();                      // a trade the list has not loaded yet
  }

  let timer = null;
  const unsub = store.onTrade((t) => {
    if (!store.mySide(t) || el.hidden || el.closest('[hidden]')) return;
    clearTimeout(timer);
    timer = setTimeout(load, 1200);   // the bridge records the fill first
  });

  return {
    /** Reload from the server and show the list (or the order being shown). */
    show() { load(); },
    /** Open one order: key = `${minute}|${side}` (minute = unix seconds at the minute start). */
    openOrder,
    destroy() { unsub(); clearTimeout(timer); el.innerHTML = ''; },
  };
}

export default mountTradeReview;
