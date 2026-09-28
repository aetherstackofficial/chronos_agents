/* ============================================================================
   app.js — Chronos Terminal shell controller.
   Topbar · LHS nav rail · hash router · RHS action panel · global status.
   Views mount into #center with ctx = {store, api, fmt, ui, charts, splitter}.
   ============================================================================ */

import { store } from './lib/store.js';
import { api } from './lib/api.js';
import { fmt } from './lib/format.js';
import { ui } from './lib/ui.js';
import { charts } from './lib/charts.js';
import { splitter } from './lib/splitter.js';

import market from './views/market.js';
import strategies from './views/strategies.js';
import results from './views/results.js';
import montecarlo from './views/montecarlo.js';
import timemachine from './views/timemachine.js';
import copilot from './views/copilot.js';
import narration from './views/narration.js';
import runs from './views/runs.js';
import lab from './views/lab.js';

const ctx = { store, api, fmt, ui, charts, splitter };

/** Fixed nav order (DESIGN.md). Group fallback if a view omits its own. */
const VIEWS = [market, strategies, results, montecarlo, timemachine, copilot, narration, lab, runs];
const GROUP_FALLBACK = {
  market: '', strategies: 'Trading', results: 'Trading',
  montecarlo: 'Lab', timemachine: 'Lab', copilot: 'Lab', narration: 'Lab', lab: 'Lab',
  runs: 'Data',
};
const ICON_FALLBACK = {
  market: 'candles', strategies: 'code', results: 'report', montecarlo: 'fan',
  timemachine: 'clock', copilot: 'sparkles', narration: 'feed', lab: 'flask', runs: 'database',
};
/** Views whose data pops out to a standalone page via the topbar ⤢. */
const POP_PAGES = { market: 'pages/trades.html', results: 'pages/results.html' };

const viewById = {};
for (const v of VIEWS) viewById[v.id] = v;

const el = {
  topbar: document.getElementById('topbar'),
  nav: document.getElementById('nav'),
  center: document.getElementById('center'),
  rhs: document.getElementById('rhs'),
  rhsReopen: document.getElementById('rhs-reopen'),
  wsBanner: document.getElementById('ws-banner'),
  shell: document.getElementById('shell'),
  mode: document.getElementById('tb-mode'),
  modeText: document.getElementById('tb-mode-text'),
  day: document.getElementById('tb-day'),
  dayN: document.getElementById('tb-day-n'),
  clock: document.getElementById('tb-clock'),
  price: document.getElementById('tb-price'),
  regime: document.getElementById('tb-regime'),
  regimeText: document.getElementById('tb-regime-text'),
  conn: document.getElementById('tb-conn'),
  start: document.getElementById('tb-start'),
  pause: document.getElementById('tb-pause'),
  reset: document.getElementById('tb-reset'),
  pop: document.getElementById('tb-pop'),
};

/* ============================== TOPBAR ================================== */

let lastPrice = null;
let reactivePulsed = false;
let paused = false;

function setModeChip(mode) {
  if (!mode) return;
  el.mode.hidden = false;
  el.modeText.textContent = mode;
  el.mode.className = `chip ${mode === 'LIVE' ? 'live' : mode === 'REPLAY' ? 'replay' : 'reactive'}`;
  if (mode === 'REACTIVE' && !reactivePulsed) {
    reactivePulsed = true;
    el.mode.classList.add('pulse');
    setTimeout(() => el.mode.classList.remove('pulse'), 12000);
  }
}

function onTick(tick) {
  setModeChip(tick.mode);
  el.day.hidden = false;
  el.dayN.textContent = tick.day_count ?? '—';
  el.clock.hidden = false;
  el.clock.textContent = fmt.time(tick.unix_time);
  el.price.hidden = false;
  el.price.textContent = fmt.inr(tick.last_price);
  if (lastPrice != null && tick.last_price !== lastPrice) {
    el.price.classList.toggle('up', tick.last_price > lastPrice);
    el.price.classList.toggle('down', tick.last_price < lastPrice);
  }
  lastPrice = tick.last_price;
  el.regime.hidden = false;
  el.regimeText.textContent = tick.regime;
  el.regime.className = `chip ${tick.regime === 'BULL' ? 'up' : tick.regime === 'BEAR' ? 'down' : ''}`;
  paused = !!tick.paused;
  el.pause.disabled = false;
  el.reset.disabled = false;
  el.pause.textContent = paused ? 'RESUME' : 'PAUSE';
}

function simConfig() {
  try { return JSON.parse(localStorage.getItem('chronos.simcfg') || 'null') || {}; } catch { return {}; }
}

/** Open the START (or RESET) configuration modal and launch the run. */
function openStartModal(title = 'Start Simulation') {
  const cfg = simConfig();
  const sectors = ['TECH', 'DEFENSE', 'PHARMA', 'OIL', 'BANK'];
  const opts = sectors.map((s) => `<option value="${s}"${cfg.sector === s ? ' selected' : ''}>${s}</option>`).join('');
  const body = `
    <div class="form-row">
      <label class="field">Symbol<input id="m-symbol" class="input mono" value="${cfg.symbol || 'TCS'}" maxlength="12"></label>
      <label class="field">Sector<select id="m-sector" class="select">${opts}</select></label>
    </div>
    <div class="form-row">
      <label class="field">Start price (₹)<input id="m-price" class="input mono" type="number" step="0.05" min="0.05" value="${cfg.price ?? 190}"></label>
      <label class="field">Seed<input id="m-seed" class="input mono" type="number" value="${cfg.seed ?? 42}"></label>
    </div>
    <p class="hint-line">One physics tick per second = one simulated market minute (09:15 → 15:30 IST).</p>`;
  const m = ui.modal({
    title,
    body,
    actions: [
      { label: 'Cancel' },
      {
        label: 'Start',
        kind: 'primary',
        onClick: async (close, btn) => {
          const payload = {
            symbol: (m.body.querySelector('#m-symbol').value || 'TCS').trim().toUpperCase(),
            sector: m.body.querySelector('#m-sector').value,
            price: parseFloat(m.body.querySelector('#m-price').value) || 190,
            seed: parseInt(m.body.querySelector('#m-seed').value, 10) || 42,
          };
          try {
            const res = await ui.busy(btn, () => api.sim.start(payload));
            localStorage.setItem('chronos.simcfg', JSON.stringify(payload));
            ui.toast(`Simulation started — ${res.run_id || payload.symbol}`, 'good');
            close();
          } catch (err) {
            ui.toast(`Start failed: ${err.detail || err.message}`, 'bad');
          }
        },
      },
    ],
  });
}

el.start.addEventListener('click', () => openStartModal());

el.pause.addEventListener('click', async () => {
  try {
    if (paused) { await api.sim.resume(); ui.toast('Simulation resumed', 'good'); }
    else { await api.sim.stop(); ui.toast('Simulation paused', 'warn'); }
  } catch (err) { ui.toast(`Control failed: ${err.detail || err.message}`, 'bad'); }
});

el.reset.addEventListener('click', async () => {
  const ok = await ui.confirm('Start a fresh run? The current market state will be replaced (the old run stays persisted).');
  if (ok) openStartModal('Reset — New Run');
});

el.pop.innerHTML = ui.icon('popout');
el.pop.addEventListener('click', () => {
  const page = POP_PAGES[currentId];
  if (page) window.open(page, '_blank', 'noopener');
});

store.onTick(onTick);
store.onStatus((connected) => {
  el.conn.classList.toggle('on', connected);
  el.conn.setAttribute('aria-label', `Engine connection: ${connected ? 'connected' : 'disconnected'}`);
  el.wsBanner.hidden = connected;
});

/* =============================== LHS NAV ================================ */

const RAIL_W = 56;

function buildNav() {
  const items = document.createElement('div');
  items.className = 'nav-items';
  let lastGroup = null;
  for (const v of VIEWS) {
    const group = v.group !== undefined && v.group !== '' ? v.group : GROUP_FALLBACK[v.id] || '';
    if (group && group !== lastGroup) {
      const g = document.createElement('div');
      g.className = 'nav-group';
      g.textContent = group;
      items.appendChild(g);
    }
    lastGroup = group || lastGroup;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'nav-item';
    btn.dataset.view = v.id;
    btn.setAttribute('aria-label', v.title);
    btn.title = v.title;
    btn.innerHTML = `${ui.icon(v.icon || ICON_FALLBACK[v.id] || 'sparkles')}<span class="nav-label">${v.title}</span>`;
    btn.addEventListener('click', () => { location.hash = `#/${v.id}`; });
    items.appendChild(btn);
  }
  el.nav.appendChild(items);

  const foot = document.createElement('div');
  foot.className = 'nav-foot';
  const collapse = document.createElement('button');
  collapse.type = 'button';
  collapse.className = 'nav-item nav-collapse';
  collapse.setAttribute('aria-label', 'Collapse navigation');
  collapse.innerHTML = `${ui.icon('chevron')}<span class="nav-label">Collapse</span>`;
  collapse.addEventListener('click', () => setRail(!el.nav.classList.contains('rail')));
  foot.appendChild(collapse);
  el.nav.appendChild(foot);
}

function setRail(rail) {
  if (rail) {
    const w = el.nav.getBoundingClientRect().width;
    if (w > RAIL_W + 8) localStorage.setItem('chronos.nav.w', String(Math.round(w)));
    el.nav.style.flex = `0 1 ${RAIL_W}px`;
  } else {
    const w = parseInt(localStorage.getItem('chronos.nav.w') || '224', 10);
    el.nav.style.flex = `${w} 1 0px`;
  }
  localStorage.setItem('chronos.nav.rail', rail ? '1' : '0');
}

/* Rail class follows real width, so splitter drags stay coherent. */
new ResizeObserver(() => {
  el.nav.classList.toggle('rail', el.nav.getBoundingClientRect().width < 96);
}).observe(el.nav);

function setActiveNav(id) {
  el.nav.querySelectorAll('.nav-item[data-view]').forEach((b) => {
    b.classList.toggle('active', b.dataset.view === id);
  });
}

/* ================================ ROUTER ================================ */

let current = null;
let currentId = null;

/**
 * Hash router: '#/market' (default). Unmounts the active view, mounts the
 * next into #center with the shared ctx. Unknown routes fall back to market.
 */
async function route() {
  const raw = (location.hash.replace(/^#\/?/, '') || 'market').split('?')[0];
  const view = viewById[raw] || viewById.market;
  if (!viewById[raw] && location.hash && raw !== 'market') {
    history.replaceState(null, '', '#/market');
  }
  if (view === current) return;
  if (current && typeof current.unmount === 'function') {
    try { current.unmount(); } catch (err) { console.error('[shell] unmount failed', err); }
  }
  el.center.innerHTML = '';
  current = view;
  currentId = view.id;
  setActiveNav(view.id);
  el.pop.hidden = !POP_PAGES[view.id];
  document.title = `Chronos Terminal — ${view.title}`;
  try {
    await view.mount(el.center, ctx);
  } catch (err) {
    console.error(`[shell] mount ${view.id} failed`, err);
    el.center.innerHTML = `<div class="empty">${ui.icon('flask')}<div>The ${view.title} view failed to load.<br><span class="dim">${(err && err.message) || err}</span></div></div>`;
  }
}

window.addEventListener('hashchange', route);

/* ============================== RHS PANEL =============================== */

const rhsState = { events: [], stratTimer: null };

function rhsSection(title, actionsHtml = '') {
  const sec = document.createElement('section');
  sec.className = 'rhs-section';
  sec.innerHTML = `<div class="rhs-section-head"><span>${title}</span><span class="actions">${actionsHtml}</span></div><div class="rhs-section-body"></div>`;
  return sec;
}

function buildRhs() {
  const head = document.createElement('div');
  head.className = 'rhs-head';
  head.innerHTML = '<span class="rhs-title">Actions</span>';
  const hide = document.createElement('button');
  hide.type = 'button';
  hide.className = 'btn btn-ghost btn-icon';
  hide.setAttribute('aria-label', 'Collapse side panel');
  hide.innerHTML = ui.icon('chevron');
  hide.addEventListener('click', () => setRhsCollapsed(true));
  head.appendChild(hide);
  el.rhs.appendChild(head);

  const scroll = document.createElement('div');
  scroll.className = 'rhs-scroll';
  el.rhs.appendChild(scroll);

  /* -- primary CTAs -- */
  const cta = document.createElement('div');
  cta.className = 'rhs-cta';
  const newsBtn = document.createElement('button');
  newsBtn.type = 'button';
  newsBtn.className = 'cta-big';
  newsBtn.innerHTML = `<span class="cta-n">1</span><span>Ingest News<span class="cta-sub">score a headline into the market</span></span>${ui.icon('news')}`;
  newsBtn.addEventListener('click', openNewsModal);
  const upBtn = document.createElement('button');
  upBtn.type = 'button';
  upBtn.className = 'cta-big';
  upBtn.innerHTML = `<span class="cta-n">2</span><span>Upload Script<span class="cta-sub">save a strategy to the library</span></span>${ui.icon('upload')}`;
  upBtn.addEventListener('click', openUploadModal);
  cta.appendChild(newsBtn);
  cta.appendChild(upBtn);
  scroll.appendChild(cta);

  /* -- running strategies -- */
  const strat = rhsSection('Running Strategies');
  scroll.appendChild(strat);
  rhsState.stratBody = strat.querySelector('.rhs-section-body');
  rhsState.stratBody.innerHTML = '<span class="dim">none running</span>';

  /* -- event feed -- */
  const feed = rhsSection('Event Feed');
  scroll.appendChild(feed);
  rhsState.feedBody = feed.querySelector('.rhs-section-body');
  rhsState.feedBody.innerHTML = '<span class="dim">waiting for events…</span>';

  /* -- market pulse -- */
  const pulse = rhsSection('Market Pulse', `<button type="button" class="btn btn-ghost btn-icon btn-sm" id="pulse-refresh" aria-label="Refresh narration">${ui.icon('refresh')}</button>`);
  scroll.appendChild(pulse);
  rhsState.pulseBody = pulse.querySelector('.rhs-section-body');
  rhsState.pulseBody.innerHTML = '<div class="pulse-line dim">press ↻ to ask the market why</div>';
  pulse.querySelector('#pulse-refresh').addEventListener('click', refreshPulse);

  el.rhsReopen.addEventListener('click', () => setRhsCollapsed(false));

  pollStrategies();
  rhsState.stratTimer = setInterval(pollStrategies, 3000);
  store.onEvent(onFeedEvent);
}

async function refreshPulse(e) {
  const btn = e.currentTarget;
  try {
    const res = await ui.busy(btn, () => api.narrate());
    rhsState.pulseBody.innerHTML = `<div class="pulse-line">${escapeHtml(res.narration || '—')}</div>`
      + `<span class="chip pulse-src">${res.source || 'engine'}</span>`;
  } catch (err) {
    ui.toast(`Narration failed: ${err.detail || err.message}`, 'bad');
  }
}

async function pollStrategies() {
  let list;
  try { list = await api.strategy.list(); } catch { return; }
  if (!Array.isArray(list)) return;
  if (!list.length) {
    rhsState.stratBody.innerHTML = '<span class="dim">none running</span>';
    return;
  }
  rhsState.stratBody.innerHTML = list.map((s) => `
    <div class="mini-strat">
      <span class="sdot ${s.status === 'running' ? 'running' : 'stopped'}" aria-hidden="true"></span>
      <span class="sname" title="${escapeAttr(s.name)}">${escapeHtml(s.name)}</span>
      <span class="sagent">${escapeHtml(s.agent_id || '')}</span>
    </div>`).join('');
}

const FEED_META = {
  news: (e) => ({ icon: '🔮', cls: (e.score ?? 0) >= 0 ? 'up' : 'down', text: `${e.headline} (${fmt.num(e.score, 2)})` }),
  mode_change: (e) => ({ icon: '⚡', cls: 'reactive', text: `Mode → ${e.mode}${e.trigger_agent ? ` · triggered by ${e.trigger_agent}` : ''}` }),
  liquidation: (e) => ({ icon: '↺', cls: 'down', text: `${e.agent_id} liquidated — ledger reset` }),
  session: (e) => ({ icon: '◷', cls: 'muted', text: `Session ${String(e.action || '').replace('_', ' ')}${e.day_count ? ` · day ${e.day_count}` : ''}` }),
};

function onFeedEvent(evt) {
  const meta = (FEED_META[evt.kind] || FEED_META.session)(evt);
  const t = store.state.tick ? fmt.time(store.state.tick.unix_time) : '';
  rhsState.events.unshift({ ...meta, time: t });
  rhsState.events = rhsState.events.slice(0, 8);
  rhsState.feedBody.innerHTML = rhsState.events.map((it) => `
    <div class="feed-item ${it.cls}">
      <span class="fico" aria-hidden="true">${it.icon}</span>
      <span class="ftxt">${escapeHtml(it.text)}</span>
      <span class="ftime">${it.time}</span>
    </div>`).join('');
}

function openNewsModal() {
  const m = ui.modal({
    title: 'Ingest News',
    body: `
      <label class="field">Headline
        <textarea id="m-headline" class="input" rows="3" placeholder="e.g. TCS wins record $2B cloud deal with EU banking consortium"></textarea>
      </label>
      <p class="hint-line">The oracle scores the headline (Gemini, with deterministic fallback) and the shock trades into the book. The score arrives in the Event Feed.</p>`,
    actions: [
      { label: 'Cancel' },
      {
        label: 'Ingest',
        kind: 'primary',
        onClick: async (close, btn) => {
          const headline = m.body.querySelector('#m-headline').value.trim();
          if (!headline) { ui.toast('Write a headline first', 'warn'); return; }
          try {
            await ui.busy(btn, () => api.news(headline));
            ui.toast('Headline sent — score arrives via the event feed', 'good');
            close();
          } catch (err) { ui.toast(`News failed: ${err.detail || err.message}`, 'bad'); }
        },
      },
    ],
  });
}

function openUploadModal() {
  const m = ui.modal({
    title: 'Upload Script',
    wide: true,
    body: `
      <div class="form-row">
        <label class="field grow">Name<input id="u-name" class="input" placeholder="my_strategy"></label>
        <label class="field">From file<input id="u-file" class="input" type="file" accept=".py"></label>
      </div>
      <label class="field">Python code
        <textarea id="u-code" class="input code" rows="12" spellcheck="false" placeholder="from runner.sdk import Strategy&#10;&#10;class UserStrategy(Strategy):&#10;    def on_tick(self, state): ..."></textarea>
      </label>
      <p class="hint-line">Saved scripts are real files in the library — load, edit and run them any time from Strategies.</p>`,
    actions: [
      { label: 'Cancel' },
      { label: 'Save to Library', onClick: (close, btn) => saveUpload(m, close, btn, false) },
      { label: 'Save & Run', kind: 'primary', onClick: (close, btn) => saveUpload(m, close, btn, true) },
    ],
  });
  const file = m.body.querySelector('#u-file');
  file.addEventListener('change', () => {
    const f = file.files && file.files[0];
    if (!f) return;
    const nameEl = m.body.querySelector('#u-name');
    if (!nameEl.value) nameEl.value = f.name.replace(/\.py$/i, '');
    const reader = new FileReader();
    reader.onload = () => { m.body.querySelector('#u-code').value = String(reader.result || ''); };
    reader.readAsText(f);
  });
}

async function saveUpload(m, close, btn, run) {
  const name = m.body.querySelector('#u-name').value.trim() || 'untitled';
  const code = m.body.querySelector('#u-code').value;
  if (!code.trim()) { ui.toast('The script is empty', 'warn'); return; }
  try {
    const saved = await ui.busy(btn, () => api.scripts.save(name, code));
    ui.toast(`Saved “${saved.name}” to the library`, 'good');
    close();
    if (run) {
      const res = await api.scripts.run(saved.script_id);
      ui.toast(`Running as ${res.agent_id} (${res.backend || 'sandbox'})`, 'good');
    }
  } catch (err) {
    ui.toast(`${run ? 'Save & run' : 'Save'} failed: ${err.detail || err.message}`, 'bad');
  }
}

/* ============================ SHELL SPLITTERS =========================== */

let shellSplit = null;

function setRhsCollapsed(collapsed) {
  if (collapsed) shellSplit.collapse(2);
  else shellSplit.restore(2);
  el.rhsReopen.hidden = !collapsed;
  localStorage.setItem('chronos.rhs.collapsed', collapsed ? '1' : '0');
}

/* ------------------------------- helpers -------------------------------- */

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function escapeAttr(s) { return escapeHtml(s); }

/* ================================= BOOT ================================= */

buildNav();
buildRhs();

shellSplit = splitter.hSplit(el.shell, {
  sizes: [224, 1200, 320],
  min: [RAIL_W, 480, 240],
  collapsible: [2],
  storageKey: 'chronos.split.shell',
});

if (localStorage.getItem('chronos.nav.rail') === '1') setRail(true);
if (localStorage.getItem('chronos.rhs.collapsed') === '1') setRhsCollapsed(true);

route();
