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
import { runstate } from './lib/runstate.js';

import market from './views/market.js';
import strategies from './views/strategies.js';
import results from './views/results.js';
import timemachine from './views/timemachine.js';
import runs from './views/runs.js';
import lab from './views/lab.js';

const ctx = { store, api, fmt, ui, charts, splitter };

/** Fixed nav order (DESIGN.md). Group fallback if a view omits its own. */
// The old "Your Data" page now lives on the main page (views/run_setup.js, the New run panel).
const VIEWS = [market, strategies, results, timemachine, lab, runs];
const GROUP_FALLBACK = {
  market: '', strategies: 'Trading', results: 'Trading',
  timemachine: 'Lab', lab: 'Lab',
  runs: 'Data',
};
const ICON_FALLBACK = {
  market: 'candles', strategies: 'code', results: 'report',
  timemachine: 'clock', lab: 'flask', runs: 'database',
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
  sym: document.getElementById('tb-sym'),
  conn: document.getElementById('tb-conn'),
  pause: document.getElementById('tb-pause'),
  stop: document.getElementById('tb-stop'),
  stopMenu: document.getElementById('tb-stop-menu'),
  pop: document.getElementById('tb-pop'),
  theme: document.getElementById('tb-theme'),
};

/* ============================= LIGHT / DARK ============================= */

const THEME_KEY = 'chronos.theme';
const SUN = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4" fill="none" stroke="currentColor" stroke-width="1.8"/>'
  + '<path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6M5.3 18.7l1.6-1.6M17.1 6.9l1.6-1.6" '
  + 'stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
const MOON = el.theme.innerHTML;
const isDark = () => document.documentElement.dataset.theme === 'dark';
function paintThemeButton() {
  el.theme.innerHTML = isDark() ? SUN : MOON;
  el.theme.title = isDark() ? 'Light mode' : 'Dark mode';
  el.theme.setAttribute('aria-label', el.theme.title);
}
el.theme.addEventListener('click', () => {
  const dark = !isDark();
  if (dark) document.documentElement.dataset.theme = 'dark';
  else delete document.documentElement.dataset.theme;
  try { localStorage.setItem(THEME_KEY, dark ? 'dark' : 'light'); } catch { /* not remembered */ }
  paintThemeButton();
  rerender();          // charts and the code editor read their colours when they are built
});
paintThemeButton();

/* ============================== TOPBAR ================================== */

let lastPrice = null;
let paused = false;
let btMode = null;            // 'free' | 'backtest' (GET /api/mode), for the status label after a hand-over

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
/** '11 Nov 2026' from the tick's unix time (the sim clock is IST written as UTC, so read the UTC parts). */
function dateOf(unix) {
  if (!Number.isFinite(unix)) return '—';
  const d = new Date(unix * 1000);
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

/** Plain-words market status: [label, colour class, tooltip]. */
function statusOf(tick, rs) {
  if (rs.stopped) return ['Stopped', 'off', 'This run is frozen. Use New run to start another.'];
  if (tick.paused) return ['Paused', 'off', 'The market is paused. Press Resume to carry on.'];
  if (tick.mode === 'REPLAY') return ['Real data', 'replay', 'Playing the real market data minute by minute.'];
  if (tick.mode === 'REACTIVE') {
    return btMode === 'backtest'
      ? ['Backtest', 'live', 'The real data keeps playing; your trades move it by their market impact.']
      : ['Simulated traders', 'live', 'After your first trade the simulated traders set the price.'];
  }
  return ['Simulated', 'live', 'A simulated stock: the price comes from the simulated traders.'];
}

async function refreshBtMode() {
  try { btMode = (await api.mode.get()).mode; } catch { btMode = null; }
  renderStatus();
}

function renderStatus() {
  const tick = store.latestTick();
  if (!tick || !tick.run_id) return;
  const [label, cls, tip] = statusOf(tick, runstate.get());
  el.mode.hidden = false;
  el.mode.className = `tb-state ${cls}`;
  el.mode.title = tip;
  el.modeText.textContent = label;
}

function onTick(tick) {
  const live = !!tick.run_id;          // an idle engine (nothing started yet) ticks without a run_id
  for (const e of [el.sym, el.day, el.clock, el.price, el.mode]) e.hidden = !live;
  if (!live) return;
  el.sym.hidden = !tick.symbol;
  el.sym.textContent = tick.symbol || '';
  el.day.hidden = false;
  el.dayN.textContent = dateOf(tick.unix_time);          // the data's own date, not the sim's day counter
  el.clock.hidden = false;
  el.clock.textContent = fmt.time(tick.unix_time);
  el.price.hidden = false;
  el.price.textContent = fmt.inr(tick.last_price);
  if (lastPrice != null && tick.last_price !== lastPrice) {
    el.price.classList.toggle('up', tick.last_price > lastPrice);
    el.price.classList.toggle('down', tick.last_price < lastPrice);
  }
  lastPrice = tick.last_price;
  paused = !!tick.paused;
  el.pause.textContent = paused ? 'Resume' : 'Pause';
  renderStatus();
}

/** Pause/Resume and Stop are shown only while a run is on; a stopped run offers only New run. */
function renderControls(rs) {
  const on = rs.hasRun && rs.view === 'live';
  el.pause.hidden = !on || rs.stopped;
  el.stop.hidden = !on;
  el.stop.textContent = rs.stopped ? 'New run' : 'Stop';
  el.stop.setAttribute('aria-haspopup', rs.stopped ? 'false' : 'menu');
  closeStopMenu();
  renderStatus();
}

function closeStopMenu() {
  el.stopMenu.hidden = true;
  el.stop.setAttribute('aria-expanded', 'false');
}

el.pause.addEventListener('click', async () => {
  try {
    if (paused) await api.sim.resume();
    else await api.sim.stop();
  } catch (err) { ui.toast(`Control failed: ${err.detail || err.message}`, 'bad'); }
});

el.stop.addEventListener('click', (e) => {
  e.stopPropagation();
  if (runstate.get().stopped) { runstate.newRun().catch((err) => ui.toast(`New run failed: ${err.detail || err.message}`, 'bad')); return; }
  const open = el.stopMenu.hidden;
  el.stopMenu.hidden = !open;
  el.stop.setAttribute('aria-expanded', open ? 'true' : 'false');
  if (open) el.stopMenu.querySelector('button').focus();
});
el.stopMenu.addEventListener('click', async (e) => {
  const b = e.target.closest('[data-act]');
  if (!b) return;
  closeStopMenu();
  try {
    if (b.dataset.act === 'stop') await runstate.stop();
    else await runstate.newRun();
  } catch (err) { ui.toast(`Stop failed: ${err.detail || err.message}`, 'bad'); }
});
document.addEventListener('click', (e) => { if (!e.target.closest('.tb-menu-wrap')) closeStopMenu(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeStopMenu(); });

runstate.onChange(renderControls);
store.onEvent((e) => { if (e.kind === 'mode_change') refreshBtMode(); });
let lastRun = null;
store.onTick((t) => { if (t.run_id !== lastRun) { lastRun = t.run_id; refreshBtMode(); } });
renderControls(runstate.get());

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
      g.textContent = group.charAt(0).toUpperCase() + group.slice(1).toLowerCase();   // 'LAB' -> 'Lab'
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
  document.title = `Chronos · ${view.title}`;
  try {
    await view.mount(el.center, ctx);
  } catch (err) {
    console.error(`[shell] mount ${view.id} failed`, err);
    el.center.innerHTML = `<div class="empty">${ui.icon('flask')}<div>The ${view.title} view failed to load.<br><span class="dim">${(err && err.message) || err}</span></div></div>`;
  }
}

window.addEventListener('hashchange', route);

/** Build the current view again (after a theme switch); its data lives in the store, so nothing is lost. */
async function rerender() {
  const view = current;
  if (!view) return;
  try { if (typeof view.unmount === 'function') view.unmount(); } catch (err) { console.error('[shell] unmount failed', err); }
  el.center.innerHTML = '';
  try { await view.mount(el.center, ctx); } catch (err) { console.error(`[shell] mount ${view.id} failed`, err); }
}

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

  /* -- running strategies (Stop each, or Stop all) -- */
  const strat = rhsSection('Running Strategies', '<button type="button" class="rhs-stopall" data-stopall hidden>Stop all</button>');
  scroll.appendChild(strat);
  rhsState.stratBody = strat.querySelector('.rhs-section-body');
  rhsState.stratBody.innerHTML = '<span class="dim">none running</span>';
  rhsState.stopAll = strat.querySelector('[data-stopall]');
  rhsState.stopAll.addEventListener('click', async () => {
    let list = [];
    try { list = await api.strategy.list(); } catch { /* nothing to stop */ }
    const running = list.filter((s) => s.status === 'running');
    rhsState.stopAll.disabled = true;
    try {
      await Promise.all(running.map((s) => api.strategy.stop(s.strategy_id)));
      ui.toast(`Stopped ${running.length} strateg${running.length === 1 ? 'y' : 'ies'}`, 'good');
    } catch (err) { ui.toast(`Stop failed: ${err.detail || err.message}`, 'bad'); }
    rhsState.stopAll.disabled = false;
    pollStrategies();
  });
  rhsState.stratBody.addEventListener('click', async (e) => {
    const b = e.target.closest('[data-stop]');
    if (!b) return;
    b.disabled = true;
    try {
      await api.strategy.stop(b.dataset.stop);
      ui.toast('Strategy stopped', 'good');
    } catch (err) { ui.toast(`Stop failed: ${err.detail || err.message}`, 'bad'); b.disabled = false; }
    pollStrategies();
  });

  /* -- event feed -- */
  const feed = rhsSection('Event Feed');
  scroll.appendChild(feed);
  rhsState.feedBody = feed.querySelector('.rhs-section-body');
  rhsState.feedBody.innerHTML = '<span class="dim">waiting for events…</span>';

  el.rhsReopen.addEventListener('click', () => setRhsCollapsed(false));

  pollStrategies();
  rhsState.stratTimer = setInterval(pollStrategies, 3000);
  store.onEvent(onFeedEvent);
}

async function pollStrategies() {
  let list;
  try { list = await api.strategy.list(); } catch { return; }
  if (!Array.isArray(list)) return;
  // only what is running (the stopped ones are on the Strategies page)
  const running = list.filter((s) => s.status === 'running');
  rhsState.stopAll.hidden = running.length < 2;
  if (!running.length) {
    rhsState.stratBody.innerHTML = '<span class="dim">none running</span>';
    return;
  }
  rhsState.stratBody.innerHTML = running.map((s) => `
    <div class="mini-strat">
      <span class="sdot running" aria-hidden="true"></span>
      <span class="sname" title="${escapeAttr(s.name)}">${escapeHtml(s.name)}</span>
      <span class="sagent">${escapeHtml(s.agent_id || '')}</span>
      <button type="button" class="rhs-stop" data-stop="${escapeAttr(s.strategy_id)}" title="Stop this strategy">Stop</button>
    </div>`).join('');
}

const FEED_META = {
  news: (e) => ({ icon: '🔮', cls: (e.score ?? 0) >= 0 ? 'up' : 'down', text: `${e.headline} (${fmt.num(e.score, 2)})` }),
  mode_change: (e) => ({ icon: '↳', cls: 'reactive', text: e.trigger_agent === 'HANDOVER'
    ? 'Real data handed over to the simulated traders'
    : `First trade by ${e.trigger_agent}: the market now reacts to your orders` }),
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
