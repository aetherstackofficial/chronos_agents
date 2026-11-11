/* ============================================================================
   views/run_setup.js — the "New run" panel on the right of the main page.

   It replaces the old "Your Data" page and the topbar START modal: choose the data
   (a stock from Yahoo Finance, your own CSV, or a simulated stock), the market mode,
   and press Start market. One button does it all:
     Yahoo/CSV: POST /api/replay/yahoo | /api/replay/upload -> /api/replay/start
                -> /api/mode (mode, when the data runs out) -> /api/sim/resume
     Simulated: POST /api/sim/start -> /api/sim/resume
   The page turns live by itself when the new run's first tick arrives (lib/runstate.js).
   ============================================================================ */

import { runstate } from '../lib/runstate.js';
import { SDK_TEMPLATE } from './strategies.js';

const STYLE_ID = 'rs-style';
const CSS = `
.rs{display:flex;flex-direction:column;height:100%;min-height:0;background:var(--bg-1)}
.rs-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;padding:14px 16px 10px;border-bottom:1px solid var(--border)}
.rs-head h3{font-size:15px;font-weight:600;color:var(--text-hi);margin:0}
.rs-head p{margin:2px 0 0;color:var(--text-mut);font-size:12px}
.rs-back{all:unset;cursor:pointer;color:var(--accent);font-size:12px;white-space:nowrap;padding-top:2px}
.rs-back:hover{text-decoration:underline}
.rs-body{flex:1 1 auto;min-height:0;overflow:auto;padding:4px 16px 16px;display:flex;flex-direction:column}
.rs-sec{padding:14px 0;border-bottom:1px solid var(--border);display:flex;flex-direction:column;gap:10px}
.rs-sec:last-child{border-bottom:none}
.rs-label{font-size:12px;font-weight:600;color:var(--text-hi)}
.rs-tabs{display:flex;background:var(--bg-2);border-radius:4px;padding:2px;gap:2px}
.rs-tabs button{all:unset;flex:1;text-align:center;cursor:pointer;padding:5px 6px;border-radius:3px;font-size:12px;color:var(--text-mut)}
.rs-tabs button:hover{color:var(--text-hi)}
.rs-tabs button[aria-selected="true"]{background:var(--bg-3);color:var(--text-hi)}
.rs-tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.rs-row{display:flex;gap:8px}
.rs-row>label{flex:1;min-width:0}
.rs label.field{gap:4px;font-size:12px}
.rs .input,.rs .select{padding:6px 8px;font-size:13px;border-radius:4px;background:var(--bg-0)}
.rs-hint{margin:0;color:var(--text-dim);font-size:11.5px;line-height:1.45}
.rs-opt{display:flex;gap:10px;align-items:flex-start;padding:9px 10px;border:1px solid var(--border);border-radius:4px;cursor:pointer}
.rs-opt:hover{border-color:var(--border-strong)}
.rs-opt:has(input:checked){border-color:var(--accent);background:var(--accent-soft)}
.rs-opt input{margin:2px 0 0;accent-color:var(--accent)}
.rs-opt b{display:block;font-size:13px;font-weight:600;color:var(--text-hi)}
.rs-opt span{display:block;font-size:12px;color:var(--text-mut);line-height:1.4}
.rs details summary{cursor:pointer;font-size:12px;color:var(--text-mut)}
.rs details[open] summary{margin-bottom:8px}
.rs-foot{flex:0 0 auto;border-top:1px solid var(--border);padding:12px 16px 14px;display:flex;flex-direction:column;gap:8px;background:var(--bg-1)}
.rs-start{width:100%;padding:10px 14px;font-size:14px}
.rs-msg{font-size:12px;color:var(--text-mut);white-space:pre-wrap;min-height:0}
.rs-msg.err{color:var(--down)}
.rs [hidden]{display:none!important}
.rs-label-row{display:flex;align-items:center;justify-content:space-between;gap:8px}
.rs-link{all:unset;cursor:pointer;font-size:12px;color:var(--text-hi);padding:2px 6px;border-radius:3px}
.rs-link:hover{background:var(--bg-3)}
.rs-link:focus-visible{outline:2px solid var(--accent)}
.rs-danger{color:var(--down)}
.rs-strat{display:flex;flex-direction:column;gap:8px}
.rs-src{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.rs-src>button,.rs-file{all:unset;cursor:pointer;font-size:12px;padding:5px 8px;border:1px solid var(--border-strong);
  border-radius:4px;color:var(--text)}
.rs-src>button:hover,.rs-file:hover{background:var(--bg-3)}
.rs-src .select{width:auto;flex:1;min-width:120px;padding:5px 8px;font-size:12px}
.rs-code{width:100%;min-height:220px;resize:vertical;font:12px/1.5 var(--mono);color:var(--text-hi);background:var(--bg-2);
  border:1px solid var(--border-strong);border-radius:4px;padding:8px;tab-size:4;white-space:pre;overflow:auto}
.rs-code:focus{outline:none;border-color:var(--accent)}
.rs-run{display:flex;align-items:center;gap:8px;font-size:12px;padding:4px 0}
.rs-run .nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text-hi)}
.rs-run .id{color:var(--text-mut);font-size:11px}
`;

/** Stocks offered (every symbol checked against Yahoo's 1-minute data; .NS = NSE). Tata Motors is absent on
 *  purpose: it was split into new tickers and TATAMOTORS.NS returns nothing. */
const STOCKS = {
  TECH: [['TCS.NS', 'TCS'], ['INFY.NS', 'Infosys'], ['WIPRO.NS', 'Wipro'], ['HCLTECH.NS', 'HCL Technologies']],
  BANK: [['HDFCBANK.NS', 'HDFC Bank'], ['ICICIBANK.NS', 'ICICI Bank'], ['SBIN.NS', 'State Bank of India'],
         ['KOTAKBANK.NS', 'Kotak Mahindra Bank'], ['AXISBANK.NS', 'Axis Bank']],
  OIL: [['RELIANCE.NS', 'Reliance Industries'], ['ONGC.NS', 'ONGC'], ['BPCL.NS', 'BPCL']],
  PHARMA: [['SUNPHARMA.NS', 'Sun Pharma'], ['DRREDDY.NS', "Dr. Reddy's"], ['CIPLA.NS', 'Cipla']],
  DEFENSE: [['HAL.NS', 'Hindustan Aeronautics'], ['BEL.NS', 'Bharat Electronics']],
};
const SECTORS = Object.keys(STOCKS);
const OTHER = '__other__';
const MAX_DAYS = { '1m': 29, '15m': 59 };       // Yahoo keeps 1m bars ~30 days, 15m ~60 (the server checks too)
const PREFS_KEY = 'chronos.runsetup';

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function loadPrefs() { try { return JSON.parse(localStorage.getItem(PREFS_KEY) || 'null') || {}; } catch { return {}; } }
function savePrefs(p) { try { localStorage.setItem(PREFS_KEY, JSON.stringify(p)); } catch { /* not remembered */ } }

/**
 * Render the panel into `el`.
 * @param {HTMLElement} el
 * @param {object} ctx  the shell's {api, ui, ...}
 * @returns {{destroy():void, refresh():void}}
 */
export function mountRunSetup(el, ctx) {
  if (!document.getElementById(STYLE_ID)) {
    const st = document.createElement('style');
    st.id = STYLE_ID;
    st.textContent = CSS;
    document.head.appendChild(st);
  }
  const { api, ui } = ctx;
  const p = { tab: 'yahoo', stock: 'RELIANCE.NS', interval: '1m', days: 5, mode: 'free', end: 'continue',
              seed: 42, simSymbol: 'TCS', simSector: 'TECH', simPrice: 190, csvSymbol: 'CUSTOM',
              strat: false, stratName: 'big_block', stratCode: '', ...loadPrefs() };

  const stockOptions = SECTORS.map((sec) => `<optgroup label="${sec}">${STOCKS[sec].map(([sym, name]) =>
    `<option value="${sym}" data-sector="${sec}">${esc(name)}</option>`).join('')}</optgroup>`).join('');
  const sectorOptions = SECTORS.map((s) => `<option value="${s}">${s}</option>`).join('');

  el.innerHTML = `
<div class="rs">
  <div class="rs-head">
    <div><h3>New run</h3><p>Pick the data and the mode, then start the market.</p></div>
    <button type="button" class="rs-back" data-r="back" hidden>Back to market</button>
  </div>
  <div class="rs-body">
    <div class="rs-sec">
      <div class="rs-label">Data</div>
      <div class="rs-tabs" role="tablist" aria-label="Data source">
        <button type="button" role="tab" data-tab="yahoo">Stock</button>
        <button type="button" role="tab" data-tab="csv">Your CSV</button>
        <button type="button" role="tab" data-tab="sim">Simulated</button>
      </div>

      <div data-pane="yahoo">
        <div class="rs-row"><label class="field">Stock
          <select class="select" data-r="stock">${stockOptions}<option value="${OTHER}">Other (Yahoo symbol)…</option></select></label></div>
        <div class="rs-row" data-r="custom-row" hidden>
          <label class="field">Yahoo symbol<input class="input" data-r="custom" placeholder="e.g. ITC.NS" maxlength="20"></label>
          <label class="field">Sector<select class="select" data-r="csector">${sectorOptions}</select></label>
        </div>
        <div class="rs-row" style="margin-top:8px">
          <label class="field">Candles<select class="select" data-r="interval">
            <option value="1m">1 minute</option><option value="15m">15 minutes</option></select></label>
          <label class="field">Days of history<input class="input" data-r="days" type="number" min="1" max="29"></label>
        </div>
        <p class="rs-hint" style="margin-top:8px">Downloaded from Yahoo Finance. 1-minute data goes back about 30 days,
          15-minute about 60. Each candle plays as one market minute.</p>
      </div>

      <div data-pane="csv" hidden>
        <div class="rs-row"><label class="field">CSV file<input class="input" data-r="file" type="file" accept=".csv,text/csv"></label></div>
        <div class="rs-row" style="margin-top:8px"><label class="field">Symbol<input class="input" data-r="csvsym" maxlength="16"></label></div>
        <p class="rs-hint" style="margin-top:8px">1-minute candles: open, high, low, close, volume and a date + time
          (or datetime) column. Up to 10 MB; a bad file is rejected with the line and the problem.</p>
      </div>

      <div data-pane="sim" hidden>
        <div class="rs-row">
          <label class="field">Symbol<input class="input" data-r="simsym" maxlength="12"></label>
          <label class="field">Sector<select class="select" data-r="simsec">${sectorOptions}</select></label>
        </div>
        <div class="rs-row" style="margin-top:8px"><label class="field">Start price (₹)<input class="input" data-r="simprice" type="number" step="0.05" min="0.05"></label></div>
        <p class="rs-hint" style="margin-top:8px">No real data: the price comes from the simulated traders alone.</p>
      </div>
    </div>

    <div class="rs-sec" data-r="mode-sec">
      <div class="rs-label">Mode</div>
      <label class="rs-opt"><input type="radio" name="rs-mode" value="free">
        <span><b>Free market</b><span>The real data plays until your first trade. From then on the simulated traders set the price.</span></span></label>
      <label class="rs-opt"><input type="radio" name="rs-mode" value="backtest">
        <span><b>Backtest</b><span>The real data keeps playing; your orders move it by their market impact.</span></span></label>
      <label class="field">When the data runs out<select class="select" data-r="end">
        <option value="continue">Keep trading with the simulated traders</option>
        <option value="pause">Pause the market</option></select></label>
    </div>

    <div class="rs-sec" data-r="strat-sec">
      <div class="rs-label-row"><span class="rs-label">Strategy</span>
        <button type="button" class="rs-link" data-r="add-strat">+ Add strategy</button></div>
      <p class="rs-hint" data-r="strat-hint">Optional. A strategy added here starts together with the market.</p>
      <div class="rs-strat" data-r="strat" hidden>
        <div class="rs-row"><label class="field">Name<input class="input" data-r="sname" maxlength="48"></label></div>
        <div class="rs-src">
          <button type="button" data-r="demo" title="The demo: one big order sent in slices (Big Block Trader)">Demo strategy</button>
          <select class="select" data-r="lib" aria-label="Load a strategy from the library"><option value="">From library…</option></select>
          <label class="rs-file">Upload .py<input type="file" accept=".py,text/x-python,text/plain" data-r="sfile" hidden></label>
        </div>
        <textarea class="rs-code" data-r="code" spellcheck="false" aria-label="Strategy code (Python)"></textarea>
        <div class="rs-label-row"><span class="rs-hint">Python with <span style="font-family:var(--mono)">runner.sdk</span>. It is checked before the market starts.</span>
          <button type="button" class="rs-link rs-danger" data-r="rm-strat">Remove</button></div>
      </div>
    </div>

    <div class="rs-sec" data-r="running-sec" hidden>
      <div class="rs-label-row"><span class="rs-label">Strategies running now</span>
        <button type="button" class="rs-link rs-danger" data-r="stop-all">Stop all</button></div>
      <p class="rs-hint">They keep trading in the next run too. Stop the ones you do not want in it.</p>
      <div data-r="running"></div>
    </div>

    <details class="rs-sec"><summary>Advanced</summary>
      <div class="rs-row"><label class="field">Random seed<input class="input" data-r="seed" type="number"></label></div>
      <p class="rs-hint">Same seed and same data give the same simulated traders' behaviour.</p>
    </details>
  </div>
  <div class="rs-foot">
    <div class="rs-msg" data-r="msg" role="status" aria-live="polite" hidden></div>
    <button type="button" class="btn btn-primary rs-start" data-r="start">Start market</button>
  </div>
</div>`;

  const $ = (r) => el.querySelector(`[data-r="${r}"]`);
  const msg = $('msg');
  const startBtn = $('start');
  const say = (text, err = false) => { msg.textContent = text; msg.classList.toggle('err', err); msg.hidden = !text; };

  /* ---- fill in the remembered choices ---- */
  const stockSel = $('stock');
  if ([...stockSel.options].some((o) => o.value === p.stock)) stockSel.value = p.stock;
  else { stockSel.value = OTHER; $('custom').value = p.stock || ''; }
  $('interval').value = p.interval;
  $('days').value = p.days;
  $('end').value = p.end;
  $('seed').value = p.seed;
  $('simsym').value = p.simSymbol;
  $('simsec').value = p.simSector;
  $('simprice').value = p.simPrice;
  $('csvsym').value = p.csvSymbol;
  el.querySelector(`input[name="rs-mode"][value="${p.mode}"]`).checked = true;

  /* ---- tabs ---- */
  const setTab = (tab) => {
    p.tab = tab;
    el.querySelectorAll('[data-tab]').forEach((b) => b.setAttribute('aria-selected', b.dataset.tab === tab ? 'true' : 'false'));
    el.querySelectorAll('[data-pane]').forEach((d) => { d.hidden = d.dataset.pane !== tab; });
    $('mode-sec').hidden = tab === 'sim';             // no real data, so no replay mode
    sync();
  };
  el.querySelector('.rs-tabs').addEventListener('click', (e) => {
    const b = e.target.closest('[data-tab]');
    if (b) setTab(b.dataset.tab);
  });

  const isOther = () => stockSel.value === OTHER;
  const symbol = () => (isOther() ? $('custom').value : stockSel.value).trim().toUpperCase();
  const sector = () => (isOther() ? $('csector').value : stockSel.selectedOptions[0].dataset.sector);
  const ready = () => (p.tab === 'yahoo' ? symbol().length > 0
    : p.tab === 'csv' ? $('file').files.length > 0 : $('simsym').value.trim().length > 0);
  function sync() {
    $('custom-row').hidden = !isOther();
    const max = MAX_DAYS[$('interval').value];
    $('days').max = String(max);
    if (Number($('days').value) > max) $('days').value = String(max);
    startBtn.disabled = busy || !ready();
  }
  ['stock', 'custom', 'interval', 'file', 'simsym'].forEach((r) => {
    $(r).addEventListener('input', sync);
    $(r).addEventListener('change', () => { sync(); say(''); });
  });

  /* ---- strategy that starts with the market ---- */
  const stratBox = $('strat');
  const codeEl = $('code');
  const nameEl = $('sname');
  const showStrat = (on) => {
    p.strat = on;
    stratBox.hidden = !on;
    $('add-strat').hidden = on;
    $('strat-hint').hidden = on;
    if (on && !codeEl.value.trim()) { codeEl.value = SDK_TEMPLATE; nameEl.value = nameEl.value || 'big_block'; }
  };
  codeEl.value = p.stratCode || '';
  nameEl.value = p.stratName || 'big_block';
  $('add-strat').addEventListener('click', () => { showStrat(true); codeEl.focus(); });
  $('rm-strat').addEventListener('click', () => showStrat(false));
  $('demo').addEventListener('click', () => { codeEl.value = SDK_TEMPLATE; nameEl.value = 'big_block'; say(''); });
  codeEl.addEventListener('keydown', (e) => {                   // Tab indents (4 spaces) instead of leaving the box
    if (e.key !== 'Tab' || e.shiftKey) return;
    e.preventDefault();
    const a = codeEl.selectionStart;
    codeEl.setRangeText('    ', a, codeEl.selectionEnd, 'end');
  });
  $('sfile').addEventListener('change', () => {
    const f = $('sfile').files[0];
    if (!f) return;
    const rd = new FileReader();
    rd.onload = () => { codeEl.value = String(rd.result || ''); nameEl.value = f.name.replace(/\.py$/i, ''); say(''); };
    rd.readAsText(f);
    $('sfile').value = '';
  });
  const libSel = $('lib');
  api.scripts.list().then((list) => {
    for (const sc of list || []) {
      const o = document.createElement('option');
      o.value = sc.script_id;
      o.textContent = sc.name;
      libSel.appendChild(o);
    }
  }).catch(() => { libSel.hidden = true; });
  libSel.addEventListener('change', async () => {
    const id = libSel.value;
    if (!id) return;
    try {
      const sc = await api.scripts.get(id);
      codeEl.value = sc.code || '';
      nameEl.value = sc.name || nameEl.value;
      say('');
    } catch (err) { say(`Could not load it: ${err.detail || err.message}`, true); }
    libSel.value = '';
  });
  showStrat(!!p.strat);

  /* ---- strategies already running (they carry on into the next run) ---- */
  const runHost = $('running');
  const renderRunning = async () => {
    let list = [];
    try { list = (await api.strategy.list()).filter((x) => x.status === 'running'); } catch { /* none shown */ }
    $('running-sec').hidden = !list.length;
    $('stop-all').hidden = list.length < 2;
    runHost.innerHTML = list.map((x) => `<div class="rs-run"><span class="nm">${esc(x.name)} <span class="id">${esc(x.agent_id)}</span></span>
      <button type="button" class="rs-link rs-danger" data-stop="${esc(x.strategy_id)}">Stop</button></div>`).join('');
  };
  runHost.addEventListener('click', async (e) => {
    const b = e.target.closest('[data-stop]');
    if (!b) return;
    b.disabled = true;
    try { await api.strategy.stop(b.dataset.stop); } catch (err) { ui.toast(`Stop failed: ${err.detail || err.message}`, 'bad'); }
    renderRunning();
  });
  $('stop-all').addEventListener('click', async () => {
    let list = [];
    try { list = (await api.strategy.list()).filter((x) => x.status === 'running'); } catch { /* nothing */ }
    try { await Promise.all(list.map((x) => api.strategy.stop(x.strategy_id))); } catch (err) {
      ui.toast(`Stop failed: ${err.detail || err.message}`, 'bad');
    }
    renderRunning();
  });
  renderRunning();
  const runTimer = setInterval(() => { if (el.offsetParent !== null) renderRunning(); }, 4000);

  /* ---- Back to market (only after "New run" froze a run) ---- */
  const back = $('back');
  back.addEventListener('click', () => runstate.backToMarket());
  const refresh = () => { back.hidden = !runstate.get().hasRun; renderRunning(); };

  /* ---- Start market ---- */
  let busy = false;
  const remember = () => {
    const mode = el.querySelector('input[name="rs-mode"]:checked').value;
    Object.assign(p, {
      stock: symbol(), interval: $('interval').value, days: parseInt($('days').value, 10) || 5,
      mode, end: $('end').value, seed: parseInt($('seed').value, 10) || 42,
      simSymbol: $('simsym').value.trim().toUpperCase() || 'TCS', simSector: $('simsec').value,
      simPrice: parseFloat($('simprice').value) || 190, csvSymbol: $('csvsym').value.trim() || 'CUSTOM',
      stratName: nameEl.value.trim() || 'my_strategy', stratCode: codeEl.value,
    });
    savePrefs(p);
  };

  startBtn.addEventListener('click', async () => {
    if (busy || !ready()) return;
    remember();
    busy = true;
    sync();
    const label = startBtn.textContent;
    const step = (t) => { startBtn.textContent = t; };
    let stage = 'Start';
    const withStrat = p.strat && codeEl.value.trim();
    try {
      if (p.strat && !withStrat) throw new Error('the strategy box is empty (add code, or Remove it)');
      if (withStrat) {
        stage = 'Strategy check';
        step('Checking the strategy…');
        const chk = await api.strategy.check(codeEl.value);
        if (!chk.ok) {
          const errs = (chk.errors || []).slice(0, 3)
            .map((x) => `${x.line ? `line ${x.line}: ` : ''}${x.message}`).join('\n');
          throw new Error(`the strategy has a problem, the market was not started:\n${errs}`);
        }
      }
      if (p.tab === 'sim') {
        stage = 'Start';
        step('Starting…');
        await api.sim.start({ symbol: p.simSymbol, sector: p.simSector, price: p.simPrice, seed: p.seed });
      } else {
        let saved;
        if (p.tab === 'yahoo') {
          stage = 'Download';
          step('Downloading from Yahoo…');
          saved = await api.replayYahoo({ symbol: p.stock, interval: p.interval, days: p.days });
        } else {
          stage = 'Upload';
          step('Checking the file…');
          saved = await api.replayUpload($('file').files[0]);
        }
        stage = 'Replay';
        step('Loading the data…');
        await api.replay({
          csv_path: saved.csv_path, seed: p.seed,
          ...(p.tab === 'yahoo' ? { symbol: p.stock, sector: sector() } : { symbol: p.csvSymbol }),
        });
        stage = 'Mode';
        try {
          await api.mode.set({ mode: p.mode, on_tape_end: p.end });
        } catch (err) {
          if (err.status !== 501) throw err;      // an engine without mode switches still replays
        }
        say(p.tab === 'yahoo'
          ? `${saved.bars.toLocaleString()} candles of ${saved.symbol}, ${saved.first_bar} to ${saved.last_bar}.`
          : `${saved.bars.toLocaleString()} candles from ${saved.filename}.`);
      }
      if (withStrat) {
        stage = 'Strategy';
        step('Starting the strategy…');
        const r = await api.strategy.run(p.stratName, codeEl.value);
        ui.toast(`Strategy ${p.stratName} running as ${r.agent_id}`, 'good');
      }
      stage = 'Start';
      await api.sim.resume();
      ui.toast('Market started', 'good');
    } catch (err) {
      say(`${stage} failed: ${(err && (err.detail || err.message)) || err}`, true);
    } finally {
      busy = false;
      startBtn.textContent = label;
      sync();
    }
  });

  setTab(p.tab);
  refresh();
  const unsub = runstate.onChange(refresh);
  return {
    refresh,
    destroy() { unsub(); clearInterval(runTimer); el.innerHTML = ''; },
  };
}

export default mountRunSetup;
