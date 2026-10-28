// ============================================================================
// CHRONOS — "Your Data" view: replay REAL market data through the engine.
// Two sources, one outcome (a CSV saved on the server, replayed from that file):
//   1. Pick a stock   -> POST /api/replay/yahoo  (Yahoo Finance bars)
//   2. Upload a CSV   -> POST /api/replay/upload (user's own 1-minute bars)
// then POST /api/replay/start with the returned csv_path.
// Via ctx.api.replayYahoo(cfg) / replayUpload(file) / replay(cfg).
// History plays back exactly until one of the user's strategies trades; then
// the engine latches to REACTIVE (same as the Time Machine).
// ============================================================================

const STYLE_ID = 'rd-view-style';
const CSS = `
.rd-root{height:100%;min-height:0;overflow:auto;padding:var(--s-4);display:flex;flex-direction:column;gap:var(--s-4)}
.rd-head{display:flex;align-items:flex-end;justify-content:space-between;gap:var(--s-4);flex-wrap:wrap}
.rd-head h2{display:flex;align-items:center;gap:var(--s-2);font-size:var(--fs-lg)}
.rd-head h2 svg{width:18px;height:18px;color:var(--accent-2)}
.rd-head p{margin:4px 0 0;color:var(--text-mut);max-width:70ch}
.rd-seed{width:110px}
.rd-card{display:flex;flex-direction:column;gap:var(--s-3)}
.rd-card h3{display:flex;align-items:center;gap:var(--s-2);font-size:var(--fs-md)}
.rd-card h3 svg{width:16px;height:16px;color:var(--accent-2)}
.rd-card p{margin:0;color:var(--text-mut);max-width:75ch}
.rd-form{display:flex;flex-wrap:wrap;align-items:flex-end;gap:var(--s-3)}
.rd-form label.field{min-width:120px}
.rd-form label.rd-stock{width:230px}
.rd-form label.rd-wide{width:280px}
.rd-form label.rd-num{width:100px}
.rd-form label[hidden]{display:none}
.rd-note{color:var(--text-dim);font-size:var(--fs-xs)}
.rd-msg{font-size:var(--fs-sm);white-space:pre-wrap}
.rd-msg.err{color:var(--bad)}
.rd-msg.ok{color:var(--good)}
.rd-banner{border-left:3px solid var(--reactive);display:flex;flex-direction:column;gap:var(--s-1)}
.rd-banner strong{color:var(--text-hi)}
.rd-banner p{margin:0;color:var(--text-mut)}
/* display:flex above would otherwise beat the hidden attribute */
.rd-banner[hidden],.rd-msg[hidden]{display:none}
`;

/**
 * Stocks offered in the picker, grouped by the sector the engine understands
 * (TECH / BANK / OIL / PHARMA / DEFENSE — used by the news oracle). Every
 * symbol here was checked against Yahoo's 1-minute data. Indian stocks need
 * the .NS (NSE) suffix. Tata Motors is absent on purpose: it was split into
 * new tickers and TATAMOTORS.NS returns nothing.
 */
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
// Yahoo limits (server enforces the same): 1m ≈ last 29 days, 15m ≈ last 59.
const MAX_DAYS = { '1m': 29, '15m': 59 };

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

function ic(ctx, name) {
  try {
    const i = ctx.ui.icon(name);
    return typeof i === 'string' ? i : ((i && i.outerHTML) || '');
  } catch (e) { return ''; }
}

let state = null;

export default {
  id: 'replaydata', title: 'Your Data', icon: 'upload', group: 'Replay',

  mount(el, ctx) {
    if (!document.getElementById(STYLE_ID)) {
      const st = document.createElement('style');
      st.id = STYLE_ID; st.textContent = CSS;
      document.head.appendChild(st);
    }
    state = { el, ctx };

    const stockOptions = SECTORS.map((sec) => `
        <optgroup label="${sec}">${STOCKS[sec].map(([sym, name]) =>
          `<option value="${sym}" data-sector="${sec}">${esc(name)} (${sym})</option>`).join('')}</optgroup>`).join('');

    el.innerHTML = `
<div class="rd-root">
  <div class="rd-head">
    <div>
      <h2>${ic(ctx, 'upload')} Your Data</h2>
      <p>Replay real market data through the matching engine: pick a stock and bring its prices
         from Yahoo Finance, or upload your own CSV. History plays back exactly until one of your
         strategies trades — then the market becomes reactive and the future is yours to change.</p>
    </div>
    <label class="field rd-seed">Seed
      <input class="input mono" id="rd-seed" type="number" value="42" aria-label="Replay seed">
    </label>
  </div>

  <div class="card rd-banner" id="rd-banner" hidden>
    <strong>Replay started.</strong>
    <p>Watch it on the <a href="#/market">Market</a> page. The moment one of your strategies'
       orders fills, the engine latches to <span class="chip reactive">REACTIVE</span> permanently
       and history stops. Go to <a href="#/strategies">Strategies</a> to trade.</p>
  </div>

  <!-- Source 1: pick a stock, bring its data from Yahoo Finance. -->
  <section class="card rd-card" aria-labelledby="rd-y-title">
    <h3 id="rd-y-title">${ic(ctx, 'candles')} Pick a stock (Yahoo Finance)</h3>
    <p>Choose a stock and how much history you want. The server downloads the 1-minute or
       15-minute bars from Yahoo Finance, saves them as a CSV, and replays from that file.</p>
    <div class="rd-form">
      <label class="field rd-stock">Stock
        <select class="select" id="rd-stock" aria-label="Stock">
          ${stockOptions}
          <option value="${OTHER}">Other… (type a Yahoo symbol)</option>
        </select>
      </label>
      <label class="field rd-wide" id="rd-custom-wrap" hidden>Yahoo symbol
        <input class="input mono" id="rd-custom" type="text" placeholder="e.g. ITC.NS" maxlength="20" aria-label="Yahoo symbol">
      </label>
      <label class="field" id="rd-sector-wrap" hidden>Sector
        <select class="select" id="rd-sector" aria-label="Sector">
          ${SECTORS.map((s) => `<option value="${s}">${s}</option>`).join('')}
        </select>
      </label>
      <label class="field">Interval
        <select class="select" id="rd-int" aria-label="Bar interval">
          <option value="1m">1 minute</option>
          <option value="15m">15 minutes</option>
        </select>
      </label>
      <label class="field rd-num">Calendar days back
        <input class="input mono" id="rd-days" type="number" value="5" min="1" max="29" aria-label="Calendar days of data">
      </label>
      <button class="btn btn-primary" id="rd-yahoo-btn">Download &amp; Replay</button>
    </div>
    <div class="rd-note">Indian stocks need <span class="mono">.NS</span> (NSE) or <span class="mono">.BO</span> (BSE).
      Yahoo keeps 1-minute bars for about 30 days and 15-minute bars for about 60 days; 15-minute bars
      replay one per simulated minute. Yahoo is unofficial and may rate-limit — if so, try again in a few minutes.</div>
  </section>

  <!-- Source 2: the user's own CSV. -->
  <section class="card rd-card" aria-labelledby="rd-u-title">
    <h3 id="rd-u-title">${ic(ctx, 'upload')} Upload your own CSV</h3>
    <p>A 1-minute bar file with <span class="mono">open, high, low, close, volume</span> plus
       <span class="mono">date</span> and <span class="mono">time</span> columns (or one
       <span class="mono">datetime</span> column). Up to 10 MB. Bad files are rejected with the exact
       line and problem.</p>
    <div class="rd-form">
      <label class="field rd-wide">CSV file
        <input class="input" id="rd-file" type="file" accept=".csv,text/csv" aria-label="CSV file to replay">
      </label>
      <label class="field">Symbol
        <input class="input mono" id="rd-symbol" type="text" value="CUSTOM" maxlength="16" aria-label="Symbol for this replay">
      </label>
      <button class="btn btn-primary" id="rd-upload-btn" disabled>Upload &amp; Replay</button>
    </div>
  </section>

  <div class="rd-msg" id="rd-msg" role="status" aria-live="polite" hidden></div>
</div>`;

    const $ = (sel) => el.querySelector(sel);
    const msg = $('#rd-msg');
    const spin = (txt) => `<span class="spinner" aria-hidden="true"></span> ${txt}`;
    const show = (text, kind) => { msg.textContent = text; msg.className = `rd-msg ${kind}`; msg.hidden = false; };

    /**
     * Shared flow for both buttons: step 1 `getCsv()` saves a CSV on the server
     * and returns its path; step 2 replays that file. Handles button state and
     * shows the server's own error text under the cards (not only a toast).
     * @param {HTMLButtonElement} btn
     * @param {string} stepName  step-1 name for error text ("Download"/"Upload")
     * @param {string} busyText  button text during step 1
     * @param {() => Promise<object>} getCsv  resolves to the saved-file info
     * @param {{symbol?:string, sector?:string}} opts  replay symbol / sector
     * @param {(saved:object)=>string} describe  success sentence for the message
     * @param {() => boolean} canRun  whether to re-enable the button afterwards
     */
    const fetchThenReplay = async (btn, stepName, busyText, getCsv, opts, describe, canRun) => {
      const label = btn.innerHTML;
      btn.disabled = true;
      msg.hidden = true;
      let step = stepName;
      try {
        btn.innerHTML = spin(busyText);
        const saved = await getCsv();
        step = 'Replay';
        btn.innerHTML = spin('Starting replay…');
        const seed = parseInt($('#rd-seed').value, 10);
        const res = await ctx.api.replay({
          csv_path: saved.csv_path,
          seed: Number.isFinite(seed) ? seed : 42,
          ...(opts.symbol ? { symbol: opts.symbol } : {}),
          ...(opts.sector ? { sector: opts.sector } : {}),
        });
        show(`${describe(saved)}\nReplay started (run ${res.run_id ?? '?'}, ${res.bars ?? saved.bars} bars).`, 'ok');
        ctx.ui.toast(`Replay started — ${res.bars ?? saved.bars} bars`, 'good');
        $('#rd-banner').hidden = false;
      } catch (e) {
        // textContent (via show): the server text is never interpreted as HTML.
        show(`${step} failed: ${(e && e.message) || e}`, 'err');
        ctx.ui.toast(`${step} failed`, 'bad');
      } finally {
        btn.innerHTML = label;
        btn.disabled = !canRun();
      }
    };

    // ---- Source 1: Yahoo ----------------------------------------------------
    const stockSel = $('#rd-stock');
    const customWrap = $('#rd-custom-wrap');
    const sectorWrap = $('#rd-sector-wrap');
    const intSel = $('#rd-int');
    const daysIn = $('#rd-days');
    const yBtn = $('#rd-yahoo-btn');
    const isOther = () => stockSel.value === OTHER;
    const ySymbol = () => (isOther() ? $('#rd-custom').value : stockSel.value).trim().toUpperCase();
    const ySector = () => (isOther() ? $('#rd-sector').value
      : stockSel.selectedOptions[0].dataset.sector);
    const yCanRun = () => ySymbol().length > 0;
    const syncYahoo = () => {
      customWrap.hidden = sectorWrap.hidden = !isOther();
      yBtn.disabled = !yCanRun();
    };
    stockSel.addEventListener('change', () => { syncYahoo(); msg.hidden = true; });
    $('#rd-custom').addEventListener('input', syncYahoo);
    // The day limit follows the interval (the server enforces it as well).
    intSel.addEventListener('change', () => {
      const max = MAX_DAYS[intSel.value];
      daysIn.max = String(max);
      if (Number(daysIn.value) > max) daysIn.value = String(max);
    });
    syncYahoo();
    yBtn.addEventListener('click', () => {
      if (!yCanRun()) return;
      const symbol = ySymbol();
      fetchThenReplay(yBtn, 'Download', 'Downloading from Yahoo…',
        () => ctx.api.replayYahoo({
          symbol, interval: intSel.value, days: parseInt(daysIn.value, 10) || 5,
        }),
        { symbol, sector: ySector() },
        (s) => `Downloaded ${s.bars} ${s.interval} bars of ${s.symbol} (${s.first_bar} to ${s.last_bar}), saved as ${s.filename}.`,
        yCanRun);
    });

    // ---- Source 2: upload ---------------------------------------------------
    const fileIn = $('#rd-file');
    const upBtn = $('#rd-upload-btn');
    const hasFile = () => fileIn.files.length > 0;
    fileIn.addEventListener('change', () => { upBtn.disabled = !hasFile(); msg.hidden = true; });
    upBtn.addEventListener('click', () => {
      if (!hasFile()) return;
      fetchThenReplay(upBtn, 'Upload', 'Checking file…',
        () => ctx.api.replayUpload(fileIn.files[0]),
        { symbol: $('#rd-symbol').value.trim() },
        (s) => `Uploaded ${s.filename}: ${s.bars} bars, saved on the server.`,
        hasFile);
    });
  },

  unmount() {
    if (!state) return;
    if (state.el) state.el.innerHTML = '';
    state = null;
  },
};
