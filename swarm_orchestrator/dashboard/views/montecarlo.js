// ============================================================================
// CHRONOS — Monte-Carlo counterfactual fan view (FE-FEAT)
// Consumes: POST /api/montecarlo via ctx.api.montecarlo(cfg)   (PROTOCOL_V2 §3)
// Renders:  ctx.charts.fanChart (p5-p95 / p25-p75 bands, p50 median, samples)
// ============================================================================

const STYLE_ID = 'mc-view-style';
const CSS = `
.mc-root{height:100%;min-height:0;overflow:auto;padding:var(--s-4);display:flex;flex-direction:column;gap:var(--s-4)}
.mc-hero{display:flex;flex-direction:column;gap:var(--s-2)}
.mc-hero h2{display:flex;align-items:center;gap:var(--s-2);font-size:var(--fs-lg)}
.mc-hero h2 svg{width:18px;height:18px;color:var(--accent-2)}
.mc-hero p{margin:0;color:var(--text-mut);max-width:72ch}
.mc-hero .mc-lede{color:var(--text);font-weight:500}
.mc-fields{display:flex;flex-wrap:wrap;gap:var(--s-4);align-items:flex-end}
.mc-fields label.field{width:130px}
.mc-shock{display:flex;flex-wrap:wrap;gap:var(--s-4);align-items:flex-end;padding:var(--s-3);
  border:1px dashed var(--border-strong);border-radius:var(--r-md);background:var(--bg-0)}
.mc-shock.off label.field{opacity:.45;pointer-events:none}
.mc-shock-toggle{display:flex;align-items:center;gap:var(--s-2);font-size:var(--fs-sm);color:var(--text-mut);cursor:pointer;user-select:none}
.mc-shock-toggle input{accent-color:var(--accent)}
.mc-chart-host{height:340px;min-height:0;position:relative}
.mc-chart-host .empty{position:absolute;inset:0}
.mc-bottom{display:grid;grid-template-columns:1fr;gap:var(--s-4)}
.mc-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:var(--s-3)}
.term-wrap{display:flex;flex-direction:column;gap:var(--s-2)}
.term-track{position:relative;height:28px;background:var(--bg-0);border:1px solid var(--border);border-radius:var(--r-full);overflow:hidden}
.term-band{position:absolute;top:0;bottom:0;border-radius:var(--r-full)}
.term-band.outer{background:var(--accent-soft)}
.term-band.inner{background:var(--accent);opacity:.35}
.term-mark{position:absolute;top:3px;bottom:3px;width:2px;border-radius:1px}
.term-mark.median{background:var(--accent-2)}
.term-mark.mean{background:var(--warn)}
.term-scale{display:flex;justify-content:space-between;color:var(--text-mut);font-size:var(--fs-xs)}
.term-legend{display:flex;gap:var(--s-3);flex-wrap:wrap;font-size:var(--fs-xs);color:var(--text-mut)}
.term-legend .key{display:inline-flex;align-items:center;gap:6px}
.term-legend .swatch{width:14px;height:8px;border-radius:2px;display:inline-block}
.term-legend .swatch.outer{background:var(--accent-soft)}
.term-legend .swatch.inner{background:var(--accent);opacity:.5}
.term-legend .swatch.median{background:var(--accent-2);width:3px;height:12px}
.term-legend .swatch.mean{background:var(--warn);width:3px;height:12px}
`;

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

function ic(ctx, name) {
  try {
    const i = ctx.ui.icon(name);
    return typeof i === 'string' ? i : ((i && i.outerHTML) || '');
  } catch (e) { return ''; }
}

function busy(btn, on) {
  if (!btn) return;
  if (on) {
    btn.dataset.label = btn.innerHTML;
    btn.innerHTML = '<span class="spinner" aria-hidden="true"></span> Running…';
    btn.disabled = true;
  } else {
    if (btn.dataset.label) btn.innerHTML = btn.dataset.label;
    btn.disabled = false;
  }
}

function disposeChart(h) {
  if (!h) return;
  try {
    if (typeof h.dispose === 'function') h.dispose();
    else if (typeof h.remove === 'function') h.remove();
    else if (h.chart && typeof h.chart.remove === 'function') h.chart.remove();
  } catch (e) { /* already gone */ }
}

let state = null;

export default {
  id: 'montecarlo', title: 'Monte-Carlo', icon: 'fan', group: 'LAB',

  async mount(el, ctx) {
    if (!document.getElementById(STYLE_ID)) {
      const st = document.createElement('style');
      st.id = STYLE_ID; st.textContent = CSS;
      document.head.appendChild(st);
    }
    state = { el, ctx, chart: null };

    const tick = (ctx.store.latestTick && ctx.store.latestTick()) || ctx.store.state.tick || null;
    const defSymbol = (tick && tick.symbol) || 'TCS';
    const defPrice = (tick && tick.last_price) || ctx.store.state.price || 190;

    el.innerHTML = `
<div class="mc-root">
  <section class="card mc-hero">
    <h2>${ic(ctx, 'fan')} Monte-Carlo Counterfactual Fan</h2>
    <p class="mc-lede">1,000 futures your order could have caused — a distribution, not a point estimate.</p>
    <p>Each path is a fully independent matching-engine universe: same venue physics, different seed.
       Optionally inject a shock order to see how YOUR size would bend the distribution.</p>
  </section>

  <section class="panel">
    <div class="panel-head"><span>Simulation Config</span></div>
    <div class="panel-body">
      <form class="mc-fields" id="mc-form">
        <label class="field">Symbol
          <input class="input mono" id="mc-symbol" value="${esc(defSymbol)}" required aria-label="Symbol">
        </label>
        <label class="field">Start Price (₹)
          <input class="input mono" id="mc-price" type="number" step="0.05" min="0.05" value="${Number(defPrice).toFixed(2)}" required aria-label="Start price">
        </label>
        <label class="field"># Paths
          <input class="input mono" id="mc-paths" type="number" min="2" max="500" value="60" required aria-label="Number of paths">
        </label>
        <label class="field">Horizon (min)
          <input class="input mono" id="mc-horizon" type="number" min="5" max="750" value="90" required aria-label="Horizon in minutes">
        </label>
        <div class="mc-shock off" id="mc-shock">
          <label class="mc-shock-toggle"><input type="checkbox" id="mc-shock-on" aria-label="Enable shock order"> Inject shock order</label>
          <label class="field">At minute
            <input class="input mono" id="mc-shock-min" type="number" min="1" value="10" aria-label="Shock minute">
          </label>
          <label class="field">Side
            <select class="select" id="mc-shock-side" aria-label="Shock side">
              <option value="BUY">BUY</option><option value="SELL">SELL</option>
            </select>
          </label>
          <label class="field">Qty
            <input class="input mono" id="mc-shock-qty" type="number" min="1" value="500" aria-label="Shock quantity">
          </label>
        </div>
        <button class="btn btn-primary" id="mc-run" type="submit">${ic(ctx, 'fan')} Run Simulation</button>
      </form>
    </div>
  </section>

  <section class="panel">
    <div class="panel-head"><span>Price Fan — P5–P95</span><span class="actions mono" id="mc-fan-meta"></span></div>
    <div class="panel-body flush">
      <div class="mc-chart-host" id="mc-chart">
        <div class="empty">${ic(ctx, 'fan')}<div>No simulation yet.<br>Configure the fan above and press <strong>Run Simulation</strong>.</div></div>
      </div>
    </div>
  </section>

  <section class="mc-bottom" id="mc-bottom" hidden>
    <div class="panel">
      <div class="panel-head"><span>Terminal Price Distribution</span></div>
      <div class="panel-body">
        <div class="term-wrap" id="mc-terminal"></div>
      </div>
    </div>
    <div class="mc-stats" id="mc-stats"></div>
  </section>
</div>`;

    const form = el.querySelector('#mc-form');
    const shockBox = el.querySelector('#mc-shock');
    const shockOn = el.querySelector('#mc-shock-on');
    shockOn.addEventListener('change', () => shockBox.classList.toggle('off', !shockOn.checked));

    form.addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const btn = el.querySelector('#mc-run');
      const cfg = {
        symbol: el.querySelector('#mc-symbol').value.trim().toUpperCase() || 'TCS',
        price: parseFloat(el.querySelector('#mc-price').value) || defPrice,
        n_paths: Math.max(2, Math.min(500, parseInt(el.querySelector('#mc-paths').value, 10) || 60)),
        horizon: Math.max(5, Math.min(750, parseInt(el.querySelector('#mc-horizon').value, 10) || 90)),
      };
      if (shockOn.checked) {
        cfg.shock = {
          minute: Math.max(1, parseInt(el.querySelector('#mc-shock-min').value, 10) || 10),
          side: el.querySelector('#mc-shock-side').value,
          qty: Math.max(1, parseInt(el.querySelector('#mc-shock-qty').value, 10) || 500),
        };
      }
      busy(btn, true);
      try {
        const res = await ctx.api.montecarlo(cfg);
        renderResult(res);
        ctx.ui.toast(`Simulated ${res.n_paths} independent futures over ${res.horizon} minutes`, 'good');
      } catch (e) {
        ctx.ui.toast('Monte-Carlo failed: ' + ((e && e.message) || e), 'bad');
      } finally {
        busy(btn, false);
      }
    });

    const renderResult = (res) => {
      const fmt = ctx.fmt;
      const host = el.querySelector('#mc-chart');
      disposeChart(state.chart);
      host.innerHTML = '';
      try {
        state.chart = ctx.charts.fanChart(host, res);
      } catch (e) {
        host.innerHTML = `<div class="empty">Chart failed to render: ${esc((e && e.message) || e)}</div>`;
      }
      el.querySelector('#mc-fan-meta').textContent =
        `${res.symbol} · ${res.n_paths} paths · ${res.horizon} min · start ${fmt.inr(res.start_price)}`;

      const t = res.terminal || {};
      const bottom = el.querySelector('#mc-bottom');
      bottom.hidden = false;

      const min = Number(t.min), max = Number(t.max);
      const range = (max - min) || 1;
      const pos = (v) => (((Number(v) - min) / range) * 100).toFixed(2) + '%';
      const width = (a, b) => (((Number(b) - Number(a)) / range) * 100).toFixed(2) + '%';
      el.querySelector('#mc-terminal').innerHTML = `
        <div class="term-track" role="img" aria-label="Terminal price distribution from ${fmt.inr(min)} to ${fmt.inr(max)}">
          <div class="term-band outer" style="left:${pos(t.p5)};width:${width(t.p5, t.p95)}" title="P5–P95 ${fmt.inr(t.p5)} → ${fmt.inr(t.p95)}"></div>
          <div class="term-band inner" style="left:${pos(t.p25)};width:${width(t.p25, t.p75)}" title="P25–P75 ${fmt.inr(t.p25)} → ${fmt.inr(t.p75)}"></div>
          <div class="term-mark median" style="left:${pos(t.p50)}" title="Median ${fmt.inr(t.p50)}"></div>
          <div class="term-mark mean" style="left:${pos(t.mean)}" title="Mean ${fmt.inr(t.mean)}"></div>
        </div>
        <div class="term-scale mono"><span>${fmt.inr(min)}</span><span>${fmt.inr(max)}</span></div>
        <div class="term-legend">
          <span class="key"><span class="swatch outer"></span>P5–P95</span>
          <span class="key"><span class="swatch inner"></span>P25–P75</span>
          <span class="key"><span class="swatch median"></span>Median</span>
          <span class="key"><span class="swatch mean"></span>Mean</span>
        </div>`;

      el.querySelector('#mc-stats').innerHTML = [
        ['Median (P50)', fmt.inr(t.p50)],
        ['P5', fmt.inr(t.p5)],
        ['P95', fmt.inr(t.p95)],
        ['Mean', fmt.inr(t.mean)],
        ['Min', fmt.inr(t.min)],
        ['Max', fmt.inr(t.max)],
      ].map(([k, v]) => `<div class="card stat"><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');
    };
  },

  unmount() {
    if (!state) return;
    disposeChart(state.chart);
    if (state.el) state.el.innerHTML = '';
    state = null;
  },
};
