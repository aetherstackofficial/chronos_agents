// ============================================================================
// CHRONOS — Time Machine crisis-replay view (FE-FEAT)
// Consumes: GET /api/timemachine/scenarios, POST /api/timemachine/start
//           via ctx.api.timemachine.scenarios()/start(id, seed)  (PROTOCOL_V2 §5)
// WS:       store onTick — live mode/day/minute status line (PROTOCOL §5.1)
// ============================================================================

const STYLE_ID = 'tm-view-style';
const CSS = `
.tm-root{height:100%;min-height:0;overflow:auto;padding:var(--s-4);display:flex;flex-direction:column;gap:var(--s-4)}
.tm-head{display:flex;align-items:flex-end;justify-content:space-between;gap:var(--s-4);flex-wrap:wrap}
.tm-head h2{display:flex;align-items:center;gap:var(--s-2);font-size:var(--fs-lg)}
.tm-head h2 svg{width:18px;height:18px;color:var(--accent-2)}
.tm-head p{margin:4px 0 0;color:var(--text-mut);max-width:70ch}
.tm-seed{display:flex;align-items:flex-end;gap:var(--s-3)}
.tm-seed label.field{width:110px}
.tm-status{display:flex;align-items:center;gap:var(--s-3);font-size:var(--fs-sm);color:var(--text-mut)}
.tm-status .mono{color:var(--text-hi)}
.tm-banner{border-left:3px solid var(--reactive);display:flex;flex-direction:column;gap:var(--s-1)}
.tm-banner strong{color:var(--text-hi)}
.tm-banner p{margin:0;color:var(--text-mut)}
.tm-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:var(--s-4)}
.tm-card{display:flex;flex-direction:column;gap:var(--s-3);transition:border-color var(--dur-2) var(--ease),transform var(--dur-2) var(--ease)}
.tm-card:hover{transform:translateY(-2px);border-color:var(--border-accent)}
.tm-card-head{display:flex;align-items:flex-start;justify-content:space-between;gap:var(--s-2)}
.tm-card-head h3{font-size:var(--fs-md)}
.tm-card p{margin:0;color:var(--text-mut);flex:1 1 auto}
.tm-meta{color:var(--text-dim);font-size:var(--fs-xs)}
.tm-skel{height:170px}
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

const TAG_CLASS = { crash: 'down', squeeze: 'accent', shock: 'warn' };

let state = null;

export default {
  id: 'timemachine', title: 'Time Machine', icon: 'clock', group: 'LAB',

  async mount(el, ctx) {
    if (!document.getElementById(STYLE_ID)) {
      const st = document.createElement('style');
      st.id = STYLE_ID; st.textContent = CSS;
      document.head.appendChild(st);
    }
    state = { el, ctx, unsub: null };

    el.innerHTML = `
<div class="tm-root">
  <div class="tm-head">
    <div>
      <h2>${ic(ctx, 'clock')} Time Machine</h2>
      <p>Re-live a market crisis bar-by-bar through the real matching engine. History plays back
         exactly — until one of your strategies trades. Then the REACTIVE latch flips and the
         future is yours to break.</p>
    </div>
    <div class="tm-seed">
      <label class="field">Seed
        <input class="input mono" id="tm-seed" type="number" value="42" aria-label="Replay seed">
      </label>
    </div>
  </div>

  <div class="tm-status" id="tm-status" role="status" aria-live="polite">
    <span class="chip" id="tm-mode"><span class="dot"></span><span id="tm-mode-txt">OFFLINE</span></span>
    <span id="tm-clock" class="mono">—</span>
  </div>

  <div class="card tm-banner" id="tm-banner" hidden>
    <strong>REPLAY armed — the REACTIVE latch is live.</strong>
    <p>The market is now re-living this crisis through the real order book. The moment YOUR
       strategy's order fills, the mode latches to <span class="chip reactive">REACTIVE</span>
       permanently: the replayer stops re-anchoring, history diverges, and the swarm owns the
       price. Go to <a href="#/strategies">Strategies</a> to intervene.</p>
  </div>

  <div class="tm-grid" id="tm-grid">
    <div class="skeleton tm-skel"></div>
    <div class="skeleton tm-skel"></div>
    <div class="skeleton tm-skel"></div>
  </div>
</div>`;

    const modeChip = el.querySelector('#tm-mode');
    const modeTxt = el.querySelector('#tm-mode-txt');
    const clockEl = el.querySelector('#tm-clock');
    const paint = (tick) => {
      if (!tick) return;
      const mode = tick.mode || ctx.store.state.mode || 'LIVE';
      modeChip.className = 'chip ' + (String(mode).toLowerCase() || '');
      if (mode === 'REACTIVE') modeChip.classList.add('pulse');
      modeTxt.textContent = mode;
      clockEl.textContent =
        `DAY ${tick.day_count ?? '—'} · min ${tick.market_minute ?? '—'} · ${ctx.fmt.time(tick.unix_time)} · ${ctx.fmt.inr(tick.last_price)}`;
      if (mode === 'REACTIVE') el.querySelector('#tm-banner').hidden = false;
    };
    paint((ctx.store.latestTick && ctx.store.latestTick()) || ctx.store.state.tick);
    state.unsub = ctx.store.onTick(paint);

    const grid = el.querySelector('#tm-grid');
    try {
      const scenarios = await ctx.api.timemachine.scenarios();
      if (!state) return; // unmounted mid-flight
      if (!scenarios || !scenarios.length) {
        grid.innerHTML = `<div class="empty">${ic(ctx, 'clock')}<div>No scenarios available from the feature service.</div></div>`;
        return;
      }
      grid.innerHTML = scenarios.map((s) => `
        <article class="card tm-card">
          <div class="tm-card-head">
            <h3>${esc(s.name)}</h3>
            <span class="chip ${TAG_CLASS[s.tag] || ''}">${esc(s.tag || 'scenario')}</span>
          </div>
          <p>${esc(s.description)}</p>
          <div class="tm-meta mono">${esc(s.symbol || '')} · ${s.bars ?? '?'} bars · 1-min tape</div>
          <button class="btn btn-primary" data-id="${esc(s.id)}" aria-label="Replay ${esc(s.name)}">
            ${ic(ctx, 'clock')} Replay This Crisis
          </button>
        </article>`).join('');

      grid.querySelectorAll('button[data-id]').forEach((btn) => {
        btn.addEventListener('click', async () => {
          const label = btn.innerHTML;
          btn.innerHTML = '<span class="spinner" aria-hidden="true"></span> Loading tape…';
          btn.disabled = true;
          try {
            const seed = parseInt(el.querySelector('#tm-seed').value, 10);
            const res = await ctx.api.timemachine.start(btn.dataset.id, Number.isFinite(seed) ? seed : 42);
            ctx.ui.toast(`Replay started — ${res.bars ?? '?'} bars queued (run ${res.run_id ?? '?'})`, 'good');
            el.querySelector('#tm-banner').hidden = false;
          } catch (e) {
            ctx.ui.toast('Replay failed: ' + ((e && e.message) || e), 'bad');
          } finally {
            btn.innerHTML = label;
            btn.disabled = false;
          }
        });
      });
    } catch (e) {
      if (!state) return;
      grid.innerHTML = `<div class="empty">${ic(ctx, 'clock')}<div>Could not load scenarios.<br><span class="mono">${esc((e && e.message) || e)}</span></div></div>`;
      ctx.ui.toast('Failed to load Time-Machine scenarios', 'bad');
    }
  },

  unmount() {
    if (!state) return;
    if (typeof state.unsub === 'function') state.unsub();
    if (state.el) state.el.innerHTML = '';
    state = null;
  },
};
