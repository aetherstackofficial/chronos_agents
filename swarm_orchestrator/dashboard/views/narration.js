// ============================================================================
// CHRONOS — Explainable Market Narration view (FE-FEAT)
// Consumes: POST /api/narrate via ctx.api.narrate()   (PROTOCOL_V2 §6)
// WS:       store latest tick for timestamps          (PROTOCOL §5.1)
// ============================================================================

const STYLE_ID = 'nr-view-style';
const CSS = `
.nr-root{height:100%;min-height:0;padding:var(--s-4);display:flex;flex-direction:column;gap:var(--s-4)}
.nr-hero h2{display:flex;align-items:center;gap:var(--s-2);font-size:var(--fs-lg)}
.nr-hero h2 svg{width:18px;height:18px;color:var(--accent-2)}
.nr-hero p{margin:6px 0 0;color:var(--text-mut);max-width:70ch}
.nr-panel{flex:1 1 auto;min-height:0}
.nr-feed{display:flex;flex-direction:column;gap:var(--s-3)}
.nr-card{display:flex;flex-direction:column;gap:var(--s-2);border-left:3px solid var(--accent);animation:nr-in var(--dur-3) var(--ease)}
@keyframes nr-in{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:none}}
.nr-card p{margin:0;color:var(--text);line-height:1.6}
.nr-meta{display:flex;align-items:center;gap:var(--s-2);font-size:var(--fs-xs);color:var(--text-mut)}
.nr-auto{display:inline-flex;align-items:center;gap:6px}
.nr-auto.on{color:var(--accent-2)}
.nr-auto .dot{width:7px;height:7px;border-radius:50%;background:currentColor}
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

const MAX_CARDS = 50;

let state = null;

export default {
  id: 'narration', title: 'Narration', icon: 'feed', group: 'LAB',

  async mount(el, ctx) {
    if (!document.getElementById(STYLE_ID)) {
      const st = document.createElement('style');
      st.id = STYLE_ID; st.textContent = CSS;
      document.head.appendChild(st);
    }
    state = { el, ctx, timer: null, inflight: false, count: 0 };

    el.innerHTML = `
<div class="nr-root">
  <section class="card nr-hero">
    <h2>${ic(ctx, 'feed')} Market Narration</h2>
    <p>The explainable market: ask WHY the tape is doing what it is doing. The narrator reads the
       live tick — regime, order-flow imbalance, spread, who is winning — and answers in plain
       language.</p>
  </section>

  <section class="panel nr-panel">
    <div class="panel-head">
      <span>Narration Feed</span>
      <span class="actions">
        <button class="btn btn-sm btn-ghost nr-auto" id="nr-auto" aria-pressed="false" title="Refresh automatically every 15 seconds">
          <span class="dot" aria-hidden="true"></span> Auto 15s
        </button>
        <button class="btn btn-sm btn-primary" id="nr-explain">${ic(ctx, 'feed')} Explain Now</button>
      </span>
    </div>
    <div class="panel-body">
      <div class="nr-feed" id="nr-feed">
        <div class="empty" id="nr-empty">${ic(ctx, 'feed')}
          <div>No narrations yet.<br>Press <strong>Explain Now</strong> to ask the market why —
          or flip on <strong>Auto</strong> for a running commentary.</div>
        </div>
      </div>
    </div>
  </section>
</div>`;

    const feed = el.querySelector('#nr-feed');
    const explainBtn = el.querySelector('#nr-explain');
    const autoBtn = el.querySelector('#nr-auto');

    const explain = async (viaAuto) => {
      if (!state || state.inflight) return;
      state.inflight = true;
      const label = explainBtn.innerHTML;
      if (!viaAuto) {
        explainBtn.innerHTML = '<span class="spinner" aria-hidden="true"></span> Thinking…';
        explainBtn.disabled = true;
      }
      try {
        const res = await ctx.api.narrate();
        if (!state) return;
        const tick = (ctx.store.latestTick && ctx.store.latestTick()) || ctx.store.state.tick || null;
        const ts = (tick && tick.unix_time) || Math.floor(Date.now() / 1000);
        const gemini = res.source === 'gemini';
        const emptyEl = el.querySelector('#nr-empty');
        if (emptyEl) emptyEl.remove();
        const card = document.createElement('article');
        card.className = 'card nr-card';
        card.innerHTML = `
          <div class="nr-meta">
            <span class="mono">${ctx.fmt.time(ts)}</span>
            ${tick ? `<span class="mono">DAY ${tick.day_count} · min ${tick.market_minute}</span>` : ''}
            <span class="chip ${gemini ? 'accent' : ''}" title="${gemini ? 'Narrated by Gemini' : 'Deterministic heuristic composed from the live numbers'}">${esc(res.source || 'heuristic')}</span>
          </div>
          <p>${esc(res.narration)}</p>`;
        feed.prepend(card);
        state.count += 1;
        while (feed.children.length > MAX_CARDS) feed.lastElementChild.remove();
      } catch (e) {
        if (!viaAuto) ctx.ui.toast('Narration failed: ' + ((e && e.message) || e), 'bad');
      } finally {
        if (state) state.inflight = false;
        explainBtn.innerHTML = label;
        explainBtn.disabled = false;
      }
    };

    explainBtn.addEventListener('click', () => explain(false));

    autoBtn.addEventListener('click', () => {
      if (!state) return;
      const on = autoBtn.getAttribute('aria-pressed') !== 'true';
      autoBtn.setAttribute('aria-pressed', String(on));
      autoBtn.classList.toggle('on', on);
      if (on) {
        explain(true);
        state.timer = setInterval(() => explain(true), 15000);
        ctx.ui.toast('Auto-narration on — refreshing every 15 s', 'good');
      } else {
        clearInterval(state.timer);
        state.timer = null;
      }
    });
  },

  unmount() {
    if (!state) return;
    if (state.timer) clearInterval(state.timer);
    if (state.el) state.el.innerHTML = '';
    state = null;
  },
};
