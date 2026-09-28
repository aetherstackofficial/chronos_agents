// ============================================================================
// CHRONOS — LLM Strategy Copilot view (FE-FEAT)
// Consumes: POST /api/copilot via ctx.api.copilot(description)   (PROTOCOL_V2 §4)
//           POST /api/scripts via ctx.api.scripts.save(name, code) (PROTOCOL_V2 §1)
// Monaco:   uses the AMD loader / global monaco already loaded by index.html,
//           with a <pre> fallback if it is not ready.
// ============================================================================

const STYLE_ID = 'cp-view-style';
const CSS = `
.cp-root{height:100%;min-height:0;overflow:auto;padding:var(--s-4);display:grid;grid-template-columns:minmax(300px,420px) 1fr;gap:var(--s-4);align-content:start}
@media (max-width:980px){.cp-root{grid-template-columns:1fr}}
.cp-left,.cp-right{display:flex;flex-direction:column;gap:var(--s-4);min-width:0}
.cp-hero h2{display:flex;align-items:center;gap:var(--s-2);font-size:var(--fs-lg)}
.cp-hero h2 svg{width:18px;height:18px;color:var(--accent-2)}
.cp-hero p{margin:6px 0 0;color:var(--text-mut)}
.cp-prompt textarea{min-height:130px;resize:vertical;font-family:var(--font)}
.cp-chips{display:flex;flex-wrap:wrap;gap:var(--s-2)}
.cp-chips .chip{cursor:pointer;transition:border-color var(--dur-1) var(--ease),color var(--dur-1) var(--ease)}
.cp-chips .chip:hover{border-color:var(--accent);color:var(--text-hi)}
.cp-actions{display:flex;gap:var(--s-2);flex-wrap:wrap;align-items:center}
.cp-code-host{height:360px;min-height:0;background:var(--bg-0)}
.cp-code-host pre{margin:0;height:100%;overflow:auto;padding:var(--s-3);font-family:var(--mono);font-size:var(--fs-sm);color:var(--text);white-space:pre}
.cp-meta{display:flex;align-items:center;gap:var(--s-2)}
.cp-explain{margin:0;color:var(--text);line-height:1.6}
.cp-save{display:flex;gap:var(--s-2);align-items:center;flex-wrap:wrap}
.cp-save .input{width:200px}
.cp-empty{min-height:240px}
`;

const EXAMPLES = [
  'Buy 10 shares when RSI drops below 30, sell them all when RSI rises above 70.',
  'Mean-revert to VWAP: buy 5 when price is 0.5% below VWAP, sell 5 when 0.5% above.',
  'Momentum: buy 10 when price is above VWAP and order-flow imbalance is positive; flatten when OFI turns negative.',
  'Scalp the spread: place a limit buy at best bid and a limit sell at best ask, 5 shares each, whenever the spread is wider than 0.15.',
];

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

function ic(ctx, name) {
  try {
    const i = ctx.ui.icon(name);
    return typeof i === 'string' ? i : ((i && i.outerHTML) || '');
  } catch (e) { return ''; }
}

function ensureMonaco() {
  return new Promise((resolve) => {
    if (window.monaco && window.monaco.editor) return resolve(window.monaco);
    const req = window.require;
    if (typeof req === 'function' && typeof req.config === 'function') {
      try {
        req.config({ paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.0/min/vs' } });
      } catch (e) { /* already configured by the shell */ }
      let done = false;
      const finish = () => { if (!done) { done = true; resolve(window.monaco || null); } };
      try { req(['vs/editor/editor.main'], finish, finish); } catch (e) { finish(); }
      setTimeout(finish, 5000);
    } else {
      resolve(null);
    }
  });
}

let state = null;

export default {
  id: 'copilot', title: 'Copilot', icon: 'sparkles', group: 'LAB',

  async mount(el, ctx) {
    if (!document.getElementById(STYLE_ID)) {
      const st = document.createElement('style');
      st.id = STYLE_ID; st.textContent = CSS;
      document.head.appendChild(st);
    }
    state = { el, ctx, editor: null, code: '' };

    el.innerHTML = `
<div class="cp-root">
  <div class="cp-left">
    <section class="card cp-hero">
      <h2>${ic(ctx, 'sparkles')} Strategy Copilot</h2>
      <p>Describe your strategy in plain English. The copilot writes a runnable
         <span class="mono">UserStrategy</span> against the Chronos SDK — you review the code
         before anything runs. Nothing executes without your say-so.</p>
    </section>

    <section class="panel">
      <div class="panel-head"><span>Describe Your Strategy</span></div>
      <div class="panel-body cp-prompt">
        <textarea class="input" id="cp-desc" rows="6" aria-label="Strategy description"
          placeholder="e.g. Buy 10 shares when RSI drops below 30, sell when it crosses 70…"></textarea>
        <div class="cp-chips" id="cp-chips" style="margin-top:var(--s-3)">
          ${EXAMPLES.map((x, i) => `<button type="button" class="chip" data-i="${i}" title="${esc(x)}">${esc(x.slice(0, 38))}…</button>`).join('')}
        </div>
        <div class="cp-actions" style="margin-top:var(--s-4)">
          <button class="btn btn-primary" id="cp-gen">${ic(ctx, 'sparkles')} Generate</button>
        </div>
      </div>
    </section>
  </div>

  <div class="cp-right">
    <section class="panel">
      <div class="panel-head">
        <span>Generated Strategy</span>
        <span class="actions cp-meta" id="cp-meta"></span>
      </div>
      <div class="panel-body flush">
        <div class="cp-code-host" id="cp-code">
          <div class="empty cp-empty">${ic(ctx, 'sparkles')}<div>No code yet.<br>Describe a strategy and press <strong>Generate</strong>.</div></div>
        </div>
      </div>
    </section>
    <section class="card" id="cp-explain-card" hidden>
      <p class="cp-explain" id="cp-explain"></p>
    </section>
    <section class="card cp-save" id="cp-save-row" hidden>
      <input class="input mono" id="cp-name" value="copilot_strategy" aria-label="Script name">
      <button class="btn" id="cp-savebtn">Save to Library</button>
      <button class="btn btn-ghost" id="cp-open">Open in Strategies →</button>
    </section>
  </div>
</div>`;

    const desc = el.querySelector('#cp-desc');
    el.querySelector('#cp-chips').addEventListener('click', (ev) => {
      const chip = ev.target.closest('[data-i]');
      if (!chip) return;
      desc.value = EXAMPLES[parseInt(chip.dataset.i, 10)];
      desc.focus();
    });

    const showCode = async (code) => {
      const host = el.querySelector('#cp-code');
      const monaco = await ensureMonaco();
      if (!state) return;
      if (monaco && monaco.editor) {
        if (state.editor) {
          state.editor.setValue(code);
        } else {
          host.innerHTML = '';
          state.editor = monaco.editor.create(host, {
            value: code, language: 'python', theme: 'vs-dark', readOnly: true,
            minimap: { enabled: false }, fontSize: 12, lineNumbers: 'on',
            scrollBeyondLastLine: false, automaticLayout: true, padding: { top: 10 },
          });
        }
      } else {
        host.innerHTML = `<pre class="mono" tabindex="0" aria-label="Generated strategy code">${esc(code)}</pre>`;
      }
    };

    el.querySelector('#cp-gen').addEventListener('click', async () => {
      const text = desc.value.trim();
      if (!text) { ctx.ui.toast('Describe your strategy first', 'warn'); return; }
      const btn = el.querySelector('#cp-gen');
      const label = btn.innerHTML;
      btn.innerHTML = '<span class="spinner" aria-hidden="true"></span> Generating…';
      btn.disabled = true;
      try {
        const res = await ctx.api.copilot(text);
        if (!state) return;
        state.code = res.code || '';
        await showCode(state.code);
        const gemini = res.source === 'gemini';
        el.querySelector('#cp-meta').innerHTML =
          `<span class="chip ${gemini ? 'accent' : ''}" title="${gemini ? 'Generated by Gemini' : 'Deterministic template built from your description (no LLM key or LLM failed)'}">${esc(res.source || 'template')}</span>`;
        const ex = el.querySelector('#cp-explain-card');
        ex.hidden = !res.explanation;
        el.querySelector('#cp-explain').textContent = res.explanation || '';
        el.querySelector('#cp-save-row').hidden = false;
        const slug = text.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 28);
        el.querySelector('#cp-name').value = slug || 'copilot_strategy';
        ctx.ui.toast('Strategy generated — review the code before running', 'good');
      } catch (e) {
        ctx.ui.toast('Copilot failed: ' + ((e && e.message) || e), 'bad');
      } finally {
        btn.innerHTML = label;
        btn.disabled = false;
      }
    });

    el.querySelector('#cp-savebtn').addEventListener('click', async () => {
      if (!state || !state.code) { ctx.ui.toast('Nothing to save yet', 'warn'); return; }
      const btn = el.querySelector('#cp-savebtn');
      const name = el.querySelector('#cp-name').value.trim() || 'copilot_strategy';
      const label = btn.innerHTML;
      btn.innerHTML = '<span class="spinner" aria-hidden="true"></span>';
      btn.disabled = true;
      try {
        const meta = await ctx.api.scripts.save(name, state.code);
        ctx.ui.toast(`Saved "${name}" to the script library (${(meta && meta.script_id) || 'ok'})`, 'good');
      } catch (e) {
        ctx.ui.toast('Save failed: ' + ((e && e.message) || e), 'bad');
      } finally {
        btn.innerHTML = label;
        btn.disabled = false;
      }
    });

    el.querySelector('#cp-open').addEventListener('click', () => {
      if (!state || !state.code) { ctx.ui.toast('Generate a strategy first', 'warn'); return; }
      try { sessionStorage.setItem('chronos.copilot.code', state.code); } catch (e) { /* storage full */ }
      location.hash = '#/strategies';
    });
  },

  unmount() {
    if (!state) return;
    if (state.editor) { try { state.editor.dispose(); } catch (e) { /* gone */ } }
    if (state.el) state.el.innerHTML = '';
    state = null;
  },
};
