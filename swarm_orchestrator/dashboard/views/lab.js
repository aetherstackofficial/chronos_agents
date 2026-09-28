// ============================================================================
// CHRONOS — Lab (experimental) view (FE-FEAT)
// Consumes: POST /api/lab/redteam {count} · POST /api/lab/redteam/stop
//           (optional feature endpoints, PROTOCOL_V2 §8; honest CLI fallback
//            `python worker.py --role redteam --count N` when absent)
// ============================================================================

const STYLE_ID = 'lab-view-style';
const CSS = `
.lab-root{height:100%;min-height:0;overflow:auto;padding:var(--s-4);display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:var(--s-4);align-content:start}
.lab-hero{grid-column:1/-1}
.lab-hero h2{display:flex;align-items:center;gap:var(--s-2);font-size:var(--fs-lg)}
.lab-hero h2 svg{width:18px;height:18px;color:var(--accent-2)}
.lab-hero p{margin:6px 0 0;color:var(--text-mut);max-width:76ch}
.lab-card{display:flex;flex-direction:column;gap:var(--s-3)}
.lab-card h3{display:flex;align-items:center;gap:var(--s-2);font-size:var(--fs-md)}
.lab-card p{margin:0;color:var(--text-mut);line-height:1.6}
.lab-card .note{font-size:var(--fs-sm);color:var(--text-dim);border-left:2px solid var(--border-strong);padding-left:var(--s-3)}
.lab-controls{display:flex;gap:var(--s-2);align-items:flex-end;flex-wrap:wrap}
.lab-controls label.field{width:110px}
.lab-cmd{display:flex;align-items:center;gap:var(--s-2);background:var(--bg-0);border:1px solid var(--border);border-radius:var(--r-md);padding:var(--s-2) var(--s-3)}
.lab-cmd code{flex:1 1 auto;font-size:var(--fs-sm);color:var(--text-hi);overflow-x:auto;white-space:nowrap}
.lab-roadmap{border-style:dashed}
.lab-roadmap ul{margin:0;padding-left:1.2em;color:var(--text-mut);line-height:1.8}
.lab-roadmap li strong{color:var(--text)}
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

function busy(btn, on, text) {
  if (!btn) return;
  if (on) {
    btn.dataset.label = btn.innerHTML;
    btn.innerHTML = '<span class="spinner" aria-hidden="true"></span> ' + (text || '');
    btn.disabled = true;
  } else {
    if (btn.dataset.label) btn.innerHTML = btn.dataset.label;
    btn.disabled = false;
  }
}

let state = null;

export default {
  id: 'lab', title: 'Lab', icon: 'flask', group: 'LAB',

  async mount(el, ctx) {
    if (!document.getElementById(STYLE_ID)) {
      const st = document.createElement('style');
      st.id = STYLE_ID; st.textContent = CSS;
      document.head.appendChild(st);
    }
    state = { el, ctx };

    el.innerHTML = `
<div class="lab-root">
  <section class="card lab-hero">
    <h2>${ic(ctx, 'flask')} Lab</h2>
    <p>Experimental capabilities. Everything here is honestly labeled: what runs today runs for
       real through the engine; what is roadmap is marked roadmap and never faked.</p>
  </section>

  <section class="card lab-card">
    <h3>${ic(ctx, 'flask')} Adversarial Red-Team <span class="chip warn">HEURISTIC</span></h3>
    <p>Launches real predatory taker agents (<span class="mono">REDTEAM_i</span>) against the live
       book. They watch order-flow imbalance and short-horizon momentum, aggressively fade
       extended moves, and sweep thin books — modeling the adverse selection your strategy would
       face from professional counterparties on a real venue.</p>
    <p class="note">Honest scope: these are hand-tuned heuristics, not trained adversaries.
       An RL-trained red-team is on the roadmap. They connect like any worker — SUB market data,
       DEALER orders — no special privileges.</p>
    <div class="lab-controls">
      <label class="field"># Agents
        <input class="input mono" id="lab-count" type="number" min="1" max="8" value="2" aria-label="Number of red-team agents">
      </label>
      <button class="btn btn-warn" id="lab-launch">${ic(ctx, 'flask')} Launch Red-Team</button>
      <button class="btn btn-ghost" id="lab-stop">Stop Red-Team</button>
    </div>
    <div id="lab-fallback" hidden>
      <p class="note">The bridge does not expose the red-team endpoint in this deployment.
         Launch it directly from a terminal instead:</p>
      <div class="lab-cmd">
        <code class="mono" id="lab-cmd-text">python worker.py --role redteam --count 2</code>
        <button class="btn btn-sm" id="lab-copy" aria-label="Copy command">Copy</button>
      </div>
    </div>
  </section>

  <section class="card lab-card lab-roadmap">
    <h3>${ic(ctx, 'flask')} Cross-Venue Fragmentation <span class="chip">ROADMAP — NOT YET BUILT</span></h3>
    <p>This capability does not exist yet; nothing here is simulated or mocked. When built, it
       will run a second matching engine as an independent venue for the same symbol and study
       real fragmentation effects:</p>
    <ul>
      <li><strong>Two books, one symbol</strong> — separate depth, separate prints, a consolidated tape.</li>
      <li><strong>Latency-modeled smart order routing</strong> — agents choose a venue per order under configurable one-way delays.</li>
      <li><strong>Cross-venue arbitrageurs</strong> — real agents closing dislocations, so the spread between venues is emergent.</li>
      <li><strong>NBBO-style consolidated view</strong> in the Market screen, with per-venue attribution.</li>
    </ul>
    <p class="note">Tracking as Phase-3 work; the engine core is already transport-free, which is
       what makes multiple venue instances feasible.</p>
  </section>
</div>`;

    const countEl = el.querySelector('#lab-count');
    const fallback = el.querySelector('#lab-fallback');
    const cmdText = el.querySelector('#lab-cmd-text');
    const getCount = () => Math.max(1, Math.min(8, parseInt(countEl.value, 10) || 2));

    const showFallback = () => {
      cmdText.textContent = `python worker.py --role redteam --count ${getCount()}`;
      fallback.hidden = false;
    };

    el.querySelector('#lab-launch').addEventListener('click', async () => {
      const btn = el.querySelector('#lab-launch');
      const count = getCount();
      busy(btn, true, 'Launching…');
      try {
        const r = await fetch('/api/lab/redteam', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ count }),
        });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        await r.json().catch(() => ({}));
        ctx.ui.toast(`Red-team launched — ${count} predatory agent${count > 1 ? 's' : ''} live`, 'good');
        fallback.hidden = true;
      } catch (e) {
        showFallback();
        ctx.ui.toast('Red-team endpoint unavailable — use the CLI command shown below', 'warn');
      } finally {
        busy(btn, false);
      }
    });

    el.querySelector('#lab-stop').addEventListener('click', async () => {
      const btn = el.querySelector('#lab-stop');
      busy(btn, true, 'Stopping…');
      try {
        const r = await fetch('/api/lab/redteam/stop', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
        });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        ctx.ui.toast('Red-team stopped', 'good');
      } catch (e) {
        ctx.ui.toast('Could not stop via API: ' + ((e && e.message) || e), 'bad');
      } finally {
        busy(btn, false);
      }
    });

    el.querySelector('#lab-copy').addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(cmdText.textContent);
        ctx.ui.toast('Command copied to clipboard', 'good');
      } catch (e) {
        ctx.ui.toast('Copy failed — select the text manually', 'warn');
      }
    });
  },

  unmount() {
    if (!state) return;
    if (state.el) state.el.innerHTML = '';
    state = null;
  },
};
