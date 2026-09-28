// ============================================================================
// CHRONOS — Runs persistence explorer (FE-FEAT)
// Consumes: GET /api/runs                       via ctx.api.runs()
//           GET /api/runs/{id}/equity[?agent_id] via ctx.api.equity(runId, agentId)
//           (PROTOCOL §6; equity without agent returns {agents:[...]})
// ============================================================================

const STYLE_ID = 'runs-view-style';
const CSS = `
.runs-root{height:100%;min-height:0;padding:var(--s-4);display:grid;grid-template-columns:minmax(300px,380px) 1fr;gap:var(--s-4)}
@media (max-width:980px){.runs-root{grid-template-columns:1fr;overflow:auto}}
.runs-list,.runs-detail{min-height:0}
.runs-list .panel-body{padding:0}
.runs-table tbody tr{cursor:pointer}
.runs-table tbody tr.sel{background:var(--accent-soft)}
.runs-table tbody tr.sel td:first-child{box-shadow:inset 3px 0 0 var(--accent)}
.runs-detail .panel-body{display:flex;flex-direction:column;gap:var(--s-4)}
.runs-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:var(--s-3)}
.runs-chart{height:300px;min-height:0;position:relative}
.runs-chart .empty{position:absolute;inset:0}
.runs-picker{display:flex;align-items:center;gap:var(--s-2);text-transform:none;letter-spacing:0}
.runs-picker .select{width:auto;min-width:150px;padding:5px 9px;font-size:var(--fs-sm)}
.runs-skel{height:40px;margin:var(--s-2) var(--s-3)}
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

function disposeChart(h) {
  if (!h) return;
  try {
    if (typeof h.dispose === 'function') h.dispose();
    else if (typeof h.remove === 'function') h.remove();
    else if (h.chart && typeof h.chart.remove === 'function') h.chart.remove();
  } catch (e) { /* gone */ }
}

function setLineData(h, data) {
  if (!h) return;
  if (typeof h.setData === 'function') h.setData(data);
  else if (h.series && typeof h.series.setData === 'function') h.series.setData(data);
}

const tokenColor = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || undefined;

let state = null;

export default {
  id: 'runs', title: 'Runs', icon: 'database', group: 'DATA',

  async mount(el, ctx) {
    if (!document.getElementById(STYLE_ID)) {
      const st = document.createElement('style');
      st.id = STYLE_ID; st.textContent = CSS;
      document.head.appendChild(st);
    }
    state = { el, ctx, chart: null, run: null };

    el.innerHTML = `
<div class="runs-root">
  <section class="panel runs-list">
    <div class="panel-head"><span>Recorded Runs</span><span class="actions"><button class="btn btn-sm btn-ghost" id="runs-refresh" aria-label="Refresh runs">↻</button></span></div>
    <div class="panel-body flush" id="runs-body">
      <div class="skeleton runs-skel"></div>
      <div class="skeleton runs-skel"></div>
      <div class="skeleton runs-skel"></div>
    </div>
  </section>
  <section class="panel runs-detail">
    <div class="panel-head">
      <span>Equity Curve</span>
      <span class="actions runs-picker" id="runs-picker" hidden>
        <label for="runs-agent">Agent</label>
        <select class="select mono" id="runs-agent" aria-label="Select agent"></select>
      </span>
    </div>
    <div class="panel-body" id="runs-detail-body">
      <div class="empty">${ic(ctx, 'database')}<div>Select a run to inspect its persisted equity curves.<br>Every tick, every trade, every ledger — replayable from SQLite.</div></div>
    </div>
  </section>
</div>`;

    const listBody = el.querySelector('#runs-body');
    const detailBody = el.querySelector('#runs-detail-body');
    const pickerWrap = el.querySelector('#runs-picker');
    const pickerSel = el.querySelector('#runs-agent');

    const loadEquity = async (runId, agentId) => {
      detailBody.innerHTML = `<div class="skeleton" style="height:300px"></div>`;
      try {
        const res = await ctx.api.equity(runId, agentId);
        if (!state) return;
        const points = (res && res.points) || [];
        if (!points.length) {
          detailBody.innerHTML = `<div class="empty">${ic(ctx, 'database')}<div>No equity snapshots recorded for <span class="mono">${esc(agentId)}</span> in this run.</div></div>`;
          return;
        }
        const values = points.map((p) => Number(p[1]));
        const start = values[0];
        const end = values[values.length - 1];
        const delta = end - start;
        let peak = -Infinity, maxDD = 0;
        for (const v of values) {
          if (v > peak) peak = v;
          if (peak > 0) maxDD = Math.max(maxDD, (peak - v) / peak);
        }
        const fmt = ctx.fmt;
        const cls = delta > 0 ? 'num-up' : (delta < 0 ? 'num-down' : '');
        detailBody.innerHTML = `
          <div class="runs-stats">
            <div class="card stat"><span class="k">Start Equity</span><span class="v">${fmt.inr(start)}</span></div>
            <div class="card stat"><span class="k">End Equity</span><span class="v">${fmt.inr(end)}</span></div>
            <div class="card stat"><span class="k">Δ P&amp;L</span><span class="v ${cls}">${delta >= 0 ? '+' : ''}${fmt.inr(delta)}</span></div>
            <div class="card stat"><span class="k">Max Drawdown</span><span class="v ${maxDD > 0 ? 'num-down' : ''}">${fmt.num(maxDD * 100, 2)}%</span></div>
            <div class="card stat"><span class="k">Points</span><span class="v">${points.length}</span></div>
          </div>
          <div class="runs-chart" id="runs-chart"></div>`;
        disposeChart(state.chart);
        state.chart = null;
        try {
          const host = el.querySelector('#runs-chart');
          state.chart = ctx.charts.lineChart(host, { color: tokenColor('--accent') });
          setLineData(state.chart, points.map((p) => ({ time: Number(p[0]), value: Number(p[1]) })));
        } catch (e) {
          const host = el.querySelector('#runs-chart');
          if (host) host.innerHTML = `<div class="empty">Chart failed: ${esc((e && e.message) || e)}</div>`;
        }
      } catch (e) {
        if (!state) return;
        detailBody.innerHTML = `<div class="empty">${ic(ctx, 'database')}<div>Failed to load equity.<br><span class="mono">${esc((e && e.message) || e)}</span></div></div>`;
      }
    };

    const selectRun = async (run, tr) => {
      state.run = run;
      listBody.querySelectorAll('tr.sel').forEach((r) => r.classList.remove('sel'));
      if (tr) tr.classList.add('sel');
      pickerWrap.hidden = true;
      detailBody.innerHTML = `<div class="skeleton" style="height:300px"></div>`;
      try {
        const res = await ctx.api.equity(run.run_id);
        if (!state || state.run !== run) return;
        if (res && Array.isArray(res.agents)) {
          if (!res.agents.length) {
            detailBody.innerHTML = `<div class="empty">${ic(ctx, 'database')}<div>No agent snapshots persisted for this run.</div></div>`;
            return;
          }
          const preferred = res.agents.find((a) => String(a).startsWith('STRAT_')) || res.agents[0];
          pickerSel.innerHTML = res.agents.map((a) =>
            `<option value="${esc(a)}" ${a === preferred ? 'selected' : ''}>${esc(a)}</option>`).join('');
          pickerWrap.hidden = false;
          await loadEquity(run.run_id, preferred);
        } else if (res && Array.isArray(res.points)) {
          await loadEquity(run.run_id); // backend defaulted an agent; render what we got
        } else {
          detailBody.innerHTML = `<div class="empty">${ic(ctx, 'database')}<div>Unexpected equity response for this run.</div></div>`;
        }
      } catch (e) {
        if (!state) return;
        detailBody.innerHTML = `<div class="empty">${ic(ctx, 'database')}<div>Failed to load run.<br><span class="mono">${esc((e && e.message) || e)}</span></div></div>`;
      }
    };

    pickerSel.addEventListener('change', () => {
      if (state && state.run) loadEquity(state.run.run_id, pickerSel.value);
    });

    const loadRuns = async () => {
      try {
        const runs = await ctx.api.runs();
        if (!state) return;
        if (!runs || !runs.length) {
          listBody.innerHTML = `<div class="empty">${ic(ctx, 'database')}<div>No runs recorded yet.<br>Start a simulation — every run persists to SQLite automatically.</div></div>`;
          return;
        }
        runs.sort((a, b) => String(b.started_at || '').localeCompare(String(a.started_at || '')));
        listBody.innerHTML = `
          <table class="grid runs-table">
            <thead><tr><th scope="col">Run</th><th scope="col">Started</th><th scope="col">Sym</th><th scope="col">Mode</th><th scope="col">Seed</th></tr></thead>
            <tbody>${runs.map((r, i) => `
              <tr data-i="${i}" tabindex="0" role="button" aria-label="Inspect run ${esc(r.run_id)}">
                <td class="mono" title="${esc(r.run_id)}">${esc(String(r.run_id).slice(0, 18))}</td>
                <td class="mono num-dim">${esc(String(r.started_at || '').replace('T', ' ').slice(0, 19))}</td>
                <td class="mono">${esc(r.symbol)}</td>
                <td><span class="chip ${String(r.mode || '').toLowerCase()}">${esc(r.mode)}</span></td>
                <td class="mono num-dim">${esc(r.seed)}</td>
              </tr>`).join('')}
            </tbody>
          </table>`;
        listBody.querySelectorAll('tbody tr').forEach((tr) => {
          const go = () => selectRun(runs[parseInt(tr.dataset.i, 10)], tr);
          tr.addEventListener('click', go);
          tr.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); go(); } });
        });
      } catch (e) {
        if (!state) return;
        listBody.innerHTML = `<div class="empty">${ic(ctx, 'database')}<div>Failed to load runs.<br><span class="mono">${esc((e && e.message) || e)}</span></div></div>`;
      }
    };

    el.querySelector('#runs-refresh').addEventListener('click', loadRuns);
    await loadRuns();
  },

  unmount() {
    if (!state) return;
    disposeChart(state.chart);
    if (state.el) state.el.innerHTML = '';
    state = null;
  },
};
