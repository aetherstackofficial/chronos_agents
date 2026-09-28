/* ============================================================================
   views/strategies.js — Monaco editor + script library + running strategies.
   Pillar-2 front door: write / save / run sandboxed strategies (SDK §10).
   ============================================================================ */

const SDK_TEMPLATE = `from runner.sdk import Strategy

class UserStrategy(Strategy):
    """Buy weakness below VWAP, exit strength above it. Runs as agent STRAT_xxxx."""

    def on_start(self, config):
        self.inventory = 0

    def on_tick(self, state):
        px, vwap, rsi = state["last_price"], state["vwap"], state["rsi"]
        if px < vwap * 0.998 and rsi < 35 and self.inventory < 500:
            return [{"action": "BUY", "type": "MARKET", "qty": 50}]
        if self.inventory > 0 and (px > vwap * 1.002 or rsi > 65):
            return [{"action": "SELL", "type": "MARKET", "qty": self.inventory}]
        return []

    def on_fill(self, ack):
        self.inventory = ack["pos"]
`;

const COPILOT_KEY = 'chronos.copilot.code';
const STRAT_POLL_MS = 3000;
const LOGS_POLL_MS = 2000;

let monacoPromise = null;
function loadMonaco() {
  if (monacoPromise) return monacoPromise;
  monacoPromise = new Promise((resolve) => {
    if (!window.require || !window.require.config) { resolve(null); return; }
    try {
      window.require.config({ paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.0/min/vs' } });
      window.require(['vs/editor/editor.main'], () => resolve(window.monaco || null), () => resolve(null));
    } catch { resolve(null); }
  });
  return monacoPromise;
}

export default {
  id: 'strategies',
  title: 'Strategies',
  icon: 'code',
  group: 'Trading',

  _splits: [],
  _timers: [],
  _editor: null,
  _fallback: null,
  _els: {},
  _scriptId: null,

  async mount(el, ctx) {
    const { api, fmt, ui, splitter } = ctx;

    el.innerHTML = `
      <div class="view-root"><div class="view-cols" data-role="cols">

        <section class="panel" aria-label="Strategy editor">
          <div class="panel-head"><span>Strategy Editor — Python</span>
            <span class="actions">
              <span class="chip" data-role="loaded-chip" hidden>editing <b data-role="loaded-name"></b></span>
              <span class="chip" data-role="backend-chip" hidden></span>
            </span>
          </div>
          <div class="editor-toolbar">
            <input class="input mono" data-role="name" value="mean_rev" aria-label="Strategy name" maxlength="48">
            <button type="button" class="btn btn-sm" data-role="save">Save to Library</button>
            <button type="button" class="btn btn-primary btn-sm" data-role="saverun">Save &amp; Run</button>
            <button type="button" class="btn btn-sm" data-role="run" title="Run without saving">${ui.icon('play')} Run</button>
            <button type="button" class="btn btn-ghost btn-sm" data-role="new" title="Reset editor to the SDK template">${ui.icon('plus')} New</button>
          </div>
          <div class="monaco-host" data-role="editor" aria-label="Python strategy code editor"></div>
        </section>

        <div class="view-rows" data-role="side">
          <section class="panel" aria-label="Script library">
            <div class="panel-head"><span>Script Library</span>
              <span class="actions"><button type="button" class="btn btn-ghost btn-icon btn-sm" data-role="lib-refresh" aria-label="Refresh library">${ui.icon('refresh')}</button></span>
            </div>
            <div class="panel-body flush scroll-y" data-role="lib"></div>
          </section>
          <section class="panel" aria-label="Running strategies">
            <div class="panel-head"><span>Running Strategies</span></div>
            <div class="panel-body flush scroll-y" data-role="running"></div>
          </section>
        </div>

      </div></div>`;

    const q = (r) => el.querySelector(`[data-role="${r}"]`);
    this._els = {
      name: q('name'), editor: q('editor'), lib: q('lib'), running: q('running'),
      backendChip: q('backend-chip'), loadedChip: q('loaded-chip'), loadedName: q('loaded-name'),
    };
    this._scriptId = null;

    this._splits.push(splitter.hSplit(q('cols'), {
      sizes: [62, 38], min: [380, 300], storageKey: 'chronos.split.strategies',
    }));
    this._splits.push(splitter.vSplit(q('side'), {
      sizes: [52, 48], min: [150, 150], storageKey: 'chronos.split.strategies.side',
    }));

    /* ---- editor (Monaco, textarea fallback) ---- */
    let initial = SDK_TEMPLATE;
    const handoff = sessionStorage.getItem(COPILOT_KEY);
    if (handoff) {
      initial = handoff;
      sessionStorage.removeItem(COPILOT_KEY);
      ui.toast('Copilot code loaded into the editor', 'good');
    }

    this._els.editor.innerHTML = '<div class="skeleton skel-block" aria-hidden="true"></div>';
    const monaco = await loadMonaco();
    if (!this._els.editor.isConnected) return; // view switched while Monaco loaded
    this._els.editor.innerHTML = '';
    if (monaco) {
      this._editor = monaco.editor.create(this._els.editor, {
        value: initial,
        language: 'python',
        theme: 'vs-dark',
        fontSize: 13,
        fontFamily: getComputedStyle(document.documentElement).getPropertyValue('--mono').trim() || 'JetBrains Mono, monospace',
        minimap: { enabled: false },
        automaticLayout: true,
        scrollBeyondLastLine: false,
        padding: { top: 12 },
        renderLineHighlight: 'gutter',
        tabSize: 4,
      });
    } else {
      this._fallback = document.createElement('textarea');
      this._fallback.className = 'input code monaco-fallback';
      this._fallback.value = initial;
      this._fallback.spellcheck = false;
      this._fallback.setAttribute('aria-label', 'Python strategy code');
      this._els.editor.appendChild(this._fallback);
      ui.toast('Monaco unavailable — using a plain editor', 'warn');
    }

    const getCode = () => (this._editor ? this._editor.getValue() : this._fallback ? this._fallback.value : '');
    const setCode = (v) => { if (this._editor) this._editor.setValue(v); else if (this._fallback) this._fallback.value = v; };
    const setLoaded = (id, name) => {
      this._scriptId = id;
      this._els.loadedChip.hidden = !id;
      if (name) this._els.loadedName.textContent = name;
    };
    const showBackend = (backend) => {
      if (!backend) return;
      this._els.backendChip.hidden = false;
      this._els.backendChip.textContent = backend;
      this._els.backendChip.title = backend === 'docker'
        ? 'Docker sandbox: resource-limited container'
        : 'Subprocess sandbox: psutil watchdog — honest note: not a security boundary';
    };

    /* ---- save / run actions ---- */
    const saveToLibrary = async (btn) => {
      const name = this._els.name.value.trim() || 'untitled';
      const code = getCode();
      if (!code.trim()) { ui.toast('The strategy is empty', 'warn'); return null; }
      try {
        const saved = await ui.busy(btn, () => (this._scriptId
          ? api.scripts.update(this._scriptId, { name, code })
          : api.scripts.save(name, code)));
        setLoaded(saved.script_id, saved.name);
        ui.toast(`Saved “${saved.name}” to the library`, 'good');
        renderLibrary();
        return saved;
      } catch (err) {
        ui.toast(`Save failed: ${err.detail || err.message}`, 'bad');
        return null;
      }
    };

    q('save').addEventListener('click', (e) => saveToLibrary(e.currentTarget));

    q('saverun').addEventListener('click', async (e) => {
      const saved = await saveToLibrary(e.currentTarget);
      if (!saved) return;
      try {
        const res = await api.scripts.run(saved.script_id);
        showBackend(res.backend);
        ui.toast(`Running as ${res.agent_id} — results are being recorded`, 'good');
        renderRunning();
      } catch (err) { ui.toast(`Run failed: ${err.detail || err.message}`, 'bad'); }
    });

    q('run').addEventListener('click', async (e) => {
      const code = getCode();
      if (!code.trim()) { ui.toast('The strategy is empty', 'warn'); return; }
      try {
        const res = await ui.busy(e.currentTarget, () => api.strategy.run(this._els.name.value.trim() || 'untitled', code));
        showBackend(res.backend);
        ui.toast(`Running as ${res.agent_id}`, 'good');
        renderRunning();
      } catch (err) { ui.toast(`Run failed: ${err.detail || err.message}`, 'bad'); }
    });

    q('new').addEventListener('click', () => {
      setCode(SDK_TEMPLATE);
      setLoaded(null);
      this._els.name.value = 'mean_rev';
    });

    /* ---- script library ---- */
    const renderLibrary = async () => {
      const host = this._els.lib;
      host.innerHTML = '<div class="skeleton skel-line"></div><div class="skeleton skel-line"></div>';
      let list;
      try { list = await api.scripts.list(); } catch (err) {
        host.innerHTML = `<div class="empty">${ui.icon('database')}<div>Library unavailable<br><span class="dim">${err.detail || err.message}</span></div></div>`;
        return;
      }
      if (!host.isConnected) return;
      if (!Array.isArray(list) || !list.length) {
        host.innerHTML = `<div class="empty">${ui.icon('database')}<div>No saved scripts yet.<br><span class="dim">Save the editor's strategy to build your library.</span></div></div>`;
        return;
      }
      host.innerHTML = list.map((s) => `
        <div class="lib-item" data-id="${s.script_id}">
          <span class="lname" title="${esc(s.name)}">${esc(s.name)}</span>
          <span class="lmeta">${(s.size != null) ? `${fmt.num(s.size, 0)} B` : ''}</span>
          <span class="actions">
            <button type="button" class="btn btn-ghost btn-icon btn-sm" data-act="load" aria-label="Load ${esc(s.name)} into the editor" title="Load">${ui.icon('code')}</button>
            <button type="button" class="btn btn-ghost btn-icon btn-sm" data-act="runscript" aria-label="Run ${esc(s.name)}" title="Run">${ui.icon('play')}</button>
            <button type="button" class="btn btn-ghost btn-icon btn-sm" data-act="rename" aria-label="Rename ${esc(s.name)}" title="Rename">${ui.icon('cog')}</button>
            <button type="button" class="btn btn-ghost btn-icon btn-sm" data-act="del" aria-label="Delete ${esc(s.name)}" title="Delete">${ui.icon('trash')}</button>
          </span>
        </div>`).join('');
    };

    this._els.lib.addEventListener('click', async (e) => {
      const btn = e.target.closest('button[data-act]');
      if (!btn) return;
      const id = btn.closest('.lib-item').dataset.id;
      const act = btn.dataset.act;
      try {
        if (act === 'load') {
          const s = await ui.busy(btn, () => api.scripts.get(id));
          setCode(s.code || '');
          this._els.name.value = s.name || 'untitled';
          setLoaded(s.script_id, s.name);
          ui.toast(`Loaded “${s.name}”`, 'good');
        } else if (act === 'runscript') {
          const res = await ui.busy(btn, () => api.scripts.run(id));
          showBackend(res.backend);
          ui.toast(`Running as ${res.agent_id} — results are being recorded`, 'good');
          renderRunning();
        } else if (act === 'rename') {
          const m = ui.modal({
            title: 'Rename Script',
            body: '<label class="field">New name<input id="rn-name" class="input"></label>',
            actions: [
              { label: 'Cancel' },
              {
                label: 'Rename',
                kind: 'primary',
                onClick: async (close, b) => {
                  const name = m.body.querySelector('#rn-name').value.trim();
                  if (!name) { ui.toast('Name cannot be empty', 'warn'); return; }
                  try {
                    await ui.busy(b, () => api.scripts.update(id, { name }));
                    ui.toast('Renamed', 'good');
                    close();
                    renderLibrary();
                    if (this._scriptId === id) setLoaded(id, name);
                  } catch (err) { ui.toast(`Rename failed: ${err.detail || err.message}`, 'bad'); }
                },
              },
            ],
          });
        } else if (act === 'del') {
          const ok = await ui.confirm('Delete this script from the library? The file is removed.');
          if (!ok) return;
          await api.scripts.del(id);
          ui.toast('Script deleted', 'good');
          if (this._scriptId === id) setLoaded(null);
          renderLibrary();
        }
      } catch (err) {
        ui.toast(`${act} failed: ${err.detail || err.message}`, 'bad');
      }
    });

    q('lib-refresh').addEventListener('click', renderLibrary);

    /* ---- running strategies (3 s poll) ---- */
    const renderRunning = async () => {
      let list;
      try { list = await api.strategy.list(); } catch { return; }
      const host = this._els.running;
      if (!host || !host.isConnected) return;
      if (!Array.isArray(list) || !list.length) {
        host.innerHTML = `<div class="empty">${ui.icon('agent')}<div>Nothing running.<br><span class="dim">Launched strategies trade as first-class agents.</span></div></div>`;
        return;
      }
      host.innerHTML = `<table class="grid" aria-label="Running strategies">
        <thead><tr><th>Name</th><th>Agent</th><th>Status</th><th class="right"></th></tr></thead>
        <tbody>${list.map((s) => `
          <tr data-id="${s.strategy_id}">
            <td title="${esc(s.name)}">${esc(s.name)}</td>
            <td class="mono">${esc(s.agent_id || '')}</td>
            <td><span class="chip ${s.status === 'running' ? 'up' : ''}">${esc(s.status || '')}</span></td>
            <td class="right">
              <button type="button" class="btn btn-ghost btn-icon btn-sm" data-act="logs" aria-label="View logs for ${esc(s.name)}" title="Logs">${ui.icon('logs')}</button>
              <button type="button" class="btn btn-ghost btn-icon btn-sm" data-act="stop" aria-label="Stop ${esc(s.name)}" title="Stop" ${s.status !== 'running' ? 'disabled' : ''}>${ui.icon('pause')}</button>
            </td>
          </tr>`).join('')}</tbody></table>`;
    };

    this._els.running.addEventListener('click', async (e) => {
      const btn = e.target.closest('button[data-act]');
      if (!btn) return;
      const id = btn.closest('tr').dataset.id;
      if (btn.dataset.act === 'stop') {
        try {
          await ui.busy(btn, () => api.strategy.stop(id));
          ui.toast('Strategy stopped (CANCEL_ALL sent)', 'good');
          renderRunning();
        } catch (err) { ui.toast(`Stop failed: ${err.detail || err.message}`, 'bad'); }
      } else if (btn.dataset.act === 'logs') {
        openLogs(id);
      }
    });

    const openLogs = (id) => {
      const d = ui.drawer({
        title: `Logs — ${id}`,
        body: '<pre class="logs-pre" data-role="logs">loading…</pre>',
      });
      const pre = d.body.querySelector('[data-role="logs"]');
      const pull = async () => {
        try {
          const res = await api.strategy.logs(id);
          const stick = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
          pre.textContent = res.logs || '(no output yet)';
          if (stick) pre.scrollTop = pre.scrollHeight;
        } catch (err) { pre.textContent = `logs unavailable: ${err.detail || err.message}`; }
      };
      pull();
      const t = setInterval(pull, LOGS_POLL_MS);
      this._timers.push(t);
      d.onClose(() => clearInterval(t));
    };

    renderLibrary();
    renderRunning();
    this._timers.push(setInterval(renderRunning, STRAT_POLL_MS));
  },

  unmount() {
    this._timers.forEach(clearInterval);
    this._timers = [];
    this._splits.forEach((s) => s.destroy());
    this._splits = [];
    if (this._editor) { this._editor.dispose(); this._editor = null; }
    this._fallback = null;
    this._els = {};
  },
};

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
