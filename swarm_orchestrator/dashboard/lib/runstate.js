/* ============================================================================
   lib/runstate.js — what the main page shows: the run set-up or the live market.

   The engine only knows "running" and "paused". The page adds two states of its own,
   remembered per run (run_id) so a reload keeps them:
     * stopped  — "Stop the market": the run is frozen where it is; the chart, book and
                  tape stay on screen, and only "New run" is offered.
     * new run  — "New run": the run is frozen too, and the right side shows the set-up
                  panel again; the old chart stays until the next run starts.
   A new run_id (the next run's first tick) clears both by itself.
   ============================================================================ */

import { store } from './store.js';
import { api } from './api.js';

const KEY = 'chronos.runstate';
const listeners = new Set();

function load() {
  try { return JSON.parse(localStorage.getItem(KEY) || 'null') || {}; } catch { return {}; }
}
function save(s) {
  try { localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* private window: not remembered */ }
}

let saved = load();                 // {stopped: run_id|null, setup: run_id|null}
const runId = () => (store.latestTick() || {}).run_id || null;

function emit() {
  for (const fn of listeners) {
    try { fn(runstate.get()); } catch (err) { console.error('[runstate] listener error', err); }
  }
}

export const runstate = {
  /**
   * @returns {{hasRun:boolean, view:'setup'|'live', stopped:boolean, paused:boolean, runId:string|null}}
   *   view = 'setup' before any run, or after "New run" until the next run starts.
   */
  get() {
    const id = runId();                 // an idle engine (nothing started yet) ticks without a run_id
    const t = id ? store.latestTick() : null;
    const setup = !t || saved.setup === id;
    return {
      hasRun: !!t,
      view: setup ? 'setup' : 'live',
      stopped: !!id && saved.stopped === id,
      paused: !!(t && t.paused),
      runId: id,
    };
  },

  /** Subscribe to changes of get(); called with the new state. @returns unsubscribe */
  onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); },

  /** "Stop the market": pause the engine and freeze this run (no resume; New run starts the next one). */
  async stop() {
    const id = runId();
    if (!id) return;
    await api.sim.stop();
    saved = { ...saved, stopped: id };
    save(saved);
    emit();
  },

  /** "New run": freeze the run where it is and show the set-up panel; the chart stays until the next run. */
  async newRun() {
    const id = runId();
    if (id) {
      try { await api.sim.stop(); } catch { /* already paused or engine idle: the set-up still opens */ }
    }
    saved = { ...saved, stopped: id, setup: id };
    save(saved);
    emit();
  },

  /** Leave the set-up panel and go back to the (frozen) market without starting anything. */
  backToMarket() {
    saved = { ...saved, setup: null };
    save(saved);
    emit();
  },
};

// The next run's first tick (a new run_id) makes the page live again; pause/resume changes are passed on too.
let lastKey = '';
store.onTick((t) => {
  const key = `${t.run_id}|${!!t.paused}`;
  if (key !== lastKey) { lastKey = key; emit(); }
});

export default runstate;
