/* ============================================================================
   lib/api.js — every Chronos REST endpoint as a promise method.
   Contracts: PROTOCOL.md §6 (bridge) + PROTOCOL_V2.md §1–§6 (feature layer).
   Throws ApiError (status + server detail) on any !res.ok.
   ============================================================================ */

/** Error carrying the HTTP status and the server's `detail` message. */
export class ApiError extends Error {
  /**
   * @param {number} status HTTP status code (0 = network failure)
   * @param {string} detail server-provided detail or statusText
   * @param {string} path   request path, for debugging
   */
  constructor(status, detail, path) {
    super(detail || `HTTP ${status}`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
    this.path = path;
  }
}

/**
 * Core fetch wrapper. GET when body is undefined, otherwise JSON body.
 * @returns {Promise<any>} parsed JSON (or null for empty responses)
 */
async function req(path, { method = 'GET', body } = {}) {
  let res;
  try {
    res = await fetch(path, {
      method,
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    throw new ApiError(0, `network error: ${err.message}`, path);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      if (data && data.detail) detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    } catch { /* non-JSON error body — keep statusText */ }
    throw new ApiError(res.status, detail, path);
  }
  const text = await res.text();
  return text ? JSON.parse(text) : null;
}

const get = (path) => req(path);
const post = (path, body = {}) => req(path, { method: 'POST', body });
const put = (path, body = {}) => req(path, { method: 'PUT', body });
const del = (path) => req(path, { method: 'DELETE' });
const enc = encodeURIComponent;

export const api = {
  /* ---- PROTOCOL §6 — simulation control ---- */
  sim: {
    /** POST /api/sim/start {symbol, sector, price, seed?} → {status, run_id} */
    start: (cfg) => post('/api/sim/start', cfg),
    /** POST /api/sim/stop (PAUSE) → {status} */
    stop: () => post('/api/sim/stop', {}),
    /** POST /api/sim/resume → {status} */
    resume: () => post('/api/sim/resume', {}),
  },

  /** POST /api/news {headline} → {status:"accepted"}; score arrives on the event topic. */
  news: (headline) => post('/api/news', { headline }),

  /** POST /api/replay/start {csv_path, symbol?, sector?, seed?} → {status, run_id, bars} */
  replay: (cfg) => post('/api/replay/start', cfg),

  /** GET /api/health → {status, engine} */
  health: () => get('/api/health'),

  /* ---- PROTOCOL §6 — sandboxed strategies ---- */
  strategy: {
    /** POST /api/strategy/run {name, code} → {strategy_id, agent_id, backend?} */
    run: (name, code) => post('/api/strategy/run', { name, code }),
    /** POST /api/strategy/stop {strategy_id} → {status} */
    stop: (id) => post('/api/strategy/stop', { strategy_id: id }),
    /** GET /api/strategy/list → [{strategy_id, agent_id, name, status, started_at}] */
    list: () => get('/api/strategy/list'),
    /** GET /api/strategy/{id}/logs → {logs} */
    logs: (id) => get(`/api/strategy/${enc(id)}/logs`),
  },

  /* ---- PROTOCOL_V2 §1 — script library ---- */
  scripts: {
    /** GET /api/scripts → [{script_id, name, language, saved_at, updated_at, size}] */
    list: () => get('/api/scripts'),
    /** GET /api/scripts/{id} → {script_id, name, code, ...} */
    get: (id) => get(`/api/scripts/${enc(id)}`),
    /** POST /api/scripts {name, code} → saved meta */
    save: (name, code) => post('/api/scripts', { name, code }),
    /** PUT /api/scripts/{id} {name?, code?} → updated meta */
    update: (id, p) => put(`/api/scripts/${enc(id)}`, p),
    /** DELETE /api/scripts/{id} → {status:"deleted"} */
    del: (id) => del(`/api/scripts/${enc(id)}`),
    /** POST /api/scripts/{id}/run → {strategy_id, agent_id, backend, script_id} */
    run: (id) => post(`/api/scripts/${enc(id)}/run`, {}),
  },

  /* ---- PROTOCOL_V2 §2 — results / reports ---- */
  results: {
    /** GET /api/results → strategy summaries */
    list: () => get('/api/results'),
    /** GET /api/results/{sid} → full strategy_detail */
    detail: (id) => get(`/api/results/${enc(id)}`),
    /** GET /api/results/{sid}/trades[?day=N] → {trades:[...]} */
    trades: (id, day) => get(`/api/results/${enc(id)}/trades${day != null ? `?day=${enc(day)}` : ''}`),
    /** Plain URL string — trades CSV download (streams from the bridge). */
    csvUrl: (id) => `/api/results/${enc(id)}/trades.csv`,
    /** Plain URL string — per-day outcomes CSV download. */
    outcomesCsvUrl: (id) => `/api/results/${enc(id)}/outcomes.csv`,
    /** Plain URL string — PDF report download. */
    pdfUrl: (id) => `/api/results/${enc(id)}/report.pdf`,
  },

  /* ---- PROTOCOL_V2 §3 — Monte-Carlo counterfactual fan ---- */
  /** POST /api/montecarlo {symbol, price, n_paths?, horizon?, seed_base?, shock?} → fan dict */
  montecarlo: (cfg) => post('/api/montecarlo', cfg),

  /* ---- PROTOCOL_V2 §4 — copilot ---- */
  /** POST /api/copilot {description} → {code, explanation, source} */
  copilot: (description) => post('/api/copilot', { description }),

  /* ---- PROTOCOL_V2 §5 — time machine ---- */
  timemachine: {
    /** GET /api/timemachine/scenarios → [{id, name, description, symbol, sector, bars, tag}] */
    scenarios: () => get('/api/timemachine/scenarios'),
    /** POST /api/timemachine/start {scenario_id, seed?} → {status, run_id, bars, scenario} */
    start: (id, seed) => post('/api/timemachine/start', seed != null ? { scenario_id: id, seed } : { scenario_id: id }),
  },

  /* ---- PROTOCOL_V2 §6 — narration ---- */
  /** POST /api/narrate {} → {narration, source} */
  narrate: () => post('/api/narrate', {}),

  /* ---- PROTOCOL §6 — persistence explorer ---- */
  /** GET /api/runs → [{run_id, started_at, symbol, mode, seed}] */
  runs: () => get('/api/runs'),
  /** GET /api/runs/{run_id}/equity?agent_id=X → {points: [[seq, equity], ...]} */
  equity: (runId, agentId) =>
    get(`/api/runs/${enc(runId)}/equity${agentId ? `?agent_id=${enc(agentId)}` : ''}`),
};

export default api;
