/* ============================================================================
   lib/store.js — the single live-data connection for every Chronos page.
   One WebSocket to /ws (same origin, ws/wss auto), auto-reconnect with a 2 s
   backoff, parsing the bridge's {topic, data} envelope (PROTOCOL §6) into the
   tick / trade / event topics of §5. PROTOCOL_V2 §7 is the exact public API.
   ============================================================================ */

const RECONNECT_MS = 2000;
const TRADE_RING = 200;

/** Per-topic subscriber sets. 'status' fires on connection-state changes. */
const listeners = {
  tick: new Set(),
  trade: new Set(),
  event: new Set(),
  status: new Set(),
  reset: new Set(),   // a different run started: history was cleared
};

/** Ring buffer of the last TRADE_RING trade payloads (§5.2), oldest first. */
const trades = [];

/**
 * Candle history of the CURRENT run, oldest first. Views are destroyed when you
 * switch pages, so the chart cannot keep its own history: it is kept here (the
 * socket and this store live for the whole session) and the Market view redraws
 * from it when it is shown again. Cleared only when a new run starts.
 */
const BAR_CAP = 6000;
const bars = [];
let runId = null;

let ws = null;
let reconnectTimer = null;
let closedByPage = false;

/**
 * Register `fn` on a listener set.
 * @returns {() => void} unsubscribe function
 */
function subscribe(set, fn) {
  set.add(fn);
  return () => set.delete(fn);
}

/** Invoke every subscriber, isolating callback errors from the socket loop. */
function emit(set, payload) {
  for (const fn of set) {
    try { fn(payload); } catch (err) { console.error('[store] subscriber error', err); }
  }
}

/**
 * The live-market store singleton (PROTOCOL_V2 §7).
 * `state` is mutated in place — cheap to read from any render loop.
 */
export const store = {
  /** Live state, updated in place on every tick / status change. */
  state: {
    /** @type {object|null} last full §5.1 tick payload */
    tick: null,
    /** @type {boolean} WebSocket connection state */
    connected: false,
    /** @type {string|null} "LIVE" | "REPLAY" | "REACTIVE" */
    mode: null,
    /** @type {number|null} sim day counter */
    day: null,
    /** @type {number|null} last trade price */
    price: null,
    /** @type {string|null} "BULL" | "BEAR" | "RANGING" */
    regime: null,
  },

  /** Subscribe to §5.1 tick snapshots. @returns {() => void} unsubscribe */
  onTick(fn) { return subscribe(listeners.tick, fn); },

  /** Subscribe to §5.2 trade prints. @returns {() => void} unsubscribe */
  onTrade(fn) { return subscribe(listeners.trade, fn); },

  /** Subscribe to §5.3 events (news / mode_change / liquidation / session). */
  onEvent(fn) { return subscribe(listeners.event, fn); },

  /**
   * Subscribe to connection-state changes; called with `true|false`.
   * The current state is replayed immediately so UIs can initialize.
   */
  onStatus(fn) {
    const unsub = subscribe(listeners.status, fn);
    try { fn(store.state.connected); } catch (err) { console.error('[store] status subscriber error', err); }
    return unsub;
  },

  /**
   * Subscribe to "a different run started" (new START, replay, stock or reset).
   * Fired before that run's first tick; the candle history and tape are already cleared.
   * @returns {() => void} unsubscribe
   */
  onReset(fn) { return subscribe(listeners.reset, fn); },

  /**
   * @returns {{time:number, open:number, high:number, low:number, close:number, volume:number}[]}
   *   every candle of the current run so far, oldest first (a copy)
   */
  bars() { return bars.slice(); },

  /** @returns {object|null} the most recent tick payload, or null before START */
  latestTick() { return store.state.tick; },

  /**
   * @param {number} [n=200] how many trades to return (max 200 retained)
   * @returns {object[]} the last n trade payloads, oldest first
   */
  recentTrades(n = TRADE_RING) { return trades.slice(-n); },
};

/**
 * Topic dispatcher — the ONLY writer of market state.
 * Applies a parsed {topic, data} envelope to `store.state`, maintains the
 * trade ring buffer, then fans the payload out to topic subscribers.
 * @param {string} topic "tick" | "trade" | "event"
 * @param {object} data  the §5 payload for that topic
 */
function dispatch(topic, data) {
  if (topic === 'tick') {
    if (data.run_id && data.run_id !== runId) {
      // A new run (new stock / START / replay / reset): drop the old run's candles and prints.
      const hadRun = runId !== null;
      runId = data.run_id;
      if (hadRun) {
        bars.length = 0;
        trades.length = 0;
        emit(listeners.reset, data);
      }
    }
    recordBar(data);
    store.state.tick = data;
    store.state.mode = data.mode ?? store.state.mode;
    store.state.day = data.day_count ?? store.state.day;
    store.state.price = data.last_price ?? store.state.price;
    store.state.regime = data.regime ?? store.state.regime;
    emit(listeners.tick, data);
  } else if (topic === 'trade') {
    trades.push(data);
    if (trades.length > TRADE_RING) trades.splice(0, trades.length - TRADE_RING);
    emit(listeners.trade, data);
  } else if (topic === 'event') {
    if (data.kind === 'mode_change' && data.mode) store.state.mode = data.mode;
    emit(listeners.event, data);
  }
}

/**
 * Add the tick's closed bar to the history: same minute replaces (a paused engine
 * re-sends its last bar), a later minute appends, an older one is ignored so the
 * series stays strictly ascending for the chart.
 */
function recordBar(tick) {
  if (!tick.bar || !Number.isFinite(tick.unix_time)) return;
  const time = tick.unix_time - (tick.unix_time % 60);
  const b = { time, open: tick.bar.o, high: tick.bar.h, low: tick.bar.l, close: tick.bar.c, volume: tick.bar.v };
  const last = bars[bars.length - 1];
  if (last && last.time === time) bars[bars.length - 1] = b;
  else if (!last || time > last.time) {
    bars.push(b);
    if (bars.length > BAR_CAP) bars.splice(0, bars.length - BAR_CAP);
  }
}

/** Update connection state and broadcast it if it changed. */
function setConnected(connected) {
  if (store.state.connected === connected) return;
  store.state.connected = connected;
  emit(listeners.status, connected);
}

/** Open the socket; schedule a reconnect on any close/error (2 s backoff). */
function connect() {
  const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
  try {
    ws = new WebSocket(`${proto}${location.host}/ws`);
  } catch (err) {
    console.error('[store] websocket open failed', err);
    scheduleReconnect();
    return;
  }

  ws.onopen = () => setConnected(true);

  ws.onmessage = (msg) => {
    let envelope;
    try { envelope = JSON.parse(msg.data); } catch { return; }
    if (envelope && envelope.topic && envelope.data) dispatch(envelope.topic, envelope.data);
  };

  ws.onclose = () => {
    setConnected(false);
    if (!closedByPage) scheduleReconnect();
  };

  ws.onerror = () => { try { ws.close(); } catch { /* already closing */ } };
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => { reconnectTimer = null; connect(); }, RECONNECT_MS);
}

window.addEventListener('beforeunload', () => {
  closedByPage = true;
  try { ws && ws.close(); } catch { /* ignore */ }
});

connect();

export default store;
