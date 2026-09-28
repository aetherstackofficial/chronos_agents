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
};

/** Ring buffer of the last TRADE_RING trade payloads (§5.2), oldest first. */
const trades = [];

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
