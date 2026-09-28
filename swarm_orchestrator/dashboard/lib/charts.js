/* ============================================================================
   lib/charts.js — lightweight-charts@4.2.0 wrappers themed to design tokens.
   Dark surfaces, recessive grid, no scale borders, one value axis per chart.
   Also exports depthBars — a pure-DOM order-book depth renderer.
   ============================================================================ */

/** Read a design token off :root, with a fallback for early paints. */
function tok(name, fallback = '') {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function hexToRgb(hex) {
  const h = hex.replace('#', '');
  const s = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  const n = parseInt(s, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Opaque blend of `fg` over `bg` at alpha `t` — for stacked fan bands. */
function mix(fg, bg, t) {
  const a = hexToRgb(fg);
  const b = hexToRgb(bg);
  const c = a.map((v, i) => Math.round(v * t + b[i] * (1 - t)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function rgba(hex, a) {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a})`;
}

function lc() { return window.LightweightCharts || null; }

/** Shared chart options — the single source of chart theming. */
function baseOptions(extra = {}) {
  const LC = lc();
  return {
    autoSize: true,
    layout: {
      background: { type: 'solid', color: 'transparent' },
      textColor: tok('--text-mut', '#7c889b'),
      fontFamily: tok('--mono', 'JetBrains Mono, monospace'),
      fontSize: 11,
      attributionLogo: false,
    },
    grid: {
      vertLines: { color: tok('--border', '#1e2637') },
      horzLines: { color: tok('--border', '#1e2637') },
    },
    rightPriceScale: { borderVisible: false },
    timeScale: { borderVisible: false, timeVisible: true, secondsVisible: false, rightOffset: 3 },
    crosshair: {
      mode: LC.CrosshairMode.Normal,
      vertLine: { color: rgba(tok('--accent-2', '#00d1ff'), 0.5), width: 1, style: LC.LineStyle.Dotted, labelBackgroundColor: tok('--bg-3', '#161d2e') },
      horzLine: { color: rgba(tok('--accent-2', '#00d1ff'), 0.5), width: 1, style: LC.LineStyle.Dotted, labelBackgroundColor: tok('--bg-3', '#161d2e') },
    },
    ...extra,
  };
}

/** Render a graceful in-panel notice when the charts CDN is unreachable. */
function chartUnavailable(el) {
  el.innerHTML = '<div class="chart-offline">chart library unavailable — check network and reload</div>';
  return {
    update() {}, setData() {}, dispose() { el.innerHTML = ''; }, crosshairReadout() {}, chart: null,
  };
}

/* ------------------------------ candleChart ------------------------------ */

/**
 * Candle + volume chart (volume rides a hidden overlay scale → one value axis).
 * Accepts bars as {time, o,h,l,c,v} or {time, open,high,low,close,volume}.
 * @returns {{update(bar):void, setData(bars):void, dispose():void,
 *            crosshairReadout(fn):void, chart:object}}
 */
export function candleChart(el) {
  const LC = lc();
  if (!LC) return chartUnavailable(el);

  const up = tok('--up', '#22c98a');
  const down = tok('--down', '#ff5470');
  const chart = LC.createChart(el, baseOptions());

  const candles = chart.addCandlestickSeries({
    upColor: up, downColor: down, borderVisible: false,
    wickUpColor: up, wickDownColor: down,
    priceFormat: { type: 'price', precision: 2, minMove: 0.05 },
  });
  candles.priceScale().applyOptions({ scaleMargins: { top: 0.06, bottom: 0.26 } });

  const vol = chart.addHistogramSeries({
    priceScaleId: 'vol',
    priceFormat: { type: 'volume' },
    lastValueVisible: false,
    priceLineVisible: false,
  });
  chart.priceScale('vol').applyOptions({ visible: false, scaleMargins: { top: 0.8, bottom: 0 } });

  let lastBar = null;
  const norm = (b) => ({
    time: b.time,
    open: b.o ?? b.open, high: b.h ?? b.high, low: b.l ?? b.low, close: b.c ?? b.close,
    volume: b.v ?? b.volume ?? 0,
  });
  const volPoint = (b) => ({
    time: b.time, value: b.volume,
    color: rgba(b.close >= b.open ? up : down, 0.42),
  });

  return {
    chart,
    update(bar) {
      const b = norm(bar);
      lastBar = b;
      candles.update({ time: b.time, open: b.open, high: b.high, low: b.low, close: b.close });
      vol.update(volPoint(b));
    },
    setData(bars) {
      const bs = bars.map(norm);
      lastBar = bs[bs.length - 1] || null;
      candles.setData(bs.map((b) => ({ time: b.time, open: b.open, high: b.high, low: b.low, close: b.close })));
      vol.setData(bs.map(volPoint));
      chart.timeScale().fitContent();
    },
    /** fn(bar|null): hovered OHLCV, or null when the pointer leaves the plot. */
    crosshairReadout(fn) {
      chart.subscribeCrosshairMove((param) => {
        if (!param || param.time === undefined) { fn(null); return; }
        const c = param.seriesData.get(candles);
        const v = param.seriesData.get(vol);
        fn(c ? { time: param.time, open: c.open, high: c.high, low: c.low, close: c.close, volume: v ? v.value : null } : null);
      });
    },
    dispose() { chart.remove(); },
  };
}

/* ------------------------------- lineChart ------------------------------- */

/**
 * Single-series line (area-glow) chart.
 * @param {HTMLElement} el
 * @param {{color?:string, tickFormatter?:(t:number)=>string, timeVisible?:boolean,
 *          priceFormatter?:(v:number)=>string}} [opts]
 */
export function lineChart(el, opts = {}) {
  const LC = lc();
  if (!LC) return chartUnavailable(el);
  const color = opts.color || tok('--accent', '#3d7bff');

  const options = baseOptions();
  if (opts.tickFormatter) {
    options.timeScale = { ...options.timeScale, timeVisible: false, tickMarkFormatter: opts.tickFormatter };
    options.localization = { timeFormatter: opts.tickFormatter };
  }
  if (opts.timeVisible === false) options.timeScale = { ...options.timeScale, timeVisible: false };
  const chart = LC.createChart(el, options);

  const series = chart.addAreaSeries({
    lineColor: color, lineWidth: 2,
    topColor: rgba(color, 0.16), bottomColor: rgba(color, 0.0),
    priceLineVisible: false,
    priceFormat: opts.priceFormatter
      ? { type: 'custom', formatter: opts.priceFormatter, minMove: 0.01 }
      : { type: 'price', precision: 2, minMove: 0.01 },
  });
  series.priceScale().applyOptions({ scaleMargins: { top: 0.12, bottom: 0.1 } });

  let last = null;
  return {
    chart,
    update(point) { last = point; series.update(point); },
    setData(points) { last = points[points.length - 1] || null; series.setData(points); chart.timeScale().fitContent(); },
    /** fn(point|null): hovered {time,value}, or null off-plot. */
    crosshairReadout(fn) {
      chart.subscribeCrosshairMove((param) => {
        if (!param || param.time === undefined) { fn(null); return; }
        const p = param.seriesData.get(series);
        fn(p ? { time: param.time, value: p.value } : null);
      });
    },
    dispose() { chart.remove(); },
  };
}

/* -------------------------------- fanChart ------------------------------- */

/**
 * Monte-Carlo percentile fan: p5–p95 light band, p25–p75 darker band,
 * p50 median accent line, plus thin muted sample paths.
 * Bands are painted with OPAQUE blends layered top-down (p95→p75→p25→p5),
 * so each area fill "cuts" the one beneath it — no additive transparency mud.
 * @param {HTMLElement} el
 * @param {{percentiles:{p5:number[],p25:number[],p50:number[],p75:number[],p95:number[]},
 *          sample_paths?:number[][]}} fanData
 * @returns {{dispose():void, chart:object}}
 */
export function fanChart(el, fanData) {
  const LC = lc();
  if (!LC) return chartUnavailable(el);

  const accent = tok('--accent', '#3d7bff');
  const bg = tok('--bg-1', '#0b0f17');
  const bandLight = mix(accent, bg, 0.1);
  const bandDark = mix(accent, bg, 0.22);

  const options = baseOptions();
  options.timeScale = {
    ...options.timeScale, timeVisible: false,
    tickMarkFormatter: (t) => `${t}m`,
  };
  options.localization = { timeFormatter: (t) => `minute ${t}` };
  const chart = LC.createChart(el, options);

  const pts = (arr) => arr.map((v, i) => ({ time: i, value: v }));
  const band = (data, fill) => {
    const s = chart.addAreaSeries({
      lineColor: 'transparent', lineWidth: 1,
      topColor: fill, bottomColor: fill,
      priceLineVisible: false, lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    s.setData(data);
    return s;
  };

  const p = fanData.percentiles || {};
  if (Array.isArray(p.p95)) band(pts(p.p95), bandLight);
  if (Array.isArray(p.p75)) band(pts(p.p75), bandDark);
  if (Array.isArray(p.p25)) band(pts(p.p25), bandLight);
  if (Array.isArray(p.p5)) band(pts(p.p5), bg);

  const samples = (fanData.sample_paths || []).slice(0, 8);
  const sampleColor = rgba(tok('--text-dim', '#525d70'), 0.55);
  for (const path of samples) {
    const s = chart.addLineSeries({
      color: sampleColor, lineWidth: 1,
      priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
    });
    s.setData(pts(path));
  }

  if (Array.isArray(p.p50)) {
    const median = chart.addLineSeries({
      color: accent, lineWidth: 2, priceLineVisible: false, lastValueVisible: true,
    });
    median.setData(pts(p.p50));
  }

  chart.timeScale().fitContent();
  return { chart, dispose() { chart.remove(); } };
}

/* ------------------------------- depthBars ------------------------------- */

/**
 * Order-book depth renderer (pure DOM — not lightweight-charts).
 * Rows are reused across updates so the depth fills animate via CSS.
 * @param {HTMLElement} el
 * @param {{side:'bid'|'ask', onFormat?:{price:(n:number)=>string, qty:(n:number)=>string}}} opts
 * @returns {{update(levels:Array<[number,number]>, max:number):void, dispose():void}}
 */
export function depthBars(el, { side, onFormat } = {}) {
  el.classList.add('db-side', side === 'ask' ? 'db-asks' : 'db-bids');
  const fp = (onFormat && onFormat.price) || ((n) => n.toFixed(2));
  const fq = (onFormat && onFormat.qty) || ((n) => String(n));
  const rows = [];

  function makeRow() {
    const row = document.createElement('div');
    row.className = 'db-row';
    row.innerHTML = '<div class="db-fill"></div><span class="db-price mono"></span><span class="db-qty mono"></span>';
    el.appendChild(row);
    return {
      row,
      fill: row.querySelector('.db-fill'),
      price: row.querySelector('.db-price'),
      qty: row.querySelector('.db-qty'),
    };
  }

  return {
    update(levels, max) {
      const m = Math.max(1, max || 0);
      while (rows.length < levels.length) rows.push(makeRow());
      rows.forEach((r, i) => {
        const lvl = levels[i];
        if (!lvl) { r.row.style.display = 'none'; return; }
        r.row.style.display = '';
        r.price.textContent = fp(lvl[0]);
        r.qty.textContent = fq(lvl[1]);
        r.fill.style.width = `${Math.min(100, (lvl[1] / m) * 100).toFixed(1)}%`;
      });
    },
    dispose() { el.innerHTML = ''; },
  };
}

export const charts = { candleChart, lineChart, fanChart, depthBars };
export default charts;
