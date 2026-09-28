/* ============================================================================
   lib/format.js — Chronos number/time formatting singleton (PROTOCOL_V2 §7).
   All money is INR. All times are UTC (the sim epoch is IST-as-UTC).
   ============================================================================ */

const INR2 = new Intl.NumberFormat('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const COMPACT = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });
const numCache = new Map();

/** Grouped en-IN formatter with `d` fraction digits (cached). */
function groupFmt(d) {
  if (!numCache.has(d)) {
    numCache.set(d, new Intl.NumberFormat('en-IN', { minimumFractionDigits: d, maximumFractionDigits: d }));
  }
  return numCache.get(d);
}

function finite(n) { return typeof n === 'number' && Number.isFinite(n); }

export const fmt = {
  /** ₹ + en-IN grouping, 2dp. fmt.inr(1234.5) → "₹1,234.50". Falls back to "—". */
  inr(n) {
    if (!finite(n)) return '—';
    return '₹' + INR2.format(n);
  },

  /** Explicit-sign number, 2dp. signed(12.3) → "+12.30", signed(-4) → "-4.00". */
  signed(n) {
    if (!finite(n)) return '—';
    return (n >= 0 ? '+' : '-') + INR2.format(Math.abs(n));
  },

  /**
   * Percentage, 1dp. Accepts a fraction (0.62 → "62.0%") or an
   * already-percent value (62 → "62.0%"): |n| <= 1 is treated as a fraction.
   */
  pct(n) {
    if (!finite(n)) return '—';
    const v = Math.abs(n) <= 1 ? n * 100 : n;
    return v.toFixed(1) + '%';
  },

  /** Compact rupees: compact(1200000) → "₹1.2M". */
  compact(n) {
    if (!finite(n)) return '—';
    return '₹' + COMPACT.format(n);
  },

  /** Grouped number with d fraction digits (default 2). num(5400, 0) → "5,400". */
  num(n, d = 2) {
    if (!finite(n)) return '—';
    return groupFmt(d).format(n);
  },

  /** UTC clock from unix seconds: time(1767258960) → "09:16". */
  time(unix) {
    if (!finite(unix)) return '—';
    const d = new Date(unix * 1000);
    const hh = String(d.getUTCHours()).padStart(2, '0');
    const mm = String(d.getUTCMinutes()).padStart(2, '0');
    return `${hh}:${mm}`;
  },

  /** UTC date + clock from unix seconds: dtime(...) → "01 Jan 09:16". */
  dtime(unix) {
    if (!finite(unix)) return '—';
    const d = new Date(unix * 1000);
    const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const dd = String(d.getUTCDate()).padStart(2, '0');
    return `${dd} ${MONTHS[d.getUTCMonth()]} ${this.time(unix)}`;
  },
};

export default fmt;
