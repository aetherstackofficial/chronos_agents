/* ============================================================================
   lib/splitter.js — draggable column/row splitters for flexbox children.
   Inserts .splitter elements between the container's children, drives sizing
   through flex-grow weights (1 unit = 1px while dragging), persists layouts
   to localStorage, double-click (or Enter) collapses/restores a pane.
   Pointer events; keyboard accessible (arrow keys resize).
   ============================================================================ */

const KEY_STEP = 24; // px per arrow-key press

/**
 * @typedef {Object} SplitOptions
 * @property {number[]} [sizes]       initial weights (any units, e.g. [70,30])
 * @property {number[]} [min]         min size in px per pane
 * @property {boolean|number[]|'first'|'last'} [collapsible] which panes may collapse
 * @property {string}  [storageKey]   localStorage key for persisted weights
 */

/**
 * Core splitter factory.
 * @param {HTMLElement} container flex container whose CHILDREN become panes
 * @param {'col'|'row'} dir       col = vertical splitters (resize widths)
 * @param {SplitOptions} opts
 * @returns {{destroy():void, toggle(i:number):void, collapse(i:number):void,
 *            restore(i:number):void, isCollapsed(i:number):boolean}}
 */
function split(container, dir, opts = {}) {
  const { sizes, min = [], collapsible, storageKey } = opts;
  const panes = Array.from(container.children);
  if (panes.length < 2) return { destroy() {}, toggle() {}, collapse() {}, restore() {}, isCollapsed() { return false; } };

  const sizeProp = dir === 'col' ? 'width' : 'height';
  const minProp = dir === 'col' ? 'minWidth' : 'minHeight';
  const axis = dir === 'col' ? 'clientX' : 'clientY';
  const splitters = [];
  const prevWeight = new Array(panes.length).fill(0);
  const collapsed = new Array(panes.length).fill(false);

  function canCollapse(i) {
    if (collapsible === true) return true;
    if (collapsible === 'first') return i === 0;
    if (collapsible === 'last') return i === panes.length - 1;
    if (Array.isArray(collapsible)) return collapsible.includes(i);
    return false;
  }

  /* ---- weights ---- */

  function loadWeights() {
    if (storageKey) {
      try {
        const saved = JSON.parse(localStorage.getItem(storageKey) || 'null');
        if (Array.isArray(saved) && saved.length === panes.length && saved.every((w) => typeof w === 'number' && w >= 0)) {
          return saved;
        }
      } catch { /* corrupt entry — fall through */ }
    }
    if (Array.isArray(sizes) && sizes.length === panes.length) return sizes.slice();
    return panes.map(() => 1);
  }

  function applyWeights(weights) {
    panes.forEach((p, i) => {
      p.style.flex = `${weights[i]} 1 0px`;
      p.style[minProp] = collapsed[i] ? '0px' : `${min[i] || 0}px`;
    });
  }

  function currentWeights() {
    return panes.map((p) => parseFloat(p.style.flexGrow || '1') || 0);
  }

  function persist() {
    if (!storageKey) return;
    try { localStorage.setItem(storageKey, JSON.stringify(currentWeights())); } catch { /* quota */ }
  }

  /** Freeze every pane's weight to its current pixel size (1 unit = 1px). */
  function freezeToPixels() {
    const px = panes.map((p) => p.getBoundingClientRect()[sizeProp]);
    panes.forEach((p, i) => { p.style.flex = `${px[i]} 1 0px`; });
    return px;
  }

  /* ---- collapse / restore ---- */

  function collapse(i) {
    if (collapsed[i]) return;
    prevWeight[i] = currentWeights()[i] || 1;
    collapsed[i] = true;
    panes[i].classList.add('pane-collapsed');
    panes[i].style.flex = '0 1 0px';
    panes[i].style[minProp] = '0px';
    persist();
  }

  function restore(i) {
    if (!collapsed[i]) return;
    collapsed[i] = false;
    panes[i].classList.remove('pane-collapsed');
    panes[i].style.flex = `${prevWeight[i] || 1} 1 0px`;
    panes[i].style[minProp] = `${min[i] || 0}px`;
    persist();
  }

  function toggle(i) { collapsed[i] ? restore(i) : collapse(i); }

  /* ---- drag ---- */

  function makeSplitter(aIdx) {
    const bIdx = aIdx + 1;
    const el = document.createElement('div');
    el.className = `splitter ${dir}`;
    el.setAttribute('role', 'separator');
    el.setAttribute('aria-orientation', dir === 'col' ? 'vertical' : 'horizontal');
    el.tabIndex = 0;
    el.setAttribute('aria-label', 'Resize panels (arrow keys); double-click or Enter to collapse');

    let drag = null;

    el.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      const px = freezeToPixels();
      drag = { start: e[axis], aPx: px[aIdx], bPx: px[bIdx] };
      el.classList.add('dragging');
    });

    el.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const d = e[axis] - drag.start;
      const total = drag.aPx + drag.bPx;
      const minA = collapsed[aIdx] ? 0 : (min[aIdx] || 0);
      const minB = collapsed[bIdx] ? 0 : (min[bIdx] || 0);
      const a = Math.min(Math.max(drag.aPx + d, minA), total - minB);
      panes[aIdx].style.flex = `${a} 1 0px`;
      panes[bIdx].style.flex = `${total - a} 1 0px`;
    });

    const endDrag = () => {
      if (!drag) return;
      drag = null;
      el.classList.remove('dragging');
      persist();
    };
    el.addEventListener('pointerup', endDrag);
    el.addEventListener('pointercancel', endDrag);

    el.addEventListener('dblclick', () => {
      const target = canCollapse(aIdx) && !canCollapse(bIdx) ? aIdx
        : canCollapse(bIdx) && !canCollapse(aIdx) ? bIdx
          : canCollapse(bIdx) ? bIdx : canCollapse(aIdx) ? aIdx : -1;
      if (target >= 0) toggle(target);
    });

    el.addEventListener('keydown', (e) => {
      const inc = dir === 'col' ? ['ArrowRight'] : ['ArrowDown'];
      const dec = dir === 'col' ? ['ArrowLeft'] : ['ArrowUp'];
      let d = 0;
      if (inc.includes(e.key)) d = KEY_STEP;
      else if (dec.includes(e.key)) d = -KEY_STEP;
      else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.dispatchEvent(new Event('dblclick')); return; }
      else return;
      e.preventDefault();
      const px = freezeToPixels();
      const total = px[aIdx] + px[bIdx];
      const a = Math.min(Math.max(px[aIdx] + d, min[aIdx] || 0), total - (min[bIdx] || 0));
      panes[aIdx].style.flex = `${a} 1 0px`;
      panes[bIdx].style.flex = `${total - a} 1 0px`;
      persist();
    });

    container.insertBefore(el, panes[bIdx]);
    splitters.push(el);
  }

  /* ---- init ---- */

  container.style.display = 'flex';
  container.style.flexDirection = dir === 'col' ? 'row' : 'column';
  applyWeights(loadWeights());
  for (let i = 0; i < panes.length - 1; i++) makeSplitter(i);

  return {
    destroy() {
      splitters.forEach((s) => s.remove());
      panes.forEach((p) => { p.style.flex = ''; p.style[minProp] = ''; p.classList.remove('pane-collapsed'); });
    },
    toggle, collapse, restore,
    isCollapsed: (i) => !!collapsed[i],
  };
}

/** Column splitters — children sit side by side, widths resize. */
export function hSplit(container, opts) { return split(container, 'col', opts); }

/** Row splitters — children stack, heights resize. */
export function vSplit(container, opts) { return split(container, 'row', opts); }

/** Back-compat alias. */
export const makeResizable = hSplit;

export const splitter = { hSplit, vSplit, makeResizable };
export default splitter;
