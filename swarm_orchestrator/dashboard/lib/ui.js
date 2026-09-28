/* ============================================================================
   lib/ui.js — Chronos UI primitives: toasts, modals, drawers, confirm, icons.
   All styling comes from design-tokens.css + app-shell.css classes.
   ============================================================================ */

/* ------------------------------- icons --------------------------------- */
/* Crisp 1.5px-stroke line icons on a 24-grid, currentColor. */

const P = (d) => `<path d="${d}"/>`;

const ICONS = {
  candles: P('M7 4v3') + '<rect x="5" y="7" width="4" height="8" rx="1"/>' + P('M7 15v5')
    + P('M17 3v4') + '<rect x="15" y="7" width="4" height="10" rx="1"/>' + P('M17 17v4'),
  code: P('M8.5 8 5 12l3.5 4') + P('M15.5 8 19 12l-3.5 4') + P('M13 5.5l-2 13'),
  report: P('M7 3h7l4 4v13a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z') + P('M14 3v4h4')
    + P('M9.5 16.5v-3') + P('M12 16.5v-6') + P('M14.5 16.5v-4'),
  fan: P('M4 19C9 18 14 14 19.5 4.5') + P('M4 19c6-.5 11-3 15.5-9') + P('M4 19c6.5 0 12-.7 16-4')
    + P('M4 19h16'),
  clock: '<circle cx="12" cy="12" r="8.5"/>' + P('M12 7.5V12l3 2'),
  sparkles: P('M12 4.5 13.4 9l4.6 1.5-4.6 1.5L12 16.5 10.6 12 6 10.5 10.6 9z')
    + P('M18.5 15.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z'),
  feed: P('M4 5.5h16') + P('M4 10.5h16') + P('M4 15.5h10') + P('M4 20h6'),
  database: '<ellipse cx="12" cy="5.5" rx="7.5" ry="2.8"/>'
    + P('M4.5 5.5v13c0 1.55 3.35 2.8 7.5 2.8s7.5-1.25 7.5-2.8v-13')
    + P('M4.5 12c0 1.55 3.35 2.8 7.5 2.8s7.5-1.25 7.5-2.8'),
  flask: P('M9.5 3.5h5') + P('M10.5 3.5v5L5 18.5A1.6 1.6 0 0 0 6.4 21h11.2a1.6 1.6 0 0 0 1.4-2.5L13.5 8.5v-5')
    + P('M7.5 15h9'),
  news: P('M4 6.5h13v13H5.5A1.5 1.5 0 0 1 4 18z') + P('M17 9.5h2.5V18a1.5 1.5 0 0 1-3 0')
    + P('M7 10h7') + P('M7 13.5h7') + P('M7 16.5h4.5'),
  upload: P('M12 15V4.5') + P('M8 8.5 12 4.5l4 4') + P('M4.5 15.5v3A1.5 1.5 0 0 0 6 20h12a1.5 1.5 0 0 0 1.5-1.5v-3'),
  close: P('M6 6l12 12') + P('M18 6 6 18'),
  chevron: P('m9 5.5 6.5 6.5L9 18.5'),
  popout: P('M13.5 4.5H19a.5.5 0 0 1 .5.5v5.5') + P('M19.2 4.8 11 13')
    + P('M18 13.5V19a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1h5.5'),
  play: P('M8 5.5v13l10.5-6.5z'),
  pause: P('M8.5 5.5v13') + P('M15.5 5.5v13'),
  reset: P('M5 12a7 7 0 1 0 2-4.9') + P('M5 4.5V8h3.5'),
  download: P('M12 4.5V15') + P('m8 11.5 4 4 4-4') + P('M4.5 16v2.5A1.5 1.5 0 0 0 6 20h12a1.5 1.5 0 0 0 1.5-1.5V16'),
  trash: P('M4.5 6.5h15') + P('M9 6.5V4.8a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v1.7')
    + P('M6.5 6.5 7.4 19a1.6 1.6 0 0 0 1.6 1.5h6a1.6 1.6 0 0 0 1.6-1.5l.9-12.5')
    + P('M10 10.5v6') + P('M14 10.5v6'),
  plus: P('M12 5.5v13') + P('M5.5 12h13'),
  refresh: P('M19 12a7 7 0 1 1-2-4.9') + P('M19 4.5V8h-3.5'),
  logs: P('M5.5 4.5h13') + P('M5.5 9h13') + P('M5.5 13.5h8') + P('M5.5 18h5') + P('m15.5 15.5 2 2 3.5-3.5'),
  agent: '<circle cx="12" cy="8" r="3.8"/>' + P('M4.8 20c1.2-3.6 4-5.3 7.2-5.3s6 1.7 7.2 5.3'),
  cog: '<circle cx="12" cy="12" r="3.4"/>'
    + P('M12 3.2v2.2') + P('M12 18.6v2.2') + P('M20.8 12h-2.2') + P('M5.4 12H3.2')
    + P('m18.2 5.8-1.6 1.6') + P('m7.4 16.6-1.6 1.6') + P('m18.2 18.2-1.6-1.6') + P('m7.4 7.4L5.8 5.8'),
};

/* Names used interchangeably around the app. */
ICONS.strategy = ICONS.code;
ICONS.results = ICONS.report;
ICONS['pop-out'] = ICONS.popout;
ICONS.lab = ICONS.flask;
ICONS.runs = ICONS.database;

/**
 * @param {string} name icon key (see ICONS)
 * @returns {string} inline SVG markup (1.5px stroke, currentColor)
 */
function icon(name) {
  const body = ICONS[name] || ICONS.sparkles;
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" `
    + `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
}

/* ------------------------------- toasts -------------------------------- */

/**
 * Show a transient toast bottom-right.
 * @param {string} msg
 * @param {'good'|'bad'|'warn'|'info'} [kind='info']
 */
function toast(msg, kind = 'info') {
  let host = document.getElementById('toasts');
  if (!host) {
    host = document.createElement('div');
    host.id = 'toasts';
    document.body.appendChild(host);
  }
  const el = document.createElement('div');
  el.className = `toast ${kind === 'info' ? '' : kind}`.trim();
  el.setAttribute('role', 'status');
  el.textContent = msg;
  host.appendChild(el);
  const ttl = kind === 'bad' ? 6500 : 4000;
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transition = 'opacity var(--dur-2) var(--ease)';
    setTimeout(() => el.remove(), 240);
  }, ttl);
  return el;
}

/* ---------------------------- modal / drawer ---------------------------- */

function resolveBody(container, body) {
  if (body == null) return;
  if (typeof body === 'string') container.innerHTML = body;
  else container.appendChild(body);
}

/**
 * Open a modal dialog. Closes on backdrop click, Esc, or `close()`.
 * @param {{title:string, body:string|Node, actions?:Array<{label:string,kind?:string,onClick?:(close:()=>void)=>void}>, wide?:boolean}} opts
 * @returns {{close:()=>void, el:HTMLElement, body:HTMLElement}}
 */
function modal({ title, body, actions = [], wide = false }) {
  const backdrop = document.createElement('div');
  backdrop.className = 'ui-backdrop';
  const box = document.createElement('div');
  box.className = `ui-modal${wide ? ' wide' : ''}`;
  box.setAttribute('role', 'dialog');
  box.setAttribute('aria-modal', 'true');
  box.setAttribute('aria-label', title);

  const head = document.createElement('div');
  head.className = 'ui-modal-head';
  head.innerHTML = `<h3>${title}</h3>`;
  const closeBtn = document.createElement('button');
  closeBtn.className = 'btn btn-ghost btn-icon';
  closeBtn.setAttribute('aria-label', 'Close dialog');
  closeBtn.innerHTML = icon('close');
  head.appendChild(closeBtn);

  const bodyEl = document.createElement('div');
  bodyEl.className = 'ui-modal-body';
  resolveBody(bodyEl, body);

  box.appendChild(head);
  box.appendChild(bodyEl);

  let foot = null;
  if (actions.length) {
    foot = document.createElement('div');
    foot.className = 'ui-modal-foot';
    for (const a of actions) {
      const b = document.createElement('button');
      b.className = `btn ${a.kind ? `btn-${a.kind}` : ''}`.trim();
      b.textContent = a.label;
      b.addEventListener('click', () => (a.onClick ? a.onClick(close, b) : close()));
      foot.appendChild(b);
    }
    box.appendChild(foot);
  }

  backdrop.appendChild(box);
  document.body.appendChild(backdrop);

  const onKey = (e) => { if (e.key === 'Escape') close(); };
  const close = () => {
    document.removeEventListener('keydown', onKey);
    backdrop.classList.add('closing');
    setTimeout(() => backdrop.remove(), 140);
  };
  closeBtn.addEventListener('click', close);
  backdrop.addEventListener('pointerdown', (e) => { if (e.target === backdrop) close(); });
  document.addEventListener('keydown', onKey);

  requestAnimationFrame(() => {
    const first = box.querySelector('input, textarea, select, button:not([aria-label="Close dialog"])');
    (first || closeBtn).focus();
  });

  return { close, el: box, body: bodyEl };
}

/**
 * Open a slide-in drawer.
 * @param {{title:string, body:string|Node, side?:'right'|'left'}} opts
 * @returns {{close:()=>void, el:HTMLElement, body:HTMLElement, onClose:(fn:()=>void)=>void}}
 */
function drawer({ title, body, side = 'right' }) {
  const backdrop = document.createElement('div');
  backdrop.className = 'ui-backdrop drawer-backdrop';
  const box = document.createElement('div');
  box.className = `ui-drawer ${side}`;
  box.setAttribute('role', 'dialog');
  box.setAttribute('aria-label', title);

  const head = document.createElement('div');
  head.className = 'ui-modal-head';
  head.innerHTML = `<h3>${title}</h3>`;
  const closeBtn = document.createElement('button');
  closeBtn.className = 'btn btn-ghost btn-icon';
  closeBtn.setAttribute('aria-label', 'Close drawer');
  closeBtn.innerHTML = icon('close');
  head.appendChild(closeBtn);

  const bodyEl = document.createElement('div');
  bodyEl.className = 'ui-drawer-body';
  resolveBody(bodyEl, body);

  box.appendChild(head);
  box.appendChild(bodyEl);
  backdrop.appendChild(box);
  document.body.appendChild(backdrop);

  let closeCb = null;
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  const close = () => {
    document.removeEventListener('keydown', onKey);
    if (closeCb) { try { closeCb(); } catch { /* ignore */ } }
    backdrop.classList.add('closing');
    setTimeout(() => backdrop.remove(), 180);
  };
  closeBtn.addEventListener('click', close);
  backdrop.addEventListener('pointerdown', (e) => { if (e.target === backdrop) close(); });
  document.addEventListener('keydown', onKey);
  requestAnimationFrame(() => closeBtn.focus());

  return { close, el: box, body: bodyEl, onClose: (fn) => { closeCb = fn; } };
}

/**
 * Confirm dialog. @returns {Promise<boolean>}
 */
function confirm(msg) {
  return new Promise((resolve) => {
    let settled = false;
    const settle = (v, close) => { if (!settled) { settled = true; resolve(v); } close(); };
    const m = modal({
      title: 'Confirm',
      body: `<p class="ui-confirm-msg">${msg}</p>`,
      actions: [
        { label: 'Cancel', onClick: (close) => settle(false, close) },
        { label: 'Confirm', kind: 'primary', onClick: (close) => settle(true, close) },
      ],
    });
    // Backdrop / Esc close resolves false after the animation delay.
    const obs = new MutationObserver(() => {
      if (!document.body.contains(m.el)) {
        if (!settled) { settled = true; resolve(false); }
        obs.disconnect();
      }
    });
    obs.observe(document.body, { childList: true, subtree: true });
  });
}

/**
 * Put a button into a loading state while an async action runs.
 * @param {HTMLButtonElement} btn
 * @param {() => Promise<any>} fn
 */
async function busy(btn, fn) {
  const prev = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner" aria-hidden="true"></span>';
  try {
    return await fn();
  } finally {
    btn.disabled = false;
    btn.innerHTML = prev;
  }
}

export const ui = { toast, modal, drawer, confirm, icon, busy };
export default ui;
