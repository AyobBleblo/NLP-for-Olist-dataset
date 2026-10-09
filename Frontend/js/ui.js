/**
 * js/ui.js — tiny dependency-free UI kit.
 *
 * Provides the shared shell pieces every page uses: icon markup, toast
 * notifications, a modal, skeleton loaders, empty/error states, star
 * ratings, pagination and accessible DOM helpers.
 */

import { escapeHtml, hueOf, initials, money, sentimentBucket, sentimentLabel, prettyCategory } from './format.js?v=2';

/* ------------------------------------------------------------------ *
 * Icons — inline SVG (no external font dependency)
 * ------------------------------------------------------------------ */

const ICON_PATHS = {
  cart: '<circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/><path d="M1 1h4l2.7 13.4a2 2 0 0 0 2 1.6h9.7a2 2 0 0 0 2-1.6L23 6H6"/>',
  search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
  sparkles: '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6.3 6.3l2.8 2.8M14.9 14.9l2.8 2.8M17.7 6.3l-2.8 2.8M9.1 14.9l-2.8 2.8"/>',
  box: '<path d="M21 8v8a2 2 0 0 1-1 1.7l-7 4a2 2 0 0 1-2 0l-7-4A2 2 0 0 1 3 16V8a2 2 0 0 1 1-1.7l7-4a2 2 0 0 1 2 0l7 4A2 2 0 0 1 21 8Z"/><path d="m3.3 7 8.7 5 8.7-5M12 22V12"/>',
  receipt: '<path d="M4 2v20l2-1 2 1 2-1 2 1 2-1 2 1 2-1 2 1V2l-2 1-2-1-2 1-2-1-2 1-2-1-2 1Z"/><path d="M8 7h8M8 11h8M8 15h5"/>',
  chart: '<path d="M3 3v18h18"/><path d="m7 15 3-4 3 3 4-6"/>',
  terminal: '<path d="m4 17 6-6-6-6"/><path d="M12 19h8"/>',
  book: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z"/>',
  close: '<path d="M18 6 6 18M6 6l12 12"/>',
  trash: '<path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>',
  minus: '<path d="M5 12h14"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  arrowLeft: '<path d="M19 12H5M12 19l-7-7 7-7"/>',
  arrowRight: '<path d="M5 12h14M12 5l7 7-7 7"/>',
  refresh: '<path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/><path d="M3 21v-5h5"/>',
  alert: '<path d="M12 9v4M12 17h.01"/><circle cx="12" cy="12" r="10"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
  user: '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
  heart: '<path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1-1.1a5.5 5.5 0 0 0-7.8 7.8l1.1 1L12 21l7.7-7.6 1.1-1a5.5 5.5 0 0 0 0-7.8Z"/>',
  star: '<path d="m12 2 3.1 6.3 6.9 1-5 4.9 1.2 6.8L12 17.8 5.8 21l1.2-6.8-5-4.9 6.9-1Z"/>',
  layers: '<path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 17l9 5 9-5"/>',
  database: '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
  git: '<circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M6 21V9a9 9 0 0 0 9 9"/>',
  play: '<path d="m5 3 14 9-14 9V3Z"/>',
  external: '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><path d="M15 3h6v6M10 14 21 3"/>',
  menu: '<path d="M3 6h18M3 12h18M3 18h18"/>',
  filter: '<path d="M22 3H2l8 9.5V19l4 2v-8.5L22 3Z"/>',
  tag: '<path d="M12.6 2.6a2 2 0 0 0-1.4-.6H4a2 2 0 0 0-2 2v7.2a2 2 0 0 0 .6 1.4l8.2 8.2a2 2 0 0 0 2.8 0l7.2-7.2a2 2 0 0 0 0-2.8Z"/><path d="M7 7h.01"/>',
  activity: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
  shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/>',
  brain: '<path d="M12 5a3 3 0 0 0-6 0 3 3 0 0 0-3 3 3 3 0 0 0 1 2.2A3 3 0 0 0 6 16a3 3 0 0 0 6 0Z"/><path d="M12 5a3 3 0 0 1 6 0 3 3 0 0 1 3 3 3 3 0 0 1-1 2.2A3 3 0 0 1 18 16a3 3 0 0 1-6 0Z"/><path d="M12 5v14"/>',
  upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M17 8l-5-5-5 5"/><path d="M12 3v12"/>',
  lock: '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
  creditCard: '<rect x="2" y="5" width="20" height="14" rx="2"/><path d="M2 10h20"/>',
  truck: '<rect x="1" y="3" width="15" height="13"/><polygon points="16 8 20 8 23 11 23 16 16 16 16 8"/><circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/>',
  award: '<circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/>',
  zap: '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
};

export function icon(name, size = 18, cls = '') {
  const path = ICON_PATHS[name] || ICON_PATHS.info;
  return `<svg class="icon ${cls}" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${path}</svg>`;
}

/* ------------------------------------------------------------------ *
 * DOM helpers
 * ------------------------------------------------------------------ */

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function el(tag, attrs = {}, html = '') {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (v === true) node.setAttribute(k, '');
    else node.setAttribute(k, v);
  }
  if (html) node.innerHTML = html;
  return node;
}

/* ------------------------------------------------------------------ *
 * Toast notifications
 * ------------------------------------------------------------------ */

let toastHost = null;

function ensureToastHost() {
  if (!toastHost) {
    toastHost = el('div', {
      class: 'toast-host',
      id: 'toast-host',
      role: 'status',
      'aria-live': 'polite',
    });
    document.body.appendChild(toastHost);
  }
  return toastHost;
}

export function toast(message, { type = 'info', title = '', timeout = 4200 } = {}) {
  const host = ensureToastHost();
  const icons = { success: 'check', error: 'alert', info: 'info', warn: 'alert' };
  const node = el('div', { class: `toast toast--${type}` }, `
    <span class="toast__icon">${icon(icons[type] || 'info', 18)}</span>
    <div class="toast__body">
      ${title ? `<p class="toast__title">${escapeHtml(title)}</p>` : ''}
      <p class="toast__msg">${escapeHtml(message)}</p>
    </div>
    <button class="toast__close" type="button" aria-label="Dismiss notification">${icon('close', 14)}</button>
  `);
  const dismiss = () => {
    node.classList.add('is-leaving');
    setTimeout(() => node.remove(), 220);
  };
  node.querySelector('.toast__close').addEventListener('click', dismiss);
  host.appendChild(node);
  requestAnimationFrame(() => node.classList.add('is-visible'));
  if (timeout) setTimeout(dismiss, timeout);
  return dismiss;
}

/* ------------------------------------------------------------------ *
 * Modal
 * ------------------------------------------------------------------ */

export function modal({ title, body, actions = [], size = 'md' }) {
  const overlay = el('div', { class: 'modal-overlay', role: 'dialog', 'aria-modal': 'true' });
  const box = el('div', { class: `modal modal--${size}` });
  box.innerHTML = `
    <header class="modal__head">
      <h2 class="modal__title">${escapeHtml(title || '')}</h2>
      <button class="modal__close" type="button" aria-label="Close dialog">${icon('close', 16)}</button>
    </header>
    <div class="modal__body"></div>
    <footer class="modal__foot"></footer>
  `;
  const bodyHost = box.querySelector('.modal__body');
  if (typeof body === 'string') bodyHost.innerHTML = body;
  else if (body) bodyHost.appendChild(body);

  const foot = box.querySelector('.modal__foot');
  const close = () => {
    overlay.classList.remove('is-visible');
    setTimeout(() => overlay.remove(), 200);
    document.removeEventListener('keydown', onKey);
  };
  actions.forEach((a) => {
    const btn = el('button', { class: `btn ${a.class || 'btn--ghost'}`, type: 'button', text: a.label });
    btn.addEventListener('click', () => {
      const result = a.onClick ? a.onClick() : undefined;
      if (result !== false) close();
    });
    foot.appendChild(btn);
  });
  if (!actions.length) foot.remove();

  function onKey(e) {
    if (e.key === 'Escape') close();
  }
  box.querySelector('.modal__close').addEventListener('click', close);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) close();
  });
  document.addEventListener('keydown', onKey);

  overlay.appendChild(box);
  document.body.appendChild(overlay);
  requestAnimationFrame(() => overlay.classList.add('is-visible'));
  const focusable = box.querySelector('button, [href], input, select, textarea');
  focusable?.focus();
  return { close, node: box };
}

/* ------------------------------------------------------------------ *
 * Loading / empty / error states
 * ------------------------------------------------------------------ */

export function skeletonCards(count = 8, cls = 'product-card') {
  return Array.from({ length: count })
    .map(
      () => `
      <article class="card ${cls} skeleton-card" aria-hidden="true">
        <div class="skeleton skeleton--media"></div>
        <div class="skeleton skeleton--line w-60"></div>
        <div class="skeleton skeleton--line w-90"></div>
        <div class="skeleton skeleton--line w-40"></div>
      </article>`,
    )
    .join('');
}

export function skeletonRows(count = 5) {
  return Array.from({ length: count })
    .map(
      () => `
      <div class="skeleton-row" aria-hidden="true">
        <div class="skeleton skeleton--line w-40"></div>
        <div class="skeleton skeleton--line w-90"></div>
        <div class="skeleton skeleton--line w-60"></div>
      </div>`,
    )
    .join('');
}

export function emptyState({ title = 'Nothing here yet', body = '', action = '', iconName = 'box' } = {}) {
  return `
    <div class="state state--empty">
      <span class="state__icon">${icon(iconName, 26)}</span>
      <h3 class="state__title">${escapeHtml(title)}</h3>
      ${body ? `<p class="state__body">${escapeHtml(body)}</p>` : ''}
      ${action}
    </div>`;
}

export function errorState({ title = 'Something went wrong', body = '', retry = true } = {}) {
  return `
    <div class="state state--error">
      <span class="state__icon">${icon('alert', 26)}</span>
      <h3 class="state__title">${escapeHtml(title)}</h3>
      ${body ? `<p class="state__body">${escapeHtml(body)}</p>` : ''}
      ${retry ? `<button class="btn btn--primary" type="button" data-action="retry">${icon('refresh', 16)} Retry</button>` : ''}
    </div>`;
}

/* ------------------------------------------------------------------ *
 * Product art: real images when present, deterministic fallback
 * ------------------------------------------------------------------ */

export function productArt(product, size = 'md') {
  const name = product?.name_translated || product?.name || product?.external_id || 'Product';
  const url = product?.image_url;
  const hue = hueOf(product?.external_id || name);
  const fb = `<span class="art__fallback" style="--art-hue:${hue}">${escapeHtml(initials(name))}</span>`;
  return `
    <div class="art art--${size}" style="--art-hue:${hue}">
      ${fb}
      ${url ? `<img src="${escapeHtml(url)}" alt="${escapeHtml(name)}" loading="lazy" decoding="async" onerror="this.remove()">` : ''}
    </div>`;
}

export function stars(rating, size = 14) {
  const r = Number(rating) || 0;
  return `<span class="stars" role="img" aria-label="${r} out of 5">${Array.from({ length: 5 })
    .map((_, i) => `<span class="stars__item ${i < Math.round(r) ? 'is-on' : ''}">${icon('star', size)}</span>`)
    .join('')}</span>`;
}

export function sentimentPill(avg) {
  const bucket = sentimentBucket(avg);
  const n = Number(avg);
  const text = bucket === 'none' ? sentimentLabel(avg) : `${sentimentLabel(avg)} · ${n.toFixed(2)}`;
  return `<span class="pill pill--${bucket}" title="Product sentiment index (PSI) — mean of P(positive) − P(negative) over reviews">${icon('heart', 13)} ${escapeHtml(text)}</span>`;
}

export function categoryChip(slug) {
  return `<span class="chip">${icon('tag', 13)} ${escapeHtml(prettyCategory(slug))}</span>`;
}

/* ------------------------------------------------------------------ *
 * Product card
 * ------------------------------------------------------------------ */

export function productCard(product, { badge = '', compact = false } = {}) {
  const price = money(product.price);
  const name = product.name_translated || product.name || product.external_id;
  const sub = product.name && product.name_translated ? product.name : '';
  return `
    <article class="card product-card ${compact ? 'product-card--compact' : ''}">
      <a class="product-card__media" href="product.html?id=${encodeURIComponent(product.external_id)}" aria-label="${escapeHtml(name)}">
        ${productArt(product)}
        ${badge ? `<span class="product-card__badge">${badge}</span>` : ''}
      </a>
      <div class="product-card__body">
        ${categoryChip(product.category_name)}
        <h3 class="product-card__title">
          <a href="product.html?id=${encodeURIComponent(product.external_id)}">${escapeHtml(name)}</a>
        </h3>
        ${sub ? `<p class="product-card__sub">${escapeHtml(sub)}</p>` : ''}
        <div class="product-card__meta">
          ${sentimentPill(product.avg_sentiment_score)}
          <span class="pill pill--ghost" title="total_purchases from the Olist order items">${icon('activity', 13)} ${Number(product.total_purchases || 0)} sold</span>
        </div>
      </div>
      <footer class="product-card__foot">
        <span class="price">${price}</span>
        <button class="btn btn--primary btn--sm" type="button" data-add-to-cart="${escapeHtml(product.external_id)}">
          ${icon('cart', 15)} Add
        </button>
      </footer>
    </article>`;
}

/* ------------------------------------------------------------------ *
 * Pagination
 * ------------------------------------------------------------------ */

export function pagination({ page, pages, total, onPage }) {
  const wrap = el('nav', { class: 'pagination', 'aria-label': 'Pagination' });
  if (pages <= 1) return wrap;

  const mk = (label, target, { disabled = false, active = false, aria } = {}) => {
    const b = el('button', {
      class: `page-btn ${active ? 'is-active' : ''}`,
      type: 'button',
      html: label,
      'aria-label': aria || `Page ${target}`,
      'aria-current': active ? 'page' : null,
      disabled: disabled || active,
    });
    if (!disabled && !active) b.addEventListener('click', () => onPage(target));
    return b;
  };

  wrap.appendChild(mk(icon('arrowLeft', 15), page - 1, { disabled: page <= 1, aria: 'Previous page' }));

  const windowSize = 2;
  const nums = new Set([1, pages, page, page - 1, page + 1, page - windowSize, page + windowSize]);
  const list = [...nums].filter((n) => n >= 1 && n <= pages).sort((a, b) => a - b);
  let prev = 0;
  list.forEach((n) => {
    if (prev && n - prev > 1) wrap.appendChild(el('span', { class: 'page-gap', text: '…' }));
    wrap.appendChild(mk(String(n), n, { active: n === page }));
    prev = n;
  });

  wrap.appendChild(mk(icon('arrowRight', 15), page + 1, { disabled: page >= pages, aria: 'Next page' }));
  return wrap;
}

/* ------------------------------------------------------------------ *
 * Bar / gauge visualisations (CSS only, no chart library needed)
 * ------------------------------------------------------------------ */

export function bar(value, max, cls = '') {
  const v = Number(value) || 0;
  const m = Number(max) || 1;
  const width = Math.max(2, Math.min(100, (v / m) * 100));
  return `<span class="bar ${cls}"><span class="bar__fill" style="width:${width}%"></span></span>`;
}

export function gauge(value) {
  const n = Math.max(0, Math.min(1, Number(value) || 0));
  const bucket = sentimentBucket(n);
  return `
    <div class="gauge" style="--gauge-value:${(n * 100).toFixed(1)}">
      <svg viewBox="0 0 120 64" class="gauge__svg" aria-hidden="true">
        <path class="gauge__track" d="M10 58a50 50 0 0 1 100 0" fill="none" stroke-width="10" stroke-linecap="round"/>
        <path class="gauge__value gauge__value--${bucket}" d="M10 58a50 50 0 0 1 100 0" fill="none" stroke-width="10" stroke-linecap="round"
              stroke-dasharray="157" stroke-dashoffset="${(157 * (1 - n)).toFixed(1)}"/>
      </svg>
      <span class="gauge__label">${n.toFixed(2)}</span>
    </div>`;
}
