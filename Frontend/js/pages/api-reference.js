/**
 * js/pages/api-reference.js — renders every endpoint from js/config.js
 * (transcribed from backend/store/urls.py) with live "send request" buttons.
 */

import { initPage, bootstrapData } from '../main.js';
import { API, BACKEND_NOTES, ENDPOINTS, RECSYS, safeGet, safeSet } from '../config.js';
import { describeError } from '../api.js';
import { $, icon, toast } from '../ui.js';
import { escapeHtml } from '../format.js';

initPage({ active: 'api-reference.html', title: 'API Reference' });

const BASE_KEY = 'olist.api.testBase';
const defaultBase = () => (API.mode === 'remote' ? API.remoteBase : new URL('tables/', document.baseURI).href);

const baseInput = $('#base-input');
baseInput.value = safeGet(BASE_KEY) || defaultBase();

function currentBase() {
  return baseInput.value.trim() || defaultBase();
}

function paintBaseNote() {
  const base = currentBase();
  const isLocal = base.includes('/tables/');
  $('#api-base-pill').textContent = isLocal ? 'Local store' : 'Django API';
  $('#base-note').innerHTML = isLocal
    ? `Currently pointing at the platform table API. Set this to <code>${escapeHtml(API.remoteBase)}</code> to hit the real Django backend — but note that <code>settings.py</code> installs no CORS middleware, so cross-origin calls will be blocked until django-cors-headers is added.`
    : `Pointing at a Django-style base URL. Every endpoint below is relative to it, exactly as <code>backend/store/urls.py</code> defines them.`;
}

baseInput.addEventListener('input', () => {
  safeSet(BASE_KEY, baseInput.value.trim());
  paintBaseNote();
});
$('#reset-base').addEventListener('click', () => {
  baseInput.value = defaultBase();
  safeSet(BASE_KEY, baseInput.value);
  paintBaseNote();
});

/* ------------------------------------------------------------------ *
 * Endpoint rendering
 * ------------------------------------------------------------------ */

function paramsTable(endpoint) {
  if (!endpoint.params.length) return '<p class="text-sm text-dim">No parameters.</p>';
  return `<table class="param-table"><tbody>${endpoint.params
    .map(
      (p) => `<tr><td>${escapeHtml(p.name)}</td><td><span class="tag">${escapeHtml(p.type)}</span></td><td class="text-dim">${escapeHtml(p.note)}</td></tr>`,
    )
    .join('')}</tbody></table>`;
}

function tryForm(endpoint) {
  if (endpoint.id === 'root' || endpoint.id === 'category-detail') return '';
  const fields = [];

  if (['cart-get', 'cart-add', 'cart-update', 'cart-remove', 'cart-checkout', 'recommendations'].includes(endpoint.id)) {
    fields.push(
      `<label class="field"><span class="text-xs text-dim">customer_external_id</span><input class="input" data-param="customer_external_id" value="c37cc6c1a59d81460a3059744f7ada1c" spellcheck="false"></label>`,
    );
  }
  if (endpoint.id === 'products') {
    fields.push(
      `<label class="field"><span class="text-xs text-dim">search</span><input class="input" data-param="search" placeholder="e.g. sports"></label>`,
      `<label class="field"><span class="text-xs text-dim">category</span><input class="input" data-param="category" placeholder="id or slug"></label>`,
      `<label class="field"><span class="text-xs text-dim">ordering</span><input class="input" data-param="ordering" value="-total_purchases,id"></label>`,
      `<label class="field"><span class="text-xs text-dim">page</span><input class="input" data-param="page" value="1"></label>`,
    );
  }
  if (endpoint.id === 'product-detail') {
    fields.push(
      `<label class="field"><span class="text-xs text-dim">lookup_value</span><input class="input" data-param="lookup_value" placeholder="numeric id or 32-char external_id"></label>`,
    );
  }
  if (endpoint.id === 'categories') {
    fields.push(`<label class="field"><span class="text-xs text-dim">search</span><input class="input" data-param="search" placeholder="e.g. sports"></label>`);
  }
  if (endpoint.id === 'recommendations') {
    fields.push(`<label class="field"><span class="text-xs text-dim">n</span><input class="input" data-param="n" value="5"></label>`);
  }
  if (endpoint.body) {
    fields.push(
      `<label class="field"><span class="text-xs text-dim">request body (JSON)</span><input class="input" data-body="1" value='${escapeHtml(JSON.stringify(endpoint.body))}' spellcheck="false"></label>`,
    );
  }
  if (!fields.length) return '';

  return `
    <div class="try-form">
      <p class="text-xs text-dim">Send this request from the browser using the base URL above.</p>
      <div class="try-form__row">${fields.join('')}</div>
      <div class="row">
        <button class="btn btn--primary btn--sm" type="button" data-send="${endpoint.id}">${icon('play', 14)} Send request</button>
        <span class="text-xs text-dim" data-status="${endpoint.id}"></span>
      </div>
      <pre class="code-block" data-output="${endpoint.id}" style="display:none"></pre>
    </div>`;
}

function endpointCard(endpoint) {
  return `
    <article class="endpoint" data-endpoint="${endpoint.id}">
      <button class="endpoint__head" type="button" aria-expanded="false">
        <span class="method method--${endpoint.method.toLowerCase()}">${endpoint.method}</span>
        <span>
          <span class="endpoint__title">${escapeHtml(endpoint.title)}</span>
          <span class="endpoint__path">${escapeHtml(endpoint.path)}</span>
        </span>
        <span class="endpoint__chevron">${icon('arrowRight', 16)}</span>
      </button>
      <div class="endpoint__body">
        <p class="text-sm text-soft" style="margin-bottom:16px">${escapeHtml(endpoint.description)}</p>
        <div class="grid grid--2" style="gap:18px">
          <div>
            <h3 class="text-xs text-dim" style="text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px">Parameters</h3>
            ${paramsTable(endpoint)}
          </div>
          <div>
            <h3 class="text-xs text-dim" style="text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px">Response shape</h3>
            <pre class="code-block">${escapeHtml(JSON.stringify(endpoint.response, null, 2))}</pre>
          </div>
        </div>
        ${tryForm(endpoint)}
      </div>
    </article>`;
}

/* ------------------------------------------------------------------ *
 * Request execution
 * ------------------------------------------------------------------ */

function buildUrl(endpoint) {
  const card = document.querySelector(`[data-endpoint="${endpoint.id}"]`);
  const base = currentBase().replace(/\/$/, '');
  let path = endpoint.path.replace(/^\/api/, '');
  const query = new URLSearchParams();

  card?.querySelectorAll('[data-param]').forEach((inputEl) => {
    const key = inputEl.dataset.param;
    const value = inputEl.value.trim();
    if (!value) return;
    if (['category', 'search', 'ordering', 'page', 'n'].includes(key)) query.set(key, value);
    else path = path.replace(`<${key}>`, encodeURIComponent(value));
  });

  if (endpoint.id === 'products' && !query.has('page')) query.set('page', '1');
  const qs = query.toString();
  return `${base}${path}${qs ? `?${qs}` : ''}`;
}

async function send(endpoint) {
  const card = document.querySelector(`[data-endpoint="${endpoint.id}"]`);
  const statusEl = card.querySelector(`[data-status="${endpoint.id}"]`);
  const outEl = card.querySelector(`[data-output="${endpoint.id}"]`);
  const btn = card.querySelector(`[data-send="${endpoint.id}"]`);

  const url = buildUrl(endpoint);
  let body;
  const bodyInput = card.querySelector('[data-body]');
  if (bodyInput && endpoint.body) {
    try {
      body = JSON.parse(bodyInput.value);
    } catch {
      statusEl.innerHTML = `<span style="color:var(--bad)">Invalid JSON body</span>`;
      return;
    }
  }

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Sending';
  statusEl.textContent = url;
  outEl.style.display = 'block';
  outEl.textContent = '…';

  const started = performance.now();
  try {
    const res = await fetch(url, {
      method: endpoint.method,
      headers: body ? { 'Content-Type': 'application/json', Accept: 'application/json' } : { Accept: 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    });
    const elapsed = Math.round(performance.now() - started);
    const text = await res.text();
    let payload;
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
    statusEl.innerHTML = `<span class="pill pill--${res.ok ? 'positive' : 'negative'}">HTTP ${res.status}</span> <span class="text-xs text-dim">${elapsed} ms</span>`;
    outEl.textContent = typeof payload === 'string' ? payload.slice(0, 4000) : JSON.stringify(payload, null, 2).slice(0, 8000);
    toast(`${endpoint.method} ${endpoint.path} → ${res.status}`, {
      type: res.ok ? 'success' : 'error',
      title: res.ok ? 'Request completed' : 'Request failed',
    });
  } catch (err) {
    statusEl.innerHTML = `<span class="pill pill--negative">network error</span>`;
    outEl.textContent = describeError(err);
    toast(describeError(err), { type: 'error', title: 'Request blocked' });
  } finally {
    btn.disabled = false;
    btn.innerHTML = `${icon('play', 14)} Send request`;
  }
}

/* ------------------------------------------------------------------ *
 * Boot
 * ------------------------------------------------------------------ */

function renderGaps() {
  $('#gaps').innerHTML = Object.values(BACKEND_NOTES)
    .map(
      (n) => `<div class="card card--pad reveal">
        <h3 style="font-size:0.95rem;margin-bottom:8px">${escapeHtml(n.title)}</h3>
        <p class="text-sm text-soft">${escapeHtml(n.body)}</p>
      </div>`,
    )
    .join('');
}

async function main() {
  paintBaseNote();

  const groups = [...new Set(ENDPOINTS.map((e) => e.group))];
  $('#endpoint-list').innerHTML = groups
    .map(
      (group) => `
      <section class="section">
        <div class="section-head"><div><h2>${escapeHtml(group)}</h2><p>${ENDPOINTS.filter((e) => e.group === group).length} route(s)</p></div></div>
        <div class="stack">${ENDPOINTS.filter((e) => e.group === group).map(endpointCard).join('')}</div>
      </section>`,
    )
    .join('');

  renderGaps();

  document.querySelectorAll('.endpoint__head').forEach((head) => {
    head.addEventListener('click', () => {
      const card = head.closest('.endpoint');
      const open = card.classList.toggle('is-open');
      head.setAttribute('aria-expanded', String(open));
    });
  });

  document.querySelectorAll('[data-send]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const endpoint = ENDPOINTS.find((e) => e.id === btn.dataset.send);
      if (endpoint) send(endpoint);
    });
  });

  // Recompute-weights summary so the tuning reference is discoverable here too.
  $('#endpoint-list').insertAdjacentHTML(
    'beforeend',
    `<section class="section">
      <div class="section-head"><div><h2>Tuning constants</h2><p>Loaded from <code>recsys_backend/settings.py</code>; the same values drive this frontend's ranker.</p></div></div>
      <div class="grid grid--kpi">
        ${[
          ['RECSYS_SVD_WEIGHT', RECSYS.svdWeight, 'SVD / collaborative weight'],
          ['RECSYS_CBF_WEIGHT', RECSYS.cbfWeight, 'content-based weight'],
          ['RECSYS_ALPHA', RECSYS.alpha, 'product-bias damping'],
          ['RECSYS_BETA', RECSYS.beta, 'category-affinity fallback'],
          ['RECSYS_GAMMA', RECSYS.gamma, 'sentiment weight'],
          ['RECSYS_BASE', RECSYS.base, 'additive baseline'],
          ['RECSYS_MAX_PER_CAT', RECSYS.maxPerCat, 'diversity cap'],
          ['CBF_TEXT_WEIGHT', RECSYS.textWeight, 'TF-IDF vs numeric split'],
        ]
          .map(
            ([name, value, note]) => `<div class="card kpi"><p class="kpi__label">${escapeHtml(name)}</p><p class="kpi__value">${escapeHtml(String(value))}</p><p class="kpi__delta">${escapeHtml(note)}</p></div>`,
          )
          .join('')}
      </div>
    </section>`,
  );
}

await bootstrapData();
await main();
