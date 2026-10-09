/**
 * js/pages/catalog.js — catalogue with the API's real search / filter /
 * ordering / pagination parameters.
 */

import { initPage, bootstrapData, debounce, query } from '../main.js';
import { api, describeError } from '../api.js';
import { $, emptyState, errorState, pagination, productCard, skeletonCards } from '../ui.js';
import { escapeHtml, sentimentBucket } from '../format.js';

initPage({ active: 'catalog.html', title: 'Catalogue' });

const state = {
  search: query('search'),
  category: query('category'),
  ordering: query('ordering', '-total_purchases,id'),
  tone: 'all',
  page: Number(query('page', '1')) || 1,
  categories: [],
};

const els = {
  grid: $('#product-grid'),
  footer: $('#catalog-footer'),
  count: $('#result-count'),
  summary: $('#filter-summary'),
  search: $('#search-input'),
  category: $('#category-select'),
  ordering: $('#ordering-select'),
  sentiment: $('#sentiment-filter'),
  clear: $('#clear-filters'),
};

function syncUrl() {
  const url = new URL(location.href);
  const set = (k, v) => (v ? url.searchParams.set(k, v) : url.searchParams.delete(k));
  set('search', state.search);
  set('category', state.category);
  set('ordering', state.ordering === '-total_purchases,id' ? '' : state.ordering);
  set('page', state.page > 1 ? String(state.page) : '');
  history.replaceState(null, '', url);
}

async function loadCategories() {
  try {
    state.categories = await api.getCategories();
    els.category.innerHTML =
      `<option value="">All categories (${state.categories.length})</option>` +
      state.categories
        .map(
          (c) =>
            `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name_translated)} · ${c.product_count}</option>`,
        )
        .join('');
    els.category.value = state.category;
  } catch (err) {
    els.category.innerHTML = `<option value="">Categories unavailable</option>`;
    void err;
  }
}

function paintControls() {
  els.search.value = state.search;
  els.ordering.value = state.ordering;
  els.sentiment.querySelectorAll('button').forEach((b) => {
    b.classList.toggle('is-active', b.dataset.tone === state.tone);
  });
  const bits = [];
  if (state.search) bits.push(`search “${state.search}”`);
  if (state.category) bits.push(`category ${state.category}`);
  if (state.tone !== 'all') bits.push(`sentiment ${state.tone}`);
  if (state.ordering !== '-total_purchases,id') bits.push(`ordered by ${state.ordering}`);
  els.summary.textContent = bits.length ? bits.join(' · ') : 'No filters applied';
}

async function load() {
  els.grid.innerHTML = skeletonCards(8);
  els.footer.innerHTML = '';
  els.count.textContent = 'Loading…';
  syncUrl();
  paintControls();

  try {
    const data = await api.getProducts({
      search: state.search,
      category: state.category,
      ordering: state.ordering,
      page: state.page,
      sentiment: state.tone !== 'all' ? state.tone : '',
    });

    const results = data.results || [];
    const total = data.count || 0;
    const pages = data.pages || 1;
    const page = data.page || state.page;
    state.page = page;

    els.count.textContent = `${total} product${total === 1 ? '' : 's'}`;
    els.summary.textContent += ` · Page ${page} of ${pages}`;

    if (!results.length) {
      els.grid.innerHTML = '';
      els.footer.innerHTML = emptyState({
        title: 'No products match these filters',
        body: 'Try a different search term, category or sentiment bucket.',
        iconName: 'search',
      });
      els.footer.insertAdjacentHTML(
        'beforeend',
        `<div class="center" style="margin-top:18px"><button class="btn btn--ghost" type="button" id="reset-empty">Clear all filters</button></div>`,
      );
      $('#reset-empty')?.addEventListener('click', resetAll);
      return;
    }

    els.grid.innerHTML = results
      .map((p, i) => productCard(p, { badge: page === 1 && i < 3 && !state.search ? 'Top seller' : '' }))
      .join('');

    els.footer.innerHTML = '';
    const nav = pagination({
      page,
      pages,
      total,
      onPage: (target) => {
        state.page = target;
        load();
        document.getElementById('catalog-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      },
    });
    els.footer.appendChild(nav);
  } catch (err) {
    els.grid.innerHTML = '';
    els.count.textContent = 'Error';
    els.footer.innerHTML = errorState({ title: 'Could not load the catalogue', body: describeError(err) });
    els.footer.querySelector('[data-action="retry"]')?.addEventListener('click', load);
  }
}

function resetAll() {
  state.search = '';
  state.category = '';
  state.ordering = '-total_purchases,id';
  state.tone = 'all';
  state.page = 1;
  els.category.value = '';
  load();
}

/* ---- wiring ---- */

els.search.addEventListener(
  'input',
  debounce(() => {
    state.search = els.search.value.trim();
    state.page = 1;
    load();
  }),
);

els.category.addEventListener('change', () => {
  state.category = els.category.value;
  state.page = 1;
  load();
});

els.ordering.addEventListener('change', () => {
  state.ordering = els.ordering.value;
  state.page = 1;
  load();
});

els.sentiment.addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-tone]');
  if (!btn) return;
  state.tone = btn.dataset.tone;
  state.page = 1;
  load();
});

els.clear.addEventListener('click', resetAll);

(async function boot() {
  await bootstrapData();
  await loadCategories();
  paintControls();
  await load();
})();
