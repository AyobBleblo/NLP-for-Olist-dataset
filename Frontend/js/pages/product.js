/**
 * js/pages/product.js — product detail card.
 * Uses GET /api/products/<lookup_value>/ which accepts either the numeric PK
 * or the 32-char Olist external_id (exactly how the backend resolves it).
 */

import { initPage, bootstrapData, query, observeReveals } from '../main.js';
import { api, describeError } from '../api.js';
import { getCustomerId } from '../session.js';
import { refreshCartBadge } from '../layout.js';
import {
  $, bar, categoryChip, errorState, gauge, icon, productCard, sentimentPill, skeletonCards, stars, toast,
} from '../ui.js';
import { dateShort, escapeHtml, money, num, prettyCategory } from '../format.js';

initPage({ active: 'catalog.html', title: 'Product' });

const host = $('#product-detail');
const lookup = query('id');

function dims(product) {
  const d = product.dims || {};
  const rows = [
    ['Weight', d.weight_g ? `${num(d.weight_g)} g` : '—'],
    ['Length', d.length_cm ? `${num(d.length_cm)} cm` : '—'],
    ['Height', d.height_cm ? `${num(d.height_cm)} cm` : '—'],
    ['Width', d.width_cm ? `${num(d.width_cm)} cm` : '—'],
  ];
  return rows.map(([k, v]) => `<div class="kv"><dt>${k}</dt><dd>${escapeHtml(v)}</dd></div>`).join('');
}

function reviewCard(r) {
  return `
    <article class="card card--pad reveal" style="gap:10px">
      <div class="row row--between">
        ${stars(r.rating)}
        <span class="pill pill--${r.sentiment === 'positive' ? 'positive' : r.sentiment === 'negative' ? 'negative' : 'mixed'}">
          ${escapeHtml(r.sentiment || 'unscored')} · ${r.sentiment_score.toFixed(2)}
        </span>
      </div>
      <p class="text-sm">${escapeHtml(r.comment_text_translated || r.comment_text)}</p>
      <p class="text-xs text-dim">PT: ${escapeHtml(r.comment_text)}</p>
      <p class="text-xs text-dim">${icon('info', 12)} ${escapeHtml(dateShort(r.created_at))} · review ${escapeHtml(r.external_id.slice(0, 10))}…</p>
    </article>`;
}

function paint(product, reviews, related) {
  const name = product.name_translated || product.name;
  $('#crumb-current').textContent = name;
  document.title = `${name} · Olist Recommender`;

  host.innerHTML = `
    <div class="split split--detail">
      <div class="stack stack--lg">
        <div class="card card--flush reveal">
          <div class="art art--lg" style="--art-hue:${(parseInt(product.external_id.slice(0, 2), 16) % 360)}">
            <span class="art__fallback">${escapeHtml(name.replace(/#[0-9A-F]+/i, '').trim().split(/\s+/).slice(0, 2).map((w) => w[0]).join('').toUpperCase())}</span>
            ${product.image_url ? `<img src="${escapeHtml(product.image_url)}" alt="${escapeHtml(name)}" onerror="this.remove()">` : ''}
          </div>
        </div>

        <div class="card card--pad reveal">
          <div class="row row--between" style="margin-bottom:12px">
            <h2 style="font-size:1rem">Review sentiment</h2>
            ${sentimentPill(product.avg_sentiment_score)}
          </div>
          <div class="row row--wrap" style="gap:26px;align-items:center">
            ${gauge(product.avg_sentiment_score)}
            <div style="flex:1;min-width:220px" class="stack stack--sm">
              <p class="text-sm text-soft">
                Sentiment Index aggregates authentic buyer reviews scored with DistilBERT NLP to measure customer satisfaction.
              </p>
              <div class="kv"><dt>Verified reviews</dt><dd>${num(reviews.length)}</dd></div>
              <div class="kv"><dt>Satisfaction</dt><dd>${product.avg_sentiment_score >= 0.7 ? 'Highly Positive' : product.avg_sentiment_score >= 0.4 ? 'Positive' : 'Mixed'}</dd></div>
              <div class="kv"><dt>Availability</dt><dd style="color:var(--ok);font-weight:600">In Stock · Ready to ship</dd></div>
            </div>
          </div>
        </div>

        <div class="section--tight">
          <div class="section-head"><div><h2>Customer Reviews (${reviews.length})</h2><p>Verified buyer reviews with English translations.</p></div></div>
          <div class="stack" id="reviews-host">
            ${reviews.length ? reviews.slice(0, 6).map(reviewCard).join('') : '<div class="state state--empty"><h3 class="state__title">No reviews yet for this product</h3><p class="state__body">Be the first to review this product after purchase!</p></div>'}
          </div>
        </div>
      </div>

      <aside class="stack stack--lg">
        <div class="card card--pad reveal">
          ${categoryChip(product.category_name_original)}
          <h1 style="font-size:1.5rem;margin:10px 0 4px">${escapeHtml(name)}</h1>
          <p class="text-sm text-dim">${escapeHtml(product.name)}</p>
          <p class="price price--lg" style="margin:18px 0 6px">${escapeHtml(money(product.price))}</p>
          <p class="text-xs text-dim">Prices in USD ($) · Inclusive of taxes</p>

          <div class="stack stack--sm" style="margin-top:22px">
            <button class="btn btn--primary btn--block btn--lg" type="button" id="add-btn">${icon('cart', 17)} Add to cart</button>
            <button class="btn btn--outline btn--block" type="button" id="buy-now">${icon('check', 16)} Add &amp; go to cart</button>
          </div>

          <div class="stack stack--sm" style="margin-top:22px">
            <div class="kv"><dt>SKU</dt><dd class="mono">#${product.id}</dd></div>
            <div class="kv"><dt>Product Code</dt><dd class="mono truncate" style="max-width:190px" title="${escapeHtml(product.external_id)}">${escapeHtml(product.external_id.slice(0, 12))}…</dd></div>
            <div class="kv"><dt>Department</dt><dd>${escapeHtml(product.category_name)}</dd></div>
            <div class="kv"><dt>Total Sold</dt><dd>${num(product.total_purchases)} orders</dd></div>
          </div>
        </div>

        <div class="card card--pad reveal">
          <h2 style="font-size:0.95rem;margin-bottom:12px">Shipping specifications</h2>
          ${dims(product)}
          <p class="text-xs text-dim" style="margin-top:12px">
            Packaged dimensions for secure nationwide delivery across Brazil.
          </p>
        </div>

        <div class="card card--pad reveal">
          <h2 style="font-size:0.95rem;margin-bottom:12px">Popularity &amp; Sales Velocity</h2>
          ${bar(product.total_purchases, Math.max(40, product.total_purchases), product.total_purchases > 15 ? 'bar--ok' : '')}
          <p class="text-xs text-dim" style="margin-top:10px">${num(product.total_purchases)} verified purchases across Brazil.</p>
        </div>
      </aside>
    </div>

    <section class="section">
      <div class="section-head">
        <div><h2>More from ${escapeHtml(prettyCategory(product.category_name_original))}</h2><p>Popular items from the same department.</p></div>
        <a class="btn btn--ghost btn--sm" href="catalog.html?category=${encodeURIComponent(product.category_name_original)}">Browse department</a>
      </div>
      <div class="grid grid--products" id="related-grid">${skeletonCards(4)}</div>
    </section>`;

  const addBtn = $('#add-btn');
  const buyBtn = $('#buy-now');

  const add = async (thenGo) => {
    addBtn.disabled = true;
    buyBtn.disabled = true;
    const original = addBtn.innerHTML;
    addBtn.innerHTML = '<span class="spinner"></span> Adding';
    try {
      await api.addToCart(getCustomerId(), { product_id: product.id, quantity: 1 });
      toast(`${name} added to your cart`, { type: 'success', title: 'Added to cart' });
      refreshCartBadge();
      if (thenGo) location.href = 'cart.html';
      else {
        addBtn.innerHTML = `${icon('check', 16)} In cart`;
        setTimeout(() => {
          addBtn.innerHTML = original;
        }, 1600);
      }
    } catch (err) {
      toast(describeError(err), { type: 'error', title: 'Could not add to cart' });
      addBtn.innerHTML = original;
    } finally {
      addBtn.disabled = false;
      buyBtn.disabled = false;
    }
  };

  addBtn.addEventListener('click', () => add(false));
  buyBtn.addEventListener('click', () => add(true));

  loadRelated(product);
  observeReveals(host);
}

async function loadRelated(product) {
  const grid = $('#related-grid');
  try {
    const data = await api.getProducts({ category: product.category_name_original, ordering: '-total_purchases', page: 1, pageSize: 500 });
    const others = data.results.filter((p) => p.external_id !== product.external_id).slice(0, 4);
    grid.innerHTML = others.length
      ? others.map((p) => productCard(p, { compact: true })).join('')
      : `<p class="text-sm text-dim">No other products in this category.</p>`;
  } catch (err) {
    grid.innerHTML = `<p class="text-sm" style="color:var(--bad)">${escapeHtml(describeError(err))}</p>`;
  }
}

function paintError(err) {
  host.innerHTML = errorState({
    title: err?.status === 404 ? 'Product not found' : 'Could not load this product',
    body:
      err?.status === 404
        ? `No product matches the lookup value “${lookup}”. The endpoint accepts a numeric id or a 32-char external_id.`
        : describeError(err),
  });
  host.querySelector('[data-action="retry"]')?.addEventListener('click', load);
  host.insertAdjacentHTML(
    'beforeend',
    `<div class="center" style="margin-top:20px"><a class="btn btn--ghost" href="catalog.html">${icon('arrowLeft', 16)} Back to catalogue</a></div>`,
  );
}

async function load() {
  if (!lookup) {
    paintError({ status: 404, message: 'No ?id= parameter supplied.' });
    return;
  }
  host.innerHTML = `<div class="grid grid--products">${skeletonCards(2)}</div>`;
  try {
    const product = await api.getProduct(lookup);
    const reviews = await api.getReviews(product.external_id);
    const sorted = [...reviews].sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
    paint(product, sorted);
  } catch (err) {
    paintError(err);
  }
}

(async function boot() {
  await bootstrapData();
  await load();
})();
