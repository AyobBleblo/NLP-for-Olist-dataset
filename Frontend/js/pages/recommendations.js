/**
 * js/pages/recommendations.js — the recommendation shelf.
 * Renders GET /api/recommendations/<customer_id>/?n= and explains the score
 * decomposition using the real RECSYS_* weights.
 */

import { initPage, bootstrapData, query, observeReveals } from '../main.js';
import { api, describeError } from '../api.js';
import { getCustomerId, getPersona, onCustomerChange, setCustomerId } from '../session.js';
import { $, errorState, icon, productArt, sentimentPill, skeletonCards, toast } from '../ui.js';
import { RECSYS } from '../config.js';
import { escapeHtml, money, prettyCategory, score as fmtScore } from '../format.js';
initPage({ active: 'recommendations.html', title: 'Recommendations' });

const state = { n: 10 };

if (query('customer')) setCustomerId(query('customer'));

const recHost = $('#rec-section');
const personaHost = $('#persona-summary');

function personaSummary() {
  const p = getPersona();
  return `
    <div class="card card--pad">
      <div class="row row--wrap" style="gap:20px">
        <span class="persona-card__avatar" style="--accent:${p.accent}">${escapeHtml(p.name.split(' ').map((s) => s[0]).join(''))}</span>
        <div style="flex:1;min-width:200px">
          <h2 style="font-size:1.1rem">${escapeHtml(p.name)}</h2>
          <p class="text-sm text-soft">${escapeHtml(p.persona)}</p>
          <p class="text-xs text-dim" style="margin-top:4px">${icon('user', 12)} Shopper Profile: ${escapeHtml(p.name)}</p>
        </div>
        <div class="stack stack--sm" style="min-width:180px">
          <div class="kv"><dt>Location</dt><dd>${escapeHtml(p.city)}, ${escapeHtml(p.state)}</dd></div>
          <div class="kv"><dt>Preferred department</dt><dd>${escapeHtml(p.category_en)}</dd></div>
          <div class="kv"><dt>Displayed items</dt><dd>${state.n} curated</dd></div>
        </div>
      </div>
    </div>`;
}

function recCard(row) {
  const p = row.product;
  const b = row.breakdown || {};
  const bars = [
    ['SVD (CF)', b.svd, RECSYS.svdWeight * (RECSYS.base + 2)],
    ['CBF (TF-IDF + numeric)', b.cbf, RECSYS.cbfWeight],
    ['Sentiment', b.sentiment + 0.15, 0.3],
  ];
  return `
    <article class="card rec-card reveal">
      <span class="rec-card__rank" title="Rank ${row.rank}">${row.rank}</span>
      <a class="rec-card__media" href="product.html?id=${encodeURIComponent(p.external_id)}" aria-label="${escapeHtml(p.name_translated)}">
        ${productArt(p, 'sm')}
      </a>
      <div class="rec-card__main">
        <div class="row row--between" style="gap:12px">
          <div style="min-width:0">
            <a href="product.html?id=${encodeURIComponent(p.external_id)}" style="font-weight:600;font-size:0.95rem">${escapeHtml(p.name_translated)}</a>
            <p class="text-xs text-dim">${escapeHtml(p.name)}</p>
          </div>
          <div class="rec-card__score">
            <strong>${fmtScore(row.score, 4)}</strong>
            <span class="text-xs text-dim">score</span>
          </div>
        </div>

        <div class="row row--wrap" style="gap:8px">
          <span class="tag">${icon('tag', 12)} ${escapeHtml(prettyCategory(p.category_name))}</span>
          <span class="tag">${escapeHtml(money(p.price))}</span>
          <span class="tag">${icon('activity', 12)} ${p.total_purchases} sold</span>
          ${sentimentPill(p.avg_sentiment_score)}
        </div>

        ${
          row.breakdown && (row.breakdown.svd || row.breakdown.cbf)
            ? `
        <div class="rec-card__breakdown">
          ${bars
            .map(
              ([label, value, max]) => `
            <span style="flex:1;min-width:130px">
              <span style="display:flex;justify-content:space-between;gap:8px">
                <span>${escapeHtml(label)}</span><b>${Number(value || 0).toFixed(3)}</b>
              </span>
              <span class="bar" style="margin-top:4px"><span class="bar__fill" style="width:${Math.max(3, Math.min(100, ((Number(value) || 0) / max) * 100))}%"></span></span>
            </span>`,
            )
            .join('')}
        </div>`
            : `
        <div class="row" style="margin-top:8px">
          <span class="tag" style="background:rgba(99,102,241,0.12);color:#a5b4fc;font-size:0.75rem">
            ${icon('sparkles', 12)} SVD Collaborative Filtering + CBF Hybrid Score
          </span>
        </div>`
        }

        <p class="text-xs text-dim">
          ${icon('shield', 12)} Verified In-Stock · Free Shipping on orders over $150
        </p>
      </div>
      <div class="rec-card__actions">
        <button class="btn btn--primary btn--sm" type="button" data-add-to-cart="${escapeHtml(p.external_id)}">${icon('cart', 14)} Add</button>
        <a class="btn btn--ghost btn--sm" href="product.html?id=${encodeURIComponent(p.external_id)}">Details</a>
      </div>
    </article>`;
}

function explainer() {
  return `
    <div class="card card--pad">
      <h2 style="font-size:1.05rem;margin-bottom:6px">How Your Recommendations Are Ranked</h2>
      <p class="text-sm text-dim" style="margin-bottom:18px">Our hybrid recommendation engine blends customer behavioral signals, verified review sentiment, and product features to produce your curated list.</p>
      <div class="grid grid--kpi" style="margin-bottom:20px">
        <div class="kpi"><p class="kpi__label">Taste Match</p><p class="kpi__value">60%</p><p class="kpi__delta">Collaborative Filtering</p></div>
        <div class="kpi"><p class="kpi__label">Attribute Match</p><p class="kpi__value">40%</p><p class="kpi__delta">Content Similarity</p></div>
        <div class="kpi"><p class="kpi__label">Sentiment Boost</p><p class="kpi__value">+0.30</p><p class="kpi__delta">Verified Review NLP</p></div>
        <div class="kpi"><p class="kpi__label">Diversity Cap</p><p class="kpi__value">Max 2</p><p class="kpi__delta">Items per category</p></div>
      </div>
      <div class="grid grid--2">
        <div class="fact"><span>Collaborative Taste Filtering</span><p>Identifies items loved by shoppers who share similar purchase histories and product rating affinities.</p></div>
        <div class="fact"><span>Feature &amp; Dimension Similarity</span><p>Compares descriptions, specifications, and physical attributes to items you have previously explored.</p></div>
        <div class="fact"><span>Verified Review Intelligence</span><p>Boosts products with overwhelmingly verified positive reviews using deep learning sentiment classification.</p></div>
        <div class="fact"><span>Freshness &amp; Diversity Protection</span><p>Ensures varied recommendations across distinct departments so you always discover new products.</p></div>
      </div>
    </div>`;
}

function renderPurchaseHistory(orders) {
  const historyHost = $('#purchase-history-section');
  if (!historyHost) return;

  const allItems = [];
  (orders || []).forEach((o) => {
    (o.items || []).forEach((it) => {
      allItems.push({ ...it, purchased_at: o.purchased_at });
    });
  });

  if (!allItems.length) {
    historyHost.innerHTML = `
      <div class="card card--pad" style="border-left: 3px solid var(--brand, #6366f1);">
        <div class="row row--between" style="gap:16px">
          <div>
            <h3 style="font-size:0.98rem;margin-bottom:4px;display:flex;align-items:center;gap:8px">
              ${icon('receipt', 16)} Purchase History: 0 past orders (Cold-Start Persona)
            </h3>
            <p class="text-xs text-dim">
              This customer hasn't purchased anything yet. The recommendations below are initialized by their category interest tags.
              Once they check out an order, this section will display their purchases and the recommendations will dynamically adapt!
            </p>
          </div>
          <a class="btn btn--ghost btn--sm" href="catalog.html">${icon('box', 14)} Make a purchase</a>
        </div>
      </div>`;
    return;
  }

  historyHost.innerHTML = `
    <div class="card card--pad">
      <div class="row row--between" style="margin-bottom:14px;gap:12px">
        <div>
          <h3 style="font-size:1rem;display:flex;align-items:center;gap:8px">
            ${icon('receipt', 18)} Customer Purchase History (${allItems.length} item${allItems.length > 1 ? 's' : ''})
          </h3>
          <p class="text-xs text-dim">
            The recommendation model analyzes these purchased items & categories to score and rank unpurchased products below.
          </p>
        </div>
        <span class="tag" style="background:rgba(16,185,129,0.12);color:#34d399">
          ${icon('check', 12)} ${orders.length} Approved Order${orders.length > 1 ? 's' : ''}
        </span>
      </div>

      <div class="row row--wrap" style="gap:12px">
        ${allItems
          .map((item) => {
            const p = item.product;
            const title = p?.name_translated || item.product_name || item.product_external_id;
            const cat = p?.category_name || 'Item';
            return `
            <div class="card card--pad" style="flex:1;min-width:240px;max-width:340px;background:rgba(255,255,255,0.03);padding:12px">
              <div class="row" style="gap:10px">
                ${p ? productArt(p, 'sm') : '<div class="art art--sm"></div>'}
                <div style="min-width:0;flex:1">
                  <p style="font-weight:600;font-size:0.88rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                    ${escapeHtml(title)}
                  </p>
                  <p class="text-xs text-dim">${icon('tag', 11)} ${escapeHtml(prettyCategory(cat))} · ${escapeHtml(money(item.price))}</p>
                </div>
              </div>
            </div>`;
          })
          .join('')}
      </div>
    </div>`;
}

async function load() {
  personaHost.innerHTML = personaSummary();
  recHost.innerHTML = `<div class="stack">${skeletonCards(4, 'rec-card')}</div>`;

  const persona = getPersona();
  try {
    const [data, orders] = await Promise.all([
      api.getRecommendations(persona.id, state.n),
      api.getOrders(persona.id),
    ]);

    renderPurchaseHistory(orders);

    if (!data.results.length) {
      recHost.innerHTML = `
        <div class="state state--empty">
          <span class="state__icon">${icon('heart', 26)}</span>
          <h3 class="state__title">No recommendations for this customer</h3>
          <p class="state__body">The backend returns an empty list when the Recommendation table holds no rows for the id.
          In the repo that means <code>compute_recommendations</code> has not been run yet.</p>
          <button class="btn btn--primary" type="button" id="compute-btn">Compute now</button>
        </div>`;
      $('#compute-btn')?.addEventListener('click', compute);
      return;
    }

    recHost.innerHTML = `
      <div class="section-head">
        <div>
          <h2>Recommended For You (${data.count} Products)</h2>
          <p>Personalized for ${escapeHtml(persona.name)} · ordered by recommendation score DESC</p>
        </div>
        <button class="btn btn--ghost btn--sm" type="button" id="recompute">${icon('refresh', 15)} Recompute</button>
      </div>
      <div class="stack">${data.results.map(recCard).join('')}</div>`;

    $('#recompute')?.addEventListener('click', compute);
    observeReveals(recHost);
  } catch (err) {
    recHost.innerHTML = errorState({ title: 'Could not load recommendations', body: describeError(err) });
    recHost.querySelector('[data-action="retry"]')?.addEventListener('click', load);
  }
}

async function compute() {
  const btn = document.getElementById('recompute') || document.getElementById('compute-btn');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Working';
  }
  try {
    if (api.mode === 'remote') {
      await load();
      toast(`Fetched recommendations from Django backend for ${getPersona().name}`, {
        type: 'success',
        title: 'Recommendations updated',
      });
    } else {
      const { recomputeForCustomer } = await import('../recompute.js');
      const result = await recomputeForCustomer(getPersona().id, state.n);
      toast(`${result.written} recommendation rows written for ${getPersona().name}`, {
        type: 'success',
        title: 'compute_recommendations complete',
      });
      await load();
    }
  } catch (err) {
    toast(describeError(err), { type: 'error', title: 'Action failed' });
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `${icon('refresh', 15)} Refresh`;
    }
  }
}

$('#n-select').addEventListener('change', (e) => {
  state.n = Number(e.target.value);
  load();
});

onCustomerChange(() => load());

(async function boot() {
  await bootstrapData();
  await load();
  $('#score-explainer').innerHTML = explainer();
})();
