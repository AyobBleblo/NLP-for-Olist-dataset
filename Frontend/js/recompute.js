/**
 * js/recompute.js — writes Recommendation rows using the ported engine.
 *
 * This is the browser-side counterpart of the offline management command
 * `manage.py compute_recommendations` (backend/store/management/commands/
 * compute_recommendations.py). The backend never scores during a request;
 * this module keeps the same separation by running as an explicit action.
 */

import { API } from './config.js';
import { api } from './api.js';
import { buildIdf, computeRecommendations } from './engine.js';

const tablesBase = () => API.localBase;

async function post(table, row) {
  const res = await fetch(`${tablesBase()}${table}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(row),
  });
  if (!res.ok && res.status !== 409) throw new Error(`${table}: HTTP ${res.status}`);
}

async function del(table, id) {
  const res = await fetch(`${tablesBase()}${table}/${id}`, { method: 'DELETE' });
  if (!res.ok && res.status !== 404 && res.status !== 204) throw new Error(`${table}: HTTP ${res.status}`);
}

async function all(table, limit = 500) {
  const out = [];
  let page = 1;
  for (;;) {
    const res = await fetch(`${tablesBase()}${table}?page=${page}&limit=${limit}`, { headers: { Accept: 'application/json' } });
    if (!res.ok) break;
    const json = await res.json();
    const rows = json?.data ?? [];
    out.push(...rows);
    const total = Number(json?.total ?? out.length);
    if (!rows.length || out.length >= total || page > 20) break;
    page += 1;
  }
  return out;
}

/**
 * Recompute the shelf for one customer, mirroring `--customer-id`.
 * Clears that customer's existing rows first (mirrors `--clear`).
 */
export async function recomputeForCustomer(customerExternalId, topN = 10) {
  if (API.mode === 'remote') {
    throw new Error(
      'Recompute is an offline management command in the backend. In remote mode run ' +
        '`uv run python manage.py compute_recommendations --clear` instead.',
    );
  }

  const [rawProducts, rawOrders, rawItems, rawCats, existing] = await Promise.all([
    all('products'),
    all('orders'),
    all('order_items'),
    all('categories'),
    all('recommendations'),
  ]);

  const catIndex = new Map(rawCats.map((c) => [c.name, c]));
  const products = rawProducts.map((r) => ({
    external_id: r.external_id,
    category_name: r.category_name,
    price: Number(r.price || 0),
    avg_sentiment_score: Number(r.avg_sentiment_score || 0),
    total_purchases: Number(r.total_purchases || 0),
    dims: {
      weight_g: Number(r.product_weight_g || 0),
      length_cm: Number(r.product_length_cm || 0),
      height_cm: Number(r.product_height_cm || 0),
      width_cm: Number(r.product_width_cm || 0),
    },
    category_name_translated: catIndex.get(r.category_name)?.name_translated || r.category_name,
  }));

  const myOrderIds = new Set(
    rawOrders.filter((o) => o.customer_external_id === customerExternalId).map((o) => o.external_id),
  );
  const purchases = rawItems.filter((i) => myOrderIds.has(i.order_external_id));

  const idf = buildIdf(products);
  const result = computeRecommendations({
    customer: { id: customerExternalId },
    products,
    purchases,
    idf,
    topN,
  });

  // Clear this customer's previous rows (compute_recommendations --clear).
  const mine = existing.filter((r) => r.customer_external_id === customerExternalId);
  for (const row of mine) await del('recommendations', row.id);

  const now = new Date().toISOString();
  let written = 0;
  for (const row of result.results) {
    await post('recommendations', {
      id: `rec_${customerExternalId.slice(0, 8)}_${row.rank}_${Date.now().toString(36)}`,
      customer_external_id: customerExternalId,
      product_external_id: row.product.external_id,
      rank: row.rank,
      score: Math.round(row.score * 10000) / 10000,
      svd_component: Math.round(row.breakdown.svd * 10000) / 10000,
      cbf_component: Math.round(row.breakdown.cbf * 10000) / 10000,
      sentiment_component: Math.round(row.breakdown.sentiment * 10000) / 10000,
      generated_at: now,
    });
    written += 1;
  }

  void api;
  return { written, pool: result.pool, profile: result.profile, results: result.results };
}
