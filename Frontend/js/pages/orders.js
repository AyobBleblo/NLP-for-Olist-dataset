/**
 * js/pages/orders.js — orders produced by the checkout endpoint.
 *
 * NOTE: SETUP.md advertises GET /api/orders/ but store/urls.py does not define
 * it. This page therefore reads the order rows the store actually holds and
 * clearly flags the missing endpoint instead of pretending it exists.
 */

import { initPage, bootstrapData, query } from '../main.js';
import { api, describeError } from '../api.js';
import { getCustomerId, getPersona, onCustomerChange, setCustomerId } from '../session.js';
import { siteBanner } from '../layout.js';
import { $, emptyState, errorState, icon, productArt, skeletonRows } from '../ui.js';
import { BACKEND_NOTES } from '../config.js';
import { dateTime, escapeHtml, money, num, relativeTime } from '../format.js';

initPage({ active: 'orders.html', title: 'Orders' });

if (query('customer')) setCustomerId(query('customer'));

const host = $('#orders-section');

const STATUS_TONE = {
  delivered: 'positive',
  shipped: 'brand',
  processing: 'mixed',
  invoiced: 'brand',
  approved: 'positive',
  created: 'ghost',
  canceled: 'negative',
  unavailable: 'negative',
};

function orderCard(order, productIndex) {
  const lines = order.items
    .map((item) => {
      const p = item.product || productIndex?.get(item.product_external_id);
      const name = p ? p.name_translated : (item.product_name || item.product_external_id);
      return `
        <div class="order-line">
          ${p ? productArt(p, 'sm') : '<div class="art art--sm"></div>'}
          <div class="order-line__main">
            <strong>${p ? `<a href="product.html?id=${encodeURIComponent(p.external_id)}">${escapeHtml(name)}</a>` : escapeHtml(name)}</strong>
            <small>${escapeHtml(money(item.price))} × ${num(item.quantity)} · ${escapeHtml(item.product_external_id.slice(0, 14))}…</small>
          </div>
          <span class="price">${escapeHtml(money(item.price * item.quantity))}</span>
        </div>`;
    })
    .join('');

  const computed = order.items.reduce((s, i) => s + i.price * i.quantity, 0);

  return `
    <article class="order-card reveal">
      <div class="order-card__head">
        <div class="row" style="gap:12px">
          <span class="stat__icon">${icon('receipt', 18)}</span>
          <div>
            <strong class="mono text-sm">${escapeHtml(order.external_id)}</strong>
            <p class="text-xs text-dim">Placed ${escapeHtml(dateTime(order.purchased_at))} · ${escapeHtml(relativeTime(order.purchased_at))}</p>
          </div>
        </div>
        <div class="row" style="gap:10px">
          <span class="pill pill--${STATUS_TONE[order.status] || 'ghost'}">${escapeHtml(order.status)}</span>
          <span class="price">${escapeHtml(money(computed || order.total))}</span>
        </div>
      </div>
      <div class="order-card__body">${lines || '<p class="text-sm text-dim">No order items stored for this order.</p>'}</div>
    </article>`;
}

async function load() {
  const persona = getPersona();
  host.innerHTML = `<div class="card card--pad">${skeletonRows(3)}</div>`;

  try {
    const orders = await api.getOrders(persona.id);
    const allProducts = await api.getProducts({ page: 1, pageSize: 500 });
    const productIndex = new Map(allProducts.results.map((p) => [p.external_id, p]));

    if (!orders.length) {
      host.innerHTML = emptyState({
        title: `No orders for ${persona.name}`,
        body: 'Complete a checkout to create your first order — the backend copies the cart into Order + OrderItem rows.',
        action: `<div class="row"><a class="btn btn--primary" href="catalog.html">${icon('box', 16)} Browse catalogue</a>
                 <a class="btn btn--ghost" href="cart.html">${icon('cart', 16)} Open cart</a></div>`,
        iconName: 'receipt',
      });
      return;
    }

    const totalSpent = orders.reduce(
      (s, o) => s + (o.items.reduce((t, i) => t + i.price * i.quantity, 0) || o.total),
      0,
    );
    const lineCount = orders.reduce((s, o) => s + o.items.length, 0);

    host.innerHTML = `
      <div class="grid grid--kpi" style="margin-bottom:26px">
        <div class="card stat"><p class="stat__label">Orders</p><p class="stat__value">${orders.length}</p><p class="stat__hint">for ${escapeHtml(persona.name)}</p></div>
        <div class="card stat"><p class="stat__label">Total items</p><p class="stat__value">${lineCount}</p><p class="stat__hint">Purchased products</p></div>
        <div class="card stat"><p class="stat__label">Total value</p><p class="stat__value">${escapeHtml(money(totalSpent))}</p><p class="stat__hint">Lifetime purchases</p></div>
        <div class="card stat"><p class="stat__label">Latest</p><p class="stat__value" style="font-size:1.05rem">${escapeHtml(relativeTime(orders[0].purchased_at))}</p><p class="stat__hint">${escapeHtml(orders[0].status)}</p></div>
      </div>
      <div class="stack stack--lg">${orders.map((o) => orderCard(o, productIndex)).join('')}</div>`;
  } catch (err) {
    host.innerHTML = errorState({ title: 'Could not load orders', body: describeError(err) });
    host.querySelector('[data-action="retry"]')?.addEventListener('click', load);
  }
}

onCustomerChange(load);

(async function boot() {
  await bootstrapData();
  await load();
})();
