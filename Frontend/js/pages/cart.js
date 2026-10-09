/**
 * js/pages/cart.js — the five cart endpoints from backend/store/urls.py.
 * View / add / update quantity / remove / checkout, with real validation
 * error handling (quantity ≥ 1, empty-cart checkout → 400).
 */

import { initPage, bootstrapData } from '../main.js';
import { api, describeError } from '../api.js';
import { getCustomerId, getPersona, onCustomerChange } from '../session.js';
import { refreshCartBadge } from '../layout.js';
import { $, emptyState, errorState, icon, modal, productArt, skeletonRows, toast } from '../ui.js';
import { escapeHtml, money, prettyCategory } from '../format.js';

initPage({ active: 'cart.html', title: 'Cart' });

const itemsHost = $('#cart-items');
const summaryHost = $('#cart-summary');

function lineRow(item) {
  const p = item.product;
  if (!p) {
    return `<div class="cart-line"><p class="text-sm" style="color:var(--bad)">This line references a product that no longer exists (product_id ${escapeHtml(String(item.product_id))}).</p></div>`;
  }
  return `
    <article class="cart-line" data-item="${escapeHtml(item.id)}">
      <a href="product.html?id=${encodeURIComponent(p.external_id)}" aria-label="${escapeHtml(p.name_translated)}">${productArt(p, 'sm')}</a>
      <div class="cart-line__main">
        <strong><a href="product.html?id=${encodeURIComponent(p.external_id)}">${escapeHtml(p.name_translated)}</a></strong>
        <small>${escapeHtml(prettyCategory(p.category_name))} · ${escapeHtml(money(p.price))} each</small>
        <small class="mono">item_id ${escapeHtml(String(item.id))}</small>
      </div>
      <div class="cart-line__controls">
        <div class="qty" role="group" aria-label="Quantity for ${escapeHtml(p.name_translated)}">
          <button type="button" data-dec="${escapeHtml(item.id)}" ${item.quantity <= 1 ? 'disabled' : ''} aria-label="Decrease quantity">${icon('minus', 14)}</button>
          <span aria-live="polite">${item.quantity}</span>
          <button type="button" data-inc="${escapeHtml(item.id)}" aria-label="Increase quantity">${icon('plus', 14)}</button>
        </div>
        <span class="price">${escapeHtml(money(p.price * item.quantity))}</span>
        <button class="icon-btn" type="button" data-remove="${escapeHtml(item.id)}" aria-label="Remove ${escapeHtml(p.name_translated)}">${icon('trash', 15)}</button>
      </div>
    </article>`;
}

function summary(cart) {
  const p = getPersona();
  const subtotal = cart.total;
  const shipping = subtotal > 0 ? 0 : 0;
  return `
    <div class="card card--pad summary">
      <h2 style="font-size:1rem;margin-bottom:16px">Order summary</h2>
      <div class="row" style="gap:10px;margin-bottom:16px">
        <span class="persona__avatar persona__avatar--sm" style="--accent:${p.accent}">${escapeHtml(p.name.split(' ').map((s) => s[0]).join(''))}</span>
        <div>
          <strong class="text-sm">${escapeHtml(p.name)}</strong>
          <p class="text-xs text-dim">${escapeHtml(p.city)}, ${escapeHtml(p.state)}</p>
        </div>
      </div>

      <div class="summary__line"><span>Items</span><span>${cart.items.reduce((s, i) => s + i.quantity, 0)}</span></div>
      <div class="summary__line"><span>Lines</span><span>${cart.items.length}</span></div>
      <div class="summary__line"><span>Shipping</span><span>${escapeHtml(money(shipping))}</span></div>
      <div class="summary__total">
        <span>Total</span>
        <strong class="price price--lg">${escapeHtml(money(subtotal))}</strong>
      </div>

      <button class="btn btn--primary btn--block btn--lg" style="margin-top:18px" type="button" id="checkout-btn"
        ${cart.items.length ? '' : 'disabled'}>
        ${icon('check', 17)} Checkout
      </button>
      <p class="text-xs text-dim" style="margin-top:10px">
        <code>POST /api/cart/&lt;id&gt;/checkout/</code> copies every line into an Order and empties the cart.
        There is no payment step in the backend.
      </p>
      <button class="btn btn--ghost btn--block btn--sm" style="margin-top:10px" type="button" id="clear-cart" ${cart.items.length ? '' : 'disabled'}>
        ${icon('trash', 14)} Empty cart
      </button>
    </div>`;
}

async function load() {
  $('#cart-customer').textContent = getPersona().name;
  itemsHost.innerHTML = `<div class="card card--pad">${skeletonRows(3)}</div>`;
  summaryHost.innerHTML = '';

  try {
    const cart = await api.getCart(getCustomerId());
    refreshCartBadge();

    if (!cart.items.length) {
      itemsHost.innerHTML = emptyState({
        title: 'Your cart is empty',
        body: 'Add a product from the catalogue or from your recommendation shelf to get started.',
        action: `<div class="row"><a class="btn btn--primary" href="catalog.html">${icon('box', 16)} Browse catalogue</a>
                 <a class="btn btn--ghost" href="recommendations.html">${icon('heart', 16)} Recommendations</a></div>`,
        iconName: 'cart',
      });
    } else {
      itemsHost.innerHTML = `<div class="card card--pad">${cart.items.map(lineRow).join('')}</div>`;
    }
    summaryHost.innerHTML = summary(cart);
    wire();
  } catch (err) {
    itemsHost.innerHTML = errorState({ title: 'Could not load the cart', body: describeError(err) });
    itemsHost.querySelector('[data-action="retry"]')?.addEventListener('click', load);
  }
}

async function setQuantity(itemId, next) {
  if (next < 1) return;
  const row = document.querySelector(`[data-item="${CSS.escape(itemId)}"]`);
  row?.classList.add('is-busy');
  try {
    await api.updateCartItem(getCustomerId(), itemId, next);
    await load();
    window.dispatchEvent(new CustomEvent('olist:cart-changed'));
  } catch (err) {
    row?.classList.remove('is-busy');
    toast(describeError(err), { type: 'error', title: 'Quantity not updated' });
  }
}

async function removeItem(itemId) {
  try {
    await api.removeCartItem(getCustomerId(), itemId);
    toast('Item removed from your cart', { type: 'info', title: 'Cart updated' });
    await load();
    window.dispatchEvent(new CustomEvent('olist:cart-changed'));
  } catch (err) {
    toast(describeError(err), { type: 'error', title: 'Could not remove item' });
  }
}

function wire() {
  itemsHost.querySelectorAll('[data-inc]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const row = btn.closest('[data-item]');
      const current = Number(row.querySelector('.qty span').textContent);
      setQuantity(btn.dataset.inc, current + 1);
    });
  });
  itemsHost.querySelectorAll('[data-dec]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const row = btn.closest('[data-item]');
      const current = Number(row.querySelector('.qty span').textContent);
      setQuantity(btn.dataset.dec, current - 1);
    });
  });
  itemsHost.querySelectorAll('[data-remove]').forEach((btn) => {
    btn.addEventListener('click', () => removeItem(btn.dataset.remove));
  });

  $('#clear-cart')?.addEventListener('click', async () => {
    const cart = await api.getCart(getCustomerId());
    const m = modal({
      title: 'Empty the cart?',
      size: 'sm',
      body: `<p class="modal__lead">This calls <code>DELETE /api/cart/&lt;id&gt;/remove/&lt;item_id&gt;/</code> once per line (${cart.items.length} requests).</p>`,
      actions: [
        {
          label: 'Remove all items',
          class: 'btn--danger',
          onClick: async () => {
            for (const item of cart.items) {
              try {
                await api.removeCartItem(getCustomerId(), item.id);
              } catch {
                /* continue */
              }
            }
            toast('Cart emptied', { type: 'info', title: 'Cart updated' });
            await load();
            window.dispatchEvent(new CustomEvent('olist:cart-changed'));
          },
        },
        { label: 'Cancel' },
      ],
    });
    void m;
  });

  $('#checkout-btn')?.addEventListener('click', async () => {
    const btn = $('#checkout-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Processing';
    try {
      const result = await api.checkout(getCustomerId());
      toast(`Order ${result.order_external_id.slice(0, 14)}… approved with ${result.items_count} item(s)!`, {
        type: 'success',
        title: 'Order Approved',
      });
      await load();
      window.dispatchEvent(new CustomEvent('olist:cart-changed'));
      const m = modal({
        title: 'Order Approved & Saved',
        size: 'sm',
        body: `
          <div class="fact-grid">
            <div class="fact"><span>Status</span><p><span class="pill pill--positive">approved</span></p></div>
            <div class="fact"><span>order_external_id</span><p class="mono">${escapeHtml(result.order_external_id)}</p></div>
            <div class="fact"><span>items_count</span><p>${escapeHtml(String(result.items_count))}</p></div>
          </div>
          <p class="modal__note">Your order is approved and saved to your account. Personalized recommendations have also been updated based on your purchase!</p>`,
        actions: [
          { label: 'View orders', class: 'btn--primary', onClick: () => { location.href = 'orders.html'; return false; } },
          { label: 'See recommendations', class: 'btn--ghost', onClick: () => { location.href = 'recommendations.html'; return false; } },
        ],
      });
      void m;
    } catch (err) {
      toast(describeError(err), { type: 'error', title: 'Checkout failed' });
      btn.disabled = false;
      btn.innerHTML = `${icon('check', 17)} Checkout`;
    }
  });
}

onCustomerChange(load);
window.addEventListener('olist:cart-changed', () => {
  /* cart page reloads itself; badge handled by layout */
});

(async function boot() {
  await bootstrapData();
  await load();
})();
