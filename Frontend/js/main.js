/**
 * js/main.js — shared page bootstrap.
 *
 * Every page calls initPage({ active, title }) which mounts the shell,
 * wires global behaviours (add-to-cart delegation, reveal animations) and
 * exposes small helpers used across pages.
 */

import { mountLayout, refreshCartBadge } from './layout.js';
import { getCustomerId } from './session.js';
import { api, describeError } from './api.js';
import { $$, icon, toast } from './ui.js';
import { ensureSeeded } from './seed.js';

export function initPage({ active = '', title = '' } = {}) {
  mountLayout({ active, title });
  observeReveals();
  wireGlobalAddToCart();
  return { active, title };
}

/**
 * Guarantees the local store holds the demo dataset before a page queries it.
 * No-op in remote mode (the Django API owns its own database).
 */
export async function bootstrapData({ onProgress } = {}) {
  return ensureSeeded({ onProgress });
}

/* ------------------------------------------------------------------ *
 * Global "Add to cart" — works for every product card on every page
 * ------------------------------------------------------------------ */

function wireGlobalAddToCart() {
  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-add-to-cart]');
    if (!btn || btn.disabled) return;
    e.preventDefault();

    const externalId = btn.dataset.addToCart;
    const original = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>';

    try {
      const product = await api.getProduct(externalId);
      await api.addToCart(getCustomerId(), { product_id: product.id, quantity: 1 });
      btn.innerHTML = `${icon('check', 15)} Added`;
      toast(`${product.name_translated || product.name} added to your cart`, {
        type: 'success',
        title: 'Added to cart',
      });
      refreshCartBadge();
      window.dispatchEvent(new CustomEvent('olist:cart-changed'));
      setTimeout(() => {
        btn.innerHTML = original;
        btn.disabled = false;
      }, 1400);
    } catch (err) {
      btn.innerHTML = original;
      btn.disabled = false;
      toast(describeError(err), { type: 'error', title: 'Could not add to cart' });
    }
  });
}

/* ------------------------------------------------------------------ *
 * Reveal-on-scroll
 *
 * Elements are inserted dynamically after load (KPI grids, pipeline
 * steps, benchmark cards), so a one-off scan at load time is not enough.
 * A document-wide MutationObserver keeps observing newly added nodes,
 * otherwise those elements would stay at opacity: 0.
 * ------------------------------------------------------------------ */

let revealIo = null;
let revealMo = null;

function showAll(root = document) {
  $$('.reveal:not(.is-in)', root).forEach((n) => n.classList.add('is-in'));
}

export function observeReveals(root = document) {
  if (!('IntersectionObserver' in window)) {
    showAll(root);
    return;
  }

  if (!revealIo) {
    revealIo = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-in');
            revealIo.unobserve(entry.target);
          }
        });
      },
      { rootMargin: '0px 0px -8% 0px', threshold: 0.05 },
    );
  }

  const scan = (node) => {
    if (node.nodeType !== 1) return;
    if (node.classList?.contains('reveal') && !node.classList.contains('is-in')) revealIo.observe(node);
    node.querySelectorAll?.('.reveal:not(.is-in)').forEach((n) => revealIo.observe(n));
  };

  if (root !== document) scan(root);
  $$('.reveal:not(.is-in)', root === document ? document : root).forEach((n) => revealIo.observe(n));

  if (!revealMo && document.body) {
    revealMo = new MutationObserver((mutations) => {
      mutations.forEach((m) => m.addedNodes.forEach(scan));
    });
    revealMo.observe(document.body, { childList: true, subtree: true });
  }
}

/* ------------------------------------------------------------------ *
 * Small page helpers
 * ------------------------------------------------------------------ */

/** Debounce for search inputs. */
export function debounce(fn, wait = 320) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), wait);
  };
}

/** Read a query-string parameter from the current URL. */
export function query(name, fallback = '') {
  return new URLSearchParams(location.search).get(name) ?? fallback;
}
