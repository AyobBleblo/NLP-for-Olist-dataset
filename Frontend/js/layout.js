/**
 * js/layout.js — injects the shared shell (header / footer) and wires the
 * global interactions: customer switcher, cart badge, mobile nav, API-mode
 * indicator.
 */

import { APP, API, BACKEND_NOTES } from './config.js';
import { PERSONAS, getCustomerId, getPersona, onCustomerChange, setCustomerId } from './session.js';
import { api } from './api.js';
import { $, icon, modal, toast } from './ui.js';
import { escapeHtml } from './format.js';

const NAV = [
  { href: 'index.html', label: 'Home', icon: 'sparkles' },
  { href: 'catalog.html', label: 'Catalogue', icon: 'box' },
  { href: 'recommendations.html', label: 'Recommendations', icon: 'heart' },
  { href: 'orders.html', label: 'My Orders', icon: 'receipt' },
];

function navLinks(active) {
  return NAV.map(
    (n) => `<a class="nav__link ${n.href === active ? 'is-active' : ''}" href="${n.href}">
      ${icon(n.icon, 16)}<span>${escapeHtml(n.label)}</span>
    </a>`,
  ).join('');
}

function customerMenu() {
  return PERSONAS.map(
    (p) => `
    <button class="persona" type="button" data-customer="${p.id}" role="menuitemradio" aria-checked="false">
      <span class="persona__avatar" style="--accent:${p.accent}">${escapeHtml(p.name.split(' ').map((s) => s[0]).join(''))}</span>
      <span class="persona__text">
        <strong>${escapeHtml(p.name)}</strong>
        <small>${escapeHtml(p.persona)} · ${escapeHtml(p.city)}</small>
      </span>
      <span class="persona__check">${icon('check', 15)}</span>
    </button>`,
  ).join('');
}

export function mountLayout({ active = '', title = '' } = {}) {
  const header = document.createElement('header');
  header.className = 'site-header';
  header.innerHTML = `
    <div class="container site-header__inner">
      <a class="brand" href="index.html">
        <span class="brand__mark">${icon('sparkles', 20)}</span>
        <span class="brand__text">
          <strong>${escapeHtml(APP.name)}</strong>
          <small>Curated Brazilian Marketplace</small>
        </span>
      </a>

      <button class="nav-toggle" type="button" aria-expanded="false" aria-controls="primary-nav" aria-label="Toggle navigation">
        ${icon('menu', 20)}
      </button>

      <nav class="nav" id="primary-nav" aria-label="Primary">${navLinks(active)}</nav>

      <div class="site-header__actions">
        <div class="dropdown" id="customer-dropdown">
          <button class="btn btn--ghost btn--sm customer-trigger" type="button" aria-haspopup="true" aria-expanded="false">
            <span class="persona__avatar persona__avatar--sm" id="current-avatar"></span>
            <span class="customer-trigger__label" id="current-customer"></span>
            ${icon('user', 15)}
          </button>
          <div class="dropdown__menu" role="menu" aria-label="Switch customer">${customerMenu()}</div>
        </div>
        <a class="btn btn--primary btn--sm cart-btn" href="cart.html" aria-label="Open cart">
          ${icon('cart', 16)}<span>Cart</span><span class="cart-btn__count" id="cart-count">0</span>
        </a>
      </div>
    </div>`;

  const footer = document.createElement('footer');
  footer.className = 'site-footer';
  footer.innerHTML = `
    <div class="container site-footer__grid">
      <div>
        <a class="brand brand--footer" href="index.html">
          <span class="brand__mark">${icon('sparkles', 18)}</span>
          <span class="brand__text"><strong>${escapeHtml(APP.name)}</strong><small>Curated Brazilian Marketplace</small></span>
        </a>
        <p class="site-footer__note">
          Discover handpicked Brazilian products powered by verified buyer review sentiment and AI-tailored recommendations.
        </p>
      </div>
      <nav aria-label="Explore Catalogue">
        <h3>Shop</h3>
        <ul>
          <li><a href="index.html">Home</a></li>
          <li><a href="catalog.html">Full Catalogue</a></li>
          <li><a href="recommendations.html">For You (Recommendations)</a></li>
          <li><a href="orders.html">My Orders</a></li>
          <li><a href="cart.html">Shopping Cart</a></li>
        </ul>
      </nav>
      <div>
        <h3>Customer Care</h3>
        <ul>
          <li><a href="orders.html">Track Order Status</a></li>
          <li><a href="catalog.html">Shop All 30 Departments</a></li>
          <li><a href="catalog.html">Browse Best Sellers</a></li>
          <li><a href="recommendations.html">Personalized Offers</a></li>
        </ul>
      </div>
      <div>
        <h3>Store Guarantee</h3>
        <ul>
          <li><span>100% Authentic Brazilian Products</span></li>
          <li><span>Over 1,180+ Verified Reviews</span></li>
          <li><span>Real-time Hybrid Personalization</span></li>
          <li><span>Fast Nationwide Delivery</span></li>
        </ul>
      </div>
    </div>
    <div class="container site-footer__bottom">
      <span>&copy; 2026 ${escapeHtml(APP.name)} &middot; All rights reserved</span>
      <span>Authentic Brazilian E-Commerce Experience</span>
    </div>`;

  document.body.prepend(header);
  document.body.appendChild(footer);

  wireHeader(header);
  if (title) document.title = `${title} · ${APP.name}`;
  refreshCartBadge();
  return { header, footer };
}

function wireHeader(header) {
  const toggle = header.querySelector('.nav-toggle');
  const nav = header.querySelector('.nav');
  toggle?.addEventListener('click', () => {
    const open = nav.classList.toggle('is-open');
    toggle.setAttribute('aria-expanded', String(open));
  });

  const dd = header.querySelector('#customer-dropdown');
  const trigger = dd?.querySelector('.customer-trigger');
  trigger?.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = dd.classList.toggle('is-open');
    trigger.setAttribute('aria-expanded', String(open));
  });
  document.addEventListener('click', (e) => {
    if (dd && !dd.contains(e.target)) {
      dd.classList.remove('is-open');
      trigger?.setAttribute('aria-expanded', 'false');
    }
  });
  dd?.querySelectorAll('[data-customer]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.customer;
      if (id !== getCustomerId()) {
        setCustomerId(id);
        toast(`Now browsing as ${getPersona(id).name}`, { type: 'success', title: 'Customer switched' });
      }
      dd.classList.remove('is-open');
      trigger?.setAttribute('aria-expanded', 'false');
    });
  });

  const paint = () => {
    const p = getPersona();
    const avatar = header.querySelector('#current-avatar');
    const label = header.querySelector('#current-customer');
    if (avatar) {
      avatar.textContent = p.name.split(' ').map((s) => s[0]).join('');
      avatar.style.setProperty('--accent', p.accent);
    }
    if (label) label.textContent = p.name;
    dd?.querySelectorAll('[data-customer]').forEach((b) => {
      const on = b.dataset.customer === getCustomerId();
      b.classList.toggle('is-current', on);
      b.setAttribute('aria-checked', String(on));
    });
  };
  paint();
  onCustomerChange(paint);
}

export async function refreshCartBadge() {
  const badge = document.getElementById('cart-count');
  if (!badge) return;
  try {
    const cart = await api.getCart(getCustomerId());
    const count = (cart.items || []).reduce((s, i) => s + Number(i.quantity || 0), 0);
    badge.textContent = String(count);
    badge.classList.toggle('is-empty', count === 0);
  } catch {
    badge.textContent = '0';
  }
}

export function siteBanner({ title, body, tone = 'info', actions = '' }) {
  return `
    <section class="banner banner--${tone}">
      <div class="banner__icon">${icon(tone === 'warn' ? 'alert' : 'info', 20)}</div>
      <div class="banner__text">
        <h2>${escapeHtml(title)}</h2>
        <p>${body}</p>
      </div>
      ${actions ? `<div class="banner__actions">${actions}</div>` : ''}
    </section>`;
}

export { PERSONAS };
