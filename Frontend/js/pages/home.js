/**
 * js/pages/home.js — Homepage for Olist Store.
 * Production-level e-commerce storefront with live personalization.
 */

import { initPage, bootstrapData } from '../main.js';
import { api, describeError } from '../api.js';
import { getCustomerId, getPersona, PERSONAS, setCustomerId, onCustomerChange } from '../session.js';
import { $, icon, productCard, skeletonCards, stars, toast } from '../ui.js?v=2';
import { escapeHtml, money, num, prettyCategory } from '../format.js?v=2';

initPage({ active: 'index.html', title: 'Home' });

/* ------------------------------------------------------------------ *
 * Value Props / Trust Highlights
 * ------------------------------------------------------------------ */

const TRUST_PROPS = [
  {
    icon: 'shield',
    title: 'Verified Customer Reviews',
    body: 'Over 1,180+ authentic buyer reviews scored with NLP sentiment intelligence for genuine product quality.',
  },
  {
    icon: 'sparkles',
    title: 'AI-Powered Recommendations',
    body: 'Hybrid SVD collaborative filtering and content similarity tailored in real-time to your taste.',
  },
  {
    icon: 'box',
    title: '30 Curated Departments',
    body: '100 handpicked products across electronics, home decor, beauty, sports, fashion, and wellness.',
  },
  {
    icon: 'truck',
    title: 'Nationwide Delivery',
    body: 'Fast and dependable order fulfillment across all Brazilian states with live status updates.',
  },
];

function renderTrustProps() {
  const host = $('#trust-props');
  if (!host) return;
  host.innerHTML = TRUST_PROPS.map(
    (t) => `
    <article class="trust-item">
      <div class="trust-item__icon">${icon(t.icon, 22)}</div>
      <div class="trust-item__text">
        <h3>${escapeHtml(t.title)}</h3>
        <p>${escapeHtml(t.body)}</p>
      </div>
    </article>`,
  ).join('');
}

/* ------------------------------------------------------------------ *
 * Featured Categories
 * ------------------------------------------------------------------ */

const FEATURED_CATS = [
  { slug: 'informatica_acessorios', title: 'Computers & Tech', icon: 'zap', sub: 'Laptops, keyboards & peripherals' },
  { slug: 'beleza_saude', title: 'Health & Beauty', icon: 'heart', sub: 'Skincare, grooming & wellness' },
  { slug: 'relogios_presentes', title: 'Watches & Gifts', icon: 'award', sub: 'Chronographs, luxury & accessories' },
  { slug: 'esporte_lazer', title: 'Sports & Leisure', icon: 'activity', sub: 'Fitness, camping & outdoor gear' },
  { slug: 'cama_mesa_banho', title: 'Bed, Bath & Table', icon: 'box', sub: 'Linens, textiles & comfort' },
  { slug: 'moveis_decoracao', title: 'Furniture & Decor', icon: 'tag', sub: 'Modern living & interior styling' },
];

function renderFeaturedCategories() {
  const host = $('#category-grid');
  if (!host) return;
  host.innerHTML = FEATURED_CATS.map(
    (c) => `
    <a class="category-tile" href="catalog.html?category=${encodeURIComponent(c.slug)}">
      <span class="category-tile__icon">${icon(c.icon, 22)}</span>
      <div>
        <strong class="category-tile__title">${escapeHtml(c.title)}</strong>
        <span class="category-tile__sub">${escapeHtml(c.sub)}</span>
      </div>
    </a>`,
  ).join('');
}

/* ------------------------------------------------------------------ *
 * Shopper Persona Selector
 * ------------------------------------------------------------------ */

function renderPersonaCards() {
  const host = $('#persona-cards');
  if (!host) return;
  const currentId = getCustomerId();

  host.innerHTML = PERSONAS.map(
    (p) => `
    <article class="card card--pad card--hover ${p.id === currentId ? 'is-current' : ''}" style="${p.id === currentId ? 'border-color:var(--brand);box-shadow:var(--glow);' : ''}">
      <div class="row" style="gap:14px;margin-bottom:14px">
        <span class="persona__avatar" style="--accent:${p.accent}">${escapeHtml(p.name.split(' ').map((s) => s[0]).join(''))}</span>
        <div>
          <h3 style="font-size:1.02rem">${escapeHtml(p.name)}</h3>
          <p class="text-sm" style="color:var(--brand-2);font-weight:600">${escapeHtml(p.persona)}</p>
          <p class="text-xs text-dim">${escapeHtml(p.city)}, ${escapeHtml(p.state)}</p>
        </div>
      </div>
      <div class="row row--wrap" style="margin-bottom:18px">
        <span class="tag">${icon('tag', 13)} ${escapeHtml(p.category_en)}</span>
        ${p.id === currentId ? `<span class="pill pill--positive">${icon('check', 12)} Active Profile</span>` : ''}
      </div>
      <div class="row" style="gap:10px">
        <button class="btn ${p.id === currentId ? 'btn--ghost' : 'btn--primary'} btn--sm" type="button" data-switch-customer="${escapeHtml(p.id)}">
          ${p.id === currentId ? `${icon('check', 14)} Browsing as ${escapeHtml(p.name.split(' ')[0])}` : `Browse as ${escapeHtml(p.name.split(' ')[0])}`}
        </button>
        <a class="btn btn--ghost btn--sm" href="recommendations.html?customer=${encodeURIComponent(p.id)}">View For You</a>
      </div>
    </article>`,
  ).join('');

  host.querySelectorAll('[data-switch-customer]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.switchCustomer;
      if (id !== getCustomerId()) {
        setCustomerId(id);
        toast(`Browsing tailored feed for ${getPersona(id).name}`, { type: 'success', title: 'Shopper Profile Changed' });
        renderPersonaCards();
        renderHeroShelf();
      }
    });
  });
}

/* ------------------------------------------------------------------ *
 * Customer Testimonials / Verified Reviews
 * ------------------------------------------------------------------ */

const REVIEWS_SPOTLIGHT = [
  {
    quote: 'The product arrived well before the estimated delivery date, in pristine condition. Packaging was superb and the item exceeded my expectations!',
    author: 'Verified Buyer',
    location: 'São Paulo, SP',
    stars: 5,
    tag: '100% Positive Sentiment',
    product: 'High Performance PC Accessories',
  },
  {
    quote: 'Extremely satisfied with the build quality and finish. It matches the description completely and customer service was quick to provide delivery tracking.',
    author: 'Verified Buyer',
    location: 'Rio de Janeiro, RJ',
    stars: 5,
    tag: 'Verified Purchase',
    product: 'Luxury Chronograph & Fashion',
  },
  {
    quote: 'Excellent purchase experience! The recommendations shelf helped me discover matching items I did not know existed. Will definitely shop again.',
    author: 'Verified Buyer',
    location: 'Belo Horizonte, MG',
    stars: 5,
    tag: 'Top Recommendation Match',
    product: 'Home Decor & Living',
  },
];

function renderReviewsSpotlight() {
  const host = $('#home-reviews');
  if (!host) return;
  host.innerHTML = REVIEWS_SPOTLIGHT.map(
    (r) => `
    <article class="testimonial-card">
      <div class="row row--between">
        ${stars(r.stars)}
        <span class="pill pill--positive" style="font-size:0.7rem">${icon('check', 12)} ${escapeHtml(r.tag)}</span>
      </div>
      <p class="testimonial-card__quote">"${escapeHtml(r.quote)}"</p>
      <div class="testimonial-card__author">
        <div>
          <strong class="text-sm">${escapeHtml(r.author)}</strong>
          <p class="text-xs text-dim">${escapeHtml(r.location)}</p>
        </div>
        <span class="text-xs text-soft mono">${escapeHtml(r.product)}</span>
      </div>
    </article>`,
  ).join('');
}

/* ------------------------------------------------------------------ *
 * Hero Stats & Live Shelf
 * ------------------------------------------------------------------ */

async function renderHeroStats() {
  const host = $('#hero-stats');
  if (!host) return;
  try {
    const stats = await api.getStats();
    const values = [
      ['Products', num(stats.products || 100)],
      ['Categories', num(stats.categories || 30)],
      ['Reviews', num(stats.reviews || 1185)],
      ['Shopper Personas', num(stats.customers || PERSONAS.length)],
    ];
    host.innerHTML = values
      .map(([label, value]) => `<div><strong>${value}</strong><span>${escapeHtml(label)}</span></div>`)
      .join('');
  } catch {
    host.innerHTML = `
      <div><strong>100</strong><span>Products</span></div>
      <div><strong>30</strong><span>Categories</span></div>
      <div><strong>1,185+</strong><span>Reviews</span></div>
      <div><strong>5</strong><span>Personas</span></div>`;
  }
}

async function renderHeroShelf() {
  const persona = getPersona();
  const personaHost = $('#hero-persona');
  const recHost = $('#hero-recs');
  if (!personaHost || !recHost) return;

  personaHost.innerHTML = `
    <div class="row" style="gap:12px">
      <span class="persona__avatar" style="--accent:${persona.accent}">${escapeHtml(persona.name.split(' ').map((s) => s[0]).join(''))}</span>
      <div>
        <strong class="text-sm" style="display:block">${escapeHtml(persona.name)}</strong>
        <p class="text-xs text-dim">${escapeHtml(persona.persona)} · ${escapeHtml(persona.city)}</p>
      </div>
    </div>`;

  recHost.innerHTML = `<div class="skeleton skeleton--line w-90"></div><div class="skeleton skeleton--line w-60"></div>`;
  try {
    const data = await api.getRecommendations(persona.id, 4);
    if (!data.results.length) {
      recHost.innerHTML = `<p class="text-sm text-dim">No recommendations currently available for this shopper profile.</p>`;
      return;
    }
    recHost.innerHTML = data.results
      .map(
        (r) => `
        <a class="row" href="product.html?id=${encodeURIComponent(r.product.external_id)}" style="gap:12px;padding:10px 0;border-top:1px solid var(--border);text-decoration:none;color:inherit;transition:0.18s var(--ease)">
          <span class="rec-card__rank" style="width:28px;height:28px;font-size:0.82rem;flex-shrink:0">${r.rank}</span>
          <span style="flex:1;min-width:0">
            <span class="text-sm truncate" style="display:block;font-weight:600">${escapeHtml(r.product.name_translated)}</span>
            <span class="text-xs text-dim">${escapeHtml(money(r.product.price))} · ${r.product.category_name}</span>
          </span>
          ${icon('arrowRight', 15)}
        </a>`,
      )
      .join('');
  } catch (err) {
    recHost.innerHTML = `<p class="text-xs" style="color:var(--bad)">${escapeHtml(describeError(err))}</p>`;
  }
}

async function renderTopProducts() {
  const host = $('#home-products');
  if (!host) return;
  host.innerHTML = skeletonCards(8);
  try {
    const data = await api.getProducts({ ordering: '-total_purchases,id', page: 1 });
    host.innerHTML = data.results.slice(0, 8).map((p) => productCard(p, { badge: 'Best Seller' })).join('');
  } catch (err) {
    host.innerHTML = `<div class="state state--error"><h3 class="state__title">Could not load products</h3><p class="state__body">${escapeHtml(describeError(err))}</p></div>`;
  }
}

async function main() {
  renderTrustProps();
  renderFeaturedCategories();
  renderPersonaCards();
  renderReviewsSpotlight();

  await bootstrapData();

  await Promise.all([renderHeroStats(), renderHeroShelf(), renderTopProducts()]);

  onCustomerChange(() => {
    renderHeroShelf();
    renderPersonaCards();
  });
}

main();
