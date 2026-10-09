/**
 * js/config.js — Central configuration for the Olist Recommender frontend.
 *
 * Every constant here is derived from the actual backend repository
 * (see /backend). Nothing is invented:
 *
 *  - RECSYS_*  ->  backend/recsys_backend/settings.py  +  TEAM_BACKEND_GUIDE.md §3.1
 *  - ENDPOINTS ->  backend/store/urls.py  (the 11 real routes)
 *  - API modes ->  see README.md "API integration" section
 */

/* ------------------------------------------------------------------ *
 * Application metadata
 * ------------------------------------------------------------------ */

export const APP = {
  name: 'Olist Store',
  tagline: 'Curated Brazilian Marketplace with AI Recommendations',
  version: '2.0.0',
  repo: 'https://github.com/Aymanbawa/NLP-for-Olist-dataset',
};

/* ------------------------------------------------------------------ *
 * API access modes
 *
 * The backend repository ships a Django REST API. A purely static site
 * cannot execute Django, so the frontend ships a swappable adapter:
 *
 *   local  (default) — implements the exact same 11-endpoint contract on
 *                      top of the platform table API, so every page is
 *                      fully functional and persistent.
 *   remote           — talks to the real Django API at remoteBase.
 *                      Requires the backend to allow CORS (see README).
 *
 * Switch with ?api=remote  or  localStorage 'olist.api.mode'.
 * ------------------------------------------------------------------ */

const params = new URLSearchParams(location.search);
const storedMode = safeGet('olist.api.mode');
const paramMode = params.get('api');

export const API = {
  mode: (paramMode || (storedMode && storedMode !== 'local' ? storedMode : 'remote')).toLowerCase(),
  /** Django dev server base URL — from SETUP.md / PROJECT_CONTEXT.md */
  remoteBase: safeGet('olist.api.remoteBase') || 'http://127.0.0.1:8000/api/',
  /** Platform table API — relative to the site root */
  localBase: new URL('tables/', document.baseURI).href,
  /** Preview/page size used by the catalog (backend PAGE_SIZE = 20) */
  pageSize: 20,
};

/* ------------------------------------------------------------------ *
 * Recommendation engine constants — copied verbatim from
 * backend/recsys_backend/settings.py and TEAM_BACKEND_GUIDE.md §3.1 / §7
 * ------------------------------------------------------------------ */

export const RECSYS = {
  svdWeight: 0.6,      // RECSYS_SVD_WEIGHT  (lambda_svd)
  cbfWeight: 0.4,      // RECSYS_CBF_WEIGHT  (lambda_cbf)
  base: 3.0,           // RECSYS_BASE
  alpha: 0.5,          // RECSYS_ALPHA — product-bias damping
  beta: 1.0,           // RECSYS_BETA  — category-affinity fallback
  gamma: 0.3,          // RECSYS_GAMMA — sentiment weight
  textWeight: 0.7,     // CBF_TEXT_WEIGHT (omega) — TF-IDF vs numeric
  maxPerCat: 2,        // RECSYS_MAX_PER_CAT — diversity cap
  defaultN: 10,
  maxN: 100,
};

/* ------------------------------------------------------------------ *
 * The 11 real API routes (backend/store/urls.py), used by the API
 * reference page and by the remote adapter.
 * ------------------------------------------------------------------ */

export const ENDPOINTS = [
  {
    id: 'root',
    group: 'Root',
    method: 'GET',
    path: '/api/',
    title: 'API root',
    params: [],
    body: null,
    description: 'Interactive entrypoint listing every available resource.',
    response: {
      categories: 'http://127.0.0.1:8000/api/categories/',
      products: 'http://127.0.0.1:8000/api/products/',
      cart_template: 'http://127.0.0.1:8000/api/cart/<customer_id>/',
      recommendations_template: 'http://127.0.0.1:8000/api/recommendations/<customer_id>/',
    },
  },
  {
    id: 'categories',
    group: 'Catalog',
    method: 'GET',
    path: '/api/categories/',
    title: 'List categories',
    params: [
      { name: 'search', type: 'string', note: 'matches name / name_translated' },
      { name: 'ordering', type: 'string', note: 'name, name_translated, product_count' },
    ],
    body: null,
    description:
      'All product categories with the English translation and a product_count annotation. Not paginated.',
    response: [
      { id: 7, name: 'esporte_lazer', name_translated: 'sports_leisure', product_count: 14 },
    ],
  },
  {
    id: 'category-detail',
    group: 'Catalog',
    method: 'GET',
    path: '/api/categories/<id>/',
    title: 'Category detail',
    params: [{ name: 'id', type: 'int', note: 'primary key' }],
    body: null,
    description: 'Single category with its product_count.',
    response: { id: 7, name: 'esporte_lazer', name_translated: 'sports_leisure', product_count: 14 },
  },
  {
    id: 'products',
    group: 'Catalog',
    method: 'GET',
    path: '/api/products/',
    title: 'List products',
    params: [
      { name: 'category', type: 'int|slug', note: 'category id or category name' },
      { name: 'search', type: 'string', note: 'name, name_translated, category, external_id' },
      { name: 'ordering', type: 'string', note: 'price, -price, total_purchases, -total_purchases, avg_sentiment_score, id' },
      { name: 'page', type: 'int', note: 'page number (20 per page)' },
    ],
    body: null,
    description:
      'Paginated product catalogue (DRF PageNumberPagination, PAGE_SIZE = 20). Default ordering: -total_purchases, id.',
    response: {
      count: 1000,
      next: 'http://127.0.0.1:8000/api/products/?page=2',
      previous: null,
      results: [
        {
          id: 130,
          external_id: 'b9ee75a5ff19e2f2b04cc5f2491b9f8a',
          name: 'Esporte Lazer Smart #B9EE75',
          name_translated: 'Smart Sports Leisure #B9EE75',
          category_id: 7,
          category_name: 'sports_leisure',
          price: '69.90',
          avg_sentiment_score: 0.0,
          total_purchases: 18,
          image_url: 'https://picsum.photos/seed/b9ee75/400/400',
        },
      ],
    },
  },
  {
    id: 'product-detail',
    group: 'Catalog',
    method: 'GET',
    path: '/api/products/<lookup_value>/',
    title: 'Product detail',
    params: [{ name: 'lookup_value', type: 'int|hash', note: 'numeric PK or Olist external_id' }],
    body: null,
    description:
      'Full product card. Accepts either the numeric PK or the 32-char Olist external_id hash.',
    response: {
      id: 130,
      external_id: 'b9ee75a5ff19e2f2b04cc5f2491b9f8a',
      name: 'Esporte Lazer Smart #B9EE75',
      name_translated: 'Smart Sports Leisure #B9EE75',
      category_id: 7,
      category_name: 'sports_leisure',
      category_name_original: 'esporte_lazer',
      price: '69.90',
      avg_sentiment_score: 0.0,
      total_purchases: 18,
      image_url: 'https://picsum.photos/seed/b9ee75/400/400',
    },
  },
  {
    id: 'cart-get',
    group: 'Cart',
    method: 'GET',
    path: '/api/cart/<customer_external_id>/',
    title: 'View cart',
    params: [{ name: 'customer_external_id', type: 'hash', note: 'Olist customer_unique_id' }],
    body: null,
    description: 'Current cart for the customer, with embedded product summaries and a calculated total.',
    response: {
      id: 1,
      customer_id: 3,
      items: [
        {
          id: 12,
          product: { id: 130, name_translated: 'Smart Sports Leisure #B9EE75', price: '69.90' },
          product_id: 130,
          quantity: 2,
        },
      ],
      total: 139.8,
      created_at: '2026-09-30T14:26:27Z',
      updated_at: '2026-09-30T14:26:27Z',
    },
  },
  {
    id: 'cart-add',
    group: 'Cart',
    method: 'POST',
    path: '/api/cart/<customer_external_id>/add/',
    title: 'Add to cart',
    params: [],
    body: { product_id: 130, quantity: 1 },
    description:
      'Adds a product, or increments its quantity when it is already in the cart. Returns the full cart. Validation: product must exist, quantity ≥ 1.',
    response: { detail: 'returns the updated cart object (same shape as GET)' },
  },
  {
    id: 'cart-update',
    group: 'Cart',
    method: 'PATCH',
    path: '/api/cart/<customer_external_id>/update/<item_id>/',
    title: 'Update item quantity',
    params: [{ name: 'item_id', type: 'int', note: 'CartItem primary key' }],
    body: { quantity: 3 },
    description:
      'Sets an exact quantity. 400 when quantity is missing, non-integer, or < 1.',
    response: { detail: 'returns the updated cart object (same shape as GET)' },
  },
  {
    id: 'cart-remove',
    group: 'Cart',
    method: 'DELETE',
    path: '/api/cart/<customer_external_id>/remove/<item_id>/',
    title: 'Remove cart item',
    params: [{ name: 'item_id', type: 'int', note: 'CartItem primary key' }],
    body: null,
    description: 'Deletes the cart item entirely and returns the updated cart.',
    response: { detail: 'returns the updated cart object (same shape as GET)' },
  },
  {
    id: 'cart-checkout',
    group: 'Cart',
    method: 'POST',
    path: '/api/cart/<customer_external_id>/checkout/',
    title: 'Checkout',
    params: [],
    body: null,
    description:
      'Converts every cart item into an Order + OrderItems in a single transaction, increments each product total_purchases, then empties the cart. 400 when the cart is empty.',
    response: {
      detail: 'Checkout successful.',
      order_external_id: '3f2c9a...',
      order_id: 42,
      items_count: 2,
    },
  },
  {
    id: 'recommendations',
    group: 'Recommendations',
    method: 'GET',
    path: '/api/recommendations/<customer_external_id>/',
    title: 'Top-N recommendations',
    params: [
      { name: 'n', type: 'int', note: 'default 10, clamped to 1..100' },
    ],
    body: null,
    description:
      'Reads only the precomputed Recommendation table — zero ML inference in the request path. Ordered by score DESC.',
    response: {
      customer_external_id: '3e2157f91502458bc58455fd798ed58a',
      count: 2,
      results: [
        {
          id: 1,
          product: {
            id: 130,
            external_id: 'b9ee75a5ff19e2f2b04cc5f2491b9f8a',
            name: 'Esporte Lazer Smart #B9EE75',
            name_translated: 'Smart Sports Leisure #B9EE75',
            category_id: 7,
            category_name: 'sports_leisure',
            price: '69.90',
            avg_sentiment_score: 0.0,
            total_purchases: 18,
            image_url: 'https://picsum.photos/seed/b9ee75/400/400',
          },
          score: 4.1553,
          generated_at: '2026-09-30T14:26:27Z',
        },
      ],
    },
  },
  {
    id: 'orders',
    group: 'Orders',
    method: 'GET',
    path: '/api/orders/<customer_external_id>/',
    title: 'Customer orders',
    params: [{ name: 'customer_external_id', type: 'hash', note: 'Olist customer_unique_id' }],
    body: null,
    description: 'Retrieves all orders placed by the customer with embedded items and prices.',
    response: [
      {
        id: 1,
        external_id: 'ord_c37cc6c1',
        customer_external_id: 'c37cc6c1a59d81460a3059744f7ada1c',
        purchased_at: '2026-09-30T14:26:27Z',
        status: 'delivered',
        total: 299.87,
        items: [],
      },
    ],
  },
  {
    id: 'reviews',
    group: 'Catalog',
    method: 'GET',
    path: '/api/reviews/?product=<product_external_id>',
    title: 'Product reviews',
    params: [{ name: 'product', type: 'hash', note: 'filter by product external_id' }],
    body: null,
    description: 'Bilingual customer reviews with original Portuguese and NLLB-200 English translation.',
    response: [],
  },
  {
    id: 'stats',
    group: 'Root',
    method: 'GET',
    path: '/api/stats/',
    title: 'Platform stats',
    params: [],
    body: null,
    description: 'Platform aggregate counts for products, categories, orders, reviews, etc.',
    response: {
      products: 1000,
      categories: 58,
      customers: 3,
      recommendations: 15,
      orders: 3,
      reviews: 1333,
    },
  },
];

/* ------------------------------------------------------------------ *
 * Documented backend behaviour
 * ------------------------------------------------------------------ */

export const BACKEND_NOTES = {
  noAuth: {
    title: 'No authentication layer',
    body:
      'store/urls.py exposes no login, registration, token or permission classes, and ' +
      'REST_FRAMEWORK defines no DEFAULT_AUTHENTICATION_CLASSES or DEFAULT_PERMISSION_CLASSES. ' +
      'Every endpoint is anonymous. Customers are identified purely by their external_id in the URL, ' +
      'so this frontend uses a customer switcher instead of an invented login flow.',
  },
  ordersEndpoint: {
    title: 'Orders API implemented',
    body:
      'Orders are retrieved via /api/orders/<customer_id>/ directly from the Django backend ' +
      'with order items, totals and timestamps.',
  },
  offlineBatch: {
    title: 'Zero ML inference in the request path',
    body:
      'Translation, sentiment scoring and recommendation scoring are offline management commands ' +
      '(translate_products, run_sentiment_batch, compute_recommendations). The API only reads precomputed ' +
      'columns and rows. This frontend mirrors that split: the batch console computes, the API serves.',
  },
  cors: {
    title: 'CORS enabled for remote access',
    body:
      'settings.py installs django-cors-headers with CORS_ALLOW_ALL_ORIGINS = True, ' +
      'enabling full cross-origin API calls from any frontend port.',
  },
  sentimentModel: {
    title: 'Binary DistilBERT Classifier (93.34% accuracy)',
    body:
      'Fine-tuned binary classifier on translated reviews achieves 93.34% overall accuracy and 0.9205 macro F1. ' +
      'Product avg_sentiment_score denormalizes review sentiment for hybrid ranking.',
  },
};

/* ------------------------------------------------------------------ *
 * Small storage helpers (never throw in private mode)
 * ------------------------------------------------------------------ */

export function safeGet(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function safeSet(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* ignore */
  }
}

