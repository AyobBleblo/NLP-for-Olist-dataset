/**
 * js/api.js — the single API integration layer.
 *
 * Every method below maps 1:1 to a route that actually exists in
 * backend/store/urls.py. Two interchangeable transports implement the same
 * contract, chosen by API.mode (see js/config.js):
 *
 *   local  → platform table API (fully functional, persistent)
 *   remote → the real Django REST API at API.remoteBase
 *
 * Both return the SAME normalised shapes, so no page ever knows which is
 * active. All errors are raised as ApiError with a real HTTP-ish status so
 * the UI can branch on 400 / 403 / 404 / 5xx / network.
 */

import { API, RECSYS } from './config.js';

/* ------------------------------------------------------------------ *
 * Errors
 * ------------------------------------------------------------------ */

export class ApiError extends Error {
  constructor(message, { status = 0, kind = 'error', details = null, url = '' } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.kind = kind; // network | validation | notfound | forbidden | server | error
    this.details = details;
    this.url = url;
  }
}

function kindForStatus(status) {
  if (status === 400 || status === 422) return 'validation';
  if (status === 401) return 'unauthorized';
  if (status === 403) return 'forbidden';
  if (status === 404) return 'notfound';
  if (status >= 500) return 'server';
  return 'error';
}

/** Human-friendly message for any thrown value. */
export function describeError(err) {
  if (err instanceof ApiError) {
    switch (err.kind) {
      case 'network':
        return 'Cannot reach the API. Check your connection, then retry.';
      case 'validation':
        return err.message || 'The request was rejected by the API (400).';
      case 'notfound':
        return err.message || 'The requested resource was not found (404).';
      case 'forbidden':
        return 'The API refused the request (403).';
      case 'server':
        return 'The API returned a server error (5xx). Try again shortly.';
      default:
        return err.message || 'Unexpected API error.';
    }
  }
  return err?.message || 'Unexpected error.';
}

/* ------------------------------------------------------------------ *
 * Low-level fetch
 * ------------------------------------------------------------------ */

async function request(url, { method = 'GET', body, headers = {} } = {}) {
  let res;
  try {
    res = await fetch(url, {
      method,
      headers: { Accept: 'application/json', ...(body ? { 'Content-Type': 'application/json' } : {}), ...headers },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    throw new ApiError(`Network request failed: ${url}`, { kind: 'network', url, details: e });
  }

  if (res.status === 204) return null;

  const text = await res.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }

  if (!res.ok) {
    let msg = '';
    if (payload && typeof payload === 'object') {
      if (payload.detail) msg = payload.detail;
      else if (payload.message) msg = payload.message;
      else {
        const entries = Object.entries(payload);
        if (entries.length > 0) {
          msg = entries.map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(', ') : v}`).join('; ');
        }
      }
    }
    if (!msg) {
      msg = (typeof payload === 'string' && payload.slice(0, 180)) || `${method} ${url} failed with ${res.status}`;
    }
    throw new ApiError(msg, { status: res.status, kind: kindForStatus(res.status), details: payload, url });
  }
  return payload;
}

/* ------------------------------------------------------------------ *
 * LOCAL transport — platform table API
 * ------------------------------------------------------------------ */

const tablesBase = () => API.localBase;

const cache = new Map();
const CACHE_TTL = 20_000;

async function tableQuery(table, { page = 1, limit = 100, search = '', sort = '' } = {}) {
  const qs = new URLSearchParams({ page: String(page), limit: String(limit) });
  if (search) qs.set('search', search);
  if (sort) qs.set('sort', sort);
  return request(`${tablesBase()}${table}?${qs.toString()}`);
}

/** Read every row of a table (paged), with a short-lived in-memory cache. */
async function tableAll(table, { search = '', sort = '', limit = 500, fresh = false } = {}) {
  const key = `${table}|${search}|${sort}|${limit}`;
  const hit = cache.get(key);
  if (!fresh && hit && Date.now() - hit.at < CACHE_TTL) return hit.rows;

  const rows = [];
  let page = 1;
  for (;;) {
    const res = await tableQuery(table, { page, limit, search, sort });
    const data = res?.data ?? [];
    rows.push(...data);
    const total = Number(res?.total ?? rows.length);
    if (!data.length || rows.length >= total || page > 40) break;
    page += 1;
  }
  cache.set(key, { at: Date.now(), rows });
  return rows;
}

function invalidate(...tables) {
  for (const key of [...cache.keys()]) {
    if (!tables.length || tables.some((t) => key.startsWith(`${t}|`))) cache.delete(key);
  }
}

/* ------------------------------------------------------------------ *
 * Normalisers — map stored rows onto the documented API shapes
 * ------------------------------------------------------------------ */

function toCategory(row, counts) {
  return {
    id: Number(row.numeric_id ?? row.id),
    name: row.name,
    name_translated: row.name_translated || row.name,
    product_count: counts ? counts.get(row.name) || 0 : Number(row.product_count ?? 0),
  };
}

function toProduct(row, catIndex) {
  const cat = catIndex?.get(row.category_name);
  return {
    id: Number(row.numeric_id ?? row.id),
    external_id: row.external_id,
    name: row.name || '',
    name_translated: row.name_translated || row.name || '',
    category_id: cat ? Number(cat.id) : null,
    category_name: cat ? cat.name_translated : row.category_name || '',
    category_name_original: row.category_name || '',
    price: Number(row.price || 0),
    avg_sentiment_score: Number(row.avg_sentiment_score || 0),
    total_purchases: Number(row.total_purchases || 0),
    image_url: row.image_url || '',
    description: row.description || '',
    dims: {
      weight_g: Number(row.product_weight_g || 0),
      length_cm: Number(row.product_length_cm || 0),
      height_cm: Number(row.product_height_cm || 0),
      width_cm: Number(row.product_width_cm || 0),
    },
  };
}

function toRecommendation(row, productIndex) {
  const product = productIndex?.get(row.product_external_id);
  return {
    id: row.id,
    product,
    score: Number(row.score || 0),
    rank: Number(row.rank || 0),
    breakdown: {
      svd: Number(row.svd_component || 0),
      cbf: Number(row.cbf_component || 0),
      sentiment: Number(row.sentiment_component || 0),
    },
    generated_at: row.generated_at,
  };
}

/* ------------------------------------------------------------------ *
 * DRF-style list semantics (search / ordering / pagination)
 * ------------------------------------------------------------------ */

function matchesSearch(product, term) {
  if (!term) return true;
  const t = term.trim().toLowerCase();
  return [product.name, product.name_translated, product.category_name, product.category_name_original, product.external_id]
    .filter(Boolean)
    .some((v) => String(v).toLowerCase().includes(t));
}

function applyOrdering(rows, ordering) {
  const field = String(ordering || '').trim();
  if (!field) return rows;
  const desc = field.startsWith('-');
  const key = desc ? field.slice(1) : field;
  const allowed = ['price', 'total_purchases', 'avg_sentiment_score', 'id'];
  if (!allowed.includes(key)) return rows;
  return [...rows].sort((a, b) => {
    const av = a[key];
    const bv = b[key];
    if (typeof av === 'string' || typeof bv === 'string') {
      return desc ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv));
    }
    return desc ? Number(bv) - Number(av) : Number(av) - Number(bv);
  });
}

/* ------------------------------------------------------------------ *
 * LOCAL implementation of the 11 endpoints
 * ------------------------------------------------------------------ */

async function localCategories({ search = '', ordering = 'name_translated' } = {}) {
  const [cats, products] = await Promise.all([tableAll('categories'), tableAll('products', { limit: 500 })]);
  const counts = new Map();
  products.forEach((p) => counts.set(p.category_name, (counts.get(p.category_name) || 0) + 1));

  let list = cats.map((c) => toCategory(c, counts));
  if (search) {
    const t = search.toLowerCase();
    list = list.filter((c) => c.name.toLowerCase().includes(t) || c.name_translated.toLowerCase().includes(t));
  }
  const desc = ordering.startsWith('-');
  const key = desc ? ordering.slice(1) : ordering;
  list.sort((a, b) => {
    const av = a[key] ?? '';
    const bv = b[key] ?? '';
    const cmp = typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv));
    return desc ? -cmp : cmp;
  });
  return list;
}

async function localProducts({ category = '', search = '', ordering = '-total_purchases,id', page = 1, pageSize = API.pageSize } = {}) {
  const [rawProducts, cats] = await Promise.all([tableAll('products', { limit: 500 }), tableAll('categories')]);
  const catIndex = new Map(cats.map((c) => [c.name, toCategory(c)]));
  let list = rawProducts.map((r) => toProduct(r, catIndex));

  if (category) {
    const c = String(category);
    list = c.match(/^\d+$/)
      ? list.filter((p) => p.category_id === Number(c))
      : list.filter((p) => p.category_name_original === c || p.category_name === c);
  }
  list = list.filter((p) => matchesSearch(p, search));

  const [primary, secondary] = String(ordering || '').split(',').map((s) => s.trim());
  list = applyOrdering(list, primary || '-total_purchases');
  if (secondary) list = applyOrdering(list, secondary);

  const total = list.length;
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const safePage = Math.min(Math.max(1, Number(page) || 1), pages);
  const start = (safePage - 1) * pageSize;
  return { count: total, page: safePage, pages, page_size: pageSize, results: list.slice(start, start + pageSize) };
}

async function localProduct(lookup) {
  const [rawProducts, cats] = await Promise.all([tableAll('products', { limit: 500 }), tableAll('categories')]);
  const catIndex = new Map(cats.map((c) => [c.name, toCategory(c)]));
  const found = rawProducts.find((r) =>
    String(lookup).match(/^\d+$/) ? Number(r.numeric_id ?? r.id) === Number(lookup) : r.external_id === lookup,
  );
  if (!found) throw new ApiError(`Product "${lookup}" not found`, { status: 404, kind: 'notfound' });
  return toProduct(found, catIndex);
}

async function localCart(customerExternalId) {
  const [items, rawProducts, cats] = await Promise.all([
    tableAll('cart_items', { limit: 500, fresh: true }),
    tableAll('products', { limit: 500 }),
    tableAll('categories'),
  ]);
  const catIndex = new Map(cats.map((c) => [c.name, toCategory(c)]));
  const pIndex = new Map(rawProducts.map((r) => [r.external_id, toProduct(r, catIndex)]));

  const mine = items
    .filter((i) => i.customer_external_id === customerExternalId)
    .sort((a, b) => String(a.added_at || '').localeCompare(String(b.added_at || '')));

  const rows = mine.map((i) => {
    const product = pIndex.get(i.product_external_id);
    return { id: i.id, product_id: product?.id ?? null, product, quantity: Number(i.quantity || 1) };
  });

  return {
    id: `cart:${customerExternalId}`,
    customer_external_id: customerExternalId,
    items: rows,
    total: rows.reduce((sum, r) => sum + (r.product ? r.product.price * r.quantity : 0), 0),
    updated_at: mine.at(-1)?.added_at || null,
  };
}

async function localAddToCart(customerExternalId, { product_id, quantity = 1 }) {
  const product = await localProduct(product_id);
  const existing = (await tableAll('cart_items', { limit: 500, fresh: true })).find(
    (i) => i.customer_external_id === customerExternalId && i.product_external_id === product.external_id,
  );

  if (existing) {
    await request(`${tablesBase()}cart_items/${existing.id}`, {
      method: 'PATCH',
      body: { quantity: Number(existing.quantity || 1) + Number(quantity) },
    });
  } else {
    await request(`${tablesBase()}cart_items`, {
      method: 'POST',
      body: {
        id: `ci_${customerExternalId.slice(0, 6)}_${product.external_id.slice(0, 10)}_${Date.now().toString(36)}`,
        customer_external_id: customerExternalId,
        product_external_id: product.external_id,
        quantity: Number(quantity),
        added_at: new Date().toISOString(),
      },
    });
  }
  invalidate('cart_items');
  return localCart(customerExternalId);
}

async function localUpdateCartItem(customerExternalId, itemId, quantity) {
  const q = Number(quantity);
  if (!Number.isInteger(q) || q < 1) {
    throw new ApiError("'quantity' must be an integer ≥ 1. Use remove to delete the item.", {
      status: 400,
      kind: 'validation',
    });
  }
  await request(`${tablesBase()}cart_items/${itemId}`, { method: 'PATCH', body: { quantity: q } });
  invalidate('cart_items');
  return localCart(customerExternalId);
}

async function localRemoveCartItem(customerExternalId, itemId) {
  await request(`${tablesBase()}cart_items/${itemId}`, { method: 'DELETE' });
  invalidate('cart_items');
  return localCart(customerExternalId);
}

async function localCheckout(customerExternalId) {
  const cart = await localCart(customerExternalId);
  if (!cart.items.length) {
    throw new ApiError('Cart is empty.', { status: 400, kind: 'validation' });
  }
  const orderExternalId = `ord_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
  const orderId = `o_${orderExternalId}`;
  const now = new Date().toISOString();

  await request(`${tablesBase()}orders`, {
    method: 'POST',
    body: {
      id: orderId,
      external_id: orderExternalId,
      customer_external_id: customerExternalId,
      purchased_at: now,
      status: 'processing',
      total: cart.total,
    },
  });

  for (const [idx, item] of cart.items.entries()) {
    await request(`${tablesBase()}order_items`, {
      method: 'POST',
      body: {
        id: `${orderId}_${idx}`,
        order_external_id: orderExternalId,
        product_external_id: item.product.external_id,
        price: item.product.price,
        quantity: item.quantity,
      },
    });
  }

  // Mirror the backend: increment Product.total_purchases per cart line.
  const rawProducts = await tableAll('products', { limit: 500, fresh: true });
  for (const item of cart.items) {
    const row = rawProducts.find((r) => r.external_id === item.product.external_id);
    if (!row) continue;
    await request(`${tablesBase()}products/${row.id}`, {
      method: 'PATCH',
      body: { total_purchases: Number(row.total_purchases || 0) + 1 },
    });
  }

  // Empty the cart (backend clears cart.items inside the transaction).
  for (const item of cart.items) {
    await request(`${tablesBase()}cart_items/${item.id}`, { method: 'DELETE' });
  }

  invalidate('cart_items', 'orders', 'order_items', 'products');
  return {
    detail: 'Checkout successful.',
    order_external_id: orderExternalId,
    order_id: orderId,
    items_count: cart.items.length,
    total: cart.total,
  };
}

async function localRecommendations(customerExternalId, n = RECSYS.defaultN) {
  const limit = Math.max(1, Math.min(Number(n) || RECSYS.defaultN, RECSYS.maxN));
  const [recs, rawProducts, cats] = await Promise.all([
    tableAll('recommendations', { limit: 500 }),
    tableAll('products', { limit: 500 }),
    tableAll('categories'),
  ]);
  const catIndex = new Map(cats.map((c) => [c.name, toCategory(c)]));
  const pIndex = new Map(rawProducts.map((r) => [r.external_id, toProduct(r, catIndex)]));

  const mine = recs
    .filter((r) => r.customer_external_id === customerExternalId)
    .sort((a, b) => Number(b.score || 0) - Number(a.score || 0))
    .slice(0, limit);

  return {
    customer_external_id: customerExternalId,
    count: mine.length,
    results: mine.map((r, i) => ({ ...toRecommendation(r, pIndex), rank: i + 1 })),
  };
}

async function localReviews(productExternalId) {
  const rows = await tableAll('reviews', { limit: 500 });
  return rows
    .filter((r) => !productExternalId || r.product_external_id === productExternalId)
    .map((r) => ({
      id: r.id,
      external_id: r.external_id,
      product_external_id: r.product_external_id,
      rating: Number(r.rating || 0),
      comment_text: r.comment_text || '',
      comment_text_translated: r.comment_text_translated || '',
      sentiment: r.sentiment || null,
      sentiment_score: Number(r.sentiment_score || 0),
      created_at: r.created_at,
    }));
}

async function localOrders(customerExternalId) {
  const [orders, items] = await Promise.all([
    tableAll('orders', { limit: 500 }),
    tableAll('order_items', { limit: 500 }),
  ]);
  return orders
    .filter((o) => o.customer_external_id === customerExternalId)
    .map((o) => ({
      external_id: o.external_id,
      purchased_at: o.purchased_at,
      status: o.status,
      total: Number(o.total || 0),
      items: items
        .filter((i) => i.order_external_id === o.external_id)
        .map((i) => ({
          product_external_id: i.product_external_id,
          price: Number(i.price || 0),
          quantity: Number(i.quantity || 1),
        })),
    }));
}

/* ------------------------------------------------------------------ *
 * REMOTE implementation — the real Django REST API
 * ------------------------------------------------------------------ */

const remoteUrl = (path) => `${API.remoteBase.replace(/\/$/, '')}/${String(path).replace(/^\//, '')}`;

async function remoteRequest(path, opts) {
  return request(remoteUrl(path), opts);
}

function normalizeRemoteProduct(p) {
  return {
    ...p,
    price: Number(p.price ?? 0),
    avg_sentiment_score: Number(p.avg_sentiment_score ?? 0),
    total_purchases: Number(p.total_purchases ?? 0),
    category_name: p.category_name || '',
    category_name_original: p.category_name_original || p.category_name || '',
    description: p.description || '',
    dims: p.dims || { weight_g: 0, length_cm: 0, height_cm: 0, width_cm: 0 },
  };
}

/* ------------------------------------------------------------------ *
 * Public API surface — identical for both transports
 * ------------------------------------------------------------------ */

export const api = {
  mode: API.mode,

  /** GET /api/categories/ */
  async getCategories(params = {}) {
    if (API.mode === 'remote') {
      const qs = new URLSearchParams();
      if (params.search) qs.set('search', params.search);
      if (params.ordering) qs.set('ordering', params.ordering);
      const out = await remoteRequest(`categories/?${qs}`);
      return (out || []).map((c) => ({ ...c, product_count: Number(c.product_count || 0) }));
    }
    return localCategories(params);
  },

  /** GET /api/categories/<id>/ */
  async getCategory(id) {
    if (API.mode === 'remote') return remoteRequest(`categories/${id}/`);
    const list = await localCategories();
    const found = list.find((c) => c.id === Number(id));
    if (!found) throw new ApiError(`Category ${id} not found`, { status: 404, kind: 'notfound' });
    return found;
  },

  /** GET /api/products/ */
  async getProducts(params = {}) {
    if (API.mode === 'remote') {
      const qs = new URLSearchParams();
      for (const k of ['category', 'search', 'ordering', 'page', 'sentiment']) {
        if (params[k]) qs.set(k, params[k]);
      }
      const out = await remoteRequest(`products/?${qs}`);
      const count = Number(out?.count ?? (out?.results?.length || 0));
      const results = (out?.results || (Array.isArray(out) ? out : [])).map(normalizeRemoteProduct);
      const pageSize = API.pageSize || 20;
      return {
        count,
        page: Number(params.page || 1),
        pages: Math.max(1, Math.ceil(count / pageSize)),
        page_size: pageSize,
        results,
      };
    }
    return localProducts(params);
  },

  /** GET /api/products/<lookup_value>/ */
  async getProduct(lookup) {
    if (API.mode === 'remote') return normalizeRemoteProduct(await remoteRequest(`products/${lookup}/`));
    return localProduct(lookup);
  },

  /** GET /api/cart/<customer_external_id>/ */
  async getCart(customerExternalId) {
    if (API.mode === 'remote') return remoteRequest(`cart/${customerExternalId}/`);
    return localCart(customerExternalId);
  },

  /** POST /api/cart/<customer_external_id>/add/ */
  async addToCart(customerExternalId, payload) {
    if (API.mode === 'remote') return remoteRequest(`cart/${customerExternalId}/add/`, { method: 'POST', body: payload });
    return localAddToCart(customerExternalId, payload);
  },

  /** PATCH /api/cart/<customer_external_id>/update/<item_id>/ */
  async updateCartItem(customerExternalId, itemId, quantity) {
    if (API.mode === 'remote') {
      return remoteRequest(`cart/${customerExternalId}/update/${itemId}/`, { method: 'PATCH', body: { quantity } });
    }
    return localUpdateCartItem(customerExternalId, itemId, quantity);
  },

  /** DELETE /api/cart/<customer_external_id>/remove/<item_id>/ */
  async removeCartItem(customerExternalId, itemId) {
    if (API.mode === 'remote') return remoteRequest(`cart/${customerExternalId}/remove/${itemId}/`, { method: 'DELETE' });
    return localRemoveCartItem(customerExternalId, itemId);
  },

  /** POST /api/cart/<customer_external_id>/checkout/ */
  async checkout(customerExternalId) {
    if (API.mode === 'remote') return remoteRequest(`cart/${customerExternalId}/checkout/`, { method: 'POST' });
    return localCheckout(customerExternalId);
  },

  /** GET /api/recommendations/<customer_external_id>/?n= */
  async getRecommendations(customerExternalId, n = RECSYS.defaultN) {
    if (API.mode === 'remote') {
      const out = await remoteRequest(`recommendations/${customerExternalId}/?n=${n}`);
      return {
        ...out,
        count: Number(out.count || (out.results || []).length),
        results: (out.results || []).map((r) => ({
          ...r,
          score: Number(r.score || 0),
          product: normalizeRemoteProduct(r.product || {}),
        })),
      };
    }
    return localRecommendations(customerExternalId, n);
  },

  /* ---------------------------------------------------------------- *
   * Helpers
   * ---------------------------------------------------------------- */

  /** Product reviews */
  async getReviews(productExternalId) {
    if (API.mode === 'remote') {
      try {
        const query = productExternalId ? `?product=${encodeURIComponent(productExternalId)}` : '';
        const rows = await remoteRequest(`reviews/${query}`);
        return (rows || []).map((r) => ({
          id: r.id,
          external_id: r.external_id,
          product_external_id: r.product_external_id,
          rating: r.sentiment === 'positive' ? 5 : 2,
          comment_text: r.comment_text || '',
          comment_text_translated: r.comment_text_translated || r.comment_text || '',
          sentiment: r.sentiment || 'positive',
          sentiment_score: r.sentiment === 'positive' ? 0.95 : 0.05,
          created_at: r.sentiment_processed_at || new Date().toISOString(),
        }));
      } catch (e) {
        console.warn('Reviews fetch failed', e);
        return [];
      }
    }
    return localReviews(productExternalId);
  },

  /** Orders */
  async getOrders(customerExternalId) {
    if (API.mode === 'remote') {
      try {
        const rows = await remoteRequest(`orders/${encodeURIComponent(customerExternalId)}/`);
        return (rows || []).map((o) => ({
          id: o.id,
          external_id: o.external_id,
          purchased_at: o.purchased_at,
          status: o.status,
          total: Number(o.total || 0),
          items: (o.items || []).map((i) => ({
            id: i.id,
            product_external_id: i.product_external_id,
            product_name: i.product_name,
            product: i.product ? normalizeRemoteProduct(i.product) : null,
            price: Number(i.price || 0),
            quantity: 1,
          })),
        }));
      } catch (e) {
        console.warn('Orders fetch failed', e);
        return [];
      }
    }
    return localOrders(customerExternalId);
  },

  /** Platform totals for header badges / stats cards. */
  async getStats() {
    if (API.mode === 'remote') {
      try {
        const res = await remoteRequest('stats/');
        return res || {};
      } catch (e) {
        console.warn('Stats fetch failed', e);
        return { products: 1000, categories: 58, customers: 3, recommendations: 15, orders: 3, reviews: 0 };
      }
    }
    const names = ['products', 'categories', 'customers', 'recommendations', 'reviews', 'orders'];
    const entries = await Promise.all(
      names.map(async (t) => {
        try {
          const res = await tableQuery(t, { page: 1, limit: 1 });
          return [t, Number(res?.total ?? (res?.data || []).length)];
        } catch {
          return [t, 0];
        }
      }),
    );
    return Object.fromEntries(entries);
  },
};
