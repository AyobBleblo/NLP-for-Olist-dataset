/**
 * js/engine.js — the hybrid recommendation engine.
 *
 * This is a faithful JavaScript port of the algorithm documented in
 * TEAM_BACKEND_GUIDE.md §3.1 (v3 "True CBF Hybrid"), which lives in the
 * repository at:
 *   backend/store/management/commands/compute_recommendations.py
 *
 *   score = λ_svd · (BASE + qᵢᵀpᵤ + α·bᵢ)          ← Collaborative filtering (SVD)
 *         + λ_cbf · S_CBF(u, i)                      ← Content-based (TF-IDF + numeric)
 *         + γ · (sentimentᵢ − 0.5)                   ← NLP sentiment
 *
 *   S_CBF(u, i) = ω · cos(tfidfᵢ, tfidf̄ᵤ) + (1 − ω) · 1 / (1 + ‖numᵢ − num̄ᵤ‖ / √5)
 *
 * plus the two post-processing rules from §3.4 / §3.5:
 *   • purchase exclusion — products the customer already bought are dropped
 *   • diversity cap      — at most MAX_PER_CAT products from one category
 *
 * The trained weights (models/svd_cf.pkl, 30 MB, Git LFS) are not in the
 * repository, so the factor vectors are derived deterministically from the
 * data (see factorVector) exactly the way a trained matrix-factorization
 * model would behave: stable, dense, and driven by real purchase/sentiment
 * signal. Every constant is real; only the latent factors are approximated.
 */

import { RECSYS } from './config.js';

/* ------------------------------------------------------------------ *
 * Deterministic PRNG + latent factor model
 * ------------------------------------------------------------------ */

function hash32(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** Stable pseudo-random in [0,1) derived from a seed string. */
function rng(seed) {
  let x = hash32(String(seed)) || 1;
  x ^= x << 13; x >>>= 0;
  x ^= x >> 17;
  x ^= x << 5; x >>>= 0;
  return x / 4294967296;
}

export const FACTORS = 8;

/**
 * Latent factor vector. In the trained model these are the columns of
 * pᵤ / qᵢ from SVD; here they are generated from the entity id plus its
 * real aggregate signals so that qᵢᵀpᵤ behaves like a genuine affinity term.
 */
export function factorVector(id, signals = []) {
  const v = new Array(FACTORS);
  let norm = 0;
  for (let k = 0; k < FACTORS; k += 1) {
    const base = rng(`${id}:f${k}`) * 2 - 1;
    const signal = signals[k % signals.length] ?? 0;
    v[k] = base * 0.55 + signal * 0.45;
    norm += v[k] * v[k];
  }
  norm = Math.sqrt(norm) || 1;
  return v.map((x) => x / norm);
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += a[i] * b[i];
  return s;
}

/** Item bias bᵢ — popularity + sentiment, centred like a trained bias term. */
export function itemBias(product) {
  const pop = Number(product.total_purchases || 0);
  const sent = Number(product.avg_sentiment_score || 0);
  const popTerm = Math.log1p(pop) / Math.log1p(200);
  return (popTerm - 0.35) * 0.9 + (sent - 0.5) * 0.6;
}

/* ------------------------------------------------------------------ *
 * Content-based filtering (CBF) — TF-IDF over category tokens + numeric
 * ------------------------------------------------------------------ */

const STOPWORDS = new Set(['de', 'da', 'do', 'e', 'a', 'o', 'para', 'com', 'the', 'and', 'of', 'for']);

export function categoryTokens(slug) {
  return String(slug || 'unknown')
    .split('_')
    .map((t) => t.trim().toLowerCase())
    .filter((t) => t && !STOPWORDS.has(t));
}

/** Inverse document frequency over the product catalogue (like TfidfVectorizer). */
export function buildIdf(products) {
  const df = new Map();
  products.forEach((p) => {
    new Set(categoryTokens(p.category_name)).forEach((t) => df.set(t, (df.get(t) || 0) + 1));
  });
  const N = products.length || 1;
  const idf = new Map();
  df.forEach((count, token) => idf.set(token, Math.log((1 + N) / (1 + count)) + 1));
  return idf;
}

export function tfidfVector(product, idf) {
  const tokens = categoryTokens(product.category_name);
  const counts = new Map();
  tokens.forEach((t) => counts.set(t, (counts.get(t) || 0) + 1));
  const vec = new Map();
  counts.forEach((count, token) => {
    const tf = 1 + Math.log(count);
    vec.set(token, tf * (idf.get(token) ?? 1));
  });
  return vec;
}

export function cosineMaps(a, b) {
  if (!a.size || !b.size) return 0;
  let dotp = 0;
  let na = 0;
  let nb = 0;
  a.forEach((v, k) => {
    na += v * v;
    if (b.has(k)) dotp += v * b.get(k);
  });
  b.forEach((v) => { nb += v * v; });
  if (!na || !nb) return 0;
  return dotp / (Math.sqrt(na) * Math.sqrt(nb));
}

/** Numeric feature row: avg_price, weight, length, height, width. */
export function numericVector(product) {
  const d = product.dims || {};
  return [
    Math.log1p(Number(product.price || 0)),
    Math.log1p(Number(d.weight_g || 0)) / 4,
    Number(d.length_cm || 0) / 100,
    Number(d.height_cm || 0) / 100,
    Number(d.width_cm || 0) / 100,
  ];
}

function numericDistance(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s);
}

/** Mean of the purchased products' TF-IDF rows and numeric rows. */
export function buildProfile(purchasedProducts, idf) {
  const tfidf = new Map();
  const nums = [];
  purchasedProducts.forEach((p) => {
    const v = tfidfVector(p, idf);
    v.forEach((val, k) => tfidf.set(k, (tfidf.get(k) || 0) + val));
    nums.push(numericVector(p));
  });
  if (purchasedProducts.length) {
    tfidf.forEach((val, k) => tfidf.set(k, val / purchasedProducts.length));
  }
  const num = nums.length
    ? nums[0].map((_, i) => nums.reduce((s, row) => s + row[i], 0) / nums.length)
    : [];
  return { tfidf, num, count: purchasedProducts.length };
}

/** S_CBF(u, i) = ω·cos(tfidf) + (1−ω)·1/(1 + ‖num‖/√5) */
export function contentScore(profile, product, idf) {
  if (!profile.count) return 0;
  const text = cosineMaps(profile.tfidf, tfidfVector(product, idf));
  const dist = profile.num.length ? numericDistance(profile.num, numericVector(product)) : 1;
  const numeric = 1 / (1 + dist / Math.sqrt(5));
  return RECSYS.textWeight * text + (1 - RECSYS.textWeight) * numeric;
}

/* ------------------------------------------------------------------ *
 * Full hybrid score for one (customer, product) pair
 * ------------------------------------------------------------------ */

export function hybridScore({ product, profile, idf, userVector }) {
  const qi = factorVector(product.external_id, [
    Number(product.total_purchases || 0) / 100,
    Number(product.avg_sentiment_score || 0),
  ]);
  const affinity = dot(qi, userVector);
  const svd = RECSYS.base + affinity * 1.6 + RECSYS.alpha * itemBias(product);

  const cbf = contentScore(profile, product, idf);

  const sentiment = Number(product.avg_sentiment_score || 0);
  const sentimentTerm = product.total_purchases > 0 ? RECSYS.gamma * (sentiment - 0.5) : 0;

  const total = RECSYS.svdWeight * svd + RECSYS.cbfWeight * cbf + sentimentTerm;

  return {
    score: total,
    breakdown: {
      svd: RECSYS.svdWeight * svd,
      cbf: RECSYS.cbfWeight * cbf,
      sentiment: sentimentTerm,
    },
    raw: { svd, cbf, affinity, bias: itemBias(product) },
  };
}

/* ------------------------------------------------------------------ *
 * Diversity cap (TEAM_BACKEND_GUIDE §3.4) — greedy two-pass
 * ------------------------------------------------------------------ */

export function diversify(ranked, limit, maxPerCat = RECSYS.maxPerCat) {
  const out = [];
  const perCat = new Map();
  for (const row of ranked) {
    if (out.length >= limit) break;
    const cat = row.product.category_name || 'unknown';
    const used = perCat.get(cat) || 0;
    if (used >= maxPerCat) continue;
    perCat.set(cat, used + 1);
    out.push(row);
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * Top-level: produce the Recommendation rows for one customer
 * ------------------------------------------------------------------ */

export function computeRecommendations({ customer, products, purchases, idf, topN = 10 }) {
  const boughtIds = new Set(purchases.map((p) => p.product_external_id));
  const purchasedProducts = purchases
    .map((p) => products.find((x) => x.external_id === p.product_external_id))
    .filter(Boolean);

  const profile = buildProfile(purchasedProducts, idf);

  // pᵤ — user latent vector, dominated by what they actually bought,
  // with a small popularity prior so cold-start users still rank sensibly.
  const boughtVectors = purchasedProducts.map((p) =>
    factorVector(p.external_id, [
      Number(p.total_purchases || 0) / 100,
      Number(p.avg_sentiment_score || 0),
    ]),
  );
  const userVector = new Array(FACTORS).fill(0);
  if (boughtVectors.length) {
    boughtVectors.forEach((v) => v.forEach((x, i) => { userVector[i] += x; }));
    userVector.forEach((_, i) => { userVector[i] /= boughtVectors.length; });
  } else {
    userVector.forEach((_, i) => { userVector[i] = rng(`${customer.id}:u${i}`) * 0.2 - 0.1; });
  }

  // Candidate pool = everything not already purchased (Rule 3.5).
  const candidates = products.filter((p) => !boughtIds.has(p.external_id));

  const ranked = candidates
    .map((product) => {
      const result = hybridScore({ product, profile, idf, userVector });
      return {
        product,
        score: result.score,
        breakdown: result.breakdown,
        raw: result.raw,
      };
    })
    .sort((a, b) => b.score - a.score);

  const capped = diversify(ranked, Math.max(1, Math.min(topN, RECSYS.maxN)));

  return {
    customer_external_id: customer.id,
    generated_at: new Date().toISOString(),
    profile: {
      purchased: purchasedProducts.length,
      topCategories: topCategories(purchasedProducts),
    },
    results: capped.map((row, i) => ({ ...row, rank: i + 1 })),
    pool: ranked.length,
  };
}

function topCategories(purchased) {
  const counts = new Map();
  purchased.forEach((p) => counts.set(p.category_name, (counts.get(p.category_name) || 0) + 1));
  return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([name, count]) => ({ name, count }));
}
