/**
 * js/seed.js — one-time bootstrap of the local store.
 *
 * IMPORTANT — where this data comes from:
 *   • Categories are the real Olist `product_category_name` slugs together with
 *     their official `product_category_name_english` translations — the exact
 *     vocabulary the repo's import_data command consumes.
 *   • Product names use the repo's own generator (import_data.py / serializers.py):
 *         PT: "{Categoria} {Adjetivo} #{HEX6}"
 *         EN: "{Adjective} {Category} #{HEX6}"
 *     with idx = int(external_id[:2], 16) % 10 over the same 10 adjectives.
 *   • image_url uses the repo's picsum scheme (import_data.py _make_image_url).
 *   • The three customers are the real seed_test_scenario personas, with their
 *     documented ids, cities and states.
 *   • Recommendations are produced by js/engine.js — the port of
 *     compute_recommendations.py — using the real RECSYS_* constants.
 *
 * The raw Olist CSVs (99,224 reviews / 3,295 products) are NOT in the repo
 * (data/ is git-ignored), so the catalogue is a representative subset built
 * from the same generators and the same category vocabulary.
 */

import { API } from './config.js';
import { PERSONAS } from './session.js';
import { buildIdf, computeRecommendations } from './engine.js';

const tablesBase = () => API.localBase;

const ADJ_EN = ['Classic', 'Premium', 'Essential', 'Deluxe', 'Pro', 'Smart', 'Modern', 'Original', 'Urban', 'Signature'];
const ADJ_PT = ['Clássico', 'Premium', 'Essencial', 'Luxo', 'Pro', 'Smart', 'Moderno', 'Original', 'Urbano', 'Assinatura'];

/* Real Olist category vocabulary (slug -> official English translation) */
const CATEGORIES = [
  ['informatica_acessorios', 'computers_accessories'],
  ['moveis_decoracao', 'furniture_decor'],
  ['esporte_lazer', 'sports_leisure'],
  ['beleza_saude', 'health_beauty'],
  ['utilidades_domesticas', 'housewares'],
  ['ferramentas_jardim', 'garden_tools'],
  ['automotivo', 'auto'],
  ['brinquedos', 'toys'],
  ['pet_shop', 'pet_shop'],
  ['telefonia', 'telephony'],
  ['relogios_presentes', 'watches_gifts'],
  ['cool_stuff', 'cool_stuff'],
  ['malas_acessorios', 'luggage_accessories'],
  ['instrumentos_musicais', 'musical_instruments'],
  ['papelaria', 'stationery'],
];

/** Per-category sentiment tone, so PSI differences are meaningful. */
const CATEGORY_TONE = {
  informatica_acessorios: 0.78,
  moveis_decoracao: 0.86,
  esporte_lazer: 0.82,
  beleza_saude: 0.88,
  utilidades_domesticas: 0.8,
  ferramentas_jardim: 0.76,
  automotivo: 0.62,
  brinquedos: 0.84,
  pet_shop: 0.9,
  telefonia: 0.58,
  relogios_presentes: 0.8,
  cool_stuff: 0.85,
  malas_acessorios: 0.7,
  instrumentos_musicais: 0.83,
  papelaria: 0.87,
};

/* ------------------------------------------------------------------ *
 * Deterministic helpers
 * ------------------------------------------------------------------ */

function hash32(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function lcg(seed) {
  let x = hash32(seed) || 1;
  return () => {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    return x / 4294967296;
  };
}

/** 32-char Olist-style hex id. */
function hex32(seed) {
  let out = '';
  let x = hash32(seed);
  while (out.length < 32) {
    x = (Math.imul(x, 1103515245) + 12345) >>> 0;
    out += x.toString(16).padStart(8, '0');
  }
  return out.slice(0, 32);
}

const round2 = (n) => Math.round(n * 100) / 100;

function picsum(seed) {
  // Same scheme as import_data.py::_make_image_url (8 hex chars of a hash).
  return `https://picsum.photos/seed/${hex32(seed).slice(0, 8)}/400/400`;
}

/* ------------------------------------------------------------------ *
 * Dataset generation
 * ------------------------------------------------------------------ */

export function buildDataset() {
  const products = [];
  let counter = 0;

  CATEGORIES.forEach(([slug, en], catIdx) => {
    const ptTitle = slug.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
    const enTitle = en.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
    const rnd = lcg(`cat:${slug}`);

    for (let i = 0; i < 6; i += 1) {
      counter += 1;
      const external = hex32(`${slug}#${i}#${counter}`);
      const idx = parseInt(external.slice(0, 2), 16) % ADJ_EN.length;

      const price = round2(9.9 + rnd() * 250);
      const purchases = Math.floor(rnd() * 34);
      const tone = CATEGORY_TONE[slug] ?? 0.8;
      // ~32% of products have no processed reviews yet (avg_sentiment_score = 0),
      // which matches the guide's "sentiment inactive for most products" note.
      const scored = rnd() > 0.32;
      const sentiment = scored ? round2(Math.max(0.12, Math.min(0.99, tone + (rnd() - 0.5) * 0.3))) : 0;

      products.push({
        id: `p_${external.slice(0, 12)}`,
        numeric_id: counter,
        external_id: external,
        category_name: slug,
        name: `${ptTitle} ${ADJ_PT[idx]} #${external.slice(0, 6).toUpperCase()}`,
        name_translated: `${ADJ_EN[idx]} ${enTitle} #${external.slice(0, 6).toUpperCase()}`,
        price,
        avg_sentiment_score: sentiment,
        total_purchases: purchases,
        image_url: picsum(external),
        description:
          `${ADJ_EN[idx]} ${enTitle} from the Olist ${en.replace(/_/g, ' ')} assortment. ` +
          `Listed at BRL ${price.toFixed(2)} with ${purchases} recorded purchases in the imported sample.`,
        product_weight_g: Math.round(180 + rnd() * 8600),
        product_length_cm: Math.round(14 + rnd() * 60),
        product_height_cm: Math.round(4 + rnd() * 40),
        product_width_cm: Math.round(10 + rnd() * 44),
        sort: counter,
      });
    }

    void catIdx;
  });

  const categories = CATEGORIES.map(([slug, en], i) => ({
    id: `c_${slug}`,
    numeric_id: i + 1,
    name: slug,
    name_translated: en,
    sort_order: i + 1,
  }));

  /* ---- Customers + purchase history (seed_test_scenario) ---- */
  const orders = [];
  const orderItems = [];

  PERSONAS.forEach((persona) => {
    const inCat = products.filter((p) => p.category_name === persona.category_slug).slice(0, 4);
    const orderExternal = `ord_${persona.id.slice(0, 8)}`;
    const total = inCat.reduce((s, p) => s + p.price, 0);

    orders.push({
      id: `o_${orderExternal}`,
      external_id: orderExternal,
      customer_external_id: persona.id,
      purchased_at: new Date(Date.now() - 1000 * 60 * 60 * 24 * 21).toISOString(),
      status: 'delivered',
      total: round2(total),
    });

    inCat.forEach((p, i) => {
      orderItems.push({
        id: `${orderExternal}_${i}`,
        order_external_id: orderExternal,
        product_external_id: p.external_id,
        price: p.price,
        quantity: 1,
      });
    });
  });

  const customers = PERSONAS.map((p) => ({
    id: `cust_${p.id.slice(0, 10)}`,
    external_id: p.id,
    name: p.name,
    city: p.city,
    state: p.state,
    persona: p.persona,
    persona_ar: p.persona_ar,
    avatar_seed: p.id.slice(0, 6),
  }));

  /* ---- Recommendations — computed with the ported engine ---- */
  const idf = buildIdf(products);
  const recommendations = [];
  PERSONAS.forEach((persona) => {
    const mine = orderItems.filter((oi) => orders.find((o) => o.external_id === oi.order_external_id)?.customer_external_id === persona.id);
    const result = computeRecommendations({
      customer: { id: persona.id },
      products,
      purchases: mine,
      idf,
      topN: 10,
    });
    result.results.forEach((row, i) => {
      recommendations.push({
        id: `rec_${persona.id.slice(0, 8)}_${i + 1}`,
        customer_external_id: persona.id,
        product_external_id: row.product.external_id,
        rank: row.rank,
        score: round2(row.score * 10000) / 10000,
        svd_component: round2(row.breakdown.svd * 10000) / 10000,
        cbf_component: round2(row.breakdown.cbf * 10000) / 10000,
        sentiment_component: round2(row.breakdown.sentiment * 10000) / 10000,
        generated_at: new Date().toISOString(),
      });
    });
  });

  /* ---- Reviews — bilingual, mapped to the 3-class NLP pipeline ---- */
  const TEMPLATES = [
    { pt: 'Produto excelente, chegou muito antes do prazo. Recomendo!', en: 'Excellent product, arrived well before the deadline. I recommend it!', rating: 5, sentiment: 'positive', score: 0.97 },
    { pt: 'Qualidade surpreendente pelo preço pago. Muito satisfeito.', en: 'Surprising quality for the price paid. Very satisfied.', rating: 5, sentiment: 'positive', score: 0.93 },
    { pt: 'Bom custo-benefício, funciona exatamente como anunciado.', en: 'Good value for money, works exactly as advertised.', rating: 4, sentiment: 'positive', score: 0.81 },
    { pt: 'O produto é bom, mas a entrega demorou duas semanas.', en: 'The product is good, but delivery took two weeks.', rating: 3, sentiment: 'neutral', score: 0.52 },
    { pt: 'Veio com um pequeno risco na embalagem, mas funciona bem.', en: 'It arrived with a small scratch on the packaging, but it works fine.', rating: 3, sentiment: 'neutral', score: 0.47 },
    { pt: 'Veio quebrado e o vendedor não respondeu minhas mensagens.', en: 'It arrived broken and the seller never answered my messages.', rating: 1, sentiment: 'negative', score: 0.06 },
    { pt: 'Não funcionou após três dias de uso. Péssima qualidade.', en: 'It stopped working after three days. Terrible quality.', rating: 1, sentiment: 'negative', score: 0.04 },
    { pt: 'O material é bem mais frágil do que parece nas fotos.', en: 'The material is far more fragile than it looks in the photos.', rating: 2, sentiment: 'negative', score: 0.14 },
  ];

  const reviews = [];
  let reviewNo = 0;
  products.forEach((p) => {
    const rnd = lcg(`rev:${p.external_id}`);
    if (rnd() > 0.45) return; // only ~55% of the sample has written reviews
    const count = 1 + Math.floor(rnd() * 3);
    for (let i = 0; i < count; i += 1) {
      reviewNo += 1;
      // Bias the review pool toward the product's own sentiment tone.
      const pool = p.avg_sentiment_score >= 0.7
        ? [0, 1, 2, 3, 4]
        : p.avg_sentiment_score > 0.45
          ? [2, 3, 4, 5]
          : p.avg_sentiment_score > 0
            ? [3, 5, 6, 7]
            : [0, 1, 3, 4];
      const t = TEMPLATES[pool[Math.floor(rnd() * pool.length)]];
      reviews.push({
        id: `r_${hex32(`rev${reviewNo}`).slice(0, 12)}`,
        external_id: hex32(`olist-review-${p.external_id}-${i}`),
        product_external_id: p.external_id,
        rating: t.rating,
        comment_text: t.pt,
        comment_text_translated: t.en,
        sentiment: t.sentiment,
        sentiment_score: t.score,
        created_at: new Date(Date.now() - Math.floor(rnd() * 200) * 86400000).toISOString(),
      });
    }
  });

  return { categories, products, customers, orders, orderItems, recommendations, reviews };
}

/* ------------------------------------------------------------------ *
 * Table API writer
 * ------------------------------------------------------------------ */

async function countRows(table) {
  try {
    const res = await fetch(`${tablesBase()}${table}?page=1&limit=1`, { headers: { Accept: 'application/json' } });
    if (!res.ok) return 0;
    const json = await res.json();
    return Number(json?.total ?? (json?.data || []).length);
  } catch {
    return 0;
  }
}

async function postRow(table, row) {
  const res = await fetch(`${tablesBase()}${table}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(row),
  });
  if (!res.ok && res.status !== 409) {
    const text = await res.text().catch(() => '');
    throw new Error(`${table}: ${res.status} ${text.slice(0, 140)}`);
  }
}

/** Strip UI-only helper fields before persisting. */
function clean(row) {
  const { sort, ...rest } = row;
  void sort;
  return rest;
}

let seedPromise = null;

/**
 * Idempotent: only writes when the store is still empty.
 * Returns { seeded: bool, counts: object, failures: number }
 */
export function ensureSeeded({ onProgress } = {}) {
  if (seedPromise) return seedPromise;
  seedPromise = (async () => {
    if (API.mode === 'remote') {
      return { seeded: false, remote: true, counts: {} };
    }

    const existing = await countRows('products');
    const existingRecs = await countRows('recommendations');
    if (existing > 0 && existingRecs > 0) {
      return { seeded: false, counts: { products: existing, recommendations: existingRecs }, failures: 0 };
    }

    const data = buildDataset();
    const plan = [
      ['categories', data.categories],
      ['products', data.products],
      ['customers', data.customers],
      ['orders', data.orders],
      ['order_items', data.orderItems],
      ['recommendations', data.recommendations],
      ['reviews', data.reviews],
    ];
    const total = plan.reduce((s, [, rows]) => s + rows.length, 0);
    let done = 0;
    let failures = 0;

    for (const [table, rows] of plan) {
      onProgress?.(done, total, `Seeding ${table}`);
      for (const row of rows) {
        try {
          await postRow(table, clean(row));
        } catch {
          failures += 1;
        }
        done += 1;
        if (done % 12 === 0) onProgress?.(done, total, `Seeding ${table}`);
      }
    }
    onProgress?.(total, total, 'Dataset ready');

    return {
      seeded: true,
      failures,
      counts: Object.fromEntries(plan.map(([t, rows]) => [t, rows.length])),
    };
  })();
  return seedPromise;
}
