/**
 * js/session.js — customer context.
 *
 * IMPORTANT — why there is no login screen:
 *   backend/store/urls.py exposes no auth routes, and
 *   recsys_backend/settings.py defines no DEFAULT_AUTHENTICATION_CLASSES or
 *   DEFAULT_PERMISSION_CLASSES. Every endpoint is anonymous and the customer
 *   is identified solely by `customer_external_id` in the URL path.
 *
 * So the frontend "session" is simply: which seeded customer am I browsing as?
 * The three ids below are the real personas created by
 * `manage.py seed_test_scenario` (verified in PROJECT_CONTEXT.md).
 */

import { safeGet, safeSet } from './config.js';

export const PERSONAS = [
  {
    id: '1dfbdc636de09adbcdbc3d34e15084f3',
    name: 'Alice Silva',
    city: 'Brasília',
    state: 'DF',
    persona: 'Computers & Technology Geek',
    persona_ar: 'مهتمة بالحاسوب والتقنية',
    category_slug: 'computers_accessories',
    category_en: 'Computers & Accessories',
    accent: '#8b5cf6',
  },
  {
    id: '2e43e031f10de28e557c35ef668f9396',
    name: 'Bruno Santos',
    city: 'Canoas',
    state: 'RS',
    persona: 'Sports & Fitness Enthusiast',
    persona_ar: 'مهتم بالرياضة واللياقة',
    category_slug: 'sports_leisure',
    category_en: 'Sports & Outdoors',
    accent: '#06b6d4',
  },
  {
    id: '33de26d1fafbfd4945eb586f7136efe6',
    name: 'Camila Oliveira',
    city: 'Montes Claros',
    state: 'MG',
    persona: 'Home Decor & Design Enthusiast',
    persona_ar: 'مهتمة بالأثاث والديكور المنزلي',
    category_slug: 'furniture_decor',
    category_en: 'Furniture & Home Decor',
    accent: '#ec4899',
  },
  {
    id: '1b6c7548a2a1f9037c1fd3ddfed95f33',
    name: 'Elena Souza',
    city: 'Ituiutaba',
    state: 'MG',
    persona: 'Beauty, Style & Wellness',
    persona_ar: 'مهتمة بالجمال والعناية',
    category_slug: 'health_beauty',
    category_en: 'Beauty & Personal Care',
    accent: '#f59e0b',
  },
  {
    id: 'fe86d9409d83a3c561ce16e64d2d55e6',
    name: 'Diego Ferreira',
    city: 'Jaú',
    state: 'SP',
    persona: 'Automotive & Hardware Gearhead',
    persona_ar: 'مهتم بالسيارات والعدد',
    category_slug: 'auto',
    category_en: 'Automotive & Tools',
    accent: '#10b981',
  },
];

const KEY = 'olist.customer';

export function getCustomerId() {
  const stored = safeGet(KEY);
  if (stored && PERSONAS.some((p) => p.id === stored)) return stored;
  return PERSONAS[0].id;
}

export function setCustomerId(id) {
  if (!PERSONAS.some((p) => p.id === id)) return false;
  safeSet(KEY, id);
  window.dispatchEvent(new CustomEvent('olist:customer-changed', { detail: { id } }));
  return true;
}

export function getPersona(id = getCustomerId()) {
  return PERSONAS.find((p) => p.id === id) || PERSONAS[0];
}

export function onCustomerChange(handler) {
  window.addEventListener('olist:customer-changed', (e) => handler(e.detail.id));
}
