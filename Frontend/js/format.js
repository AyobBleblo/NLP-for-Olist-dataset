/**
 * js/format.js — pure formatting helpers shared by every page.
 */

export function money(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  return '$' + n.toFixed(2);
}

export function num(value, digits = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  return n.toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function score(value, digits = 4) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  return n.toFixed(digits);
}

/** "Smart Sports Leisure #B9EE75" -> initials used by the image fallback */
export function initials(text, max = 2) {
  if (!text) return '?';
  return String(text)
    .replace(/#[0-9A-F]+/i, '')
    .trim()
    .split(/\s+/)
    .slice(0, max)
    .map((w) => w[0])
    .join('')
    .toUpperCase();
}

/** Deterministic hue from any string — keeps placeholder art stable. */
export function hueOf(text) {
  let h = 0;
  const s = String(text || '');
  for (let i = 0; i < s.length; i += 1) {
    h = (h * 31 + s.charCodeAt(i)) % 360;
  }
  return h;
}

export function dateShort(value) {
  if (!value) return '—';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
}

export function dateTime(value) {
  if (!value) return '—';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function relativeTime(value) {
  if (!value) return '—';
  const then = new Date(value).getTime();
  if (Number.isNaN(then)) return '—';
  const diff = Date.now() - then;
  const abs = Math.abs(diff);
  const min = 60_000;
  const hour = 60 * min;
  const day = 24 * hour;
  const rtf = new Intl.RelativeTimeFormat('en', { numeric: 'auto' });
  if (abs < min) return rtf.format(Math.round(-diff / 1000), 'second');
  if (abs < hour) return rtf.format(Math.round(-diff / min), 'minute');
  if (abs < day) return rtf.format(Math.round(-diff / hour), 'hour');
  return rtf.format(Math.round(-diff / day), 'day');
}

export function escapeHtml(text) {
  return String(text ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[c]));
}

/** Sentiment colour bucket used across gauges, badges and charts. */
export function sentimentBucket(avg) {
  const n = Number(avg);
  if (!Number.isFinite(n) || n <= 0) return 'none';
  if (n >= 0.7) return 'positive';
  if (n >= 0.45) return 'mixed';
  return 'negative';
}

export function sentimentLabel(avg) {
  switch (sentimentBucket(avg)) {
    case 'positive':
      return 'Positive';
    case 'mixed':
      return 'Mixed';
    case 'negative':
      return 'Negative';
    default:
      return 'Not scored';
  }
}

/** Pretty-print a category slug: furniture_decor -> Furniture Decor */
export function prettyCategory(slug) {
  if (!slug) return 'Uncategorised';
  return String(slug).replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}
