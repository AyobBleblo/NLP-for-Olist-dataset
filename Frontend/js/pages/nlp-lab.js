/**
 * js/pages/nlp-lab.js — interactive sentiment lab + the 30-case benchmark.
 */

import { initPage, observeReveals } from '../main.js';
import { api } from '../api.js';
import { BENCHMARK, runPipeline } from '../nlp.js';
import { $, icon, toast } from '../ui.js';
import { BACKEND_NOTES } from '../config.js';
import { escapeHtml } from '../format.js';

initPage({ active: 'nlp-lab.html', title: 'NLP Lab' });

const input = $('#review-input');
const resultHost = $('#nlp-result');

/* ---- sample reviews pulled from the API data ---- */

async function loadSamples() {
  const host = $('#sample-reviews');
  host.innerHTML = `<p class="text-xs text-dim">Loading sample reviews from the API…</p>`;
  try {
    const reviews = await api.getReviews();
    const pool = reviews.filter((r) => r.comment_text && r.comment_text.trim());
    const picked = [];
    ['positive', 'negative'].forEach((s) => {
      const found = pool.find((r) => r.sentiment === s && !picked.includes(r));
      if (found) picked.push(found);
    });
    pool.filter((r) => !picked.includes(r)).slice(0, 3).forEach((r) => picked.push(r));

    if (!picked.length) {
      host.innerHTML = `<p class="text-xs text-dim">No stored reviews available to sample.</p>`;
      return;
    }

    host.innerHTML = `
      <p class="text-xs text-dim">Try a real stored review from the database (PT text with its NLLB translation):</p>
      <div class="row row--wrap" style="gap:8px">
        ${picked
          .slice(0, 4)
          .map(
            (r, i) => `<button class="btn btn--ghost btn--sm" type="button" data-sample="${i}">
              ${escapeHtml((r.comment_text || '').slice(0, 34))}…</button>`,
          )
          .join('')}
      </div>`;
    host.querySelectorAll('[data-sample]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const r = picked[Number(btn.dataset.sample)];
        input.value = r.comment_text;
        updateCount();
        render();
        input.focus();
      });
    });
  } catch {
    host.innerHTML = `<p class="text-xs text-dim">Sample reviews unavailable.</p>`;
  }
}

/* ---- render ---- */

const LABEL_AR = { positive: 'إيجابي', negative: 'سلبي' };

function verdictBlock(prediction) {
  return `
    <div class="prediction__verdict prediction__verdict--${prediction.label}">
      <span class="stat__icon">${icon(prediction.label === 'positive' ? 'heart' : 'alert', 22)}</span>
      <div>
        <p class="prediction__label">${escapeHtml(prediction.label)}</p>
        <p class="text-sm text-soft">${escapeHtml(LABEL_AR[prediction.label] || '')} · class id ${prediction.labelId} · confidence ${(prediction.confidence * 100).toFixed(1)}%</p>
      </div>
    </div>`;
}

function probRows(prediction) {
  const rows = [
    ['Negative (Class 0)', prediction.probabilities.negative, 'bad'],
    ['Positive (Class 1)', prediction.probabilities.positive, 'ok'],
  ];
  return rows
    .map(
      ([label, value, tone]) => `
      <div class="prob-row">
        <span>${label}</span>
        <span class="bar bar--${tone}"><span class="bar__fill" style="width:${(value * 100).toFixed(1)}%"></span></span>
        <span>${(value * 100).toFixed(1)}%</span>
      </div>`,
    )
    .join('');
}

function render() {
  const text = input.value.trim();
  if (!text) {
    resultHost.innerHTML = `
      <div class="state state--empty">
        <span class="state__icon">${icon('brain', 26)}</span>
        <h3 class="state__title">Awaiting input</h3>
        <p class="state__body">Run the pipeline to see the translation and the binary probability distribution.</p>
      </div>`;
    return;
  }

  const out = runPipeline(text);
  const { prediction, language, translation } = out;
  const psi = Math.round(((prediction.probabilities.positive - prediction.probabilities.negative + 1) / 2) * 1000) / 1000;

  resultHost.innerHTML = `
    ${verdictBlock(prediction)}

    <div class="card card--pad">
      <h2 style="font-size:0.92rem;margin-bottom:12px">Binary class probabilities</h2>
      ${probRows(prediction)}
      <p class="text-xs text-dim" style="margin-top:12px">
        Product Sentiment Index equivalent: <code>PSI = P(pos) − P(neg) = ${(prediction.probabilities.positive - prediction.probabilities.negative).toFixed(3)}</code>
        → rescaled to <b>${psi.toFixed(3)}</b> for the hybrid ranker.
      </p>
    </div>

    <div class="card card--pad">
      <h2 style="font-size:0.92rem;margin-bottom:12px">Pipeline stages</h2>
      <div class="kv"><dt>1 · Encoding repair (ftfy)</dt><dd>${translation.changed || out.repaired !== text ? 'applied' : 'no mojibake found'}</dd></div>
      <div class="kv"><dt>2 · Language detection</dt><dd class="mono">${escapeHtml(language.code)} (${(language.confidence * 100).toFixed(0)}%)</dd></div>
      <div class="kv"><dt>3 · Translation (NLLB-200)</dt><dd>${escapeHtml(translation.method === 'passthrough' ? 'skipped — already English' : 'glossary mode')}</dd></div>
      <div class="kv"><dt>4 · Classifier (DistilBERT)</dt><dd>Binary (Class 0 / 1) · lexical approximation</dd></div>
      <div class="stack stack--sm" style="margin-top:14px">
        <div>
          <p class="text-xs text-dim">English text passed to the classifier</p>
          <p class="text-sm">${escapeHtml(translation.text)}</p>
        </div>
        <div>
          <p class="text-xs text-dim">Lexical evidence</p>
          <p class="text-sm">positive hits ${prediction.evidence.positive.toFixed(2)} · negative hits ${prediction.evidence.negative.toFixed(2)}
            · contrast ${prediction.evidence.contrast ? 'yes' : 'no'} · ${prediction.evidence.tokens} tokens</p>
        </div>
      </div>
    </div>`;
}

function updateCount() {
  $('#char-count').textContent = `${input.value.length} characters`;
  const lang = input.value.trim() ? runPipeline(input.value).language : null;
  $('#lang-hint').textContent = lang
    ? `Detected language: ${lang.code === 'por_Latn' ? 'Brazilian Portuguese (por_Latn)' : lang.code === 'eng_Latn' ? 'English (eng_Latn)' : 'unknown'} — NLLB translates por_Latn → eng_Latn.`
    : 'Portuguese input is translated automatically, exactly like translate_reviews.py.';
}

/* ---- benchmark ---- */

function caseCard(row) {
  const correct = row.actual === row.expected;
  return `
    <article class="case-card ${correct ? '' : 'case-card--miss'}">
      <p class="case-card__text">${escapeHtml(row.text)}</p>
      <div class="case-card__row">
        <span class="row" style="gap:6px">
          <span class="pill pill--ghost">${escapeHtml(row.expected)}</span>
          ${icon('arrowRight', 12)}
          <span class="pill pill--${row.actual}">${escapeHtml(row.actual)}</span>
        </span>
        <span class="row" style="gap:8px">
          <span class="text-xs text-dim">L${row.level} (${escapeHtml(row.difficulty || '')})</span>
          <span class="text-xs">${(row.confidence * 100).toFixed(0)}%</span>
          ${correct ? `<span style="color:var(--ok)">${icon('check', 14)}</span>` : `<span style="color:var(--warn)">${icon('alert', 14)}</span>`}
        </span>
      </div>
      ${row.notes ? `<p class="text-xs text-dim" style="margin-top:6px">${escapeHtml(row.notes)}</p>` : ''}
    </article>`;
}

function runBenchmark({ silent = false } = {}) {
  const rows = BENCHMARK.map((c) => {
    const out = runPipeline(c.text);
    return { ...c, actual: out.prediction.label, confidence: out.prediction.confidence };
  });
  const correct = rows.filter((r) => r.actual === r.expected).length;
  const perClass = ['negative', 'positive'].map((c) => {
    const set = rows.filter((r) => r.expected === c);
    return { c, hit: set.filter((r) => r.actual === c).length, total: set.length };
  });

  $('#benchmark-grid').innerHTML = `
    <div class="card card--pad" style="grid-column:1/-1">
      <div class="row row--wrap" style="gap:26px">
        <div><p class="stat__label">Overall</p><p class="stat__value">${correct}/20</p><p class="stat__hint">${((correct / 20) * 100).toFixed(0)}% agreement with the repo's expected labels</p></div>
        ${perClass
          .map(
            (p) => `<div><p class="stat__label">${p.c}</p><p class="stat__value">${p.hit}/${p.total}</p><p class="stat__hint">${((p.hit / p.total) * 100).toFixed(0)}%</p></div>`,
          )
          .join('')}
      </div>
    </div>
    ${rows.map(caseCard).join('')}`;
  observeReveals($('#benchmark-grid'));
  if (!silent) {
    toast(`Benchmark finished: ${correct}/20 matched the expected labels`, { type: 'info', title: '20-case binary suite' });
  }
}

/* ---- pipeline explainer ---- */

const STAGES = [
  { n: 1, title: 'Download', cmd: 'downloddataset.py', body: 'kagglehub pulls the Olist CSVs (99,224 review rows).' },
  { n: 2, title: 'Translate', cmd: 'translate_reviews.py', body: 'ftfy repairs mojibake, then NLLB-200 translates por_Latn → eng_Latn in FP16 batches.' },
  { n: 3, title: 'Split', cmd: 'prepare_data.py', body: 'Maps 1–2 stars to Negative, 4–5 to Positive (dropping 3-star neutral) with an 80/10/10 stratified split.' },
  { n: 4, title: 'Fine-tune', cmd: 'train_model.py', body: 'Binary DistilBERT with AMP, dynamic padding, and AdamW optimizer.' },
  { n: 5, title: 'Evaluate', cmd: 'evaluate_model.py', body: 'Held-out test report + confusion matrix (93.34% accuracy, 0.9205 macro F1).' },
  { n: 6, title: 'Serve', cmd: 'run_sentiment_batch.py', body: 'Offline batch fills Product.avg_sentiment_score for the hybrid ranker.' },
];

/* ---- wiring ---- */

input.addEventListener('input', () => {
  updateCount();
  clearTimeout(input._t);
  input._t = setTimeout(render, 260);
});

$('#run-btn').addEventListener('click', () => {
  render();
  if (!input.value.trim()) toast('Type a review first', { type: 'warn', title: 'Nothing to analyse' });
});

$('#clear-btn').addEventListener('click', () => {
  input.value = '';
  updateCount();
  render();
  input.focus();
});

$('#run-benchmark').addEventListener('click', () => runBenchmark());

$('#nlp-pipeline').innerHTML = STAGES.map(
  (s) => `<article class="pipeline__step reveal"><span class="pipeline__num">${s.n}</span><h3>${escapeHtml(s.title)}</h3><p>${escapeHtml(s.body)}</p><code>${escapeHtml(s.cmd)}</code></article>`,
).join('');

$('#nlp-limitation').innerHTML = `
  <div class="banner banner--warn">
    <span class="banner__icon">${icon('alert', 20)}</span>
    <div class="banner__text">
      <h2>${escapeHtml(BACKEND_NOTES.sentimentModel.title)}</h2>
      <p>${escapeHtml(BACKEND_NOTES.sentimentModel.body)} Stages 1–2 are faithful, stage 3 uses a bounded glossary and stage 4 uses a transparent lexical scorer with negation, contrast and intensity handling. The label mapping, class-weighting rationale and thresholds match the documented model; only the numeric weights are approximated.</p>
    </div>
  </div>`;

$('#nlp-mode').innerHTML = `${icon('brain', 13)} lexical approximation`;

render();
updateCount();
loadSamples();
runBenchmark({ silent: true });
