/**
 * js/pages/analytics.js — evaluation dashboard.
 *
 * The model metrics are the repository's own published numbers (README §7–8
 * and models/evaluation_results.json). The catalogue charts are computed live
 * from the API responses.
 */

import { initPage, bootstrapData } from '../main.js';
import { api, describeError } from '../api.js';
import { $ } from '../ui.js';
import { escapeHtml, num } from '../format.js';

initPage({ active: 'analytics.html', title: 'Analytics' });

/* ---- Published metrics (verbatim from models/evaluation_results_binary.json) ---- */

const TRAINING_LOG = [
  { epoch: 1, train_loss: 0.2749, val_loss: 0.2323, val_acc: 93.31, val_f1: 0.9198, f1_neg: 0.8872, f1_pos: 0.9525 },
  { epoch: 2, train_loss: 0.1783, val_loss: 0.2562, val_acc: 93.66, val_f1: 0.9240, f1_neg: 0.8930, f1_pos: 0.9550 },
  { epoch: 3, train_loss: 0.1338, val_loss: 0.2560, val_acc: 93.18, val_f1: 0.9190, f1_neg: 0.8867, f1_pos: 0.9512 },
  { epoch: 4, train_loss: 0.1004, val_loss: 0.3225, val_acc: 93.34, val_f1: 0.9200, f1_neg: 0.8872, f1_pos: 0.9528 },
];

const CLASS_REPORT = [
  { label: 'Negative', precision: 0.8671, recall: 0.9109, f1: 0.8885, support: 1089 },
  { label: 'Positive', precision: 0.9626, recall: 0.9427, f1: 0.9525, support: 2651 },
];

const CONFUSION = [
  [992, 97],
  [152, 2499],
];

const CLASSIFICATION_REPORT = `              precision    recall  f1-score   support

    Negative     0.8671    0.9109    0.8885      1089
    Positive     0.9626    0.9427    0.9525      2651

    accuracy                         0.9334      3740
   macro avg     0.9149    0.9268    0.9205      3740
weighted avg     0.9348    0.9334    0.9339      3740`;

const KPIS = [
  { label: 'Test accuracy', value: '93.34%', delta: '3,740 held-out samples' },
  { label: 'Macro F1', value: '0.9205', delta: 'target ≥ 0.82 passed' },
  { label: 'Positive F1', value: '0.9525', delta: '2,651 positives · 70.9% of data' },
  { label: 'Negative F1', value: '0.8885', delta: '1,089 negatives · 29.1% of data' },
  { label: 'Weighted F1', value: '0.9339', delta: 'class-weighted performance' },
  { label: 'Training duration', value: '70.04 min', delta: '4 epochs · PyTorch FP16' },
  { label: 'Backbone', value: 'DistilBERT', delta: 'best_model_binary · 66M params' },
  { label: 'Best epoch', value: '2', delta: 'val macro F1: 0.9240 checkpoint' },
];

function renderKpis() {
  $('#eval-kpis').innerHTML = KPIS.map(
    (k) => `<div class="card kpi reveal"><p class="kpi__label">${escapeHtml(k.label)}</p><p class="kpi__value">${escapeHtml(k.value)}</p><p class="kpi__delta">${escapeHtml(k.delta)}</p></div>`,
  ).join('');
}

function chartDefaults() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#b6bdd4', font: { family: 'Inter', size: 11 }, boxWidth: 12, usePointStyle: true } },
      tooltip: { backgroundColor: 'rgba(10,14,28,.96)', borderColor: 'rgba(255,255,255,.14)', borderWidth: 1, padding: 10, titleColor: '#eef1fb', bodyColor: '#b6bdd4' },
    },
    scales: {
      x: { grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#7d86a3', font: { family: 'Inter', size: 11 } } },
      y: { grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#7d86a3', font: { family: 'Inter', size: 11 } } },
    },
  };
}

function drawModelCharts() {
  if (typeof Chart === 'undefined') return;
  Chart.defaults.font.family = 'Inter, system-ui, sans-serif';

  new Chart($('#chart-loss'), {
    type: 'line',
    data: {
      labels: TRAINING_LOG.map((e) => `Epoch ${e.epoch}`),
      datasets: [
        { label: 'Train loss', data: TRAINING_LOG.map((e) => e.train_loss), borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,.18)', fill: true, tension: 0.35, pointRadius: 4, pointBackgroundColor: '#6366f1' },
        { label: 'Val loss', data: TRAINING_LOG.map((e) => e.val_loss), borderColor: '#fb7185', backgroundColor: 'rgba(251,113,133,.14)', fill: true, tension: 0.35, pointRadius: 4, pointBackgroundColor: '#fb7185' },
      ],
    },
    options: { ...chartDefaults(), plugins: { ...chartDefaults().plugins, title: { display: true, text: 'Cross-entropy loss', color: '#eef1fb', font: { size: 13, weight: '600' } } } },
  });

  new Chart($('#chart-f1'), {
    type: 'line',
    data: {
      labels: TRAINING_LOG.map((e) => `Epoch ${e.epoch}`),
      datasets: [
        { label: 'Negative F1', data: TRAINING_LOG.map((e) => e.f1_neg), borderColor: '#fb7185', tension: 0.35, pointRadius: 4, pointBackgroundColor: '#fb7185' },
        { label: 'Positive F1', data: TRAINING_LOG.map((e) => e.f1_pos), borderColor: '#34d399', tension: 0.35, pointRadius: 4, pointBackgroundColor: '#34d399' },
        { label: 'Macro F1', data: TRAINING_LOG.map((e) => e.val_f1), borderColor: '#22d3ee', borderDash: [5, 4], tension: 0.35, pointRadius: 4, pointBackgroundColor: '#22d3ee' },
      ],
    },
    options: { ...chartDefaults(), scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, min: 0.8, max: 1 } } },
  });

  new Chart($('#chart-classes'), {
    type: 'bar',
    data: {
      labels: CLASS_REPORT.map((c) => c.label),
      datasets: [
        { label: 'Precision', data: CLASS_REPORT.map((c) => c.precision), backgroundColor: 'rgba(99,102,241,.78)', borderRadius: 6 },
        { label: 'Recall', data: CLASS_REPORT.map((c) => c.recall), backgroundColor: 'rgba(34,211,238,.72)', borderRadius: 6 },
        { label: 'F1', data: CLASS_REPORT.map((c) => c.f1), backgroundColor: 'rgba(168,85,247,.72)', borderRadius: 6 },
      ],
    },
    options: { ...chartDefaults(), scales: { ...chartDefaults().scales, y: { ...chartDefaults().scales.y, min: 0.75, max: 1 } } },
  });

  new Chart($('#chart-confusion'), {
    type: 'bar',
    data: {
      labels: ['Predicted Negative', 'Predicted Positive'],
      datasets: [
        { label: 'True Negative', data: CONFUSION[0], backgroundColor: 'rgba(251,113,133,.8)', borderRadius: 6 },
        { label: 'True Positive', data: CONFUSION[1], backgroundColor: 'rgba(52,211,153,.8)', borderRadius: 6 },
      ],
    },
    options: {
      ...chartDefaults(),
      indexAxis: 'y',
      scales: { ...chartDefaults().scales, x: { ...chartDefaults().scales.x, stacked: false }, y: { ...chartDefaults().scales.y, stacked: false } },
    },
  });
}

async function drawCatalogueCharts() {
  try {
    const [cats, products] = await Promise.all([
      api.getCategories(),
      api.getProducts({ page: 1, pageSize: 500 }),
    ]);
    const rows = products.results;

    const top = [...cats].sort((a, b) => b.product_count - a.product_count).slice(0, 10);

    if (typeof Chart !== 'undefined') {
      new Chart($('#chart-cats'), {
        type: 'bar',
        data: {
          labels: top.map((c) => c.name_translated.replace(/_/g, ' ')),
          datasets: [
            {
              label: 'Products in category',
              data: top.map((c) => c.product_count),
              backgroundColor: 'rgba(99,102,241,.78)',
              borderRadius: 6,
            },
          ],
        },
        options: { ...chartDefaults(), indexAxis: 'y', plugins: { ...chartDefaults().plugins, legend: { display: false } } },
      });

      const buckets = { Positive: 0, Mixed: 0, Negative: 0, 'Not scored': 0 };
      rows.forEach((p) => {
        const s = Number(p.avg_sentiment_score || 0);
        if (s <= 0) buckets['Not scored'] += 1;
        else if (s >= 0.7) buckets.Positive += 1;
        else if (s >= 0.45) buckets.Mixed += 1;
        else buckets.Negative += 1;
      });

      new Chart($('#chart-sentiment'), {
        type: 'doughnut',
        data: {
          labels: Object.keys(buckets),
          datasets: [
            {
              data: Object.values(buckets),
              backgroundColor: ['rgba(52,211,153,.85)', 'rgba(251,191,36,.85)', 'rgba(251,113,133,.85)', 'rgba(125,134,163,.5)'],
              borderColor: 'rgba(7,10,20,.9)',
              borderWidth: 3,
            },
          ],
        },
        options: { ...chartDefaults(), cutout: '62%', scales: {} },
      });
    }

    $('#analytics-source').textContent = `${num(rows.length)} products · ${cats.length} categories`;
  } catch (err) {
    $('#chart-cats').parentElement.innerHTML = `<p class="text-sm" style="color:var(--bad)">${escapeHtml(describeError(err))}</p>`;
  }
}

async function main() {
  renderKpis();
  $('#classification-report').textContent = CLASSIFICATION_REPORT;

  await bootstrapData();
  drawModelCharts();
  await drawCatalogueCharts();
}

// Chart.js is loaded with `defer`; wait for it before drawing.
if (document.readyState === 'complete') main();
else window.addEventListener('load', main);
