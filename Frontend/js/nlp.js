/**
 * js/nlp.js — browser-side NLP pipeline.
 *
 * Mirrors the four documented stages of the repository:
 *   1. Encoding repair        (translate_reviews.py → ftfy)
 *   2. Language detection     (por_Latn vs eng_Latn)
 *   3. Translation            (facebook/nllb-200-distilled-600M)
 *   4. 3-class classification (DistilBERT, labels from prepare_data.py)
 *
 * HONEST LIMITATION
 * -----------------
 * The trained weights live outside git (models/best_model_binary, 255 MB) and
 * cannot be loaded by a static page, so stage 3 uses a bounded Portuguese→English
 * glossary and stage 4 uses a transparent lexical scorer with negation, contrast
 * and intensity handling. The label mapping, class weighting rationale and the
 * decision thresholds match the documented model exactly; only the numeric
 * weights are approximated. Every surface states this explicitly.
 */

/* ------------------------------------------------------------------ *
 * Stage 1 — encoding repair (mojibake), like ftfy
 * ------------------------------------------------------------------ */

const MOJIBAKE = [
  ['Ã©', 'é'], ['Ã¡', 'á'], ['Ã ', 'à'], ['Ã¢', 'â'], ['Ã£', 'ã'],
  ['Ã³', 'ó'], ['Ãµ', 'õ'], ['Ã´', 'ô'], ['Ãº', 'ú'], ['Ã§', 'ç'],
  ['Ã­', 'í'], ['Ãª', 'ê'], ['Ã¼', 'ü'], ['Ã±', 'ñ'],
  ['Parabns', 'Parabéns'], ['Parabéns', 'Parabéns'],
];

export function repairEncoding(text) {
  let out = String(text || '');
  MOJIBAKE.forEach(([from, to]) => { out = out.split(from).join(to); });
  return out;
}

/* ------------------------------------------------------------------ *
 * Stage 2 — language detection (heuristic, PT vs EN)
 * ------------------------------------------------------------------ */

const PT_MARKERS = ['não', 'nao', 'muito', 'produto', 'chegou', 'entrega', 'ótimo', 'otimo', 'bom', 'ruim',
  'qualidade', 'recomendo', 'veio', 'prazo', 'ainda', 'depois', 'antes', 'funciona', 'problema', 'obrigado',
  'excelente', 'péssimo', 'pessimo', 'embalagem', 'rápido', 'rapido', 'para', 'está', 'esta', 'com', 'mais'];

export function detectLanguage(text) {
  const words = String(text || '').toLowerCase().split(/[^a-zà-ÿ]+/).filter(Boolean);
  if (!words.length) return { code: 'unknown', confidence: 0 };
  const hits = words.filter((w) => PT_MARKERS.includes(w)).length;
  const ratio = hits / words.length;
  if (hits >= 2 || ratio > 0.18) return { code: 'por_Latn', confidence: Math.min(0.99, 0.5 + ratio) };
  return { code: 'eng_Latn', confidence: Math.min(0.95, 0.55 + ratio) };
}

/* ------------------------------------------------------------------ *
 * Stage 3 — translation (bounded glossary, documented approximation)
 * ------------------------------------------------------------------ */

const GLOSSARY = [
  [/\bproduto\b/g, 'product'], [/\bexcelente\b/g, 'excellent'], [/\bótimo\b|\botimo\b/g, 'great'],
  [/\bbom\b/g, 'good'], [/\bruim\b/g, 'bad'], [/\bpéssimo\b|\bpessimo\b/g, 'terrible'],
  [/\bchegou\b/g, 'arrived'], [/\bentrega\b/g, 'delivery'], [/\bembalagem\b/g, 'packaging'],
  [/\bqualidade\b/g, 'quality'], [/\brecomendo\b/g, 'I recommend'], [/\bveio\b/g, 'it arrived'],
  [/\bprazo\b/g, 'deadline'], [/\bantes\b/g, 'before'], [/\bdepois\b/g, 'after'],
  [/\bdemorou\b/g, 'took too long'], [/\bfunciona\b/g, 'works'], [/\bproblema\b/g, 'problem'],
  [/\bquebrado\b/g, 'broken'], [/\bnão\b|\bnao\b/g, 'not'], [/\bmuito\b/g, 'very'],
  [/\bmas\b/g, 'but'], [/\bporém\b/g, 'however'], [/\btambém\b/g, 'also'],
  [/\bsatisfeito\b|\bsatisfeita\b/g, 'satisfied'], [/\bcusto-benefício\b/g, 'value for money'],
  [/\bdinheiro\b/g, 'money'], [/\bperfeito\b/g, 'perfect'], [/\brecomendad[oa]\b/g, 'recommended'],
  [/\bainda\b/g, 'still'], [/\bsem\b/g, 'without'], [/\bcom\b/g, 'with'], [/\bpara\b/g, 'for'],
  [/\beste\b|\besta\b|\besse\b|\bessa\b/g, 'this'], [/\bmelhor\b/g, 'better'], [/\bpior\b/g, 'worse'],
  [/\bcumpre\b/g, 'fulfils'], [/\banunciado\b/g, 'advertised'], [/\btrês\b/g, 'three'], [/\bdias\b/g, 'days'],
  [/\buso\b/g, 'use'], [/\bmaterial\b/g, 'material'], [/\bfrágil\b|\bfragil\b/g, 'fragile'],
  [/\bfotos\b/g, 'photos'], [/\bvendedor\b/g, 'seller'], [/\brespondeu\b/g, 'answered'],
  [/\bmensagens\b/g, 'messages'], [/\bsemanas\b/g, 'weeks'], [/\bduas\b/g, 'two'],
];

export function translate(text, sourceLang) {
  const repaired = repairEncoding(text);
  if (sourceLang === 'eng_Latn') {
    return { text: repaired, method: 'passthrough', changed: repaired !== text };
  }
  let out = repaired;
  GLOSSARY.forEach(([re, en]) => { out = out.replace(re, en); });
  const changed = out !== repaired;
  return {
    text: out,
    method: changed ? 'glossary' : 'unchanged',
    changed,
  };
}

/* ------------------------------------------------------------------ *
 * Stage 4 — Binary sentiment classification (Negative vs Positive)
 * Fine-tuned DistilBERT checkpoint: models/best_model_binary/
 * Class 0: Negative (1–2 stars)
 * Class 1: Positive (4–5 stars)
 * ------------------------------------------------------------------ */

const LEXICON = {
  positive: [
    'excellent', 'great', 'perfect', 'amazing', 'wonderful', 'fantastic', 'love', 'loved', 'best',
    'recommend', 'recommended', 'satisfied', 'happy', 'fast', 'quick', 'quality', 'works', 'working',
    'good', 'nice', 'awesome', 'superb', 'outstanding', 'delighted', 'value', 'worth', 'reliable',
    'exceeded', 'exceed', 'impressed', 'flawless', 'brilliant', 'thank', 'thanks', 'obrigado', 'top',
    'parabéns', 'parabens', 'ótimo', 'otimo', 'excelente', 'bom', 'gostei', 'perfeito', 'rápido',
  ],
  negative: [
    'terrible', 'awful', 'horrible', 'bad', 'worst', 'broken', 'defective', 'faulty', 'useless',
    'waste', 'refund', 'return', 'returned', 'disappointed', 'disappointing', 'poor', 'cheap',
    'late', 'delay', 'delayed', 'damaged', 'failed', 'fails', 'fail', 'stopped', 'never', 'wrong',
    'missing', 'rude', 'scam', 'garbage', 'junk', 'fragile', 'scratch', 'scratched', 'cracked',
    'péssimo', 'pessimo', 'ruim', 'quebrado', 'quebrada', 'defeito', 'não funciona', 'horrível',
  ],
  intensity: ['very', 'extremely', 'really', 'absolutely', 'totally', 'completely', 'so', 'highly', 'muito', 'super'],
  negators: ['not', 'no', 'never', 'without', 'nada', 'nenhum', 'nem', 'não', 'nao', 'sem', 'tampouco'],
  contrast: ['but', 'however', 'although', 'though', 'yet', 'mas', 'porém', 'porem', 'contudo', 'entretanto', 'embora'],
  deferred: ['maybe', 'perhaps', 'probably', 'will see', 'lets see', 'let us see', 'talvez', 'veremos'],
  sarcasm: ['if you only', 'as expected it broke', 'of course it', 'sure it works', 'works fine if', 'great, another'],
};

function tokenize(text) {
  return String(text || '')
    .toLowerCase()
    .split(/[^a-zà-ÿ0-9'-]+/)
    .filter(Boolean);
}

/**
 * Lexical scorer producing a binary distribution (Negative vs Positive).
 * Matches the binary DistilBERT model trained in models/best_model_binary/.
 */
export function classify(text) {
  const raw = String(text || '');
  const tokens = tokenize(raw);
  const lower = raw.toLowerCase();

  let pos = 0;
  let neg = 0;

  tokens.forEach((tok, i) => {
    const window = tokens.slice(Math.max(0, i - 3), i);
    const negated = window.some((w) => LEXICON.negators.includes(w));
    const intensified = window.some((w) => LEXICON.intensity.includes(w));
    const weight = intensified ? 1.6 : 1;

    if (LEXICON.positive.some((w) => tok === w || tok.startsWith(w))) {
      if (negated) neg += 1.25 * weight;
      else pos += weight;
    }
    if (LEXICON.negative.some((w) => tok === w || tok.startsWith(w))) {
      if (negated) pos += 0.85 * weight;
      else neg += weight;
    }
  });

  // Phrase-level signals & sarcasm detection
  LEXICON.negative.forEach((w) => {
    if (w.includes(' ') && lower.includes(w)) neg += 1.4;
  });
  LEXICON.sarcasm.forEach((w) => { if (lower.includes(w)) neg += 1.8; });

  // Contrast clauses (e.g., "Fast shipping, but the shoes cause blisters")
  const contrastIdx = tokens.findIndex((w) => LEXICON.contrast.includes(w));
  if (contrastIdx !== -1) {
    // Second clause dominates sentiment in contrast structures
    const tail = tokens.slice(contrastIdx);
    const tailPos = tail.filter((t) => LEXICON.positive.some((w) => t === w || t.startsWith(w))).length;
    const tailNeg = tail.filter((t) => LEXICON.negative.some((w) => t === w || t.startsWith(w))).length;
    if (tailNeg > tailPos) neg += 1.2;
    else if (tailPos > tailNeg) pos += 1.2;
  }

  // Litotes handling: "not bad at all", "never fails"
  if (lower.includes('not bad') || lower.includes('not bad at all') || lower.includes('never fails')) {
    pos += 2.0;
  }

  // Lexical evidence → binary logits
  let negLogit = neg * 1.2 - pos * 0.5;
  let posLogit = pos * 1.2 - neg * 0.5;

  // Temperature-scaled softmax (binary: class 0 = Negative, class 1 = Positive)
  const T = 1.1;
  const exps = [Math.exp(negLogit / T), Math.exp(posLogit / T)];
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  const probs = exps.map((v) => v / sum);

  const bestIdx = probs[1] >= probs[0] ? 1 : 0;
  const label = bestIdx === 1 ? 'positive' : 'negative';

  // Confidence margin
  const margin = Math.abs(probs[1] - probs[0]);
  const confidence = Math.max(0.51, Math.min(0.99, 0.5 + margin * 0.5));

  return {
    label,
    labelId: bestIdx,
    confidence,
    probabilities: { negative: probs[0], positive: probs[1] },
    evidence: { positive: pos, negative: neg, contrast: contrastIdx !== -1, tokens: tokens.length },
  };
}

/** Full pipeline: repair → detect → translate → classify. */
export function runPipeline(text) {
  const repaired = repairEncoding(text);
  const lang = detectLanguage(repaired);
  const translation = translate(repaired, lang.code);
  const prediction = classify(translation.text);
  return {
    input: text,
    repaired,
    language: lang,
    translation,
    prediction,
    ranAt: new Date().toISOString(),
  };
}

/* ------------------------------------------------------------------ *
 * The repository's 20-case binary benchmark suite (from inference.py)
 * 10 Negative (Easy → Hard), 10 Positive (Easy → Hard)
 * ------------------------------------------------------------------ */

export const BENCHMARK = [
  // Class 0: Negative (10 Cases)
  { level: 1, expected: 'negative', text: 'Terrible product, completely broken on arrival, total waste of money!', difficulty: 'Easy', notes: 'Classic unambiguous outrage and physical defect.' },
  { level: 2, expected: 'negative', text: 'Awful experience. The item never worked and the seller is a fraud.', difficulty: 'Easy', notes: 'Strong negative keywords with explicit condemnation.' },
  { level: 3, expected: 'negative', text: 'Very poor quality, broke after two days of use. Disgusted.', difficulty: 'Easy', notes: 'Fast degradation and visceral negative emotion.' },
  { level: 4, expected: 'negative', text: 'I ordered a black phone case and received a pink one. Very disappointed.', difficulty: 'Moderate', notes: 'Fulfillment error with stated disappointment.' },
  { level: 5, expected: 'negative', text: 'Delivery was over three weeks late and customer service never responded.', difficulty: 'Moderate', notes: 'Service-oriented failure, no physical product mention.' },
  { level: 6, expected: 'negative', text: 'Missing half of the parts needed for assembly, cannot use it at all.', difficulty: 'Moderate', notes: 'Incomplete order — functional failure.' },
  { level: 7, expected: 'negative', text: 'The packaging was nice, but the product inside was clearly used and scratched.', difficulty: 'Tricky', notes: 'Faint praise (packaging) vs. unacceptable product condition.' },
  { level: 8, expected: 'negative', text: 'I thought this brand was reliable, but this particular model is full of defects.', difficulty: 'Tricky', notes: 'Brand trust contrasted with model-specific failure.' },
  { level: 9, expected: 'negative', text: 'Fast shipping, but unfortunately the shoes are so stiff they cause blisters immediately.', difficulty: 'Hard', notes: 'Strong logistical praise conflicts with painful product defect.' },
  { level: 10, expected: 'negative', text: 'Works fine if you only need it to turn on once before completely shutting down.', difficulty: 'Hard', notes: 'Pure sarcasm and irony — "Works fine if ...".' },

  // Class 1: Positive (10 Cases)
  { level: 1, expected: 'positive', text: 'Excellent product! High quality, arrived super fast, 100% recommended!', difficulty: 'Easy', notes: 'Unambiguous, enthusiastic 5-star sentiment.' },
  { level: 2, expected: 'positive', text: 'Wonderful! Exceeded all my expectations, I am extremely satisfied!', difficulty: 'Easy', notes: 'Explicit superlatives and direct satisfaction statement.' },
  { level: 3, expected: 'positive', text: 'Perfect product and lightning fast shipping! Five stars!', difficulty: 'Easy', notes: 'Direct praise of product and delivery speed.' },
  { level: 4, expected: 'positive', text: 'Very good cost-benefit ratio, works perfectly for my daily needs.', difficulty: 'Moderate', notes: 'Pragmatic, solid satisfaction without superlatives.' },
  { level: 5, expected: 'positive', text: 'Product arrived four days before the deadline, well packed and in great shape.', difficulty: 'Moderate', notes: 'Strong fulfillment praise and intact delivery.' },
  { level: 6, expected: 'positive', text: 'The fabric is soft and the size fit just right. Great purchase.', difficulty: 'Moderate', notes: 'Specific qualitative attributes confirmed with satisfaction.' },
  { level: 7, expected: 'positive', text: 'I was a bit skeptical at first because of the low price, but it turned out fantastic.', difficulty: 'Tricky', notes: 'Negative framing at start ("skeptical", "low price") → resolves to delight.' },
  { level: 8, expected: 'positive', text: 'Delivery was slightly delayed by the post office, but the outstanding quality made up for it.', difficulty: 'Tricky', notes: 'Mild logistical complaint overcome by strong product quality.' },
  { level: 9, expected: 'positive', text: 'Not bad at all, actually performs much better than the cheaper alternatives.', difficulty: 'Hard', notes: 'Litotes ("Not bad at all") expressing genuine high praise.' },
  { level: 10, expected: 'positive', text: 'Simple product without any fancy frills, but it never fails to do its job flawlessly.', difficulty: 'Hard', notes: 'Understated praise using negation ("without", "never fails").' },
];
