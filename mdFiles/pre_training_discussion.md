# Pre-Training Discussion: Sentiment Classification on Olist Reviews

---

## Q1: Are we training an NLP classification model?

**Yes**, but let's be precise about *what kind* of classification.

Your `review_score` column has **5 classes** (1–5 stars). You have two options:

| Approach | Classes | Description |
|---|---|---|
| **5-class classification** | 1, 2, 3, 4, 5 | Predict the exact star rating |
| **3-class sentiment** | Negative, Neutral, Positive | Merge 1-2 → Negative, 3 → Neutral, 4-5 → Positive |
| **Binary sentiment** | Negative, Positive | Merge 1-2 → Negative, 4-5 → Positive (drop 3) |

### Your data distribution (reviews WITH text only — 40,949 rows):

| Score | Count | % | Category |
|---|---|---|---|
| ⭐ 1 | 8,743 | 21.4% | Negative |
| ⭐ 2 | 2,145 | 5.2% | Negative |
| ⭐ 3 | 3,556 | 8.7% | Neutral |
| ⭐ 4 | 5,970 | 14.6% | Positive |
| ⭐ 5 | 20,535 | 50.1% | Positive |

> [!IMPORTANT]
> **The data is heavily imbalanced** — score 5 alone is 50% of the dataset, while score 2 is only 5.2%. This matters for model choice.

### Recommendation
**3-class sentiment (Negative/Neutral/Positive)** is the best fit for a recommendation system because:
- It directly maps to actionable categories (bad/okay/good)
- It reduces the class imbalance problem (Negative ~26.6%, Neutral ~8.7%, Positive ~64.7%)
- 5-class is hard even for humans — the difference between 4 and 5 stars is subjective
- It's easier for downstream recommendation logic to use 3 buckets

---

## Q2: What is the most powerful algorithm?

Here's a comparison of the realistic options, ranked by **expected accuracy on this task**:

| Rank | Model | Type | Expected Accuracy | Training Time | GPU Required? |
|---|---|---|---|---|---|
| 🥇 | **Fine-tuned BERT/DistilBERT** | Transformer | ~88-92% | 20-40 min | ✅ Yes (you have one) |
| 🥈 | **Fine-tuned RoBERTa** | Transformer | ~89-93% | 30-60 min | ✅ Yes |
| 🥉 | **TF-IDF + Logistic Regression** | Classical ML | ~82-86% | 1-2 min | ❌ No |
| 4 | **TF-IDF + SVM** | Classical ML | ~81-85% | 2-5 min | ❌ No |
| 5 | **TF-IDF + Random Forest** | Classical ML | ~78-83% | 3-10 min | ❌ No |
| 6 | **LSTM/GRU** | Deep Learning | ~83-87% | 15-30 min | ✅ Recommended |
| 7 | **Naive Bayes** | Classical ML | ~75-80% | < 1 min | ❌ No |

### Why fine-tuned BERT/DistilBERT is the best choice here:

1. **You already have a GPU** — so the training cost is trivial
2. **~40K samples** is the sweet spot for fine-tuning — enough data to learn, not so much that training takes forever
3. **DistilBERT** is 60% faster than BERT with only ~3% accuracy loss — best bang for buck
4. **Pre-trained on English** — perfect since we translated to English
5. **Handles context and negation well** — "not bad" = positive, which trips up TF-IDF models

> [!TIP]
> **My recommendation: Start with DistilBERT.** It's the best trade-off of power vs. speed. If you're not happy with the results, we can upgrade to RoBERTa.

---

## Q3: Is there a train/dev/test split in this data?

**No.** The Olist dataset does **not** come with a pre-defined train/test split. We need to create one ourselves.

### Proposed split strategy:

```
Total reviews with text: 40,949
├── Train:  32,759 (80%)
├── Validation: 4,095 (10%)  ← for tuning hyperparameters
└── Test:   4,095 (10%)  ← held-out, never seen during training
```

### Important considerations:

1. **Stratified split** — We must preserve the class distribution in each split (so each split has the same % of 1-star, 2-star, etc.)
2. **No data leakage** — Some customers have multiple reviews. We should split by `order_id` to ensure the same customer's reviews don't appear in both train and test.
3. **Rows without text** — 58,275 rows have no review text at all (just a score). We'll **exclude** these from NLP training since there's nothing for the model to learn from.

---

## Q4: Was the translation step correct?

### ✅ What we did right:
1. **Translating before training** — Training on English text lets us use powerful pre-trained English models (BERT, DistilBERT, RoBERTa) which are much better than Portuguese-only models
2. **Keeping the original text** — We preserved the original Portuguese column so we can always go back
3. **Quality assessment** — We verified 97.8% of translations are usable

### ⚠️ Alternative we chose NOT to do (and why):
| Alternative | Why we skipped it |
|---|---|
| Train on Portuguese directly using `BERTimbau` | Would avoid translation errors, but BERTimbau is less mature and has fewer fine-tuning resources |
| Use multilingual BERT (`bert-base-multilingual-cased`) | Weaker than English-only BERT on English text, and weaker than BERTimbau on Portuguese |
| Use an LLM API (GPT-4, Gemini) for translation | Better quality, but expensive at 40K rows and introduces API dependency |

> [!NOTE]
> **The translation step was a valid choice.** The 1.9% hallucination rate is on very short inputs (< 10 chars like "OK", "bom") that carry minimal information anyway. For a recommendation system, this noise level is acceptable.

### One thing to be aware of:
The translation model occasionally over-generates on short inputs (e.g., "OK" → "Okay, that's it."). For training, this is fine because the model will learn from the full distribution. But if you deploy this for **real-time inference**, you should translate user queries with a better model (or use BERTimbau directly).

---

## Summary of Decisions Needed

> [!IMPORTANT]
> Please confirm these choices before we proceed:

1. **Classification type**: 3-class sentiment (Negative/Neutral/Positive) — or do you prefer 5-class or binary?
2. **Model**: DistilBERT fine-tuning — or do you want to start with a simpler baseline first?
3. **Split**: 80/10/10 stratified — good?
4. **Rows without text**: Exclude from training — agreed?
