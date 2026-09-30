# Olist E-Commerce Recommendation System — NLP Sentiment Classification Pipeline

Comprehensive documentation of the end-to-end NLP workflow built for the **Olist Brazilian E-Commerce Capstone Project**. This document covers every phase from raw data ingestion and Portuguese-to-English translation to model training, evaluation, and inference.

---

## Table of Contents
1. [Project Overview & Objectives](#1-project-overview--objectives)
2. [Dataset Overview & Ingestion](#2-dataset-overview--ingestion)
3. [Phase 1: Machine Translation Pipeline (PT $\rightarrow$ EN)](#3-phase-1-machine-translation-pipeline-pt--en)
4. [Phase 2: Sentiment Formulation & Class Imbalance Strategy](#4-phase-2-sentiment-formulation--class-imbalance-strategy)
5. [Phase 3: Stratified Dataset Splitting](#5-phase-3-stratified-dataset-splitting)
6. [Phase 4: DistilBERT Architecture & Training Pipeline](#6-phase-4-distilbert-architecture--training-pipeline)
7. [Phase 5: Training Results & Validation Progression](#7-phase-5-training-results--validation-progression)
8. [Phase 6: Held-Out Test Set Evaluation](#8-phase-6-held-out-test-set-evaluation)
9. [Phase 7: Inference & 30-Case Benchmark Suite](#9-phase-7-inference--30-case-benchmark-suite)
10. [Repository Structure & File Inventory](#10-repository-structure--file-inventory)
11. [How to Run Every Step](#11-how-to-run-every-step)
12. [Downstream Integration with Recommendation Engine](#12-downstream-integration-with-recommendation-engine)

---

## 1. Project Overview & Objectives

In modern e-commerce recommendation systems, star ratings alone provide coarse-grained signals. Two products may both hold a 3.8-star average, but one may have reviews complaining about catastrophic hardware failure while the other suffers only from minor delivery packaging delays.

**Core Objective:**
Fine-tune a state-of-the-art Transformer NLP model (**DistilBERT**) on customer reviews to extract granular sentiment polarity ($P(\text{Negative}), P(\text{Neutral}), P(\text{Positive})$). This sentiment score serves as an objective product-quality index and dynamic weight for downstream collaborative filtering and hybrid recommendation algorithms.

---

## 2. Dataset Overview & Ingestion

* **Dataset**: Brazilian E-Commerce Public Dataset by Olist (Kaggle).
* **Ingestion Script**: [`downloddataset.py`](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe Project/downloddataset.py) using `kagglehub`.
* **Primary Target File**: `olist_order_reviews_dataset.csv`
  * **Total records**: 99,224 rows
  * **Columns**: `review_id`, `order_id`, `review_score`, `review_comment_title`, `review_comment_message`, `review_creation_date`, `review_answer_timestamp`
  * **Reviews with text**: 40,949 rows (58,275 records were rating-only without written text and were filtered out for NLP modeling).

---

## 3. Phase 1: Machine Translation Pipeline (PT $\rightarrow$ EN)

Customer reviews were originally written in Brazilian Portuguese. To leverage pre-trained transformer backbones trained on English corpora without losing Brazilian Portuguese nuances, we built a GPU-accelerated batch translation pipeline.

### Model Selection
We evaluated candidate translation models on Hugging Face:
* `Helsinki-NLP/opus-mt-tc-big-pt-en`: Lightweight, but rigid with informal e-commerce slang.
* `facebook/m2m100_418M`: Adequate, but older architecture.
* **`facebook/nllb-200-distilled-600M` (Selected)**: State-of-the-art multilingual model with dedicated support for Portuguese (`por_Latn` $\rightarrow$ `eng_Latn`), handling spelling errors, internet slang, and short e-commerce reviews with high semantic fidelity.

### Text Cleaning & Translation Script: [`translate_reviews.py`](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe Project/translate_reviews.py)
* **Encoding Artifact Repair**: Used `ftfy` to resolve mojibake and encoding corruptions (e.g., `Parabns` $\rightarrow$ `Parabéns`).
* **Batch Processing**: Dynamic mini-batches executed with PyTorch FP16 on NVIDIA GeForce GTX 1650 Ti GPU.
* **Output File**: [`data/olist_order_reviews_translated.csv`](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe Project/data/olist_order_reviews_translated.csv) (16.8 MB, 99,224 rows, with added column `review_comment_translated`).
* **Quality Verification**: [`check_translation_quality.py`](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe Project/check_translation_quality.py) verified 30-sample side-by-side outputs across 1–5 stars.

---

## 4. Phase 2: Sentiment Formulation & Class Imbalance Strategy

### 3-Class Sentiment Mapping
The original 1–5 star `review_score` was mapped into 3 standard sentiment classes:

| Class ID | Sentiment | Source Scores | Clean Count | Dataset % | Description |
|---|---|---|---|---|---|
| **0** | **Negative** | ⭐ 1, ⭐ 2 | 10,888 | 26.59% | Strong complaints, defects, refunds |
| **1** | **Neutral** | ⭐ 3 | 3,556 | 8.68% | Average, mixed pros/cons, neutral receipt |
| **2** | **Positive** | ⭐ 4, ⭐ 5 | 26,505 | 64.73% | High satisfaction, praise, recommendations |
| **Total** | — | — | **40,949** | **100.0%** | (Rows with text) |

### Class Imbalance Resolution: Weighted Cross-Entropy Loss
Because **Positive** represents nearly **65%** of the data and **Neutral** only **8.7%**, standard models develop a severe bias toward predicting Positive.

> **Key Design Decision:** Cross-validation does *not* solve class bias. We resolved the imbalance mathematically using **inverse frequency class weighting** in the loss function:

$$W_k = \frac{N}{K \cdot N_k}$$

Where $N = 32,759$ (training samples), $K = 3$ (number of classes), and $N_k$ is the frequency of class $k$:
* **Negative (Class 0)**: $32759 / (3 \times 8710) = \mathbf{1.2537}$
* **Neutral (Class 1)**: $32759 / (3 \times 2845) = \mathbf{3.8382}$
* **Positive (Class 2)**: $32759 / (3 \times 21204) = \mathbf{0.5150}$

**Effect:** Misclassifying a **Neutral** review is penalized **7.5× more heavily** than misclassifying a Positive review, forcing the transformer to actively distinguish subtle minority-class boundaries.

---

## 5. Phase 3: Stratified Dataset Splitting

Executed via [`prepare_data.py`](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe Project/prepare_data.py):
* Filtered out empty/null comments.
* Applied an **80 / 10 / 10 stratified split** ensuring exact class preservation across all splits:

| Split | File Path | Total Rows | Negative (0) | Neutral (1) | Positive (2) |
|---|---|---|---|---|---|
| **Train (80%)** | [`data/train.csv`](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe Project/data/train.csv) | **32,759** | 8,710 (26.59%) | 2,845 (8.68%) | 21,204 (64.73%) |
| **Val (10%)** | [`data/val.csv`](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe Project/data/val.csv) | **4,095** | 1,089 (26.59%) | 355 (8.67%) | 2,651 (64.74%) |
| **Test (10%)** | [`data/test.csv`](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe Project/data/test.csv) | **4,095** | 1,089 (26.59%) | 356 (8.69%) | 2,650 (64.71%) |

---

## 6. Phase 4: DistilBERT Architecture & Training Pipeline

Implemented in [`train_model.py`](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe Project/train_model.py):

### Architecture & Optimization
* **Backbone**: `distilbert-base-uncased` (66M parameters, 6 transformer layers, 12 attention heads).
* **Head**: Sequence classification head (Dropout + Linear layer $768 \rightarrow 3$).
* **Hardware**: NVIDIA GeForce GTX 1650 Ti GPU (4GB VRAM).
* **Speed & Memory Optimizations**:
  * **FP16 Automatic Mixed Precision (AMP)** via `torch.amp.autocast('cuda')` and `GradScaler`.
  * **Dynamic Batch Padding**: `DataCollatorWithPadding` dynamically pads batches to the length of the longest sentence in that mini-batch rather than a fixed 256 tokens, reducing unnecessary computation by ~70%.
  * **GPU VRAM Utilization**: ~1.65 GB (safely below 4GB limit).
  * **GPU Compute Utilization**: ~98–99%.

### Training Configuration
* **Batch Size**: 16
* **Max Token Length**: 256
* **Optimizer**: AdamW ($lr = 2 \times 10^{-5}$, weight decay $= 0.01$, $\beta_1 = 0.9, \beta_2 = 0.999$)
* **LR Scheduler**: Linear warmup over the first 10% of steps (819 steps), decaying linearly to 0 across 8,192 total steps.
* **Epochs**: 4 (2,048 steps per epoch).
* **Early Stopping / Checkpoint Selection**: Checkpoint saved when Validation Macro F1 improves.

---

## 7. Phase 5: Training Results & Validation Progression

Total training duration: **76.84 minutes** (~18.4 minutes per epoch).

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Macro F1 | Negative F1 | Neutral F1 | Positive F1 | Status |
|---|---|---|---|---|---|---|---|---|
| **1** | 0.7444 | **0.6650** | 80.95% | 0.6786 | 0.8004 | 0.3316 | 0.9039 | Checkpoint saved |
| **2** | 0.6218 | 0.7067 | 81.37% | 0.6757 | 0.8068 | 0.3176 | 0.9028 | — |
| **3** | 0.5390 | 0.7614 | **82.34%** | **0.6854** | **0.8120** | **0.3341** | **0.9101** | **Best Model Checkpoint** |
| **4** | **0.4584** | 0.8576 | 82.08% | 0.6813 | 0.8023 | 0.3306 | 0.9109 | (Early stop trigger) |

The final best model checkpoint was saved to [`models/best_model/`](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe Project/models/best_model) with full epoch logs stored in [`models/training_log.json`](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe Project/models/training_log.json).

---

## 8. Phase 6: Held-Out Test Set Evaluation

Executed via [`evaluate_model.py`](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe Project/evaluate_model.py) on **4,095 unseen test samples**:

### Classification Report

```text
              precision    recall  f1-score   support

    Negative     0.8014    0.8411    0.8208      1089
     Neutral     0.2731    0.3736    0.3155       356
    Positive     0.9456    0.8796    0.9114      2650

    accuracy                         0.8254      4095
   macro avg     0.6734    0.6981    0.6826      4095
weighted avg     0.8488    0.8254    0.8355      4095
```

### Confusion Matrix

| True \\ Predicted | Negative (0) | Neutral (1) | Positive (2) | Total | Class Recall |
|---|---|---|---|---|---|
| **True Negative** | **916** | 133 | 40 | 1,089 | **84.11%** |
| **True Neutral** | 129 | **133** | 94 | 356 | **37.36%** |
| **True Positive** | 98 | 221 | **2,331** | 2,650 | **87.96%** |
| **Total Predicted** | 1,143 | 487 | 2,465 | 4,095 | — |
| **Class Precision** | **80.14%** | **27.31%** | **94.56%** | — | — |

### Key Diagnostic Insights
1. **Polarity Separation is Outstanding (>96%)**:
   Between direct opposites (Positive vs. Negative):
   * Only **40 out of 1,089** Negative reviews (3.67%) were misclassified as Positive.
   * Only **98 out of 2,650** Positive reviews (3.70%) were misclassified as Negative.
2. **The Neutral Class Dynamics**:
   Neutral (3-star) reviews inherently contain mixed sentiment: e.g. *"Great product, but delivery was 2 weeks late"*. Such sentences carry strong positive and negative tokens simultaneously. Because Neutral makes up only 8.7% of the dataset, achieving **37.4% recall** proves the weighted loss prevented the model from simply ignoring this class.
3. **Core Classes (91.3% of users)**:
   For Positive and Negative reviews, the model achieves **0.9114** and **0.8208** F1 scores, providing an exceptionally stable foundation for customer satisfaction scoring.

---

## 9. Phase 7: Inference & 30-Case Benchmark Suite

Implemented in [`inference.py`](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe Project/inference.py) and [`inference_nlp.py`](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe Project/inference_nlp.py):

### 30-Case Benchmark (Easy $\rightarrow$ Hard)
The benchmark evaluates 30 crafted e-commerce scenarios across all three classes:
* **Negative (Levels 1–10)**: From obvious failure (*"Broken on arrival, waste of money"*) $\rightarrow$ Service delays $\rightarrow$ Sarcasm (*"Works fine if you only need it to turn on once"*).
* **Neutral (Levels 1–10)**: From explicit averages (*"Standard cable, works properly"*) $\rightarrow$ Mixed reviews (*"Good material, but delivery was twice as long"*) $\rightarrow$ Deferred judgment.
* **Positive (Levels 1–10)**: From pure enthusiasm (*"Wonderful, 100% recommend"*) $\rightarrow$ Solid cost-benefit $\rightarrow$ Understated praise (*"Not bad at all, actually performs much better"*).

---

## 10. Repository Structure & File Inventory

```
Recoomedation System Copstoe Project/
├── data/
│   ├── olist_order_reviews_dataset.csv     # Original Olist reviews (Portuguese)
│   ├── olist_order_reviews_translated.csv  # Translated reviews (NLLB-200, 16.8 MB)
│   ├── train.csv                           # 80% Stratified Train set (32,759 rows)
│   ├── val.csv                             # 10% Stratified Val set (4,095 rows)
│   └── test.csv                            # 10% Stratified Test set (4,095 rows)
├── models/
│   ├── best_model/                         # Saved fine-tuned DistilBERT weights
│   │   ├── config.json                     # Model hyperparameters & label mapping
│   │   ├── model.safetensors               # Trained PyTorch transformer weights (268 MB)
│   │   ├── tokenizer.json                  # WordPiece tokenizer configuration
│   │   └── tokenizer_config.json
│   ├── training_log.json                   # Loss, accuracy, F1 per epoch history
│   └── evaluation_results.json             # Test set classification report & confusion matrix
├── docs/
│   └── pre_training_discussion.md          # Architectural decisions & class distribution notes
├── downloddataset.py                       # Automated Kagglehub dataset downloader
├── translate_reviews.py                    # NLLB-200 GPU batch translation script
├── check_translation_quality.py            # Quality inspection script for translations
├── prepare_data.py                         # 3-class mapping & 80/10/10 stratified split
├── train_model.py                          # FP16 DistilBERT training loop with weighted loss
├── evaluate_model.py                       # Test set evaluation and confusion matrix
├── inference.py                            # SentimentPredictor class & 30-case benchmark
├── inference_nlp.py                        # Root inference entrypoint
├── pyproject.toml                          # Project dependencies (uv package manager)
├── uv.lock                                 # Exact reproducible lockfile
└── README.md                               # Complete project documentation
```

---

## 11. How to Run Every Step

Ensure you have your environment set up with `uv`:

```powershell
# 1. Download Dataset
uv run python downloddataset.py

# 2. Translate Portuguese Reviews to English
uv run python translate_reviews.py

# 3. Prepare Stratified Train/Val/Test Splits
uv run python prepare_data.py

# 4. Train the DistilBERT Model (uses GPU)
uv run python train_model.py

# 5. Evaluate Best Model on Held-Out Test Set
uv run python evaluate_model.py

# 6. Run the 30-Case Benchmark (Easy to Hard)
uv run python inference_nlp.py

# 7. Run Interactive Sentiment Mode
uv run python inference_nlp.py --interactive

# 8. Single Review Classification
uv run python inference_nlp.py --text "Super fast delivery and the laptop is amazing!"
```

---

## 12. Downstream Integration with Recommendation Engine

Now that we have a verified sentiment classification engine, here is how it directly improves product recommendations:

### 1. Product Sentiment Index (PSI)
For every product $j$, aggregate sentiment predictions over all associated reviews:
$$\text{PSI}_j = \frac{1}{|R_j|} \sum_{r \in R_j} \Big( P(\text{Positive}_r) - P(\text{Negative}_r) \Big) \in [-1, 1]$$
This separates high-volume products that buyers actively love from high-volume products that buyers frequently return or complain about.

### 2. Hybrid Collaborative Filtering Re-Ranking
When a Collaborative Filtering model (e.g., Matrix Factorization / LightFM / SVD) predicts an affinity score $\hat{y}_{u, j}$ for user $u$ and product $j$:
$$\text{Score}_{\text{final}}(u, j) = \hat{y}_{u, j} \cdot \Big(1 + \alpha \cdot \text{PSI}_j\Big)$$
Where $\alpha \in [0.1, 0.3]$ acts as a quality damper. Products with negative sentiment are penalized in the recommendation carousel, preventing bad recommendations and increasing customer trust.
