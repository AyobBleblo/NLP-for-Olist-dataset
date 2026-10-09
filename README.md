# 🛒 Olist Smart Recommendation System

> **Samsung Innovation Campus — Capstone Project**
> An end-to-end AI-powered e-commerce platform built on the Brazilian [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) dataset.
> Combines Collaborative Filtering (SVD), Content-Based Filtering (TF-IDF), and NLP Sentiment Analysis (fine-tuned DistilBERT) into a **hybrid recommendation engine** served at **sub-10ms** API response times.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture at a Glance](#-architecture-at-a-glance)
3. [What Is Included in the Repo](#-what-is-included-in-the-repo)
4. [Prerequisites](#-prerequisites)
5. [Quick Start](#-quick-start-5-steps)
6. [Detailed Setup](#-detailed-setup)
7. [Running the Frontend](#-running-the-frontend)
8. [Management Commands](#-management-commands-reference)
9. [API Reference](#-api-endpoints-reference)
10. [Test Customer IDs](#-test-customer-ids)
11. [Project Structure](#-project-structure)
12. [Optional: DistilBERT Model](#-optional-distilbert-sentiment-model-setup)
13. [Troubleshooting](#-troubleshooting)

---

## 🎯 Project Overview

This project solves a real e-commerce problem: **how do you deliver personalized recommendations at web speed when the models are expensive to run?**

The answer is **Decoupled Asynchronous Intelligence**:

- 🧠 **Offline (batch):** DistilBERT scores reviews, SVD trains on purchase history, CBF builds TF-IDF product vectors — all precomputed and stored in the database.
- ⚡ **Online (live request):** Django reads from pre-indexed database rows. Zero ML inference in the request path. Response time: **< 10 ms**.
- 🔄 **On checkout:** A lightweight event recomputes only that customer's recommendations using cached vectors — no GPU needed.

### Key NLP Results

| Model | Test Accuracy | Notes |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 93.0 % | Binary sentiment, 37 K reviews |
| Fine-tuned DistilBERT (NLLB-200 translation) | 93.34 % | 249 errors / 3 740 |
| Fine-tuned DistilBERT (Qwen2.5-7B translation) | **94.0 %** | 209 errors / 3 727 — **−23 % fewer missed negatives** |

---

## 🏗️ Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────────┐
│                     OFFLINE BATCH PIPELINE                       │
│  Kaggle CSVs → Qwen2.5-7B Translation → DistilBERT Sentiment    │
│  → SVD Collaborative Filtering → TF-IDF Content-Based Filtering  │
│  → Precomputed scores written to SQLite database                 │
└────────────────────────────┬─────────────────────────────────────┘
                             │ precomputed rows
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│            DJANGO 6.1 + DRF REST API  (< 10 ms)                 │
│  11 endpoints · No ML inference in request path · CORS enabled   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ JSON
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    VANILLA JS FRONTEND                           │
│  index · catalog · recommendations · nlp-lab · cart · analytics  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📦 What Is Included in the Repo

| Component | Status | Notes |
|---|---|---|
| Django backend source | ✅ Fully included | `store/`, `recsys_backend/` |
| Vanilla JS frontend | ✅ Fully included | `Frontend/` |
| SVD Collaborative Filtering model | ✅ **Git LFS** | `models/svd_cf.pkl` (~30 MB) |
| CBF TF-IDF artifacts | ✅ **Git LFS** | `models/processed/` (~3 MB) |
| DistilBERT sentiment model | ⚠️ **Google Drive** | `models/best_model_binary/` (~255 MB) |
| Raw Olist CSV dataset | ❌ Not included | Download from Kaggle (optional) |
| SQLite database | ❌ Not included | Generated locally via seed commands |

> **You can fully run the app without the DistilBERT model.** Recommendations and the API work fine; sentiment scores just default to 0 for new reviews.

---

## ✅ Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | **3.13+** | [python.org](https://www.python.org/downloads/) |
| Git | Any recent | [git-scm.com](https://git-scm.com/) |
| Git LFS | Any recent | [git-lfs.com](https://git-lfs.com/) — required for model files |
| `uv` package manager | Latest | See Step 3 below |

> **Windows users:** Use **PowerShell** or **Windows Terminal**. Do NOT use the old `cmd.exe`.

---

## ⚡ Quick Start (5 Steps)

Copy-paste this entire block:

```bash
# 1. Clone
git clone https://github.com/AyobBleblo/NLP-for-Olist-dataset.git
cd NLP-for-Olist-dataset

# 2. Pull ML model files via Git LFS
git lfs install
git lfs pull

# 3. Install uv  (Windows PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 4. Install dependencies
uv sync

# 5. Set up database + run server
uv run python manage.py migrate
uv run python manage.py seed_olist_catalog
uv run python manage.py runserver
```

**Backend API:** `http://127.0.0.1:8000/api/`

Then open a **second terminal** for the frontend:
```bash
python -m http.server 5173 --directory Frontend
```

**Frontend:** `http://localhost:5173`

---

## 🔧 Detailed Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/AyobBleblo/NLP-for-Olist-dataset.git
cd NLP-for-Olist-dataset
```

### Step 2 — Install Git LFS and Pull Model Files

```powershell
# Windows
winget install GitHub.GitLFS
```
```bash
# Mac
brew install git-lfs
# Linux
sudo apt install git-lfs
```

```bash
git lfs install
git lfs pull
```

After this you should have:
```
models/
  svd_cf.pkl                      ← ~30 MB  (Collaborative Filtering)
  processed/
    cbf_precomputed.npz           ← Content-Based Filtering scores
    tfidf_matrix.npz              ← TF-IDF sparse matrix
    products_features.parquet     ← Product feature vectors
    tfidf_row_index.parquet       ← Row-index mapping
```

### Step 3 — Install `uv`

```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> **Alternative:** If you prefer pip/conda, activate your own virtual environment and run `pip install -r requirements.txt`. The packages are the same.

### Step 4 — Install Dependencies

```bash
uv sync
```

Key packages installed: `django 6.1`, `djangorestframework`, `django-cors-headers`, `scikit-surprise`, `torch`, `transformers`, `scikit-learn`, `pandas`, `ftfy`.

> ⚠️ **PyTorch note:** `pyproject.toml` pulls CUDA 12.4 wheels. If you only have a CPU, it still works — offline sentiment scoring will be slower, but the live API never uses the GPU.

### Step 5 — Set Up the Database

Run migrations first, then pick **one** seeding option:

#### Option A — Full real Olist catalog ⭐ Recommended

```bash
uv run python manage.py migrate
uv run python manage.py seed_olist_catalog
```

Creates: 30 categories · 100 products · 5 real customers · real reviews with DistilBERT scores · hybrid recommendations precomputed for all customers.

#### Option B — Minimal 3-customer test (fastest)

```bash
uv run python manage.py migrate
uv run python manage.py seed_test_scenario
```

Creates Alice, Bruno, and Carlos with purchase histories and recommendations. Good for quick API testing.

#### Option C — Import from raw Kaggle CSV files

```bash
uv run python manage.py migrate
uv run python manage.py import_data              # products from CSVs
uv run python manage.py import_reviews           # reviews
uv run python manage.py compute_recommendations  # hybrid scoring
```

Place the Olist CSV files in the `data/` folder. Download from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

### Step 6 — Run the Backend Server

```bash
uv run python manage.py runserver
```

API is live at `http://127.0.0.1:8000/api/`. Verify:

```bash
curl http://127.0.0.1:8000/api/
```

---

## 🌐 Running the Frontend

Open a **second terminal** (keep Django running in the first):

```bash
python -m http.server 5173 --directory Frontend
```

Open `http://localhost:5173` in your browser.

### Pages

| URL | Page |
|---|---|
| `http://localhost:5173/` | 🏠 Home — catalog, personas, sentiment reviews |
| `http://localhost:5173/catalog.html` | 📦 Catalog — search, filter, sort |
| `http://localhost:5173/recommendations.html` | 🎯 Personalized shelf with SVD/CBF/NLP score breakdown |
| `http://localhost:5173/nlp-lab.html` | 🧪 NLP Lab — live sentiment tester + 20-case benchmark |
| `http://localhost:5173/cart.html` | 🛒 Cart — add items, checkout |
| `http://localhost:5173/analytics.html` | 📊 Model performance charts |
| `http://localhost:5173/orders.html` | 📋 Order history |
| `http://localhost:5173/product.html?id=<id>` | 🔍 Product detail |

> CORS is already enabled in `settings.py` — no extra config needed.

---

## 🛠️ Management Commands Reference

| Command | What it does | When to use |
|---|---|---|
| `migrate` | Creates all DB tables | Always run first |
| `seed_olist_catalog` | 100 products, 5 customers, reviews, recommendations | ⭐ Recommended |
| `seed_test_scenario` | 3 test customers (Alice / Bruno / Carlos) | Quick API testing |
| `seed_clean_store` | Alternative catalog seeding | Optional alternative |
| `import_data` | Imports products from Olist CSV files | Only if you have raw CSVs |
| `import_reviews` | Imports reviews from translated CSV | After `import_data` |
| `compute_recommendations` | Recomputes recommendations for all customers | After `import_data` |
| `run_sentiment_batch` | DistilBERT sentiment scoring on all reviews | Needs `models/best_model_binary/` |
| `diagnose_recommendations` | Debug info for recommendation scores | Debugging only |

---

## 🔌 API Endpoints Reference

**Base URL:** `http://127.0.0.1:8000/api/`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/` | API root |
| GET | `/api/categories/` | List all categories |
| GET | `/api/categories/<id>/` | Category detail |
| GET | `/api/products/` | List products (20 per page) |
| GET | `/api/products/?search=furniture` | Search by name |
| GET | `/api/products/?category=furniture_decor` | Filter by category |
| GET | `/api/products/?ordering=-total_purchases` | Sort by popularity |
| GET | `/api/products/<id>/` | Product detail |
| GET | `/api/recommendations/<customer_id>/` | Top-10 recommendations |
| GET | `/api/recommendations/<customer_id>/?n=5` | Top-N recommendations |
| GET | `/api/cart/<customer_id>/` | View cart |
| POST | `/api/cart/<customer_id>/add/` | Add item to cart |
| PATCH | `/api/cart/<customer_id>/update/<item_id>/` | Update item quantity |
| DELETE | `/api/cart/<customer_id>/remove/<item_id>/` | Remove item |
| POST | `/api/cart/<customer_id>/checkout/` | Checkout + refresh recommendations |

### Example curl Calls

```bash
# Products list
curl http://127.0.0.1:8000/api/products/

# Search
curl "http://127.0.0.1:8000/api/products/?search=sports"

# Recommendations (top 5)
curl "http://127.0.0.1:8000/api/recommendations/c37cc6c1a59d81460a3059744f7ada1c/?n=5"

# Add to cart
curl -X POST http://127.0.0.1:8000/api/cart/c37cc6c1a59d81460a3059744f7ada1c/add/ \
     -H "Content-Type: application/json" \
     -d '{"product_id": 1, "quantity": 2}'

# Checkout
curl -X POST http://127.0.0.1:8000/api/cart/c37cc6c1a59d81460a3059744f7ada1c/checkout/
```

---

## 👤 Test Customer IDs

### After `seed_olist_catalog` (5 real Olist customers)

| # | external_id | Preference |
|---|---|---|
| 1 | `06b8999e2fba1a1fbc88172c00ba8bc7` | housewares |
| 2 | `18955e83d337fd6b2def6b18a428ac77` | computers_accessories |
| 3 | `4e7b3452b3e31ade6d16c97f64a9e8f2` | sports_leisure |
| 4 | `7a5a6efc09ef6f57c84c7a63bc58fede` | furniture_decor |
| 5 | `8d50f5eadf50201ccdcedfb9e2ac8455` | health_beauty |

### After `seed_test_scenario` (3 simple personas)

| Customer | external_id | Preference |
|---|---|---|
| Alice Silva | `c37cc6c1a59d81460a3059744f7ada1c` | furniture_decor |
| Bruno Santos | `3e2157f91502458bc58455fd798ed58a` | sports_leisure |
| Carlos Costa | `feb2a9889d236875c3510880bf9576f3` | computers_accessories |

---

## 📁 Project Structure

```
NLP-for-Olist-dataset/
│
├── manage.py
├── pyproject.toml                    # Python dependencies
├── uv.lock
│
├── recsys_backend/                   # Django project config
│   ├── settings.py                   # Settings + ML model paths
│   ├── urls.py                       # Root URL router
│   └── wsgi.py
│
├── store/                            # Main Django app
│   ├── models.py                     # 9 database models
│   ├── views.py                      # DRF API views (zero ML inference)
│   ├── serializers.py
│   ├── urls.py                       # 11 API routes
│   ├── ml/
│   │   └── loader.py                 # Singleton loaders (SVD, CBF, DistilBERT)
│   └── management/commands/
│       ├── seed_olist_catalog.py     # ⭐ Main seed command
│       ├── seed_test_scenario.py
│       ├── import_data.py
│       ├── import_reviews.py
│       ├── compute_recommendations.py
│       ├── run_sentiment_batch.py
│       └── diagnose_recommendations.py
│
├── models/
│   ├── svd_cf.pkl                    # ✅ Git LFS (~30 MB)
│   ├── best_model_binary/            # ⚠️ Google Drive (~255 MB)
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   └── processed/                    # ✅ Git LFS (~3 MB)
│       ├── cbf_precomputed.npz
│       ├── tfidf_matrix.npz
│       ├── products_features.parquet
│       └── tfidf_row_index.parquet
│
├── Frontend/
│   ├── index.html
│   ├── catalog.html
│   ├── recommendations.html
│   ├── nlp-lab.html
│   ├── cart.html
│   ├── product.html
│   ├── analytics.html
│   ├── orders.html
│   ├── css/style.css
│   └── js/
│       ├── api.js
│       ├── config.js
│       ├── ui.js
│       └── pages/
│
├── inference.py                      # Standalone DistilBERT inference
├── train_model.py                    # Model training script
├── precompute_cbf.py
├── prepare_data.py
│
├── README.md                         # This file
├── SETUP.md                          # Frontend developer quick-start
├── BACKEND_ARCHITECTURE_AND_PROCESS.md
└── TEAM_BACKEND_GUIDE.md
```

---

## 🧠 Optional: DistilBERT Sentiment Model Setup

The fine-tuned DistilBERT checkpoint (~255 MB) is not on GitHub.
You only need it to run sentiment analysis on new reviews.
Everything else works without it.

**Download from Google Drive:** *(ask the team for the link)*

Place the files at:
```
models/best_model_binary/
  config.json
  model.safetensors       ← 255 MB
  tokenizer.json
  tokenizer_config.json
```

Then run:
```bash
uv run python manage.py run_sentiment_batch
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---|---|
| `uv: command not found` | Install uv — see Step 3 |
| `git lfs: command not found` | Install Git LFS from git-lfs.com |
| `models/svd_cf.pkl not found` | Run `git lfs pull` after cloning |
| `ModuleNotFoundError` | Run `uv sync` |
| Frontend shows "Cannot connect to API" | Make sure Django is running on port 8000 |
| Port 8000 in use | Run `uv run python manage.py runserver 8001` |
| `DLL load failed` on Windows | Use `uv run python` not plain `python` |
| No recommendations shown | Run `uv run python manage.py seed_olist_catalog` |
| Sentiment score is 0.0 everywhere | Normal — download `best_model_binary/` from Drive |
| `No module named surprise` | Run `uv sync` — scikit-surprise is in pyproject.toml |

---

## 📚 Further Documentation

| File | Contents |
|---|---|
| [`BACKEND_ARCHITECTURE_AND_PROCESS.md`](./BACKEND_ARCHITECTURE_AND_PROCESS.md) | Full architecture — DB schema, API flows, ML pipeline |
| [`TEAM_BACKEND_GUIDE.md`](./TEAM_BACKEND_GUIDE.md) | Backend team reference |
| [`SETUP.md`](./SETUP.md) | Frontend developer quick-start |
| [`Frontend/README.md`](./Frontend/README.md) | Frontend architecture details |
| [`mdFiles/`](./mdFiles/) | Additional NLP methodology docs |

---

*Samsung Innovation Campus — Smart Recommendation System for E-Commerce*
*Built on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).*
