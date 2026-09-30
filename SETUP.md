# 🚀 Frontend Developer Setup Guide

> **For:** Frontend / Full-Stack developer integrating with the Olist Recommendation API  
> **Backend Stack:** Python 3.13 · Django 6.1 · Django REST Framework  
> **API Base URL (local):** `http://127.0.0.1:8000/api/`

---

## ✅ Quick Answer: What Do You Need?

| What | Where | Size |
|---|---|---|
| Project source code | This GitHub repo | — |
| Python 3.12+ | [python.org](https://python.org) | — |
| `uv` package manager | Install command below | — |
| `svd_cf.pkl` (CF model) | **Git LFS** (pulled automatically) | ~30 MB |
| `models/processed/` (CBF artifacts) | **Git LFS** (pulled automatically) | ~3 MB |
| `best_model_binary/` (Sentiment NLP) | **Google Drive link** (optional) | 255 MB |

> **You don't need the sentiment model to run the API.** Recommendations work without it — sentiment scores just default to 0. Get the CF + CBF models first (they come via Git LFS automatically).

---

## 📥 Step 1 — Clone the Repository

```bash
git clone https://github.com/AyobBleblo/NLP-for-Olist-dataset.git
cd NLP-for-Olist-dataset
```

> **Git LFS is required** to pull the model files. Install it first if you don't have it:
> - Windows: `winget install GitHub.GitLFS` or download from [git-lfs.com](https://git-lfs.com)
> - Mac: `brew install git-lfs`
> - Linux: `sudo apt install git-lfs`
>
> Then run once: `git lfs install`

After cloning, pull LFS files:
```bash
git lfs pull
```

You should now have:
```
models/
  svd_cf.pkl                        ← Collaborative Filtering (SVD)
  processed/
    cbf_precomputed.npz             ← Content-Based Filtering
    tfidf_matrix.npz
    products_features.parquet
    tfidf_row_index.parquet
```

---

## 🐍 Step 2 — Install Python & uv

### Option A — uv (Recommended, fastest)
```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Option B — Conda / Anaconda (if already installed)
Works too — just use `pip install -r requirements.txt` instead of `uv sync`.

---

## 📦 Step 3 — Install Dependencies

```bash
uv sync
```

This creates a `.venv/` and installs all packages from `pyproject.toml` automatically.

---

## 🗄️ Step 4 — Set Up the Database

The database is **not** committed to git (it's large). Run these commands in order to create and populate it:

```bash
# 1. Create all database tables
uv run python manage.py migrate

# 2. Import all 1,000 products, categories, and customers from Olist CSVs
#    NOTE: The /data/ folder with CSV files must be present.
#    If you don't have /data/, skip to Step 4b below.
uv run python manage.py import_data

# 3. Compute recommendations using SVD + CBF models
uv run python manage.py compute_recommendations
```

### Step 4b — No CSV data? Use the test scenario instead:
```bash
# Creates 3 test customers with purchase history + computes recommendations
uv run python manage.py seed_test_scenario
```

This gives you **3 working customers** with known IDs to test the API immediately.

---

## ▶️ Step 5 — Run the Server

```bash
uv run python manage.py runserver
```

The API is now live at `http://127.0.0.1:8000/api/`

---

## 🌐 API Endpoints Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/products/` | GET | List all products (paginated, 20/page) |
| `/api/products/?search=furniture` | GET | Search products by name |
| `/api/products/?category=furniture_decor` | GET | Filter by category slug |
| `/api/categories/` | GET | List all product categories |
| `/api/recommendations/<customer_id>/` | GET | Get top-5 recommendations |
| `/api/recommendations/<customer_id>/?n=10` | GET | Get top-N recommendations |
| `/api/cart/` | GET/POST | View or add to cart |
| `/api/cart/<id>/` | DELETE | Remove item from cart |
| `/api/checkout/` | POST | Place an order |
| `/api/orders/` | GET | List customer orders |
| `/api/customers/` | GET | List customers |

### Pagination
All list endpoints return paginated results:
```json
{
  "count": 1000,
  "next": "http://127.0.0.1:8000/api/products/?page=2",
  "previous": null,
  "results": [...]
}
```

---

## 🧪 Test Customer IDs (from seed_test_scenario)

| Customer | Persona | external_id | Category preference |
|---|---|---|---|
| Alice Silva | Home Decor fan | `c37cc6c1a59d81460a3059744f7ada1c` | furniture_decor |
| Bruno Santos | Sports enthusiast | `3e2157f91502458bc58455fd798ed58a` | sports_leisure |
| Carlos Costa | Tech geek | `feb2a9889d236875c3510880bf9576f3` | computers_accessories |

### Example API calls:

```bash
# Get Alice's top-5 recommendations
curl http://127.0.0.1:8000/api/recommendations/c37cc6c1a59d81460a3059744f7ada1c/?n=5

# Browse products
curl http://127.0.0.1:8000/api/products/?page=1

# Search products
curl "http://127.0.0.1:8000/api/products/?search=sports"

# Get all categories
curl http://127.0.0.1:8000/api/categories/
```

---

## 📊 Recommendation Response Format

```json
GET /api/recommendations/c37cc6c1a59d81460a3059744f7ada1c/?n=5

{
  "customer_id": "c37cc6c1a59d81460a3059744f7ada1c",
  "recommendations": [
    {
      "rank": 1,
      "product": {
        "id": "...",
        "external_id": "c50663...",
        "name": "Original Furniture Decor #C50663",
        "price": 73.35,
        "category": "furniture_decor",
        "avg_sentiment_score": 0.82,
        "total_purchases": 14
      },
      "score": 2.2574
    },
    ...
  ]
}
```

---

## 🧠 (Optional) Sentiment Model Setup

The `best_model_binary/` folder (255 MB DistilBERT) is NOT in Git due to its size.  
You only need it if you want to **run sentiment analysis on new reviews**.  

**Download from Google Drive:** *(link to be added by the Data Science team)*

Place it at:
```
models/
  best_model_binary/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
```

To run sentiment analysis on product reviews:
```bash
uv run python manage.py run_sentiment_batch
```

---

## ⚡ Full Setup (one screen — copy paste)

```powershell
# 1. Clone
git clone https://github.com/AyobBleblo/NLP-for-Olist-dataset.git
cd NLP-for-Olist-dataset

# 2. Pull LFS model files
git lfs install
git lfs pull

# 3. Install dependencies
uv sync

# 4. Set up database with test data
uv run python manage.py migrate
uv run python manage.py seed_test_scenario

# 5. Run server
uv run python manage.py runserver
```

Then open: **http://127.0.0.1:8000/api/**

---

## ❓ Troubleshooting

| Problem | Fix |
|---|---|
| `uv: command not found` | Install uv — see Step 2 above |
| `git lfs: command not found` | Install Git LFS from [git-lfs.com](https://git-lfs.com) |
| `models/svd_cf.pkl not found` | Run `git lfs pull` after cloning |
| `No recommendations for customer` | Run `uv run python manage.py seed_test_scenario` first |
| `DLL load failed` error | This is a Windows App Control issue; use `uv run` not `python` |
| Port 8000 already in use | Run `uv run python manage.py runserver 8001` |

---

## 📁 Project Structure

```
NLP-for-Olist-dataset/
├── manage.py
├── recsys_backend/          # Django project config
│   ├── settings.py          # All settings (API keys, model paths)
│   └── urls.py              # Root URL router
├── store/                   # Main Django app
│   ├── models.py            # Database models (Product, Customer, Order, etc.)
│   ├── views.py             # API views
│   ├── serializers.py       # DRF serializers
│   ├── urls.py              # API endpoint definitions
│   └── ml/
│       └── loader.py        # Model loading (SVD + CBF singletons)
├── models/                  # ML artifacts (via Git LFS)
│   ├── svd_cf.pkl           # Collaborative Filtering SVD model
│   └── processed/
│       ├── cbf_precomputed.npz   # Content-Based Filtering (numpy)
│       └── ...
├── TEAM_BACKEND_GUIDE.md    # Full technical documentation
└── SETUP.md                 # This file
```

---

*For architecture details, algorithm explanation, and full API documentation see [`TEAM_BACKEND_GUIDE.md`](./TEAM_BACKEND_GUIDE.md)*
