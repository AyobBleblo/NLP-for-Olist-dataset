# Project Context: E-Commerce Recommendation & NLP Sentiment Platform

This document serves as the master source-of-truth context for the entire project, spanning dataset engineering, NLP translation, sentiment classification, Collaborative Filtering recommendation models, and the Django REST backend.

---

## 🌟 Executive Summary

This platform is an AI-powered e-commerce backend built on the **Olist Brazilian E-Commerce Dataset**. It combines:
1. **NLP Review Translation**: Portuguese $\rightarrow$ English translation using `facebook/nllb-200-distilled-600M`.
2. **NLP Review Sentiment Analysis**: Fine-tuned binary DistilBERT classifier evaluating customer reviews (Class 0: Negative, Class 1: Positive).
3. **Collaborative Filtering Engine**: Matrix Factorization using Singular Value Decomposition (SVD via `scikit-surprise`) trained on user-item interaction matrices.
4. **Hybrid Sentiment Boost**: Blends SVD predicted ratings with NLP review sentiment scores.
5. **Production Django REST Backend**: 11 clean endpoints with sub-10ms response times (strict *Zero ML in request path* architecture).

---

## 🌟 Executive Summary

This platform is an AI-powered e-commerce backend built on the **Olist Brazilian E-Commerce Dataset**. It combines:
1. **NLP Review Translation**: Portuguese $\rightarrow$ English translation using `facebook/nllb-200-distilled-600M` & LLM pipeline.
2. **NLP Review Sentiment Analysis**: Fine-tuned binary DistilBERT classifier evaluating customer reviews (Class 0: Negative, Class 1: Positive).
3. **Collaborative Filtering Engine**: Matrix Factorization using Singular Value Decomposition (SVD via `scikit-surprise`) trained on user-item interaction matrices.
4. **Content-Based Filtering Engine**: TF-IDF text similarity and normalized physical/pricing feature vectors.
5. **True Hybrid Recommendation**: Blends SVD predicted ratings with CBF profile similarity and DistilBERT review sentiment scores, with per-category diversity caps (max 2 per category).
6. **Production Django REST Backend**: 11 clean endpoints with sub-10ms response times (strict *Zero ML in request path* architecture).

---

## 🚀 Catalog Status: Real Olist Catalog (Completed)

The store runs on the **original Olist catalog** using original 32-character hexadecimal IDs:
- **30 Balanced Categories**: High-demand, visually clear categories with certified English translations.
- **100 Flagship Products**: Selected based on presence in **both** SVD trainset and CBF index, high purchase count, and review volume.
- **100% Verified Real Photography**: High-resolution Unsplash product photography matched to each product's title and category (HTTP 200 and image/* verified).
- **5 Real SVD Customers**: Real multi-purchase Olist customers with 50-dimensional latent user vectors ($p_u$) in `models/svd_cf.pkl`.
- **1,185 Real Reviews**: Real customer reviews imported and scored with binary sentiment.
- **Zero Fallback Needed**: SVD `to_inner_iid` and `to_inner_uid` both succeed; CBF `pid_to_idx` succeeds; recommendations are 100% true hybrid.
- **Dynamic Live Checkout Refresh**: When a user purchases an item, checkout immediately updates their recommendations using the true hybrid engine.

Master seeding command:
```powershell
uv run python manage.py seed_olist_catalog
```

---

## 🏛️ System Architecture

```
                          OFFLINE BATCH & SEED PIPELINE
┌────────────────────────┬────────────────────────┬────────────────────────┬────────────────────────┐
│  1. Catalog Curation   │  2. Review Import      │  3. Sentiment Analysis │  4. Hybrid RecSys      │
│  seed_olist_catalog    │  (1,185 real bilingual │  (DistilBERT binary    │  (SVD CF + TF-IDF CBF  │
│  (100 Products,        │   reviews for catalog  │   model scoring ->     │   + DistilBERT NLP +   │
│   30 Categories,       │   items)               │   avg_sentiment_score) │   Diversity Cap)       │
│   5 SVD Customers)     │                        │                        │                        │
└───────────┬────────────┴───────────┬────────────┴───────────┬────────────┴───────────┬────────────┘
            │                        │                        │                        │
            ▼                        ▼                        ▼                        ▼
     [Products & Cats]        [English Titles]      [avg_sentiment_score]    [Recommendation Rows]
            │                        │                        │                        │
            └────────────────────────┴────────────────────────┴────────────────────────┘
                                                 │
                                                 ▼
                                     SQLite Database (db.sqlite3)
                                                 ▲
                                                 │ Fast read queries (<10ms)
                                     DJANGO REST API (/api/)
      ┌────────────────────────┬────────────────────────┬────────────────────────┬────────────────────────┐
      │ Catalog API            │ Cart API               │ Checkout API           │ Recommendation API     │
      │ /categories/           │ /cart/<cust_id>/       │ /cart/<cust_id>/       │ /recommendations/      │
      │ /products/             │ /add/, /update/, etc.  │ /checkout/ (refreshes) │ <cust_id>/?n=5         │
      └────────────────────────┴────────────────────────┴────────────────────────┴────────────────────────┘
```

---

## 🎯 The 5 Real Customer Personas (True SVD + CBF + NLP)

All 5 personas are real Olist buyers from the SVD trainset (`models/svd_cf.pkl`), each possessing 50-dimensional latent preference vectors $p_u$ and real historical orders in the catalog:

### 1. Alice Silva (Computers & Technology Geek)
- **Customer Unique ID**: `1dfbdc636de09adbcdbc3d34e15084f3`
- **Location**: Brasília, DF
- **Persona**: Tech enthusiast with multi-item purchase history in `computers_accessories`.
- **Top Hybrid Recommendations**:
  1. *Dell UltraSharp 27" 4K USB-C Hub Monitor* ($108.50) — Score: **2.405** *(computers_accessories)*
  2. *Logitech MX Master 3S Advanced Ergonomic Wireless Mouse* ($48.33) — Score: **2.373** *(computers_accessories)*
  3. *BAGSMART 6-Set Travel Compression Packing Cubes* ($168.00) — Score: **2.247** *(luggage_accessories)*
- **Live Endpoint**: [http://127.0.0.1:8000/api/recommendations/1dfbdc636de09adbcdbc3d34e15084f3/?n=5](http://127.0.0.1:8000/api/recommendations/1dfbdc636de09adbcdbc3d34e15084f3/?n=5)

### 2. Bruno Santos (Sports & Fitness Enthusiast)
- **Customer Unique ID**: `2e43e031f10de28e557c35ef668f9396`
- **Location**: Canoas, RS
- **Persona**: Active outdoor, running, and sporting goods buyer (`sports_leisure`).
- **Top Hybrid Recommendations**:
  1. *Hydro Flask 32 oz Wide Mouth Vacuum Insulated Bottle* ($23.85) — Score: **2.397** *(sports_leisure)*
  2. *Bowflex SelectTech Adjustable Dumbbells Pair* ($114.65) — Score: **2.248** *(sports_leisure)*
  3. *Ippodo Tea Japanese Uji Ceremonial Grade Matcha Powder* ($29.90) — Score: **2.206** *(drinks)*
- **Live Endpoint**: [http://127.0.0.1:8000/api/recommendations/2e43e031f10de28e557c35ef668f9396/?n=5](http://127.0.0.1:8000/api/recommendations/2e43e031f10de28e557c35ef668f9396/?n=5)

### 3. Camila Oliveira (Home Decor & Design Enthusiast)
- **Customer Unique ID**: `33de26d1fafbfd4945eb586f7136efe6`
- **Location**: Montes Claros, MG
- **Persona**: Frequent buyer of interior furniture, wall art, and decoration (`furniture_decor`).
- **Top Hybrid Recommendations**:
  1. *Rove Concepts Velvet Tufted Accent Round Pouf Ottoman* ($53.48) — Score: **2.227** *(furniture_living_room)*
  2. *Our Place Ceramic Stoneware 16-Piece Dinnerware Set* ($71.36) — Score: **2.172** *(housewares)*
  3. *Arhaus Modern Arched Brass Wall Hanging Mirror* ($71.36) — Score: **2.163** *(furniture_decor)*
- **Live Endpoint**: [http://127.0.0.1:8000/api/recommendations/33de26d1fafbfd4945eb586f7136efe6/?n=5](http://127.0.0.1:8000/api/recommendations/33de26d1fafbfd4945eb586f7136efe6/?n=5)

### 4. Elena Souza (Beauty, Style & Wellness)
- **Customer Unique ID**: `1b6c7548a2a1f9037c1fd3ddfed95f33`
- **Location**: Ituiutaba, MG
- **Persona**: Dedicated buyer of skincare, cosmetics, and fine fragrance (`health_beauty`, `perfumery`).
- **Top Hybrid Recommendations**:
  1. *Our Place Ceramic Stoneware 16-Piece Dinnerware Set* ($71.36) — Score: **2.259** *(housewares)*
  2. *Dior Miss Dior Blooming Bouquet Eau de Toilette* ($33.90) — Score: **2.209** *(perfumery)*
  3. *Diptyque Baies Luxury Scented Reed Room Diffuser* ($132.88) — Score: **2.198** *(perfumery)*
- **Live Endpoint**: [http://127.0.0.1:8000/api/recommendations/1b6c7548a2a1f9037c1fd3ddfed95f33/?n=5](http://127.0.0.1:8000/api/recommendations/1b6c7548a2a1f9037c1fd3ddfed95f33/?n=5)

### 5. Diego Ferreira (Automotive & Hardware Gearhead)
- **Customer Unique ID**: `fe86d9409d83a3c561ce16e64d2d55e6`
- **Location**: Jaú, SP
- **Persona**: Auto enthusiast and DIY mechanic (`auto`, `garden_tools`).
- **Top Hybrid Recommendations**:
  1. *iOttie Easy One Touch 5 Universal Dashboard Car Phone Mount* ($49.90) — Score: **2.338** *(auto)*
  2. *Mobil 1 Advanced Full Synthetic Motor Oil 5W-30* ($104.97) — Score: **2.333** *(auto)*
  3. *Dior Miss Dior Blooming Bouquet Eau de Toilette* ($33.90) — Score: **2.222** *(perfumery)*
- **Live Endpoint**: [http://127.0.0.1:8000/api/recommendations/fe86d9409d83a3c561ce16e64d2d55e6/?n=5](http://127.0.0.1:8000/api/recommendations/fe86d9409d83a3c561ce16e64d2d55e6/?n=5)

---

## 📡 REST API Reference (All 11 Endpoints)

Base URL: `http://127.0.0.1:8000/api/`

| Category | Method | Endpoint | Description |
|---|---|---|---|
| **Root** | `GET` | `/api/` | Interactive entrypoint with resource links |
| **Catalog** | `GET` | `/api/categories/` | List all 30 categories (with translations & product counts) |
| | `GET` | `/api/categories/<id>/` | Single category detail |
| | `GET` | `/api/products/` | Paginated product list (`?category=`, `?search=`, `?ordering=`) |
| | `GET` | `/api/products/<lookup>/` | Product detail by numeric ID or Olist external hash |
| **Cart** | `GET` | `/api/cart/<customer_id>/` | View cart items and calculated order total |
| | `POST` | `/api/cart/<customer_id>/add/` | Add item or increment quantity |
| | `PATCH` | `/api/cart/<customer_id>/update/<item_id>/` | Set item quantity |
| | `DELETE` | `/api/cart/<customer_id>/remove/<item_id>/` | Remove item from cart |
| | `POST` | `/api/cart/<customer_id>/checkout/` | Checkout cart $\rightarrow$ creates `Order`, empties cart, refreshes hybrid recs |
| **Recommendations** | `GET` | `/api/recommendations/<customer_id>/` | Retrieve Top-N precomputed recommendations (`?n=5`) |

---

## 📦 Machine Learning Models Summary

| Model | Path | Technique / Architecture | Purpose |
|---|---|---|---|
| **Translation** | HuggingFace Hub | `facebook/nllb-200-distilled-600M` | Translate Portuguese reviews and categories to English |
| **Sentiment Analysis** | `models/best_model_binary/` | DistilBERT Binary Classifier | Classify reviews into Negative (0) and Positive (1) |
| **Collaborative Filtering** | `models/svd_cf.pkl` | SVD Matrix Factorization (`scikit-surprise`) | Predict personal user ratings $\hat{r}_{u,i} = \mu + b_u + b_i + q_i^T p_u$ |
| **Content-Based Filtering** | `models/processed/cbf_precomputed.npz` | TF-IDF (193-dim text) + 5-dim Normalized Numerical | Product feature similarity and user profile matching |

---

## 💻 Master Pipeline Execution Order

```powershell
# 1. Start the live development server
uv run python manage.py runserver

# 2. Rebuild and seed the complete Real Olist Catalog (100 products, 30 categories, 5 personas, reviews, hybrid recs)
uv run python manage.py seed_olist_catalog

# 3. (Optional) Run standalone offline batch jobs if desired
uv run python manage.py compute_recommendations --top-n 10
```
