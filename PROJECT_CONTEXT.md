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

## 🏛️ System Architecture

```
                          OFFLINE BATCH PIPELINE
┌────────────────────────┬────────────────────────┬────────────────────────┬────────────────────────┐
│  1. Data Ingestion     │  2. Translation        │  3. Sentiment Analysis │  4. Recommendation     │
│  import_data           │  translate_products    │  run_sentiment_batch   │  compute_recommendations
│  (1,000 Products &     │  (NLLB-200 model       │  (DistilBERT binary    │  (SVD Matrix           │
│   58 Categories)       │   PT -> EN)            │   model scoring)       │   Factorization)       │
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
      │ /products/             │ /add/, /update/, etc.  │ /checkout/             │ <cust_id>/?n=5         │
      └────────────────────────┴────────────────────────┴────────────────────────┴────────────────────────┘
```

---

## 🎯 The 3-Customer Collaborative Filtering Test Scenario

The system includes a fully verified, seeded 3-customer scenario demonstrated via:
```powershell
uv run python manage.py seed_test_scenario
```

### Customer 1: Alice Silva (Home Decor & Furniture Fan)
- **Customer ID**: `c37cc6c1a59d81460a3059744f7ada1c`
- **Location**: São Paulo, SP
- **Persona**: Frequent buyer of furniture, lighting, and interior decoration.
- **Purchase History (4 Items)**:
  1. *Smart Furniture Decor #A5341E* ($88.94)
  2. *Essential Furniture Decor #A237DE* ($38.67)
  3. *Pro Furniture Decor #C2ECE6* ($83.08)
  4. *Classic Furniture Decor #64D0FE* ($89.18)
- **Top SVD Recommendations**:
  1. **Pro Pet Shop #A4AA7C** ($101.88) — Predicted Score: **4.545** *(pet_shop)*
  2. **Premium Toys #6F33A4** ($140.99) — Predicted Score: **4.422** *(toys)*
  3. **Modern Pet Shop #10876F** ($47.23) — Predicted Score: **4.351** *(pet_shop)*
- **Live Endpoint**: [http://127.0.0.1:8000/api/recommendations/c37cc6c1a59d81460a3059744f7ada1c/?n=5](http://127.0.0.1:8000/api/recommendations/c37cc6c1a59d81460a3059744f7ada1c/?n=5)

---

### Customer 2: Bruno Santos (Sports & Outdoor Enthusiast)
- **Customer ID**: `3e2157f91502458bc58455fd798ed58a`
- **Location**: Rio de Janeiro, RJ
- **Persona**: Active outdoor, fitness, and sporting goods consumer.
- **Purchase History (4 Items)**:
  1. *Urban Sports Leisure #583F15* ($45.33)
  2. *Signature Sports Leisure #81288D* ($110.54)
  3. *Deluxe Sports Leisure #219666* ($65.17)
  4. *Smart Sports Leisure #4B5E26* ($213.89)
- **Top SVD Recommendations**:
  1. **Pro Pet Shop #A4AA7C** ($101.88) — Predicted Score: **4.661** *(pet_shop)*
  2. **Essential Furniture Decor #A237DE** ($38.67) — Predicted Score: **4.596** *(furniture_decor)*
  3. **Smart Sports Leisure #B9EE75** ($69.90) — Predicted Score: **4.492** *(sports_leisure)*
- **Live Endpoint**: [http://127.0.0.1:8000/api/recommendations/3e2157f91502458bc58455fd798ed58a/?n=5](http://127.0.0.1:8000/api/recommendations/3e2157f91502458bc58455fd798ed58a/?n=5)

---

### Customer 3: Carlos Costa (Computers & Technology Geek)
- **Customer ID**: `feb2a9889d236875c3510880bf9576f3`
- **Location**: Curitiba, PR
- **Persona**: PC builder, tech accessories, and electronics shopper.
- **Purchase History (4 Items)**:
  1. *Premium Computers Accessories #FB6782* ($79.99)
  2. *Signature Computers Accessories #C7E027* ($112.82)
  3. *Essential Computers Accessories #7AA1AB* ($235.12)
  4. *Essential Computers Accessories #0CF3AB* ($88.50)
- **Top SVD Recommendations**:
  1. **Essential Furniture Decor #A237DE** ($38.67) — Predicted Score: **4.725** *(furniture_decor)*
  2. **Pro Pet Shop #A4AA7C** ($101.88) — Predicted Score: **4.663** *(pet_shop)*
  3. **Signature Furniture Decor #A9E189** ($49.50) — Predicted Score: **4.445** *(furniture_decor)*
- **Live Endpoint**: [http://127.0.0.1:8000/api/recommendations/feb2a9889d236875c3510880bf9576f3/?n=5](http://127.0.0.1:8000/api/recommendations/feb2a9889d236875c3510880bf9576f3/?n=5)

---

## 📡 REST API Reference (All 11 Endpoints)

Base URL: `http://127.0.0.1:8000/api/`

| Category | Method | Endpoint | Description |
|---|---|---|---|
| **Root** | `GET` | `/api/` | Interactive entrypoint with resource links |
| **Catalog** | `GET` | `/api/categories/` | List all 58 categories (with translations & product counts) |
| | `GET` | `/api/categories/<id>/` | Single category detail |
| | `GET` | `/api/products/` | Paginated product list (`?category=`, `?search=`, `?ordering=`) |
| | `GET` | `/api/products/<lookup>/` | Product detail by numeric ID or Olist external hash |
| **Cart** | `GET` | `/api/cart/<customer_id>/` | View cart items and calculated order total |
| | `POST` | `/api/cart/<customer_id>/add/` | Add item or increment quantity |
| | `PATCH` | `/api/cart/<customer_id>/update/<item_id>/` | Set item quantity |
| | `DELETE` | `/api/cart/<customer_id>/remove/<item_id>/` | Remove item from cart |
| | `POST` | `/api/cart/<customer_id>/checkout/` | Checkout cart $\rightarrow$ creates `Order`, empties cart |
| **Recommendations** | `GET` | `/api/recommendations/<customer_id>/` | Retrieve Top-N precomputed recommendations (`?n=5`) |

---

## 📦 Machine Learning Models Summary

| Model | Path | Technique / Architecture | Purpose |
|---|---|---|---|
| **Translation** | HuggingFace Hub | `facebook/nllb-200-distilled-600M` | Translate Portuguese reviews and categories to English |
| **Sentiment Analysis** | `models/best_model_binary/` | DistilBERT Binary Classifier | Classify reviews into Negative (0) and Positive (1) |
| **Collaborative Filtering** | `models/svd_cf.pkl` | SVD Matrix Factorization (`scikit-surprise`) | Predict personal user ratings $\hat{r}_{u,i} = \mu + b_u + b_i + q_i^T p_u$ |

---

## 💻 Master Pipeline Execution Order

```powershell
# 1. Start the live development server
uv run python manage.py runserver

# 2. Import products and categories (selected by photo quality)
uv run python manage.py import_data --limit 1000

# 3. Translate product and category names using NLLB-200
uv run python manage.py translate_products

# 4. Run sentiment analysis batch on reviews using DistilBERT
uv run python manage.py run_sentiment_batch

# 5. Compute SVD recommendations for all customers
uv run python manage.py compute_recommendations --top-n 10

# 6. Run the complete 3-customer verification scenario
uv run python manage.py seed_test_scenario
```
