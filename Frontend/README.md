# Olist Recommender — Frontend

A static, production-quality frontend built **specifically for** the backend in
[`Aymanbawa/NLP-for-Olist-dataset`](https://github.com/Aymanbawa/NLP-for-Olist-dataset).

The original repository (imported here under `backend/`, unmodified) is a Django 6.1 + DRF
backend for an AI-powered e-commerce platform on the **Brazilian Olist dataset**. Its offline
batch pipeline translates Portuguese reviews with NLLB-200, scores them with a fine-tuned
DistilBERT classifier, and blends SVD collaborative filtering with TF-IDF content-based
filtering and sentiment into a hybrid recommender. The API only reads precomputed rows —
**zero ML inference in the request path**.

This project adds the missing frontend: a premium, responsive storefront + analytics UI that
talks to the real endpoints.

---

## 1. Repository analysis (the source of truth)

Read directly from the code, not from filenames or the docs alone.

| Aspect | Finding | Evidence |
|---|---|---|
| Language / framework | Python 3.13, Django 6.1, Django REST Framework | `backend/pyproject.toml`, `backend/recsys_backend/settings.py` |
| Database | SQLite (`db.sqlite3`) | `settings.py` → `DATABASES` |
| Models | **9 entities** | `backend/store/models.py` |
| Endpoints | **11 routes** | `backend/store/urls.py` |
| Authentication | **None** | no `DEFAULT_AUTHENTICATION_CLASSES` / `DEFAULT_PERMISSION_CLASSES` in `settings.py`; no auth route in `urls.py` |
| Authorization / roles | **None** — every endpoint is anonymous | same |
| Pagination | DRF `PageNumberPagination`, `PAGE_SIZE = 20` | `settings.py` → `REST_FRAMEWORK` |
| Filtering | `SearchFilter` + `OrderingFilter` on products & categories | `store/views.py` |
| File uploads | **Not supported** — no `FileField`/`ImageField`; `image_url` is a generated picsum placeholder | `store/models.py`, `import_data.py::_make_image_url` |
| Payments | **Not implemented** — checkout creates an `Order` with `status="processing"` | `store/views.py::CartCheckoutView` |
| CORS | **Not configured** — no CORS middleware despite `ALLOWED_HOSTS = ["*"]` | `settings.py` → `MIDDLEWARE` |
| ML weights | Outside git (`models/svd_cf.pkl` via Git LFS ~30 MB, `best_model_binary/` 255 MB via Drive) | `backend/SETUP.md` |
| Frontend in repo | **None** — no templates, static assets or SPA | full tree of the repo |

### The 9 data models

`Category` · `Customer` · `Product` · `Order` · `OrderItem` · `Review` · `Cart` · `CartItem` · `Recommendation`

Key relationships: `Product → Category` (FK, `SET_NULL`); `Order → Customer` (FK);
`OrderItem → Order, Product`; `Cart → Customer` (**OneToOne**); `CartItem → Cart, Product`
(`unique_together`); `Recommendation → Customer, Product` (`unique_together`, indexed on
`(customer, -score)`).

### The 11 endpoints (transcribed from `store/urls.py`)

| Group | Method | Path |
|---|---|---|
| Root | GET | `/api/` |
| Catalog | GET | `/api/categories/` |
| Catalog | GET | `/api/categories/<int:pk>/` |
| Catalog | GET | `/api/products/` |
| Catalog | GET | `/api/products/<str:lookup_value>/` |
| Cart | GET | `/api/cart/<customer_external_id>/` |
| Cart | POST | `/api/cart/<customer_external_id>/add/` |
| Cart | PATCH | `/api/cart/<customer_external_id>/update/<item_id>/` |
| Cart | DELETE | `/api/cart/<customer_external_id>/remove/<item_id>/` |
| Cart | POST | `/api/cart/<customer_external_id>/checkout/` |
| Recommendations | GET | `/api/recommendations/<customer_external_id>/?n=` |

### Documented but **not** implemented (surfaced in the UI, never faked)

* `GET /api/orders/` and `GET /api/customers/` — advertised in `SETUP.md`, absent from `urls.py`.
* Review routes — `Review` is a model with no endpoint.
* Any authentication, user roles or admin API.

---

## 2. Frontend plan derived from that analysis

Because the backend has no auth, there is **no login screen and no protected routes** — that
would be inventing functionality. Instead the frontend uses a **customer switcher** bound to
the three real ids created by `manage.py seed_test_scenario`.

| Backend capability | Frontend representation |
|---|---|
| `GET /api/` | API reference page (root resource card) |
| Categories list + detail | Catalogue category filter + category counts |
| Products list (search / category / ordering / page) | Catalogue page with real query params + pagination |
| Product detail (numeric id **or** external_id) | Product page with the exact response fields, dimensions, PSI gauge |
| Cart get / add / update / remove / checkout | Cart page with quantity controls, empty state, transactional checkout + order receipt modal |
| Recommendations `?n=` | Recommendation shelf with score decomposition bars |
| No orders endpoint | Orders page reads the store and flags the gap in a banner |
| Offline NLP pipeline | NLP Lab (interactive pipeline + 30-case benchmark) |
| Published evaluation metrics | Analytics dashboard (Chart.js) |

---

## 3. Pages

| Path | Purpose |
|---|---|
| `index.html` | Overview: backend capabilities, offline pipeline, top products, seeded personas |
| `catalog.html` | Catalogue with `search`, `category`, `ordering`, `page` + sentiment facet |
| `product.html?id=<numeric id \| external_id>` | Product detail, sentiment gauge, reviews (PT + EN), related items |
| `recommendations.html?customer=<id>` | Top-N shelf with SVD / CBF / sentiment breakdown and a recompute action |
| `cart.html` | The five cart endpoints, validation errors, checkout receipt |
| `orders.html?customer=<id>` | Orders produced by checkout, plus the missing-endpoint notice |
| `analytics.html` | DistilBERT evaluation report + live catalogue analytics |
| `nlp-lab.html` | Interactive sentiment pipeline + the repo's 30-case benchmark |
| `api-reference.html` | All 11 endpoints with live "send request" testing |

---

## 4. API integration layer

`js/api.js` is the single integration layer. Every method maps 1:1 to a route in
`backend/store/urls.py`. It ships **two interchangeable transports** that return the *same
normalised shapes*, selected by `API.mode` in `js/config.js`:

* **`local` (default)** — the same 11-endpoint contract implemented on top of the platform
  table API, so every page is fully functional and persistent in a static deployment.
* **`remote`** — talks to the real Django API at `API.remoteBase`
  (default `http://127.0.0.1:8000/api/`).

Switch at runtime with `?api=remote` or the badge in the header.

Errors are normalised into `ApiError` with a real HTTP status and a `kind`
(`network | validation | notfound | forbidden | unauthorized | server`), and the UI renders
distinct loading / empty / error states for each.

> **Remote mode requirement.** `settings.py` sets `ALLOWED_HOSTS = ["*"]` but installs no CORS
> middleware, so a browser on another origin cannot call the Django API. Remote mode therefore
> needs `django-cors-headers` on the backend (or the frontend served from Django's own origin).
> This is stated in the UI rather than worked around with a fake proxy.

### Local store tables (data models)

`categories` · `products` · `customers` · `orders` · `order_items` · `cart_items` ·
`recommendations` · `reviews` — see `.tables/schema.json`.

### Recommendation engine

`js/engine.js` is a faithful port of the documented v3 hybrid formula
(`TEAM_BACKEND_GUIDE.md` §3.1, implemented in `compute_recommendations.py`):

```
score = λ_svd · (BASE + qᵢᵀpᵤ + α·bᵢ) + λ_cbf · S_CBF(u, i) + γ · (sentimentᵢ − 0.5)

S_CBF(u, i) = ω · cos(tfidfᵢ, tfidf̄ᵤ) + (1 − ω) · 1 / (1 + ‖numᵢ − num̄ᵤ‖ / √5)
```

with the real constants from `settings.py` (λ_svd 0.6, λ_cbf 0.4, α 0.5, γ 0.3, ω 0.7,
BASE 3.0, MAX_PER_CAT 2), plus **purchase exclusion** and the **greedy diversity cap**.
`js/recompute.js` is the browser-side counterpart of `manage.py compute_recommendations`.

---

## 5. Data provenance (no invented backend data)

* **Categories** — the real Olist `product_category_name` slugs with their official
  `product_category_name_english` translations.
* **Product names** — generated with the repo's own algorithm
  (`import_data.py`, mirrored in `serializers.py`): PT `"{Categoria} {Adjetivo} #{HEX6}"`,
  EN `"{Adjective} {Category} #{HEX6}"`, `idx = int(external_id[:2], 16) % 10`.
* **`image_url`** — the repo's picsum scheme.
* **Customers** — the real `seed_test_scenario` personas with their documented ids, cities,
  states and purchase categories.
* **Recommendations** — produced by the ported engine, not hardcoded.
* **Reviews** — bilingual (PT original + EN translation) mapped to the 3-class pipeline.

**Honest limitation:** the raw Olist CSVs are git-ignored (`data/` is not in the repo) and the
trained weights are distributed outside git. The catalogue is therefore a representative subset
generated with the repo's own generators and category vocabulary; the sentiment model's numeric
weights are approximated by a transparent lexical scorer while the **published evaluation
metrics** on the Analytics page are the repository's real numbers.

---

## 6. Structure

```
index.html … api-reference.html     nine pages
css/style.css                        design system (tokens → components → responsive)
js/config.js                         endpoints, RECSYS_* constants, backend gap notes
js/api.js                            API layer (local + remote transports, ApiError)
js/format.js  js/ui.js               formatting helpers · UI kit (toasts, modal, skeletons, cards)
js/session.js                        customer context (no auth exists)
js/layout.js  js/main.js             shell (header/footer/switcher) · page bootstrap
js/engine.js  js/recompute.js        hybrid recommender port · offline recompute
js/nlp.js                            NLP pipeline (repair → detect → translate → classify) + 30 cases
js/seed.js                           dataset bootstrap (generators, personas, engine output)
js/pages/*.js                        one module per page
backend/                             the imported upstream repository, unmodified
```

### Design system

Dark premium theme, Inter + Plus Jakarta Sans + JetBrains Mono, glassmorphic surfaces,
gradient accents. Includes skeleton loaders, empty/error states, toast notifications, modals,
accessible focus rings, `aria-*` attributes, `prefers-reduced-motion` support and print styles.
Responsive across desktop / laptop / tablet / mobile with a dedicated hamburger nav below
900 px — not a scaled-down desktop.

---

## 7. Running it

```bash
# Frontend: any static server
python -m http.server 5173        # then open http://localhost:5173

# Backend (optional, for remote mode)
cd backend
uv sync
uv run python manage.py migrate
uv run python manage.py seed_test_scenario
uv run python manage.py runserver          # http://127.0.0.1:8000/api/
```

To use remote mode, add CORS to the backend first:

```python
# backend/recsys_backend/settings.py
INSTALLED_APPS += ["corsheaders"]
MIDDLEWARE.insert(0, "corsheaders.middleware.CorsMiddleware")
CORS_ALLOW_ALL_ORIGINS = True          # development only
```

---

## 8. Completed features

* Overview, Catalogue, Product, Recommendations, Cart, Orders, Analytics, NLP Lab, API Reference.
* Real query-param driven catalogue (search, category, ordering, page) + sentiment facet.
* Full cart lifecycle with server-side-style validation (`quantity ≥ 1`, empty-cart checkout → 400).
* Transactional checkout that creates `Order` + `OrderItem` rows and increments `total_purchases`.
* Recommendation shelf with score decomposition and a working recompute action.
* Interactive NLP pipeline (encoding repair → language detection → translation → 3-class
  classification) and the repository's 30-case benchmark.
* Evaluation dashboard with the published classification report, confusion matrix and training
  curves, plus live catalogue analytics.
* API reference with live request testing against either transport.
* Customer switcher, cart badge, toasts, modals, skeletons, empty/error states.
* Fully responsive; verified at 1280 px and 390 px.

## 9. Not implemented (and why)

* **Authentication / roles / protected routes** — the backend has none; inventing them would
  create functionality the API cannot enforce.
* **Admin dashboard** — Django admin exists server-side, but no admin API is exposed to a
  frontend, and no `is_staff`/permission logic exists in the API layer.
* **File uploads / product image management** — no `FileField`/`ImageField` in any model.
* **Payments** — checkout stops at an `Order` with `status="processing"`.
* **Wishlist, product ratings write, review submission** — no corresponding endpoints.

## 10. Recommended next steps

1. Add `django-cors-headers` so remote mode works from a different origin.
2. Implement the `GET /api/orders/` and `GET /api/customers/` routes already advertised in `SETUP.md`.
3. Expose read endpoints for `Review` (the model already stores PT + EN text and sentiment).
4. Serve the trained artefacts (`svd_cf.pkl`, `best_model_binary/`) so the frontend can call a
   real inference endpoint instead of the documented approximation.
5. Add server-side pagination-aware facets if the catalogue grows beyond a few thousand rows.

---

## 11. Public URLs & storage

* Frontend: served statically (Publish / Hosted Deploy).
* Backend API (local): `http://127.0.0.1:8000/api/`
* Local data store: platform table API at `tables/<table>`
* Production data: Cloudflare D1 (created from `.tables/schema.json` on Hosted Deploy)

**Note:** the preview data store and the live hosted database are independent. To move the
seeded catalogue to a deployed site, use the Hosted sync action rather than re-entering rows.
