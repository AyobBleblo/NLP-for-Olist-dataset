# E-Commerce Backend & Recommendation System Guide

Welcome to the backend documentation for the **Olist E-Commerce & Product Recommendation System**. This project is built with **Django 6.1** and **Django REST Framework (DRF)**.

---

## 🏛️ System Architecture

The backend is built around a key architectural design principle:
> **Zero ML Inference in the Request Path:** All heavy AI/ML calculations (translation, DistilBERT sentiment scoring, and collaborative filtering recommendations) are performed **offline in batch jobs** and cached in the database. When users browse the website, checkout, or request recommendations, the API executes only fast ORM database queries with sub-millisecond response times.

```
                           OFFLINE BATCH PIPELINE
  [Olist CSVs] ──> import_data ──> translate_products ──> run_sentiment_batch ──> compute_recommendations
                         │                    │                      │                           │
                         ▼                    ▼                      ▼                           ▼
                  [Products & Cats]   [English Names]     [avg_sentiment_score]       [Recommendation Table]
                         │                    │                      │                           │
                         └────────────────────┴──────────────────────┴───────────────────────────┘
                                                       │
                                                       ▼
                                            SQLite Database (db.sqlite3)
                                                       ▲
                                                       │ fast reads
                                              ONLINE REST API (/api/)
                         ┌─────────────────────┼──────────────────────┬───────────────────────────┐
                         ▼                     ▼                      ▼                           ▼
                    Catalog API             Cart API             Checkout API            Recommendation API
                  /categories/          /cart/<cust_id>/      /cart/<cust_id>/          /recommendations/
                   /products/                 /add/              /checkout/                <cust_id>/
```

---

## 🎯 WHERE IS THE RECOMMENDATION PART?

The recommendation system consists of **3 interconnected layers**:

### 1. The Database Table: `Recommendation` ([store/models.py](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe%20Project/store/models.py#L210-L245))
Stores precomputed (Customer, Product, Score) pairs:
```python
class Recommendation(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="recommendations")
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="recommendations")
    score = models.FloatField(help_text="Recommendation relevance score (higher = stronger)")
    generated_at = models.DateTimeField(auto_now=True)
```

### 2. The Offline Scoring Engine: [compute_recommendations.py](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe%20Project/store/management/commands/compute_recommendations.py)
This is the **heart of your recommendation algorithms**.
- **CLI Command**:
  ```powershell
  uv run python manage.py compute_recommendations --top-n 10
  ```
- **The Core Function**:
  In `store/management/commands/compute_recommendations.py`, line 42:
  ```python
  def compute_recommendations(
      customer_db_id: int,
      all_product_ids: list[int],
      purchased_product_ids: set[int],
  ) -> list[tuple[int, float]]:
  ```
- **How to plug in your ML algorithm**:
  Replace this function's body with your algorithm:
  - **Collaborative Filtering**: Matrix Factorization / ALS (e.g., using `implicit` or `scikit-surprise`).
  - **Association Rules**: Market Basket Analysis / Apriori (e.g., `mlxtend`).
  - **Content-Based Filtering**: Product category & description vector similarity.
  - **Hybrid with NLP Sentiment**: Multiply collaborative filtering score by `(1 + product.avg_sentiment_score)` so highly-rated positive products rank higher!

### 3. The Live Online API: `RecommendationListView` ([store/views.py](file:///d:/01_Projects_Workspace/Recoomedation%20System%20Copstoe%20Project/store/views.py#L235-L280))
When the client calls `GET /api/recommendations/<customer_id>/?n=10`:
- It fetches the precomputed scores instantly from the database.
- It automatically embeds the full product card: title, image URL, category, price, and sentiment rating.
- Response time: **< 10ms**.

---

## 📡 Complete API Reference (Every Endpoint)

Base URL: `http://127.0.0.1:8000/api/`

---

### 1. API Root
Discover all available endpoints and API links.

- **Method**: `GET`
- **URL**: `/api/`
- **Response** (`200 OK`):
```json
{
  "categories": "http://127.0.0.1:8000/api/categories/",
  "products": "http://127.0.0.1:8000/api/products/",
  "cart_template": "http://127.0.0.1:8000/api/cart/<customer_id>/",
  "recommendations_template": "http://127.0.0.1:8000/api/recommendations/<customer_id>/"
}
```

---

### 2. Categories List
Retrieve all product categories with both Portuguese names, English translations, and product counts.

- **Method**: `GET`
- **URL**: `/api/categories/`
- **Query Parameters**:
  - `search`: Filter categories by name (e.g. `?search=beauty` or `?search=esporte`)
  - `ordering`: Order by field (e.g. `?ordering=-product_count` or `?ordering=name_translated`)
- **Response** (`200 OK`):
```json
[
  {
    "id": 1,
    "name": "brinquedos",
    "name_translated": "toys",
    "product_count": 35
  },
  {
    "id": 15,
    "name": "ferramentas_jardim",
    "name_translated": "garden_tools",
    "product_count": 42
  }
]
```

---

### 3. Category Detail
Retrieve details for a single category by primary key.

- **Method**: `GET`
- **URL**: `/api/categories/<int:id>/`
- **Example**: `/api/categories/1/`
- **Response** (`200 OK`):
```json
{
  "id": 1,
  "name": "brinquedos",
  "name_translated": "toys",
  "product_count": 35
}
```

---

### 4. Products List (Catalog)
Paginated list of products with filtering, search, and sorting.

- **Method**: `GET`
- **URL**: `/api/products/`
- **Query Parameters**:
  - `category`: Filter by category ID or name (e.g. `?category=1` or `?category=toys`)
  - `search`: Search products by title, category, or ID (e.g. `?search=modern`)
  - `ordering`: Sort results (e.g. `?ordering=-price`, `?ordering=price`, `?ordering=-total_purchases`, `?ordering=-avg_sentiment_score`)
  - `page`: Page number (default: 1, 20 products per page)
- **Response** (`200 OK`):
```json
{
  "count": 1000,
  "next": "http://127.0.0.1:8000/api/products/?page=2",
  "previous": null,
  "results": [
    {
      "id": 156,
      "external_id": "7fab1a1472fdd934397068931f63f3ca",
      "name": "Brinquedos Original #7FAB1A",
      "name_translated": "Original Toys #7FAB1A",
      "category_id": 1,
      "category_name": "toys",
      "price": "70.66",
      "avg_sentiment_score": 0.0,
      "total_purchases": 48,
      "image_url": "https://picsum.photos/seed/6716730e/400/400"
    }
  ]
}
```

---

### 5. Product Detail
Retrieve full details of a specific product. Works with either the database numeric ID or the Olist external hex ID.

- **Method**: `GET`
- **URL**: `/api/products/<lookup_value>/`
- **Examples**:
  - Numeric ID: `/api/products/156/`
  - Olist Hash: `/api/products/7fab1a1472fdd934397068931f63f3ca/`
- **Response** (`200 OK`):
```json
{
  "id": 156,
  "external_id": "7fab1a1472fdd934397068931f63f3ca",
  "name": "Brinquedos Original #7FAB1A",
  "name_translated": "Original Toys #7FAB1A",
  "category_id": 1,
  "category_name": "toys",
  "category_name_original": "brinquedos",
  "price": "70.66",
  "avg_sentiment_score": 0.0,
  "total_purchases": 48,
  "image_url": "https://picsum.photos/seed/6716730e/400/400"
}
```

---

### 6. View Customer Cart
View all active items in a customer's shopping cart, along with the calculated total order value.

- **Method**: `GET`
- **URL**: `/api/cart/<customer_external_id>/`
- **Example**: `/api/cart/customer_123/`
- **Response** (`200 OK`):
```json
{
  "id": 1,
  "customer_id": 12,
  "items": [
    {
      "id": 5,
      "product": {
        "id": 156,
        "external_id": "7fab1a1472fdd934397068931f63f3ca",
        "name_translated": "Original Toys #7FAB1A",
        "category_name": "toys",
        "price": "70.66",
        "image_url": "https://picsum.photos/seed/6716730e/400/400"
      },
      "quantity": 2
    }
  ],
  "total": 141.32,
  "created_at": "2026-09-28T14:30:00Z",
  "updated_at": "2026-09-28T14:35:00Z"
}
```

---

### 7. Add Item to Cart
Add a product or increment its quantity in the cart.

- **Method**: `POST`
- **URL**: `/api/cart/<customer_external_id>/add/`
- **Request Body** (`application/json`):
```json
{
  "product_id": 156,
  "quantity": 1
}
```
- **Response** (`200 OK`): Returns the updated full cart JSON.

---

### 8. Update Item Quantity in Cart
Directly modify the quantity of an item already in the cart.

- **Method**: `PATCH`
- **URL**: `/api/cart/<customer_external_id>/update/<int:item_id>/`
- **Request Body** (`application/json`):
```json
{
  "quantity": 3
}
```
- **Response** (`200 OK`): Returns the updated full cart JSON.

---

### 9. Remove Item from Cart
Remove an item from the cart completely.

- **Method**: `DELETE`
- **URL**: `/api/cart/<customer_external_id>/remove/<int:item_id>/`
- **Response** (`200 OK`): Returns the updated cart without the deleted item.

---

### 10. Checkout Cart
Converts the active cart into a confirmed `Order` and `OrderItem` records, increments `total_purchases` on products, and empties the cart.

- **Method**: `POST`
- **URL**: `/api/cart/<customer_external_id>/checkout/`
- **Response** (`201 Created`):
```json
{
  "detail": "Checkout successful.",
  "order_external_id": "ord_a9e7f4c1...",
  "order_id": 84,
  "items_count": 2
}
```

---

### 11. Recommendations API ⭐
Fetch personalized, precomputed product recommendations for a specific customer.

- **Method**: `GET`
- **URL**: `/api/recommendations/<customer_external_id>/`
- **Query Parameters**:
  - `n`: Number of recommendations to return (default: `10`, maximum: `100`)
- **Example**: `/api/recommendations/customer_123/?n=5`
- **Response** (`200 OK`):
```json
{
  "customer_external_id": "customer_123",
  "count": 5,
  "results": [
    {
      "id": 1,
      "product": {
        "id": 637,
        "external_id": "f1fe595ee7ef768b41bd9b246d13432d",
        "name_translated": "Premium Garden Tools #F1FE59",
        "category_name": "garden_tools",
        "price": "99.90",
        "avg_sentiment_score": 0.95,
        "total_purchases": 48,
        "image_url": "https://picsum.photos/seed/8d977572/400/400"
      },
      "score": 0.892,
      "generated_at": "2026-09-28T12:00:00Z"
    }
  ]
}
```

---

## 🛠️ Management Commands Reference

All background tasks are run via Django management commands:

| Command | Purpose |
|---|---|
| `uv run python manage.py import_data --limit 1000` | Imports top 1,000 products & 58 categories from Olist CSVs |
| `uv run python manage.py translate_products` | Uses NLLB-200 to translate Portuguese category/product names to English |
| `uv run python manage.py run_sentiment_batch` | Translates review text and classifies sentiment using DistilBERT |
| `uv run python manage.py compute_recommendations --top-n 10` | Computes user-product scores and populates the `Recommendation` table |
