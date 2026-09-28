"""
store/models.py — Data models for the e-commerce recommendation system.

All models follow the spec exactly.  A few design notes:
- external_id fields use CharField(max_length=64, unique=True, db_index=True)
  because Olist IDs are MD5-like hex strings.
- avg_sentiment_score and total_purchases on Product are denormalized
  aggregates refreshed by management commands, never computed at request time.
- The Recommendation table is written by an offline job and is read-only
  from the API's perspective.
- Review.sentiment uses a short string field ('positive'/'negative'/None)
  so the JSON API can return it directly without extra decoding.
"""

from django.db import models
from django.utils import timezone


# ---------------------------------------------------------------------------
# Category
# ---------------------------------------------------------------------------

class Category(models.Model):
    """
    Product category.  The raw Olist data has Portuguese category names;
    the translate_products management command populates name_translated.
    """

    name = models.CharField(
        max_length=200,
        unique=True,
        help_text="Original Portuguese category name from Olist CSV.",
    )
    name_translated = models.CharField(
        max_length=200,
        blank=True,
        default="",
        help_text="English translation (populated by translate_products command).",
    )

    class Meta:
        verbose_name_plural = "categories"
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name_translated or self.name


# ---------------------------------------------------------------------------
# Customer
# ---------------------------------------------------------------------------

class Customer(models.Model):
    """
    Represents a unique buyer in the Olist dataset.
    external_id maps to customer_unique_id in the CSV.
    """

    external_id = models.CharField(max_length=64, unique=True, db_index=True)
    city = models.CharField(max_length=100, blank=True, default="")
    state = models.CharField(max_length=2, blank=True, default="")

    class Meta:
        ordering = ["external_id"]

    def __str__(self) -> str:
        return f"Customer({self.external_id[:8]}…)"


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------

class Product(models.Model):
    """
    A single product.

    Denormalized fields:
    - avg_sentiment_score: float in [0, 1] where 1 = all Positive reviews.
      Updated by the run_sentiment_batch management command after processing
      all reviews for a product.
    - total_purchases: count of OrderItems referencing this product.
      Updated by the import_data command (and can be refreshed later).
    - image_url: deterministic placeholder generated from external_id since
      Olist ships no real product images.
    """

    external_id = models.CharField(max_length=64, unique=True, db_index=True)
    category = models.ForeignKey(
        Category,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="products",
    )
    name = models.CharField(
        max_length=300,
        blank=True,
        default="",
        help_text="Original Portuguese product name (if available in dataset).",
    )
    name_translated = models.CharField(
        max_length=300,
        blank=True,
        default="",
        help_text="English translation of product name.",
    )
    price = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    # Denormalized / precomputed — never recalculated inline on a request
    avg_sentiment_score = models.FloatField(
        default=0.0,
        help_text=(
            "Mean sentiment score across all processed reviews. "
            "0 = all Negative, 1 = all Positive.  Updated offline."
        ),
    )
    total_purchases = models.PositiveIntegerField(
        default=0,
        help_text="Number of times this product has been bought.  Updated offline.",
    )
    image_url = models.URLField(
        max_length=500,
        blank=True,
        default="",
        help_text="Deterministic placeholder URL seeded from external_id.",
    )

    class Meta:
        ordering = ["-total_purchases", "external_id"]

    def __str__(self) -> str:
        return self.name_translated or self.name or self.external_id[:12]


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

class Order(models.Model):
    """
    An order placed by a customer.
    external_id maps to order_id in the Olist CSV.
    """

    STATUS_CHOICES = [
        ("delivered", "Delivered"),
        ("shipped", "Shipped"),
        ("processing", "Processing"),
        ("canceled", "Canceled"),
        ("unavailable", "Unavailable"),
        ("invoiced", "Invoiced"),
        ("approved", "Approved"),
        ("created", "Created"),
    ]

    external_id = models.CharField(max_length=64, unique=True, db_index=True)
    customer = models.ForeignKey(
        Customer,
        on_delete=models.CASCADE,
        related_name="orders",
    )
    purchased_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="delivered",
    )

    class Meta:
        ordering = ["-purchased_at"]

    def __str__(self) -> str:
        return f"Order({self.external_id[:8]}…)"


# ---------------------------------------------------------------------------
# OrderItem
# ---------------------------------------------------------------------------

class OrderItem(models.Model):
    """
    A single product line within an Order.
    price is captured at time of purchase (may differ from current Product.price).
    """

    order = models.ForeignKey(
        Order,
        on_delete=models.CASCADE,
        related_name="items",
    )
    product = models.ForeignKey(
        Product,
        on_delete=models.CASCADE,
        related_name="order_items",
    )
    price = models.DecimalField(max_digits=10, decimal_places=2)

    class Meta:
        ordering = ["id"]

    def __str__(self) -> str:
        return f"OrderItem(order={self.order_id}, product={self.product_id})"


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------

class Review(models.Model):
    """
    A customer review for an order/product.

    Processing lifecycle:
      1. comment_text is the raw Portuguese text from the CSV.
      2. comment_text_translated is filled by run_sentiment_batch.
      3. sentiment ('positive'/'negative') is set by the same batch job.
      4. sentiment_processed_at is a timestamp for idempotency — rows with
         sentiment IS NOT NULL are skipped on subsequent runs.
    """

    SENTIMENT_POSITIVE = "positive"
    SENTIMENT_NEGATIVE = "negative"
    SENTIMENT_CHOICES = [
        (SENTIMENT_POSITIVE, "Positive"),
        (SENTIMENT_NEGATIVE, "Negative"),
    ]

    external_id = models.CharField(max_length=64, db_index=True)
    order = models.ForeignKey(
        Order,
        on_delete=models.CASCADE,
        related_name="reviews",
        null=True,
        blank=True,
    )
    product = models.ForeignKey(
        Product,
        on_delete=models.CASCADE,
        related_name="reviews",
        null=True,
        blank=True,
    )
    comment_text = models.TextField(
        blank=True,
        default="",
        help_text="Original Portuguese review text.",
    )
    comment_text_translated = models.TextField(
        blank=True,
        default="",
        help_text="English translation, produced by run_sentiment_batch.",
    )
    rating = models.PositiveSmallIntegerField(
        null=True,
        blank=True,
        help_text="Star rating (1–5).",
    )
    sentiment = models.CharField(
        max_length=10,
        choices=SENTIMENT_CHOICES,
        null=True,
        blank=True,
        help_text="Binary sentiment label.  NULL until processed by batch job.",
    )
    sentiment_processed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the sentiment was last computed.  NULL = not yet processed.",
    )

    class Meta:
        ordering = ["id"]
        indexes = [
            # Used by the batch job to find unprocessed reviews efficiently
            models.Index(fields=["sentiment"], name="review_sentiment_idx"),
        ]

    def __str__(self) -> str:
        return f"Review({self.external_id[:8]}… | {self.sentiment or 'unprocessed'})"


# ---------------------------------------------------------------------------
# Cart / CartItem
# ---------------------------------------------------------------------------

class Cart(models.Model):
    """
    A shopping cart.  One per customer (enforced by unique_together).
    created_at / updated_at for auditing and cache-busting.
    """

    customer = models.OneToOneField(
        Customer,
        on_delete=models.CASCADE,
        related_name="cart",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"Cart(customer={self.customer_id})"


class CartItem(models.Model):
    """A line item in a Cart."""

    cart = models.ForeignKey(
        Cart,
        on_delete=models.CASCADE,
        related_name="items",
    )
    product = models.ForeignKey(
        Product,
        on_delete=models.CASCADE,
        related_name="cart_items",
    )
    quantity = models.PositiveIntegerField(default=1)

    class Meta:
        unique_together = [("cart", "product")]
        ordering = ["id"]

    def __str__(self) -> str:
        return f"CartItem(product={self.product_id}, qty={self.quantity})"


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class Recommendation(models.Model):
    """
    Precomputed (customer, product) recommendation score.

    Written by the compute_recommendations management command; read-only
    from the API's perspective — the API never triggers ML inference.

    score: float produced by the recommendation algorithm (e.g. 0–1 ALS
    probability, association-rule confidence, etc.).
    generated_at: timestamp of the last compute run (used for freshness checks).
    """

    customer = models.ForeignKey(
        Customer,
        on_delete=models.CASCADE,
        related_name="recommendations",
    )
    product = models.ForeignKey(
        Product,
        on_delete=models.CASCADE,
        related_name="recommendations",
    )
    score = models.FloatField(
        help_text="Precomputed recommendation relevance score.",
    )
    generated_at = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = [("customer", "product")]
        ordering = ["-score"]
        indexes = [
            # Used by GET /recommendations/<customer_id>/
            models.Index(
                fields=["customer", "-score"],
                name="rec_customer_score_idx",
            ),
        ]

    def __str__(self) -> str:
        return (
            f"Recommendation(customer={self.customer_id}, "
            f"product={self.product_id}, score={self.score:.3f})"
        )
