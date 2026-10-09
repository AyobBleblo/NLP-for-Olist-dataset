"""
store/serializers.py — Django REST Framework serializers.

All serializers are read-optimised (select_related / prefetch_related are
handled in views, not here).  No ML inference is triggered here — only
pre-computed fields are serialized.
"""

from rest_framework import serializers

from store.models import (
    Cart,
    CartItem,
    Category,
    Customer,
    Order,
    OrderItem,
    Product,
    Recommendation,
    Review,
)


# ---------------------------------------------------------------------------
# Category
# ---------------------------------------------------------------------------

class CategorySerializer(serializers.ModelSerializer):
    """Category representation including product count."""

    product_count = serializers.IntegerField(read_only=True, default=0)

    class Meta:
        model = Category
        fields = [
            "id",
            "name",
            "name_translated",
            "product_count",
        ]
        read_only_fields = fields


ADJECTIVES_EN = [
    "Classic", "Premium", "Essential", "Deluxe", "Pro",
    "Smart", "Modern", "Original", "Urban", "Signature",
]
ADJECTIVES_PT = [
    "Clássico", "Premium", "Essencial", "Luxo", "Pro",
    "Smart", "Moderno", "Original", "Urbano", "Assinatura",
]


def _generate_display_name_en(product: Product) -> str:
    """Return English product name or a deterministic realistic title."""
    if product.name_translated:
        return product.name_translated
    if product.name:
        return product.name
    cat_name = (
        product.category.name_translated.replace("_", " ").title()
        if product.category and product.category.name_translated
        else (product.category.name.replace("_", " ").title() if product.category else "Product")
    )
    idx = int(product.external_id[:2], 16) % len(ADJECTIVES_EN) if product.external_id else 0
    adj = ADJECTIVES_EN[idx]
    return f"{adj} {cat_name} #{product.external_id[:6].upper()}"


def _generate_display_name_pt(product: Product) -> str:
    """Return Portuguese product name or a deterministic realistic title."""
    if product.name:
        return product.name
    cat_name = (
        product.category.name.replace("_", " ").title()
        if product.category and product.category.name
        else "Produto"
    )
    idx = int(product.external_id[:2], 16) % len(ADJECTIVES_PT) if product.external_id else 0
    adj = ADJECTIVES_PT[idx]
    return f"{cat_name} {adj} #{product.external_id[:6].upper()}"


# ---------------------------------------------------------------------------
# Product (used inside Recommendation, Cart responses, and catalog)
# ---------------------------------------------------------------------------

class ProductSummarySerializer(serializers.ModelSerializer):
    """Compact product representation for catalog list and embedding."""

    name = serializers.SerializerMethodField()
    name_translated = serializers.SerializerMethodField()
    category_id = serializers.IntegerField(
        source="category.id",
        default=None,
        read_only=True,
    )
    category_name = serializers.CharField(
        source="category.name_translated",
        default="",
        read_only=True,
    )

    class Meta:
        model = Product
        fields = [
            "id",
            "external_id",
            "name",
            "name_translated",
            "category_id",
            "category_name",
            "price",
            "avg_sentiment_score",
            "total_purchases",
            "image_url",
        ]
        read_only_fields = fields

    def get_name(self, obj: Product) -> str:
        return _generate_display_name_pt(obj)

    def get_name_translated(self, obj: Product) -> str:
        return _generate_display_name_en(obj)


class ProductDetailSerializer(serializers.ModelSerializer):
    """Full product detail including original Portuguese name."""

    name = serializers.SerializerMethodField()
    name_translated = serializers.SerializerMethodField()
    category_id = serializers.IntegerField(
        source="category.id",
        default=None,
        read_only=True,
    )
    category_name = serializers.CharField(
        source="category.name_translated",
        default="",
        read_only=True,
    )
    category_name_original = serializers.CharField(
        source="category.name",
        default="",
        read_only=True,
    )

    class Meta:
        model = Product
        fields = [
            "id",
            "external_id",
            "name",
            "name_translated",
            "category_id",
            "category_name",
            "category_name_original",
            "price",
            "avg_sentiment_score",
            "total_purchases",
            "image_url",
        ]
        read_only_fields = fields

    def get_name(self, obj: Product) -> str:
        return _generate_display_name_pt(obj)

    def get_name_translated(self, obj: Product) -> str:
        return _generate_display_name_en(obj)


# ---------------------------------------------------------------------------
# Cart / CartItem
# ---------------------------------------------------------------------------

class CartItemSerializer(serializers.ModelSerializer):
    """CartItem with embedded product summary."""

    product = ProductSummarySerializer(read_only=True)
    product_id = serializers.IntegerField(write_only=True)

    class Meta:
        model = CartItem
        fields = ["id", "product", "product_id", "quantity"]


class CartItemWriteSerializer(serializers.Serializer):
    """
    Used for add/update operations.
    Accepts product_id (integer FK) and quantity.
    """

    product_id = serializers.IntegerField()
    quantity = serializers.IntegerField(default=1)

    def validate_product_id(self, value):
        from store.models import Product
        if not Product.objects.filter(pk=value).exists():
            raise serializers.ValidationError(
                f"Product with id={value} does not exist."
            )
        return value

    def validate_quantity(self, value):
        if value < 1:
            raise serializers.ValidationError("Quantity must be at least 1.")
        return value


class CartSerializer(serializers.ModelSerializer):
    """Full cart with all items embedded."""

    items = CartItemSerializer(many=True, read_only=True)
    customer_id = serializers.IntegerField(source="customer.id", read_only=True)
    total = serializers.SerializerMethodField()

    class Meta:
        model = Cart
        fields = ["id", "customer_id", "items", "total", "created_at", "updated_at"]

    def get_total(self, obj) -> float:
        """Sum of (price × quantity) across all items."""
        return sum(
            float(item.product.price) * item.quantity
            for item in obj.items.all()
            if item.product
        )


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class RecommendationSerializer(serializers.ModelSerializer):
    """
    Recommendation with embedded product data.

    IMPORTANT: no ML inference happens here.  `product` data is fully
    precomputed (avg_sentiment_score, image_url, etc.).
    """

    product = ProductSummarySerializer(read_only=True)

    class Meta:
        model = Recommendation
        fields = ["id", "product", "score", "generated_at"]
        read_only_fields = fields


# ---------------------------------------------------------------------------
# Order / OrderItem
# ---------------------------------------------------------------------------

class OrderItemSerializer(serializers.ModelSerializer):
    product_external_id = serializers.CharField(source="product.external_id", read_only=True)
    product_name = serializers.CharField(source="product.name_translated", default="", read_only=True)
    product = ProductSummarySerializer(read_only=True)

    class Meta:
        model = OrderItem
        fields = ["id", "product_external_id", "product_name", "product", "price"]


class OrderSerializer(serializers.ModelSerializer):
    customer_external_id = serializers.CharField(source="customer.external_id", read_only=True)
    items = OrderItemSerializer(many=True, read_only=True)
    total = serializers.SerializerMethodField()

    class Meta:
        model = Order
        fields = [
            "id",
            "external_id",
            "customer_external_id",
            "purchased_at",
            "status",
            "total",
            "items",
        ]

    def get_total(self, obj) -> float:
        return sum(float(item.price) for item in obj.items.all())


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------

class ReviewSerializer(serializers.ModelSerializer):
    product_external_id = serializers.CharField(source="product.external_id", read_only=True)

    class Meta:
        model = Review
        fields = [
            "id",
            "external_id",
            "product_external_id",
            "rating",
            "comment_text",
            "comment_text_translated",
            "sentiment",
            "sentiment_processed_at",
        ]
