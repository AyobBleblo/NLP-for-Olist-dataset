"""
store/views.py — DRF API views for Cart and Recommendation endpoints.

ARCHITECTURE CONSTRAINT
=======================
No ML model is loaded or called in this file.  All data served here
is pre-computed and stored in the database.  The only "computation"
that happens at request time is basic ORM queries and serialization.
"""

from django.db import transaction
from django.db.models import Count
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import filters, generics, status
from rest_framework.decorators import action
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.reverse import reverse
from rest_framework.views import APIView
from rest_framework.viewsets import ViewSet

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
from store.serializers import (
    CartItemWriteSerializer,
    CartSerializer,
    CategorySerializer,
    OrderItemSerializer,
    OrderSerializer,
    ProductDetailSerializer,
    ProductSummarySerializer,
    RecommendationSerializer,
    ReviewSerializer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_create_cart(customer: Customer) -> Cart:
    """Return the customer's cart, creating one if it doesn't exist."""
    cart, _ = Cart.objects.get_or_create(customer=customer)
    return cart


def _cart_response(cart: Cart, request: Request) -> Response:
    """Serialize cart with prefetched items and return a DRF Response."""
    cart_fresh = (
        Cart.objects.prefetch_related("items__product__category")
        .get(pk=cart.pk)
    )
    serializer = CartSerializer(cart_fresh, context={"request": request})
    return Response(serializer.data)


# ---------------------------------------------------------------------------
# Cart API  (customer identified by external_id in URL)
# ---------------------------------------------------------------------------

class CartView(APIView):
    """
    Cart management for a single customer.

    URL param: customer_external_id  (the Olist customer_unique_id)

    GET  /api/cart/<customer_external_id>/
        → View the full cart with all items.

    POST /api/cart/<customer_external_id>/add/
        Body: { "product_id": <int>, "quantity": <int> }
        → Add a product to the cart, or increment its quantity if already present.

    PATCH /api/cart/<customer_external_id>/update/<item_id>/
        Body: { "quantity": <int> }
        → Update the quantity of an existing cart item.

    DELETE /api/cart/<customer_external_id>/remove/<item_id>/
        → Remove a cart item entirely.

    POST /api/cart/<customer_external_id>/checkout/
        → Copy all CartItems into a new Order + OrderItems, then clear the cart.
    """

    def _get_customer(self, external_id: str) -> Customer:
        return get_object_or_404(Customer, external_id=external_id)

    # ---- GET /api/cart/<external_id>/ ----

    def get(self, request: Request, customer_external_id: str) -> Response:
        customer = self._get_customer(customer_external_id)
        cart = _get_or_create_cart(customer)
        return _cart_response(cart, request)


class CartAddItemView(APIView):
    """POST /api/cart/<customer_external_id>/add/"""

    def post(self, request: Request, customer_external_id: str) -> Response:
        customer = get_object_or_404(Customer, external_id=customer_external_id)
        cart = _get_or_create_cart(customer)

        serializer = CartItemWriteSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        product_id: int = serializer.validated_data["product_id"]
        quantity: int = serializer.validated_data["quantity"]

        cart_item, created = CartItem.objects.get_or_create(
            cart=cart,
            product_id=product_id,
            defaults={"quantity": quantity},
        )
        if not created:
            cart_item.quantity += quantity
            cart_item.save(update_fields=["quantity"])

        # Touch updated_at on the cart
        Cart.objects.filter(pk=cart.pk).update(updated_at=timezone.now())

        return _cart_response(cart, request)


class CartUpdateItemView(APIView):
    """PATCH /api/cart/<customer_external_id>/update/<item_id>/"""

    def patch(self, request: Request, customer_external_id: str, item_id: int) -> Response:
        customer = get_object_or_404(Customer, external_id=customer_external_id)
        cart = get_object_or_404(Cart, customer=customer)
        item = get_object_or_404(CartItem, pk=item_id, cart=cart)

        quantity = request.data.get("quantity")
        if quantity is None:
            return Response(
                {"detail": "'quantity' is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            quantity = int(quantity)
        except (TypeError, ValueError):
            return Response(
                {"detail": "'quantity' must be an integer."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if quantity < 1:
            return Response(
                {"detail": "'quantity' must be ≥ 1.  Use the remove endpoint to delete."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        item.quantity = quantity
        item.save(update_fields=["quantity"])
        Cart.objects.filter(pk=cart.pk).update(updated_at=timezone.now())

        return _cart_response(cart, request)


class CartRemoveItemView(APIView):
    """DELETE /api/cart/<customer_external_id>/remove/<item_id>/"""

    def delete(self, request: Request, customer_external_id: str, item_id: int) -> Response:
        customer = get_object_or_404(Customer, external_id=customer_external_id)
        cart = get_object_or_404(Cart, customer=customer)
        item = get_object_or_404(CartItem, pk=item_id, cart=cart)
        item.delete()
        Cart.objects.filter(pk=cart.pk).update(updated_at=timezone.now())
        return _cart_response(cart, request)


from collections import defaultdict
import uuid

from store.ml.loader import get_cf_model, get_cbf_model
from store.management.commands.compute_recommendations import (
    _apply_diversity_cap,
    _build_cbf_user_profile,
    _cbf_scores_all,
    compute_recommendations_for_customer,
    MAX_PER_CAT,
)


def _refresh_customer_recommendations(customer: Customer, top_n: int = 10) -> None:
    """
    Refresh recommendation rows for a customer based on their real purchase history.
    Uses the True Hybrid engine: SVD CF + TF-IDF/Numeric CBF + NLP DistilBERT Sentiment
    with Diversity Capping (max 2 items per category).
    """
    all_prods = list(Product.objects.select_related("category").all())
    if not all_prods:
        return

    purchased_pids = set(
        OrderItem.objects.filter(order__customer=customer).values_list("product_id", flat=True)
    )
    purchased_eids = list(
        OrderItem.objects.filter(order__customer=customer).values_list("product__external_id", flat=True)
    )
    candidates = [p for p in all_prods if p.id not in purchased_pids]
    if not candidates:
        candidates = all_prods

    prod_cat_map = {p.id: p.category_id for p in all_prods}

    svd_model = get_cf_model()
    cbf_model = get_cbf_model()

    cbf_all_scores = None
    cbf_pid_to_idx = None

    if cbf_model is not None:
        cbf_pid_to_idx = cbf_model.get("pid_to_idx")
        text_prof, num_prof = _build_cbf_user_profile(purchased_eids, cbf_model)
        if text_prof is not None and num_prof is not None:
            cbf_all_scores = _cbf_scores_all(cbf_model, text_prof, num_prof)

    # Compute true hybrid scores
    scored = compute_recommendations_for_customer(
        customer=customer,
        candidate_products=candidates,
        svd_model=svd_model,
        category_affinity=None,
        product_category_map=prod_cat_map,
        cbf_model=cbf_model,
        cbf_all_scores=cbf_all_scores,
        cbf_pid_to_idx=cbf_pid_to_idx,
    )

    # Diversity filter: at most MAX_PER_CAT per category
    chosen = _apply_diversity_cap(
        ranked=scored,
        product_category_map=prod_cat_map,
        top_n=top_n,
        max_per_cat=MAX_PER_CAT,
    )

    # Save to database atomically
    now = timezone.now()
    with transaction.atomic():
        Recommendation.objects.filter(customer=customer).delete()
        recs = [
            Recommendation(
                customer=customer,
                product_id=prod_id,
                score=score,
                generated_at=now,
            )
            for prod_id, score in chosen
        ]
        Recommendation.objects.bulk_create(recs)


class CartCheckoutView(APIView):
    """
    POST /api/cart/<customer_external_id>/checkout/

    Creates a new Order + OrderItems from the current cart contents,
    marks order as approved, updates recommendations, then clears the cart.
    """

    def post(self, request: Request, customer_external_id: str) -> Response:
        customer = get_object_or_404(Customer, external_id=customer_external_id)
        cart = get_object_or_404(Cart, customer=customer)

        items = list(
            cart.items.select_related("product").all()
        )

        if not items:
            return Response(
                {"detail": "Cart is empty."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        with transaction.atomic():
            # Create a new Order with status="approved"
            new_order = Order.objects.create(
                external_id=uuid.uuid4().hex,
                customer=customer,
                purchased_at=timezone.now(),
                status="approved",
            )

            # Copy CartItems → OrderItems
            order_items = [
                OrderItem(
                    order=new_order,
                    product=item.product,
                    price=item.product.price,
                )
                for item in items
                if item.product is not None
            ]
            OrderItem.objects.bulk_create(order_items)

            # Update total_purchases on products (increment by 1 per item)
            for item in items:
                if item.product_id:
                    Product.objects.filter(pk=item.product_id).update(
                        total_purchases=Product.objects.filter(
                            pk=item.product_id
                        ).values_list("total_purchases", flat=True)[0] + 1
                    )

            # Clear the cart
            cart.items.all().delete()
            Cart.objects.filter(pk=cart.pk).update(updated_at=timezone.now())

        # Refresh recommendations for this customer based on their new purchase history!
        _refresh_customer_recommendations(customer)

        return Response(
            {
                "detail": "Checkout successful. Order approved.",
                "order_external_id": new_order.external_id,
                "order_id": new_order.pk,
                "status": new_order.status,
                "items_count": len(order_items),
                "recommendations_updated": True,
            },
            status=status.HTTP_201_CREATED,
        )


# ---------------------------------------------------------------------------
# Recommendation API (read-only)
# ---------------------------------------------------------------------------

class RecommendationListView(APIView):
    """
    GET /api/recommendations/<customer_external_id>/

    Returns the top-N precomputed recommendations for a customer,
    ordered by score DESC.

    Query params:
        n   — number of results to return (default: 10, max: 100)

    CRITICAL: this endpoint performs ZERO ML inference.
              It reads only from the Recommendation table + Product table.
    """

    DEFAULT_N = 10
    MAX_N = 100

    def get(self, request: Request, customer_external_id: str) -> Response:
        customer = get_object_or_404(Customer, external_id=customer_external_id)

        try:
            n = int(request.query_params.get("n", self.DEFAULT_N))
            n = max(1, min(n, self.MAX_N))
        except (TypeError, ValueError):
            n = self.DEFAULT_N

        recommendations = (
            Recommendation.objects.filter(customer=customer)
            .select_related("product__category")
            .order_by("-score")[:n]
        )

        serializer = RecommendationSerializer(
            recommendations, many=True, context={"request": request}
        )

        return Response(
            {
                "customer_external_id": customer_external_id,
                "count": len(serializer.data),
                "results": serializer.data,
            }
        )


# ---------------------------------------------------------------------------
# API Root
# ---------------------------------------------------------------------------

class ApiRootView(APIView):
    """
    API Root providing hyperlinks to all available resources.
    """

    def get(self, request: Request) -> Response:
        return Response(
            {
                "categories": reverse("category-list", request=request),
                "products": reverse("product-list", request=request),
                "cart_template": f"{request.build_absolute_uri('/api/cart/')}<customer_id>/",
                "recommendations_template": f"{request.build_absolute_uri('/api/recommendations/')}<customer_id>/",
                "orders_template": f"{request.build_absolute_uri('/api/orders/')}<customer_id>/",
                "reviews": reverse("review-list", request=request),
                "stats": reverse("stats", request=request),
            }
        )


# ---------------------------------------------------------------------------
# Category API
# ---------------------------------------------------------------------------

class CategoryListView(generics.ListAPIView):
    """
    GET /api/categories/
    List all categories with English translations and product count.
    Supports ordering (e.g. ?ordering=-product_count or ?ordering=name_translated).
    Supports search (e.g. ?search=beauty).
    """

    serializer_class = CategorySerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["name", "name_translated"]
    ordering_fields = ["name", "name_translated", "product_count"]
    ordering = ["name_translated"]
    pagination_class = None

    def get_queryset(self):
        return (
            Category.objects.annotate(product_count=Count("products"))
            .order_by("name_translated")
        )


class CategoryDetailView(generics.RetrieveAPIView):
    """
    GET /api/categories/<int:pk>/
    Retrieve a single category by primary key.
    """

    serializer_class = CategorySerializer

    def get_queryset(self):
        return Category.objects.annotate(product_count=Count("products"))


# ---------------------------------------------------------------------------
# Product API
# ---------------------------------------------------------------------------

class ProductListView(generics.ListAPIView):
    """
    GET /api/products/
    Paginated list of products with filtering, search, and ordering.

    Query params:
        category        — Filter by category ID (e.g. ?category=5)
        search          — Search by name, English translation, category, or product ID
        ordering        — Order by price, -price, total_purchases, -total_purchases, avg_sentiment_score
    """

    serializer_class = ProductSummarySerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = [
        "name",
        "name_translated",
        "category__name",
        "category__name_translated",
        "external_id",
    ]
    ordering_fields = ["price", "total_purchases", "avg_sentiment_score", "id"]
    ordering = ["-total_purchases", "id"]

    def get_queryset(self):
        qs = Product.objects.select_related("category").all()
        cat = self.request.query_params.get("category")
        if cat:
            if cat.isdigit():
                qs = qs.filter(category_id=int(cat))
            else:
                qs = qs.filter(category__name=cat)
        sentiment = self.request.query_params.get("sentiment")
        if sentiment:
            if sentiment == "positive":
                qs = qs.filter(avg_sentiment_score__gte=0.7)
            elif sentiment == "mixed":
                qs = qs.filter(avg_sentiment_score__gte=0.45, avg_sentiment_score__lt=0.7)
            elif sentiment == "negative":
                qs = qs.filter(avg_sentiment_score__gt=0.0, avg_sentiment_score__lt=0.45)
            elif sentiment == "none":
                qs = qs.filter(avg_sentiment_score=0.0)
        return qs


class ProductDetailView(generics.RetrieveAPIView):
    """
    GET /api/products/<str:lookup_value>/
    Retrieve a product by numeric ID or Olist external_id hex string.
    """

    serializer_class = ProductDetailSerializer

    def get_object(self):
        lookup_value = self.kwargs.get("lookup_value", "")
        if lookup_value.isdigit():
            return get_object_or_404(
                Product.objects.select_related("category"),
                pk=int(lookup_value),
            )
        return get_object_or_404(
            Product.objects.select_related("category"),
            external_id=lookup_value,
        )


# ---------------------------------------------------------------------------
# Order API
# ---------------------------------------------------------------------------

class OrderListView(APIView):
    """
    GET /api/orders/<customer_external_id>/
    List all past orders for a customer with embedded items.
    """

    def get(self, request: Request, customer_external_id: str) -> Response:
        customer = get_object_or_404(Customer, external_id=customer_external_id)
        orders = (
            Order.objects.filter(customer=customer)
            .prefetch_related("items__product__category")
            .order_by("-purchased_at")
        )
        serializer = OrderSerializer(orders, many=True)
        return Response(serializer.data)


# ---------------------------------------------------------------------------
# Review API
# ---------------------------------------------------------------------------

class ReviewListView(APIView):
    """
    GET /api/reviews/?product=<product_external_id>
    List reviews, optionally filtered by product external ID.
    """

    def get(self, request: Request) -> Response:
        product_ext = request.query_params.get("product")
        qs = Review.objects.select_related("product").all()
        if product_ext:
            qs = qs.filter(product__external_id=product_ext)
        serializer = ReviewSerializer(qs[:50], many=True)
        return Response(serializer.data)


# ---------------------------------------------------------------------------
# Stats API
# ---------------------------------------------------------------------------

class StatsView(APIView):
    """
    GET /api/stats/
    Returns aggregate counts across the platform.
    """

    def get(self, request: Request) -> Response:
        return Response(
            {
                "products": Product.objects.count(),
                "categories": Category.objects.count(),
                "customers": Customer.objects.count(),
                "recommendations": Recommendation.objects.count(),
                "orders": Order.objects.count(),
                "reviews": Review.objects.count(),
            }
        )

