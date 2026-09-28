"""
store/urls.py — URL routing for the store app.

All endpoints are prefixed with /api/ by the root urls.py.

Catalog endpoints
-----------------
  GET    /api/
  GET    /api/categories/
  GET    /api/categories/<int:pk>/
  GET    /api/products/
         ?category=<id_or_name>
         ?search=<query>
         ?ordering=<field>
  GET    /api/products/<int:pk_or_str:external_id>/

Cart endpoints
--------------
  GET    /api/cart/<customer_external_id>/
  POST   /api/cart/<customer_external_id>/add/
  PATCH  /api/cart/<customer_external_id>/update/<item_id>/
  DELETE /api/cart/<customer_external_id>/remove/<item_id>/
  POST   /api/cart/<customer_external_id>/checkout/

Recommendation endpoint
-----------------------
  GET    /api/recommendations/<customer_external_id>/
         ?n=<int>   (default 10, max 100)
"""

from django.urls import path

from store.views import (
    ApiRootView,
    CartAddItemView,
    CartCheckoutView,
    CartRemoveItemView,
    CartUpdateItemView,
    CartView,
    CategoryDetailView,
    CategoryListView,
    ProductDetailView,
    ProductListView,
    RecommendationListView,
)

urlpatterns = [
    # API Root
    path("", ApiRootView.as_view(), name="api-root"),

    # Categories
    path("categories/", CategoryListView.as_view(), name="category-list"),
    path("categories/<int:pk>/", CategoryDetailView.as_view(), name="category-detail"),

    # Products
    path("products/", ProductListView.as_view(), name="product-list"),
    path(
        "products/<str:lookup_value>/",
        ProductDetailView.as_view(),
        name="product-detail",
    ),

    # Cart
    path(
        "cart/<str:customer_external_id>/",
        CartView.as_view(),
        name="cart-detail",
    ),
    path(
        "cart/<str:customer_external_id>/add/",
        CartAddItemView.as_view(),
        name="cart-add-item",
    ),
    path(
        "cart/<str:customer_external_id>/update/<int:item_id>/",
        CartUpdateItemView.as_view(),
        name="cart-update-item",
    ),
    path(
        "cart/<str:customer_external_id>/remove/<int:item_id>/",
        CartRemoveItemView.as_view(),
        name="cart-remove-item",
    ),
    path(
        "cart/<str:customer_external_id>/checkout/",
        CartCheckoutView.as_view(),
        name="cart-checkout",
    ),

    # Recommendations (read-only)
    path(
        "recommendations/<str:customer_external_id>/",
        RecommendationListView.as_view(),
        name="recommendations-list",
    ),
]
