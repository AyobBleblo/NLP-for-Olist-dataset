"""
store/admin.py — Register models with Django admin.
"""

from django.contrib import admin

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


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ["id", "name", "name_translated"]
    search_fields = ["name", "name_translated"]


@admin.register(Customer)
class CustomerAdmin(admin.ModelAdmin):
    list_display = ["id", "external_id", "city", "state"]
    search_fields = ["external_id", "city"]


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "external_id",
        "name_translated",
        "category",
        "price",
        "avg_sentiment_score",
        "total_purchases",
    ]
    list_filter = ["category"]
    search_fields = ["external_id", "name", "name_translated"]
    readonly_fields = ["avg_sentiment_score", "total_purchases", "image_url"]


@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ["id", "external_id", "customer", "status", "purchased_at"]
    list_filter = ["status"]
    search_fields = ["external_id"]


@admin.register(OrderItem)
class OrderItemAdmin(admin.ModelAdmin):
    list_display = ["id", "order", "product", "price"]
    search_fields = ["order__external_id", "product__external_id"]


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "external_id",
        "product",
        "rating",
        "sentiment",
        "sentiment_processed_at",
    ]
    list_filter = ["sentiment", "rating"]
    search_fields = ["external_id", "comment_text"]
    readonly_fields = ["sentiment", "sentiment_processed_at", "comment_text_translated"]


class CartItemInline(admin.TabularInline):
    model = CartItem
    extra = 0


@admin.register(Cart)
class CartAdmin(admin.ModelAdmin):
    list_display = ["id", "customer", "created_at", "updated_at"]
    inlines = [CartItemInline]


@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = ["id", "customer", "product", "score", "generated_at"]
    list_filter = ["generated_at"]
    search_fields = ["customer__external_id", "product__external_id"]
    readonly_fields = ["score", "generated_at"]
