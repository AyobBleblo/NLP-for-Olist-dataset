"""
store/management/commands/seed_test_scenario.py
===============================================
Management command: python manage.py seed_test_scenario

Sets up a complete 3-customer scenario to test Collaborative Filtering:
  1. Customer 1 (Home Decor & Furniture Fan) — Buys 4 Furniture/Decor products
  2. Customer 2 (Sports & Outdoor Enthusiast) — Buys 4 Sports/Leisure products
  3. Customer 3 (Computers & Technology Geek) — Buys 4 Computer/Tech products
"""

from datetime import datetime, timezone

from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.db import transaction

from store.models import Customer, Order, OrderItem, Product, Recommendation


class Command(BaseCommand):
    help = "Seed 3 test customers with 4 purchases each and compute their recommendations."

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("\nSetting up 3-Customer Recommendation Scenario ...\n"))

        personas = [
            {
                "id": "c37cc6c1a59d81460a3059744f7ada1c",
                "name": "Alice Silva",
                "city": "Sao Paulo",
                "state": "SP",
                "persona": "Home Decor & Furniture Fan",
                "category_slug": "moveis_decoracao",
            },
            {
                "id": "3e2157f91502458bc58455fd798ed58a",
                "name": "Bruno Santos",
                "city": "Rio de Janeiro",
                "state": "RJ",
                "persona": "Sports & Outdoor Enthusiast",
                "category_slug": "esporte_lazer",
            },
            {
                "id": "feb2a9889d236875c3510880bf9576f3",
                "name": "Carlos Costa",
                "city": "Curitiba",
                "state": "PR",
                "persona": "Computers & Technology Geek",
                "category_slug": "informatica_acessorios",
            },
        ]

        created_customers = []

        with transaction.atomic():
            for p in personas:
                # 1. Create Customer
                cust, created = Customer.objects.update_or_create(
                    external_id=p["id"],
                    defaults={"city": p["city"], "state": p["state"]},
                )
                created_customers.append((cust, p))

                # 2. Pick 4 products for this persona
                products = list(
                    Product.objects.filter(category__name=p["category_slug"]).select_related("category")[:4]
                )

                if len(products) < 4:
                    products = list(Product.objects.all()[:4])

                # 3. Create an Order
                order_ext = f"ord_{p['id'][:8]}"
                order, _ = Order.objects.update_or_create(
                    external_id=order_ext,
                    customer=cust,
                    defaults={
                        "purchased_at": datetime.now(tz=timezone.utc),
                        "status": "delivered",
                    },
                )

                # 4. Create 4 OrderItems
                for item in products:
                    OrderItem.objects.get_or_create(
                        order=order,
                        product=item,
                        defaults={"price": item.price},
                    )
                    # Increment total_purchases
                    Product.objects.filter(pk=item.pk).update(
                        total_purchases=item.total_purchases + 1
                    )

        self.stdout.write(self.style.SUCCESS("[OK] 3 Customers & 12 Orders/Items successfully created in database."))

        # ------------------------------------------------------------------
        # 5. Compute Recommendations for all 3 customers
        # ------------------------------------------------------------------
        self.stdout.write(self.style.MIGRATE_HEADING("\nComputing Collaborative Filtering Recommendations ..."))
        call_command("compute_recommendations", top_n=5, clear=True)

        # ------------------------------------------------------------------
        # 6. Display Scenario Results
        # ------------------------------------------------------------------
        self.stdout.write(self.style.MIGRATE_HEADING("\n" + "=" * 80))
        self.stdout.write(self.style.MIGRATE_HEADING("   TEST SCENARIO RESULTS (COLLABORATIVE FILTERING IN ACTION)"))
        self.stdout.write(self.style.MIGRATE_HEADING("=" * 80 + "\n"))

        for cust, p in created_customers:
            self.stdout.write(
                self.style.SUCCESS(f"CUSTOMER: {p['name']} ({p['persona']})")
            )
            self.stdout.write(f"   ID: {cust.external_id} | Location: {cust.city}, {cust.state}")

            # Purchased items
            purchased_items = OrderItem.objects.filter(order__customer=cust).select_related("product__category")
            self.stdout.write("\n   PURCHASE HISTORY (4 products bought):")
            for oi in purchased_items:
                self.stdout.write(
                    f"      - {oi.product.name_translated} (${oi.price}) | Category: {oi.product.category.name_translated}"
                )

            # Recommendations
            recs = Recommendation.objects.filter(customer=cust).select_related("product__category").order_by("-score")
            self.stdout.write("\n   TOP 5 RECOMMENDATIONS GENERATED BY SVD:")
            for rank, r in enumerate(recs, 1):
                self.stdout.write(
                    f"      {rank}. {r.product.name_translated} (${r.product.price})"
                    f"\n         Predicted Score: {r.score:.4f} | Category: {r.product.category.name_translated}"
                )

            # API link
            self.stdout.write(
                self.style.WARNING(
                    f"\n   Live API Endpoint:\n"
                    f"      http://127.0.0.1:8000/api/recommendations/{cust.external_id}/?n=5\n"
                )
            )
            self.stdout.write("-" * 80 + "\n")
