"""
store/management/commands/compute_recommendations.py
====================================================
Management command: python manage.py compute_recommendations

Reads OrderItem data and writes (customer, product, score) rows into the
Recommendation table.

PLACEHOLDER NOTICE
------------------
The actual scoring algorithm is intentionally left as a stub.
The interface is clearly defined:

    compute_recommendations(customer_id: int) -> list[tuple[int, float]]

Returns a list of (product_db_id, score) pairs.  Replace the placeholder
body with your chosen algorithm (collaborative filtering, ALS, association
rules, etc.) without changing anything else in this file.

Usage
-----
  python manage.py compute_recommendations
  python manage.py compute_recommendations --customer-id <external_id>
  python manage.py compute_recommendations --top-n 10
"""

import logging
from datetime import datetime, timezone

from django.core.management.base import BaseCommand
from django.db import transaction

from store.models import Customer, OrderItem, Product, Recommendation

logger = logging.getLogger(__name__)


# ===========================================================================
# *** PLACEHOLDER — replace this function with your recommendation algorithm ***
# ===========================================================================

def compute_recommendations(
    customer_db_id: int,
    all_product_ids: list[int],
    purchased_product_ids: set[int],
) -> list[tuple[int, float]]:
    """
    Compute (product_db_id, score) pairs for a single customer.

    CURRENT IMPLEMENTATION: stub — assigns a score of 0.0 to every
    product NOT already purchased by this customer.

    REPLACE WITH:
    - Collaborative filtering (e.g. ALS via implicit / surprise)
    - Association rules (e.g. mlxtend Apriori)
    - Content-based filtering on Product features
    - Hybrid approach

    Args:
        customer_db_id:     Django PK of the Customer row.
        all_product_ids:    All product PKs in the catalogue.
        purchased_product_ids: Set of product PKs already bought by this
                               customer (to exclude from recommendations).

    Returns:
        List of (product_db_id, score) tuples, ordered DESC by score.
        The caller will take the top-N from this list.
    """
    # -----------------------------------------------------------------------
    # TODO: implement real scoring here
    # -----------------------------------------------------------------------
    candidates = [
        pid for pid in all_product_ids if pid not in purchased_product_ids
    ]
    # Placeholder: uniform 0.0 score for every candidate
    return [(pid, 0.0) for pid in candidates]


# ===========================================================================
# Management command
# ===========================================================================

class Command(BaseCommand):
    help = (
        "Compute purchase-based recommendations and write them to the "
        "Recommendation table (placeholder algorithm — replace the "
        "compute_recommendations() function body to add real scoring)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--customer-id",
            type=str,
            default=None,
            help="Process a single customer by external_id (default: all).",
        )
        parser.add_argument(
            "--top-n",
            type=int,
            default=20,
            help="Number of top recommendations to store per customer (default: 20).",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            default=False,
            help="Delete existing Recommendation rows before recomputing.",
        )

    def handle(self, *args, **options):
        customer_external_id: str | None = options["customer_id"]
        top_n: int = options["top_n"]
        clear: bool = options["clear"]

        # ------------------------------------------------------------------
        # 1. Resolve customer queryset
        # ------------------------------------------------------------------
        if customer_external_id:
            try:
                customers = [Customer.objects.get(external_id=customer_external_id)]
            except Customer.DoesNotExist:
                self.stderr.write(
                    self.style.ERROR(
                        f"Customer with external_id='{customer_external_id}' not found."
                    )
                )
                return
        else:
            customers = list(Customer.objects.all())

        self.stdout.write(f"Processing {len(customers)} customer(s) …")

        if clear:
            deleted, _ = Recommendation.objects.all().delete()
            self.stdout.write(f"  Cleared {deleted} existing recommendation rows.")

        # ------------------------------------------------------------------
        # 2. Pre-load all product IDs (once)
        # ------------------------------------------------------------------
        all_product_ids: list[int] = list(
            Product.objects.values_list("id", flat=True)
        )
        self.stdout.write(f"  {len(all_product_ids)} products in catalogue.")

        # ------------------------------------------------------------------
        # 3. Build purchase history: customer_db_id → set of product_db_ids
        # ------------------------------------------------------------------
        self.stdout.write("  Building purchase history …")
        purchase_history: dict[int, set[int]] = {}
        for item in OrderItem.objects.select_related("order").values(
            "order__customer_id", "product_id"
        ):
            cid = item["order__customer_id"]
            pid = item["product_id"]
            if cid not in purchase_history:
                purchase_history[cid] = set()
            purchase_history[cid].add(pid)

        # ------------------------------------------------------------------
        # 4. Compute and upsert recommendations
        # ------------------------------------------------------------------
        now = datetime.now(tz=timezone.utc)
        total_written = 0

        with transaction.atomic():
            for customer in customers:
                purchased = purchase_history.get(customer.pk, set())

                scored = compute_recommendations(
                    customer_db_id=customer.pk,
                    all_product_ids=all_product_ids,
                    purchased_product_ids=purchased,
                )

                # Take top-N by score
                scored.sort(key=lambda x: x[1], reverse=True)
                top_scored = scored[:top_n]

                for product_id, score in top_scored:
                    Recommendation.objects.update_or_create(
                        customer=customer,
                        product_id=product_id,
                        defaults={"score": score, "generated_at": now},
                    )
                total_written += len(top_scored)

        self.stdout.write(
            self.style.SUCCESS(
                f"\n✓ Wrote {total_written} recommendation rows for "
                f"{len(customers)} customer(s)."
            )
        )
        self.stdout.write(
            self.style.WARNING(
                "\n⚠  NOTE: compute_recommendations() is currently a STUB "
                "(all scores = 0.0).  Replace the function body in "
                "store/management/commands/compute_recommendations.py "
                "with your actual algorithm."
            )
        )
