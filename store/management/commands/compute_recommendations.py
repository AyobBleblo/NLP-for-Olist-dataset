"""
store/management/commands/compute_recommendations.py
====================================================
Management command: python manage.py compute_recommendations

Computes Collaborative Filtering recommendations using the SVD model
(trained in Google Colab / scikit-surprise and saved at models/svd_cf.pkl).
If the SVD model is not yet placed in models/, it gracefully falls back to
the Popularity baseline (ranked by total_purchases), with an optional
NLP review sentiment boost.

Algorithm (v2 — diversity-aware hybrid)
----------------------------------------
For each candidate product:

  1. BASE SCORE (SVD decomposition — no svd.predict() call):
       score = BASE + qi·pu + ALPHA * bi
     where BASE=3.0, ALPHA=0.5 are module-level constants (overridable via
     settings.RECSYS_ALPHA / settings.RECSYS_BASE).  Using ALPHA<1 dampens
     the product-bias term so user taste (qi·pu) has relative weight.

  2. COLD-START / UNKNOWN PRODUCT FALLBACK:
     - Unknown user (not in trainset): pu=zero vector, bu=0, treated same.
     - Unknown product (not in trainset): popularity fallback score
         1.0 + min(total_purchases / 10, 2.0)
       This never exceeds ~3.0, so it won't outrank any known SVD score.

  3. CATEGORY AFFINITY BOOST (content-based):
       score += BETA * affinity(category)
     affinity = fraction of customer's purchase history in that category.
     BETA default 1.0, overridable via settings.RECSYS_BETA.

  4. SENTIMENT BOOST (only when product has reviews):
       score += GAMMA * (avg_sentiment_score - 0.5)
     GAMMA default 0.3.  Products with no reviews (avg_sentiment_score=0.0
     AND review_count=0) are skipped — 0.0 is NOT treated as negative.

  5. DIVERSITY (per category cap):
     After sorting by score, picks top-N enforcing at most MAX_PER_CAT
     products per category (default 2). Remaining slots are filled from
     the leftover ranked list.

Usage
-----
  python manage.py compute_recommendations
  python manage.py compute_recommendations --customer-id <external_id>
  python manage.py compute_recommendations --top-n 10
  python manage.py compute_recommendations --clear
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

from store.ml.loader import get_cf_model
from store.models import Customer, OrderItem, Product, Recommendation

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level tunable constants — all overridable via Django settings
# ---------------------------------------------------------------------------

#: Weight applied to the product-bias term (bi).  Values < 1.0 reduce the
#: dominance of popular-item bias and let the latent dot product qi·pu steer.
ALPHA: float = float(getattr(settings, "RECSYS_ALPHA", 0.5))

#: Additive baseline so SVD scores land in a reasonable absolute range.
BASE: float = float(getattr(settings, "RECSYS_BASE", 3.0))

#: Weight for the category-affinity content-based boost.
BETA: float = float(getattr(settings, "RECSYS_BETA", 1.0))

#: Weight for the sentiment boost/penalty.
GAMMA: float = float(getattr(settings, "RECSYS_GAMMA", 0.3))

#: Maximum products from a single category in any top-N list (diversity cap).
MAX_PER_CAT: int = int(getattr(settings, "RECSYS_MAX_PER_CAT", 2))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_category_affinity(customer: Customer) -> dict[int | None, float]:
    """
    Compute the customer's per-category purchase distribution.

    Returns a dict mapping category_id -> affinity in [0, 1], where affinity
    is the fraction of the customer's purchases belonging to that category.
    The dict includes a None key for products with no category.

    This is precomputed once per customer and passed into the scoring loop —
    no per-product DB queries inside the scoring loop.
    """
    items = (
        OrderItem.objects.filter(order__customer=customer)
        .select_related("product__category")
        .values_list("product__category_id", flat=True)
    )
    counts: dict[int | None, int] = defaultdict(int)
    total = 0
    for cat_id in items:
        counts[cat_id] += 1
        total += 1

    if total == 0:
        return {}

    return {cat_id: count / total for cat_id, count in counts.items()}


def _apply_diversity_cap(
    ranked: list[tuple[int, float]],
    product_category_map: dict[int, int | None],
    top_n: int,
    max_per_cat: int,
) -> list[tuple[int, float]]:
    """
    Pick top-N from *ranked* (already sorted DESC by score) while enforcing
    at most *max_per_cat* products per category.  Remaining slots are filled
    from the leftover tail in rank order, also respecting the cap.

    Two-pass algorithm:
      Pass 1: greedily fill result up to top_n, skipping items that would
              exceed the per-category cap (they go to leftover).
      Pass 2: if result < top_n, scan leftover in rank order; items whose
              category still has room are added (cap is re-applied).

    Args:
        ranked: (product_id, score) pairs sorted descending.
        product_category_map: product_id -> category_id (or None).
        top_n: desired output length.
        max_per_cat: max products from any single category.

    Returns:
        List of (product_id, score) pairs of length min(top_n, len(ranked)).
    """
    result: list[tuple[int, float]] = []
    cat_count: dict[int | None, int] = defaultdict(int)
    leftover: list[tuple[int, float]] = []

    # Pass 1 — respect cap strictly
    for prod_id, score in ranked:
        if len(result) >= top_n:
            break
        cat_id = product_category_map.get(prod_id)
        if cat_count[cat_id] < max_per_cat:
            result.append((prod_id, score))
            cat_count[cat_id] += 1
        else:
            leftover.append((prod_id, score))

    # Pass 2 — fill remaining slots from leftover, still respecting cap
    if len(result) < top_n:
        for prod_id, score in leftover:
            if len(result) >= top_n:
                break
            cat_id = product_category_map.get(prod_id)
            if cat_count[cat_id] < max_per_cat:
                result.append((prod_id, score))
                cat_count[cat_id] += 1

    return result


# ---------------------------------------------------------------------------
# Public scoring interface
# ---------------------------------------------------------------------------


def compute_recommendations_for_customer(
    customer: Customer,
    candidate_products: list[Product],
    svd_model=None,
    category_affinity: dict[int | None, float] | None = None,
    product_category_map: dict[int, int | None] | None = None,
) -> list[tuple[int, float]]:
    """
    Score unpurchased candidate products for a customer.

    Algorithm (hybrid):
      1. SVD decomposition score: BASE + qi·pu + ALPHA*bi
         - Unknown user: pu=zero, bu=0 (cold-start, category affinity carries the load)
         - Unknown product: popularity fallback (1.0 + min(purchases/10, 2.0))
      2. Category affinity boost: BETA * affinity(product.category)
      3. Sentiment boost/penalty: GAMMA * (avg_sentiment_score - 0.5)
         (only applied when the product has at least one review)

    Args:
        customer: The Customer ORM object.
        candidate_products: Unpurchased Product objects to score.
        svd_model: Loaded surprise.SVD object (or None for popularity fallback).
        category_affinity: Precomputed {category_id: affinity} dict (optional,
            computed on-demand if not supplied).
        product_category_map: Precomputed {product_id: category_id} dict (optional).
            Used after scoring for diversity; not needed inside this function.

    Returns:
        List of (product_db_id, score) tuples, sorted descending by score.
    """
    # ------------------------------------------------------------------
    # Resolve category affinity (precomputed preferred to avoid N DB hits)
    # ------------------------------------------------------------------
    if category_affinity is None:
        category_affinity = _build_category_affinity(customer)

    # ------------------------------------------------------------------
    # SVD model components (if available)
    # ------------------------------------------------------------------
    ts = None
    pu: np.ndarray | None = None
    bu: float = 0.0

    if svd_model is not None:
        ts = svd_model.trainset
        try:
            inner_uid = ts.to_inner_uid(customer.external_id)
            pu = svd_model.pu[inner_uid]
            bu = float(svd_model.bu[inner_uid])
        except ValueError:
            # Cold-start user: pu stays None, bu stays 0
            pu = None
            bu = 0.0

    scores: list[tuple[int, float]] = []

    for prod in candidate_products:
        # ----------------------------------------------------------------
        # 1. Base SVD score (or popularity fallback for unknown products)
        # ----------------------------------------------------------------
        if svd_model is not None and ts is not None:
            try:
                inner_iid = ts.to_inner_iid(prod.external_id)
                bi = float(svd_model.bi[inner_iid])
                if pu is not None:
                    qi_pu = float(svd_model.qi[inner_iid] @ pu)
                else:
                    qi_pu = 0.0
                base_score = BASE + qi_pu + ALPHA * bi
            except ValueError:
                # Unknown product — popularity fallback (capped below known SVD range)
                base_score = 1.0 + min(float(prod.total_purchases) / 10.0, 2.0)
        else:
            # No SVD model at all — pure popularity baseline (scaled to 1–5)
            base_score = 1.0 + min(float(prod.total_purchases) / 10.0, 4.0)

        # ----------------------------------------------------------------
        # 2. Category affinity boost
        # ----------------------------------------------------------------
        cat_id = prod.category_id  # FK integer, avoids select_related fetch
        affinity = category_affinity.get(cat_id, 0.0)
        affinity_boost = BETA * affinity

        # ----------------------------------------------------------------
        # 3. Sentiment boost/penalty (skip products with no reviews)
        # ----------------------------------------------------------------
        sentiment_boost = 0.0
        if prod.total_purchases > 0 and prod.avg_sentiment_score > 0.0:
            # avg_sentiment_score in [0, 1]; centre at 0.5 so neutral=no effect
            sentiment_boost = GAMMA * (prod.avg_sentiment_score - 0.5)

        final_score = base_score + affinity_boost + sentiment_boost
        scores.append((prod.pk, round(final_score, 4)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def compute_recommendations(
    customer_db_id: int,
    all_product_ids: list[int],
    purchased_product_ids: set[int],
) -> list[tuple[int, float]]:
    """
    Backwards-compatible interface for external callers.

    Computes SVD/popularity recommendations for a single customer, excluding
    already-purchased products.  No diversity cap or bulk persistence — callers
    receive the raw ranked list.

    Args:
        customer_db_id: PK of the Customer row.
        all_product_ids: Pool of candidate product PKs.
        purchased_product_ids: Set of already-purchased product PKs to exclude.

    Returns:
        List of (product_id, score) tuples sorted descending.
    """
    try:
        customer = Customer.objects.get(pk=customer_db_id)
    except Customer.DoesNotExist:
        return []

    candidate_ids = [pid for pid in all_product_ids if pid not in purchased_product_ids]
    candidates = list(Product.objects.filter(id__in=candidate_ids))
    svd_model = get_cf_model()
    return compute_recommendations_for_customer(customer, candidates, svd_model)


# ===========================================================================
# Management command
# ===========================================================================


class Command(BaseCommand):
    help = (
        "Compute Collaborative Filtering recommendations (SVD / Popularity) "
        "and write them to the Recommendation table."
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

    def handle(self, *args, **options):  # noqa: C901
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

        if not customers:
            self.stdout.write(
                self.style.WARNING(
                    "No customers found in database. Create or import customers first."
                )
            )
            return

        self.stdout.write(f"Processing {len(customers)} customer(s) ...")

        if clear:
            deleted, _ = Recommendation.objects.all().delete()
            self.stdout.write(f"  Cleared {deleted} existing recommendation rows.")

        # ------------------------------------------------------------------
        # 2. Check SVD Collaborative Filtering Model
        # ------------------------------------------------------------------
        svd_model = get_cf_model()
        if svd_model is not None:
            self.stdout.write(
                self.style.SUCCESS("  [OK] SVD Collaborative Filtering model loaded successfully.")
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    f"  [INFO] SVD model file not found at '{getattr(settings, 'CF_MODEL_PATH', 'models/svd_cf.pkl')}'.\n"
                    "     Operating in Popularity baseline mode until svd_cf.pkl is placed in models/."
                )
            )

        # ------------------------------------------------------------------
        # 3. Pre-load all products (once, with category FK pre-fetched)
        # ------------------------------------------------------------------
        all_products = list(Product.objects.select_related("category").all())
        self.stdout.write(f"  {len(all_products)} products in catalogue.")

        # Precompute product -> category_id map (avoids attribute access in loops)
        product_category_map: dict[int, int | None] = {
            p.pk: p.category_id for p in all_products
        }

        # ------------------------------------------------------------------
        # 4. Build purchase history: customer_db_id -> set of product_db_ids
        #    Also build per-customer category purchase counts for affinity.
        # ------------------------------------------------------------------
        self.stdout.write("  Building purchase history & category affinity ...")

        # category affinity: customer_pk -> {category_id: count}
        raw_cat_counts: dict[int, dict[int | None, int]] = defaultdict(lambda: defaultdict(int))
        purchase_history: dict[int, set[int]] = defaultdict(set)

        for item in (
            OrderItem.objects
            .select_related("order", "product")
            .values("order__customer_id", "product_id", "product__category_id")
        ):
            cid = item["order__customer_id"]
            pid = item["product_id"]
            cat_id = item["product__category_id"]
            purchase_history[cid].add(pid)
            raw_cat_counts[cid][cat_id] += 1

        # Normalise category counts -> affinity in [0, 1]
        category_affinity_map: dict[int, dict[int | None, float]] = {}
        for cid, cat_counts in raw_cat_counts.items():
            total = sum(cat_counts.values())
            if total > 0:
                category_affinity_map[cid] = {
                    cat_id: cnt / total for cat_id, cnt in cat_counts.items()
                }
            else:
                category_affinity_map[cid] = {}

        # ------------------------------------------------------------------
        # 5. Compute and persist recommendations — one transaction per customer
        # ------------------------------------------------------------------
        now = datetime.now(tz=timezone.utc)
        total_written = 0

        for customer in customers:
            try:
                purchased = purchase_history.get(customer.pk, set())
                candidates = [p for p in all_products if p.pk not in purchased]
                affinity = category_affinity_map.get(customer.pk, {})

                # Score all candidates
                scored = compute_recommendations_for_customer(
                    customer=customer,
                    candidate_products=candidates,
                    svd_model=svd_model,
                    category_affinity=affinity,
                    product_category_map=product_category_map,
                )

                # Apply diversity cap to get final top-N
                top_scored = _apply_diversity_cap(
                    ranked=scored,
                    product_category_map=product_category_map,
                    top_n=top_n,
                    max_per_cat=MAX_PER_CAT,
                )

                # Persist: delete existing rows then bulk_create
                with transaction.atomic():
                    Recommendation.objects.filter(customer=customer).delete()
                    Recommendation.objects.bulk_create(
                        [
                            Recommendation(
                                customer=customer,
                                product_id=product_id,
                                score=score,
                                generated_at=now,
                            )
                            for product_id, score in top_scored
                        ],
                        ignore_conflicts=False,
                    )

                total_written += len(top_scored)

            except Exception:
                logger.exception(
                    "Unexpected error computing recommendations for customer pk=%s (%s). Skipping.",
                    customer.pk,
                    customer.external_id,
                )

        self.stdout.write(
            self.style.SUCCESS(
                f"\n[OK] Complete: Wrote {total_written} recommendation rows for "
                f"{len(customers)} customer(s)."
            )
        )
