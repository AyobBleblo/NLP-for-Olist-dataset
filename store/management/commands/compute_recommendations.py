"""
store/management/commands/compute_recommendations.py
====================================================
Management command: python manage.py compute_recommendations

Computes hybrid recommendations combining:
  1. Collaborative Filtering — SVD model (scikit-surprise, models/svd_cf.pkl)
  2. Content-Based Filtering — TF-IDF + numeric similarity from Colab Part 3
     (models/processed/tfidf_matrix.npz + products_features.parquet +
      tfidf_row_index.parquet)
  3. NLP Sentiment boost  — avg_sentiment_score from DistilBERT reviews
  4. Diversity cap        — at most MAX_PER_CAT products per category

Algorithm (v3 — true CBF hybrid)
----------------------------------
For each candidate product p and customer u:

  1. SVD SCORE:
       svd_score = BASE + qi·pu + ALPHA * bi
     Cold-start user  → pu=zero
     Unknown product  → popularity fallback (≤ 3.0)

  2. CBF SCORE (if models/processed/ artifacts are loaded):
       cbf_score = text_weight * cosine(tfidf_matrix[p], user_profile_text)
                 + (1 - text_weight) * numeric_sim(num_matrix[p], user_profile_num)
     user_profile_text = mean TF-IDF row over purchased products
     user_profile_num  = mean numeric row over purchased products
     numeric_sim = 1 / (1 + scaled_euclidean)
     Falls back to category-affinity boost if artifacts missing.

  3. SENTIMENT BOOST:
       score += GAMMA * (avg_sentiment_score - 0.5)   [only if reviews exist]

  4. FINAL SCORE:
       score = SVD_WEIGHT * svd_score + CBF_WEIGHT * cbf_score + sentiment_boost

  5. DIVERSITY CAP:
     At most MAX_PER_CAT (default 2) products from any single category.

Constants (overridable via settings.RECSYS_*):
  ALPHA=0.5, BASE=3.0, SVD_WEIGHT=0.6, CBF_WEIGHT=0.4, GAMMA=0.3, MAX_PER_CAT=2

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

import scipy.sparse as _sp
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

import numpy as np
from store.ml.loader import get_cbf_model, get_cf_model
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

#: Blend weights for CF vs CBF signal.
SVD_WEIGHT: float = float(getattr(settings, "RECSYS_SVD_WEIGHT", 0.6))
CBF_WEIGHT: float = float(getattr(settings, "RECSYS_CBF_WEIGHT", 0.4))

#: Legacy category-affinity weight (used when CBF artifacts are missing).
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
    Fallback CBF: compute the customer's per-category purchase distribution.
    Used when CBF TF-IDF artifacts are not available.
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


def _build_cbf_user_profile(
    purchased_external_ids: list[str],
    cbf: dict,
) -> tuple["np.ndarray | None", "np.ndarray | None"]:
    """
    Build a user content profile as the mean of purchased products' TF-IDF
    and numeric vectors.  Mirrors notebook cell 75 ContentBasedRecommender.

    Args:
        purchased_external_ids: list of product external_ids the user bought.
        cbf: dict returned by get_cbf_model().

    Returns:
        (text_profile, num_profile) as numpy arrays, or (None, None) if no
        purchased product is in the CBF index.
    """
    pid_to_idx = cbf["pid_to_idx"]
    tfidf_matrix = cbf["tfidf_matrix"]
    num_matrix = cbf["num_matrix"]

    idx_list = [pid_to_idx[eid] for eid in purchased_external_ids if eid in pid_to_idx]
    if not idx_list:
        return None, None

    # Text profile: mean of sparse TF-IDF rows, then L2-normalise
    text_sum = tfidf_matrix[idx_list].mean(axis=0)          # dense (1, vocab)
    text_arr = np.asarray(text_sum)                          # shape (1, vocab)
    norm = np.linalg.norm(text_arr)
    text_profile = (text_arr / norm) if norm > 0 else text_arr  # shape (1, vocab)

    # Numeric profile: mean of numeric rows
    num_profile = num_matrix[idx_list].mean(axis=0)          # shape (5,)

    return text_profile, num_profile


def _cbf_scores_all(
    cbf: dict,
    text_profile: "np.ndarray",
    num_profile: "np.ndarray",
) -> "np.ndarray":
    """
    Compute similarity of every product in the CBF index to the user profile.
    Mirrors notebook cb_scores_for_vectors().

    score[i] = text_weight * cosine(tfidf[i], text_profile)
             + (1 - text_weight) * 1 / (1 + scaled_euclidean(num[i], num_profile))
    """
    import numpy as np
    tfidf_matrix = cbf["tfidf_matrix"]
    num_matrix   = cbf["num_matrix"]
    tw = cbf["text_weight"]

    # Cosine similarity (TF-IDF rows are already L2-normalised in the notebook)
    text_sim = np.asarray(tfidf_matrix.dot(text_profile.T)).ravel()  # shape (n,)

    # Numeric similarity: 1 / (1 + scaled Euclidean)
    diff = num_matrix - num_profile          # broadcasting (n, 5)
    d = np.linalg.norm(diff, axis=1) / np.sqrt(num_matrix.shape[1])
    num_sim = 1.0 / (1.0 + d)

    return tw * text_sim + (1 - tw) * num_sim


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
    cbf_model: dict | None = None,
    cbf_all_scores: "np.ndarray | None" = None,
    cbf_pid_to_idx: dict | None = None,
) -> list[tuple[int, float]]:
    """
    Score unpurchased candidate products for a customer.

    Algorithm (v3 — true CBF hybrid):
      1. SVD score: BASE + qi·pu + ALPHA*bi
         (cold-start: pu=zero; unknown product: popularity fallback ≤ 3.0)
      2. CBF score: text_weight*cosine(TF-IDF) + (1-text_weight)*numeric_sim
         Built from mean of purchased products' feature vectors (notebook Part 3).
         Falls back to BETA * category_affinity if CBF artifacts not loaded.
      3. Sentiment boost: GAMMA * (avg_sentiment_score - 0.5)
      4. Final: SVD_WEIGHT * svd_score + CBF_WEIGHT * cbf_score + sentiment_boost

    Args:
        customer:            The Customer ORM object.
        candidate_products:  Unpurchased Product objects to score.
        svd_model:           Loaded surprise.SVD object (or None).
        category_affinity:   Fallback {category_id: affinity} (used when no CBF).
        product_category_map: {product_id: category_id} for diversity cap.
        cbf_model:           Dict from get_cbf_model() (or None).
        cbf_all_scores:      Pre-computed CBF similarity array over all CBF products.
        cbf_pid_to_idx:      pid_to_idx from the CBF model.

    Returns:
        List of (product_db_id, score) tuples, sorted descending by score.
    """
    import numpy as np

    # ------------------------------------------------------------------
    # SVD model components
    # ------------------------------------------------------------------
    ts = None
    pu: np.ndarray | None = None

    if svd_model is not None:
        ts = svd_model.trainset
        try:
            inner_uid = ts.to_inner_uid(customer.external_id)
            pu = svd_model.pu[inner_uid]
        except ValueError:
            pu = None  # cold-start user

    # ------------------------------------------------------------------
    # CBF available?
    # ------------------------------------------------------------------
    use_cbf = cbf_model is not None and cbf_all_scores is not None

    # Fallback: category affinity (only needed when CBF unavailable)
    if not use_cbf and category_affinity is None:
        category_affinity = _build_category_affinity(customer)

    scores: list[tuple[int, float]] = []

    for prod in candidate_products:
        # ----------------------------------------------------------------
        # 1. SVD score
        # ----------------------------------------------------------------
        if svd_model is not None and ts is not None:
            try:
                inner_iid = ts.to_inner_iid(prod.external_id)
                bi = float(svd_model.bi[inner_iid])
                qi_pu = float(svd_model.qi[inner_iid] @ pu) if pu is not None else 0.0
                svd_score = BASE + qi_pu + ALPHA * bi
            except ValueError:
                svd_score = 1.0 + min(float(prod.total_purchases) / 10.0, 2.0)
        else:
            svd_score = 1.0 + min(float(prod.total_purchases) / 10.0, 4.0)

        # ----------------------------------------------------------------
        # 2. CBF score (real TF-IDF) or category-affinity fallback
        # ----------------------------------------------------------------
        if use_cbf:
            idx = cbf_pid_to_idx.get(prod.external_id)
            cbf_score = float(cbf_all_scores[idx]) if idx is not None else 0.0
        else:
            cat_id = prod.category_id
            affinity = (category_affinity or {}).get(cat_id, 0.0)
            cbf_score = BETA * affinity

        # ----------------------------------------------------------------
        # 3. Sentiment boost (only products with reviews)
        # ----------------------------------------------------------------
        sentiment_boost = 0.0
        if prod.total_purchases > 0 and prod.avg_sentiment_score > 0.0:
            sentiment_boost = GAMMA * (prod.avg_sentiment_score - 0.5)

        # ----------------------------------------------------------------
        # 4. Combine
        # ----------------------------------------------------------------
        final_score = SVD_WEIGHT * svd_score + CBF_WEIGHT * cbf_score + sentiment_boost
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
        # 3. Load CBF model (TF-IDF from models/processed/)
        # ------------------------------------------------------------------
        cbf_model = get_cbf_model()
        if cbf_model is not None:
            self.stdout.write(
                self.style.SUCCESS(
                    "  [OK] CBF TF-IDF model loaded (%d products in index)." % len(cbf_model["pid_to_idx"])
                )
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    "  [INFO] CBF artifacts not found in models/processed/. "
                    "Using category-affinity fallback for content signal."
                )
            )

        # ------------------------------------------------------------------
        # 4. Pre-load all products (once, with category FK pre-fetched)
        # ------------------------------------------------------------------
        all_products = list(Product.objects.select_related("category").all())
        self.stdout.write(f"  {len(all_products)} products in catalogue.")

        product_category_map: dict[int, int | None] = {
            p.pk: p.category_id for p in all_products
        }
        # Map external_id -> db pk (for CBF lookup)
        ext_to_pk: dict[str, int] = {p.external_id: p.pk for p in all_products}

        # ------------------------------------------------------------------
        # 5. Build purchase history + category affinity (fallback only)
        # ------------------------------------------------------------------
        self.stdout.write("  Building purchase history ...")

        raw_cat_counts: dict[int, dict[int | None, int]] = defaultdict(lambda: defaultdict(int))
        # purchase_history: cust_pk -> set of product PKs
        purchase_history: dict[int, set[int]] = defaultdict(set)
        # purchase_ext_ids: cust_pk -> list of product external_ids (for CBF)
        purchase_ext_ids: dict[int, list[str]] = defaultdict(list)

        for item in (
            OrderItem.objects
            .select_related("order", "product")
            .values("order__customer_id", "product_id",
                    "product__category_id", "product__external_id")
        ):
            cid = item["order__customer_id"]
            pid = item["product_id"]
            cat_id = item["product__category_id"]
            ext_id = item["product__external_id"]
            purchase_history[cid].add(pid)
            raw_cat_counts[cid][cat_id] += 1
            purchase_ext_ids[cid].append(ext_id)

        # Fallback: category-affinity normalised
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
                import numpy as np
                purchased = purchase_history.get(customer.pk, set())
                candidates = [p for p in all_products if p.pk not in purchased]

                # ----------------------------------------------------------
                # Build per-customer CBF user profile (real TF-IDF cosine)
                # ----------------------------------------------------------
                cbf_all_scores = None
                cbf_pid_to_idx = None
                if cbf_model is not None:
                    ext_ids = purchase_ext_ids.get(customer.pk, [])
                    text_profile, num_profile = _build_cbf_user_profile(ext_ids, cbf_model)
                    if text_profile is not None:
                        cbf_all_scores = _cbf_scores_all(cbf_model, text_profile, num_profile)
                        cbf_pid_to_idx = cbf_model["pid_to_idx"]

                affinity = category_affinity_map.get(customer.pk, {})

                # Score all candidates
                scored = compute_recommendations_for_customer(
                    customer=customer,
                    candidate_products=candidates,
                    svd_model=svd_model,
                    category_affinity=affinity,
                    product_category_map=product_category_map,
                    cbf_model=cbf_model,
                    cbf_all_scores=cbf_all_scores,
                    cbf_pid_to_idx=cbf_pid_to_idx,
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
