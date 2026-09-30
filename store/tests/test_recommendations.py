"""
store/tests/test_recommendations.py
====================================
Unit tests for compute_recommendations_for_customer and helpers.

Tests cover:
  1. Unknown-user (cold-start) fallback — category affinity still personalises
  2. Purchase exclusion — purchased products never appear in results
  3. Diversity cap (MAX_PER_CAT) — no category exceeds the per-category limit
  4. Affinity boost — products in the customer's top category score higher

Run with:
    python manage.py test store.tests.test_recommendations
or:
    uv run manage.py test store.tests.test_recommendations
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from django.test import TestCase

from store.management.commands.compute_recommendations import (
    ALPHA,
    BASE,
    BETA,
    GAMMA,
    MAX_PER_CAT,
    _apply_diversity_cap,
    compute_recommendations_for_customer,
)
from store.models import Category, Customer, Order, OrderItem, Product


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_category(name: str) -> Category:
    cat, _ = Category.objects.get_or_create(name=name, defaults={"name_translated": name})
    return cat


def _make_product(
    ext_id: str,
    category: Category | None = None,
    purchases: int = 0,
    sentiment: float = 0.0,
) -> Product:
    prod, _ = Product.objects.get_or_create(
        external_id=ext_id,
        defaults={
            "category": category,
            "total_purchases": purchases,
            "avg_sentiment_score": sentiment,
            "price": "10.00",
        },
    )
    return prod


def _make_customer(ext_id: str) -> Customer:
    cust, _ = Customer.objects.get_or_create(external_id=ext_id)
    return cust


def _make_order_with_items(customer: Customer, products: list[Product]) -> Order:
    order, _ = Order.objects.get_or_create(
        external_id=f"ord_{customer.external_id[:8]}",
        customer=customer,
        defaults={"status": "delivered"},
    )
    for prod in products:
        OrderItem.objects.get_or_create(order=order, product=prod, defaults={"price": "10.00"})
    return order


def _make_svd_mock(
    user_ext_id: str,
    known_products: dict[str, tuple[float, np.ndarray]],  # ext_id -> (bi, qi)
    n_factors: int = 4,
    global_mean: float = 4.0,
) -> MagicMock:
    """
    Build a minimal mock surprise.SVD with a trainset that knows the given
    user and products.  Products not in known_products raise ValueError on
    to_inner_iid (unknown product path).
    """
    svd = MagicMock()
    ts = MagicMock()
    svd.trainset = ts
    ts.global_mean = global_mean

    # User
    inner_uid = 0
    ts.to_inner_uid.side_effect = lambda uid: inner_uid if uid == user_ext_id else (_ for _ in ()).throw(ValueError)
    pu = np.ones(n_factors) * 0.1
    bu = 0.05
    svd.pu = {inner_uid: pu}
    svd.bu = {inner_uid: bu}
    ts.ur = {inner_uid: [(0, 4.0)]}  # 1 rating

    # Products
    inner_iid_map: dict[str, int] = {}
    bi_arr: dict[int, float] = {}
    qi_arr: dict[int, np.ndarray] = {}
    ir_map: dict[int, list] = {}
    for idx, (ext_id, (bi, qi)) in enumerate(known_products.items()):
        inner_iid_map[ext_id] = idx
        bi_arr[idx] = bi
        qi_arr[idx] = qi
        ir_map[idx] = [(0, 4.0)] * 5  # 5 ratings

    def to_inner_iid(ext_id: str) -> int:
        if ext_id in inner_iid_map:
            return inner_iid_map[ext_id]
        raise ValueError(f"Unknown product: {ext_id}")

    ts.to_inner_iid.side_effect = to_inner_iid
    svd.bi = bi_arr
    svd.qi = qi_arr
    ts.ir = ir_map

    return svd


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestUnknownUserFallback(TestCase):
    """Cold-start user gets popularity fallback, but affinity still works."""

    def setUp(self):
        self.cat_sports = _make_category("sports")
        self.cat_tech = _make_category("tech")
        self.customer = _make_customer("unknown_user_ext_id")
        # 2 purchases in sports (all from test, not in SVD trainset)
        self.prod_sports1 = _make_product("sp001", self.cat_sports, purchases=5)
        self.prod_sports2 = _make_product("sp002", self.cat_sports, purchases=3)
        self.prod_tech1 = _make_product("tc001", self.cat_tech, purchases=10)
        _make_order_with_items(self.customer, [self.prod_sports1, self.prod_sports2])

    def test_cold_start_returns_scores(self):
        """Unknown user: should still return a scored list (fallback path)."""
        # SVD that does NOT know this user
        svd = _make_svd_mock(
            user_ext_id="some_other_user",
            known_products={
                "tc001": (0.3, np.ones(4) * 0.1),
            },
        )
        candidates = [self.prod_sports1, self.prod_tech1]
        # Manually pass empty affinity (no purchase history for THIS test call)
        results = compute_recommendations_for_customer(
            customer=self.customer,
            candidate_products=candidates,
            svd_model=svd,
            category_affinity={},
        )
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)
        # All scores should be positive
        for _, score in results:
            self.assertGreater(score, 0.0)

    def test_cold_start_affinity_boosts_known_category(self):
        """
        Unknown user with 100% sports affinity: sports product must outscore
        a tech product with higher raw popularity.
        """
        # SVD knows neither user nor products -> all fallback
        svd = _make_svd_mock(
            user_ext_id="some_other_user",
            known_products={},
        )
        # sports product: 3 purchases; tech product: 10 purchases (would win without affinity)
        sports_prod = _make_product("sp_x", self.cat_sports, purchases=3)
        tech_prod = _make_product("tc_x", self.cat_tech, purchases=10)

        affinity = {self.cat_sports.pk: 1.0}  # 100% sports
        results = compute_recommendations_for_customer(
            customer=self.customer,
            candidate_products=[sports_prod, tech_prod],
            svd_model=svd,
            category_affinity=affinity,
        )
        scores = {pid: score for pid, score in results}
        # Sports product should win due to BETA * 1.0 affinity boost
        self.assertGreater(scores[sports_prod.pk], scores[tech_prod.pk])


class TestPurchaseExclusion(TestCase):
    """Purchased products must never appear in recommendations."""

    def setUp(self):
        self.cat = _make_category("general")
        self.customer = _make_customer("excl_user_ext_id")
        self.purchased = _make_product("excl_p001", self.cat, purchases=100)
        self.not_purchased = _make_product("excl_p002", self.cat, purchases=1)
        _make_order_with_items(self.customer, [self.purchased])

    def test_purchased_product_absent_from_candidates(self):
        """
        The command.handle() pre-filters purchased products.  Here we verify
        that if the caller passes only unpurchased products, the result only
        contains them.
        """
        candidates = [self.not_purchased]  # purchased product intentionally excluded by caller
        results = compute_recommendations_for_customer(
            customer=self.customer,
            candidate_products=candidates,
            svd_model=None,
            category_affinity={},
        )
        returned_ids = {pid for pid, _ in results}
        self.assertNotIn(self.purchased.pk, returned_ids)
        self.assertIn(self.not_purchased.pk, returned_ids)

    def test_purchased_product_not_in_result_when_caller_includes_it(self):
        """
        Even if the caller accidentally passes a purchased product in candidates,
        it should be excluded by the management command's candidate filtering.
        Verify via compute_recommendations() wrapper.
        """
        from store.management.commands.compute_recommendations import compute_recommendations

        all_ids = [self.purchased.pk, self.not_purchased.pk]
        purchased_ids = {self.purchased.pk}
        results = compute_recommendations(self.customer.pk, all_ids, purchased_ids)
        returned_ids = {pid for pid, _ in results}
        self.assertNotIn(self.purchased.pk, returned_ids)


class TestDiversityCap(TestCase):
    """_apply_diversity_cap enforces MAX_PER_CAT across categories."""

    def test_cap_limits_per_category(self):
        cat_a_id = 1
        cat_b_id = 2
        # 4 products in cat A, 2 in cat B — ranked by score already
        ranked = [
            (101, 5.0),  # cat A
            (102, 4.8),  # cat A
            (103, 4.6),  # cat A  <- should be held back if max_per_cat=2
            (104, 4.4),  # cat A  <- same
            (201, 4.2),  # cat B
            (202, 4.0),  # cat B
        ]
        product_category_map = {
            101: cat_a_id, 102: cat_a_id, 103: cat_a_id, 104: cat_a_id,
            201: cat_b_id, 202: cat_b_id,
        }
        result = _apply_diversity_cap(ranked, product_category_map, top_n=5, max_per_cat=2)
        result_ids = [pid for pid, _ in result]

        # At most 2 from cat A
        cat_a_count = sum(1 for pid in result_ids if product_category_map[pid] == cat_a_id)
        self.assertLessEqual(cat_a_count, 2)

        # Both cat B products should appear (2 slots remain)
        self.assertIn(201, result_ids)
        self.assertIn(202, result_ids)

    def test_cap_fills_remaining_from_leftover(self):
        """
        Leftover fill also respects the cap.  With max_per_cat=2 and all 10
        products in the same single category, only 2 can ever appear regardless
        of top_n=5 because there is no other category to draw from.
        """
        cat_id = 99
        ranked = [(i, float(10 - i)) for i in range(1, 11)]  # 10 products, all same cat
        product_category_map = {i: cat_id for i in range(1, 11)}

        result = _apply_diversity_cap(ranked, product_category_map, top_n=5, max_per_cat=2)
        # Only 2 can pass (max_per_cat), no other category exists to fill remaining slots
        self.assertEqual(len(result), 2)
        # They must be the highest-scored two
        result_ids = [pid for pid, _ in result]
        self.assertIn(1, result_ids)
        self.assertIn(2, result_ids)

    def test_no_cap_violation_in_result(self):
        """Verify MAX_PER_CAT is never exceeded by the cap function."""
        cats = {i: i % 3 for i in range(20)}  # 20 products, 3 categories
        ranked = [(i, float(20 - i)) for i in range(20)]
        result = _apply_diversity_cap(ranked, cats, top_n=10, max_per_cat=2)
        per_cat: dict[int, int] = {}
        for pid, _ in result:
            c = cats[pid]
            per_cat[c] = per_cat.get(c, 0) + 1
        for cat, count in per_cat.items():
            self.assertLessEqual(count, 2, f"Cat {cat} appears {count} times, exceeds cap")


class TestAffinityBoost(TestCase):
    """Category affinity boost should directly affect scores."""

    def setUp(self):
        self.cat_a = _make_category("catA_aff")
        self.cat_b = _make_category("catB_aff")
        self.customer = _make_customer("aff_user_ext_id")
        self.prod_a = _make_product("aff_pa", self.cat_a, purchases=5)
        self.prod_b = _make_product("aff_pb", self.cat_b, purchases=5)

    def test_affinity_boost_increases_score(self):
        """A product in the customer's preferred category should score higher."""
        # No SVD — compare using popularity fallback as base
        affinity = {self.cat_a.pk: 1.0}  # 100% cat A preference
        results_with_affinity = compute_recommendations_for_customer(
            customer=self.customer,
            candidate_products=[self.prod_a, self.prod_b],
            svd_model=None,
            category_affinity=affinity,
        )
        scores = {pid: score for pid, score in results_with_affinity}

        # prod_a in cat_a (full affinity), prod_b in cat_b (zero affinity)
        # both have same purchases=5, so base is identical
        self.assertGreater(
            scores[self.prod_a.pk],
            scores[self.prod_b.pk],
            "Product in preferred category should score higher.",
        )

    def test_zero_affinity_produces_no_boost(self):
        """With affinity=0, product score equals base score."""
        affinity = {}  # no history
        results = compute_recommendations_for_customer(
            customer=self.customer,
            candidate_products=[self.prod_a],
            svd_model=None,
            category_affinity=affinity,
        )
        _, score = results[0]
        # Base popularity score for 5 purchases (no SVD model)
        expected_base = 1.0 + min(5.0 / 10.0, 4.0)
        self.assertAlmostEqual(score, expected_base, places=3)

    def test_sentiment_boost_skipped_when_no_reviews(self):
        """avg_sentiment_score=0 with no purchases means no sentiment boost."""
        prod_no_reviews = _make_product("sent_x", self.cat_a, purchases=0, sentiment=0.0)
        affinity = {}
        results = compute_recommendations_for_customer(
            customer=self.customer,
            candidate_products=[prod_no_reviews],
            svd_model=None,
            category_affinity=affinity,
        )
        _, score = results[0]
        # purchases=0 -> no sentiment boost applied; base = 1.0 + 0/10 = 1.0
        expected = 1.0
        self.assertAlmostEqual(score, expected, places=3)

    def test_sentiment_boost_applied_when_has_reviews(self):
        """Positive sentiment on a product with purchases raises score."""
        prod_reviewed = _make_product("sent_y", self.cat_a, purchases=10, sentiment=0.8)
        affinity = {}
        results = compute_recommendations_for_customer(
            customer=self.customer,
            candidate_products=[prod_reviewed],
            svd_model=None,
            category_affinity=affinity,
        )
        _, score = results[0]
        base = 1.0 + min(10.0 / 10.0, 4.0)  # = 2.0
        expected = base + GAMMA * (0.8 - 0.5)
        self.assertAlmostEqual(score, round(expected, 4), places=3)
