"""
store/management/commands/diagnose_recommendations.py
=====================================================
Read-only diagnostic command.

Usage:
    python manage.py diagnose_recommendations --customer-id <external_id>

Prints, for a given customer:
  - Number of ratings the user has in the SVD trainset
  - For the top-10 scored products: b_i, q_i*p_u, # ratings in trainset, known?
  - How many candidate products are unknown to the trainset
  - Which cause (1-3) appears dominant
"""

import logging

from django.core.management.base import BaseCommand

from store.ml.loader import get_cf_model
from store.models import Customer, OrderItem, Product

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Diagnostic: print SVD trainset membership and score decomposition for a customer."

    def add_arguments(self, parser):
        parser.add_argument(
            "--customer-id",
            type=str,
            default="3e2157f91502458bc58455fd798ed58a",
            help="Customer external_id to diagnose (default: Bruno).",
        )
        parser.add_argument(
            "--top-n",
            type=int,
            default=10,
            help="Number of top candidates to decompose.",
        )

    def handle(self, *args, **options):  # noqa: C901
        customer_id: str = options["customer_id"]
        top_n: int = options["top_n"]

        # ---------------------------------------------------------------
        # Load customer
        # ---------------------------------------------------------------
        try:
            customer = Customer.objects.get(external_id=customer_id)
        except Customer.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Customer '{customer_id}' not found."))
            return

        self.stdout.write(self.style.MIGRATE_HEADING(f"\nDIAGNOSTIC -- Customer: {customer_id}\n"))

        # ---------------------------------------------------------------
        # Load SVD model
        # ---------------------------------------------------------------
        svd = get_cf_model()
        if svd is None:
            self.stderr.write(self.style.ERROR("SVD model not found -- cannot diagnose."))
            return
        ts = svd.trainset
        self.stdout.write(f"SVD global_mean (mu): {ts.global_mean:.4f}")

        # ---------------------------------------------------------------
        # User membership in trainset
        # ---------------------------------------------------------------
        try:
            inner_uid = ts.to_inner_uid(customer_id)
            user_ratings = ts.ur[inner_uid]
            n_user_ratings = len(user_ratings)
            is_known_user = True
            pu = svd.pu[inner_uid]
            bu = svd.bu[inner_uid]
            self.stdout.write(self.style.SUCCESS(
                f"User IS in trainset -- inner_uid={inner_uid}, "
                f"n_ratings={n_user_ratings}, b_u={bu:.4f}"
            ))
        except ValueError:
            inner_uid = None
            pu = None
            bu = None
            n_user_ratings = 0
            is_known_user = False
            self.stdout.write(self.style.WARNING(
                "User is NOT in trainset (cold-start). "
                "SVD cannot personalise -- popularity/fallback will dominate."
            ))

        # ---------------------------------------------------------------
        # Purchase history (for exclusion)
        # ---------------------------------------------------------------
        purchased_ids: set[int] = set(
            OrderItem.objects.filter(order__customer=customer).values_list("product_id", flat=True)
        )
        self.stdout.write(f"\nPurchased products: {len(purchased_ids)}")

        # ---------------------------------------------------------------
        # Candidate products
        # ---------------------------------------------------------------
        all_products = list(Product.objects.select_related("category").all())
        candidates = [p for p in all_products if p.pk not in purchased_ids]
        self.stdout.write(f"Candidate products (unpurchased): {len(candidates)}")

        # ---------------------------------------------------------------
        # Classify known vs unknown products
        # ---------------------------------------------------------------
        known_count = 0
        unknown_count = 0
        scored: list[tuple[float, float, float, int, bool, Product]] = []
        # (total_score, bi, qi_pu, n_item_ratings, is_known, product)

        for prod in candidates:
            try:
                inner_iid = ts.to_inner_iid(prod.external_id)
                bi = float(svd.bi[inner_iid])
                n_item_ratings = len(ts.ir[inner_iid])
                is_known = True
                known_count += 1
                if is_known_user and pu is not None:
                    qi = svd.qi[inner_iid]
                    qi_pu = float(qi @ pu)
                else:
                    qi_pu = 0.0
                if bu is not None:
                    total = ts.global_mean + bu + bi + qi_pu
                else:
                    total = ts.global_mean + bi + qi_pu
            except ValueError:
                bi = 0.0
                qi_pu = 0.0
                n_item_ratings = 0
                is_known = False
                unknown_count += 1
                total = ts.global_mean  # fallback

            scored.append((total, bi, qi_pu, n_item_ratings, is_known, prod))

        scored.sort(key=lambda x: x[0], reverse=True)

        self.stdout.write(f"\nKnown products in trainset: {known_count}")
        self.stdout.write(f"Unknown products (not in trainset): {unknown_count}")
        pct_unknown = 100 * unknown_count / max(1, len(candidates))
        self.stdout.write(f"Unknown percentage: {pct_unknown:.1f}%\n")

        # ---------------------------------------------------------------
        # Top-N decomposition
        # ---------------------------------------------------------------
        self.stdout.write(self.style.MIGRATE_HEADING(f"TOP {top_n} SCORED CANDIDATES -- Score Decomposition\n"))
        header = f"{'#':<3} {'Product':<35} {'Cat':<20} {'Total':>7} {'b_i':>7} {'qi*pu':>7} {'#ratings':>8} {'known?':>7}"
        self.stdout.write(header)
        self.stdout.write("-" * len(header))

        for rank, (total, bi, qi_pu, n_item_ratings, is_known, prod) in enumerate(scored[:top_n], 1):
            cat = str(prod.category) if prod.category else "--"
            known_flag = "YES" if is_known else "NO"
            self.stdout.write(
                f"{rank:<3} {str(prod)[:35]:<35} {cat[:20]:<20} "
                f"{total:>7.4f} {bi:>7.4f} {qi_pu:>7.4f} {n_item_ratings:>8} {known_flag:>7}"
            )

        # ---------------------------------------------------------------
        # Dominant cause analysis
        # ---------------------------------------------------------------
        self.stdout.write(self.style.MIGRATE_HEADING("\n--- DOMINANT CAUSE ANALYSIS ---\n"))

        top_scored_subset = scored[:top_n]
        if top_scored_subset:
            bi_values = [abs(x[1]) for x in top_scored_subset]
            qi_pu_values = [abs(x[2]) for x in top_scored_subset]
            avg_bi = sum(bi_values) / len(bi_values)
            avg_qi_pu = sum(qi_pu_values) / len(qi_pu_values)
            self.stdout.write(f"Avg |b_i| in top-{top_n}: {avg_bi:.4f}")
            self.stdout.write(f"Avg |q_i*p_u| in top-{top_n}: {avg_qi_pu:.4f}")
            if not is_known_user:
                self.stdout.write(self.style.ERROR(
                    "\n>>> CAUSE 3 DOMINANT: User is cold-start (not in SVD trainset). "
                    "All scores fall back to global_mean -> same ranking for everyone."
                ))
            elif avg_qi_pu < avg_bi * 0.1:
                self.stdout.write(self.style.WARNING(
                    "\n>>> CAUSE 1 DOMINANT: |q_i*p_u| is tiny vs |b_i|. "
                    "Product bias dominates -> near-identical rankings across users."
                ))
            else:
                self.stdout.write(self.style.SUCCESS(
                    f"\n>>> SVD personalisation active: |q_i*p_u| ({avg_qi_pu:.4f}) "
                    f"is meaningful relative to |b_i| ({avg_bi:.4f})."
                ))

        if pct_unknown > 50:
            self.stdout.write(self.style.ERROR(
                f"\n>>> CAUSE 3 ALSO SIGNIFICANT: {pct_unknown:.0f}% of candidates are unknown "
                "to trainset -> they all get the same global_mean fallback score, creating ties."
            ))

        if not is_known_user:
            self.stdout.write(self.style.WARNING(
                "\n>>> CAUSE 2 CONFIRMED: Seed customers were never in the SVD training data "
                "(OrderItems created after training) -> cold-start guaranteed for all 3 test users."
            ))

        self.stdout.write("")
