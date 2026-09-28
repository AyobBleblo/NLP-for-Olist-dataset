"""
store/management/commands/import_data.py
========================================
Management command: python manage.py import_data

Imports products and categories from the Olist CSV dataset.
Products are sorted DESC by product_photos_qty (most-photographed first)
and capped at --limit.

Usage
-----
  python manage.py import_data
  python manage.py import_data --limit 500
  python manage.py import_data --data-dir path/to/csv/folder

Algorithm
---------
1. Load olist_products_dataset.csv, sort DESC by product_photos_qty,
   take the top `limit` rows.
2. Load optional product_category_name_translation.csv to populate English
   category names immediately.
3. Optionally read product prices from olist_order_items_dataset.csv if present.
4. Upsert Category and Product rows in the database (idempotent).
"""

import hashlib
import os

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from store.models import Category, Product

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = os.path.join(settings.BASE_DIR, "data")
DEFAULT_LIMIT = 1000

ADJECTIVES_EN = [
    "Classic", "Premium", "Essential", "Deluxe", "Pro",
    "Smart", "Modern", "Original", "Urban", "Signature",
]
ADJECTIVES_PT = [
    "Clássico", "Premium", "Essencial", "Luxo", "Pro",
    "Smart", "Moderno", "Original", "Urbano", "Assinatura",
]


def _make_image_url(external_id: str) -> str:
    """
    Return a deterministic placeholder image URL seeded by external_id.

    Uses the first 8 hex characters of SHA-256(external_id) as the
    Picsum seed so the URL is stable across re-imports and unique per
    product.
    """
    digest = hashlib.sha256(external_id.encode()).hexdigest()[:8]
    return f"https://picsum.photos/seed/{digest}/400/400"


class Command(BaseCommand):
    help = (
        "Import top-N products (by photo count) and their categories "
        "from the Olist CSV dataset."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--limit",
            type=int,
            default=DEFAULT_LIMIT,
            help=f"Number of products to import (default: {DEFAULT_LIMIT}).",
        )
        parser.add_argument(
            "--data-dir",
            type=str,
            default=DEFAULT_DATA_DIR,
            help="Directory containing the Olist CSV files (default: data/).",
        )

    def handle(self, *args, **options):
        limit: int = options["limit"]
        data_dir: str = os.path.abspath(options["data_dir"])

        self.stdout.write(f"Data directory : {data_dir}")
        self.stdout.write(f"Product limit  : {limit}")

        # ------------------------------------------------------------------
        # 1. Load and select products
        # ------------------------------------------------------------------
        products_csv = os.path.join(data_dir, "olist_products_dataset.csv")
        self._check_file(products_csv)

        self.stdout.write("Loading products CSV …")
        products_df = pd.read_csv(products_csv)

        # Sort by photo count descending, then take top N
        products_df = products_df.sort_values(
            "product_photos_qty", ascending=False, na_position="last"
        ).head(limit)

        product_ids: set[str] = set(products_df["product_id"].dropna().tolist())
        self.stdout.write(f"  → Selected {len(product_ids)} products.")

        # ------------------------------------------------------------------
        # 2. Load optional category name translation CSV
        # ------------------------------------------------------------------
        cat_translation: dict[str, str] = {}
        cat_csv = os.path.join(data_dir, "product_category_name_translation.csv")
        if os.path.exists(cat_csv):
            cat_df = pd.read_csv(cat_csv)
            cat_translation = dict(
                zip(
                    cat_df["product_category_name"],
                    cat_df["product_category_name_english"],
                )
            )
            self.stdout.write(f"  → Loaded {len(cat_translation)} category translations.")

        # ------------------------------------------------------------------
        # 3. Optional: Read prices and purchase count from order items CSV
        # ------------------------------------------------------------------
        price_map: dict[str, float] = {}
        purchases_map: dict[str, int] = {}
        items_csv = os.path.join(data_dir, "olist_order_items_dataset.csv")
        if os.path.exists(items_csv):
            try:
                self.stdout.write("Reading product prices from order items CSV …")
                items_df = pd.read_csv(items_csv, usecols=["product_id", "price"])
                matched_items = items_df[items_df["product_id"].isin(product_ids)]
                if not matched_items.empty:
                    price_map = (
                        matched_items.groupby("product_id")["price"]
                        .mean()
                        .round(2)
                        .to_dict()
                    )
                    purchases_map = (
                        matched_items["product_id"].value_counts().to_dict()
                    )
                self.stdout.write(f"  → Found prices for {len(price_map)} products.")
            except Exception as e:
                self.stdout.write(f"  → (Skipped optional price lookup: {e})")

        # ------------------------------------------------------------------
        # 4. Upsert Categories & Products to database (idempotent)
        # ------------------------------------------------------------------
        self.stdout.write(self.style.MIGRATE_HEADING("\nInserting into database …"))

        with transaction.atomic():
            # ---- Categories ----
            self.stdout.write("  Categories …")
            category_map: dict[str, Category] = {}
            raw_categories = (
                products_df["product_category_name"].dropna().unique().tolist()
            )
            for cat_name in raw_categories:
                cat_name_str = str(cat_name)
                cat_obj, _ = Category.objects.update_or_create(
                    name=cat_name_str,
                    defaults={
                        "name_translated": cat_translation.get(cat_name_str, ""),
                    },
                )
                category_map[cat_name_str] = cat_obj
            self.stdout.write(f"    {len(category_map)} categories upserted.")

            # ---- Products ----
            self.stdout.write("  Products …")
            product_map: dict[str, Product] = {}
            for _, row in products_df.iterrows():
                pid = str(row["product_id"])
                cat_name = (
                    str(row["product_category_name"])
                    if pd.notna(row.get("product_category_name"))
                    else None
                )
                cat_obj = category_map.get(cat_name) if cat_name else None
                price = price_map.get(pid, 0.0)
                total_purchases = purchases_map.get(pid, 0)

                # Generate clean, realistic product names
                pt_cat = cat_obj.name.replace("_", " ").title() if cat_obj else "Produto"
                en_cat = (
                    cat_obj.name_translated.replace("_", " ").title()
                    if (cat_obj and cat_obj.name_translated)
                    else pt_cat
                )
                idx = int(pid[:2], 16) % len(ADJECTIVES_EN)
                prod_name_pt = f"{pt_cat} {ADJECTIVES_PT[idx]} #{pid[:6].upper()}"
                prod_name_en = f"{ADJECTIVES_EN[idx]} {en_cat} #{pid[:6].upper()}"

                prod_obj, _ = Product.objects.update_or_create(
                    external_id=pid,
                    defaults={
                        "category": cat_obj,
                        "name": prod_name_pt,
                        "name_translated": prod_name_en,
                        "price": price,
                        "total_purchases": total_purchases,
                        "image_url": _make_image_url(pid),
                    },
                )
                product_map[pid] = prod_obj
            self.stdout.write(f"    {len(product_map)} products upserted.")

        # ------------------------------------------------------------------
        # Done
        # ------------------------------------------------------------------
        self.stdout.write(
            self.style.SUCCESS(
                f"\n✓ Import complete: "
                f"{len(category_map)} categories | "
                f"{len(product_map)} products."
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_file(self, path: str) -> None:
        if not os.path.exists(path):
            raise CommandError(f"Required CSV not found: {path}")
