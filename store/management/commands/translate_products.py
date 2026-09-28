"""
store/management/commands/translate_products.py
===============================================
Management command: python manage.py translate_products

Translates each Category.name and Product.name from Portuguese to English
using the SAME facebook/nllb-200-distilled-600M model used in
translate_reviews.py (loaded via store.ml.loader to avoid double-loading).

Skips rows that already have a non-empty translated value (idempotent).

Usage
-----
  python manage.py translate_products
  python manage.py translate_products --force    # re-translate even if done
  python manage.py translate_products --batch-size 8
"""

from django.core.management.base import BaseCommand
from django.db import transaction
from tqdm import tqdm

from store.ml.loader import translate_texts_batched
from store.models import Category, Product


class Command(BaseCommand):
    help = (
        "Translate Category names and Product names from Portuguese to English "
        "using facebook/nllb-200-distilled-600M (same model as review translation)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            default=False,
            help="Re-translate rows that already have a translation.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help="Override default GPU/CPU batch size.",
        )

    def handle(self, *args, **options):
        force: bool = options["force"]
        batch_size: int | None = options["batch_size"]

        # ------------------------------------------------------------------
        # 1. Translate categories
        # ------------------------------------------------------------------
        self.stdout.write(self.style.MIGRATE_HEADING("Translating categories …"))

        if force:
            cats = list(Category.objects.all())
        else:
            cats = list(Category.objects.filter(name_translated=""))

        self.stdout.write(f"  {len(cats)} categories to translate.")

        if cats:
            cat_texts = [c.name for c in cats]
            cat_translations = translate_texts_batched(cat_texts, batch_size=batch_size)

            with transaction.atomic():
                for cat, translated in tqdm(
                    zip(cats, cat_translations),
                    total=len(cats),
                    desc="  Saving categories",
                ):
                    cat.name_translated = translated or cat.name
                    cat.save(update_fields=["name_translated"])

            self.stdout.write(
                self.style.SUCCESS(f"  ✓ {len(cats)} categories translated.")
            )

        # ------------------------------------------------------------------
        # 2. Translate product names
        # ------------------------------------------------------------------
        self.stdout.write(self.style.MIGRATE_HEADING("Translating product names …"))

        if force:
            products = list(Product.objects.exclude(name=""))
        else:
            products = list(
                Product.objects.filter(name_translated="").exclude(name="")
            )

        self.stdout.write(f"  {len(products)} products to translate.")

        if products:
            prod_texts = [p.name for p in products]
            prod_translations = translate_texts_batched(prod_texts, batch_size=batch_size)

            with transaction.atomic():
                for prod, translated in tqdm(
                    zip(products, prod_translations),
                    total=len(products),
                    desc="  Saving products",
                ):
                    prod.name_translated = translated or prod.name
                    prod.save(update_fields=["name_translated"])

            self.stdout.write(
                self.style.SUCCESS(f"  ✓ {len(products)} product names translated.")
            )

        self.stdout.write(self.style.SUCCESS("\n✓ Translation complete."))
