"""
store/management/commands/import_reviews.py
===========================================
Import translated reviews from data/olist_order_reviews_translated.csv
and link them to imported Products, updating their avg_sentiment_score.
"""

import os
from datetime import datetime, timezone
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

from store.models import Product, Review


class Command(BaseCommand):
    help = "Import translated reviews from CSV and update product sentiment scores."

    def handle(self, *args, **options):
        csv_path = os.path.join(settings.BASE_DIR, "data", "olist_order_reviews_translated.csv")
        items_path = os.path.join(settings.BASE_DIR, "data", "olist_order_items_dataset.csv")

        if not os.path.exists(csv_path) or not os.path.exists(items_path):
            self.stdout.write(self.style.ERROR(f"CSV files not found in {settings.BASE_DIR}/data/"))
            return

        self.stdout.write("Reading product mappings and reviews...")
        product_map = {p.external_id: p for p in Product.objects.all()}
        if not product_map:
            self.stdout.write(self.style.ERROR("No products in database. Run import_data first."))
            return

        items_df = pd.read_csv(items_path, usecols=["order_id", "product_id"])
        items_matched = items_df[items_df["product_id"].isin(product_map.keys())]
        order_to_product = dict(zip(items_matched["order_id"], items_matched["product_id"]))

        revs_df = pd.read_csv(csv_path)
        revs_matched = revs_df[
            revs_df["order_id"].isin(order_to_product.keys()) &
            revs_df["review_comment_message"].notna()
        ]

        self.stdout.write(f"Found {len(revs_matched)} reviews matching imported products.")

        now = datetime.now(tz=timezone.utc)
        reviews_to_create = []
        product_sentiments = {}

        for _, row in revs_matched.iterrows():
            rev_id = str(row["review_id"])
            order_id = str(row["order_id"])
            prod_id = order_to_product[order_id]
            product = product_map[prod_id]

            score = float(row.get("review_score", 3))
            pt_text = str(row.get("review_comment_message", "") or "")
            en_text = str(row.get("review_comment_translated", "") or pt_text)

            sentiment = "positive" if score >= 4 else ("negative" if score <= 2 else "positive")
            num_sentiment = 1.0 if sentiment == "positive" else 0.0

            if prod_id not in product_sentiments:
                product_sentiments[prod_id] = []
            product_sentiments[prod_id].append(num_sentiment)

            reviews_to_create.append(
                Review(
                    external_id=rev_id,
                    product=product,
                    comment_text=pt_text,
                    comment_text_translated=en_text,
                    sentiment=sentiment,
                    sentiment_processed_at=now,
                )
            )

        with transaction.atomic():
            Review.objects.all().delete()
            Review.objects.bulk_create(reviews_to_create, batch_size=500)

            # Update avg_sentiment_score on products
            for prod_id, sents in product_sentiments.items():
                avg = sum(sents) / len(sents)
                Product.objects.filter(external_id=prod_id).update(avg_sentiment_score=round(avg, 2))

        self.stdout.write(
            self.style.SUCCESS(
                f"[OK] Successfully imported {len(reviews_to_create)} reviews and updated sentiment for {len(product_sentiments)} products."
            )
        )
