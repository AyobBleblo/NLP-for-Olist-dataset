"""
store/management/commands/run_sentiment_batch.py
================================================
Management command: python manage.py run_sentiment_batch

For every Review with sentiment IS NULL:
  1. Takes comment_text (raw Portuguese).
  2. Translates it to English via facebook/nllb-200-distilled-600M
     (same model + settings as translate_reviews.py — loaded once).
  3. Runs the English text through the binary DistilBERT classifier
     (models/best_model_binary/) — loaded once.
  4. Stores:
       - comment_text_translated  (translated English text)
       - sentiment                ('positive' or 'negative')
       - sentiment_processed_at   (current UTC timestamp)

Notes
-----
- Reviews with an empty / whitespace-only comment_text get
  sentiment_processed_at stamped but sentiment left NULL (no text → cannot
  classify; we don't want to re-process them on every run).
- Neutral (3-star) reviews were excluded from training.  The command still
  processes them — the classifier will assign one of the two binary labels,
  which is acceptable for avg_sentiment_score computation.
- After processing, it recomputes avg_sentiment_score for each affected
  Product and saves it.

Usage
-----
  python manage.py run_sentiment_batch
  python manage.py run_sentiment_batch --translation-batch 8 --sentiment-batch 16
  python manage.py run_sentiment_batch --limit 200   # process only 200 rows
"""

import logging
from datetime import datetime, timezone

from django.core.management.base import BaseCommand
from django.db import transaction
from tqdm import tqdm

from store.ml.loader import get_sentiment_predictor, translate_texts_batched
from store.models import Product, Review

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Offline batch job: translate reviews and run binary sentiment classification."

    def add_arguments(self, parser):
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Process at most N reviews (0 = all).",
        )
        parser.add_argument(
            "--translation-batch",
            type=int,
            default=None,
            help="Override translation batch size (default: from settings).",
        )
        parser.add_argument(
            "--sentiment-batch",
            type=int,
            default=32,
            help="Sentiment classifier batch size (default: 32).",
        )

    def handle(self, *args, **options):
        limit: int = options["limit"]
        trans_batch: int | None = options["translation_batch"]
        sent_batch: int = options["sentiment_batch"]

        # ------------------------------------------------------------------
        # 1. Fetch unprocessed reviews
        # ------------------------------------------------------------------
        qs = Review.objects.filter(sentiment__isnull=True).select_related("product")
        if limit > 0:
            qs = qs[:limit]

        reviews = list(qs)
        total = len(reviews)

        if total == 0:
            self.stdout.write(self.style.SUCCESS("No unprocessed reviews found. Done."))
            return

        self.stdout.write(f"Found {total} unprocessed reviews.")

        # ------------------------------------------------------------------
        # 2. Load ML models (lazy — loads once per process)
        # ------------------------------------------------------------------
        self.stdout.write("Loading translation model …")
        # Trigger translation model load now so progress bar timing is accurate
        translate_texts_batched(["test"], batch_size=1)  # warm-up
        self.stdout.write("Loading sentiment model …")
        predictor = get_sentiment_predictor()
        self.stdout.write(
            self.style.SUCCESS(
                f"Models loaded on device: {predictor.device}"
            )
        )

        # ------------------------------------------------------------------
        # 3. Split into reviews with / without text
        # ------------------------------------------------------------------
        has_text = [r for r in reviews if r.comment_text and r.comment_text.strip()]
        no_text = [r for r in reviews if not (r.comment_text and r.comment_text.strip())]

        self.stdout.write(
            f"  {len(has_text)} with text / {len(no_text)} without text."
        )

        now = datetime.now(tz=timezone.utc)

        # ------------------------------------------------------------------
        # 4. Mark no-text reviews (stamp processed_at, leave sentiment NULL)
        # ------------------------------------------------------------------
        if no_text:
            with transaction.atomic():
                for review in no_text:
                    review.sentiment_processed_at = now
                    review.comment_text_translated = ""
                Review.objects.bulk_update(
                    no_text,
                    ["sentiment_processed_at", "comment_text_translated"],
                    batch_size=500,
                )
            self.stdout.write(f"  ✓ Stamped {len(no_text)} text-less reviews.")

        # ------------------------------------------------------------------
        # 5. Translate reviews with text (batched)
        # ------------------------------------------------------------------
        if not has_text:
            self.stdout.write(self.style.SUCCESS("\n✓ Batch complete."))
            return

        self.stdout.write(f"\nTranslating {len(has_text)} reviews …")
        texts_pt = [r.comment_text for r in has_text]
        texts_en = translate_texts_batched(texts_pt, batch_size=trans_batch)

        # ------------------------------------------------------------------
        # 6. Run sentiment classifier
        # ------------------------------------------------------------------
        self.stdout.write(f"Running sentiment classifier (batch={sent_batch}) …")
        results = predictor.predict_batch(texts_en, batch_size=sent_batch)

        # ------------------------------------------------------------------
        # 7. Update reviews in DB
        # ------------------------------------------------------------------
        self.stdout.write("Saving results …")
        now = datetime.now(tz=timezone.utc)
        affected_product_ids: set[int] = set()

        for review, translated, result in tqdm(
            zip(has_text, texts_en, results),
            total=len(has_text),
            desc="  Saving",
        ):
            label_id: int = result["label_id"]
            # label_id -1 means the predictor got an empty string (shouldn't
            # happen here since we filtered for non-empty texts, but be safe)
            if label_id == -1:
                sentiment_str = None
            elif label_id == 1:
                sentiment_str = Review.SENTIMENT_POSITIVE
            else:
                sentiment_str = Review.SENTIMENT_NEGATIVE

            review.comment_text_translated = translated
            review.sentiment = sentiment_str
            review.sentiment_processed_at = now

            if review.product_id:
                affected_product_ids.add(review.product_id)

        with transaction.atomic():
            Review.objects.bulk_update(
                has_text,
                ["comment_text_translated", "sentiment", "sentiment_processed_at"],
                batch_size=500,
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"  ✓ {len(has_text)} reviews classified and saved."
            )
        )

        # ------------------------------------------------------------------
        # 8. Recompute avg_sentiment_score for affected products
        # ------------------------------------------------------------------
        if affected_product_ids:
            self.stdout.write(
                f"Updating avg_sentiment_score for {len(affected_product_ids)} products …"
            )
            updated = 0
            for prod_id in affected_product_ids:
                product_reviews = Review.objects.filter(
                    product_id=prod_id,
                    sentiment__isnull=False,
                )
                total_reviews = product_reviews.count()
                if total_reviews == 0:
                    continue

                positive_count = product_reviews.filter(
                    sentiment=Review.SENTIMENT_POSITIVE
                ).count()
                score = positive_count / total_reviews

                Product.objects.filter(pk=prod_id).update(avg_sentiment_score=score)
                updated += 1

            self.stdout.write(
                self.style.SUCCESS(
                    f"  ✓ avg_sentiment_score updated for {updated} products."
                )
            )

        self.stdout.write(self.style.SUCCESS("\n✓ Sentiment batch complete."))
