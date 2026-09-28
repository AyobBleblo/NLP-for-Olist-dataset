"""
store/ml/loader.py — Singleton loaders for all ML models.

DESIGN PRINCIPLE
================
All heavyweight model loading is isolated here.  Nothing in views.py,
serializers.py, or signals.py may import torch/transformers directly.

Each public getter follows the same pattern:
  - A module-level variable holds the singleton (None until first call).
  - get_X() initialises it on first call, then returns the cached instance.
  - Thread-safety note: Django dev server is single-threaded; for
    production (Gunicorn/uvicorn) use --preload or a dedicated worker
    initializer — the lazy approach is safe enough for the batch commands
    that call these functions.

MODELS MANAGED HERE
===================
1. SentimentPredictor  — binary DistilBERT (Negative / Positive)
   Path: settings.SENTIMENT_MODEL_DIR (models/best_model_binary/)
   Source: inference.SentimentPredictor (already in the project root)

2. Translation pipeline — facebook/nllb-200-distilled-600M
   Same model + tokenizer used in translate_reviews.py.
   Produces: Brazilian Portuguese → English translations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from django.conf import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentiment predictor singleton
# ---------------------------------------------------------------------------

_sentiment_predictor = None  # type: ignore[var-annotated]


def get_sentiment_predictor():
    """
    Return the cached SentimentPredictor, loading it on first call.

    Imports inference.SentimentPredictor from the project root
    (it is already well-tested and handles device selection internally).
    """
    global _sentiment_predictor
    if _sentiment_predictor is None:
        # Import from project-root inference.py — already on sys.path
        # because manage.py adds BASE_DIR to sys.path.
        from inference import SentimentPredictor  # type: ignore[import]

        model_dir = settings.SENTIMENT_MODEL_DIR
        logger.info("Loading sentiment model from %s", model_dir)
        _sentiment_predictor = SentimentPredictor(model_dir=model_dir)
        logger.info(
            "Sentiment model ready on device: %s",
            _sentiment_predictor.device,
        )
    return _sentiment_predictor


# ---------------------------------------------------------------------------
# Translation model singleton
# ---------------------------------------------------------------------------

_translation_tokenizer = None
_translation_model = None
_translation_device = None


def get_translation_components():
    """
    Return (tokenizer, model, device) for the NLLB-200 translation model,
    loading on first call.

    Mirrors the load_model() function in translate_reviews.py exactly:
      - AutoTokenizer with src_lang=por_Latn
      - AutoModelForSeq2SeqLM in fp16 on CUDA (fp32 on CPU)
    """
    global _translation_tokenizer, _translation_model, _translation_device

    if _translation_model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name: str = settings.TRANSLATION_MODEL_NAME
        src_lang: str = settings.TRANSLATION_SRC_LANG

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Loading translation model '%s' on device %s …", model_name, device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if device.type == "cuda":
            model = model.half()  # fp16 to fit in 4 GB VRAM

        model = model.to(device)
        model.eval()

        _translation_tokenizer = tokenizer
        _translation_model = model
        _translation_device = device

        logger.info("Translation model loaded.")

    return _translation_tokenizer, _translation_model, _translation_device


# ---------------------------------------------------------------------------
# Translation helper (mirrors translate_reviews.translate_batch exactly)
# ---------------------------------------------------------------------------

def translate_texts(texts: list[str]) -> list[str]:
    """
    Translate a list of Portuguese strings to English using NLLB-200.

    Mirrors translate_reviews.translate_batch() — same tokenizer settings,
    same forced_bos_token_id trick, same fallback to row-by-row on batch error.

    Args:
        texts: List of Portuguese strings (may include empty strings; they
               are returned as-is).

    Returns:
        List of English strings in the same order as `texts`.
    """
    if not texts:
        return []

    tokenizer, model, device = get_translation_components()

    tgt_lang: str = settings.TRANSLATION_TGT_LANG
    tgt_lang_id: int = tokenizer.convert_tokens_to_ids(tgt_lang)

    max_length: int = 512
    max_new_tokens: int = 300

    def _translate_batch(batch: list[str]) -> list[str]:
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            tokens = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_new_tokens=max_new_tokens,
                max_length=None,
            )

        return tokenizer.batch_decode(tokens, skip_special_tokens=True)

    results: list[str] = []
    for i, text in enumerate(texts):
        if not text or not str(text).strip():
            results.append("")
            continue

        try:
            translated = _translate_batch([text])
            results.append(translated[0])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Translation failed for text index %d (%r…): %s",
                i,
                text[:60],
                exc,
            )
            results.append("")

    return results


def translate_texts_batched(texts: list[str], batch_size: int | None = None) -> list[str]:
    """
    Translate texts in GPU-friendly batches.

    Uses settings.TRANSLATION_GPU_BATCH_SIZE / TRANSLATION_CPU_BATCH_SIZE
    if batch_size is not supplied.

    Args:
        texts:      List of Portuguese strings.
        batch_size: Override the default batch size.

    Returns:
        List of English translations in input order.
    """
    if not texts:
        return []

    _, _, device = get_translation_components()

    if batch_size is None:
        if device.type == "cuda":
            batch_size = getattr(settings, "TRANSLATION_GPU_BATCH_SIZE", 16)
        else:
            batch_size = getattr(settings, "TRANSLATION_CPU_BATCH_SIZE", 8)

    tokenizer, model, device = get_translation_components()
    tgt_lang: str = settings.TRANSLATION_TGT_LANG
    tgt_lang_id: int = tokenizer.convert_tokens_to_ids(tgt_lang)

    results: list[str] = []

    for start in range(0, len(texts), batch_size):
        batch = [str(t) if t and str(t).strip() else "" for t in texts[start : start + batch_size]]

        # Split out empty strings — no point running them through the model
        non_empty_idx = [i for i, t in enumerate(batch) if t]
        non_empty_texts = [batch[i] for i in non_empty_idx]

        batch_results: list[str] = [""] * len(batch)

        if non_empty_texts:
            try:
                inputs = tokenizer(
                    non_empty_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)

                with torch.no_grad():
                    tokens = model.generate(
                        **inputs,
                        forced_bos_token_id=tgt_lang_id,
                        max_new_tokens=300,
                        max_length=None,
                    )

                translated = tokenizer.batch_decode(tokens, skip_special_tokens=True)

                for idx, trans in zip(non_empty_idx, translated):
                    batch_results[idx] = trans

            except Exception as exc:  # noqa: BLE001
                # Row-by-row fallback (mirrors translate_reviews.py)
                logger.warning("Batch translation failed (%s). Falling back row-by-row.", exc)
                for idx in non_empty_idx:
                    text = batch[idx]
                    try:
                        inputs = tokenizer(
                            [text],
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512,
                        ).to(device)
                        with torch.no_grad():
                            tokens = model.generate(
                                **inputs,
                                forced_bos_token_id=tgt_lang_id,
                                max_new_tokens=300,
                                max_length=None,
                            )
                        batch_results[idx] = tokenizer.batch_decode(
                            tokens, skip_special_tokens=True
                        )[0]
                    except Exception as inner_exc:  # noqa: BLE001
                        logger.warning(
                            "Row-by-row translation failed for %r…: %s",
                            text[:60],
                            inner_exc,
                        )
                        batch_results[idx] = ""

        results.extend(batch_results)

    return results
