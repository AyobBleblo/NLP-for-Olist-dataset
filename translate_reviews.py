"""
Translate Brazilian Portuguese e-commerce reviews to English.

Uses facebook/nllb-200-distilled-600M for accurate multilingual translation.
NLLB-200 explicitly supports Brazilian Portuguese (por_Latn) and fits within
a GTX 1650 Ti (4GB VRAM) with batch inference.

Usage:
    uv run translate_reviews.py --sample 30    # Test on 30 rows first
    uv run translate_reviews.py                # Translate full dataset
"""

import argparse
import io
import sys
import time

# Force UTF-8 output on Windows to avoid cp1252 encoding crashes
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import ftfy
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "por_Latn"  # Portuguese (Latin script) — covers Brazilian Portuguese
TGT_LANG = "eng_Latn"  # English (Latin script)

INPUT_CSV = "data/olist_order_reviews_dataset.csv"
OUTPUT_CSV = "data/olist_order_reviews_translated.csv"
TEXT_COL = "review_comment_message"
TRANSLATED_COL = "review_comment_translated"

# Batch sizes – smaller for 600M model on 4GB VRAM
GPU_BATCH_SIZE = 16
CPU_BATCH_SIZE = 8
MAX_TOKEN_LENGTH = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fix_encoding(text: str) -> str:
    """Fix mojibake / encoding issues (e.g. Parab�ns -> Parabéns)."""
    if not isinstance(text, str):
        return text
    return ftfy.fix_text(text)


def load_model(device: torch.device):
    """Download (if needed) and load model + tokenizer."""
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Use float16 on GPU to save VRAM
    if device.type == "cuda":
        model = model.half()

    model = model.to(device)
    model.eval()
    print("Model loaded successfully.\n")
    return tokenizer, model


def translate_batch(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> list[str]:
    """Translate a batch of texts. Returns list of translated strings."""
    # Get the target language token id for forced_bos
    tgt_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    try:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
        ).to(device)

        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_new_tokens=300,
                max_length=None,
            )

        results = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return results

    except Exception as e:
        # Fallback: try translating one-by-one so a single bad row
        # doesn't kill the whole batch
        print(f"\n⚠ Batch failed ({e}). Falling back to row-by-row...")
        results = []
        for text in texts:
            try:
                inputs = tokenizer(
                    [text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_TOKEN_LENGTH,
                ).to(device)
                with torch.no_grad():
                    tokens = model.generate(
                        **inputs,
                        forced_bos_token_id=tgt_lang_id,
                        max_new_tokens=300,
                        max_length=None,
                    )
                result = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
                results.append(result)
            except Exception as inner_e:
                print(f"  ✗ Skipped row ({inner_e}): {text[:60]}...")
                results.append("")
        return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Translate PT-BR reviews to English")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Translate only N rows for quality check (0 = full dataset)",
    )
    args = parser.parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = GPU_BATCH_SIZE if device.type == "cuda" else CPU_BATCH_SIZE

    # --- Load data ---
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total rows: {len(df)}")

    # Identify rows with actual review text
    has_text_mask = df[TEXT_COL].notna() & (df[TEXT_COL].str.strip() != "")
    print(f"Rows with review text: {has_text_mask.sum()}")
    print(f"Rows without text (will be skipped): {(~has_text_mask).sum()}\n")

    # --- Fix encoding on all non-null reviews ---
    df.loc[has_text_mask, TEXT_COL] = df.loc[has_text_mask, TEXT_COL].apply(fix_encoding)

    # --- Sample mode ---
    if args.sample > 0:
        sample_indices = df.loc[has_text_mask].head(args.sample).index
        texts_to_translate = df.loc[sample_indices, TEXT_COL].tolist()
        print(f"=== SAMPLE MODE: translating {len(texts_to_translate)} rows ===\n")
    else:
        sample_indices = None
        texts_to_translate = df.loc[has_text_mask, TEXT_COL].tolist()
        print(f"=== FULL MODE: translating {len(texts_to_translate)} rows ===\n")

    # --- Load model ---
    tokenizer, model = load_model(device)

    # --- Translate in batches ---
    all_translations = []
    start_time = time.time()

    for i in tqdm(range(0, len(texts_to_translate), batch_size), desc="Translating"):
        batch = texts_to_translate[i : i + batch_size]
        translated = translate_batch(batch, tokenizer, model, device)
        all_translations.extend(translated)

    elapsed = time.time() - start_time
    speed = len(texts_to_translate) / elapsed if elapsed > 0 else 0
    print(f"\nTranslation complete: {len(texts_to_translate)} rows in {elapsed:.1f}s ({speed:.1f} rows/sec)")

    # --- Sample mode: print comparison table ---
    if args.sample > 0:
        print("\n" + "=" * 100)
        print("ORIGINAL vs TRANSLATED (side-by-side)")
        print("=" * 100)
        for idx, (orig, trans) in enumerate(
            zip(texts_to_translate, all_translations), 1
        ):
            print(f"\n--- Row {idx} ---")
            print(f"  PT: {orig}")
            print(f"  EN: {trans}")
        print("\n" + "=" * 100)
        print("Review the translations above. If quality looks good, run without --sample:")
        print(f"  uv run translate_reviews.py")
        return

    # --- Full mode: save to CSV ---
    # Initialize translated column as empty string
    df[TRANSLATED_COL] = ""
    # Fill in translations for rows that had text
    translation_indices = df.loc[has_text_mask].index
    df.loc[translation_indices, TRANSLATED_COL] = all_translations

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved translated dataset to: {OUTPUT_CSV}")
    print(f"  Total rows: {len(df)}")
    print(f"  Translated rows: {len(all_translations)}")
    print(f"  New column: '{TRANSLATED_COL}'")


if __name__ == "__main__":
    main()
