"""
Quality assessment for translated reviews.

Checks for:
1. Empty/missing translations
2. Hallucinations (output much longer than input)
3. Untranslated text (output ≈ input)
4. Very short input issues (< 5 chars)
5. Random sample for manual review
"""

import io
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np


INPUT_CSV = "data/olist_order_reviews_translated.csv"
TEXT_COL = "review_comment_message"
TRANSLATED_COL = "review_comment_translated"


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Total rows: {len(df)}")

    # Filter to rows that had original text
    has_text = df[TEXT_COL].notna() & (df[TEXT_COL].str.strip() != "")
    translated = df[has_text].copy()
    print(f"Rows with original text: {len(translated)}")

    has_translation = translated[TRANSLATED_COL].notna() & (translated[TRANSLATED_COL].str.strip() != "")
    print(f"Rows with translation: {has_translation.sum()}")
    print(f"Rows missing translation: {(~has_translation).sum()}")

    translated = translated[has_translation].copy()

    # --- Length analysis ---
    translated["orig_len"] = translated[TEXT_COL].str.len()
    translated["trans_len"] = translated[TRANSLATED_COL].str.len()
    translated["len_ratio"] = translated["trans_len"] / translated["orig_len"]

    print("\n" + "=" * 80)
    print("LENGTH STATISTICS")
    print("=" * 80)
    print(f"  Avg original length:    {translated['orig_len'].mean():.0f} chars")
    print(f"  Avg translated length:  {translated['trans_len'].mean():.0f} chars")
    print(f"  Avg length ratio:       {translated['len_ratio'].mean():.2f}x")
    print(f"  Median length ratio:    {translated['len_ratio'].median():.2f}x")

    # --- Check 1: Possible hallucinations (translation >> original) ---
    hallucinated = translated[translated["len_ratio"] > 3.0]
    print(f"\n⚠ Possible hallucinations (translation > 3x input): {len(hallucinated)}")
    if len(hallucinated) > 0:
        for _, row in hallucinated.head(5).iterrows():
            print(f"  PT ({row['orig_len']} chars): {row[TEXT_COL][:80]}")
            print(f"  EN ({row['trans_len']} chars): {row[TRANSLATED_COL][:80]}")
            print()

    # --- Check 2: Untranslated (output ≈ input) ---
    translated["same"] = translated[TEXT_COL].str.lower().str.strip() == translated[TRANSLATED_COL].str.lower().str.strip()
    untranslated = translated[translated["same"]]
    print(f"⚠ Possibly untranslated (output = input): {len(untranslated)}")
    if len(untranslated) > 0:
        for _, row in untranslated.head(5).iterrows():
            print(f"  PT: {row[TEXT_COL][:80]}")
            print(f"  EN: {row[TRANSLATED_COL][:80]}")
            print()

    # --- Check 3: Very short inputs (prone to errors) ---
    short_inputs = translated[translated["orig_len"] <= 5]
    print(f"⚠ Very short inputs (≤5 chars): {len(short_inputs)}")
    if len(short_inputs) > 0:
        sample_short = short_inputs.sample(min(10, len(short_inputs)), random_state=42)
        for _, row in sample_short.iterrows():
            print(f"  PT: '{row[TEXT_COL]}' → EN: '{row[TRANSLATED_COL]}'")
        print()

    # --- Check 4: Sentiment spot-check ---
    # Look for known negative words in PT that should map to negative in EN
    print("=" * 80)
    print("SENTIMENT SPOT-CHECK (negative PT reviews)")
    print("=" * 80)
    neg_keywords = ["péssimo", "horrível", "terrível", "lixo", "não recomendo", "pior", "decepcion"]
    neg_mask = translated[TEXT_COL].str.lower().str.contains("|".join(neg_keywords), na=False)
    neg_reviews = translated[neg_mask]
    print(f"Found {len(neg_reviews)} reviews with negative PT keywords")
    if len(neg_reviews) > 0:
        sample_neg = neg_reviews.sample(min(10, len(neg_reviews)), random_state=42)
        for _, row in sample_neg.iterrows():
            print(f"  PT: {row[TEXT_COL][:100]}")
            print(f"  EN: {row[TRANSLATED_COL][:100]}")
            print()

    # --- Random sample for manual review ---
    print("=" * 80)
    print("RANDOM SAMPLE (20 rows for manual review)")
    print("=" * 80)
    sample = translated.sample(20, random_state=123)
    for idx, (_, row) in enumerate(sample.iterrows(), 1):
        print(f"\n--- Sample {idx} (score: {row['review_score']}) ---")
        print(f"  PT: {row[TEXT_COL]}")
        print(f"  EN: {row[TRANSLATED_COL]}")

    # --- Summary ---
    total = len(translated)
    print("\n" + "=" * 80)
    print("QUALITY SUMMARY")
    print("=" * 80)
    print(f"  Total translated:       {total}")
    print(f"  Possible hallucinations: {len(hallucinated)} ({100*len(hallucinated)/total:.1f}%)")
    print(f"  Untranslated:           {len(untranslated)} ({100*len(untranslated)/total:.1f}%)")
    print(f"  Very short inputs:      {len(short_inputs)} ({100*len(short_inputs)/total:.1f}%)")
    print(f"  Normal translations:    {total - len(hallucinated) - len(untranslated)} ({100*(total - len(hallucinated) - len(untranslated))/total:.1f}%)")


if __name__ == "__main__":
    main()
