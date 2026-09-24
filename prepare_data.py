"""
================================================================================
prepare_data.py  —  BINARY MODE (Negative / Positive only)
================================================================================

Data Preparation for Binary Sentiment Classification:
------------------------------------------------------
This script prepares the translated Olist customer reviews for 2-class (binary)
sentiment classification by:

1. Loading the translated CSV (data/olist_order_reviews_translated.csv)
2. Filtering out rows with no review text (58,275 rating-only records removed)
3. **Dropping all 3-star (Neutral) reviews** — 3,556 records removed.
   Rationale: 3-star reviews carry inherently ambiguous mixed sentiment (partial
   praise + partial complaint) and significantly reduce model decision boundary clarity.
4. Remapping the remaining scores to binary integer labels:
       review_score 1, 2  ->  0  (Negative)
       review_score 4, 5  ->  1  (Positive)
5. Performing an 80/10/10 stratified split that preserves the Negative/Positive ratio.
6. Saving three CSV files: data/train.csv, data/val.csv, data/test.csv

Binary Class Distribution (reviews with text, excluding Neutral):
    Total Kept:   37,393 reviews
    Negative (0): 10,888 reviews (29.1%)
    Positive (1): 26,505 reviews (70.9%)

Expected Imbalance (manageable for binary):
    The 70.9% Positive skew is handled via weighted CrossEntropyLoss in train_model.py.
================================================================================
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
DATA_DIR: Path = Path(__file__).resolve().parent / "data"
INPUT_CSV: Path = DATA_DIR / "olist_order_reviews_translated.csv"
TRAIN_CSV: Path = DATA_DIR / "train.csv"
VAL_CSV: Path = DATA_DIR / "val.csv"
TEST_CSV: Path = DATA_DIR / "test.csv"

# ==============================================================================
# BINARY CLASS MAPPING
# ==============================================================================
"""
SCORE_TO_SENTIMENT:
    Maps review_score (1–5) to binary integer class IDs.
    3-star reviews are NOT included — they are filtered before applying this map.
    review_score 1 -> Class 0 (Negative)
    review_score 2 -> Class 0 (Negative)
    review_score 4 -> Class 1 (Positive)
    review_score 5 -> Class 1 (Positive)

SENTIMENT_NAMES:
    Human-readable class labels indexed by integer class ID.
    Used for display and verification printing.
"""
SCORE_TO_SENTIMENT: Dict[int, int] = {
    1: 0,  # Negative
    2: 0,  # Negative
    4: 1,  # Positive
    5: 1,  # Positive
}

SENTIMENT_NAMES: Dict[int, str] = {
    0: "Negative",
    1: "Positive",
}


def main() -> None:
    """
    Executes the full binary data preparation pipeline:

    Steps:
    ------
    1. Load and validate the translated reviews CSV.
    2. Drop rows without translated review text.
    3. Drop all Neutral (3-star) reviews to produce a clean binary dataset.
    4. Map remaining review scores to binary class IDs (0: Negative, 1: Positive).
    5. Perform an 80/10/10 stratified split preserving the class ratio.
    6. Save train.csv, val.csv, and test.csv to data/ directory.
    7. Print class distribution verification tables for all three splits.
    """
    print(f"Loading raw translated reviews from: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}\n"
            "Run translate_reviews.py first."
        )

    df = pd.read_csv(INPUT_CSV)
    print(f"Total raw records loaded:         {len(df):,}")

    # Step 1: Remove rows with no translated text
    has_text = df["review_comment_translated"].notna() & (
        df["review_comment_translated"].astype(str).str.strip() != ""
    )
    df_with_text = df[has_text].copy()
    print(f"Reviews with translated text:     {len(df_with_text):,} "
          f"(dropped {len(df) - len(df_with_text):,} without text)")

    # Step 2: Drop Neutral (3-star) reviews
    neutral_mask = df_with_text["review_score"] == 3
    n_neutral_dropped = neutral_mask.sum()
    df_binary = df_with_text[~neutral_mask].copy()
    print(f"Neutral (3-star) rows dropped:    {n_neutral_dropped:,}")
    print(f"Clean binary rows remaining:      {len(df_binary):,}")

    # Step 3: Map scores to binary labels
    df_binary["sentiment"] = df_binary["review_score"].map(SCORE_TO_SENTIMENT)
    if df_binary["sentiment"].isna().any():
        unmapped = df_binary[df_binary["sentiment"].isna()]["review_score"].unique()
        raise ValueError(
            f"Found unmapped review scores: {unmapped}. "
            "Check SCORE_TO_SENTIMENT mapping."
        )
    df_binary["sentiment"] = df_binary["sentiment"].astype(int)
    df_binary["sentiment_name"] = df_binary["sentiment"].map(SENTIMENT_NAMES)

    # Step 4: Stratified 80/10/10 split
    train_df, temp_df = train_test_split(
        df_binary,
        test_size=0.20,
        random_state=42,
        stratify=df_binary["sentiment"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["sentiment"],
    )

    # Step 5: Report and verify distributions
    print("\n" + "=" * 60)
    print("BINARY DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"Train set: {len(train_df):,} samples "
          f"({len(train_df)/len(df_binary)*100:.1f}%)")
    print(f"Val set:   {len(val_df):,} samples "
          f"({len(val_df)/len(df_binary)*100:.1f}%)")
    print(f"Test set:  {len(test_df):,} samples "
          f"({len(test_df)/len(df_binary)*100:.1f}%)")

    def print_distribution(name: str, split_df: pd.DataFrame) -> None:
        """Prints the class count and percentage for a given split DataFrame."""
        print(f"\n--- {name} Class Distribution ---")
        counts = split_df["sentiment"].value_counts().sort_index()
        for cls_idx, count in counts.items():
            pct = (count / len(split_df)) * 100
            print(f"  Class {cls_idx} ({SENTIMENT_NAMES[cls_idx]:<8}): "
                  f"{count:6,d}  ({pct:5.2f}%)")

    print_distribution("Total Binary Dataset", df_binary)
    print_distribution("Train Set", train_df)
    print_distribution("Val Set", val_df)
    print_distribution("Test Set", test_df)

    # Step 6: Save to disk
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print("\n" + "=" * 60)
    print("Files saved successfully:")
    print(f"  Train: {TRAIN_CSV} ({TRAIN_CSV.stat().st_size / 1024:.1f} KB)")
    print(f"  Val:   {VAL_CSV} ({VAL_CSV.stat().st_size / 1024:.1f} KB)")
    print(f"  Test:  {TEST_CSV} ({TEST_CSV.stat().st_size / 1024:.1f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
