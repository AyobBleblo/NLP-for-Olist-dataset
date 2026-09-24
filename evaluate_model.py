"""
================================================================================
evaluate_model.py — BINARY MODE (Negative / Positive only)
================================================================================

Evaluates the best fine-tuned binary DistilBERT model on the held-out test set
(data/test.csv — Neutral reviews already removed by prepare_data.py).

What this script does:
-----------------------
1. Loads the binary model checkpoint from models/best_model_binary/
2. Runs inference on all test samples
3. Computes:
    - Overall Accuracy
    - Macro F1 (unweighted mean across Negative and Positive)
    - Weighted F1 (support-weighted across both classes)
    - Per-class Precision, Recall, F1 (Negative + Positive)
    - 2×2 Confusion Matrix
4. Checks the following binary success criteria:
    * Accuracy  > 88%
    * Macro F1  > 0.82
    * Negative F1 > 0.78  (harder class — only 29% of test samples)
5. Displays high-confidence and low-confidence sample predictions
6. Saves evaluation results to models/evaluation_results_binary.json

Expected Binary Performance (vs 3-class baseline):
---------------------------------------------------
    3-class Accuracy:   82.54%   →   Binary target:  > 88%
    3-class Macro F1:   0.6826   →   Binary target:  > 0.82
    Neutral F1:         0.3155   →   Dropped entirely
================================================================================
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

# ==============================================================================
# PATH CONFIGURATION (BINARY MODE)
# ==============================================================================
"""
Path Constants:
- BEST_MODEL_DIR:  Binary model checkpoint (saved by binary train_model.py).
                   Separate from the 3-class model at models/best_model/.
- TEST_CSV:        Binary test split — Neutral rows already removed by prepare_data.py.
- RESULTS_JSON:    Saves binary evaluation report separately from the 3-class report.
"""
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
BEST_MODEL_DIR: Path = MODELS_DIR / "best_model_binary"
TEST_CSV: Path = DATA_DIR / "test.csv"
RESULTS_JSON: Path = MODELS_DIR / "evaluation_results_binary.json"

# ==============================================================================
# BINARY CLASS CONFIGURATION
# ==============================================================================
"""
CLASS_NAMES:
    Ordered list of class labels by integer index.
    Class 0 → Negative (1–2 star reviews)
    Class 1 → Positive (4–5 star reviews)
    Neutral (3-star) has been dropped from the dataset entirely.

NUM_CLASSES: 2 (binary)

BATCH_SIZE:
    Larger batch size (64) is safe during inference since no gradients
    are stored and FP16 AMP is used for fast forward passes.

MAX_LENGTH:
    Matches training truncation (256 subword tokens).
"""
CLASS_NAMES: List[str] = ["Negative", "Positive"]
NUM_CLASSES: int = 2
BATCH_SIZE: int = 64
MAX_LENGTH: int = 256


# ==============================================================================
# DATASET CLASS (shared structure with train_model.py)
# ==============================================================================
class ReviewDataset(Dataset):
    """
    PyTorch Dataset for tokenizing binary-labeled review text.

    Attributes:
        texts (List[str]):         Raw translated review strings.
        labels (List[int]):        Binary class IDs: 0 (Negative) or 1 (Positive).
        tokenizer (AutoTokenizer): Pretrained DistilBERT WordPiece tokenizer.
        max_length (int):          Maximum token sequence length (default 256).
    """

    def __init__(
        self,
        texts,
        labels,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ) -> None:
        """
        Args:
            texts:      Iterable of review text strings.
            labels:     Iterable of integer class IDs (0 or 1).
            tokenizer:  Hugging Face tokenizer (loaded from binary model dir).
            max_length: Sequence truncation limit.
        """
        self.texts: List[str] = list(texts)
        self.labels: List[int] = list(labels)
        self.tokenizer = tokenizer
        self.max_length: int = max_length

    def __len__(self) -> int:
        """Returns total number of test samples."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Tokenizes a single review and returns input_ids, attention_mask, and label.
        Padding is deferred to DataCollatorWithPadding for dynamic batch-level padding.
        """
        text: str = str(self.texts[idx])
        label: int = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        encoding["label"] = label
        return encoding


def main() -> None:
    """
    End-to-end binary test set evaluation pipeline.

    Stages:
    -------
    1. Validate that the binary model checkpoint and test CSV exist.
    2. Load binary model from models/best_model_binary/ and set to eval mode.
    3. Tokenize and build DataLoader over the binary test set.
    4. Run FP16 forward passes to collect all predictions and class probabilities.
    5. Compute accuracy, macro/weighted F1, per-class F1, and confusion matrix.
    6. Check binary success criteria thresholds.
    7. Display 3 sample predictions per class (highest and lowest confidence).
    8. Serialize full results to models/evaluation_results_binary.json.
    """
    print("=" * 65)
    print("DISTILBERT BINARY SENTIMENT — TEST SET EVALUATION")
    print("=" * 65)

    if not BEST_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Binary model not found at: {BEST_MODEL_DIR}\n"
            "Please run: uv run python train_model.py"
        )
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found at: {TEST_CSV}\n"
            "Please run: uv run python prepare_data.py"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp: bool = torch.cuda.is_available()
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # Load binary test split
    test_df = pd.read_csv(TEST_CSV)
    print(f"Test samples: {len(test_df):,}")

    # Verify no Neutral class leaked into the test set
    unique_labels = sorted(test_df["sentiment"].unique())
    assert unique_labels == [0, 1], (
        f"Expected binary labels [0, 1] but found {unique_labels} in test.csv. "
        "Re-run prepare_data.py."
    )

    print(f"\nLoading binary model checkpoint from: {BEST_MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR)
    model.to(device)
    model.eval()
    print("Model loaded. Running inference...")
    sys.stdout.flush()

    test_dataset = ReviewDataset(
        test_df["review_comment_translated"],
        test_df["sentiment"],
        tokenizer,
        max_length=MAX_LENGTH,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

            preds = torch.argmax(probs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds_arr = np.array(all_preds)
    all_labels_arr = np.array(all_labels)
    all_probs_arr = np.array(all_probs)

    # ==============================================================================
    # COMPUTE METRICS
    # ==============================================================================
    acc = accuracy_score(all_labels_arr, all_preds_arr)
    macro_f1 = f1_score(all_labels_arr, all_preds_arr, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels_arr, all_preds_arr, average="weighted", zero_division=0)
    clf_report_dict = classification_report(
        all_labels_arr,
        all_preds_arr,
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True,
    )
    clf_report_text = classification_report(
        all_labels_arr,
        all_preds_arr,
        target_names=CLASS_NAMES,
        digits=4,
    )
    conf_mat = confusion_matrix(all_labels_arr, all_preds_arr)

    print("\n" + "=" * 65)
    print("BINARY CLASSIFICATION REPORT")
    print("=" * 65)
    print(clf_report_text)

    # ==============================================================================
    # CONFUSION MATRIX (Binary: 2×2)
    # ==============================================================================
    print("CONFUSION MATRIX (rows = True, columns = Predicted):")
    print(f"{'':15} {'Predicted Neg':>15} {'Predicted Pos':>15}")
    for i, row_label in enumerate(CLASS_NAMES):
        row_str = "  ".join(f"{count:15d}" for count in conf_mat[i])
        print(f"True {row_label:<10}: {row_str}")

    # ==============================================================================
    # SUCCESS CRITERIA — BINARY THRESHOLDS
    # ==============================================================================
    """
    Binary Success Criteria (stricter than 3-class because task is simpler):
    - Accuracy  > 88%  : Binary classification should be significantly more accurate.
    - Macro F1  > 0.82 : Both classes should be well-captured.
    - Neg F1    > 0.78 : Minority class (29%) — critical for recommendation systems.
    """
    neg_f1 = clf_report_dict["Negative"]["f1-score"]
    pos_f1 = clf_report_dict["Positive"]["f1-score"]
    crit_acc = acc >= 0.88
    crit_macro = macro_f1 >= 0.82
    crit_neg = neg_f1 >= 0.78

    print("\n" + "=" * 65)
    print("BINARY SUCCESS CRITERIA EVALUATION")
    print("=" * 65)
    print(
        f"1. Overall Accuracy  > 88%:  {acc*100:6.2f}% "
        f"-> {'PASSED ✓' if crit_acc else 'FAILED ✗'}"
    )
    print(
        f"2. Macro F1-Score    > 0.82: {macro_f1:6.4f} "
        f"-> {'PASSED ✓' if crit_macro else 'FAILED ✗'}"
    )
    print(
        f"3. Negative Class F1 > 0.78: {neg_f1:6.4f} "
        f"-> {'PASSED ✓' if crit_neg else 'FAILED ✗'}"
    )
    print(
        f"4. Positive Class F1 (info): {pos_f1:6.4f}"
    )

    all_passed = crit_acc and crit_macro and crit_neg
    print(f"\nOverall: {'ALL CRITERIA PASSED ✓' if all_passed else 'ONE OR MORE CRITERIA FAILED ✗'}")

    # ==============================================================================
    # SAMPLE PREDICTIONS (3 per class)
    # ==============================================================================
    print("\n" + "=" * 65)
    print("SAMPLE PREDICTIONS (3 per class)")
    print("=" * 65)
    test_df = test_df.copy()
    test_df["pred_sentiment"] = all_preds_arr
    test_df["pred_name"] = [CLASS_NAMES[p] for p in all_preds_arr]
    test_df["confidence"] = [float(np.max(p)) for p in all_probs_arr]

    for c_idx, c_name in enumerate(CLASS_NAMES):
        print(f"\nPredicted as {c_name.upper()}:")
        # Show 3 correctly predicted samples (highest confidence)
        correct_mask = (test_df["pred_sentiment"] == c_idx) & (test_df["sentiment"] == c_idx)
        subset = test_df[correct_mask].nlargest(3, "confidence")
        for _, row in subset.iterrows():
            txt = str(row["review_comment_translated"])[:90].replace("\n", " ")
            true_name = CLASS_NAMES[int(row["sentiment"])]
            print(
                f"  [CORRECT | Conf: {row['confidence']:.2%}] "
                f"(True: {true_name}): \"{txt}\""
            )

    # ==============================================================================
    # SAVE EVALUATION RESULTS
    # ==============================================================================
    results = {
        "mode": "binary",
        "model_dir": str(BEST_MODEL_DIR),
        "test_samples": int(len(test_df)),
        "class_names": CLASS_NAMES,
        "overall_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": {
            "Negative": float(neg_f1),
            "Positive": float(pos_f1),
        },
        "classification_report": clf_report_dict,
        "confusion_matrix": conf_mat.tolist(),
        "success_criteria": {
            "accuracy_target_88pct": {
                "threshold": 0.88,
                "value": float(acc),
                "passed": bool(crit_acc),
            },
            "macro_f1_target_82": {
                "threshold": 0.82,
                "value": float(macro_f1),
                "passed": bool(crit_macro),
            },
            "negative_f1_target_78": {
                "threshold": 0.78,
                "value": float(neg_f1),
                "passed": bool(crit_neg),
            },
            "all_passed": bool(all_passed),
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print(f"Binary evaluation results saved to: {RESULTS_JSON}")
    print("=" * 65)


if __name__ == "__main__":
    main()
