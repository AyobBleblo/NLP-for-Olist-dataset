"""
================================================================================
DISTILBERT BINARY SENTIMENT CLASSIFICATION TRAINING PIPELINE
================================================================================

Overview:
---------
This module fine-tunes a pretrained DistilBERT transformer (distilbert-base-uncased)
to perform 2-class (binary) sentiment analysis — Negative vs. Positive — on
translated e-commerce customer reviews from the Brazilian Olist dataset.

Binary vs 3-class:
------------------
The previous 3-class model (Negative/Neutral/Positive) achieved:
    - Overall Accuracy: 82.54%
    - Macro F1:         0.6826
    - Neutral F1:       0.3155  (hardest class — semantically ambiguous)

By dropping all 3-star Neutral reviews from training and evaluation, we gain:
    - Cleaner, more separated decision boundary
    - Expected accuracy improvement to ~88–91%
    - Simpler model output: pure polarity (thumbs up / thumbs down)

Class Distribution (after dropping Neutral):
    Total kept:    37,393 reviews
    Negative (0):  10,888 reviews  (29.1%)
    Positive (1):  26,505 reviews  (70.9%)

The 70.9% / 29.1% skew is handled via inverse-frequency class weighting:
    W(Negative) = N / (2 * N_neg) ≈ 1.72
    W(Positive) = N / (2 * N_pos) ≈ 0.70

Model Outputs:
--------------
Best binary model checkpoint saved to:  models/best_model_binary/
Training log saved to:                  models/training_log_binary.json
================================================================================
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

# ==============================================================================
# DIRECTORY & FILE PATH CONFIGURATION
# ==============================================================================
"""
Path Configuration (Binary Mode):
- BASE_DIR:          Absolute root directory of the project.
- DATA_DIR:          Contains binary train.csv, val.csv (Neutral rows removed).
- MODELS_DIR:        Root directory for model artifacts.
- BEST_MODEL_DIR:    Saves binary model weights separately from the 3-class model
                     at models/best_model_binary/ to preserve the original model.
- TRAINING_LOG_PATH: Binary training history JSON file.
"""
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
BEST_MODEL_DIR: Path = MODELS_DIR / "best_model_binary"
TRAINING_LOG_PATH: Path = MODELS_DIR / "training_log_binary.json"

TRAIN_CSV: Path = DATA_DIR / "train.csv"
VAL_CSV: Path = DATA_DIR / "val.csv"

# ==============================================================================
# MODEL & TRAINING HYPERPARAMETERS
# ==============================================================================
"""
Hyperparameters (Binary Mode):
- MODEL_NAME:    DistilBERT base uncased — same backbone as 3-class model.
- NUM_CLASSES:   2 (Binary: Negative=0, Positive=1).
- BATCH_SIZE:    16 — fits comfortably within 4GB VRAM with FP16 AMP.
- MAX_LENGTH:    256 — maximum subword token truncation length.
- LEARNING_RATE: 2e-5 — standard BERT fine-tuning learning rate.
- WEIGHT_DECAY:  0.01 — L2 regularization on non-bias/non-LayerNorm parameters.
- EPOCHS:        4 — binary tasks typically converge faster, but 4 is safe.
- WARMUP_RATIO:  0.10 — linear warmup over 10% of total steps.
- SEED:          42 — reproducibility seed.
- CLASS_NAMES:   ["Negative", "Positive"] — ordered by class index.
"""
MODEL_NAME: str = "distilbert-base-uncased"
NUM_CLASSES: int = 2
BATCH_SIZE: int = 16
MAX_LENGTH: int = 256
LEARNING_RATE: float = 2e-5
WEIGHT_DECAY: float = 0.01
EPOCHS: int = 4
WARMUP_RATIO: float = 0.1
SEED: int = 42

CLASS_NAMES: List[str] = ["Negative", "Positive"]


# ==============================================================================
# DATASET CLASS
# ==============================================================================
class ReviewDataset(Dataset):
    """
    Custom PyTorch Dataset for binary sentiment classification.

    Tokenizes customer review strings on-the-fly with DistilBERT's WordPiece
    tokenizer. Padding is intentionally deferred to DataCollatorWithPadding
    so each mini-batch is padded dynamically to its own maximum length,
    saving GPU compute and memory.

    Attributes:
        texts (List[str]):          Raw review text strings.
        labels (List[int]):         Integer class IDs — 0 (Negative), 1 (Positive).
        tokenizer (AutoTokenizer):  Pretrained Hugging Face tokenizer.
        max_length (int):           Sequence truncation cutoff (default 256).
    """

    def __init__(
        self,
        texts: Union[List[str], "pd.Series"],
        labels: Union[List[int], "pd.Series"],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ) -> None:
        """
        Args:
            texts:      Review strings (list or pandas Series).
            labels:     Target class IDs (0 or 1).
            tokenizer:  Pretrained tokenizer for subword encoding.
            max_length: Maximum sequence length after truncation.
        """
        self.texts: List[str] = list(texts)
        self.labels: List[int] = list(labels)
        self.tokenizer = tokenizer
        self.max_length: int = max_length

    def __len__(self) -> int:
        """Returns total number of review samples."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Tokenizes a single review and returns a model-ready encoding dict.

        Note: No static padding applied here — DataCollatorWithPadding handles
        dynamic batch-level padding in the DataLoader.

        Returns:
            Dict with keys: 'input_ids', 'attention_mask', 'label'.
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


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds across numpy, PyTorch, and CUDA for full reproducibility.

    Args:
        seed: Integer seed value (default 42).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp_enabled: bool = True,
) -> Dict[str, Any]:
    """
    Runs binary classification inference on the validation set.

    Disables gradient computation (torch.no_grad), runs FP16 forward passes,
    and accumulates predictions to compute cross-entropy loss and all F1 metrics.

    Args:
        model:       Fine-tuned DistilBERT sequence classification model.
        val_loader:  DataLoader of validation mini-batches.
        loss_fn:     Weighted binary CrossEntropyLoss.
        device:      Computation device ('cuda' or 'cpu').
        amp_enabled: Enable FP16 mixed precision (True when GPU available).

    Returns:
        Dict containing:
            val_loss (float):        Average cross-entropy loss.
            val_acc (float):         Overall classification accuracy.
            val_macro_f1 (float):    Unweighted mean F1 across both classes.
            val_weighted_f1 (float): Support-weighted F1 score.
            per_class_f1 (Dict):     {'Negative': float, 'Positive': float}
            all_preds (List[int]):   All predicted class IDs.
            all_labels (List[int]):  All ground-truth class IDs.
    """
    model.eval()
    total_loss: float = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    return {
        "val_loss": avg_loss,
        "val_acc": acc,
        "val_macro_f1": macro_f1,
        "val_weighted_f1": weighted_f1,
        "per_class_f1": {
            CLASS_NAMES[i]: float(per_class_f1[i]) for i in range(NUM_CLASSES)
        },
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def main() -> None:
    """
    Executes the full end-to-end binary DistilBERT training workflow.

    Workflow Stages:
    ----------------
    1. Seed & directory initialization.
    2. Load binary train.csv / val.csv (Neutral rows already excluded by prepare_data.py).
    3. Compute inverse-frequency class weights for weighted CrossEntropyLoss.
    4. Load distilbert-base-uncased with a 2-class classification head.
    5. Build ReviewDatasets and DataLoaders with dynamic collation.
    6. Configure AdamW optimizer (with weight-decay grouping) and linear warmup scheduler.
    7. Train for EPOCHS with FP16 AMP + gradient clipping.
    8. After each epoch, evaluate on val set and save the best Macro F1 checkpoint
       to models/best_model_binary/.
    9. Persist full training telemetry to models/training_log_binary.json.
    """
    set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp: bool = torch.cuda.is_available()

    print("=" * 65)
    print("DISTILBERT BINARY SENTIMENT TRAINING (Negative / Positive)")
    print("=" * 65)
    print(f"Device:            {device}")
    if torch.cuda.is_available():
        print(f"GPU Name:          {torch.cuda.get_device_name(0)}")
        print(f"Mixed Precision:   Enabled (FP16 AMP)")
    print(f"Classes:           {NUM_CLASSES} ({' / '.join(CLASS_NAMES)})")
    print(f"Batch Size:        {BATCH_SIZE}")
    print(f"Learning Rate:     {LEARNING_RATE}")
    print(f"Epochs:            {EPOCHS}")
    print(f"Max Token Length:  {MAX_LENGTH}")
    print(f"Best Model Dir:    {BEST_MODEL_DIR}")
    sys.stdout.flush()

    # Load binary splits (Neutral rows already removed by prepare_data.py)
    print("\nLoading binary dataset splits...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples:   {len(val_df):,}")

    # Validate no Neutral labels remain (safety check)
    unique_labels = sorted(train_df["sentiment"].unique())
    assert unique_labels == [0, 1], (
        f"Expected binary labels [0, 1] but found {unique_labels}. "
        "Re-run prepare_data.py."
    )

    # Compute binary class weights: W_k = N_total / (2 * Count_k)
    class_counts = train_df["sentiment"].value_counts().sort_index()
    total_train = len(train_df)
    weights: List[float] = [
        total_train / (NUM_CLASSES * class_counts[i]) for i in range(NUM_CLASSES)
    ]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    print("\nBinary Class Imbalance Handling - Loss Weights:")
    for i, name in enumerate(CLASS_NAMES):
        count = class_counts[i]
        pct = (count / total_train) * 100
        print(
            f"  Class {i} ({name:<8}): count={count:6,d}  "
            f"({pct:5.2f}%)  ->  weight={weights[i]:.4f}"
        )
    sys.stdout.flush()

    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Load pretrained DistilBERT with 2-class head
    print(f"\nLoading pretrained {MODEL_NAME} with {NUM_CLASSES}-class head...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label={i: name for i, name in enumerate(CLASS_NAMES)},
        label2id={name: i for i, name in enumerate(CLASS_NAMES)},
    )
    model.to(device)

    # Build datasets and DataLoaders
    train_dataset = ReviewDataset(
        train_df["review_comment_translated"],
        train_df["sentiment"],
        tokenizer,
        max_length=MAX_LENGTH,
    )
    val_dataset = ReviewDataset(
        val_df["review_comment_translated"],
        val_df["sentiment"],
        tokenizer,
        max_length=MAX_LENGTH,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
    )

    # AdamW optimizer — exclude 1D params (bias, LayerNorm) from weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    total_steps: int = len(train_loader) * EPOCHS
    warmup_steps: int = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # FP16 gradient scaler
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Total training steps: {total_steps:,} (warmup: {warmup_steps:,})")
    print("\n" + "=" * 65)
    print("STARTING BINARY TRAINING LOOP")
    print("=" * 65)
    sys.stdout.flush()

    best_val_macro_f1: float = -1.0
    best_epoch: int = -1
    history: List[Dict[str, Any]] = []
    start_total_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        running_loss: float = 0.0
        step_count: int = 0

        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        sys.stdout.flush()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            step_count += 1

            # Log every 250 steps or at the final step
            if step % 250 == 0 or step == len(train_loader):
                cur_loss = running_loss / step_count
                lr_cur = scheduler.get_last_lr()[0]
                pct = (step / len(train_loader)) * 100
                print(
                    f"  Step {step:4d}/{len(train_loader)} ({pct:5.1f}%) | "
                    f"Train Loss: {cur_loss:.4f} | LR: {lr_cur:.2e}"
                )
                sys.stdout.flush()

        epoch_train_loss = running_loss / step_count
        epoch_time = time.time() - epoch_start

        # Validation after each epoch
        print("Evaluating on validation set...")
        sys.stdout.flush()
        val_metrics = evaluate(model, val_loader, loss_fn, device, amp_enabled=use_amp)

        print(f"Epoch {epoch} Summary ({epoch_time:.1f}s):")
        print(f"  Train Loss:     {epoch_train_loss:.4f}")
        print(f"  Val Loss:       {val_metrics['val_loss']:.4f}")
        print(f"  Val Accuracy:   {val_metrics['val_acc']*100:.2f}%")
        print(f"  Val Macro F1:   {val_metrics['val_macro_f1']:.4f}")
        print(f"  Val Weight F1:  {val_metrics['val_weighted_f1']:.4f}")
        print(
            f"  Per-class F1:   "
            f"Neg={val_metrics['per_class_f1']['Negative']:.4f} | "
            f"Pos={val_metrics['per_class_f1']['Positive']:.4f}"
        )
        sys.stdout.flush()

        epoch_record = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": val_metrics["val_loss"],
            "val_accuracy": val_metrics["val_acc"],
            "val_macro_f1": val_metrics["val_macro_f1"],
            "val_weighted_f1": val_metrics["val_weighted_f1"],
            "per_class_f1": val_metrics["per_class_f1"],
            "epoch_duration_sec": epoch_time,
        }
        history.append(epoch_record)

        # Save checkpoint if this is the best Macro F1 seen so far
        if val_metrics["val_macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["val_macro_f1"]
            best_epoch = epoch
            print(
                f"  >>> New best Macro F1: {best_val_macro_f1:.4f}! "
                f"Saving checkpoint to {BEST_MODEL_DIR}..."
            )
            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)
            sys.stdout.flush()

    total_duration = time.time() - start_total_time

    print("\n" + "=" * 65)
    print("BINARY TRAINING FINISHED")
    print("=" * 65)
    print(f"Total time:        {total_duration/60:.2f} minutes")
    print(f"Best Val Macro F1: {best_val_macro_f1:.4f} (at Epoch {best_epoch})")
    print(f"Best model saved:  {BEST_MODEL_DIR}")

    # Persist training telemetry
    log_data = {
        "mode": "binary",
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "class_weights": {
            CLASS_NAMES[i]: float(weights[i]) for i in range(NUM_CLASSES)
        },
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "total_training_duration_seconds": total_duration,
        "epochs_history": history,
    }
    with open(TRAINING_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    print(f"Training log saved: {TRAINING_LOG_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
