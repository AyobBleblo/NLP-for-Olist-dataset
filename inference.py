"""
================================================================================
inference.py — BINARY MODE (Negative / Positive only)
================================================================================

Inference module and benchmark suite for the fine-tuned binary DistilBERT model.

Binary Classes:
    Class 0 → Negative  (1–2 star reviews)
    Class 1 → Positive  (4–5 star reviews)
    Neutral (3-star) has been dropped from training and evaluation.

Usage:
------
  1. Programmatic (import into your code):
        from inference import SentimentPredictor
        predictor = SentimentPredictor()
        result = predictor.predict("Excellent product, fast shipping!")
        print(result)

  2. Single text prediction via CLI:
        uv run python inference.py --text "Terrible quality, broke instantly"

  3. Interactive terminal mode:
        uv run python inference.py --interactive

  4. Run 20-case Easy-to-Hard benchmark (10 Neg, 10 Pos):
        uv run python inference.py --test-suite

  5. Use the 3-class model (legacy fallback):
        uv run python inference.py --model-dir models/best_model --test-suite
================================================================================
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==============================================================================
# PATH & CLASS CONFIGURATION (BINARY MODE)
# ==============================================================================
"""
DEFAULT_MODEL_DIR:
    Points to the binary model checkpoint at models/best_model_binary/.
    Override with --model-dir flag to load the legacy 3-class model if needed.

CLASS_LABELS:
    Maps integer class ID to human-readable sentiment label.
    Class 0 → Negative (low-scoring, 1–2 star reviews)
    Class 1 → Positive (high-scoring, 4–5 star reviews)
"""
BASE_DIR: Path = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR: Path = BASE_DIR / "models" / "best_model_binary"

CLASS_LABELS: Dict[int, str] = {
    0: "Negative",
    1: "Positive",
}


# ==============================================================================
# PREDICTOR CLASS
# ==============================================================================
class SentimentPredictor:
    """
    Binary sentiment predictor backed by a fine-tuned DistilBERT model.

    Loads the model checkpoint once on instantiation and exposes two methods:
    - predict()       — single review string inference
    - predict_batch() — batched inference for multiple reviews

    Attributes:
        model_dir (Path):          Path to the binary model checkpoint directory.
        device (torch.device):     Computation device (CUDA or CPU).
        tokenizer (AutoTokenizer): Loaded DistilBERT WordPiece tokenizer.
        model:                     Loaded DistilBERT classification model in eval mode.
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
        device: str = None,
    ) -> None:
        """
        Initializes the predictor by loading model and tokenizer from disk.

        Args:
            model_dir: Path to the saved model directory. Defaults to
                       models/best_model_binary/ (binary model).
                       Pass 'models/best_model' to load the legacy 3-class model.
            device:    Force a specific device ('cuda' or 'cpu'). Auto-selects
                       CUDA if available when left as None.

        Raises:
            FileNotFoundError: If model_dir does not exist on disk.
        """
        self.model_dir: Path = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found at: {self.model_dir}\n"
                "Run train_model.py first to generate the binary model checkpoint."
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict:
        """
        Classifies a single review string as Negative or Positive.

        Args:
            text: Raw English review text string. Empty or whitespace-only
                  strings return label 'Unknown' with zero confidence.

        Returns:
            Dict with fields:
                text (str):              Input review text.
                label_id (int):          Predicted class integer (0 or 1).
                label (str):             Predicted class name ('Negative' or 'Positive').
                confidence (float):      Softmax probability of the predicted class.
                probabilities (Dict):    Full softmax distribution over all classes.
                                         Keys: 'Negative', 'Positive'.
        """
        if not text or not str(text).strip():
            return {
                "text": text,
                "label_id": -1,
                "label": "Unknown",
                "confidence": 0.0,
                "probabilities": {name: 0.0 for name in CLASS_LABELS.values()},
            }

        inputs = self.tokenizer(
            str(text),
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        pred_id = int(np.argmax(probs))
        pred_label = CLASS_LABELS[pred_id]
        confidence = float(probs[pred_id])

        return {
            "text": text,
            "label_id": pred_id,
            "label": pred_label,
            "confidence": confidence,
            "probabilities": {
                CLASS_LABELS[i]: float(probs[i]) for i in range(len(CLASS_LABELS))
            },
        }

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Classifies multiple review strings in one or more batched forward passes.

        Processes texts in chunks of `batch_size` to avoid OOM errors on large inputs.

        Args:
            texts:      List of raw review strings to classify.
            batch_size: Number of reviews to process per batch (default 32).
                        Reduce this value if GPU memory is insufficient.

        Returns:
            List of prediction dicts (same structure as predict() output),
            in the same order as the input `texts` list.
        """
        results: List[Dict] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [str(t) for t in texts[i : i + batch_size]]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs_batch = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for t, probs in zip(batch_texts, probs_batch):
                pred_id = int(np.argmax(probs))
                results.append(
                    {
                        "text": t,
                        "label_id": pred_id,
                        "label": CLASS_LABELS[pred_id],
                        "confidence": float(probs[pred_id]),
                        "probabilities": {
                            CLASS_LABELS[k]: float(probs[k])
                            for k in range(len(CLASS_LABELS))
                        },
                    }
                )
        return results


# ==============================================================================
# BINARY BENCHMARK TEST SUITE (20 Cases)
# ==============================================================================
"""
TEST_SUITE:
    20 curated English-language review cases for binary classification evaluation.
    Organized into two groups of 10, each ranked by difficulty (Level 1 = Easy,
    Level 10 = Hard).

    Difficulty Tiers:
        Easy (Levels 1–3):     Unambiguous, keyword-rich sentiment signal.
        Moderate (Levels 4–6): Clear polarity with minor complexity.
        Tricky (Levels 7–8):   Mixed signals, mild contrast, requires inference.
        Hard (Levels 9–10):    Irony, sarcasm, litotes, or deferred judgment.

    Note: The 10 Neutral test cases from the previous 3-class benchmark have been
    removed. They are semantically out-of-distribution for the binary model.
"""
TEST_SUITE = [
    # ------------------------------------------------------------------
    # CLASS 0: NEGATIVE (10 Cases: Easy → Hard)
    # ------------------------------------------------------------------
    {
        "class_id": 0,
        "class_name": "Negative",
        "level": 1,
        "difficulty": "Easy",
        "text": "Terrible product, completely broken on arrival, total waste of money!",
        "notes": "Classic unambiguous outrage and physical defect.",
    },
    {
        "class_id": 0,
        "class_name": "Negative",
        "level": 2,
        "difficulty": "Easy",
        "text": "Awful experience. The item never worked and the seller is a fraud.",
        "notes": "Strong negative keywords with explicit condemnation.",
    },
    {
        "class_id": 0,
        "class_name": "Negative",
        "level": 3,
        "difficulty": "Easy",
        "text": "Very poor quality, broke after two days of use. Disgusted.",
        "notes": "Fast degradation and visceral negative emotion.",
    },
    {
        "class_id": 0,
        "class_name": "Negative",
        "level": 4,
        "difficulty": "Moderate",
        "text": "I ordered a black phone case and received a pink one. Very disappointed.",
        "notes": "Fulfillment error with stated disappointment.",
    },
    {
        "class_id": 0,
        "class_name": "Negative",
        "level": 5,
        "difficulty": "Moderate",
        "text": "Delivery was over three weeks late and customer service never responded.",
        "notes": "Service-oriented failure, no physical product mention.",
    },
    {
        "class_id": 0,
        "class_name": "Negative",
        "level": 6,
        "difficulty": "Moderate",
        "text": "Missing half of the parts needed for assembly, cannot use it at all.",
        "notes": "Incomplete order — functional failure.",
    },
    {
        "class_id": 0,
        "class_name": "Negative",
        "level": 7,
        "difficulty": "Tricky",
        "text": "The packaging was nice, but the product inside was clearly used and scratched.",
        "notes": "Faint praise (packaging) vs. unacceptable product condition.",
    },
    {
        "class_id": 0,
        "class_name": "Negative",
        "level": 8,
        "difficulty": "Tricky",
        "text": "I thought this brand was reliable, but this particular model is full of defects.",
        "notes": "Brand trust contrasted with model-specific failure.",
    },
    {
        "class_id": 0,
        "class_name": "Negative",
        "level": 9,
        "difficulty": "Hard",
        "text": "Fast shipping, but unfortunately the shoes are so stiff they cause blisters immediately.",
        "notes": "Strong logistical praise conflicts with painful product defect.",
    },
    {
        "class_id": 0,
        "class_name": "Negative",
        "level": 10,
        "difficulty": "Hard",
        "text": "Works fine if you only need it to turn on once before completely shutting down.",
        "notes": "Pure sarcasm and irony — 'Works fine if ...'.",
    },
    # ------------------------------------------------------------------
    # CLASS 1: POSITIVE (10 Cases: Easy → Hard)
    # ------------------------------------------------------------------
    {
        "class_id": 1,
        "class_name": "Positive",
        "level": 1,
        "difficulty": "Easy",
        "text": "Excellent product! High quality, arrived super fast, 100% recommended!",
        "notes": "Unambiguous, enthusiastic 5-star sentiment.",
    },
    {
        "class_id": 1,
        "class_name": "Positive",
        "level": 2,
        "difficulty": "Easy",
        "text": "Wonderful! Exceeded all my expectations, I am extremely satisfied!",
        "notes": "Explicit superlatives and direct satisfaction statement.",
    },
    {
        "class_id": 1,
        "class_name": "Positive",
        "level": 3,
        "difficulty": "Easy",
        "text": "Perfect product and lightning fast shipping! Five stars!",
        "notes": "Direct praise of product and delivery speed.",
    },
    {
        "class_id": 1,
        "class_name": "Positive",
        "level": 4,
        "difficulty": "Moderate",
        "text": "Very good cost-benefit ratio, works perfectly for my daily needs.",
        "notes": "Pragmatic, solid satisfaction without superlatives.",
    },
    {
        "class_id": 1,
        "class_name": "Positive",
        "level": 5,
        "difficulty": "Moderate",
        "text": "Product arrived four days before the deadline, well packed and in great shape.",
        "notes": "Strong fulfillment praise and intact delivery.",
    },
    {
        "class_id": 1,
        "class_name": "Positive",
        "level": 6,
        "difficulty": "Moderate",
        "text": "The fabric is soft and the size fit just right. Great purchase.",
        "notes": "Specific qualitative attributes confirmed with satisfaction.",
    },
    {
        "class_id": 1,
        "class_name": "Positive",
        "level": 7,
        "difficulty": "Tricky",
        "text": "I was a bit skeptical at first because of the low price, but it turned out fantastic.",
        "notes": "Negative framing at start ('skeptical', 'low price') → resolves to delight.",
    },
    {
        "class_id": 1,
        "class_name": "Positive",
        "level": 8,
        "difficulty": "Tricky",
        "text": "Delivery was slightly delayed by the post office, but the outstanding quality made up for it.",
        "notes": "Mild logistical complaint overcome by strong product quality.",
    },
    {
        "class_id": 1,
        "class_name": "Positive",
        "level": 9,
        "difficulty": "Hard",
        "text": "Not bad at all, actually performs much better than the cheaper alternatives.",
        "notes": "Litotes ('Not bad at all') expressing genuine high praise.",
    },
    {
        "class_id": 1,
        "class_name": "Positive",
        "level": 10,
        "difficulty": "Hard",
        "text": "Simple product without any fancy frills, but it never fails to do its job flawlessly.",
        "notes": "Understated praise using negation ('without', 'never fails').",
    },
]


# ==============================================================================
# TEST SUITE RUNNER
# ==============================================================================
def run_test_suite(predictor: SentimentPredictor) -> None:
    """
    Runs the 20-case binary sentiment benchmark against the loaded predictor.

    Displays per-case results with predicted vs expected label, confidence,
    probability breakdown, and difficulty tier. Prints a summary table by
    class and by difficulty tier at the end.

    Args:
        predictor: Initialized SentimentPredictor instance.
    """
    print("\n" + "=" * 80)
    print("RUNNING 20-CASE BINARY INFERENCE BENCHMARK (10 Neg, 10 Pos)")
    print("Ranked from Level 1 (EASY) to Level 10 (HARD)")
    print("=" * 80)
    sys.stdout.flush()

    total_tests = len(TEST_SUITE)
    correct_count: int = 0
    by_class_correct: Dict[int, int] = {0: 0, 1: 0}
    by_class_total: Dict[int, int] = {0: 0, 1: 0}
    by_diff_correct: Dict[str, int] = {"Easy": 0, "Moderate": 0, "Tricky": 0, "Hard": 0}
    by_diff_total: Dict[str, int] = {"Easy": 0, "Moderate": 0, "Tricky": 0, "Hard": 0}

    current_class = None

    for i, test in enumerate(TEST_SUITE, 1):
        if test["class_id"] != current_class:
            current_class = test["class_id"]
            print("\n" + "-" * 80)
            print(
                f">>> TESTING TARGET CLASS: {test['class_name'].upper()} "
                f"(Class {current_class})"
            )
            print("-" * 80)

        result = predictor.predict(test["text"])
        pred_label = result["label"]
        pred_id = result["label_id"]
        conf = result["confidence"]
        probs = result["probabilities"]

        is_correct = pred_id == test["class_id"]
        if is_correct:
            correct_count += 1
            by_class_correct[test["class_id"]] += 1
            by_diff_correct[test["difficulty"]] += 1
            verdict = "[PASS ✓]"
        else:
            verdict = f"[FAIL → {pred_label}]"

        by_class_total[test["class_id"]] += 1
        by_diff_total[test["difficulty"]] += 1

        print(
            f"Case {i:2d}/{total_tests} | Lvl {test['level']:2d} ({test['difficulty']:<8}) | "
            f"Exp: {test['class_name']:<8} | Got: {pred_label:<8} | Conf: {conf:6.2%} | {verdict}"
        )
        print(f"   Text: \"{test['text']}\"")
        prob_str = (
            f"Neg: {probs['Negative']*100:5.1f}% | "
            f"Pos: {probs['Positive']*100:5.1f}%"
        )
        print(f"   Probs: [{prob_str}] | Notes: {test['notes']}")
        print()
        sys.stdout.flush()

    print("=" * 80)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 80)
    print(
        f"Total Accuracy: {correct_count}/{total_tests} "
        f"({correct_count/total_tests*100:.1f}%)\n"
    )

    print("--- By Target Class ---")
    for cid in [0, 1]:
        cname = CLASS_LABELS[cid]
        c_acc = by_class_correct[cid] / by_class_total[cid] * 100
        print(
            f"  Class {cid} ({cname:<8}): "
            f"{by_class_correct[cid]:2d}/{by_class_total[cid]:2d} ({c_acc:5.1f}%)"
        )

    print("\n--- By Difficulty Tier ---")
    for diff in ["Easy", "Moderate", "Tricky", "Hard"]:
        d_acc = by_diff_correct[diff] / by_diff_total[diff] * 100
        print(
            f"  {diff:<10}: "
            f"{by_diff_correct[diff]:2d}/{by_diff_total[diff]:2d} ({d_acc:5.1f}%)"
        )
    print("=" * 80)


# ==============================================================================
# INTERACTIVE MODE
# ==============================================================================
def interactive_mode(predictor: SentimentPredictor) -> None:
    """
    Launches an interactive terminal loop for live binary sentiment prediction.

    Reads one review at a time from stdin, runs the predictor, and prints
    the binary verdict (Negative / Positive) with confidence and probabilities.
    Type 'exit', 'quit', or 'q' to terminate.

    Args:
        predictor: Initialized SentimentPredictor instance.
    """
    print("\n" + "=" * 65)
    print("BINARY INTERACTIVE SENTIMENT PREDICTOR")
    print("Enter any review in English (or 'exit' / 'quit' to stop):")
    print("=" * 65)
    while True:
        try:
            text = input("\nEnter Review > ").strip()
            if text.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive mode.")
                break
            if not text:
                continue
            res = predictor.predict(text)
            p = res["probabilities"]
            print(f"  Verdict:    {res['label'].upper()} (Confidence: {res['confidence']:.2%})")
            print(f"  Neg: {p['Negative']:.1%} | Pos: {p['Positive']:.1%}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================
def main() -> None:
    """
    CLI entry point for the binary sentiment inference module.

    Modes:
    ------
    --text TEXT        : Predict sentiment for a single review string.
    --interactive      : Launch the interactive terminal prediction loop.
    --test-suite       : Run the 20-case Easy-to-Hard binary benchmark.
    (default)          : Same as --test-suite if no flags are provided.

    --model-dir PATH   : Override the default binary model path
                         (default: models/best_model_binary/).
                         Use 'models/best_model' for the legacy 3-class model.
    """
    parser = argparse.ArgumentParser(
        description="DistilBERT Binary Sentiment Inference (Negative / Positive)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Path to saved binary model (default: models/best_model_binary/)",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Single review text to classify",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal mode",
    )
    parser.add_argument(
        "--test-suite",
        action="store_true",
        help="Run 20-case binary Easy-to-Hard benchmark",
    )

    args = parser.parse_args()

    print(f"Loading binary SentimentPredictor from: {args.model_dir}")
    predictor = SentimentPredictor(model_dir=args.model_dir)
    print(f"Model ready | Device: {predictor.device}")

    if args.text:
        res = predictor.predict(args.text)
        print("\nPrediction Result:")
        print(f"  Text:       \"{res['text']}\"")
        print(f"  Sentiment:  {res['label']} (Class {res['label_id']})")
        print(f"  Confidence: {res['confidence']:.2%}")
        print("  Distribution:")
        for name, prob in res["probabilities"].items():
            print(f"    - {name:<8}: {prob:6.2%}")
    elif args.interactive:
        interactive_mode(predictor)
    else:
        # Default: run test suite when no flag is given
        run_test_suite(predictor)


if __name__ == "__main__":
    main()
