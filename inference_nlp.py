"""
inference_nlp.py
Binary Sentiment Classification Inference and 20-Case Benchmark (Easy to Hard).

Binary Classes:
    Class 0  →  Negative  (1–2 star reviews)
    Class 1  →  Positive  (4–5 star reviews)
    (Neutral 3-star reviews dropped from training and evaluation)

Usage in Terminal:
    uv run python inference_nlp.py
    (Runs the full 20-case binary benchmark — 10 Negative, 10 Positive)

Interactive mode:
    uv run python inference_nlp.py --interactive

Single text prediction:
    uv run python inference_nlp.py --text "Great product, highly recommend!"

Use legacy 3-class model (old model):
    uv run python inference_nlp.py --model-dir models/best_model
"""

from inference import main

if __name__ == "__main__":
    main()
