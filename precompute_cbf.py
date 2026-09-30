"""
precompute_cbf.py
=================
Run with SYSTEM Python (Anaconda) — NOT with uv:
    python precompute_cbf.py

Reads the CBF Colab artifacts from models/processed/ and saves a single
precomputed NumPy archive (models/processed/cbf_precomputed.npz) that the
Django backend can load with numpy only — no pandas, no scipy, no sklearn.

Output file keys:
    tfidf_data, tfidf_indices, tfidf_indptr, tfidf_shape  — CSR sparse matrix
    num_matrix    — float64 (n_products, 5)  log1p + z-score normalised
    product_ids   — U64 array of external_id strings (row order)
    text_weight   — scalar float (default 0.7)
"""

import os
import sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(BASE, "models", "processed")

tfidf_path   = os.path.join(PROCESSED, "tfidf_matrix.npz")
features_path = os.path.join(PROCESSED, "products_features.parquet")
row_idx_path  = os.path.join(PROCESSED, "tfidf_row_index.parquet")
out_path      = os.path.join(PROCESSED, "cbf_precomputed.npz")

print("Loading TF-IDF matrix ...")
import scipy.sparse as sp
tfidf_matrix = sp.load_npz(tfidf_path)

print("Loading product features ...")
import pandas as pd
features_df = pd.read_parquet(features_path)
row_idx_df  = pd.read_parquet(row_idx_path)

print(f"TF-IDF shape: {tfidf_matrix.shape}")
print(f"Features shape: {features_df.shape}")
print(f"Row index length: {len(row_idx_df)}")

# --- Numeric matrix (mirror notebook: log1p + z-score) ---
num_cols = ["avg_price", "product_weight_g",
            "product_length_cm", "product_height_cm", "product_width_cm"]
available_cols = [c for c in num_cols if c in features_df.columns]
print(f"Numeric columns used: {available_cols}")

num_raw = features_df[available_cols].fillna(0).values.astype(np.float64)
log_mat = np.log1p(num_raw)
mean = log_mat.mean(axis=0)
std  = log_mat.std(axis=0)
std[std == 0] = 1.0
num_matrix = (log_mat - mean) / std

print(f"Numeric matrix shape: {num_matrix.shape}")

# --- Product IDs (row order = tfidf_row_index.parquet) ---
product_ids = row_idx_df["product_id"].values.astype(str)

# --- Save CSR sparse matrix components ---
tfidf_csr = tfidf_matrix.tocsr()

TEXT_WEIGHT = 0.7

print(f"\nSaving precomputed CBF to: {out_path}")
np.savez_compressed(
    out_path,
    tfidf_data    = tfidf_csr.data.astype(np.float32),
    tfidf_indices = tfidf_csr.indices,
    tfidf_indptr  = tfidf_csr.indptr,
    tfidf_shape   = np.array(tfidf_csr.shape),
    num_matrix    = num_matrix.astype(np.float32),
    product_ids   = product_ids,
    text_weight   = np.array([TEXT_WEIGHT]),
)

size_mb = os.path.getsize(out_path) / 1024 / 1024
print(f"[OK] Saved: cbf_precomputed.npz  ({size_mb:.1f} MB)")
print(f"     Products: {len(product_ids)}")
print(f"     TF-IDF vocab: {tfidf_csr.shape[1]}")
print(f"     Text weight: {TEXT_WEIGHT}")
