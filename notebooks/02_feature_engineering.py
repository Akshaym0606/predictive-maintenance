import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.data_pipeline import load_and_process
from src.feature_engineering import build_features, save_scaler

# ── PATHS ────────────────────────────────────────────────────
RAW_FILE        = "../data/raw/train_FD001.txt"
CLEAN_FILE      = "../data/processed/train_FD001_clean.csv"
FEATURES_FILE   = "../data/processed/train_FD001_features.csv"

# ── STEP 1 — Load clean data ─────────────────────────────────
print("Loading clean data...")
df = load_and_process(RAW_FILE, CLEAN_FILE)

# ── STEP 2 — Build features ───────────────────────────────────
df_features, scaler = build_features(df, fit_scaler=True)

# ── STEP 3 — Save feature matrix ──────────────────────────────
df_features.to_csv(FEATURES_FILE, index=False)
print(f"\nFeature matrix saved to: {FEATURES_FILE}")

# ── STEP 4 — Save scaler ──────────────────────────────────────
save_scaler(scaler, path="../models/scaler.pkl")

# ── STEP 5 — Inspect results ──────────────────────────────────
print("\n── Feature Matrix Info ─────────────────────────")
print(f"Shape          : {df_features.shape}")
print(f"Total features : {df_features.shape[1]}")
print(f"\nAll columns:")
for col in df_features.columns:
    print(f"  {col}")

print("\n── Sample — Engine 1, first 5 rows ─────────────")
engine1 = df_features[df_features["engine_id"] == 1].head()
print(engine1[["engine_id", "cycle", 
               "sensor_4", "sensor_4_rmean_10", 
               "sensor_4_rstd_10", "sensor_4_lag_3",
               "will_fail"]].to_string())