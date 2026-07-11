import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import io

# ── Load model and scaler once when server starts ─────────────
MODEL_PATH  = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model  = None
    scaler = None

# ── Create FastAPI app ─────────────────────────────────────────
app = FastAPI(
    title="Predictive Maintenance API",
    description="Predict turbofan engine failure using sensor readings",
    version="1.0.0"
)

# ── Define input schema ────────────────────────────────────────
class SensorReading(BaseModel):
    engine_id:        int
    cycle:            int
    sensor_2:         float
    sensor_3:         float
    sensor_4:         float
    sensor_7:         float
    sensor_9:         float
    sensor_11:        float
    sensor_12:        float
    sensor_14:        float
    sensor_17:        float
    sensor_20:        float
    sensor_21:        float
    sensor_4_rmean_5:   float
    sensor_4_rstd_5:    float
    sensor_4_rmean_10:  float
    sensor_4_rstd_10:   float
    sensor_4_rmean_30:  float
    sensor_4_rstd_30:   float
    sensor_7_rmean_5:   float
    sensor_7_rstd_5:    float
    sensor_7_rmean_10:  float
    sensor_7_rstd_10:   float
    sensor_7_rmean_30:  float
    sensor_7_rstd_30:   float
    sensor_9_rmean_5:   float
    sensor_9_rstd_5:    float
    sensor_9_rmean_10:  float
    sensor_9_rstd_10:   float
    sensor_9_rmean_30:  float
    sensor_9_rstd_30:   float
    sensor_14_rmean_5:  float
    sensor_14_rstd_5:   float
    sensor_14_rmean_10: float
    sensor_14_rstd_10:  float
    sensor_14_rmean_30: float
    sensor_14_rstd_30:  float
    sensor_3_rmean_5:   float
    sensor_3_rstd_5:    float
    sensor_3_rmean_10:  float
    sensor_3_rstd_10:   float
    sensor_3_rmean_30:  float
    sensor_3_rstd_30:   float
    sensor_17_rmean_5:  float
    sensor_17_rstd_5:   float
    sensor_17_rmean_10: float
    sensor_17_rstd_10:  float
    sensor_17_rmean_30: float
    sensor_17_rstd_30:  float
    sensor_4_lag_1:     float
    sensor_4_lag_3:     float
    sensor_4_lag_5:     float
    sensor_7_lag_1:     float
    sensor_7_lag_3:     float
    sensor_7_lag_5:     float
    sensor_9_lag_1:     float
    sensor_9_lag_3:     float
    sensor_9_lag_5:     float
    sensor_14_lag_1:    float
    sensor_14_lag_3:    float
    sensor_14_lag_5:    float
    sensor_3_lag_1:     float
    sensor_3_lag_3:     float
    sensor_3_lag_5:     float
    sensor_17_lag_1:    float
    sensor_17_lag_3:    float
    sensor_17_lag_5:    float
    cycle_norm:         float


# ── Helper function ────────────────────────────────────────────
def make_prediction(df: pd.DataFrame) -> list:
    """
    Takes a DataFrame of feature rows.
    Scales them using the saved scaler.
    Returns predictions and confidence scores.
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Check server logs."
        )

    # Drop identifier columns — model doesn't use these
    drop_cols = ["engine_id", "cycle", "RUL", "will_fail"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]

    # ── THE FIX ───────────────────────────────────────────────
    # cycle_norm was excluded from scaling during training
    # so the scaler doesn't know about it
    # We must scale only the columns the scaler was fitted on
    # then add cycle_norm back separately
    cols_to_scale  = [c for c in feature_cols if c != "cycle_norm"]
    cols_no_scale  = [c for c in feature_cols if c == "cycle_norm"]

    X_to_scale = X[cols_to_scale]

    # Scale only known columns
    X_scaled = scaler.transform(X_to_scale)
    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=cols_to_scale
    )

    # Add cycle_norm back unscaled
    if cols_no_scale:
        X_scaled_df["cycle_norm"] = X["cycle_norm"].values

    # Get the exact column order the model was trained on
    # model.feature_names_in_ stores this automatically
    if hasattr(model, "feature_names_in_"):
        X_final = X_scaled_df[model.feature_names_in_]
    else:
        X_final = X_scaled_df
    # ── END FIX ───────────────────────────────────────────────

    # Get predictions and probabilities
    predictions   = model.predict(X_final)
    probabilities = model.predict_proba(X_final)[:, 1]

    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        results.append({
            "row_index"  : i,
            "prediction" : int(pred),
            "confidence" : round(float(prob), 4),
            "message"    : (
                "⚠️ Engine will fail within 30 cycles — schedule maintenance"
                if pred == 1
                else "✅ Engine is operating safely"
            ),
            "risk_level" : (
                "HIGH"   if prob >= 0.7
                else "MEDIUM" if prob >= 0.4
                else "LOW"
            )
        })

    return results


# ── Endpoint 1 — Health check ──────────────────────────────────
@app.get("/")
def root():
    return {
        "status"  : "running",
        "message" : "Predictive Maintenance API is live",
        "docs"    : "/docs"
    }


# ── Endpoint 2 — Single prediction ────────────────────────────
@app.post("/predict")
def predict(reading: SensorReading):
    df = pd.DataFrame([reading.dict()])
    results = make_prediction(df)
    result  = results[0]

    return {
        "engine_id"  : reading.engine_id,
        "cycle"      : reading.cycle,
        "prediction" : result["prediction"],
        "confidence" : result["confidence"],
        "risk_level" : result["risk_level"],
        "message"    : result["message"]
    }


# ── Endpoint 3 — Batch prediction ─────────────────────────────
@app.post("/batch")
async def batch_predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted"
        )

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    print(f"Batch request received: {len(df)} rows")

    results = make_prediction(df)

    return {
        "total_engines"   : len(df),
        "predictions"     : results,
        "high_risk_count" : sum(1 for r in results
                                if r["risk_level"] == "HIGH"),
        "summary"         : (
            f"{sum(1 for r in results if r['prediction']==1)} "
            f"engines predicted to fail within 30 cycles"
        )
    }