import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# These are the sensors we identified as most informative in EDA
# sensor_4 and sensor_7 had strongest correlation with RUL
# sensor_9 and sensor_14 had highest variance
# We engineer features from all of them
TOP_SENSORS = ["sensor_4", "sensor_7", "sensor_9", 
               "sensor_14", "sensor_3", "sensor_17"]

# Window sizes for rolling features
# 5  = short term trend (last 5 cycles)
# 10 = medium term trend
# 30 = long term trend
WINDOWS = [5, 10, 30]

# Lag periods
# How many cycles back to look
LAGS = [1, 3, 5]


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each top sensor, calculate rolling mean and rolling std
    over multiple window sizes.
    
    We group by engine_id first — rolling features should only
    look at cycles from the SAME engine, not bleed across engines.
    """
    for sensor in TOP_SENSORS:
        if sensor not in df.columns:
            continue  # skip if this sensor was dropped in pipeline
            
        for window in WINDOWS:
            # Rolling mean — smoothed trend
            df[f"{sensor}_rmean_{window}"] = (
                df.groupby("engine_id")[sensor]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            
            # Rolling std — instability measure
            df[f"{sensor}_rstd_{window}"] = (
                df.groupby("engine_id")[sensor]
                .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
            )
    
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each top sensor, add the sensor value from N cycles ago.
    This captures the rate of change over time.
    
    Again grouped by engine_id — lag should not cross engine boundaries.
    """
    for sensor in TOP_SENSORS:
        if sensor not in df.columns:
            continue
            
        for lag in LAGS:
            df[f"{sensor}_lag_{lag}"] = (
                df.groupby("engine_id")[sensor]
                .transform(lambda x: x.shift(lag).bfill())
            )
    
    return df


def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple cycle-based features.
    These tell the model how far into its life the engine is.
    """
    # What cycle is this engine on
    df["cycle_norm"] = (
        df.groupby("engine_id")["cycle"]
        .transform(lambda x: x / x.max())
    )
    
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns the model doesn't need.
    setting_1, setting_2, setting_3 are operational settings
    that don't change with degradation in FD001.
    """
    cols_to_drop = ["setting_1", "setting_2", "setting_3"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    return df


def normalize_features(df: pd.DataFrame, 
                        scaler=None, 
                        fit: bool = True):
    """
    Scale all feature columns to mean=0, std=1.
    
    Why normalize?
    XGBoost doesn't strictly need it but other models like
    Logistic Regression do. Normalizing now means your
    feature matrix works with ANY model you try.
    
    fit=True  → fit the scaler on this data (use for training data)
    fit=False → use existing scaler (use for test/new data)
    """
    # These columns are not features — don't scale them
    exclude_cols = ["engine_id", "cycle", "RUL", 
                    "will_fail", "cycle_norm"]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    if fit:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    
    return df, scaler


def save_scaler(scaler, path: str = "models/scaler.pkl") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")


def build_features(df: pd.DataFrame, 
                   fit_scaler: bool = True,
                   scaler=None):
    """
    Master function — runs the full feature engineering pipeline.
    
    fit_scaler=True  → training data (fit + transform)
    fit_scaler=False → new data (transform only, pass existing scaler)
    """
    print("Building features...")
    print(f"  Input shape: {df.shape}")
    
    df = add_rolling_features(df)
    print(f"  After rolling features: {df.shape}")
    
    df = add_lag_features(df)
    print(f"  After lag features: {df.shape}")
    
    df = add_cycle_features(df)
    
    df = drop_unnecessary_columns(df)
    
    df, scaler = normalize_features(df, scaler=scaler, fit=fit_scaler)
    
    print(f"  Final shape: {df.shape}")
    print("Feature engineering complete.")
    
    return df, scaler
