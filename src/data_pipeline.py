import pandas as pd
import numpy as np
import os

# NASA CMAPSS dataset has no headers — we define them manually
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]       # 21 sensors
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]      # 3 operational settings

COLUMN_NAMES = ["engine_id", "cycle"] + SETTING_COLS + SENSOR_COLS


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load a raw CMAPSS txt file into a DataFrame.
    Handles the space-separated format with no header.
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",       # space-separated
        header=None,
        names=COLUMN_NAMES,
        index_col=False
    )
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life (RUL) for each engine.
    RUL = max cycle for that engine - current cycle
    """
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycles, on="engine_id", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)

    return df


def add_failure_label(df: pd.DataFrame, threshold: int = 30) -> pd.DataFrame:
    """
    Binary label: 1 if engine will fail within `threshold` cycles, else 0.
    This is what our classifier will predict.
    """
    df["will_fail"] = (df["RUL"] <= threshold).astype(int)
    return df


def drop_low_variance_sensors(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Some sensors in CMAPSS have near-zero variance — they carry no info.
    Drop them to reduce noise.
    """
    sensor_data = df[SENSOR_COLS]
    variances = sensor_data.var()
    low_var = variances[variances < threshold].index.tolist()

    if low_var:
        print(f"Dropping low-variance sensors: {low_var}")
        df.drop(columns=low_var, inplace=True)

    return df


def save_processed(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to: {output_path}")


def load_and_process(raw_filepath: str, output_filepath: str) -> pd.DataFrame:
    """
    Full pipeline: load → add RUL → add label → drop useless sensors → save
    """
    print(f"Loading data from: {raw_filepath}")
    df = load_raw_data(raw_filepath)
    print(f"  Shape: {df.shape}")

    df = add_rul(df)
    df = add_failure_label(df, threshold=30)
    df = drop_low_variance_sensors(df)

    save_processed(df, output_filepath)
    print("Pipeline complete.")

    return df