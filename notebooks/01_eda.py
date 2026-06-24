# ============================================================
# Notebook 01 — Exploratory Data Analysis
# Run this cell by cell in Jupyter, or as a plain Python script
# ============================================================

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from src.data_pipeline import load_and_process, SENSOR_COLS, SETTING_COLS

# ── CONFIG ───────────────────────────────────────────────────
RAW_FILE   = "../data/raw/train_FD001.txt"
CLEAN_FILE = "../data/processed/train_FD001_clean.csv"

# ── 1. LOAD & PROCESS ────────────────────────────────────────
df = load_and_process(RAW_FILE, CLEAN_FILE)

print("\n── Basic Info ──────────────────────────────────")
print(f"Total rows     : {len(df):,}")
print(f"Engines        : {df['engine_id'].nunique()}")
print(f"Columns        : {df.shape[1]}")
print(f"Missing values : {df.isnull().sum().sum()}")
print(f"\nClass balance (will_fail):")
print(df["will_fail"].value_counts(normalize=True).round(3).to_string())

# ── 2. RUL DISTRIBUTION ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("RUL Distribution", fontsize=14, fontweight="bold")

axes[0].hist(df["RUL"], bins=50, color="#4A90D9", edgecolor="white")
axes[0].set_title("All cycles")
axes[0].set_xlabel("RUL (cycles remaining)")
axes[0].set_ylabel("Count")

# RUL at first observed cycle per engine (health at start)
first_rul = df.groupby("engine_id")["RUL"].max()
axes[1].hist(first_rul, bins=30, color="#E07B54", edgecolor="white")
axes[1].set_title("Max RUL per engine (engine lifespan)")
axes[1].set_xlabel("Max RUL (cycles)")

plt.tight_layout()
plt.savefig("../data/processed/plot_rul_distribution.png", dpi=120)
plt.show()
print("Saved: plot_rul_distribution.png")

# ── 3. SENSOR VARIANCE — which sensors are informative? ──────
# Remaining sensors after low-variance drop
remaining_sensors = [c for c in df.columns if c.startswith("sensor_")]

variances = df[remaining_sensors].var().sort_values(ascending=False)

plt.figure(figsize=(12, 4))
variances.plot(kind="bar", color="#5BA85F", edgecolor="white")
plt.title("Sensor Variance (higher = more informative)", fontweight="bold")
plt.ylabel("Variance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../data/processed/plot_sensor_variance.png", dpi=120)
plt.show()
print("Saved: plot_sensor_variance.png")

# ── 4. TOP SENSOR TRENDS OVER LIFECYCLE ──────────────────────
# Pick top 6 sensors by variance
top_sensors = variances.head(6).index.tolist()

# Average sensor value at each % of engine life
df["life_pct"] = df["cycle"] / df.groupby("engine_id")["cycle"].transform("max")
df["life_bin"] = pd.cut(df["life_pct"], bins=20, labels=False)

avg_by_bin = df.groupby("life_bin")[top_sensors].mean()

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle("Sensor Trends Over Engine Lifecycle\n(0% = new engine → 100% = near failure)",
             fontsize=13, fontweight="bold")

colors = ["#4A90D9", "#E07B54", "#5BA85F", "#9B59B6", "#F39C12", "#1ABC9C"]

for ax, sensor, color in zip(axes.flatten(), top_sensors, colors):
    ax.plot(avg_by_bin.index * 5, avg_by_bin[sensor], color=color, linewidth=2)
    ax.set_title(sensor, fontweight="bold")
    ax.set_xlabel("Life consumed (%)")
    ax.set_ylabel("Avg value")
    ax.axvline(x=85, color="red", linestyle="--", alpha=0.5, label="85% life")

plt.tight_layout()
plt.savefig("../data/processed/plot_sensor_trends.png", dpi=120)
plt.show()
print("Saved: plot_sensor_trends.png")

# ── 5. CORRELATION HEATMAP — top sensors vs RUL ──────────────
corr_data = df[top_sensors + ["RUL"]].corr()
rul_corr = corr_data["RUL"].drop("RUL").sort_values()

plt.figure(figsize=(8, 4))
colors_corr = ["#E07B54" if v < 0 else "#4A90D9" for v in rul_corr]
rul_corr.plot(kind="barh", color=colors_corr)
plt.title("Sensor Correlation with RUL\n(negative = sensor rises as engine degrades)",
          fontweight="bold")
plt.xlabel("Pearson Correlation")
plt.axvline(x=0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("../data/processed/plot_rul_correlation.png", dpi=120)
plt.show()
print("Saved: plot_rul_correlation.png")

# ── 6. SINGLE ENGINE DEEP-DIVE ───────────────────────────────
engine_id = 1
engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")

fig, axes = plt.subplots(3, 2, figsize=(12, 9))
fig.suptitle(f"Engine #{engine_id} — Full Lifecycle Sensor Readings",
             fontsize=13, fontweight="bold")

for ax, sensor, color in zip(axes.flatten(), top_sensors, colors):
    ax.plot(engine_df["cycle"], engine_df[sensor], color=color, linewidth=1.5)
    ax.set_title(sensor)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Value")
    # Mark last 30 cycles (failure zone)
    failure_start = engine_df["cycle"].max() - 30
    ax.axvline(x=failure_start, color="red", linestyle="--",
               alpha=0.7, label="Failure zone starts")

axes[0][0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("../data/processed/plot_engine1_lifecycle.png", dpi=120)
plt.show()
print("Saved: plot_engine1_lifecycle.png")

# ── SUMMARY ──────────────────────────────────────────────────
print("\n── Week 1 Complete! ────────────────────────────")
print(f"Informative sensors to use in Week 2: {top_sensors}")
print("Plots saved to data/processed/")
print("Processed CSV saved to data/processed/train_FD001_clean.csv")
print("\nNext step → 02_feature_engineering.ipynb")
