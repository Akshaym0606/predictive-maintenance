import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, 
    recall_score, roc_auc_score,
    classification_report
)

# ── PATHS ─────────────────────────────────────────────────────
FEATURES_FILE = "data/processed/train_FD001_features.csv"
MODEL_PATH    = "models/model.pkl"
MLFLOW_URI    = "mlruns"


def load_features(filepath: str):
    """
    Load the feature matrix and split into
    X (features) and y (target).
    
    We drop engine_id, cycle, RUL from X because:
    - engine_id is just an identifier, not a feature
    - cycle is encoded in cycle_norm already
    - RUL is what will_fail was derived from — 
      including it would be cheating (data leakage)
    """
    df = pd.read_csv(filepath)
    
    drop_cols = ["engine_id", "cycle", "RUL", "will_fail"]
    X = df.drop(columns=drop_cols)
    y = df["will_fail"]
    
    print(f"Feature matrix shape : {X.shape}")
    print(f"Target distribution  :")
    print(f"  Safe (0)    : {(y==0).sum()} ({(y==0).mean():.1%})")
    print(f"  Failing (1) : {(y==1).sum()} ({(y==1).mean():.1%})")
    
    return X, y


def split_data(X, y, test_size: float = 0.2):
    """
    Split into train and test sets.
    
    stratify=y ensures both splits have the same
    class balance — 73% safe, 27% failing in both
    train and test. Without stratify, you might get
    an unlucky split where test has very few failures.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,       # reproducible split
        stratify=y             # preserve class balance
    )
    
    print(f"\nTrain size : {X_train.shape[0]} rows")
    print(f"Test size  : {X_test.shape[0]} rows")
    
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test) -> dict:
    """
    Calculate all metrics for a trained model.
    Returns a dictionary of metric name → value.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        "f1"        : f1_score(y_test, y_pred),
        "precision" : precision_score(y_test, y_pred),
        "recall"    : recall_score(y_test, y_pred),
        "auc"       : roc_auc_score(y_test, y_prob)
    }


def train_and_log(model, model_name: str, params: dict,
                  X_train, X_test, y_train, y_test) -> dict:
    """
    Train one model and log everything to MLflow.
    
    MLflow tracks:
    - params  → the settings you used (n_estimators etc)
    - metrics → f1, precision, recall, auc
    - model   → the actual trained model file
    """
    with mlflow.start_run(run_name=model_name):
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate(model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, model_name)
        
        # Print results
        print(f"  F1        : {metrics['f1']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  AUC       : {metrics['auc']:.4f}")
        
    return metrics


def save_best_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\nBest model saved to: {path}")


def run_training():
    """
    Master function — runs the full training pipeline.
    Trains 3 models, logs all to MLflow, saves the best.
    """
    
    # ── Setup MLflow ─────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("predictive-maintenance")
    
    # ── Load data ────────────────────────────────────────
    print("=" * 50)
    print("Loading features...")
    print("=" * 50)
    X, y = load_features(FEATURES_FILE)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # ── Define 3 models ──────────────────────────────────
    experiments = [
        {
            "name"   : "LogisticRegression",
            "model"  : LogisticRegression(
                           max_iter=1000,
                           random_state=42
                       ),
            "params" : {
                "model_type" : "LogisticRegression",
                "max_iter"   : 1000
            }
        },
        {
            "name"   : "RandomForest",
            "model"  : RandomForestClassifier(
                           n_estimators=100,
                           max_depth=10,
                           random_state=42,
                           n_jobs=-1
                       ),
            "params" : {
                "model_type"   : "RandomForest",
                "n_estimators" : 100,
                "max_depth"    : 10
            }
        },
        {
            "name"   : "XGBoost",
            "model"  : XGBClassifier(
                           n_estimators=200,
                           max_depth=6,
                           learning_rate=0.05,
                           subsample=0.8,
                           colsample_bytree=0.8,
                           use_label_encoder=False,
                           eval_metric="logloss",
                           random_state=42
                       ),
            "params" : {
                "model_type"       : "XGBoost",
                "n_estimators"     : 200,
                "max_depth"        : 6,
                "learning_rate"    : 0.05,
                "subsample"        : 0.8,
                "colsample_bytree" : 0.8
            }
        }
    ]
    
    # ── Train all models ─────────────────────────────────
    print("\n" + "=" * 50)
    print("Training models...")
    print("=" * 50)
    
    results = []
    
    for exp in experiments:
        metrics = train_and_log(
            model      = exp["model"],
            model_name = exp["name"],
            params     = exp["params"],
            X_train    = X_train,
            X_test     = X_test,
            y_train    = y_train,
            y_test     = y_test
        )
        results.append({
            "name"    : exp["name"],
            "model"   : exp["model"],
            "metrics" : metrics
        })
    
    # ── Pick best model by F1 ────────────────────────────
    print("\n" + "=" * 50)
    print("Results Summary")
    print("=" * 50)
    print(f"{'Model':<25} {'F1':>8} {'Precision':>10} "
          f"{'Recall':>8} {'AUC':>8}")
    print("-" * 65)
    
    for r in results:
        m = r["metrics"]
        print(f"{r['name']:<25} {m['f1']:>8.4f} "
              f"{m['precision']:>10.4f} {m['recall']:>8.4f} "
              f"{m['auc']:>8.4f}")
    
    best = max(results, key=lambda x: x["metrics"]["f1"])
    print(f"\n🏆 Best model : {best['name']}")
    print(f"   F1 Score   : {best['metrics']['f1']:.4f}")
    
    # ── Save best model ──────────────────────────────────
    save_best_model(best["model"], MODEL_PATH)
    
    # ── Detailed report for best model ───────────────────
    print("\n── Detailed Classification Report ──────────────")
    y_pred = best["model"].predict(X_test)
    print(classification_report(y_test, y_pred, 
                                 target_names=["Safe", "Will Fail"]))
    
    print("\n── Feature Importance (Top 15) ─────────────────")
    if hasattr(best["model"], "feature_importances_"):
        importances = pd.Series(
            best["model"].feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
        
        for feat, imp in importances.head(15).items():
            bar = "█" * int(imp * 100)
            print(f"  {feat:<35} {imp:.4f} {bar}")
    
    return best


if __name__ == "__main__":
    run_training()