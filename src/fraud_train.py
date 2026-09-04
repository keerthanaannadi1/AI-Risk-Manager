"""
fraud_train.py
--------------
Model training for Phase 2: Fraud Transaction Scorer.

What this does:
  1. Loads and prepares data using fraud_features.build_features()
  2. Splits into 70% train / 20% validation / 10% test (stratified)
  3. Trains three models: XGBoost, LightGBM, Voting Ensemble
  4. Tunes decision threshold on the validation set
  5. Picks the best model by AUPRC on validation set
  6. Evaluates the best model on the test set (once, final numbers)
  7. Saves the best model to models/fraud_risk_model.pkl

Minimum targets (from docs/evaluation-strategy.md):
  Precision >= 0.80  (stricter than Phase 1 — blocking legit transactions hurts customers)
  Recall    >= 0.70
  AUPRC     >= 0.75

Usage:
    python src/fraud_train.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(ROOT, "data", "raw", "transactions.csv")
MODEL_DIR  = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_risk_model.pkl")

sys.path.insert(0, ROOT)
from src.fraud_features import build_features

# ── Minimum targets ────────────────────────────────────────────────────────────
# Precision is stricter than Phase 1 because blocking a legit transaction
# directly hurts a real customer trying to make a purchase.
MIN_PRECISION = 0.80
MIN_RECALL    = 0.70
MIN_AUPRC     = 0.75

# ── Hyperparameters ────────────────────────────────────────────────────────────
# scale_pos_weight tells XGBoost to penalise missing a fraud case more heavily.
# Formula: (number of negatives) / (number of positives) = 8500 / 1500 ≈ 5.67
SCALE_POS_WEIGHT = 8500 / 1500

XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": SCALE_POS_WEIGHT,
    "eval_metric":      "aucpr",
    "random_state":     42,
    "n_jobs":           -1,
}

LGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "class_weight":     "balanced",  # LightGBM equivalent of scale_pos_weight
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":          -1,
}

# Starting threshold — will be tuned on the validation set
THRESHOLD = 0.40


# ── Helpers ────────────────────────────────────────────────────────────────────

def evaluate(model, X, y, threshold=THRESHOLD, label="") -> dict:
    """
    Evaluate a model at a given threshold.
    Returns a dict of metrics and prints a summary.
    """
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    precision = precision_score(y, preds, zero_division=0)
    recall    = recall_score(y, preds, zero_division=0)
    f1        = f1_score(y, preds, zero_division=0)
    auc_roc   = roc_auc_score(y, proba)
    auprc     = average_precision_score(y, proba)
    cm        = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()

    if label:
        print(f"\n  ── {label} ──")
    print(f"  Threshold  : {threshold:.2f}")
    print(f"  Precision  : {precision:.4f}  (target >= {MIN_PRECISION})")
    print(f"  Recall     : {recall:.4f}  (target >= {MIN_RECALL})")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  AUC-ROC    : {auc_roc:.4f}")
    print(f"  AUPRC      : {auprc:.4f}  (target >= {MIN_AUPRC})")
    print(f"  Confusion Matrix:")
    print(f"    TN={tn:>4}  FP={fp:>4}  (FP = legit transactions wrongly blocked)")
    print(f"    FN={fn:>4}  TP={tp:>4}  (FN = fraud transactions missed)")

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "auc_roc":   auc_roc,
        "auprc":     auprc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def tune_threshold(model, X_val, y_val) -> float:
    """
    Find the threshold that maximises F1 on the validation set,
    while keeping precision >= MIN_PRECISION.
    This ensures we don't block too many legit transactions to catch fraud.
    """
    proba = model.predict_proba(X_val)[:, 1]
    best_f1, best_thresh = 0.0, 0.5

    for thresh in np.arange(0.20, 0.85, 0.01):
        preds = (proba >= thresh).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        f = f1_score(y_val, preds, zero_division=0)
        if p >= MIN_PRECISION and f > best_f1:
            best_f1, best_thresh = f, thresh

    print(f"  Best threshold (F1={best_f1:.4f}): {best_thresh:.2f}")
    return best_thresh


def check_targets(metrics: dict, model_name: str) -> bool:
    """Check whether a model meets the minimum precision/recall/AUPRC targets."""
    ok = (
        metrics["precision"] >= MIN_PRECISION and
        metrics["recall"]    >= MIN_RECALL    and
        metrics["auprc"]     >= MIN_AUPRC
    )
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"\n  {model_name} targets: {status}")
    return ok


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  AI Risk Manager — Model Training")
    print("  Phase 2: Fraud Transaction Scorer")
    print(SEP)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("\n  [1/6] Loading data ...")
    df = pd.read_csv(DATA_PATH)
    X, y = build_features(df)
    print(f"  X shape : {X.shape}")
    print(f"  y dist  : {(y==0).sum():,} legit / {(y==1).sum():,} fraud")

    # ── 2. Split 70 / 20 / 10 ─────────────────────────────────────────────────
    print("\n  [2/6] Splitting data (70 train / 20 val / 10 test, stratified) ...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2222, stratify=y_temp, random_state=42
        # 0.2222 of 90% ≈ 20% of total
    )
    print(f"  Train : {len(X_train):,}  ({y_train.mean()*100:.1f}% fraud)")
    print(f"  Val   : {len(X_val):,}  ({y_val.mean()*100:.1f}% fraud)")
    print(f"  Test  : {len(X_test):,}  ({y_test.mean()*100:.1f}% fraud)")

    # ── 3. Train models ────────────────────────────────────────────────────────
    print("\n  [3/6] Training models ...")

    print("\n  Training XGBoost ...")
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print("  Training LightGBM ...")
    lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )

    print("  Building Voting Ensemble (XGBoost + LightGBM) ...")
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb_model), ("lgb", lgb_model)],
        voting="soft",   # averages probabilities from both models
    )
    ensemble.fit(X_train, y_train)

    # ── 4. Tune threshold on validation set ────────────────────────────────────
    print("\n  [4/6] Tuning decision threshold on validation set ...")
    print("\n  XGBoost:")
    xgb_thresh = tune_threshold(xgb_model, X_val, y_val)
    print("  LightGBM:")
    lgb_thresh = tune_threshold(lgb_model, X_val, y_val)
    print("  Ensemble:")
    ens_thresh = tune_threshold(ensemble, X_val, y_val)

    # ── 5. Evaluate on validation set ─────────────────────────────────────────
    print(f"\n  [5/6] Validation set results ...")
    xgb_val = evaluate(xgb_model, X_val, y_val, xgb_thresh, "XGBoost (val)")
    lgb_val = evaluate(lgb_model, X_val, y_val, lgb_thresh, "LightGBM (val)")
    ens_val = evaluate(ensemble,  X_val, y_val, ens_thresh, "Ensemble (val)")

    # Pick best model by AUPRC on validation set
    scores = {
        "XGBoost":  (xgb_model, xgb_thresh, xgb_val),
        "LightGBM": (lgb_model, lgb_thresh, lgb_val),
        "Ensemble": (ensemble,  ens_thresh, ens_val),
    }
    best_name = max(scores, key=lambda k: scores[k][2]["auprc"])
    best_model, best_thresh, best_val = scores[best_name]
    print(f"\n  Best model on validation: {best_name}  (AUPRC={best_val['auprc']:.4f})")

    # ── 6. Final evaluation on test set (never touched before) ────────────────
    print(f"\n  [6/6] Final evaluation on test set (held-out) ...")
    test_metrics = evaluate(
        best_model, X_test, y_test, best_thresh,
        f"{best_name} — FINAL TEST"
    )

    passed = check_targets(test_metrics, best_name)

    # ── Save model ─────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    save_obj = {
        "model":           best_model,
        "model_name":      best_name,
        "threshold":       best_thresh,
        "metrics":         test_metrics,
        "feature_columns": list(X.columns),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(save_obj, f)

    print(f"\n  Model saved → {MODEL_PATH}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  TRAINING COMPLETE")
    print(SEP)
    print(f"\n  Model        : {best_name}")
    print(f"  Threshold    : {best_thresh:.2f}")
    print(f"  Precision    : {test_metrics['precision']:.4f}  (target >= {MIN_PRECISION})")
    print(f"  Recall       : {test_metrics['recall']:.4f}  (target >= {MIN_RECALL})")
    print(f"  F1           : {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC      : {test_metrics['auc_roc']:.4f}")
    print(f"  AUPRC        : {test_metrics['auprc']:.4f}  (target >= {MIN_AUPRC})")
    print(f"\n  FP (legit blocked) : {test_metrics['fp']}")
    print(f"  FN (fraud missed)  : {test_metrics['fn']}")
    print(f"\n  Targets met  : {'✅ YES' if passed else '❌ NO — review model'}")
    print(f"{SEP}\n")

    return test_metrics


if __name__ == "__main__":
    main()
