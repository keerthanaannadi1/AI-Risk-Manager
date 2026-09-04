"""
chargeback_train.py
-------------------
Model training for Phase 3: Chargeback Evidence Responder.

What this does:
  1. Loads and prepares data using chargeback_features.build_features()
  2. Splits into 70% train / 20% validation / 10% test (stratified)
  3. Trains three models: XGBoost, LightGBM, Voting Ensemble
  4. Tunes decision threshold on the validation set
  5. Picks the best model by AUPRC on validation set
  6. Evaluates the best model on the test set (once, final numbers)
  7. Saves the best model to models/chargeback_model.pkl

Note on class balance:
  60/40 split (wins/losses) — no scale_pos_weight needed.
  The model sees enough examples of both classes naturally.

Minimum targets:
  Precision >= 0.75  (don't send merchants to fight unwinnable disputes)
  Recall    >= 0.75  (don't miss winnable disputes — every win = money recovered)
  AUPRC     >= 0.75

Usage:
    python src/chargeback_train.py
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
    confusion_matrix,
)
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(ROOT, "data", "raw", "chargebacks.csv")
MODEL_DIR  = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "chargeback_model.pkl")

sys.path.insert(0, ROOT)
from src.chargeback_features import build_features

# ── Minimum targets ────────────────────────────────────────────────────────────
MIN_PRECISION = 0.75
MIN_RECALL    = 0.75
MIN_AUPRC     = 0.75

# ── Hyperparameters ────────────────────────────────────────────────────────────
# No scale_pos_weight needed — 60/40 split is balanced enough
XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
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
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":          -1,
}

THRESHOLD = 0.50   # start at 0.50 since classes are balanced


# ── Helpers ────────────────────────────────────────────────────────────────────

def evaluate(model, X, y, threshold=THRESHOLD, label="") -> dict:
    """Evaluate a model at a given threshold. Returns metrics dict."""
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
    print(f"    TN={tn:>4}  FP={fp:>4}  (FP = lost disputes predicted as wins)")
    print(f"    FN={fn:>4}  TP={tp:>4}  (FN = winnable disputes missed)")

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
    while keeping both precision AND recall >= MIN_PRECISION.
    Equal weight — we care about both not missing wins and not wasting effort.
    """
    proba = model.predict_proba(X_val)[:, 1]
    best_f1, best_thresh = 0.0, 0.5

    for thresh in np.arange(0.20, 0.85, 0.01):
        preds = (proba >= thresh).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f = f1_score(y_val, preds, zero_division=0)
        if p >= MIN_PRECISION and r >= MIN_RECALL and f > best_f1:
            best_f1, best_thresh = f, thresh

    print(f"  Best threshold (F1={best_f1:.4f}): {best_thresh:.2f}")
    return best_thresh


def check_targets(metrics: dict, model_name: str) -> bool:
    """Check whether a model meets the minimum targets."""
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
    print("  Phase 3: Chargeback Evidence Responder")
    print(SEP)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("\n  [1/6] Loading data ...")
    df = pd.read_csv(DATA_PATH)
    X, y = build_features(df)
    print(f"  X shape : {X.shape}")
    print(f"  y dist  : {(y==0).sum():,} losses / {(y==1).sum():,} wins")

    # ── 2. Split 70 / 20 / 10 ─────────────────────────────────────────────────
    print("\n  [2/6] Splitting data (70 train / 20 val / 10 test, stratified) ...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2222, stratify=y_temp, random_state=42
    )
    print(f"  Train : {len(X_train):,}  ({y_train.mean()*100:.1f}% wins)")
    print(f"  Val   : {len(X_val):,}  ({y_val.mean()*100:.1f}% wins)")
    print(f"  Test  : {len(X_test):,}  ({y_test.mean()*100:.1f}% wins)")

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
        voting="soft",
    )
    ensemble.fit(X_train, y_train)

    # ── 4. Tune threshold ──────────────────────────────────────────────────────
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

    scores = {
        "XGBoost":  (xgb_model, xgb_thresh, xgb_val),
        "LightGBM": (lgb_model, lgb_thresh, lgb_val),
        "Ensemble": (ensemble,  ens_thresh, ens_val),
    }
    best_name = max(scores, key=lambda k: scores[k][2]["auprc"])
    best_model, best_thresh, best_val = scores[best_name]
    print(f"\n  Best model on validation: {best_name}  (AUPRC={best_val['auprc']:.4f})")

    # ── 6. Final evaluation on test set ───────────────────────────────────────
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
    print(f"\n  FP (lost disputes predicted as wins) : {test_metrics['fp']}")
    print(f"  FN (winnable disputes missed)        : {test_metrics['fn']}")
    print(f"\n  Targets met  : {'✅ YES' if passed else '❌ NO — review model'}")
    print(f"{SEP}\n")

    return test_metrics


if __name__ == "__main__":
    main()
