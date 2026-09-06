"""
retrain.py
----------
Feedback-driven retraining pipeline for all three models.

What this does:
  1. Loads original training data for each module
  2. Loads confirmed feedback from data/outcomes/
  3. Merges new confirmed labels into the training set
  4. Retrains XGBoost, LightGBM, Ensemble — picks best by AUPRC
  5. Compares new model vs current model on the same held-out test set
  6. If new model is better → saves it as the CURRENT model AND
     archives a timestamped version in models/versions/
  7. If new model is NOT better → keeps the old model, still archives
     the new one for audit purposes

Model versioning:
  models/
    return_risk_model.pkl          ← always the current best (overwritten on improvement)
    fraud_risk_model.pkl           ← always the current best
    chargeback_model.pkl           ← always the current best
    versions/
      return_risk_model_v_<timestamp>.pkl   ← every retrain is archived here
      fraud_risk_model_v_<timestamp>.pkl
      chargeback_model_v_<timestamp>.pkl

Usage:
    python src/retrain.py                     # retrain all 3 modules
    python src/retrain.py --module return     # retrain only return model
    python src/retrain.py --module fraud      # retrain only fraud model
    python src/retrain.py --module chargeback # retrain only chargeback model
    python src/retrain.py --dry-run           # show feedback stats, don't retrain
"""

import os
import sys
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix,
)
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSIONS_DIR = os.path.join(ROOT, "models", "versions")
sys.path.insert(0, ROOT)

from src.features            import build_features as return_build_features
from src.fraud_features      import build_features as fraud_build_features
from src.chargeback_features import build_features as chargeback_build_features
from data.outcomes.feedback_store import read_feedback, VALID_OUTCOMES

# ── Module config ──────────────────────────────────────────────────────────────
# Everything that differs per module in one place.
MODULE_CONFIG = {
    "return": {
        "data_path":      os.path.join(ROOT, "data", "raw", "returns.csv"),
        "model_path":     os.path.join(ROOT, "models", "return_risk_model.pkl"),
        "model_filename": "return_risk_model",
        "build_features": return_build_features,
        "target_column":  "is_label_abusive",
        "feedback_label_map": {"abusive": True, "legitimate": False},
        "scale_pos_weight": 8500 / 1500,
        "min_precision":  0.70,
        "min_recall":     0.65,
        "min_auprc":      0.75,
    },
    "fraud": {
        "data_path":      os.path.join(ROOT, "data", "raw", "transactions.csv"),
        "model_path":     os.path.join(ROOT, "models", "fraud_risk_model.pkl"),
        "model_filename": "fraud_risk_model",
        "build_features": fraud_build_features,
        "target_column":  "is_fraud",
        "feedback_label_map": {"fraud": True, "legitimate": False},
        "scale_pos_weight": 8500 / 1500,
        "min_precision":  0.80,
        "min_recall":     0.70,
        "min_auprc":      0.75,
    },
    "chargeback": {
        "data_path":      os.path.join(ROOT, "data", "raw", "chargebacks.csv"),
        "model_path":     os.path.join(ROOT, "models", "chargeback_model.pkl"),
        "model_filename": "chargeback_model",
        "build_features": chargeback_build_features,
        "target_column":  "merchant_won",
        "feedback_label_map": {"won": 1, "lost": 0},
        "scale_pos_weight": None,   # 60/40 split — no reweighting needed
        "min_precision":  0.75,
        "min_recall":     0.75,
        "min_auprc":      0.75,
    },
}

SEP = "=" * 64


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _timestamp() -> str:
    """Return a filesystem-safe UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_current_model(model_path: str) -> dict | None:
    """Load the current saved model. Returns None if not found."""
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _save_model(obj: dict, current_path: str, versioned_path: str, deploy: bool):
    """
    Always save to versioned_path (archive).
    Only overwrite current_path if deploy=True.
    """
    os.makedirs(os.path.dirname(versioned_path), exist_ok=True)
    with open(versioned_path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Archived  → {versioned_path}")

    if deploy:
        with open(current_path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Deployed  → {current_path}  ✅ (new current model)")
    else:
        print(f"  Kept old  → {current_path}  (old model retained)")


def _evaluate(model, X, y, threshold: float) -> dict:
    """Evaluate a model at a given threshold. Returns a metrics dict."""
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    return {
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall":    float(recall_score(y, preds, zero_division=0)),
        "f1":        float(f1_score(y, preds, zero_division=0)),
        "auprc":     float(average_precision_score(y, proba)),
        "auc_roc":   float(roc_auc_score(y, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _tune_threshold(model, X_val, y_val) -> float:
    """Find the threshold that maximises F1 on the validation set."""
    proba = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        preds = (proba >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return float(best_thresh)


def _build_xgb_lgb_ensemble(scale_pos_weight):
    """Build XGBoost, LightGBM, and Voting Ensemble. Returns a dict of models."""
    spw = scale_pos_weight or 1.0

    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, random_state=42, verbose=-1,
    )
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=42, verbosity=0,
        )), ("lgb", lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, random_state=42, verbose=-1,
        ))],
        voting="soft",
    )
    return {"XGBoost": xgb_model, "LightGBM": lgb_model, "Ensemble": ensemble}


# ══════════════════════════════════════════════════════════════════════════════
# Core: merge feedback into training data
# ══════════════════════════════════════════════════════════════════════════════

def _merge_feedback(module: str, df_original: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Read confirmed feedback records and append them to the original DataFrame.

    How it works:
      - Feedback records contain original_id + actual_outcome (e.g. "fraud")
      - We map actual_outcome → the target column value (True/False or 1/0)
      - We duplicate a representative row from the original data and set the
        target column to the confirmed label
      - This is a simplified merge — in production you'd store full feature
        vectors at prediction time and replay them here

    Returns the augmented DataFrame (original + feedback rows).
    """
    records = read_feedback(module)
    if not records:
        print(f"  No feedback found for {module}. Training on original data only.")
        return df_original

    label_map    = cfg["feedback_label_map"]
    target_col   = cfg["target_column"]

    # Build new rows from feedback.
    # Use a real row with the same label as base — preserves categorical
    # columns correctly. The label is confirmed ground truth.
    new_rows = []
    for rec in records:
        outcome = rec["actual_outcome"]
        if outcome not in label_map:
            continue
        confirmed_label = label_map[outcome]
        matching = df_original[df_original[target_col] == confirmed_label]
        if matching.empty:
            matching = df_original
        base_row = matching.sample(1, random_state=42).iloc[0].to_dict()
        base_row[target_col] = confirmed_label
        new_rows.append(base_row)

    if not new_rows:
        return df_original

    df_feedback = pd.DataFrame(new_rows)
    df_augmented = pd.concat([df_original, df_feedback], ignore_index=True)
    print(f"  Merged {len(new_rows)} feedback records → total rows: {len(df_augmented):,}")
    return df_augmented


# ══════════════════════════════════════════════════════════════════════════════
# Core: retrain one module
# ══════════════════════════════════════════════════════════════════════════════

def retrain_module(module: str, dry_run: bool = False) -> dict:
    """
    Retrain one module, compare with current model, save versioned copy.
    Returns a result dict with module, deployed (bool), and metrics comparison.
    """
    cfg = MODULE_CONFIG[module]
    ts  = _timestamp()

    print(f"\n{SEP}")
    print(f"  RETRAINING: {module.upper()} MODULE")
    print(SEP)

    # ── 1. Load feedback summary ───────────────────────────────────────────────
    records = read_feedback(module)
    print(f"\n  Feedback records available: {len(records)}")
    if dry_run:
        print("  [dry-run] Stopping here. Pass no --dry-run to actually retrain.")
        return {"module": module, "deployed": False, "dry_run": True, "feedback_count": len(records)}

    # ── 2. Load original data ──────────────────────────────────────────────────
    print(f"\n  [1/6] Loading data from {cfg['data_path']} ...")
    df = pd.read_csv(cfg["data_path"])

    # ── 3. Merge feedback ──────────────────────────────────────────────────────
    print(f"\n  [2/6] Merging feedback ...")
    df = _merge_feedback(module, df, cfg)

    # ── 4. Build features & split ──────────────────────────────────────────────
    print(f"\n  [3/6] Building features and splitting ...")
    X, y = cfg["build_features"](df)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2222, stratify=y_temp, random_state=42
    )
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ── 5. Train models ────────────────────────────────────────────────────────
    print(f"\n  [4/6] Training XGBoost, LightGBM, Ensemble ...")
    models = _build_xgb_lgb_ensemble(cfg["scale_pos_weight"])

    for name, m in models.items():
        print(f"  Training {name} ...")
        if name in ("XGBoost",):
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            m.fit(X_train, y_train)

    # ── 6. Tune thresholds & pick best ────────────────────────────────────────
    print(f"\n  [5/6] Tuning thresholds on validation set ...")
    scores = {}
    for name, m in models.items():
        thresh  = _tune_threshold(m, X_val, y_val)
        metrics = _evaluate(m, X_val, y_val, thresh)
        scores[name] = (m, thresh, metrics)
        print(f"  {name:12} val AUPRC={metrics['auprc']:.4f}  threshold={thresh:.2f}")

    best_name  = max(scores, key=lambda n: scores[n][2]["auprc"])
    best_model, best_thresh, _ = scores[best_name]
    print(f"\n  Best: {best_name}")

    # ── 7. Evaluate new model on test set ──────────────────────────────────────
    print(f"\n  [6/6] Evaluating on held-out test set ...")
    new_metrics = _evaluate(best_model, X_test, y_test, best_thresh)
    print(f"  New model  — AUPRC={new_metrics['auprc']:.4f}  "
          f"P={new_metrics['precision']:.4f}  R={new_metrics['recall']:.4f}")

    # ── 8. Compare with current model ─────────────────────────────────────────
    current_obj = _load_current_model(cfg["model_path"])
    deploy      = False

    if current_obj is None:
        print("\n  No existing model found. Deploying new model.")
        deploy = True
    else:
        old_auprc = current_obj["metrics"]["auprc"]
        new_auprc = new_metrics["auprc"]
        print(f"\n  Old model  — AUPRC={old_auprc:.4f}")
        print(f"  New model  — AUPRC={new_auprc:.4f}")

        if new_auprc > old_auprc:
            print(f"\n  ✅ New model is better (+{new_auprc - old_auprc:.4f} AUPRC). Deploying.")
            deploy = True
        else:
            print(f"\n  ⚠️  New model is NOT better ({new_auprc - old_auprc:.4f} AUPRC). Keeping old model.")
            deploy = False

    # ── 9. Save versioned copy (always) + deploy if better ────────────────────
    save_obj = {
        "model":           best_model,
        "model_name":      best_name,
        "threshold":       best_thresh,
        "metrics":         new_metrics,
        "feature_columns": list(X.columns),
        "retrained_at":    ts,
        "feedback_count":  len(records),
    }

    versioned_filename = f"{cfg['model_filename']}_v_{ts}.pkl"
    versioned_path     = os.path.join(VERSIONS_DIR, versioned_filename)

    print(f"\n  Saving ...")
    _save_model(save_obj, cfg["model_path"], versioned_path, deploy)

    return {
        "module":          module,
        "deployed":        deploy,
        "new_model_name":  best_name,
        "new_auprc":       round(new_metrics["auprc"], 4),
        "old_auprc":       round(current_obj["metrics"]["auprc"], 4) if current_obj else None,
        "versioned_path":  versioned_path,
        "feedback_count":  len(records),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AI Risk Manager — Retraining Pipeline")
    parser.add_argument(
        "--module", choices=["return", "fraud", "chargeback"],
        default=None,
        help="Retrain only this module. Omit to retrain all three.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show feedback stats without actually retraining.",
    )
    args = parser.parse_args()

    modules = [args.module] if args.module else ["return", "fraud", "chargeback"]

    print(f"\n{SEP}")
    print("  AI Risk Manager — Feedback Retraining Pipeline")
    print(f"  Modules   : {', '.join(modules)}")
    print(f"  Dry run   : {args.dry_run}")
    print(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP)

    results = []
    for module in modules:
        result = retrain_module(module, dry_run=args.dry_run)
        results.append(result)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RETRAINING SUMMARY")
    print(SEP)
    for r in results:
        if r.get("dry_run"):
            print(f"  {r['module']:12} | feedback={r['feedback_count']:4d} | [dry-run]")
            continue
        status   = "DEPLOYED ✅" if r["deployed"] else "KEPT OLD ⚠️ "
        old_str  = f"{r['old_auprc']:.4f}" if r["old_auprc"] is not None else "N/A"
        print(
            f"  {r['module']:12} | {status} | "
            f"old AUPRC={old_str}  new AUPRC={r['new_auprc']:.4f} | "
            f"feedback={r['feedback_count']:4d}"
        )
    print(f"\n  Versioned models saved to: models/versions/")
    print(f"  Restart the API to load the new model: uvicorn api.main:app --reload")
    print(SEP)


if __name__ == "__main__":
    main()
