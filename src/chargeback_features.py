"""
chargeback_features.py
----------------------
Feature engineering for Phase 3: Chargeback Evidence Responder.

Responsibilities:
  - Drop identifier, date, and leakage columns
  - Encode categorical columns (text → numbers)
  - Ensure correct data types for all features
  - Return a clean (X, y) pair ready for model training

Usage:
    from src.chargeback_features import build_features
    X, y = build_features(df)
"""

import pandas as pd
import numpy as np

# ── Categorical encodings ─────────────────────────────────────────────────────
# These must stay fixed forever.
# If you add a new category, add it here AND retrain the model.

CATEGORY_ENCODING = {
    "electronics": 0,
    "apparel":     1,
    "books":       2,
    "home":        3,
    "luxury":      4,
    "gift_cards":  5,
}

PAYMENT_ENCODING = {
    "card":       0,
    "upi":        1,
    "netbanking": 2,
    "wallet":     3,
    "cod":        4,
}

DISPUTE_REASON_ENCODING = {
    "not_received":         0,
    "not_authorized":       1,
    "not_as_described":     2,
    "duplicate_charge":     3,
    "credit_not_processed": 4,
}

# ── Columns to drop before training ───────────────────────────────────────────

# Identifiers — no signal
ID_COLUMNS = [
    "dispute_id",
    "customer_id",
    "merchant_id",
]

# Raw date strings — already captured as days_to_dispute
DATE_COLUMNS = [
    "transaction_date",
    "dispute_date",
]

# Target column — this is y, not a feature
TARGET_COLUMN = "merchant_won"

# ── Final feature list the model trains on ─────────────────────────────────────
# Defined explicitly so the order is always consistent.
# The API must send features in this exact order.

FEATURE_COLUMNS = [
    # Transaction details
    "order_value",
    "product_category",         # encoded
    "payment_method",           # encoded
    "days_to_dispute",
    "delivery_days",
    "dispute_reason",           # encoded

    # Evidence available
    "delivery_confirmed",
    "customer_signed_delivery",
    "otp_used",
    "ip_logs_available",
    "order_confirmation_sent",
    "refund_issued",

    # Customer behaviour
    "customer_contacted_support",
    "customer_account_age_days",
    "customer_total_orders",
    "customer_past_disputes",
    "customer_avg_order_value",

    # Merchant
    "merchant_chargeback_rate",

    # Engineered features
    "is_quick_dispute",
    "is_late_dispute",
    "is_high_value",
    "is_first_order",
    "is_new_account",
    "is_serial_disputer",
    "is_cod",
    "evidence_score",
    "dispute_difficulty",
    "order_value_normalized",
]


# ── Core function ──────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Transform raw chargebacks DataFrame into (X, y) for model training.

    Steps:
      1. Extract target column (merchant_won) as y
      2. Drop identifier and date columns
      3. Encode categorical text columns to integers
      4. Convert boolean columns to int (0/1)
      5. Select and order final feature columns
      6. Validate — no missing values, all numeric

    Args:
        df: Raw DataFrame loaded from data/raw/chargebacks.csv

    Returns:
        X: DataFrame with shape (n_samples, n_features)
        y: Series with shape (n_samples,) — 0 = merchant lost, 1 = merchant won
    """
    df = df.copy()

    # ── Step 1: Extract target ─────────────────────────────────────────────────
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")
    y = df[TARGET_COLUMN].astype(int)

    # ── Step 2: Drop columns not used in training ──────────────────────────────
    cols_to_drop = ID_COLUMNS + DATE_COLUMNS + [TARGET_COLUMN]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # ── Step 3: Encode categoricals ───────────────────────────────────────────
    df["product_category"] = (
        df["product_category"]
        .str.lower()
        .map(CATEGORY_ENCODING)
    )

    df["payment_method"] = (
        df["payment_method"]
        .str.lower()
        .map(PAYMENT_ENCODING)
    )

    df["dispute_reason"] = (
        df["dispute_reason"]
        .str.lower()
        .map(DISPUTE_REASON_ENCODING)
    )

    # ── Step 4: Convert booleans to int ───────────────────────────────────────
    bool_cols = [
        "delivery_confirmed", "customer_signed_delivery",
        "otp_used", "ip_logs_available", "order_confirmation_sent",
        "refund_issued", "customer_contacted_support",
        "is_quick_dispute", "is_late_dispute", "is_high_value",
        "is_first_order", "is_new_account", "is_serial_disputer", "is_cod",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # ── Step 5: Select and order final feature columns ─────────────────────────
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Expected feature columns missing from data: {missing_cols}"
        )

    X = df[FEATURE_COLUMNS].copy()

    # ── Step 6: Validate ───────────────────────────────────────────────────────
    missing_values = X.isnull().sum()
    cols_with_nulls = missing_values[missing_values > 0]
    if len(cols_with_nulls) > 0:
        raise ValueError(
            f"Missing values found after feature engineering:\n{cols_with_nulls}"
        )

    _validate_dtypes(X)

    return X, y


def _validate_dtypes(X: pd.DataFrame) -> None:
    """Ensure all feature columns are numeric. Raise clearly if not."""
    non_numeric = [
        col for col in X.columns
        if not pd.api.types.is_numeric_dtype(X[col])
    ]
    if non_numeric:
        raise ValueError(
            f"Non-numeric columns found after feature engineering: {non_numeric}\n"
            "All features must be numeric before passing to the model."
        )


# ── Utility functions ──────────────────────────────────────────────────────────

def get_feature_names() -> list[str]:
    """Return the ordered list of feature column names."""
    return FEATURE_COLUMNS.copy()


def summarize_features(X: pd.DataFrame, y: pd.Series) -> None:
    """Print a quick summary of the feature matrix."""
    print("=" * 55)
    print("  FEATURE MATRIX SUMMARY")
    print("=" * 55)
    print(f"\n  Samples   : {len(X):,}")
    print(f"  Features  : {X.shape[1]}")
    print(f"\n  Target distribution:")
    print(f"    Merchant lost (0) : {(y == 0).sum():,}  ({(y == 0).mean()*100:.1f}%)")
    print(f"    Merchant won  (1) : {(y == 1).sum():,}  ({(y == 1).mean()*100:.1f}%)")
    print(f"\n  Feature dtypes:")
    for col, dtype in X.dtypes.items():
        print(f"    {col:<40} {dtype}")
    print(f"\n  Missing values : {'None ✓' if X.isnull().sum().sum() == 0 else X.isnull().sum()}")
    print("=" * 55)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "raw", "chargebacks.csv"
    )

    print("\n  Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} records with {len(df.columns)} columns.")

    print("\n  Running build_features()...")
    X, y = build_features(df)

    summarize_features(X, y)
    print("\n  Feature engineering complete. Ready for model training.\n")
