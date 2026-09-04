"""
fraud_features.py
-----------------
Feature engineering for Phase 2: Fraud Transaction Scorer.

Responsibilities:
  - Drop identifier, date, string, and leakage columns
  - Encode categorical columns (text → numbers)
  - Ensure correct data types for all features
  - Return a clean (X, y) pair ready for model training

Usage:
    from src.fraud_features import build_features
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

# ── Columns to drop before training ──────────────────────────────────────────

# Identifiers — unique IDs carry no signal for fraud detection
ID_COLUMNS = [
    "transaction_id",
    "customer_id",
    "merchant_id",
]

# Raw string location/device columns — already captured as engineered features:
#   device_id         → is_new_device
#   transaction_city  → is_different_city
#   billing_city      → address_mismatch
#   shipping_city     → address_mismatch
STRING_COLUMNS = [
    "device_id",
    "transaction_city",
    "billing_city",
    "shipping_city",
]

# Raw datetime — already captured as hour_of_day and day_of_week
DATE_COLUMNS = [
    "transaction_datetime",
]

# Leakage — fraud_pattern is derived FROM the label.
# In production this column won't exist, so we must not train on it.
LEAKAGE_COLUMNS = [
    "fraud_pattern",
]

# Target column — this is y, not a feature
TARGET_COLUMN = "is_fraud"

# ── Final feature list the model trains on ─────────────────────────────────────
# Defined explicitly so the order is always consistent.
# The API must send features in this exact order.

FEATURE_COLUMNS = [
    # Transaction details
    "order_value",
    "product_category",         # encoded
    "payment_method",           # encoded
    "hour_of_day",
    "day_of_week",
    "failed_attempts",
    "is_vpn",

    # Customer history
    "customer_account_age_days",
    "customer_total_past_orders",
    "customer_avg_order_value",
    "customer_past_fraud_flags",
    "customer_orders_1h",
    "customer_orders_24h",

    # Merchant
    "merchant_fraud_rate",

    # Engineered features
    "is_night_transaction",
    "is_weekend",
    "is_new_device",
    "is_different_city",
    "is_high_value",
    "is_first_order",
    "is_new_account",
    "address_mismatch",
    "multiple_failed_attempts",
    "high_velocity_1h",
    "order_value_normalized",
    "new_account_high_value_card",
    "payment_risk_score",
    "category_risk_score",
]


# ── Core function ──────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Transform raw transactions DataFrame into (X, y) for model training.

    Steps:
      1. Extract target column (is_fraud) as y
      2. Drop identifier, string, date, leakage, and target columns
      3. Encode categorical text columns to integers
      4. Convert boolean columns to int (0/1)
      5. Select and order final feature columns
      6. Validate — no missing values, all numeric

    Args:
        df: Raw DataFrame loaded from data/raw/transactions.csv

    Returns:
        X: DataFrame with shape (n_samples, n_features)
        y: Series with shape (n_samples,) — 0 = legitimate, 1 = fraud
    """
    df = df.copy()

    # ── Step 1: Extract target ─────────────────────────────────────────────────
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")
    y = df[TARGET_COLUMN].astype(int)

    # ── Step 2: Drop columns not used in training ──────────────────────────────
    cols_to_drop = ID_COLUMNS + STRING_COLUMNS + DATE_COLUMNS + LEAKAGE_COLUMNS + [TARGET_COLUMN]
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

    # ── Step 4: Convert booleans to int ───────────────────────────────────────
    bool_cols = [
        "is_vpn", "is_night_transaction", "is_weekend",
        "is_new_device", "is_different_city", "is_high_value",
        "is_first_order", "is_new_account", "address_mismatch",
        "multiple_failed_attempts", "high_velocity_1h",
        "new_account_high_value_card",
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
    print(f"    Legitimate (0) : {(y == 0).sum():,}  ({(y == 0).mean()*100:.1f}%)")
    print(f"    Fraud      (1) : {(y == 1).sum():,}  ({(y == 1).mean()*100:.1f}%)")
    print(f"\n  Feature dtypes:")
    for col, dtype in X.dtypes.items():
        print(f"    {col:<40} {dtype}")
    print(f"\n  Missing values : {'None ✓' if X.isnull().sum().sum() == 0 else X.isnull().sum()}")
    print("=" * 55)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "raw", "transactions.csv"
    )

    print("\n  Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} records with {len(df.columns)} columns.")

    print("\n  Running build_features()...")
    X, y = build_features(df)

    summarize_features(X, y)
    print("\n  Feature engineering complete. Ready for model training.\n")
