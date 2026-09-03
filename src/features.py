"""
features.py
-----------
Feature engineering for Phase 1: Return Risk Scorer.

Responsibilities:
  - Drop identifier and leakage columns
  - Encode categorical columns (text → numbers)
  - Ensure correct data types for all features
  - Return a clean (X, y) pair ready for model training

Usage:
    from src.features import build_features
    X, y = build_features(df)
"""

import pandas as pd
import numpy as np

# ── Categorical encodings (match data-dictionary.md) ─────────────────────────

CATEGORY_ENCODING = {
    "books":       0,
    "home":        1,
    "apparel":     2,
    "electronics": 3,
    "luxury":      4,
}

REASON_ENCODING = {
    "defective":     0,
    "wrong_item":    1,
    "quality_issue": 2,
    "changed_mind":  3,
    "not_needed":    4,
}

PAYMENT_ENCODING = {
    "cod":        0,
    "wallet":     1,
    "netbanking": 2,
    "upi":        3,
    "card":       4,
}

# ── Columns that must be dropped before training ──────────────────────────────

# Identifiers — no signal, just unique IDs
ID_COLUMNS = [
    "return_id",
    "order_id",
    "customer_id",
    "merchant_id",
]

# Raw date strings — already captured as days_to_return
DATE_COLUMNS = [
    "order_date",
    "return_request_date",
]

# Label leakage — abuse_pattern is derived FROM the label.
# In production this column won't exist, so we must not train on it.
LEAKAGE_COLUMNS = [
    "abuse_pattern",
]

# Target column — this is y, not a feature
TARGET_COLUMN = "is_label_abusive"

# ── Final feature list the model trains on ────────────────────────────────────
# Defined explicitly so the order is always consistent (important for SHAP).

FEATURE_COLUMNS = [
    # Customer history
    "customer_total_orders",
    "customer_total_returns",
    "customer_account_age_days",
    "customer_returns_30d",
    "customer_orders_30d",
    "customer_avg_order_value",

    # Order details
    "order_value",
    "payment_method",           # encoded

    # Return behaviour
    "days_to_return",
    "return_reason",            # encoded
    "support_contacted",
    "images_submitted",

    # Product
    "product_category",         # encoded

    # Merchant
    "merchant_return_rate",

    # Pre-computed features (from generate_data.py)
    "return_rate_lifetime",
    "return_rate_30d",
    "is_near_deadline",
    "is_same_day_return",
    "is_first_order",
    "is_new_account",
    "no_images_high_value",
    "no_support_contact",
    "category_risk_score",
    "return_reason_risk",
    "order_value_normalized",
]


# ── Core function ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Transform raw returns DataFrame into (X, y) for model training.

    Steps:
      1. Drop identifier, date, leakage, and target columns
      2. Encode categorical text columns to integers
      3. Convert boolean columns to int (0/1)
      4. Validate no missing values remain
      5. Return X (features) and y (target)

    Args:
        df: Raw DataFrame loaded from data/raw/returns.csv

    Returns:
        X: DataFrame with shape (n_samples, n_features)
        y: Series with shape (n_samples,) — 0 = legitimate, 1 = abusive
    """
    df = df.copy()

    # ── Step 1: Extract target ────────────────────────────────────────────────
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")
    y = df[TARGET_COLUMN].astype(int)

    # ── Step 2: Drop columns not used in training ─────────────────────────────
    cols_to_drop = ID_COLUMNS + DATE_COLUMNS + LEAKAGE_COLUMNS + [TARGET_COLUMN]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # ── Step 3: Encode categoricals ───────────────────────────────────────────
    df["product_category"] = (
        df["product_category"]
        .str.lower()
        .map(CATEGORY_ENCODING)
    )

    df["return_reason"] = (
        df["return_reason"]
        .str.lower()
        .map(REASON_ENCODING)
    )

    df["payment_method"] = (
        df["payment_method"]
        .str.lower()
        .map(PAYMENT_ENCODING)
    )

    # ── Step 4: Convert booleans to int ──────────────────────────────────────
    bool_cols = ["support_contacted", "images_submitted"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # ── Step 5: Select and order final feature columns ────────────────────────
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Expected feature columns missing from data: {missing_cols}")

    X = df[FEATURE_COLUMNS].copy()

    # ── Step 6: Validate ──────────────────────────────────────────────────────
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
            f"All features must be numeric before passing to the model."
        )


# ── Utility ───────────────────────────────────────────────────────────────────

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
    print(f"    Abusive    (1) : {(y == 1).sum():,}  ({(y == 1).mean()*100:.1f}%)")
    print(f"\n  Feature dtypes:")
    for col, dtype in X.dtypes.items():
        print(f"    {col:<35} {dtype}")
    print(f"\n  Missing values : {'None ✓' if X.isnull().sum().sum() == 0 else X.isnull().sum()}")
    print("=" * 55)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "returns.csv")

    print("\n  Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} records with {len(df.columns)} columns.")

    print("\n  Running build_features()...")
    X, y = build_features(df)

    summarize_features(X, y)
    print("\n  Feature engineering complete. Ready for model training.\n")
