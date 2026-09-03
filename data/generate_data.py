"""
generate_data.py
----------------
Generates ~10,000 synthetic return records for Phase 1: Return Risk Scorer.

Output: data/raw/returns.csv

Class distribution:
  ~85% legitimate returns  (is_label_abusive = False)
  ~15% abusive returns     (is_label_abusive = True)

Abusive patterns simulated:
  - Wardrobing        : buys apparel/luxury, returns quickly after use
  - Swap fraud        : returns electronics/luxury with no images
  - Serial returner   : lifetime return rate > 70%
  - Refund fishing    : claims defective/not-received, no support contact, no images

All fields in this file map to the data-dictionary.md spec.
Engineered features (return_rate_lifetime, days_to_return, etc.) are computed
here so that EDA and feature engineering can use them directly.

References:
  - SMOTE: Chawla et al., JAIR 2002  (class imbalance handling)
  - XGBoost: Chen & Guestrin, KDD 2016 (primary model)
"""

import os
import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_IN")
fake.seed_instance(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
NUM_RECORDS   = 10_000
ABUSE_RATE    = 0.15          # 15% abusive returns
NUM_CUSTOMERS = 2_000         # customer pool
NUM_MERCHANTS = 200
POLICY_DAYS   = 30            # return window
OUTPUT_PATH   = os.path.join(os.path.dirname(__file__), "raw", "returns.csv")

# ── Lookup tables (from data-dictionary.md) ───────────────────────────────────
PRODUCT_CATEGORIES = ["electronics", "apparel", "books", "home", "luxury"]
CATEGORY_RISK      = {
    "electronics": 0.8,
    "apparel":     0.6,
    "books":       0.2,
    "home":        0.4,
    "luxury":      0.9,
}

RETURN_REASONS = [
    "defective", "wrong_item", "not_needed", "quality_issue", "changed_mind"
]
REASON_RISK = {
    "defective":     0.3,
    "wrong_item":    0.4,
    "not_needed":    0.8,
    "quality_issue": 0.5,
    "changed_mind":  0.7,
}

PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet", "cod"]

# ── Customer pool ─────────────────────────────────────────────────────────────

def build_customer_pool(n: int) -> dict:
    """
    Pre-generate customer profiles so every record for a customer is
    internally consistent (same account age, same avg order value, etc.).

    ~12% of customers are flagged as abusers — these drive the serial-returner
    pattern. Wardrobing, swap fraud and refund fishing can come from any customer.
    """
    customers = {}
    for i in range(n):
        cid       = f"C_{10000 + i}"
        is_abuser = random.random() < 0.12

        if is_abuser:
            total_orders     = random.randint(5, 40)
            return_rate      = random.uniform(0.70, 0.95)
            total_returns    = int(total_orders * return_rate)
            account_age      = random.randint(10, 400)
            orders_30d       = random.randint(1, 6)
            returns_30d      = min(orders_30d, random.randint(1, orders_30d))
            avg_order_value  = round(np.random.lognormal(mean=8.5, sigma=0.6), 2)
        else:
            total_orders     = random.randint(1, 60)
            return_rate      = random.uniform(0.0, 0.20)
            total_returns    = int(total_orders * return_rate)
            account_age      = random.randint(30, 1500)
            orders_30d       = random.randint(0, 5)
            returns_30d      = random.randint(0, min(2, orders_30d)) if orders_30d > 0 else 0
            avg_order_value  = round(np.random.lognormal(mean=8.2, sigma=0.7), 2)

        # clamp avg_order_value to a realistic INR range
        avg_order_value = float(np.clip(avg_order_value, 100, 40_000))

        customers[cid] = {
            "is_abuser":          is_abuser,
            "total_orders":       max(total_orders, 1),
            "total_returns":      total_returns,
            "account_age_days":   account_age,
            "orders_30d":         orders_30d,
            "returns_30d":        returns_30d,
            "avg_order_value":    avg_order_value,
        }
    return customers


# ── Merchant pool ─────────────────────────────────────────────────────────────

def build_merchant_pool(n: int) -> dict:
    """
    Pre-generate merchant profiles.
    Some merchants have higher baseline return rates (bad product quality,
    wrong catalogue, etc.) — this adds realistic noise.
    """
    merchants = {}
    for i in range(n):
        mid = f"M_{5000 + i}"
        # merchant's baseline return rate — affects how abusive a return looks
        # high-volume merchants often have higher return rates legitimately
        merchant_return_rate = round(random.uniform(0.03, 0.25), 3)
        primary_category     = random.choice(PRODUCT_CATEGORIES)
        merchants[mid] = {
            "merchant_return_rate": merchant_return_rate,
            "primary_category":     primary_category,
        }
    return merchants


# ── Record builders ───────────────────────────────────────────────────────────

def _base_record(record_id, cid, mid, customer, merchant,
                 category, order_value, order_date, days_to_return,
                 return_reason, support_contacted, images_submitted,
                 is_abusive, abuse_pattern):
    """
    Assemble the full record dict from components.
    Stores both raw fields AND pre-computed feature columns so EDA is easy.
    """
    return_date = order_date + timedelta(days=days_to_return)

    total_orders  = customer["total_orders"]
    total_returns = customer["total_returns"]
    account_age   = customer["account_age_days"]
    orders_30d    = customer["orders_30d"]
    returns_30d   = customer["returns_30d"]
    avg_ov        = customer["avg_order_value"]

    # ── Pre-computed features (match data-dictionary.md engineered features) ──
    return_rate_lifetime = round(total_returns / total_orders, 4) if total_orders > 0 else 0.0
    return_rate_30d      = round(returns_30d / orders_30d, 4) if orders_30d > 0 else 0.0
    is_near_deadline     = int(days_to_return >= (POLICY_DAYS - 2))
    is_same_day_return   = int(days_to_return <= 1)
    is_first_order       = int(total_orders == 1)
    is_new_account       = int(account_age <= 30)
    no_images_high_value = int((not images_submitted) and (order_value > 5000))
    no_support_contact   = int(not support_contacted)
    category_risk_score  = CATEGORY_RISK[category]
    return_reason_risk   = REASON_RISK[return_reason]
    order_value_norm     = round(order_value / avg_ov, 4) if avg_ov > 0 else 1.0

    return {
        # ── Raw identifier fields ──────────────────────────────────────────────
        "return_id":                   f"R_{record_id:05d}",
        "order_id":                    f"O_{random.randint(10000, 99999)}",
        "customer_id":                 cid,
        "merchant_id":                 mid,

        # ── Raw order fields ───────────────────────────────────────────────────
        "product_category":            category,
        "order_value":                 round(order_value, 2),
        "order_date":                  order_date.strftime("%Y-%m-%d"),
        "return_request_date":         return_date.strftime("%Y-%m-%d"),
        "return_reason":               return_reason,
        "payment_method":              random.choice(PAYMENT_METHODS),

        # ── Raw customer behaviour fields ──────────────────────────────────────
        "support_contacted":           support_contacted,
        "images_submitted":            images_submitted,
        "customer_total_orders":       total_orders,
        "customer_total_returns":      total_returns,
        "customer_account_age_days":   account_age,
        "customer_returns_30d":        returns_30d,
        "customer_orders_30d":         orders_30d,
        "customer_avg_order_value":    round(avg_ov, 2),

        # ── Raw merchant fields ────────────────────────────────────────────────
        "merchant_return_rate":        merchant["merchant_return_rate"],

        # ── Pre-computed feature columns ───────────────────────────────────────
        "days_to_return":              days_to_return,
        "return_rate_lifetime":        return_rate_lifetime,
        "return_rate_30d":             return_rate_30d,
        "is_near_deadline":            is_near_deadline,
        "is_same_day_return":          is_same_day_return,
        "is_first_order":              is_first_order,
        "is_new_account":              is_new_account,
        "no_images_high_value":        no_images_high_value,
        "no_support_contact":          no_support_contact,
        "category_risk_score":         category_risk_score,
        "return_reason_risk":          return_reason_risk,
        "order_value_normalized":      order_value_norm,

        # ── Labels ─────────────────────────────────────────────────────────────
        "abuse_pattern":               abuse_pattern,   # for debugging/EDA
        "is_label_abusive":            is_abusive,      # target variable
    }


def generate_legitimate_record(record_id, cid, mid, customer, merchant):
    """Simulate a normal, non-abusive return."""
    category    = random.choices(
        PRODUCT_CATEGORIES, weights=[0.25, 0.35, 0.15, 0.20, 0.05]
    )[0]
    order_value = float(np.clip(np.random.lognormal(mean=8.5, sigma=0.8), 100, 50_000))
    order_date  = fake.date_between(start_date="-180d", end_date="-10d")

    # Legitimate: returns happen 3–25 days in, not right at the deadline
    days_to_ret = random.randint(3, min(25, POLICY_DAYS - 3))

    return_reason     = random.choices(
        RETURN_REASONS, weights=[0.35, 0.25, 0.15, 0.15, 0.10]
    )[0]
    support_contacted = random.random() < 0.65
    images_submitted  = random.random() < 0.60

    return _base_record(
        record_id, cid, mid, customer, merchant,
        category, order_value, order_date, days_to_ret,
        return_reason, support_contacted, images_submitted,
        is_abusive=False, abuse_pattern="none",
    )


def generate_abusive_record(record_id, cid, mid, customer, merchant):
    """
    Simulate one of four abuse patterns.
    Each pattern has distinct signals that the model should learn to pick up.
    """
    pattern = random.choices(
        ["wardrobing", "swap_fraud", "serial_returner", "refund_fishing"],
        weights=[0.25, 0.25, 0.30, 0.20],
    )[0]

    # ── Wardrobing ────────────────────────────────────────────────────────────
    if pattern == "wardrobing":
        category          = random.choice(["apparel", "luxury"])
        order_value       = random.uniform(1500, 15_000)
        order_date        = fake.date_between(start_date="-60d", end_date="-3d")
        days_to_ret       = random.randint(1, 4)           # used and returned quickly
        return_reason     = random.choice(["not_needed", "changed_mind"])
        support_contacted = False
        images_submitted  = random.random() < 0.15         # rarely submits images

    # ── Swap Fraud ────────────────────────────────────────────────────────────
    elif pattern == "swap_fraud":
        category          = random.choice(["electronics", "luxury"])
        order_value       = random.uniform(8_000, 50_000)
        order_date        = fake.date_between(start_date="-30d", end_date="-5d")
        days_to_ret       = random.randint(5, 15)
        return_reason     = random.choice(["defective", "quality_issue"])
        support_contacted = random.random() < 0.20
        images_submitted  = False                          # never — hides the swap

    # ── Serial Returner ───────────────────────────────────────────────────────
    elif pattern == "serial_returner":
        category          = random.choice(PRODUCT_CATEGORIES)
        order_value       = float(np.clip(np.random.lognormal(mean=8.5, sigma=0.8), 200, 30_000))
        order_date        = fake.date_between(start_date="-90d", end_date="-5d")
        days_to_ret       = random.randint(2, POLICY_DAYS - 1)
        return_reason     = random.choice(RETURN_REASONS)
        support_contacted = random.random() < 0.25
        images_submitted  = random.random() < 0.20
        # override customer history to match serial pattern
        customer = customer.copy()
        customer["total_orders"]  = random.randint(8, 40)
        customer["total_returns"] = int(customer["total_orders"] * random.uniform(0.70, 0.95))
        customer["returns_30d"]   = random.randint(2, 5)
        customer["orders_30d"]    = random.randint(2, 6)

    # ── Refund Fishing ────────────────────────────────────────────────────────
    else:  # refund_fishing
        category          = random.choice(["electronics", "home", "apparel"])
        order_value       = random.uniform(500, 10_000)
        order_date        = fake.date_between(start_date="-30d", end_date="-2d")
        # claims immediately OR right at deadline — both suspicious
        days_to_ret = random.choice([
            random.randint(0, 2),
            random.randint(POLICY_DAYS - 2, POLICY_DAYS),
        ])
        days_to_ret       = min(days_to_ret, POLICY_DAYS)
        return_reason     = random.choice(["defective", "wrong_item"])
        support_contacted = False                          # never contacts support
        images_submitted  = False                         # no evidence

    return _base_record(
        record_id, cid, mid, customer, merchant,
        category, order_value, order_date, days_to_ret,
        return_reason, support_contacted, images_submitted,
        is_abusive=True, abuse_pattern=pattern,
    )


# ── Main generation ───────────────────────────────────────────────────────────

def generate(n: int = NUM_RECORDS) -> pd.DataFrame:
    print(f"  Building customer pool ({NUM_CUSTOMERS} customers) ...")
    customers = build_customer_pool(NUM_CUSTOMERS)

    print(f"  Building merchant pool ({NUM_MERCHANTS} merchants) ...")
    merchants = build_merchant_pool(NUM_MERCHANTS)

    customer_ids = list(customers.keys())
    merchant_ids = list(merchants.keys())

    abuser_ids = [c for c, v in customers.items() if v["is_abuser"]]
    normal_ids = [c for c, v in customers.items() if not v["is_abuser"]]
    if not abuser_ids:
        abuser_ids = customer_ids

    num_abusive = int(n * ABUSE_RATE)
    num_legit   = n - num_abusive

    records = []

    print(f"  Generating {num_legit:,} legitimate records ...")
    for i in range(num_legit):
        cid = random.choice(normal_ids or customer_ids)
        mid = random.choice(merchant_ids)
        records.append(
            generate_legitimate_record(i + 1, cid, mid, customers[cid], merchants[mid])
        )

    print(f"  Generating {num_abusive:,} abusive records ...")
    for i in range(num_abusive):
        # 75% of abusive records come from known abuser customers
        cid = random.choice(abuser_ids) if random.random() < 0.75 else random.choice(customer_ids)
        mid = random.choice(merchant_ids)
        records.append(
            generate_abusive_record(num_legit + i + 1, cid, mid, customers[cid], merchants[mid])
        )

    df = pd.DataFrame(records)
    # shuffle so abusive records aren't all at the end
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


# ── Validation report ─────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    """Print a concise data quality report after generation."""
    SEP = "=" * 58
    print(f"\n{SEP}")
    print("  DATA VALIDATION REPORT")
    print(SEP)

    total       = len(df)
    abuse_n     = df["is_label_abusive"].sum()
    abuse_pct   = abuse_n / total * 100

    print(f"\n  Total records       : {total:,}")
    print(f"  Total columns       : {len(df.columns)}")
    print(f"\n  Legitimate returns  : {total - abuse_n:,}  ({100 - abuse_pct:.1f}%)")
    print(f"  Abusive returns     : {abuse_n:,}  ({abuse_pct:.1f}%)")

    # Abuse pattern breakdown
    print(f"\n  Abuse pattern breakdown:")
    for pat, cnt in df["abuse_pattern"].value_counts().items():
        print(f"    {pat:<20} {cnt:>5,}  ({cnt/total*100:.1f}%)")

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"\n  Missing values      : {'None ✓' if len(missing) == 0 else ''}")
    for col, count in missing.items():
        print(f"    {col}: {count}")

    # Category distribution
    print(f"\n  Product category distribution:")
    for cat, cnt in df["product_category"].value_counts().items():
        print(f"    {cat:<15} {cnt:>5,}  ({cnt/total*100:.1f}%)")

    # Return reason distribution
    print(f"\n  Return reason distribution:")
    for reason, cnt in df["return_reason"].value_counts().items():
        print(f"    {reason:<20} {cnt:>5,}  ({cnt/total*100:.1f}%)")

    # Order value stats
    print(f"\n  Order value stats (INR):")
    ov = df["order_value"]
    print(f"    min    : ₹{ov.min():>10,.2f}")
    print(f"    mean   : ₹{ov.mean():>10,.2f}")
    print(f"    median : ₹{ov.median():>10,.2f}")
    print(f"    max    : ₹{ov.max():>10,.2f}")

    # Feature sanity check
    print(f"\n  Feature sanity (abusive records only):")
    ab = df[df["is_label_abusive"] == True]
    print(f"    No images submitted    : {(~ab['images_submitted']).sum():,}  "
          f"({(~ab['images_submitted']).mean()*100:.1f}%)")
    print(f"    No support contacted   : {(~ab['support_contacted']).sum():,}  "
          f"({(~ab['support_contacted']).mean()*100:.1f}%)")
    print(f"    Avg return_rate_lifetime (abuse)  : "
          f"{ab['return_rate_lifetime'].mean():.3f}")
    print(f"    Avg return_rate_lifetime (legit)  : "
          f"{df[df['is_label_abusive'] == False]['return_rate_lifetime'].mean():.3f}")
    print(f"    Avg days_to_return (abuse)        : "
          f"{ab['days_to_return'].mean():.1f}")
    print(f"    Avg days_to_return (legit)        : "
          f"{df[df['is_label_abusive'] == False]['days_to_return'].mean():.1f}")

    print(f"\n  Columns generated:")
    for col in df.columns:
        print(f"    {col}")

    print(f"\n{SEP}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("\n  AI Risk Manager — Synthetic Data Generator")
    print("  Phase 1: Return Risk Scorer\n")

    df = generate(NUM_RECORDS)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved → {OUTPUT_PATH}")

    validate(df)
    print("  Done. Ready for EDA and feature engineering.\n")


if __name__ == "__main__":
    main()
