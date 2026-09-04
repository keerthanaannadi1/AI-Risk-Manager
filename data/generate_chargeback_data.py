"""
generate_chargeback_data.py
---------------------------
Generates ~8,000 synthetic chargeback dispute records for Phase 3:
Chargeback Evidence Responder.

Output: data/raw/chargebacks.csv

Class distribution:
  ~60% merchant wins   (merchant_won = True)
  ~40% merchant losses (merchant_won = False)

Dispute reasons simulated:
  - not_received       : customer claims product never arrived
  - not_authorized     : customer claims they never made the transaction
  - not_as_described   : product was different from what was shown
  - duplicate_charge   : customer claims they were charged twice
  - credit_not_processed: customer returned item but got no refund

Win/loss patterns:
  Merchant wins when:
    - Delivery is confirmed (tracking + signature)
    - OTP/2FA was used during purchase
    - IP/device logs are available
    - Customer contacted support before disputing
    - Customer disputed late (many days after delivery)

  Merchant loses when:
    - No delivery proof
    - No OTP/2FA
    - Customer is first-time buyer
    - High-value item
    - Customer disputed quickly after delivery

All data is synthetic. No real customer, transaction, or merchant data is used.
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
NUM_RECORDS   = 8_000
WIN_RATE      = 0.60       # 60% merchant wins
NUM_CUSTOMERS = 2_000
NUM_MERCHANTS = 200
OUTPUT_PATH   = os.path.join(os.path.dirname(__file__), "raw", "chargebacks.csv")

# ── Dispute reasons and their base difficulty (how hard to win) ───────────────
# not_authorized is hardest to win — bank almost always sides with customer
# duplicate_charge is easiest to win — easy to prove with transaction logs
DISPUTE_REASONS = [
    "not_received",
    "not_authorized",
    "not_as_described",
    "duplicate_charge",
    "credit_not_processed",
]

DISPUTE_DIFFICULTY = {
    "not_received":        0.5,   # medium — need delivery proof
    "not_authorized":      0.8,   # hard — need IP logs + OTP
    "not_as_described":    0.6,   # medium-hard — subjective
    "duplicate_charge":    0.2,   # easy — transaction logs prove it
    "credit_not_processed": 0.4,  # medium-easy — refund logs prove it
}

# ── Product categories ─────────────────────────────────────────────────────────
PRODUCT_CATEGORIES = ["electronics", "apparel", "books", "home", "luxury", "gift_cards"]

PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet", "cod"]


# ══════════════════════════════════════════════════════════════════════════════
# Customer pool
# ══════════════════════════════════════════════════════════════════════════════

def build_customer_pool(n: int) -> dict:
    customers = {}
    for i in range(n):
        cid = f"C_{10000 + i}"

        # ~10% of customers are serial disputers — they file chargebacks repeatedly
        is_serial_disputer = random.random() < 0.10

        if is_serial_disputer:
            account_age         = random.randint(30, 500)
            total_orders        = random.randint(5, 30)
            past_disputes       = random.randint(2, 8)
            avg_order_value     = random.uniform(500, 8000)
        else:
            account_age         = random.randint(60, 1800)
            total_orders        = random.randint(1, 80)
            past_disputes       = random.randint(0, 1)
            avg_order_value     = float(np.clip(np.random.lognormal(8.2, 0.7), 100, 30_000))

        customers[cid] = {
            "is_serial_disputer":  is_serial_disputer,
            "account_age_days":    account_age,
            "total_orders":        max(total_orders, 1),
            "past_disputes":       past_disputes,
            "avg_order_value":     round(avg_order_value, 2),
        }
    return customers


# ══════════════════════════════════════════════════════════════════════════════
# Merchant pool
# ══════════════════════════════════════════════════════════════════════════════

def build_merchant_pool(n: int) -> dict:
    merchants = {}
    for i in range(n):
        mid = f"M_{5000 + i}"
        merchants[mid] = {
            "primary_category":      random.choice(PRODUCT_CATEGORIES),
            "merchant_chargeback_rate": round(random.uniform(0.01, 0.15), 3),
            # Whether the merchant has good evidence practices
            "has_good_evidence_practices": random.random() < 0.60,
        }
    return merchants


# ══════════════════════════════════════════════════════════════════════════════
# Record assembler
# Takes all components and builds the full row dict with engineered features
# ══════════════════════════════════════════════════════════════════════════════

def _build_record(
    record_id, cid, mid, customer, merchant,
    order_value, category, payment_method,
    dispute_reason, transaction_dt, dispute_dt,
    delivery_confirmed, delivery_days,
    customer_signed_delivery, otp_used,
    ip_logs_available, order_confirmation_sent,
    customer_contacted_support, refund_issued,
    merchant_won,
):
    account_age      = customer["account_age_days"]
    total_orders     = customer["total_orders"]
    past_disputes    = customer["past_disputes"]
    avg_ov           = customer["avg_order_value"]

    days_to_dispute  = (dispute_dt - transaction_dt).days

    # ── Engineered features ───────────────────────────────────────────────────

    # Is the dispute filed very quickly? (within 3 days — suspicious)
    is_quick_dispute = int(days_to_dispute <= 3)

    # Is the dispute filed very late? (after 60 days — weaker claim)
    is_late_dispute = int(days_to_dispute > 60)

    # High value order — harder to win without strong evidence
    is_high_value = int(order_value > 5_000)

    # First order — no history to reference, harder to defend
    is_first_order = int(total_orders == 1)

    # New account — less trust history
    is_new_account = int(account_age <= 30)

    # Serial disputer — customer has filed multiple chargebacks before
    is_serial_disputer = int(past_disputes >= 2)

    # Strong evidence score — how much evidence does the merchant have?
    # Range 0 to 5 — each piece of evidence adds 1 point
    evidence_score = (
        int(delivery_confirmed) +
        int(customer_signed_delivery) +
        int(otp_used) +
        int(ip_logs_available) +
        int(order_confirmation_sent)
    )

    # Dispute difficulty — how hard is this type of dispute to win?
    dispute_difficulty = DISPUTE_DIFFICULTY[dispute_reason]

    # Order value normalized by customer average
    order_value_normalized = round(order_value / avg_ov, 4) if avg_ov > 0 else 1.0

    # COD payment — no card dispute possible, auto-win for not_authorized
    is_cod = int(payment_method == "cod")

    return {
        # ── Identifiers ───────────────────────────────────────────────────────
        "dispute_id":       f"D_{record_id:05d}",
        "customer_id":      cid,
        "merchant_id":      mid,

        # ── Transaction fields ─────────────────────────────────────────────────
        "order_value":          round(order_value, 2),
        "product_category":     category,
        "payment_method":       payment_method,
        "transaction_date":     transaction_dt.strftime("%Y-%m-%d"),
        "dispute_date":         dispute_dt.strftime("%Y-%m-%d"),
        "days_to_dispute":      days_to_dispute,
        "dispute_reason":       dispute_reason,

        # ── Delivery fields ────────────────────────────────────────────────────
        "delivery_confirmed":       int(delivery_confirmed),
        "delivery_days":            delivery_days,
        "customer_signed_delivery": int(customer_signed_delivery),

        # ── Evidence fields ────────────────────────────────────────────────────
        "otp_used":                 int(otp_used),
        "ip_logs_available":        int(ip_logs_available),
        "order_confirmation_sent":  int(order_confirmation_sent),
        "refund_issued":            int(refund_issued),

        # ── Customer behaviour ─────────────────────────────────────────────────
        "customer_contacted_support":  int(customer_contacted_support),
        "customer_account_age_days":   account_age,
        "customer_total_orders":       total_orders,
        "customer_past_disputes":      past_disputes,
        "customer_avg_order_value":    round(avg_ov, 2),

        # ── Merchant fields ────────────────────────────────────────────────────
        "merchant_chargeback_rate":    merchant["merchant_chargeback_rate"],

        # ── Engineered features ────────────────────────────────────────────────
        "is_quick_dispute":        is_quick_dispute,
        "is_late_dispute":         is_late_dispute,
        "is_high_value":           is_high_value,
        "is_first_order":          is_first_order,
        "is_new_account":          is_new_account,
        "is_serial_disputer":      is_serial_disputer,
        "is_cod":                  is_cod,
        "evidence_score":          evidence_score,
        "dispute_difficulty":      dispute_difficulty,
        "order_value_normalized":  order_value_normalized,

        # ── Label ──────────────────────────────────────────────────────────────
        "merchant_won": int(merchant_won),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Winning dispute generator
# Merchant has strong evidence — delivery confirmed, OTP used, logs available
# ══════════════════════════════════════════════════════════════════════════════

def generate_winning_record(record_id, cid, mid, customer, merchant):
    category       = random.choice(PRODUCT_CATEGORIES)
    order_value    = float(np.clip(np.random.lognormal(8.2, 0.7), 100, 20_000))
    payment_method = random.choices(
        PAYMENT_METHODS, weights=[0.30, 0.35, 0.15, 0.12, 0.08]
    )[0]

    dispute_reason = random.choices(
        DISPUTE_REASONS, weights=[0.35, 0.20, 0.20, 0.15, 0.10]
    )[0]

    days_ago         = random.randint(30, 365)
    transaction_dt   = datetime.now() - timedelta(days=days_ago)
    days_to_dispute  = random.randint(10, 90)   # disputed after reasonable time
    dispute_dt       = transaction_dt + timedelta(days=days_to_dispute)

    delivery_days    = random.randint(2, 7)

    # Winning merchants have strong evidence
    delivery_confirmed      = True
    customer_signed         = random.random() < 0.75
    otp_used                = random.random() < 0.80
    ip_logs_available       = True
    order_confirmation_sent = True
    customer_contacted      = random.random() < 0.55
    refund_issued           = False

    # COD with not_authorized is always a win — can't dispute cash payment
    if payment_method == "cod" and dispute_reason == "not_authorized":
        merchant_won = True
    else:
        merchant_won = True

    return _build_record(
        record_id, cid, mid, customer, merchant,
        order_value, category, payment_method,
        dispute_reason, transaction_dt, dispute_dt,
        delivery_confirmed, delivery_days,
        customer_signed, otp_used,
        ip_logs_available, order_confirmation_sent,
        customer_contacted, refund_issued,
        merchant_won=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Losing dispute generator
# Merchant lacks evidence — no delivery proof, no OTP, no logs
# ══════════════════════════════════════════════════════════════════════════════

def generate_losing_record(record_id, cid, mid, customer, merchant):
    category       = random.choice(PRODUCT_CATEGORIES)
    order_value    = float(np.clip(np.random.lognormal(8.5, 0.6), 500, 50_000))
    payment_method = random.choices(
        PAYMENT_METHODS, weights=[0.50, 0.20, 0.15, 0.10, 0.05]
    )[0]

    # Harder dispute types are more common in losses
    dispute_reason = random.choices(
        DISPUTE_REASONS, weights=[0.40, 0.30, 0.15, 0.08, 0.07]
    )[0]

    days_ago        = random.randint(10, 180)
    transaction_dt  = datetime.now() - timedelta(days=days_ago)
    days_to_dispute = random.randint(1, 20)    # disputed quickly
    dispute_dt      = transaction_dt + timedelta(days=days_to_dispute)

    delivery_days   = random.randint(5, 15)    # slow delivery

    # Losing merchants lack evidence
    delivery_confirmed      = random.random() < 0.30   # often no proof
    customer_signed         = False
    otp_used                = random.random() < 0.25   # often no OTP
    ip_logs_available       = random.random() < 0.35   # often no logs
    order_confirmation_sent = random.random() < 0.50
    customer_contacted      = random.random() < 0.20   # customer went straight to bank
    refund_issued           = random.random() < 0.15   # merchant rarely pre-empted

    return _build_record(
        record_id, cid, mid, customer, merchant,
        order_value, category, payment_method,
        dispute_reason, transaction_dt, dispute_dt,
        delivery_confirmed, delivery_days,
        customer_signed, otp_used,
        ip_logs_available, order_confirmation_sent,
        customer_contacted, refund_issued,
        merchant_won=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main generation
# ══════════════════════════════════════════════════════════════════════════════

def generate(n: int = NUM_RECORDS) -> pd.DataFrame:
    print(f"  Building customer pool ({NUM_CUSTOMERS} customers) ...")
    customers = build_customer_pool(NUM_CUSTOMERS)

    print(f"  Building merchant pool ({NUM_MERCHANTS} merchants) ...")
    merchants = build_merchant_pool(NUM_MERCHANTS)

    customer_ids = list(customers.keys())
    merchant_ids = list(merchants.keys())

    num_wins   = int(n * WIN_RATE)
    num_losses = n - num_wins

    records = []

    print(f"  Generating {num_wins:,} winning dispute records ...")
    for i in range(num_wins):
        cid = random.choice(customer_ids)
        mid = random.choice(merchant_ids)
        records.append(
            generate_winning_record(i + 1, cid, mid, customers[cid], merchants[mid])
        )

    print(f"  Generating {num_losses:,} losing dispute records ...")
    for i in range(num_losses):
        cid = random.choice(customer_ids)
        mid = random.choice(merchant_ids)
        records.append(
            generate_losing_record(num_wins + i + 1, cid, mid, customers[cid], merchants[mid])
        )

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Validation report
# ══════════════════════════════════════════════════════════════════════════════

def validate(df: pd.DataFrame) -> None:
    SEP = "=" * 58
    print(f"\n{SEP}")
    print("  DATA VALIDATION REPORT")
    print(SEP)

    total    = len(df)
    wins     = df["merchant_won"].sum()
    win_pct  = wins / total * 100

    print(f"\n  Total records        : {total:,}")
    print(f"  Total columns        : {len(df.columns)}")
    print(f"\n  Merchant wins        : {wins:,}  ({win_pct:.1f}%)")
    print(f"  Merchant losses      : {total - wins:,}  ({100 - win_pct:.1f}%)")

    print(f"\n  Dispute reason breakdown:")
    for reason, cnt in df["dispute_reason"].value_counts().items():
        win_rate = df[df["dispute_reason"] == reason]["merchant_won"].mean() * 100
        print(f"    {reason:<25} {cnt:>5,}  (win rate: {win_rate:.1f}%)")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"\n  Missing values       : {'None' if len(missing) == 0 else ''}")
    for col, cnt in missing.items():
        print(f"    {col}: {cnt}")

    print(f"\n  Evidence sanity (wins vs losses):")
    wins_df   = df[df["merchant_won"] == True]
    losses_df = df[df["merchant_won"] == False]
    signals = [
        ("delivery_confirmed",       "Delivery confirmed"),
        ("otp_used",                 "OTP used"),
        ("ip_logs_available",        "IP logs available"),
        ("customer_signed_delivery", "Customer signed"),
        ("order_confirmation_sent",  "Order confirmation sent"),
        ("customer_contacted_support", "Customer contacted support"),
    ]
    print(f"    {'Signal':<35} {'Wins %':<10} {'Losses %':<10}")
    print(f"    {'-'*55}")
    for col, label in signals:
        wp = wins_df[col].mean() * 100
        lp = losses_df[col].mean() * 100
        print(f"    {label:<35} {wp:<10.1f} {lp:<10.1f}")

    print(f"\n  Avg order value:")
    print(f"    Wins   : Rs.{wins_df['order_value'].mean():>10,.2f}")
    print(f"    Losses : Rs.{losses_df['order_value'].mean():>10,.2f}")

    print(f"\n  Avg days to dispute:")
    print(f"    Wins   : {wins_df['days_to_dispute'].mean():.1f} days")
    print(f"    Losses : {losses_df['days_to_dispute'].mean():.1f} days")

    print(f"\n  Avg evidence score (0-5):")
    print(f"    Wins   : {wins_df['evidence_score'].mean():.2f}")
    print(f"    Losses : {losses_df['evidence_score'].mean():.2f}")

    print(f"\n  Columns generated:")
    for col in df.columns:
        print(f"    {col}")
    print(f"\n{SEP}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n  AI Risk Manager — Synthetic Data Generator")
    print("  Phase 3: Chargeback Evidence Responder\n")

    df = generate(NUM_RECORDS)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved → {OUTPUT_PATH}")

    validate(df)
    print("  Done. Ready for EDA and feature engineering.\n")


if __name__ == "__main__":
    main()
