"""
generate_fraud_data.py
----------------------
Generates ~10,000 synthetic payment transaction records for Phase 2: Fraud Transaction Scorer.

Output: data/raw/transactions.csv

Class distribution:
  ~85% legitimate transactions  (is_fraud = False)
  ~15% fraudulent transactions  (is_fraud = True)

Fraud patterns simulated:
  - Card Testing       : small amounts, multiple failures, night time, new device
  - Account Takeover   : new device + new city + immediate high-value purchase
  - Triangulation Fraud: stolen card, billing/shipping address mismatch, electronics/gift cards
  - Velocity Abuse     : many orders in short time from new account

All data is synthetic. No real customer, transaction, or merchant data is used.
"""

import os
import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

# ── Reproducibility ───────────────────────────────────────────────────────────
# Setting seeds means every run produces the exact same 10,000 rows.
# This is critical so train/test splits are always consistent.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_IN")
fake.seed_instance(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
NUM_RECORDS   = 10_000   # total rows to generate
FRAUD_RATE    = 0.15     # 15% fraudulent = ~1,500 fraud rows
NUM_CUSTOMERS = 2_000    # pool of unique customers
NUM_MERCHANTS = 200      # pool of unique merchants
OUTPUT_PATH   = os.path.join(os.path.dirname(__file__), "raw", "transactions.csv")

# ── Product categories and their fraud risk scores ────────────────────────────
# Gift cards and electronics are easiest to resell → highest fraud risk
PRODUCT_CATEGORIES = ["electronics", "apparel", "books", "home", "luxury", "gift_cards"]
CATEGORY_RISK = {
    "electronics": 0.8,
    "gift_cards":  0.9,   # highest — instant resale value
    "luxury":      0.85,
    "apparel":     0.4,
    "books":       0.1,   # lowest — not worth stealing a card for
    "home":        0.3,
}

# ── Payment methods and their fraud risk scores ───────────────────────────────
# Card is highest risk — fraudsters steal card details
# Netbanking is hardest to abuse — requires full account credentials
PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet", "cod"]
PAYMENT_RISK = {
    "card":       0.9,
    "upi":        0.3,
    "netbanking": 0.2,
    "wallet":     0.4,
    "cod":        0.5,
}

# ── Indian cities for location simulation ─────────────────────────────────────
# Each customer has a "usual city". A transaction from a different city is a red flag.
CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
    "Surat", "Kanpur", "Nagpur", "Indore", "Bhopal",
]


# Customer pool
# Each customer has a fixed profile: usual city, usual device, account age, etc.
# This lets us detect anomalies — a transaction from a different device or city
# is suspicious only because we know the customer's normal behaviour.


def build_customer_pool(n: int) -> dict:
    customers = {}
    for i in range(n):
        cid = f"C_{10000 + i}"

        # 12% of customers are pre-flagged as known bad actors
        # These customers drive most of the fraud records
        is_known_fraudster = random.random() < 0.12

        if is_known_fraudster:
            account_age       = random.randint(5, 120)    # newer accounts
            total_past_orders = random.randint(1, 10)
            avg_order_value   = float(np.clip(np.random.lognormal(8.5, 0.5), 200, 30_000))
        else:
            account_age       = random.randint(60, 1800)  # older, established accounts
            total_past_orders = random.randint(3, 100)
            avg_order_value   = float(np.clip(np.random.lognormal(8.2, 0.7), 100, 40_000))

        customers[cid] = {
            "is_known_fraudster": is_known_fraudster,
            "account_age_days":   account_age,
            "total_past_orders":  max(total_past_orders, 1),
            "avg_order_value":    round(avg_order_value, 2),
            "usual_city":         random.choice(CITIES),
            # Each customer has 1-3 known devices. New device = red flag.
            "known_devices":      [f"DEV_{random.randint(1000,9999)}"
                                   for _ in range(random.randint(1, 3))],
            "past_fraud_flags":   random.randint(0, 2) if is_known_fraudster else 0,
        }
    return customers


# Merchant pool
# Each merchant has a category focus and a baseline fraud rate


def build_merchant_pool(n: int) -> dict:
    merchants = {}
    for i in range(n):
        mid = f"M_{5000 + i}"
        merchants[mid] = {
            "primary_category":    random.choice(PRODUCT_CATEGORIES),
            "merchant_fraud_rate": round(random.uniform(0.02, 0.20), 3),
        }
    return merchants

# Record assembler
# Takes all the raw components and builds the full row dict.
# Also computes all engineered features so EDA can use them directly.


def _build_record(
    record_id, cid, mid, customer, merchant,
    order_value, category, payment_method,
    transaction_dt, device_id, transaction_city,
    billing_city, shipping_city,
    failed_attempts, is_vpn,
    customer_orders_1h, customer_orders_24h,
    is_fraud, fraud_pattern,
):
    account_age       = customer["account_age_days"]
    total_past_orders = customer["total_past_orders"]
    avg_ov            = customer["avg_order_value"]
    usual_city        = customer["usual_city"]
    known_devices     = customer["known_devices"]
    past_fraud_flags  = customer["past_fraud_flags"]

    hour_of_day  = transaction_dt.hour
    day_of_week  = transaction_dt.weekday()   # 0=Monday, 6=Sunday

    # ── Engineered features ───────────────────────────────────────────────────

    # Night transaction: before 6am or after 11pm — fraudsters prefer off-hours
    is_night_transaction = int(hour_of_day < 6 or hour_of_day >= 23)

    # Weekend: slightly higher fraud on weekends (less monitoring)
    is_weekend = int(day_of_week >= 5)

    # New device: device not in the customer's known device list
    is_new_device = int(device_id not in known_devices)

    # Different city: transaction city is not the customer's usual city
    is_different_city = int(transaction_city != usual_city)

    # High value: order value above ₹5,000
    is_high_value = int(order_value > 5_000)

    # First order: customer has never ordered before
    is_first_order = int(total_past_orders == 1)

    # New account: account less than 30 days old
    is_new_account = int(account_age <= 30)

    # Address mismatch: billing city ≠ shipping city (drop address pattern)
    address_mismatch = int(billing_city != shipping_city)

    # Multiple failures: more than 2 failed attempts before success = card testing
    multiple_failed_attempts = int(failed_attempts > 2)

    # High velocity: more than 3 orders in the last hour from this customer
    high_velocity_1h = int(customer_orders_1h > 3)

    # Order value normalized: how unusual is this order compared to customer's average?
    # A ₹20,000 order from someone who normally spends ₹500 is suspicious
    order_value_normalized = round(order_value / avg_ov, 4) if avg_ov > 0 else 1.0

    # New account + high value + card = very suspicious combination
    new_account_high_value_card = int(
        is_new_account and is_high_value and payment_method == "card"
    )

    # Risk scores from lookup tables
    payment_risk_score  = PAYMENT_RISK[payment_method]
    category_risk_score = CATEGORY_RISK[category]

    return {
        # ── Identifiers ───────────────────────────────────────────────────────
        "transaction_id":   f"T_{record_id:05d}",
        "customer_id":      cid,
        "merchant_id":      mid,

        # ── Raw transaction fields ─────────────────────────────────────────────
        "order_value":          round(order_value, 2),
        "product_category":     category,
        "payment_method":       payment_method,
        "transaction_datetime": transaction_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "hour_of_day":          hour_of_day,
        "day_of_week":          day_of_week,
        "device_id":            device_id,
        "transaction_city":     transaction_city,
        "billing_city":         billing_city,
        "shipping_city":        shipping_city,
        "failed_attempts":      failed_attempts,
        "is_vpn":               int(is_vpn),

        # ── Raw customer fields ────────────────────────────────────────────────
        "customer_account_age_days": account_age,
        "customer_total_past_orders": total_past_orders,
        "customer_avg_order_value":  round(avg_ov, 2),
        "customer_past_fraud_flags": past_fraud_flags,
        "customer_orders_1h":        customer_orders_1h,
        "customer_orders_24h":       customer_orders_24h,

        # ── Raw merchant fields ────────────────────────────────────────────────
        "merchant_fraud_rate": merchant["merchant_fraud_rate"],

        # ── Engineered features ────────────────────────────────────────────────
        "is_night_transaction":        is_night_transaction,
        "is_weekend":                  is_weekend,
        "is_new_device":               is_new_device,
        "is_different_city":           is_different_city,
        "is_high_value":               is_high_value,
        "is_first_order":              is_first_order,
        "is_new_account":              is_new_account,
        "address_mismatch":            address_mismatch,
        "multiple_failed_attempts":    multiple_failed_attempts,
        "high_velocity_1h":            high_velocity_1h,
        "order_value_normalized":      order_value_normalized,
        "new_account_high_value_card": new_account_high_value_card,
        "payment_risk_score":          payment_risk_score,
        "category_risk_score":         category_risk_score,

        # ── Labels ─────────────────────────────────────────────────────────────
        "fraud_pattern":  fraud_pattern,   # for EDA/debugging only
        "is_fraud":       is_fraud,        # target variable
    }

# Legitimate transaction generator
# Normal customers: shop during the day, use known devices, ship to own address


def generate_legitimate_record(record_id, cid, mid, customer, merchant):
    category    = random.choices(
        PRODUCT_CATEGORIES, weights=[0.25, 0.30, 0.15, 0.20, 0.05, 0.05]
    )[0]

    # Legitimate order values follow a log-normal distribution around ₹1,500
    order_value = float(np.clip(np.random.lognormal(mean=8.2, sigma=0.7), 100, 20_000))

    # Normal shopping hours: 8am to 11pm
    hour        = random.randint(8, 22)
    days_ago    = random.randint(1, 180)
    dt          = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23))
    dt          = dt.replace(hour=hour, minute=random.randint(0, 59))

    # Known device and usual city — no anomaly
    device_id   = random.choice(customer["known_devices"])
    city        = customer["usual_city"]

    # Billing and shipping city usually match for legitimate orders
    billing_city  = city
    shipping_city = city if random.random() < 0.85 else random.choice(CITIES)

    # Legitimate users rarely fail more than once
    failed_attempts = random.choices([0, 1, 2], weights=[0.70, 0.20, 0.10])[0]

    payment_method = random.choices(
        PAYMENT_METHODS, weights=[0.25, 0.40, 0.15, 0.12, 0.08]
    )[0]

    return _build_record(
        record_id, cid, mid, customer, merchant,
        order_value, category, payment_method,
        dt, device_id, city,
        billing_city, shipping_city,
        failed_attempts, is_vpn=False,
        customer_orders_1h=random.randint(0, 1),
        customer_orders_24h=random.randint(1, 3),
        is_fraud=False, fraud_pattern="none",
    )

# Fraud pattern generators
# Each pattern has distinct signals the model should learn to detect


def generate_fraudulent_record(record_id, cid, mid, customer, merchant):
    # Pick one of four fraud patterns randomly
    pattern = random.choices(
        ["card_testing", "account_takeover", "triangulation", "velocity_abuse"],
        weights=[0.25, 0.25, 0.30, 0.20],
    )[0]

    # ── Pattern 1: Card Testing ───────────────────────────────────────────────
    # Fraudster steals a card and places tiny orders to check if it works.
    # Signals: small amount, many failures, 1-5am, new device, VPN
    if pattern == "card_testing":
        order_value     = random.uniform(1, 99)        # tiny test amount
        category        = random.choice(["books", "home"])  # cheap items
        payment_method  = "card"                       # always card
        hour            = random.randint(1, 5)         # middle of the night
        failed_attempts = random.randint(3, 10)        # many failures = testing
        device_id       = f"DEV_{random.randint(9000, 9999)}"  # new device
        city            = random.choice(CITIES)        # random city
        is_vpn          = random.random() < 0.75       # usually on VPN
        billing_city    = city
        shipping_city   = city
        orders_1h       = random.randint(4, 15)        # many attempts in one hour

    # ── Pattern 2: Account Takeover ───────────────────────────────────────────
    # Fraudster gains access to a real customer's account, then places a
    # large order immediately from a new device and different city.
    # Signals: new device + new city + high value + immediate purchase
    elif pattern == "account_takeover":
        order_value     = random.uniform(8_000, 50_000)   # high value
        category        = random.choice(["electronics", "luxury", "gift_cards"])
        payment_method  = "card"
        hour            = random.randint(0, 23)
        failed_attempts = random.randint(0, 1)            # uses saved card — no failures
        device_id       = f"DEV_{random.randint(9000, 9999)}"  # new device
        city            = random.choice([c for c in CITIES if c != customer["usual_city"]])
        is_vpn          = random.random() < 0.50
        billing_city    = customer["usual_city"]          # saved address from real account
        shipping_city   = city                            # ship to fraudster's location
        orders_1h       = random.randint(1, 2)

        # Override customer to look like an established account (was taken over)
        customer = customer.copy()
        customer["account_age_days"]   = random.randint(200, 1000)
        customer["total_past_orders"]  = random.randint(10, 50)

    # ── Pattern 3: Triangulation Fraud ────────────────────────────────────────
    # Fraudster takes orders on their own fake store, pays supplier using stolen
    # card, ships to real customer's address. Supplier (merchant) loses money.
    # Signals: billing ≠ shipping, card, electronics/gift_cards, high value
    elif pattern == "triangulation":
        order_value     = random.uniform(3_000, 30_000)
        category        = random.choice(["electronics", "gift_cards", "luxury"])
        payment_method  = "card"
        hour            = random.randint(8, 22)          # normal hours — less suspicious
        failed_attempts = random.randint(0, 2)
        device_id       = random.choice(
            customer["known_devices"] + [f"DEV_{random.randint(9000,9999)}"]
        )
        city            = customer["usual_city"]
        is_vpn          = random.random() < 0.30
        billing_city    = random.choice(CITIES)          # stolen card's billing city
        shipping_city   = random.choice(                 # ship to real customer
            [c for c in CITIES if c != billing_city]
        )
        orders_1h       = random.randint(1, 3)

    # ── Pattern 4: Velocity Abuse ─────────────────────────────────────────────
    # New account places many orders very quickly before being detected.
    # Signals: new account, many orders in 1h, medium-high value, card/wallet
    else:  # velocity_abuse
        order_value     = random.uniform(500, 8_000)
        category        = random.choice(["electronics", "apparel", "gift_cards"])
        payment_method  = random.choice(["card", "wallet"])
        hour            = random.randint(10, 20)
        failed_attempts = random.randint(0, 2)
        device_id       = f"DEV_{random.randint(9000, 9999)}"  # new device
        city            = random.choice(CITIES)
        is_vpn          = random.random() < 0.40
        billing_city    = city
        shipping_city   = city
        orders_1h       = random.randint(5, 20)          # very high velocity

        # Override customer to look like a brand new account
        customer = customer.copy()
        customer["account_age_days"]  = random.randint(1, 15)
        customer["total_past_orders"] = random.randint(1, 5)

    # Build the transaction datetime
    days_ago = random.randint(1, 180)
    dt = datetime.now() - timedelta(days=days_ago)
    dt = dt.replace(hour=hour, minute=random.randint(0, 59))

    return _build_record(
        record_id, cid, mid, customer, merchant,
        order_value, category, payment_method,
        dt, device_id, city,
        billing_city, shipping_city,
        failed_attempts, is_vpn,
        customer_orders_1h=orders_1h,
        customer_orders_24h=orders_1h + random.randint(0, 5),
        is_fraud=True, fraud_pattern=pattern,
    )

# Main generation function
# Builds the full dataset by calling legitimate and fraud generators


def generate(n: int = NUM_RECORDS) -> pd.DataFrame:
    print(f"  Building customer pool ({NUM_CUSTOMERS} customers) ...")
    customers = build_customer_pool(NUM_CUSTOMERS)

    print(f"  Building merchant pool ({NUM_MERCHANTS} merchants) ...")
    merchants = build_merchant_pool(NUM_MERCHANTS)

    customer_ids = list(customers.keys())
    merchant_ids = list(merchants.keys())

    # Separate known fraudsters from normal customers
    fraudster_ids = [c for c, v in customers.items() if v["is_known_fraudster"]]
    normal_ids    = [c for c, v in customers.items() if not v["is_known_fraudster"]]
    if not fraudster_ids:
        fraudster_ids = customer_ids

    num_fraud = int(n * FRAUD_RATE)
    num_legit = n - num_fraud

    records = []

    print(f"  Generating {num_legit:,} legitimate records ...")
    for i in range(num_legit):
        cid = random.choice(normal_ids or customer_ids)
        mid = random.choice(merchant_ids)
        records.append(
            generate_legitimate_record(i + 1, cid, mid, customers[cid], merchants[mid])
        )

    print(f"  Generating {num_fraud:,} fraudulent records ...")
    for i in range(num_fraud):
        # 75% of fraud comes from known fraudster customer profiles
        cid = random.choice(fraudster_ids) if random.random() < 0.75 else random.choice(customer_ids)
        mid = random.choice(merchant_ids)
        records.append(
            generate_fraudulent_record(num_legit + i + 1, cid, mid, customers[cid], merchants[mid])
        )

    df = pd.DataFrame(records)

    # Shuffle so all fraud rows aren't at the bottom
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df

# Validation report
# Prints a summary after generation so we can verify the data looks right

def validate(df: pd.DataFrame) -> None:
    SEP = "=" * 58
    print(f"\n{SEP}")
    print("  DATA VALIDATION REPORT")
    print(SEP)

    total     = len(df)
    fraud_n   = df["is_fraud"].sum()
    fraud_pct = fraud_n / total * 100

    print(f"\n  Total records        : {total:,}")
    print(f"  Total columns        : {len(df.columns)}")
    print(f"\n  Legitimate           : {total - fraud_n:,}  ({100 - fraud_pct:.1f}%)")
    print(f"  Fraudulent           : {fraud_n:,}  ({fraud_pct:.1f}%)")

    print(f"\n  Fraud pattern breakdown:")
    for pat, cnt in df["fraud_pattern"].value_counts().items():
        print(f"    {pat:<25} {cnt:>5,}  ({cnt/total*100:.1f}%)")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"\n  Missing values       : {'None ✓' if len(missing) == 0 else ''}")
    for col, cnt in missing.items():
        print(f"    {col}: {cnt}")

    print(f"\n  Payment method distribution:")
    for pm, cnt in df["payment_method"].value_counts().items():
        print(f"    {pm:<15} {cnt:>5,}  ({cnt/total*100:.1f}%)")

    print(f"\n  Product category distribution:")
    for cat, cnt in df["product_category"].value_counts().items():
        print(f"    {cat:<15} {cnt:>5,}  ({cnt/total*100:.1f}%)")

    print(f"\n  Order value stats (INR):")
    ov = df["order_value"]
    print(f"    min    : ₹{ov.min():>10,.2f}")
    print(f"    mean   : ₹{ov.mean():>10,.2f}")
    print(f"    median : ₹{ov.median():>10,.2f}")
    print(f"    max    : ₹{ov.max():>10,.2f}")

    print(f"\n  Feature sanity (fraud vs legit):")
    fr = df[df["is_fraud"] == True]
    lg = df[df["is_fraud"] == False]
    print(f"    Avg order_value        fraud: ₹{fr['order_value'].mean():>10,.2f}  "
          f"legit: ₹{lg['order_value'].mean():>10,.2f}")
    print(f"    Night transactions     fraud: {fr['is_night_transaction'].mean()*100:.1f}%  "
          f"legit: {lg['is_night_transaction'].mean()*100:.1f}%")
    print(f"    New device             fraud: {fr['is_new_device'].mean()*100:.1f}%  "
          f"legit: {lg['is_new_device'].mean()*100:.1f}%")
    print(f"    Address mismatch       fraud: {fr['address_mismatch'].mean()*100:.1f}%  "
          f"legit: {lg['address_mismatch'].mean()*100:.1f}%")
    print(f"    Multiple failed att.   fraud: {fr['multiple_failed_attempts'].mean()*100:.1f}%  "
          f"legit: {lg['multiple_failed_attempts'].mean()*100:.1f}%")
    print(f"    High velocity (1h)     fraud: {fr['high_velocity_1h'].mean()*100:.1f}%  "
          f"legit: {lg['high_velocity_1h'].mean()*100:.1f}%")

    print(f"\n  Columns generated:")
    for col in df.columns:
        print(f"    {col}")

    print(f"\n{SEP}\n")

# Entry point

def main():
    print("\n  AI Risk Manager — Synthetic Data Generator")
    print("  Phase 2: Fraud Transaction Scorer\n")

    df = generate(NUM_RECORDS)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved → {OUTPUT_PATH}")

    validate(df)
    print("  Done. Ready for EDA and feature engineering.\n")


if __name__ == "__main__":
    main()
