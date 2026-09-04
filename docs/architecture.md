# Architecture

## What This System Does

AI Risk Manager is a backend scoring API. It exposes three ML models as REST endpoints. Each endpoint takes a set of features about an event (a return request, a payment transaction, or a chargeback dispute), runs it through a trained model, and returns a risk score with a recommended action.

The system is defence-only. It never blocks or penalises anything on its own. It scores and recommends — the merchant's system decides what to actually do.

---

## How a Request Flows Through the System

Every module follows the same pattern end to end:

```
Incoming event (return / transaction / chargeback)
        │
        ▼
FastAPI endpoint receives the request
Pydantic validates all fields — bad input returns 422 immediately
        │
        ▼
Features are mapped and encoded
(categoricals → integers, booleans → 0/1, columns ordered to match training)
        │
        ▼
Model runs predict_proba() → probability between 0.0 and 1.0
        │
        ▼
Score is bucketed against thresholds → Low / Medium / High
        │
        ▼
Response returned: score, label, recommendation, model name, timestamp
```

---

## Module Status

| Module | Endpoint | Status |
|--------|----------|--------|
| Return Risk Scorer | `POST /score/return` | ✅ Complete |
| Fraud Transaction Scorer | `POST /score/transaction` | ✅ Complete |
| Chargeback Evidence Responder | `POST /chargeback/analyze` | ✅ Complete |

---

## Module 1: Return Risk Scorer

Scores a return request as Low / Medium / High abuse risk.

**What it detects:** Wardrobing (buy, use, return), swap fraud (return a different item), serial returners, refund fishing.

**Model:** XGBoost + LightGBM soft-voting ensemble  
**Saved to:** `models/return_risk_model.pkl`  
**Training data:** 10,000 synthetic return records (85% legitimate / 15% abusive)

### Decision thresholds

| Score | Label | Default recommendation |
|-------|-------|------------------------|
| < 0.40 | Low | Auto-approve |
| 0.40 – 0.70 | Medium | Flag for manual review |
| ≥ 0.70 | High | Hold return, manual review required |

### Key features the model uses

```
customer_total_orders       how many orders this customer has placed lifetime
customer_total_returns      how many returns lifetime
return_rate_lifetime        returns / total orders (lifetime)
return_rate_30d             returns / orders in the last 30 days
days_to_return              days between purchase and return request
return_reason               defective / wrong_item / quality_issue / changed_mind / not_needed
images_submitted            did the customer attach photos?
support_contacted           did they contact support before requesting the return?
product_category            electronics / apparel / books / home / luxury
order_value                 value of the order being returned
is_near_deadline            is the return request close to the policy deadline?
is_new_account              account less than 30 days old?
category_risk_score         pre-computed risk weight for the product category
```

### What the API returns

```json
{
  "risk_score": 0.87,
  "risk_label": "High",
  "recommendation": "Hold return. Manual review required before processing.",
  "model_name": "Ensemble",
  "threshold_used": 0.40,
  "scored_at": "2026-09-04T11:43:00+00:00"
}
```

---

## Module 2: Fraud Transaction Scorer

Scores a payment transaction for fraud risk in real time.

**What it detects:** Stolen card usage, account takeover, velocity abuse (many orders in a short window), new-account high-value fraud, VPN/proxy transactions, address mismatches.

**Model:** LightGBM (selected over XGBoost and ensemble on validation AUPRC)  
**Saved to:** `models/fraud_risk_model.pkl`  
**Training data:** 10,000 synthetic transaction records (~10% fraud)

### Decision thresholds

| Score | Label | Default recommendation |
|-------|-------|------------------------|
| < 0.40 | Low | Approve transaction |
| 0.40 – 0.70 | Medium | Flag for review, consider step-up auth (OTP/2FA) |
| ≥ 0.70 | High | Block transaction, manual review required |

### Key features the model uses

```
order_value                 transaction amount in INR
payment_method              card / upi / netbanking / wallet / cod
hour_of_day                 hour the transaction was placed (0–23)
failed_attempts             number of failed payment attempts before success
is_vpn                      was the transaction made through a VPN or proxy?
customer_account_age_days   how old is the customer account?
customer_orders_1h          how many orders this customer placed in the last hour
customer_orders_24h         how many orders in the last 24 hours
customer_past_fraud_flags   how many times this customer was previously flagged
is_new_device               device not seen on this account before?
is_different_city           shipping city different from customer's usual city?
address_mismatch            billing and shipping address mismatch?
multiple_failed_attempts    more than 2 failed attempts before this transaction?
high_velocity_1h            more than 3 orders in the last hour?
new_account_high_value_card account < 30 days old, order > ₹5000, paying by card?
payment_risk_score          pre-computed risk weight for the payment method
```

### What the API returns

```json
{
  "risk_score": 0.94,
  "risk_label": "High",
  "recommendation": "Block transaction. Manual review required before processing.",
  "model_name": "LightGBM",
  "threshold_used": 0.38,
  "scored_at": "2026-09-04T11:43:00+00:00"
}
```

---

## Module 3: Chargeback Evidence Responder

When a bank raises a chargeback dispute, this module scores how likely the merchant is to win and tells them exactly what evidence they have and what's missing.

**What it scores:** Winnability of the dispute (Weak / Moderate / Strong), based on the evidence the merchant holds and the nature of the dispute.

**Model:** XGBoost + LightGBM soft-voting ensemble  
**Saved to:** `models/chargeback_model.pkl`  
**Training data:** 8,000 synthetic chargeback records (~60% merchant wins)

### Winability thresholds

| Score | Label | Default recommendation |
|-------|-------|------------------------|
| < 0.40 | Weak | Consider settling with the customer to avoid the chargeback fee |
| 0.40 – 0.70 | Moderate | Collect any missing evidence before submitting |
| ≥ 0.70 | Strong | Submit all available evidence immediately |

### Key features the model uses

```
dispute_reason              not_received / not_authorized / not_as_described / duplicate_charge / credit_not_processed
days_to_dispute             how many days after the transaction the dispute was raised
delivery_confirmed          is there a tracking number confirming delivery?
customer_signed_delivery    did the customer sign for the delivery?
otp_used                    was OTP or 2FA used during the original purchase?
ip_logs_available           are device and IP logs available for the transaction?
order_confirmation_sent     was a confirmation sent to the customer at time of purchase?
refund_issued               has a refund already been issued?
customer_contacted_support  did the customer reach out before disputing?
customer_past_disputes      how many previous disputes this customer has raised
is_serial_disputer          customer has 2 or more past disputes?
evidence_score              count of the 7 evidence fields that are true (0–7)
dispute_difficulty          pre-computed difficulty weight for the dispute reason type
```

### What the API returns

The chargeback endpoint returns more than just a score — it builds a full evidence checklist:

```json
{
  "winability_score": 0.73,
  "winability_label": "Strong",
  "recommendation": "Strong case. Submit all available evidence immediately. Likely to win.",
  "evidence_score": 5,
  "evidence_present": [
    "Delivery confirmed (tracking number available)",
    "OTP / 2FA used during purchase",
    "IP address and device logs available",
    "Order confirmation sent to customer email/SMS",
    "Customer contacted support before disputing"
  ],
  "evidence_missing": [
    "Customer signed for delivery",
    "Refund already issued to customer"
  ],
  "model_name": "Ensemble",
  "threshold_used": 0.45,
  "scored_at": "2026-09-04T11:43:00+00:00"
}
```

---

## API

All three modules are served from a single FastAPI app.

```
uvicorn api.main:app --reload
# Interactive docs at: http://127.0.0.1:8000/docs
```

### Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| GET | `/` | Health check — confirms all three models are loaded |
| GET | `/model/return/info` | Return model metrics and threshold |
| GET | `/model/fraud/info` | Fraud model metrics and threshold |
| GET | `/model/chargeback/info` | Chargeback model metrics and threshold |
| POST | `/score/return` | Score a return request |
| POST | `/score/transaction` | Score a payment transaction |
| POST | `/chargeback/analyze` | Analyze a chargeback dispute |

All three models are loaded once at server startup from their `.pkl` files. If a model file is missing, the server refuses to start with a clear error message pointing to the training script that needs to be run first.

### Health check response

```json
{
  "status": "ok",
  "models_loaded": {
    "return_risk_scorer": "Ensemble",
    "fraud_transaction_scorer": "LightGBM",
    "chargeback_responder": "Ensemble"
  },
  "message": "AI Risk Manager is live. All three models ready."
}
```

---

## Project Structure

```
AI-Risk-Manager/
├── data/
│   ├── generate_data.py              # generates synthetic return records
│   ├── generate_fraud_data.py        # generates synthetic transaction records
│   ├── generate_chargeback_data.py   # generates synthetic chargeback records
│   └── raw/
│       ├── returns.csv               # 10,000 records
│       ├── transactions.csv          # 10,000 records
│       └── chargebacks.csv           # 8,000 records
├── src/
│   ├── features.py                   # feature engineering for returns
│   ├── fraud_features.py             # feature engineering for transactions
│   ├── chargeback_features.py        # feature engineering for chargebacks
│   ├── train.py                      # trains the return risk model
│   ├── fraud_train.py                # trains the fraud model
│   └── chargeback_train.py           # trains the chargeback model
├── api/
│   └── main.py                       # FastAPI app — all three endpoints
├── models/
│   ├── return_risk_model.pkl         # saved return model
│   ├── fraud_risk_model.pkl          # saved fraud model
│   └── chargeback_model.pkl          # saved chargeback model
├── tests/
│   ├── test_model.py                 # 24 tests for the return module
│   ├── test_fraud_model.py           # 25 tests for the fraud module
│   └── test_chargeback_model.py      # 36 tests for the chargeback module
├── notebooks/
│   ├── exploration.ipynb             # EDA for returns
│   ├── exploration_fraud.ipynb       # EDA for transactions
│   └── exploration_chargeback.ipynb  # EDA for chargebacks
└── docs/
    ├── architecture.md               # this file
    ├── problem-statement.md
    ├── evaluation-strategy.md
    ├── data-dictionary.md
    └── roadmap.md
```

---

## Technology choices

| Component | Choice | Why |
|-----------|--------|-----|
| ML models | XGBoost + LightGBM | Both handle tabular data well and train fast. Ensemble of the two consistently outperforms either alone on this dataset. |
| API | FastAPI | Pydantic validation means bad inputs are rejected before they reach the model. Auto-generates interactive docs at `/docs`. |
| Data generation | Faker + NumPy | Produces realistic-looking synthetic records without any real PII. |
| Data processing | Pandas | Standard for tabular data work in Python. |
| Model serialisation | pickle | Simple and works. The models are only loaded locally — no untrusted sources. |
| Testing | Pytest | Fixtures load the test split once per session using the same random seed as training, so metric assertions are deterministic. |

---

## Running the full pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data
python data/generate_data.py
python data/generate_fraud_data.py
python data/generate_chargeback_data.py

# 3. Train all three models
python src/train.py
python src/fraud_train.py
python src/chargeback_train.py

# 4. Run tests (expects 85 passed)
pytest tests/ -v

# 5. Start the API
uvicorn api.main:app --reload
```
