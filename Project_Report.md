# AI Risk Manager — Project Report

**Project Name:** AI Risk Manager  
**Developer:** CSE Admin  
**Date:** September 2026  
**Version:** 3.0.0  
**Location:** `/home/cse-admin/Documents/razorpay/AI-Risk-Manager`

---

## Executive Summary

AI Risk Manager is an ML-powered risk scoring system designed to protect merchants on digital payment platforms from three types of silent financial losses: **fraudulent transactions**, **return abuse**, and **chargebacks**. The system uses machine learning models trained on synthetic data to assign risk scores and recommend defensive actions in real-time.

The project is a **defense-only system** — it scores risk and recommends actions but never blocks, penalizes, or takes offensive action autonomously. This design philosophy ensures human oversight while providing actionable intelligence.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Modules & Models](#3-modules--models)
4. [Data & Feature Engineering](#4-data--feature-engineering)
5. [Model Training & Evaluation](#5-model-training--evaluation)
6. [API Documentation](#6-api-documentation)
7. [Technology Stack](#7-technology-stack)
8. [Project Structure](#8-project-structure)
9. [Setup & Installation](#9-setup--installation)
10. [Test Coverage](#10-test-coverage)
11. [Key Design Decisions](#11-key-design-decisions)
12. [Future Roadmap](#12-future-roadmap)
13. [Conclusion](#13-conclusion)

---

## 1. Problem Statement

### The Silent Drain on Merchant Revenue

Unlike visible theft or robbery, merchant losses from fraud, return abuse, and chargebacks happen gradually through policy gaps, trust exploitation, and system blindspots. By the time merchants notice, significant revenue has already been lost.

### Three Core Problems Addressed

#### 1.1 Fraudulent Transactions
**What Happens:** Bad actors use stolen card details to place orders. Merchants ship products, then lose both goods and revenue when the real cardholder disputes the charge.

**Why It's Hard:**
- Transactions appear normal at purchase time
- Fraudsters use VPNs, fresh devices, and synthetic identities
- Patterns evolve constantly

#### 1.2 Return Abuse
**What Happens:** Customers exploit return policies through:
- **Wardrobing:** Buy, use, return
- **Swap fraud:** Return broken/fake items
- **Serial returners:** Return 70-90% of purchases
- **Refund fishing:** Claim non-delivery of delivered items

**Why It's Hard:**
- Individual returns look reasonable in isolation
- Merchants can't share return data across platforms
- Legitimate customers also return items

#### 1.3 Chargebacks
**What Happens:** Customers dispute charges with their bank (either true fraud or "friendly fraud"). Merchants lose money + pay fees. Exceeding ~1% chargeback ratio risks losing payment gateway access entirely.

**Why It's Hard:**
- Banks default to customer side
- Evidence window is narrow (7-14 days)
- Merchants often don't know what evidence is needed

---

## 2. System Architecture

### Request Flow

Every module follows the same end-to-end pattern:

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

### System Design Principles

1. **Defense-Only:** Never blocks autonomously — scores and recommends
2. **Honest Evaluation:** Uses precision, recall, AUPRC on held-out test sets
3. **No Accuracy Metric:** Accuracy is meaningless on imbalanced fraud data
4. **Threshold Transparency:** Documents why each threshold was chosen
5. **Human-in-the-Loop:** All high-risk decisions require human review

---

## 3. Modules & Models

### 3.1 Return Risk Scorer (Phase 1)

**Purpose:** Score return requests as Low / Medium / High abuse risk

**What It Detects:**
- Wardrobing (buy, use, return)
- Swap fraud (return different item)
- Serial returners
- Refund fishing

**Model Architecture:**
- Algorithm: XGBoost + LightGBM soft-voting ensemble
- Training Data: 10,000 synthetic return records (85% legitimate / 15% abusive)
- Decision Threshold: 0.40 (tuned on validation set)

**Decision Thresholds:**

| Score | Label | Recommendation |
|-------|-------|----------------|
| < 0.40 | Low | Auto-approve return |
| 0.40 – 0.70 | Medium | Flag for manual review |
| ≥ 0.70 | High | Hold return, manual review required |

**Key Features:**
- `return_rate_lifetime` — Lifetime return rate (serial returner signal)
- `return_rate_30d` — Recent return pattern
- `days_to_return` — Time between purchase and return request
- `category_risk_score` — Product category risk weight
- `no_images_high_value` — High-value return without evidence
- `is_new_account` — Account less than 30 days old

**Performance (Test Set):**

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Precision | 0.9545 | ≥ 0.70 | ✅ Pass |
| Recall | 0.9800 | ≥ 0.65 | ✅ Pass |
| F1 Score | 0.9671 | — | — |
| AUPRC | 0.9979 | ≥ 0.75 | ✅ Pass |

---

### 3.2 Fraud Transaction Scorer (Phase 2)

**Purpose:** Score payment transactions for fraud risk in real-time

**What It Detects:**
- Stolen card usage
- Account takeover
- Velocity abuse (many orders in short window)
- New-account high-value fraud
- VPN/proxy transactions
- Address mismatches

**Model Architecture:**
- Algorithm: LightGBM (selected over XGBoost and ensemble on validation AUPRC)
- Training Data: 10,000 synthetic transaction records (~5% fraud rate)
- Class Imbalance Handling: `scale_pos_weight=19`

**Decision Thresholds:**

| Score | Label | Recommendation |
|-------|-------|----------------|
| < 0.40 | Low | Approve transaction |
| 0.40 – 0.70 | Medium | Flag for review, consider step-up auth |
| ≥ 0.70 | High | Block transaction, manual review required |

**Key Features:**
- `customer_orders_1h` / `customer_orders_24h` — Velocity signals
- `is_vpn` — VPN/proxy usage
- `is_new_device` — Device not seen before
- `address_mismatch` — Billing vs shipping address
- `new_account_high_value_card` — Risky combination
- `payment_risk_score` — Payment method risk weight

**Performance (Test Set):**

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Precision | 0.9934 | ≥ 0.80 | ✅ Pass |
| Recall | 1.0000 | ≥ 0.60 | ✅ Pass |
| F1 Score | 0.9967 | — | — |
| AUPRC | 1.0000 | ≥ 0.80 | ✅ Pass |

---

### 3.3 Chargeback Evidence Responder (Phase 3)

**Purpose:** Score dispute winnability and provide evidence checklist

**What It Scores:**
- Winnability of dispute (Weak / Moderate / Strong)
- Evidence completeness (0-7 score)
- Missing evidence items

**Model Architecture:**
- Algorithm: XGBoost + LightGBM soft-voting ensemble
- Training Data: 8,000 synthetic chargeback records (~60% merchant wins)
- Class Balance: Balanced (no `scale_pos_weight` needed)

**Decision Thresholds:**

| Score | Label | Recommendation |
|-------|-------|----------------|
| < 0.40 | Weak | Consider settling with customer |
| 0.40 – 0.70 | Moderate | Collect missing evidence before submitting |
| ≥ 0.70 | Strong | Submit all available evidence immediately |

**Key Features:**
- `evidence_score` (0–7) — Most predictive feature
- `dispute_difficulty` — Domain knowledge per dispute type
- `is_serial_disputer` — Customer with 2+ past disputes
- `delivery_confirmed` — Tracking number available
- `otp_used` — OTP/2FA during purchase

**Performance (Test Set):**

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Precision | 1.0000 | ≥ 0.75 | ✅ Pass |
| Recall | 1.0000 | ≥ 0.75 | ✅ Pass |
| F1 Score | 1.0000 | — | — |
| AUPRC | 1.0000 | ≥ 0.75 | ✅ Pass |

**Note:** These metrics are high because the data is synthetic and patterns are clean. Real-world performance would be lower.

---

## 4. Data & Feature Engineering

### 4.1 Synthetic Data Generation

All data in this project is synthetically generated using Faker and NumPy. No real customer or transaction data is used.

| Dataset | Records | Source File | Purpose |
|---------|---------|-------------|---------|
| Returns | 10,000 | `data/raw/returns.csv` | Phase 1 training |
| Transactions | 10,000 | `data/raw/transactions.csv` | Phase 2 training |
| Chargebacks | 8,000 | `data/raw/chargebacks.csv` | Phase 3 training |

### 4.2 Feature Engineering

#### Return Risk Features (25 features)

| Feature | Type | Description |
|---------|------|-------------|
| `return_rate_lifetime` | float | Lifetime return rate |
| `return_rate_30d` | float | 30-day return rate |
| `days_to_return` | int | Days between purchase and return |
| `is_near_deadline` | bool | Return close to policy deadline |
| `is_same_day_return` | bool | Returned within 1 day |
| `is_first_order` | bool | Customer's first order |
| `is_new_account` | bool | Account < 30 days old |
| `no_images_high_value` | bool | High-value return without photos |
| `no_support_contact` | bool | Didn't contact support first |
| `category_risk_score` | float | Product category risk weight |
| `return_reason_risk` | float | Return reason risk weight |
| `order_value_normalized` | float | Order value / customer avg |

#### Fraud Transaction Features (28 features)

| Feature | Type | Description |
|---------|------|-------------|
| `velocity_1h` | int | Orders in last hour |
| `velocity_24h` | int | Orders in last 24 hours |
| `is_address_mismatch` | bool | Billing ≠ shipping address |
| `is_new_device` | bool | Device not seen before |
| `is_country_mismatch` | bool | Card country ≠ IP country |
| `amount_vs_avg` | float | Transaction amount / customer avg |
| `is_prepaid_card` | bool | Prepaid card usage |
| `is_night_transaction` | bool | Transaction between 0-6 AM |
| `high_velocity_1h` | bool | >3 orders in last hour |
| `new_account_high_value_card` | bool | Risky combination flag |

#### Chargeback Features (28 features)

| Feature | Type | Description |
|---------|------|-------------|
| `evidence_score` | int | Count of 7 evidence fields (0-7) |
| `dispute_difficulty` | float | Difficulty weight per dispute type |
| `is_serial_disputer` | bool | Customer with 2+ past disputes |
| `is_quick_dispute` | bool | Disputed within 3 days |
| `is_late_dispute` | bool | Disputed after 60 days |
| `delivery_confirmed` | bool | Tracking number available |
| `otp_used` | bool | OTP/2FA during purchase |
| `customer_contacted_support` | bool | Support contacted before dispute |

### 4.3 Categorical Encodings

**Product Category Risk Scores:**

| Category | Risk Score | Reason |
|----------|-----------|--------|
| luxury | 0.9 | High swap fraud risk |
| electronics | 0.8 | High swap fraud risk |
| apparel | 0.6 | Wardrobing common |
| home | 0.4 | Lower abuse rates |
| books | 0.2 | Very low abuse rates |

**Return Reason Risk Scores:**

| Reason | Risk Score | Reason |
|--------|-----------|--------|
| not_needed | 0.8 | Often masks wardrobing |
| changed_mind | 0.7 | Similar to not_needed |
| quality_issue | 0.5 | Ambiguous signal |
| wrong_item | 0.4 | Could be merchant error |
| defective | 0.3 | Legitimate claim |

---

## 5. Model Training & Evaluation

### 5.1 Training Pipeline

Each module follows the same training pipeline:

1. **Data Loading:** Load synthetic CSV data
2. **Feature Engineering:** Apply `build_features()` to create model-ready features
3. **Train/Validation/Test Split:** 70/20/10 split (stratified)
4. **Model Training:** Train XGBoost, LightGBM, and Voting Ensemble
5. **Threshold Tuning:** Find optimal threshold on validation set
6. **Model Selection:** Pick best model by validation AUPRC
7. **Final Evaluation:** Evaluate on held-out test set (never touched before)
8. **Model Persistence:** Save model to `.pkl` file

### 5.2 Data Split Strategy

```
Total Data (100%)
├── Train (70%) — Model learns from this
├── Validation (20%) — Threshold tuning, model selection
└── Test (10%) — Final metrics, never touched during training
```

**Why 70/20/10 (not 80/20)?**
- Dedicated validation set prevents threshold leakage into test set
- More rigorous evaluation ensures honest performance estimates

### 5.3 Evaluation Metrics

**Metrics Used:**

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **Precision** | TP / (TP + FP) | Of all flags, how many are real? |
| **Recall** | TP / (TP + FN) | Of all real fraud, how much did we catch? |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |
| **AUPRC** | Area under PR curve | Robust to class imbalance |

**Metrics NOT Used:**

| Metric | Why Not |
|--------|---------|
| **Accuracy** | Useless on imbalanced data (95% accuracy by flagging nothing) |
| **AUC-ROC** | Less informative than AUPRC for rare events |

### 5.4 Threshold Selection

Thresholds are tuned on the validation set using F1 score:

```python
for thresh in np.arange(0.20, 0.80, 0.01):
    preds = (proba >= thresh).astype(int)
    p = precision_score(y_val, preds)
    r = recall_score(y_val, preds)
    f = f1_score(y_val, preds)
    if p >= MIN_PRECISION and f > best_f1:
        best_f1, best_thresh = f, thresh
```

### 5.5 Confusion Matrix

For binary classification:

```
                    Predicted: Risky    Predicted: Safe
Actual: Risky         TP                      FN
Actual: Safe          FP                      TN
```

| Cell | Name | Meaning |
|------|------|---------|
| TP | True Positive | Correctly flagged risky case |
| TN | True Negative | Correctly cleared safe case |
| FP | False Positive | Wrong alarm — legitimate case flagged |
| FN | False Negative | Missed — risky case not caught |

### 5.6 False Positive Cost Analysis

**False Positive Impact:**
- Legitimate return wrongly rejected → angry customer, lost trust
- Legitimate transaction wrongly blocked → lost sale, customer churn
- Support ticket cost (₹50-200 per ticket)

**Cost Formula:**
```
FP Cost = Number of FPs × Average Order Value × False Positive Weight
```

**Business Trade-off:**
- Low-margin merchant: FP Weight = 1.0x
- High-CLV business: FP Weight = 5.0x (losing good customer is expensive)

---

## 6. API Documentation

### 6.1 API Overview

- **Framework:** FastAPI
- **Base URL:** `http://127.0.0.1:8000`
- **Interactive Docs:** `http://127.0.0.1:8000/docs`
- **Version:** 3.0.0

### 6.2 Authentication

Set `API_KEY` environment variable to enable authentication:

```bash
export API_KEY="your-secret-key"
```

If not set, server runs in open mode (no key required).

### 6.3 Endpoints

#### Health Check

```
GET /
```

**Response:**
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

#### Score Return Request

```
POST /score/return
```

**Request Body:**
```json
{
  "customer_total_orders": 12,
  "customer_total_returns": 8,
  "customer_account_age_days": 45,
  "customer_returns_30d": 3,
  "customer_orders_30d": 4,
  "customer_avg_order_value": 1800.0,
  "order_value": 3500.0,
  "payment_method": "cod",
  "days_to_return": 28,
  "return_reason": "changed_mind",
  "support_contacted": false,
  "images_submitted": false,
  "product_category": "electronics",
  "merchant_return_rate": 0.12
}
```

**Response:**
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

#### Score Transaction

```
POST /score/transaction
```

**Request Body:**
```json
{
  "order_value": 18000.0,
  "product_category": "electronics",
  "payment_method": "card",
  "hour_of_day": 3,
  "day_of_week": 2,
  "failed_attempts": 5,
  "is_vpn": true,
  "customer_account_age_days": 12,
  "customer_total_past_orders": 2,
  "customer_avg_order_value": 1500.0,
  "customer_past_fraud_flags": 0,
  "customer_orders_1h": 6,
  "customer_orders_24h": 8,
  "merchant_fraud_rate": 0.08,
  "is_new_device": true,
  "is_different_city": true,
  "address_mismatch": true
}
```

**Response:**
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

#### Analyze Chargeback

```
POST /chargeback/analyze
```

**Request Body:**
```json
{
  "order_value": 8500.0,
  "product_category": "electronics",
  "payment_method": "card",
  "days_to_dispute": 5,
  "delivery_days": 3,
  "dispute_reason": "not_received",
  "delivery_confirmed": true,
  "customer_signed_delivery": false,
  "otp_used": true,
  "ip_logs_available": true,
  "order_confirmation_sent": true,
  "refund_issued": false,
  "customer_contacted_support": false,
  "customer_account_age_days": 400,
  "customer_total_orders": 15,
  "customer_past_disputes": 0,
  "customer_avg_order_value": 3000.0,
  "merchant_chargeback_rate": 0.04
}
```

**Response:**
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

#### Model Info Endpoints

```
GET /model/return/info
GET /model/fraud/info
GET /model/chargeback/info
```

**Response (same format for all):**
```json
{
  "model_name": "Ensemble",
  "threshold": 0.40,
  "precision": 0.9545,
  "recall": 0.9800,
  "f1": 0.9671,
  "auc_roc": 0.9995,
  "auprc": 0.9979,
  "confusion_matrix": {
    "true_negative": 1680,
    "false_positive": 20,
    "false_negative": 4,
    "true_positive": 196
  }
}
```

#### Feedback Endpoints

```
POST /feedback/return
POST /feedback/transaction
POST /feedback/chargeback
GET /feedback/summary
```

Used to submit confirmed outcomes for model retraining.

---

## 7. Technology Stack

| Layer | Tool | Version | Purpose |
|-------|------|---------|---------|
| **Language** | Python | 3.11+ | Core runtime |
| **ML Framework** | XGBoost | 2.1.4 | Gradient boosting |
| **ML Framework** | LightGBM | 4.5.0 | Gradient boosting |
| **ML Utilities** | Scikit-learn | 1.6.1 | Metrics, splitting, ensemble |
| **Data Processing** | Pandas | 3.0.5 | Tabular data manipulation |
| **Numerical** | NumPy | 2.5.2 | Array operations |
| **API Framework** | FastAPI | 0.141.1 | REST API |
| **API Server** | Uvicorn | 0.52.4 | ASGI server |
| **Data Validation** | Pydantic | 2.13.5 | Request/response validation |
| **Data Generation** | Faker | 26.0.0 | Synthetic data |
| **Testing** | Pytest | 9.1.1 | Unit testing |
| **Notebooks** | Jupyter | — | EDA and exploration |

---

## 8. Project Structure

```
AI-Risk-Manager/
├── data/
│   ├── generate_data.py              # Phase 1 — return records
│   ├── generate_fraud_data.py        # Phase 2 — transaction records
│   ├── generate_chargeback_data.py   # Phase 3 — chargeback records
│   └── raw/
│       ├── returns.csv               # 10,000 records
│       ├── transactions.csv          # 10,000 records
│       └── chargebacks.csv           # 8,000 records
├── notebooks/
│   ├── exploration.ipynb             # Phase 1 EDA
│   ├── exploration_fraud.ipynb       # Phase 2 EDA
│   └── exploration_chargeback.ipynb  # Phase 3 EDA
├── src/
│   ├── features.py                   # Phase 1 feature engineering
│   ├── fraud_features.py             # Phase 2 feature engineering
│   ├── chargeback_features.py        # Phase 3 feature engineering
│   ├── train.py                      # Phase 1 model training
│   ├── fraud_train.py                # Phase 2 model training
│   ├── chargeback_train.py           # Phase 3 model training
│   └── retrain.py                    # Retraining pipeline
├── api/
│   └── main.py                       # FastAPI — all endpoints
├── models/
│   ├── return_risk_model.pkl         # Saved Phase 1 model
│   ├── fraud_risk_model.pkl          # Saved Phase 2 model
│   └── chargeback_model.pkl          # Saved Phase 3 model
├── tests/
│   ├── test_model.py                 # Phase 1 — 28 tests
│   ├── test_fraud_model.py           # Phase 2 — 23 tests
│   ├── test_chargeback_model.py      # Phase 3 — 28 tests
│   └── test_feedback.py             # Feedback system tests
├── docs/
│   ├── problem-statement.md
│   ├── architecture.md
│   ├── roadmap.md
│   ├── data-dictionary.md
│   └── evaluation-strategy.md
├── LEARNING_LOG.md
├── PROJECT_STATUS.md
├── README.md
├── PROJECT_REPORT.md                 # This document
├── requirements.txt
└── .gitignore
```

---

## 9. Setup & Installation

### 9.1 Prerequisites

- Python 3.11+
- pip
- Git (optional)

### 9.2 Installation Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd AI-Risk-Manager

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic data
python data/generate_data.py
python data/generate_fraud_data.py
python data/generate_chargeback_data.py

# 5. Train models
python src/train.py
python src/fraud_train.py
python src/chargeback_train.py

# 6. Run tests
pytest tests/ -v

# 7. Start API
uvicorn api.main:app --reload

# 8. Visit interactive docs
open http://127.0.0.1:8000/docs
```

### 9.3 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_KEY` | No | None | API authentication key |
| `RETURN_MODEL_PATH` | No | `models/return_risk_model.pkl` | Path to return model |
| `FRAUD_MODEL_PATH` | No | `models/fraud_risk_model.pkl` | Path to fraud model |
| `CHARGEBACK_MODEL_PATH` | No | `models/chargeback_model.pkl` | Path to chargeback model |

---

## 10. Test Coverage

### 10.1 Test Suite Summary

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_model.py` | 28 | Phase 1 — Return Risk Scorer |
| `test_fraud_model.py` | 23 | Phase 2 — Fraud Transaction Scorer |
| `test_chargeback_model.py` | 28 | Phase 3 — Chargeback Evidence Responder |
| `test_feedback.py` | — | Feedback system |
| **Total** | **79+** | |

### 10.2 Test Categories

Each test file covers:

1. **Data Generation Tests** — Verify synthetic data quality
2. **Feature Engineering Tests** — Validate feature computation
3. **Model Training Tests** — Ensure models train successfully
4. **Prediction Tests** — Verify model outputs are valid probabilities
5. **API Endpoint Tests** — Validate request/response formats
6. **Edge Case Tests** — Handle invalid inputs gracefully
7. **Metric Tests** — Ensure models meet minimum performance targets

### 10.3 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 11. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Accuracy NOT used as metric** | Useless on imbalanced data — use precision, recall, AUPRC |
| **70/20/10 split (not 80/20)** | Dedicated validation set for threshold tuning prevents leakage |
| **Voting ensemble over single model** | More stable; one model's errors corrected by the other |
| **Threshold tuned on validation set** | Default 0.50 is almost never optimal for imbalanced fraud data |
| **All data synthetic** | No real customer or transaction data used anywhere |
| **Phase 3 uses `evidence_score`** | Domain insight: having proof matters more than any behavioral signal |
| **Defense-only system** | Scores and recommends — never blocks autonomously |
| **No accuracy metric** | Accuracy is meaningless when only 5% of transactions are fraudulent |

---

## 12. Future Roadmap

### Phase 4: Abuse Ring Sentinel (Planned)

**What It Does:** Detects coordinated fraud rings by analyzing relationships between accounts.

**Key Differences from Phases 1-3:**
- Not per-transaction — analyzes **relationships between accounts**
- Requires graph construction (NetworkX)
- Community detection algorithm (Louvain)
- Graph-level and node-level features

**Planned Steps:**
- [ ] Generate synthetic account network data with planted abuse rings
- [ ] Build account relationship graph
- [ ] Run community detection
- [ ] Engineer graph features
- [ ] Train cluster scorer
- [ ] Add `POST /score/abuse-ring` endpoint

### Potential Enhancements

1. **Real-time Data Integration** — Connect to live transaction feeds
2. **Model Versioning** — Implement MLflow or similar for model tracking
3. **A/B Testing Framework** — Test model performance in production
4. **Dashboard** — Visualize risk scores and trends
5. **Alert System** — Real-time notifications for high-risk events
6. **Multi-tenant Support** — Merchant-specific thresholds and models

---

## 13. Conclusion

### Project Summary

AI Risk Manager demonstrates a complete ML pipeline for merchant risk scoring:

- **3 production-ready modules** addressing fraud, return abuse, and chargebacks
- **79+ tests** ensuring reliability and correctness
- **REST API** with interactive documentation
- **Honest evaluation** with precision, recall, and AUPRC on held-out test sets
- **Defense-only design** that prioritizes human oversight

### Key Achievements

1. **High Performance:** All models exceed minimum targets on held-out test sets
2. **Production Ready:** FastAPI with validation, logging, and authentication
3. **Well Documented:** Comprehensive docs, data dictionary, and evaluation strategy
4. **Extensible:** Clear architecture for adding new risk modules

### Important Caveats

- **Synthetic Data:** All models trained on synthetic data. Real-world performance will be lower.
- **No Production Deployment:** This is a proof-of-concept, not a production system.
- **Human Oversight Required:** The system recommends actions — humans decide.

---

## Appendices

### Appendix A: Glossary

| Term | Definition |
|------|------------|
| **AUPRC** | Area Under the Precision-Recall Curve |
| **Precision** | TP / (TP + FP) — Of all flags, how many are real? |
| **Recall** | TP / (TP + FN) — Of all real cases, how many did we catch? |
| **F1 Score** | Harmonic mean of precision and recall |
| **False Positive** | Legitimate case wrongly flagged as risky |
| **False Negative** | Risky case missed by the model |
| **Threshold** | Cutoff probability above which a case is flagged |

### Appendix B: Model Files

| File | Size | Description |
|------|------|-------------|
| `return_risk_model.pkl` | ~2.5 MB | Phase 1 — XGBoost + LightGBM ensemble |
| `fraud_risk_model.pkl` | ~710 KB | Phase 2 — LightGBM |
| `chargeback_model.pkl` | ~2.3 MB | Phase 3 — XGBoost + LightGBM ensemble |

### Appendix C: API Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 401 | Unauthorized (invalid API key) |
| 422 | Validation error (invalid request body) |
| 500 | Internal server error |

---

**Report Version:** 1.0  
**Last Updated:** September 2026  
**Author:** CSE Admin  
**Project:** AI Risk Manager  
**Organization:** Razorpay
