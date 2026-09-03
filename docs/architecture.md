# Architecture

## System Overview

AI Risk Manager is a modular, ML-powered risk scoring system. Each module is independently deployable and follows the same core pattern: receive an event, compute features, run the model, return a risk score with an explanation.

---

## High-Level Data Flow

```
┌─────────────────────────────────────────────────────┐
│                  Incoming Event                      │
│   (Return request / Transaction / Chargeback)        │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│               Feature Engineering                    │
│  Convert raw event + customer history into signals   │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                  ML Model                            │
│     XGBoost classifier → risk probability (0–1)     │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              Threshold + Decision Layer              │
│   Low (<0.3) / Medium (0.3–0.7) / High (>0.7)      │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                  API Response                        │
│   { risk_score, risk_level, recommended_action,     │
│     top_features }                                   │
└─────────────────────────────────────────────────────┘
```

---

## Module Architecture

Each module follows the same internal structure:

```
Module (e.g. Return Risk Scorer)
├── Data Layer          → raw order + customer data
├── Feature Layer       → engineered signals
├── Model Layer         → trained XGBoost model
├── Decision Layer      → threshold → risk level
└── API Layer           → FastAPI endpoint
```

---

## Module 1: Return Risk Scorer

### Input
```json
{
  "customer_id": "C123",
  "order_id": "O456",
  "return_reason": "defective",
  "days_since_purchase": 25,
  "product_category": "electronics"
}
```

### Feature Engineering
The raw input is enriched with customer history pulled from the database:

| Feature | Description |
|---------|-------------|
| `return_rate_30d` | Customer returns / orders in last 30 days |
| `return_rate_lifetime` | Customer returns / total orders ever |
| `days_to_return` | Days between purchase and return request |
| `category_risk_score` | Risk weight of product category |
| `account_age_days` | Days since customer account was created |
| `order_value` | Value of the order being returned |
| `is_first_order` | Whether this is the customer's first order |
| `support_contacted` | Whether customer contacted support before returning |

### Model
- Algorithm: XGBoost binary classifier
- Output: probability of return being abusive (0.0 to 1.0)
- Saved to: `models/return_risk_model.pkl`

### Decision Thresholds
| Score Range | Risk Level | Action |
|-------------|-----------|--------|
| 0.0 – 0.30 | Low | Auto-approve |
| 0.30 – 0.70 | Medium | Flag for review |
| 0.70 – 1.0 | High | Require photo proof / reject |

### Output
```json
{
  "risk_score": 0.82,
  "risk_level": "HIGH",
  "recommended_action": "Request photo evidence before approving",
  "top_features": [
    {"feature": "return_rate_lifetime", "value": 0.78, "contribution": "high"},
    {"feature": "days_to_return", "value": 29, "contribution": "medium"},
    {"feature": "category_risk_score", "value": 0.9, "contribution": "high"}
  ]
}
```

---

## Module 2: Fraud Transaction Scorer (Planned)

### Input
```json
{
  "transaction_id": "T789",
  "customer_id": "C123",
  "amount": 15000,
  "payment_method": "card",
  "device_fingerprint": "abc123",
  "ip_address": "103.21.x.x",
  "shipping_address": "...",
  "billing_address": "..."
}
```

### Key Features
- Transaction velocity (orders in last 1h / 24h)
- Device seen before (yes/no)
- IP risk score
- Address match (billing vs shipping)
- Order value vs customer average
- Account age at time of transaction
- Payment method risk (prepaid, international)

### Model
- XGBoost classifier
- Trained on imbalanced data (fraud is rare — ~0.5% of transactions)
- Uses SMOTE or class weights to handle imbalance

---

## Module 3: Chargeback Evidence Responder (Planned)

### Input
- Chargeback notification from payment gateway
- Order ID to look up

### Process
1. Pull all evidence records for the order (delivery logs, IP, device, login history, communication)
2. Score winnability of the dispute
3. Auto-generate structured evidence document

### Output
- Win probability score (0.0 to 1.0)
- Evidence package (structured JSON + formatted PDF)
- Recommended action: fight / concede / escalate

---

## Module 4: Abuse Ring Sentinel (Planned)

### Input
- Stream of account events (registrations, logins, transactions)

### Process
1. Build account relationship graph (shared device, IP, phone, email domain, card BIN)
2. Run community detection algorithm (Louvain or connected components)
3. Score each cluster for coordinated abuse signals

### Output
- Cluster ID + accounts in cluster
- Abuse probability for the cluster
- Shared signals that triggered the alert

---

## API Design

All modules expose REST endpoints via FastAPI.

### Base URL
```
http://localhost:8000
```

### Endpoints

| Method | Path | Module |
|--------|------|--------|
| POST | `/score/return` | Return Risk Scorer |
| POST | `/score/transaction` | Fraud Transaction Scorer |
| POST | `/score/chargeback` | Chargeback Responder |
| GET | `/health` | Health check |

### Health Check Response
```json
{
  "status": "ok",
  "models_loaded": ["return_risk_model"],
  "version": "0.1.0"
}
```

---

## Technology Choices and Why

| Component | Choice | Why |
|-----------|--------|-----|
| ML algorithm | XGBoost | Best performance on tabular data; handles missing values; fast inference |
| API framework | FastAPI | Async, fast, built-in validation with Pydantic, auto-generates API docs |
| Data generation | Faker + NumPy | Realistic synthetic data without real PII |
| Data processing | Pandas | Industry standard for tabular data manipulation |
| Model serialization | joblib/pickle | Simple, fast, works with scikit-learn and XGBoost |
| Testing | Pytest | Standard Python testing framework |

---

## Project Folder to Module Mapping

```
data/generate_data.py     → generates synthetic training data for all modules
src/features.py           → feature engineering (shared utilities)
src/train.py              → trains and saves model to models/
src/predict.py            → loads model and scores a single input
api/main.py               → FastAPI app with all endpoints
tests/test_model.py       → precision/recall tests on held-out test set
notebooks/exploration.ipynb → EDA, feature importance, threshold analysis
```

---

## Deployment (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python data/generate_data.py

# Train the model
python src/train.py

# Start the API
uvicorn api.main:app --reload

# Run tests
pytest tests/
```
