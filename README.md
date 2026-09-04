# AI Risk Manager

> Stop merchants from losing money to fraud, returns, and chargebacks using ML-powered risk scoring.

---

## What This Project Does

Merchants on payment platforms like Razorpay lose money in three silent ways:

- **Fraud** — stolen cards used to place orders; merchant ships the product, then loses the money
- **Return abuse** — customers exploit return policies (wardrobing, swap fraud, serial returners)
- **Chargebacks** — customers dispute transactions with their bank; merchant loses money + pays a fee

This project builds ML-based detectors for each of these loss types, with honest evaluation using precision, recall, and false-positive cost on a held-out test set.

---

## Scope

This is a **defense-only** system. It scores risk and recommends actions. It does not block, penalize, or take any offensive action against any party.

---

## Modules

| Module | What It Does | Status |
|--------|-------------|--------|
| Return Risk Scorer | Scores a return request as Low / Medium / High risk | ✅ Complete |
| Fraud Transaction Scorer | Flags suspicious transactions in real time | ✅ Complete |
| Chargeback Evidence Responder | Scores dispute winnability + builds evidence checklist | ✅ Complete |

---

## Model Results (held-out test set)

| Module | Model | Precision | Recall | F1 | AUPRC |
|--------|-------|-----------|--------|----|-------|
| Return Risk Scorer | XGBoost + LightGBM Ensemble | 0.9545 | 0.9800 | 0.9671 | 0.9979 |
| Fraud Transaction Scorer | LightGBM | 0.9934 | 1.0000 | 0.9967 | 1.0000 |
| Chargeback Evidence Responder | XGBoost + LightGBM Ensemble | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

All metrics are on a held-out test set never seen during training. Thresholds are tuned on a separate validation set — not the test set.

> **Note:** These numbers are high because the data is synthetic and the patterns are clean. Real-world performance on production data would be lower. See [Evaluation Strategy](docs/evaluation-strategy.md) for methodology.

---

## API Endpoints

Start the API:

```bash
source venv/bin/activate
uvicorn api.main:app --reload
# Docs at: http://127.0.0.1:8000/docs
```

| Method | Endpoint | What It Does |
|--------|----------|-------------|
| GET | `/` | Health check — all three models loaded |
| GET | `/model/return/info` | Return model metrics and threshold |
| GET | `/model/fraud/info` | Fraud model metrics and threshold |
| GET | `/model/chargeback/info` | Chargeback model metrics and threshold |
| POST | `/score/return` | Score a return request |
| POST | `/score/transaction` | Score a payment transaction for fraud |
| POST | `/chargeback/analyze` | Analyze a chargeback dispute — winability + evidence checklist |

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.11+ |
| Data generation | Faker, NumPy |
| Data processing | Pandas |
| ML model | XGBoost, LightGBM, Scikit-learn |
| API | FastAPI |
| Testing | Pytest |
| Notebooks | Jupyter |

---

## Project Structure

```
AI-Risk-Manager/
├── data/
│   ├── generate_data.py                # Phase 1 — return records
│   ├── generate_fraud_data.py          # Phase 2 — transaction records
│   ├── generate_chargeback_data.py     # Phase 3 — chargeback records
│   └── raw/
│       ├── returns.csv                 # 10,000 records
│       ├── transactions.csv            # 10,000 records
│       └── chargebacks.csv             # 8,000 records
├── notebooks/
│   ├── exploration.ipynb               # Phase 1 EDA
│   ├── exploration_fraud.ipynb         # Phase 2 EDA
│   └── exploration_chargeback.ipynb    # Phase 3 EDA
├── src/
│   ├── features.py                     # Phase 1 feature engineering
│   ├── fraud_features.py               # Phase 2 feature engineering
│   ├── chargeback_features.py          # Phase 3 feature engineering
│   ├── train.py                        # Phase 1 model training
│   ├── fraud_train.py                  # Phase 2 model training
│   └── chargeback_train.py             # Phase 3 model training
├── api/
│   └── main.py                         # FastAPI — all 3 endpoints
├── models/
│   ├── return_risk_model.pkl           # Saved Phase 1 model
│   ├── fraud_risk_model.pkl            # Saved Phase 2 model
│   └── chargeback_model.pkl            # Saved Phase 3 model
├── tests/
│   ├── test_model.py                   # Phase 1 — 24 tests
│   ├── test_fraud_model.py             # Phase 2 — 25 tests
│   └── test_chargeback_model.py        # Phase 3 — 36 tests
├── docs/
│   ├── problem-statement.md
│   ├── architecture.md
│   ├── roadmap.md
│   ├── data-dictionary.md
│   └── evaluation-strategy.md
├── LEARNING_LOG.md                     # Session-by-session build log
└── README.md
```

---

## Running the Project

### 1. Setup

```bash
git clone <repo-url>
cd AI-Risk-Manager
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate data

```bash
python data/generate_data.py           # Phase 1
python data/generate_fraud_data.py     # Phase 2
python data/generate_chargeback_data.py  # Phase 3
```

### 3. Train models

```bash
python src/train.py                    # Phase 1
python src/fraud_train.py              # Phase 2
python src/chargeback_train.py         # Phase 3
```

### 4. Run tests

```bash
pytest tests/ -v
# Expected: 85 passed
```

### 5. Start the API

```bash
uvicorn api.main:app --reload
```

---

## Evaluation Philosophy

Every model in this project is evaluated with:

- **Precision** — of all flagged cases, how many are actually risky?
- **Recall** — of all actually risky cases, how many did we catch?
- **False positive cost** — every legitimate case wrongly flagged is a real customer hurt
- **Threshold analysis** — what precision/recall tradeoff works best for the business?
- **AUPRC** — area under the precision-recall curve; robust to class imbalance

Accuracy is never used as a metric. On imbalanced data (e.g. 5% fraud rate), a model that flags nothing achieves 95% accuracy and is completely useless.

No model ships without these numbers documented on a held-out test set.

---

## Documentation

- [Problem Statement](docs/problem-statement.md)
- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [Data Dictionary](docs/data-dictionary.md)
- [Evaluation Strategy](docs/evaluation-strategy.md)
- [Learning Log](LEARNING_LOG.md)

---

## Who This Is For

- Developers learning applied ML in the fintech/risk domain
- Engineers building fraud or returns infrastructure at payment companies
- Anyone interested in honest, measured ML systems — not just accuracy numbers

---

## Disclaimer

All data used in this project is **synthetically generated**. No real customer, transaction, or merchant data is used anywhere. Model performance numbers reflect synthetic data patterns and should not be taken as indicative of real-world production performance.
