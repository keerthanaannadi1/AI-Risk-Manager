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
| Return Risk Scorer | Scores a return request as Low / Medium / High risk | 🔨 In Progress |
| Fraud Transaction Scorer | Flags suspicious transactions in real time | 📋 Planned |
| Chargeback Evidence Responder | Auto-collects and formats dispute evidence | 📋 Planned |
| Abuse Ring Sentinel | Detects coordinated multi-account abuse | 📋 Planned |

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.11+ |
| Data generation | Faker, NumPy |
| Data processing | Pandas |
| ML model | XGBoost, Scikit-learn |
| API | FastAPI |
| Testing | Pytest |
| Notebooks | Jupyter |

---

## Project Structure

```
AI-Risk-Manager/
├── data/
│   ├── generate_data.py          # Synthetic dataset generation
│   └── raw/                      # Generated CSV files
├── notebooks/
│   └── exploration.ipynb         # EDA and feature analysis
├── src/
│   ├── features.py               # Feature engineering
│   ├── train.py                  # Model training
│   └── predict.py                # Scoring logic
├── api/
│   └── main.py                   # FastAPI endpoints
├── models/
│   └── return_risk_model.pkl     # Saved trained model
├── tests/
│   └── test_model.py             # Precision/recall tests
├── docs/
│   ├── problem-statement.md      # Detailed problem breakdown
│   ├── architecture.md           # System design and data flow
│   ├── roadmap.md                # Phased build plan
│   ├── data-dictionary.md        # All fields and features explained
│   └── evaluation-strategy.md   # How models are measured
└── README.md
```

---

## Evaluation Philosophy

Every model in this project is evaluated with:

- **Precision** — of all flagged cases, how many are actually risky?
- **Recall** — of all actually risky cases, how many did we catch?
- **False positive cost** — every legitimate transaction wrongly blocked is a real customer hurt
- **Threshold analysis** — what precision/recall tradeoff works best for the business?

No model ships without these numbers documented.

---

## Documentation

- [Problem Statement](docs/problem-statement.md)
- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [Data Dictionary](docs/data-dictionary.md)
- [Evaluation Strategy](docs/evaluation-strategy.md)

---

## Who This Is For

- Developers learning applied ML in the fintech/risk domain
- Engineers building fraud or returns infrastructure at payment companies
- Anyone interested in honest, measured ML systems — not just accuracy numbers

---

## Disclaimer

All data used in this project is **synthetically generated**. No real customer, transaction, or merchant data is used anywhere.
