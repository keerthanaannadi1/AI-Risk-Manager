# AI Risk Manager — Project Status

## Last Updated
2026-08-29

## Project Location
/home/cse-admin/Documents/razorpay/AI-Risk-Manager

## What This Project Is
ML-powered risk scoring system for merchants on payment platforms.
Detects fraud, return abuse, and chargebacks before the merchant loses money.
Defense-only system — scores risk and recommends actions, never blocks or penalizes.

---

## Current Status: DOCUMENTATION COMPLETE, CODING NOT STARTED

### What Is Done ✅
- `README.md` — full project overview
- `docs/problem-statement.md` — all 4 problems explained in detail
- `docs/architecture.md` — system design, data flow, API design, JSON formats
- `docs/roadmap.md` — step-by-step build plan with checkboxes
- `docs/data-dictionary.md` — every field and engineered feature defined
- `docs/evaluation-strategy.md` — precision, recall, false-positive cost explained

### What Is NOT Done Yet ❌ (no code written at all)
- `data/` folder — empty, `generate_data.py` not written
- `src/` folder — empty, no `features.py`, `train.py`, `predict.py`
- `api/` folder — empty, no `main.py`
- `models/` folder — empty, no trained model
- `tests/` folder — empty, no `test_model.py`
- `notebooks/` folder — empty, no `exploration.ipynb`

---

## Build Order (agreed)

Build ONE module completely before moving to the next.

### Phase 1: Return Risk Scorer ← START HERE
**Status: NOT STARTED**

| Step | Task | Status |
|------|------|--------|
| 1.1 | Write `data/generate_data.py` — generate ~10,000 synthetic return records | ❌ |
| 1.2 | EDA in `notebooks/exploration.ipynb` — class balance, distributions, correlations | ❌ |
| 1.3 | Write `src/features.py` — feature engineering (return rates, days_to_return, category risk) | ❌ |
| 1.4 | Write `src/train.py` — XGBoost classifier, 80/20 split, save model | ❌ |
| 1.5 | Evaluate — precision-recall curve, confusion matrix, false positive cost, threshold | ❌ |
| 1.6 | Write `api/main.py` — FastAPI with `POST /score/return` endpoint | ❌ |
| 1.7 | Write `tests/test_model.py` — assert precision >= 0.70, recall >= 0.65 | ❌ |

### Phase 2: Fraud Transaction Scorer
**Status: PLANNED — start after Phase 1 is fully done**

### Phase 3: Chargeback Evidence Responder
**Status: PLANNED — start after Phase 2 is fully done**

### Phase 4: Abuse Ring Sentinel
**Status: PLANNED — start after Phase 3 is fully done**

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

## Key Decisions Made
- Return Risk Scorer first — most intuitive features, cleanest data structure, best for learning
- Accuracy is NOT used as a metric — use precision, recall, AUPRC, and false positive cost
- Every module must have a model card with honest metrics before it is considered done
- Minimum targets for Phase 1: Precision ≥ 0.70, Recall ≥ 0.65, AUPRC ≥ 0.75
- All data is synthetic — no real customer or transaction data

---

## NEXT ACTION
**Step 1.1 — Write `data/generate_data.py`**

This script will:
- Generate ~10,000 synthetic return records using Faker and NumPy
- Include both normal returns and abusive return patterns
- Save the data to `data/raw/returns.csv`

Tell Kiro: *"Let's start Step 1.1 — write the data generation script"*
