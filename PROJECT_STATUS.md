# AI Risk Manager — Project Status

## Last Updated
2026-09-04

## Project Location
/home/cse-admin/Documents/razorpay/AI-Risk-Manager

## What This Project Is
ML-powered risk scoring system for merchants on payment platforms.
Detects fraud, return abuse, and chargebacks before the merchant loses money.
Defense-only system — scores risk and recommends actions, never blocks or penalizes.

---

## Current Status: PHASE 3 COMPLETE — 3 of 4 Modules Live

---

## Phase 1: Return Risk Scorer ✅ COMPLETE

| Step | Task | Status |
|------|------|--------|
| 1.1 | `data/generate_data.py` — 10,000 synthetic return records | ✅ |
| 1.2 | `notebooks/exploration.ipynb` — EDA, 11 sections | ✅ |
| 1.3 | `src/features.py` — feature engineering, 25 features | ✅ |
| 1.4 | `src/train.py` — XGBoost + LightGBM voting ensemble | ✅ |
| 1.5 | Evaluation — precision-recall curve, confusion matrix, threshold tuning | ✅ |
| 1.6 | `api/main.py` — `POST /score/return` endpoint | ✅ |
| 1.7 | `tests/test_model.py` — 28 tests passing | ✅ |

**Final test set metrics (Voting Ensemble, threshold=0.21):**

| Metric | Score | Target |
|--------|-------|--------|
| Precision | 0.9545 | ≥ 0.70 ✅ |
| Recall | 0.9800 | ≥ 0.65 ✅ |
| AUPRC | 0.9979 | ≥ 0.75 ✅ |

---

## Phase 2: Fraud Transaction Scorer ✅ COMPLETE

| Step | Task | Status |
|------|------|--------|
| 2.1 | `data/generate_fraud_data.py` — 10,000 transaction records (5% fraud) | ✅ |
| 2.2 | `notebooks/exploration_fraud.ipynb` — EDA | ✅ |
| 2.3 | `src/fraud_features.py` — 28 features including velocity and device signals | ✅ |
| 2.4 | `src/fraud_train.py` — XGBoost + LightGBM ensemble, `scale_pos_weight=19` | ✅ |
| 2.5 | `api/main.py` — `POST /score/transaction` endpoint | ✅ |
| 2.6 | `tests/test_fraud_model.py` — 23 tests passing | ✅ |

**What's different from Phase 1:**
- Severe class imbalance: 95% legitimate / 5% fraud (vs 85/15 in Phase 1)
- Velocity features: `customer_orders_1h`, `customer_orders_24h`
- Device and session signals: `is_vpn`, `is_new_device`, `address_mismatch`
- Interaction feature: `new_account_high_value_card`

---

## Phase 3: Chargeback Evidence Responder ✅ COMPLETE

| Step | Task | Status |
|------|------|--------|
| 3.1 | `data/generate_chargeback_data.py` — 8,000 dispute records (60% wins) | ✅ |
| 3.2 | `notebooks/exploration_chargeback.ipynb` — EDA | ✅ |
| 3.3 | `src/chargeback_features.py` — 28 features including evidence score | ✅ |
| 3.4 | `src/chargeback_train.py` — XGBoost + LightGBM ensemble, balanced classes | ✅ |
| 3.5 | `api/main.py` — `POST /chargeback/analyze` endpoint (v3.0.0) | ✅ |
| 3.6 | `tests/test_chargeback_model.py` — 28 tests passing | ✅ |

**What's different from Phases 1 and 2:**
- Not a risk score — a **winability score** (can the merchant win this dispute?)
- API response includes an **evidence checklist**: what the merchant has vs what's missing
- Balanced class split (60/40) — no `scale_pos_weight` needed
- `dispute_difficulty` encodes domain knowledge per dispute type
- `evidence_score` (0–7) is the single most predictive feature

**Final test set metrics:**

| Metric | Score | Target |
|--------|-------|--------|
| Precision | ≥ 0.75 | ≥ 0.75 ✅ |
| Recall | ≥ 0.75 | ≥ 0.75 ✅ |
| AUPRC | ≥ 0.75 | ≥ 0.75 ✅ |

---

## Phase 4: Abuse Ring Sentinel 📋 PLANNED

**Depends on:** Phase 3 complete ✅ — ready to start

**What's different from all previous phases:**
- Not per-transaction — analyzes **relationships between accounts**
- Requires graph construction (NetworkX)
- Community detection algorithm (Louvain)
- Graph-level and node-level features (not tabular)

**Steps planned:**
- [ ] Generate synthetic account network data with planted abuse rings
- [ ] Build account relationship graph
- [ ] Run community detection
- [ ] Engineer graph features
- [ ] Train cluster scorer
- [ ] Add `POST /score/abuse-ring` endpoint

---

## API — Current Endpoints (v3.0.0)

| Method | Path | Module | Status |
|--------|------|--------|--------|
| GET | `/` | Health check (all 3 models) | ✅ Live |
| GET | `/model/return/info` | Phase 1 metrics | ✅ Live |
| GET | `/model/fraud/info` | Phase 2 metrics | ✅ Live |
| GET | `/model/chargeback/info` | Phase 3 metrics | ✅ Live |
| POST | `/score/return` | Return Risk Scorer | ✅ Live |
| POST | `/score/transaction` | Fraud Transaction Scorer | ✅ Live |
| POST | `/chargeback/analyze` | Chargeback Responder | ✅ Live |

Start the API:
```bash
cd /home/cse-admin/Documents/razorpay/AI-Risk-Manager
source venv/bin/activate
uvicorn api.main:app --reload
```
Then visit: http://127.0.0.1:8000/docs

---

## Test Suite — Current Coverage

| File | Tests | Covers |
|------|-------|--------|
| `tests/test_model.py` | 28 | Phase 1 model + API |
| `tests/test_fraud_model.py` | 23 | Phase 2 model + API |
| `tests/test_chargeback_model.py` | 28 | Phase 3 model + API |
| **Total** | **79** | |

Run all tests:
```bash
cd /home/cse-admin/Documents/razorpay/AI-Risk-Manager
source venv/bin/activate
pytest tests/ -v
```

---

## Files — Complete Inventory

```
AI-Risk-Manager/
├── data/
│   ├── generate_data.py              ✅ Phase 1 data generator
│   ├── generate_fraud_data.py        ✅ Phase 2 data generator
│   ├── generate_chargeback_data.py   ✅ Phase 3 data generator
│   └── raw/
│       ├── returns.csv               ✅ 10,000 records
│       ├── transactions.csv          ✅ 10,000 records
│       └── chargebacks.csv           ✅ 8,000 records
├── notebooks/
│   ├── exploration.ipynb             ✅ Phase 1 EDA
│   ├── exploration_fraud.ipynb       ✅ Phase 2 EDA
│   └── exploration_chargeback.ipynb  ✅ Phase 3 EDA
├── src/
│   ├── features.py                   ✅ Phase 1 feature engineering
│   ├── fraud_features.py             ✅ Phase 2 feature engineering
│   ├── chargeback_features.py        ✅ Phase 3 feature engineering
│   ├── train.py                      ✅ Phase 1 model training
│   ├── fraud_train.py                ✅ Phase 2 model training
│   └── chargeback_train.py           ✅ Phase 3 model training
├── models/
│   ├── return_risk_model.pkl         ✅ 2.5 MB
│   ├── fraud_risk_model.pkl          ✅ 710 KB
│   └── chargeback_model.pkl          ✅ 2.3 MB
├── api/
│   └── main.py                       ✅ v3.0.0 — all 3 phases
├── tests/
│   ├── test_model.py                 ✅ 28 tests — Phase 1
│   ├── test_fraud_model.py           ✅ 23 tests — Phase 2
│   └── test_chargeback_model.py      ✅ 28 tests — Phase 3
├── docs/
│   ├── problem-statement.md          ✅
│   ├── architecture.md               ✅
│   ├── roadmap.md                    ✅
│   ├── data-dictionary.md            ✅
│   └── evaluation-strategy.md       ✅
├── LEARNING_LOG.md                   ✅ Sessions 1–3 documented
└── README.md                         ✅
```

---

## Key Decisions Made (All Phases)

| Decision | Reason |
|----------|--------|
| Accuracy NOT used as metric | Useless on imbalanced data — use precision, recall, AUPRC |
| 70/20/10 split (not 80/20) | Dedicated validation set for threshold tuning prevents leakage |
| Voting ensemble over single model | More stable; one model's errors corrected by the other |
| Threshold tuned on validation set | Default 0.50 is almost never optimal for imbalanced fraud data |
| All data synthetic | No real customer or transaction data used anywhere |
| Phase 3 uses `evidence_score` | Domain insight: having proof matters more than any behavioral signal |

---

## NEXT ACTION
**Phase 4 — Abuse Ring Sentinel**

Start with: `data/generate_ring_data.py`
- Generate ~500 accounts with planted abuse rings (shared devices, IPs, card BINs)
- Add ~5,000 normal accounts for contrast
- Save to `data/raw/account_network.csv`
