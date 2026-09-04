# AI Risk Manager — Project Learning Log

> A daily log of what was built, what decisions were made, and what terms were learned.
> Updated as the project progresses.

---

## Session 1 — 2026-09-03

### What We Built Today

#### Step 1.1 — Data Generation (`data/generate_data.py`)

**What was done:**
- Wrote a synthetic data generator that creates 10,000 return records
- The data simulates real-world return patterns on a payment platform like Razorpay
- 85% of records are legitimate returns, 15% are abusive

**Four abuse patterns simulated:**
| Pattern | How it works | Signal |
|---------|-------------|--------|
| Wardrobing | Customer buys apparel/luxury, wears it, returns within 1–4 days | Very fast return, no images |
| Swap Fraud | Customer returns a broken/fake item in place of the original | No images submitted, high-value item |
| Serial Returner | Customer returns 70–95% of everything they buy | High lifetime return rate |
| Refund Fishing | Claims item never arrived or was defective to get free refund | No support contact, no images, claims at deadline |

**Columns generated (33 total):**
- Raw fields: `order_value`, `product_category`, `return_reason`, `payment_method`, `support_contacted`, `images_submitted`, customer history fields
- Pre-computed features: `return_rate_lifetime`, `days_to_return`, `is_near_deadline`, `is_new_account`, `no_images_high_value`, etc.
- Label: `is_label_abusive` (target variable)
- Debug: `abuse_pattern` (for EDA only, not used in model)

**Fix made mid-session:**
- `is_new_account` was nearly always 0 (only 56/10,000 records) because account age minimum was too high
- Fixed: injected 150 abusive records with account age 1–30 days
- Changed threshold from 30 days → 90 days
- Result: 23.7% of abusive records now have new accounts vs 4.0% of legitimate — a 6x signal difference

---

#### Step 1.2 — EDA Notebook (`notebooks/exploration.ipynb`)

**What was done:**
- Created a Jupyter notebook with 11 analysis sections
- Installed: `jupyter`, `matplotlib`, `seaborn`, `ipykernel`

**What the notebook shows:**
1. Class balance confirmation (85/15 split)
2. Abuse pattern breakdown
3. Mean feature values: abusive vs legitimate
4. Boxplots for key numeric features
5. Binary feature rates (% of each group with flag = True)
6. Category and return reason breakdown
7. Numeric feature distributions with histograms
8. Correlation heatmap
9. Order value analysis
10. EDA summary with key numbers

**Key findings from EDA:**
- Abusive customers have a lifetime return rate of **0.686** vs **0.075** for legitimate — a 9x difference
- 90.5% of abusive records have no images submitted
- 86.5% of abusive records never contacted support
- `return_rate_lifetime` and `no_support_contact` are the strongest signals

---

#### Step 1.3 — Feature Engineering (`src/features.py`)

**What was done:**
- Wrote `build_features(df)` function that transforms raw CSV into model-ready (X, y)
- Drops 8 columns: IDs, date strings, `abuse_pattern` (leakage), and the target
- Encodes 3 text columns to numbers

**Categorical encodings:**
| Column | Values → Numbers |
|--------|----------------|
| `product_category` | books=0, home=1, apparel=2, electronics=3, luxury=4 |
| `return_reason` | defective=0, wrong_item=1, quality_issue=2, changed_mind=3, not_needed=4 |
| `payment_method` | cod=0, wallet=1, netbanking=2, upi=3, card=4 |

**Output:**
- X: 25 numeric feature columns
- y: 0 (legitimate) or 1 (abusive)
- Validates: no missing values, all columns numeric

---

#### Step 1.4 — Model Training (`src/train.py`)

**What was done:**
- Split data: 70% train / 20% validation / 10% test (all stratified)
- Trained 3 models: XGBoost, LightGBM, Voting Ensemble
- Tuned decision threshold on validation set (picked threshold that maximises F1 while keeping precision ≥ 0.70)
- Final evaluation on held-out test set
- Saved best model to `models/return_risk_model.pkl`

**Final test set results (Voting Ensemble, threshold=0.21):**
| Metric | Score | Target |
|--------|-------|--------|
| Precision | 0.9545 | ≥ 0.70 ✅ |
| Recall | 0.9800 | ≥ 0.65 ✅ |
| F1 | 0.9671 | — |
| AUC-ROC | 0.9996 | — |
| AUPRC | 0.9979 | ≥ 0.75 ✅ |

**Confusion matrix on test set:**
```
                  Predicted Legit   Predicted Abusive
Actual Legit           843               7  (false alarms)
Actual Abusive           3  (missed)   147  (caught)
```

---

### Decisions Made Today

| Decision | Reason |
|----------|--------|
| 70/20/10 split instead of 80/20 | Dedicated validation set for threshold tuning prevents data leakage |
| XGBoost + LightGBM voting ensemble | Consistently outperforms single models on tabular fraud data |
| `scale_pos_weight = 5.67` | Compensates for 85/15 class imbalance |
| `is_new_account` threshold = 90 days | 30 days was too strict — only 56/10,000 records triggered it |
| Threshold = 0.21 (not default 0.50) | Tuned on validation set to maximise F1 while keeping precision ≥ 0.70 |

---

## Glossary — Terms Learned

### Machine Learning Basics

**Binary Classification**
A problem where the model predicts one of two outcomes: 0 or 1, Yes or No, Legitimate or Abusive.

**Target Variable (Label)**
The column the model is trying to predict. In this project: `is_label_abusive`.

**Feature**
Any input column the model uses to make its prediction. Example: `return_rate_lifetime`, `order_value`.

**Feature Engineering**
The process of transforming raw data into better inputs for the model. Example: computing `return_rate_lifetime = total_returns / total_orders` from raw counts.

**Training Set**
The data the model learns from. (70% in this project = 7,000 records)

**Validation Set**
Data the model never trains on, used to tune settings like the decision threshold. (20% = 2,000 records)

**Test Set**
Completely held-out data, only touched once at the very end to get the final honest numbers. (10% = 1,000 records)

**Stratified Split**
Splitting data so each subset has the same class balance as the original. Example: all three splits have 15% abusive records.

---

### Model Evaluation

**Precision**
Of all the returns the model flagged as abusive, what fraction actually were abusive?
```
Precision = True Positives / (True Positives + False Positives)
```
High precision = few false alarms on legitimate customers.

**Recall (Sensitivity)**
Of all the actually abusive returns, what fraction did the model catch?
```
Recall = True Positives / (True Positives + False Negatives)
```
High recall = few abusers slipping through.

**F1 Score**
The balance between precision and recall. Useful when you care about both.
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**AUC-ROC**
Area Under the ROC Curve. Measures how well the model separates the two classes across all thresholds. Score of 1.0 = perfect, 0.5 = random guessing.

**AUPRC (Average Precision)**
Area Under the Precision-Recall Curve. More informative than AUC-ROC when classes are imbalanced. The key metric for fraud detection.

**Confusion Matrix**
A 2×2 table showing correct and incorrect predictions:
```
                  Predicted: Legit   Predicted: Abusive
Actual: Legit        TN (correct)      FP (false alarm)
Actual: Abusive      FN (missed)       TP (correct catch)
```
- TN = True Negative (legit, correctly cleared)
- FP = False Positive (legit, wrongly flagged)
- FN = False Negative (abusive, missed)
- TP = True Positive (abusive, correctly caught)

**False Positive**
A legitimate customer wrongly flagged as abusive. Real cost: customer frustration, delayed refund, lost trust.

**False Negative**
An abusive return that slipped through. Real cost: merchant loses money.

**Decision Threshold**
The probability cutoff above which the model says "abusive". Default is 0.50, but tuned to 0.21 in this project to balance precision and recall.

---

### Algorithms

**XGBoost (Extreme Gradient Boosting)**
A powerful tree-based ML algorithm. Builds many small decision trees one at a time, each one correcting the errors of the previous. Industry standard for tabular fraud detection data.

**LightGBM**
Similar to XGBoost but faster. Also tree-based gradient boosting. Often matches or beats XGBoost on the same data.

**Voting Ensemble**
Combining multiple models by averaging their probability outputs (soft voting). More stable than any single model because one model's mistakes get corrected by the others.

**scale_pos_weight**
An XGBoost parameter that tells the model to pay more attention to the minority class (abusive returns). Set to `negative_count / positive_count`. In this project: `8500 / 1500 = 5.67`.

**SMOTE (Synthetic Minority Oversampling Technique)**
A technique for handling class imbalance by creating synthetic samples of the minority class. An alternative to `scale_pos_weight`. Referenced in the code but not used — `scale_pos_weight` was sufficient.

**SHAP (SHapley Additive exPlanations)**
A method for explaining why a model made a specific prediction. It assigns an importance score to each feature for each individual prediction. Planned for the API step.

---

### Data Concepts

**Class Imbalance**
When one class is much rarer than the other. In this project: 85% legitimate vs 15% abusive. Models trained without handling this tend to just predict "legitimate" for everything.

**Data Leakage**
When information that wouldn't be available in production accidentally gets into the training data. Example: including `abuse_pattern` as a feature — the model would "cheat" by using the label to predict the label.

**Synthetic Data**
Artificially generated data that mimics real-world patterns. Used in this project because real transaction data is private and sensitive.

**EDA (Exploratory Data Analysis)**
Examining your data before modeling to understand distributions, spot problems, and confirm signals are present. The "check your ingredients before cooking" step.

**Label Encoding**
Converting text categories to integers. Example: `electronics → 3`. Used for `product_category`, `return_reason`, `payment_method`.

**Correlation**
A measure of how strongly two variables move together. Range: -1 to +1. High absolute value = strong relationship. Used in the heatmap to see which features are most related to the abuse label.

---

### Project Architecture

**FastAPI**
A Python web framework for building APIs. Will be used in Step 1.6 to serve the model as a REST endpoint.

**Pickle (.pkl)**
Python's format for saving any object (including trained models) to disk so they can be loaded later without retraining.

**Jupyter Notebook (.ipynb)**
An interactive file format where you write Python code in cells and see output (charts, tables) right below each cell. Used for EDA.

**venv (Virtual Environment)**
An isolated Python environment for a project so installed packages don't conflict with other projects on the same machine.

---

## Current Status

| Step | Task | Status |
|------|------|--------|
| 1.1 | Data generation | ✅ Done |
| 1.2 | EDA notebook | ✅ Done |
| 1.3 | Feature engineering | ✅ Done |
| 1.4 | Model training | ✅ Done |
| 1.5 | Evaluation (curves + cost analysis) | 🔲 Next |
| 1.6 | FastAPI endpoint | 🔲 Pending |
| 1.7 | Tests | 🔲 Pending |
| Phase 2 | Fraud Transaction Scorer | 📋 Planned |
| Phase 3 | Chargeback Evidence Responder | 📋 Planned |
| Phase 4 | Abuse Ring Sentinel | 📋 Planned |

---

## Files Created So Far

```
AI-Risk-Manager/
├── data/
│   ├── generate_data.py      ✅ Synthetic data generator (448 lines)
│   └── raw/
│       └── returns.csv       ✅ 10,000 records, 33 columns
├── notebooks/
│   └── exploration.ipynb     ✅ EDA notebook (11 sections)
├── src/
│   ├── features.py           ✅ Feature engineering (242 lines)
│   └── train.py              ✅ Model training (282 lines)
├── models/
│   └── return_risk_model.pkl ✅ Saved Voting Ensemble model
├── docs/                     ✅ All documentation (pre-existing)
└── PROJECT_STATUS.md         (to be updated)
```

---

*Log maintained by Kiro. Updated at end of each session.*

---

## Session 2 — 2026-09-04

### What We Built Today

#### Step 2.1 — Fraud Data Generation (`data/generate_fraud_data.py`)

**What was done:**
- Wrote a synthetic data generator that creates ~10,000 payment transaction records
- 95% legitimate transactions, 5% fraudulent (realistic real-world ratio)
- Uses Faker with Indian locale (`en_IN`) for realistic names, addresses, and phone numbers

**Five fraud patterns simulated:**
| Pattern | Signal |
|---------|--------|
| Stolen card | New account, card payment, high value, late night, VPN |
| Account takeover | New device, different city, address mismatch |
| Velocity abuse | Many orders in 1 hour (high_velocity_1h) |
| Card testing | Multiple failed attempts before success |
| First-party fraud | New account + high value + card = new_account_high_value_card |

---

#### Step 2.2 — EDA Notebook (`notebooks/exploration_fraud.ipynb`)

**Key findings:**
- Fraudulent transactions have a mean order value of ~₹16,000 vs ~₹2,800 for legitimate
- 89% of fraud happens via card; legitimate transactions are spread across UPI, netbanking, COD
- `high_velocity_1h`, `multiple_failed_attempts`, and `new_account_high_value_card` are the strongest binary signals

---

#### Step 2.3 — Feature Engineering (`src/fraud_features.py`)

- 28 features including velocity signals (`customer_orders_1h`, `customer_orders_24h`), device signals (`is_new_device`, `is_vpn`), and interaction features (`new_account_high_value_card`)
- Same pattern as Phase 1: explicit `FEATURE_COLUMNS` list, categorical encoding dictionaries, `build_features()` function

---

#### Step 2.4 — Model Training (`src/fraud_train.py`)

**Split:** 70% train / 20% validation / 10% test (stratified, `random_state=42`)

**Class imbalance handling:** `scale_pos_weight = 19` (95% / 5% = ~19x more negatives)

**Final test set results (best model):**

| Metric | Score | Target |
|--------|-------|--------|
| Precision | > 0.80 | ≥ 0.80 ✅ |
| Recall | > 0.70 | ≥ 0.70 ✅ |
| AUPRC | > 0.75 | ≥ 0.75 ✅ |

---

#### Step 2.5 — API Endpoint (`api/main.py` updated)

- Added `POST /score/transaction` endpoint for fraud scoring
- Added `GET /model/fraud/info` endpoint
- Health check at `GET /` now reports both return and fraud models loaded

---

#### Step 2.6 — Tests (`tests/test_fraud_model.py`)

- 23 tests covering model loading, metrics, API health, scoring endpoint, and input validation
- High-risk payload: 3am, new device, new account, 5 failed attempts, VPN, high-value electronics
- Low-risk payload: 2pm, old account, known device, books, UPI, zero failures

---

### Decisions Made in Session 2

| Decision | Reason |
|----------|--------|
| `scale_pos_weight = 19` | 95/5 imbalance needs aggressive upweighting |
| `gift_cards` added to product categories | Common fraud vector, missing from Phase 1 |
| `high_velocity_1h` as interaction feature | Single strongest fraud signal in EDA |

---

## Session 3 — 2026-09-04

### What We Built Today

#### Step 3.1 — Chargeback Data Generation (`data/generate_chargeback_data.py`)

**What was done:**
- Wrote a synthetic data generator creating 8,000 chargeback dispute records
- 60% merchant wins / 40% merchant losses (deliberately balanced — no `scale_pos_weight` needed)
- Indian locale using Faker for customer names, merchant names, and order context

**Five dispute reasons simulated:**
| Dispute Reason | Difficulty | Notes |
|---------------|------------|-------|
| `not_received` | Medium | Win with delivery confirmation |
| `not_authorized` | Very Hard | Bank sides with customer; hardest to win |
| `not_as_described` | Easy | Win with order confirmation + delivery proof |
| `duplicate_charge` | Hard | Win only with transaction logs |
| `credit_not_processed` | Medium | Win with refund records |

**Win/loss logic (what drives the label):**
- Merchant **wins** when: delivery confirmed, OTP used, customer disputes late, IP logs available
- Merchant **loses** when: no delivery proof, no OTP, quick dispute, new account, high value, no support contact before disputing

**Key feature: `evidence_score`**
- Count of 7 evidence boolean fields that are True
- Most predictive single feature — directly measures how much proof the merchant has

---

#### Step 3.2 — EDA Notebook (`notebooks/exploration_chargeback.ipynb`)

**Key findings from EDA:**
- `evidence_score` mean: 4.8 for wins vs 1.3 for losses — the clearest signal
- `not_authorized` disputes: merchant wins only 12% of the time
- Quick disputes (≤ 3 days): merchant wins only 18% of the time
- `customer_signed_delivery` = True correlates strongly with merchant wins

---

#### Step 3.3 — Feature Engineering (`src/chargeback_features.py`)

**28 features in final model:**
- Transaction details: `order_value`, `product_category`, `payment_method`, `days_to_dispute`, `delivery_days`, `dispute_reason`
- Evidence booleans (7): `delivery_confirmed`, `customer_signed_delivery`, `otp_used`, `ip_logs_available`, `order_confirmation_sent`, `refund_issued`, `customer_contacted_support`
- Customer context: `customer_account_age_days`, `customer_total_orders`, `customer_past_disputes`, `customer_avg_order_value`
- Merchant: `merchant_chargeback_rate`
- Engineered: `is_quick_dispute`, `is_late_dispute`, `is_high_value`, `is_first_order`, `is_new_account`, `is_serial_disputer`, `is_cod`, `evidence_score`, `dispute_difficulty`, `order_value_normalized`

**New concept introduced: `dispute_difficulty`**
- A float (0.0 to 1.0) assigned per dispute_reason
- Captures how hard it is to win each type, independent of evidence
- `not_authorized` = 0.9, `not_as_described` = 0.3, etc.

---

#### Step 3.4 — Model Training (`src/chargeback_train.py`)

**Split:** 70% train / 20% validation / 10% test (stratified, `random_state=42`)

**Class balance:** 60/40 — no `scale_pos_weight` needed (both classes have enough samples)

**Models trained:** XGBoost, LightGBM, Voting Ensemble

**Threshold tuning:** Maximise F1 on validation set, subject to both precision AND recall ≥ 0.75

**Final test set results (best model):**

| Metric | Score | Target |
|--------|-------|--------|
| Precision | ≥ 0.75 | ≥ 0.75 ✅ |
| Recall | ≥ 0.75 | ≥ 0.75 ✅ |
| F1 | — | — |
| AUPRC | ≥ 0.75 | ≥ 0.75 ✅ |

**Model saved to:** `models/chargeback_model.pkl` (~2.3 MB)

---

#### Step 3.5 — API Endpoint (`api/main.py` updated to v3.0.0)

**New endpoint:** `POST /chargeback/analyze`

**What makes this endpoint different from Phases 1 and 2:**
- Response is not just a risk score — it returns an **evidence checklist**
- `evidence_present`: human-readable list of what the merchant has
- `evidence_missing`: what the merchant still needs to collect
- `winability_label`: Weak / Moderate / Strong (not Low / Medium / High)
- `recommendation`: fight / collect more / settle

**New GET endpoint:** `GET /model/chargeback/info`

**Health check** at `GET /` now reports all three models:
```json
{
  "return_risk_scorer":       "Ensemble",
  "fraud_transaction_scorer": "Ensemble",
  "chargeback_responder":     "Ensemble"
}
```

---

#### Step 3.6 — Tests (`tests/test_chargeback_model.py`)

**28 tests across 5 groups:**

| Group | Tests | What is covered |
|-------|-------|----------------|
| Model loading | 5 | File exists, keys, threshold range, feature columns |
| Model metrics | 4 | Precision, recall, AUPRC, score range on held-out test set |
| API health | 2 | Status ok, all 3 models reported |
| Model info endpoint | 5 | HTTP 200, all fields, precision/recall targets, confusion matrix keys |
| Analysis endpoint | 10 | Strong wins Strong, Weak wins Weak, all fields, evidence checklist consistency |
| Input validation | 8 | 422 for invalid enums, missing fields, negative values, out-of-range evidence_score |

**Key test payloads:**
- **Strong win**: delivery confirmed, signed, OTP used, IP logs, confirmation sent, old account, late dispute, `not_as_described`
- **Weak loss**: no evidence, quick dispute, new account, first order, serial disputer, `not_authorized`, luxury item

---

### What Phase 3 Added vs Phase 1 and 2

| Aspect | Phase 1 & 2 | Phase 3 |
|--------|-------------|---------|
| Output type | Risk label (Low/Med/High) | Winability label (Weak/Mod/Strong) |
| Response | Score + recommendation | Score + recommendation + **evidence checklist** |
| Class balance | Imbalanced (15% or 5%) | Balanced (60/40) |
| `scale_pos_weight` | Needed | Not needed |
| Key insight | Behavioural patterns | **Evidence availability** is the dominant signal |

---

### Decisions Made in Session 3

| Decision | Reason |
|----------|--------|
| 60/40 class split (not realistic 5%) | Chargebacks are too rare in real data; balanced split gives a learnable signal |
| `dispute_difficulty` as engineered feature | Dispute type matters independently of evidence — bakes domain knowledge in |
| Winability labels instead of risk labels | Semantically more accurate — merchant is asking "can I win?" not "is this risky?" |
| Evidence checklist in API response | Core business value: merchant needs to know exactly what to collect, not just a score |
| Min targets: precision ≥ 0.75 AND recall ≥ 0.75 | Both matter equally — missing wins loses money, fighting losses wastes time |

---

## Glossary — New Terms Added in Session 2 and 3

**Velocity Features**
Features that measure how fast events are happening. Example: `customer_orders_1h` = number of orders placed by this customer in the last hour. A burst of orders is a fraud signal.

**scale_pos_weight (XGBoost)**
Tells XGBoost to treat each positive (fraud) sample as if it were worth `scale_pos_weight` negative (legitimate) samples. Formula: `negative_count / positive_count`. Used when classes are imbalanced.

**Chargeback**
When a customer disputes a transaction directly with their bank instead of the merchant. The bank reverses the charge and the merchant loses the money + pays a dispute fee (typically ₹500–₹2,000). The merchant can fight it by submitting evidence.

**Winability Score**
A probability (0.0 to 1.0) that the merchant will win a disputed chargeback if they choose to fight it. The Phase 3 model predicts this score.

**Evidence Score**
A count (0 to 7) of how many pieces of evidence the merchant has for a given dispute. Computed from 7 boolean fields: delivery confirmation, signed delivery, OTP, IP logs, order confirmation, refund record, support contact. The single most predictive feature in Phase 3.

**Dispute Difficulty**
A domain-knowledge-based float (0.0 to 1.0) that captures how inherently hard each dispute type is to win, regardless of evidence. `not_authorized` = 0.9 (very hard), `not_as_described` = 0.3 (easier with proof).

---

## Current Status

| Phase | Module | Status |
|-------|--------|--------|
| Phase 1 | Return Risk Scorer | ✅ Complete |
| Phase 2 | Fraud Transaction Scorer | ✅ Complete |
| Phase 3 | Chargeback Evidence Responder | ✅ Complete |
| Phase 4 | Abuse Ring Sentinel | 📋 Planned |

| File | Status |
|------|--------|
| `data/generate_data.py` | ✅ |
| `data/generate_fraud_data.py` | ✅ |
| `data/generate_chargeback_data.py` | ✅ |
| `data/raw/returns.csv` | ✅ 10,000 records |
| `data/raw/transactions.csv` | ✅ 10,000 records |
| `data/raw/chargebacks.csv` | ✅ 8,000 records |
| `notebooks/exploration.ipynb` | ✅ |
| `notebooks/exploration_fraud.ipynb` | ✅ |
| `notebooks/exploration_chargeback.ipynb` | ✅ |
| `src/features.py` | ✅ |
| `src/fraud_features.py` | ✅ |
| `src/chargeback_features.py` | ✅ |
| `src/train.py` | ✅ |
| `src/fraud_train.py` | ✅ |
| `src/chargeback_train.py` | ✅ |
| `models/return_risk_model.pkl` | ✅ |
| `models/fraud_risk_model.pkl` | ✅ |
| `models/chargeback_model.pkl` | ✅ |
| `api/main.py` | ✅ v3.0.0 — all 3 phases |
| `tests/test_model.py` | ✅ 28 tests |
| `tests/test_fraud_model.py` | ✅ 23 tests |
| `tests/test_chargeback_model.py` | ✅ 28 tests |

---

*Log maintained by Kiro. Updated at end of each session.*