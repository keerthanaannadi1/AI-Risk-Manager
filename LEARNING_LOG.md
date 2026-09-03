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
