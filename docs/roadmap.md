# Roadmap

## Guiding Principle

Build one thing at a time. Finish it completely. Measure it honestly. Then move to the next.

A single working module with documented precision and recall is worth more than four half-built ones.

---

## Phase 1: Return Risk Scorer
**Status: 🔨 In Progress**
**Goal: A working scorer with honest metrics on a held-out test set**

### Step 1.1 — Data Generation
- [ ] Write `data/generate_data.py` to produce synthetic return records
- [ ] Generate ~10,000 return records with realistic distributions
- [ ] Include both normal and abusive return patterns
- [ ] Save to `data/raw/returns.csv`

**What you'll learn:** Python, Faker, NumPy, realistic data distributions

---

### Step 1.2 — Exploratory Data Analysis
- [ ] Load data in `notebooks/exploration.ipynb`
- [ ] Check class balance (what % are abusive returns?)
- [ ] Plot feature distributions (return rate, days to return, category breakdown)
- [ ] Identify which features correlate most with abuse
- [ ] Check for missing values and outliers

**What you'll learn:** Pandas, Matplotlib/Seaborn, data intuition

---

### Step 1.3 — Feature Engineering
- [ ] Write `src/features.py` with all feature transformations
- [ ] Compute customer-level aggregates (return rate, order count, account age)
- [ ] Encode categorical variables (product category, return reason)
- [ ] Handle missing values with sensible defaults
- [ ] Write unit tests for each feature function

**What you'll learn:** Feature engineering, categorical encoding, aggregation logic

---

### Step 1.4 — Model Training
- [ ] Write `src/train.py`
- [ ] Split data 80% train / 20% test (stratified split to preserve class balance)
- [ ] Train XGBoost classifier
- [ ] Save model to `models/return_risk_model.pkl`
- [ ] Log training metrics (AUC, precision, recall at default threshold)

**What you'll learn:** XGBoost, train/test split, model serialization

---

### Step 1.5 — Evaluation
- [ ] Evaluate model on held-out test set
- [ ] Plot precision-recall curve
- [ ] Plot confusion matrix
- [ ] Compute false positive cost at different thresholds
- [ ] Choose and document the operating threshold with justification
- [ ] Document final metrics in README

**What you'll learn:** Precision, recall, F1, confusion matrix, threshold tuning, cost analysis

---

### Step 1.6 — API
- [ ] Write `api/main.py` with FastAPI
- [ ] Implement `POST /score/return` endpoint
- [ ] Load model at startup
- [ ] Return risk score, risk level, recommended action, and top features
- [ ] Test with sample requests

**What you'll learn:** FastAPI, Pydantic, REST APIs, model serving

---

### Step 1.7 — Testing
- [ ] Write `tests/test_model.py`
- [ ] Assert precision >= target on test set
- [ ] Assert recall >= target on test set
- [ ] Test API endpoint with known inputs

**What you'll learn:** Pytest, writing assertion-based tests, reproducibility

---

### Phase 1 Completion Criteria
- Model trained and saved
- Precision and recall documented on held-out test set
- False positive cost calculated and documented
- API running and responding correctly
- Tests passing

**Milestone: First working, measured, deployable module ✓**

---

## Phase 2: Fraud Transaction Scorer
**Status: 📋 Planned**
**Depends on: Phase 1 complete**

### What's different from Phase 1
- Much more imbalanced data (fraud is ~0.5% of transactions vs ~15% for return abuse)
- Velocity features require time-series thinking (orders per hour, per day)
- Device and IP features require new data fields
- Model needs SMOTE or class weight handling

### Steps
- [ ] Generate synthetic transaction data with fraud patterns
- [ ] Add velocity and device features to feature engineering
- [ ] Handle class imbalance (SMOTE or `scale_pos_weight` in XGBoost)
- [ ] Train and evaluate model
- [ ] Add `POST /score/transaction` endpoint
- [ ] Document metrics

**New skills:** Imbalanced learning, velocity features, real-time scoring patterns

---

## Phase 3: Chargeback Evidence Responder
**Status: 📋 Planned**
**Depends on: Phase 2 complete**

### What's different from Phase 1 and 2
- Not pure classification — involves evidence retrieval and document generation
- Optionally uses an LLM to write the dispute response text
- Needs a "winnability" scorer (how likely are we to win this dispute?)

### Steps
- [ ] Define evidence schema (what fields constitute evidence)
- [ ] Build evidence collector (pulls from order, transaction, delivery records)
- [ ] Train winnability scorer
- [ ] Build document generator (template + data → formatted response)
- [ ] Optionally integrate LLM for natural language response writing
- [ ] Add `POST /score/chargeback` endpoint

**New skills:** Document generation, retrieval logic, LLM integration (optional)

---

## Phase 4: Abuse Ring Sentinel
**Status: 📋 Planned**
**Depends on: Phase 3 complete**

### What's different from all previous phases
- Not per-transaction — analyzes relationships between accounts
- Requires graph construction and community detection
- Much more complex feature space (graph features vs tabular features)

### Steps
- [ ] Generate synthetic account network data with planted abuse rings
- [ ] Build account relationship graph using NetworkX
- [ ] Run community detection (Louvain algorithm)
- [ ] Engineer graph-level and node-level features
- [ ] Train abuse cluster scorer
- [ ] Add `/score/abuse-ring` endpoint

**New skills:** Graph analysis, NetworkX, community detection, network features

---

## Timeline (Rough Estimate)

| Phase | Estimated Duration | Key Deliverable |
|-------|--------------------|-----------------|
| Phase 1: Return Risk Scorer | 3–4 weeks | Working model + API + metrics |
| Phase 2: Fraud Scorer | 2–3 weeks | Working model + API + metrics |
| Phase 3: Chargeback Responder | 2–3 weeks | Evidence package + win scorer |
| Phase 4: Abuse Ring Sentinel | 3–4 weeks | Graph-based cluster scorer |

Total: ~10–14 weeks of focused part-time work.

These are learning estimates — if you already know Python and Pandas well, phases will go faster.

---

## What "Done" Means for Each Phase

A phase is done when:
1. The model is trained and saved
2. Precision and recall are documented on the held-out test set
3. False positive cost is calculated
4. The API endpoint works and returns the correct response format
5. At least one test is passing that asserts minimum precision/recall

Not done until the numbers are written down.
