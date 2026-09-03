# Evaluation Strategy

## Why Evaluation Matters More Than Accuracy

The most important rule in this project:

> **Accuracy is a useless metric for fraud and risk detection.**

Here's why. Suppose only 1% of transactions are fraud. A model that flags nothing at all — that just says "not fraud" for every single transaction — has 99% accuracy. But it catches zero fraud cases. It is completely useless.

The metrics that actually matter are **precision**, **recall**, and **the cost of being wrong in each direction**.

---

## The Core Metrics

### Precision

**Definition:** Of all the cases your model flagged as risky, what fraction were actually risky?

```
Precision = True Positives / (True Positives + False Positives)
```

**In plain English:** When you raise an alarm, how often are you right?

**Low precision means:** Too many false alarms. Legitimate customers get blocked, questioned, or rejected. This destroys customer trust.

**Example:** You flagged 100 returns as abusive. Only 40 were actually abusive. Precision = 40%.

---

### Recall

**Definition:** Of all the actually risky cases, what fraction did your model catch?

```
Recall = True Positives / (True Positives + False Negatives)
```

**In plain English:** Of all the real fraud, how much did you find?

**Low recall means:** Too many misses. Fraudsters get through. The merchant keeps losing money.

**Example:** There were 200 actually abusive returns. Your model caught 160. Recall = 80%.

---

### The Tradeoff

Precision and recall pull in opposite directions. To increase recall (catch more real fraud), you lower the threshold — but this also flags more legitimate cases, hurting precision. To increase precision (fewer false alarms), you raise the threshold — but this lets more real fraud through, hurting recall.

```
High threshold → High precision, Low recall
                 (fewer alarms, but miss more real fraud)

Low threshold  → Low precision, High recall
                 (catch more real fraud, but more false alarms)
```

The right operating point depends on the business context.

---

## False Positive Cost

A false positive is a legitimate case that your model wrongly flags as risky.

In this context:
- A legitimate return wrongly rejected = angry customer, lost trust, potential dispute
- A legitimate transaction wrongly blocked = lost sale, customer frustration, possible churn

**False positive cost must be quantified, not ignored.**

### How We Calculate It

For the Return Risk Scorer:

```
False Positive Cost = (Number of FP) × (Cost per FP)

Where Cost per FP includes:
  - Average order value lost (customer abandons)
  - Support ticket cost (₹50–200 per ticket)
  - Probability of customer churn × customer lifetime value
```

For a simplified calculation, we use:

```
FP Cost = Number of FPs × Average Order Value × False Positive Weight
```

The False Positive Weight is a business parameter. A low-margin merchant might set it to 1.0x. A high-CLV business might set it to 5x (because losing a good customer is very expensive).

---

## The Confusion Matrix

For every model in this project, we report the full confusion matrix on the held-out test set:

```
                    Predicted: Abusive   Predicted: Legitimate
Actual: Abusive         TP                      FN
Actual: Legitimate      FP                      TN
```

| Cell | Name | Meaning |
|------|------|---------|
| TP (True Positive) | Correctly flagged | Abusive return caught ✓ |
| TN (True Negative) | Correctly cleared | Legitimate return approved ✓ |
| FP (False Positive) | Wrong alarm | Legitimate return wrongly rejected ✗ |
| FN (False Negative) | Missed case | Abusive return not caught ✗ |

---

## The Precision-Recall Curve

Rather than reporting metrics at a single threshold, we plot the full precision-recall curve across all possible thresholds. This shows the full tradeoff space and helps choose the right operating point.

```
Precision
   |
1.0|  *
   |   *
0.8|    *
   |      *
0.6|         *
   |              *
0.4|                    *
   |__________________________
        0.4  0.6  0.8  1.0   Recall
```

We also report **Area Under the Precision-Recall Curve (AUPRC)** as a single summary number. Unlike AUC-ROC, AUPRC is robust to class imbalance and better reflects real-world performance on rare-event detection.

---

## Threshold Selection

The threshold is the cutoff above which a score becomes a "flag." We do not use the default 0.5 threshold blindly.

### Method

1. Plot precision and recall at every threshold from 0.1 to 0.9
2. Compute F-beta score for each threshold

```
F-beta = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

- **β = 1** (F1): Equal weight to precision and recall
- **β = 0.5**: Penalizes false positives more (use when FP cost is high)
- **β = 2**: Penalizes false negatives more (use when missing fraud is very costly)

3. Select the threshold that maximizes the appropriate F-beta score given the business context

4. Document the chosen threshold and the reasoning behind it

---

## Minimum Acceptable Metrics

Before a module is considered "done," it must meet these minimum bars on the held-out test set:

### Return Risk Scorer

| Metric | Minimum Target | Notes |
|--------|---------------|-------|
| Precision | ≥ 0.70 | Of all flags, at least 70% are real abuse |
| Recall | ≥ 0.65 | Catch at least 65% of all actual abuse |
| AUPRC | ≥ 0.75 | Overall model quality |
| FP Rate | ≤ 0.10 | At most 10% of legitimate returns get flagged |

### Fraud Transaction Scorer

| Metric | Minimum Target | Notes |
|--------|---------------|-------|
| Precision | ≥ 0.80 | Higher bar because blocking a payment is more disruptive |
| Recall | ≥ 0.60 | Fraud is rare — catching 60% is already useful |
| AUPRC | ≥ 0.80 | |
| FP Rate | ≤ 0.05 | Very low false alarm rate required for payment blocking |

These targets are starting points. They will be revised once we see real model performance.

---

## What Gets Reported for Each Model

Every trained model ships with a model card that contains:

```
Model: Return Risk Scorer v1.0
Date: YYYY-MM-DD
Training data: 8,000 synthetic return records
Test data: 2,000 held-out records (never seen during training)

Metrics on test set:
  Precision:     0.74
  Recall:        0.68
  F1 Score:      0.71
  AUPRC:         0.79
  
Operating threshold: 0.62
  (chosen to maximize F-0.5, prioritizing precision)

Confusion Matrix:
  TP: 204   FN: 96
  FP: 72    TN: 1628

False Positive Analysis:
  72 legitimate returns wrongly flagged
  Estimated FP cost: 72 × ₹1200 avg order value = ₹86,400
  
  This is offset by:
  204 abusive returns caught × ₹1200 avg value = ₹244,800 protected
  
  Net benefit: ₹158,400 per 2,000 return requests evaluated

Known limitations:
  - Model trained on synthetic data; real-world performance may differ
  - Does not account for merchant-specific return policies
  - Threshold should be re-tuned per merchant context
```

---

## What We Never Do

- Report accuracy as the primary metric
- Pick the threshold that maximizes accuracy
- Evaluate on training data
- Ship a model without documenting its false positive rate
- Claim a model "works" without a held-out test set evaluation
