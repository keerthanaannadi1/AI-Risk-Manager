"""
feedback_store.py
-----------------
Outcomes store for the AI Risk Manager feedback & retraining pipeline.

What this does:
  - Defines the schema for each feedback record
  - Writes confirmed outcomes to CSV files (one per module)
  - Reads feedback records back for use in retraining

One CSV file per module:
  data/outcomes/return_feedback.csv
  data/outcomes/fraud_feedback.csv
  data/outcomes/chargeback_feedback.csv

Each row represents one confirmed real-world outcome — what the model
predicted vs what actually happened. These are used by src/retrain.py
to improve the model over time.

CSV schema (all three files share the same columns):
  record_id       — unique ID for this feedback record
  original_id     — the transaction/return/dispute ID from the original request
  module          — return | fraud | chargeback
  risk_score      — the score the model gave at prediction time (0.0 to 1.0)
  predicted_label — what the model said: Low / Medium / High (or Weak/Moderate/Strong)
  actual_outcome  — what actually happened (confirmed by merchant)
  is_correct      — 1 if the model was right, 0 if it was wrong
  submitted_at    — when the merchant submitted this feedback (ISO timestamp)
"""

import os
import uuid
import csv
from datetime import datetime, timezone
from typing import Literal

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTCOMES_DIR = os.path.join(ROOT, "outcomes")

FEEDBACK_FILES = {
    "return":     os.path.join(OUTCOMES_DIR, "return_feedback.csv"),
    "fraud":      os.path.join(OUTCOMES_DIR, "fraud_feedback.csv"),
    "chargeback": os.path.join(OUTCOMES_DIR, "chargeback_feedback.csv"),
}

# ── CSV column header ─────────────────────────────────────────────────────────
COLUMNS = [
    "record_id",
    "original_id",
    "module",
    "risk_score",
    "predicted_label",
    "actual_outcome",
    "is_correct",
    "submitted_at",
]

# ── Allowed actual outcomes per module ────────────────────────────────────────
VALID_OUTCOMES = {
    "return":     ["abusive", "legitimate"],
    "fraud":      ["fraud", "legitimate"],
    "chargeback": ["won", "lost"],
}

# ── Mapping: which actual outcomes count as "the risky class was correct" ─────
# Used to compute is_correct:
#   is_correct = 1 if model predicted High/Strong AND outcome is the bad class
#                  OR model predicted Low/Weak   AND outcome is the good class
RISKY_OUTCOME = {
    "return":     "abusive",
    "fraud":      "fraud",
    "chargeback": "won",   # for chargeback, "won" = merchant wins = model was right to flag Strong
}

SAFE_LABELS = {
    "return":     "Low",
    "fraud":      "Low",
    "chargeback": "Weak",
}


def _ensure_file(module: str) -> str:
    """Create the CSV file with headers if it doesn't exist yet."""
    path = FEEDBACK_FILES[module]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
    return path


def write_feedback(
    module: Literal["return", "fraud", "chargeback"],
    original_id: str,
    risk_score: float,
    predicted_label: str,
    actual_outcome: str,
) -> dict:
    """
    Write one confirmed outcome to the feedback store.

    Parameters
    ----------
    module          : "return", "fraud", or "chargeback"
    original_id     : the transaction/return/dispute ID from the original request
    risk_score      : the probability score the model produced (0.0 to 1.0)
    predicted_label : the risk label the model returned (Low / Medium / High etc.)
    actual_outcome  : what actually happened — must be in VALID_OUTCOMES[module]

    Returns
    -------
    The written record as a dict.
    """
    if module not in FEEDBACK_FILES:
        raise ValueError(f"module must be one of: {list(FEEDBACK_FILES.keys())}")

    if actual_outcome not in VALID_OUTCOMES[module]:
        raise ValueError(
            f"actual_outcome for module '{module}' must be one of: "
            f"{VALID_OUTCOMES[module]}"
        )

    # is_correct: did the model correctly identify the risky case?
    # High/Strong predicted AND actual is the bad outcome → correct
    # Low/Weak predicted    AND actual is the good outcome → correct
    risky  = RISKY_OUTCOME[module]
    safe   = SAFE_LABELS[module]
    predicted_risky = predicted_label != safe   # Medium or High/Strong
    outcome_risky   = actual_outcome == risky

    is_correct = int(predicted_risky == outcome_risky)

    record = {
        "record_id":      str(uuid.uuid4()),
        "original_id":    original_id,
        "module":         module,
        "risk_score":     round(risk_score, 4),
        "predicted_label": predicted_label,
        "actual_outcome": actual_outcome,
        "is_correct":     is_correct,
        "submitted_at":   datetime.now(timezone.utc).isoformat(),
    }

    path = _ensure_file(module)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writerow(record)

    return record


def read_feedback(module: str) -> list[dict]:
    """
    Read all feedback records for a given module.

    Returns a list of dicts, one per row. Empty list if no feedback yet.
    """
    if module not in FEEDBACK_FILES:
        raise ValueError(f"module must be one of: {list(FEEDBACK_FILES.keys())}")

    path = FEEDBACK_FILES[module]
    if not os.path.exists(path):
        return []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def feedback_summary() -> dict:
    """
    Return a summary of all feedback collected so far.
    Useful for the API health check and monitoring.
    """
    summary = {}
    for module in FEEDBACK_FILES:
        records = read_feedback(module)
        if not records:
            summary[module] = {"total": 0, "correct": 0, "accuracy": None}
            continue
        total   = len(records)
        correct = sum(int(r["is_correct"]) for r in records)
        summary[module] = {
            "total":    total,
            "correct":  correct,
            "accuracy": round(correct / total, 4) if total > 0 else None,
        }
    return summary
