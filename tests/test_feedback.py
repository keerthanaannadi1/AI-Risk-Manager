"""
tests/test_feedback.py
----------------------
Automated tests for Phase 5: Feedback & Retraining Pipeline.

What is tested:
  1.  POST /feedback/return        — valid abusive outcome accepted
  2.  POST /feedback/return        — valid legitimate outcome accepted
  3.  POST /feedback/return        — invalid outcome rejected (422)
  4.  POST /feedback/return        — invalid predicted_label rejected (422)
  5.  POST /feedback/transaction   — valid fraud outcome accepted
  6.  POST /feedback/transaction   — valid legitimate outcome accepted
  7.  POST /feedback/transaction   — invalid outcome rejected (422)
  8.  POST /feedback/chargeback    — valid won outcome accepted
  9.  POST /feedback/chargeback    — valid lost outcome accepted
  10. POST /feedback/chargeback    — invalid outcome rejected (422)
  11. GET  /feedback/summary       — returns all three modules in summary
  12. is_correct=1 when model was right (High + abusive)
  13. is_correct=0 when model was wrong (High + legitimate)
  14. is_correct=1 when model was right (Low + legitimate)
  15. feedback_store write + read round trip
  16. feedback_store rejects unknown module
  17. feedback_store rejects invalid outcome
  18. feedback_summary counts correctly
  19. retrain.py --dry-run runs without error
  20. retrain.py runs to completion and creates versioned files

Run with:
    pytest tests/test_feedback.py -v
"""

import os
import sys
import shutil
import subprocess
import tempfile

import pytest
from fastapi.testclient import TestClient

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from api.main import app
from data.outcomes.feedback_store import (
    write_feedback,
    read_feedback,
    feedback_summary,
    FEEDBACK_FILES,
)

client = TestClient(app)

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_feedback_files():
    """
    Before each test: back up existing feedback CSVs.
    After each test:  restore them so tests don't pollute the real store.
    """
    backups = {}
    for module, path in FEEDBACK_FILES.items():
        if os.path.exists(path):
            backups[module] = open(path, "rb").read()

    yield  # run the test

    # Restore originals
    for module, path in FEEDBACK_FILES.items():
        if module in backups:
            with open(path, "wb") as f:
                f.write(backups[module])
        elif os.path.exists(path):
            os.remove(path)


# ══════════════════════════════════════════════════════════════════════════════
# 1-4: POST /feedback/return
# ══════════════════════════════════════════════════════════════════════════════

def test_feedback_return_abusive():
    """Test 1 — valid abusive return feedback is accepted."""
    resp = client.post("/feedback/return", json={
        "original_id":     "RET-001",
        "risk_score":      0.88,
        "predicted_label": "High",
        "actual_outcome":  "abusive",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["module"] == "return"
    assert data["actual_outcome"] == "abusive"
    assert data["original_id"] == "RET-001"


def test_feedback_return_legitimate():
    """Test 2 — valid legitimate return feedback is accepted."""
    resp = client.post("/feedback/return", json={
        "original_id":     "RET-002",
        "risk_score":      0.10,
        "predicted_label": "Low",
        "actual_outcome":  "legitimate",
    })
    assert resp.status_code == 200
    assert resp.json()["actual_outcome"] == "legitimate"


def test_feedback_return_invalid_outcome():
    """Test 3 — invalid outcome value returns 422."""
    resp = client.post("/feedback/return", json={
        "original_id":     "RET-003",
        "risk_score":      0.5,
        "predicted_label": "High",
        "actual_outcome":  "maybe",        # not allowed
    })
    assert resp.status_code == 422


def test_feedback_return_invalid_label():
    """Test 4 — invalid predicted_label returns 422."""
    resp = client.post("/feedback/return", json={
        "original_id":     "RET-004",
        "risk_score":      0.5,
        "predicted_label": "Unknown",      # not allowed
        "actual_outcome":  "abusive",
    })
    assert resp.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# 5-7: POST /feedback/transaction
# ══════════════════════════════════════════════════════════════════════════════

def test_feedback_transaction_fraud():
    """Test 5 — valid fraud outcome accepted."""
    resp = client.post("/feedback/transaction", json={
        "original_id":     "TX-001",
        "risk_score":      0.95,
        "predicted_label": "High",
        "actual_outcome":  "fraud",
    })
    assert resp.status_code == 200
    assert resp.json()["module"] == "fraud"


def test_feedback_transaction_legitimate():
    """Test 6 — valid legitimate outcome accepted."""
    resp = client.post("/feedback/transaction", json={
        "original_id":     "TX-002",
        "risk_score":      0.08,
        "predicted_label": "Low",
        "actual_outcome":  "legitimate",
    })
    assert resp.status_code == 200
    assert resp.json()["actual_outcome"] == "legitimate"


def test_feedback_transaction_invalid_outcome():
    """Test 7 — invalid outcome returns 422."""
    resp = client.post("/feedback/transaction", json={
        "original_id":     "TX-003",
        "risk_score":      0.5,
        "predicted_label": "High",
        "actual_outcome":  "suspicious",   # not allowed
    })
    assert resp.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# 8-10: POST /feedback/chargeback
# ══════════════════════════════════════════════════════════════════════════════

def test_feedback_chargeback_won():
    """Test 8 — valid won outcome accepted."""
    resp = client.post("/feedback/chargeback", json={
        "original_id":     "CB-001",
        "risk_score":      0.91,
        "predicted_label": "Strong",
        "actual_outcome":  "won",
    })
    assert resp.status_code == 200
    assert resp.json()["module"] == "chargeback"


def test_feedback_chargeback_lost():
    """Test 9 — valid lost outcome accepted."""
    resp = client.post("/feedback/chargeback", json={
        "original_id":     "CB-002",
        "risk_score":      0.22,
        "predicted_label": "Weak",
        "actual_outcome":  "lost",
    })
    assert resp.status_code == 200
    assert resp.json()["actual_outcome"] == "lost"


def test_feedback_chargeback_invalid_outcome():
    """Test 10 — invalid outcome returns 422."""
    resp = client.post("/feedback/chargeback", json={
        "original_id":     "CB-003",
        "risk_score":      0.5,
        "predicted_label": "Strong",
        "actual_outcome":  "pending",      # not allowed
    })
    assert resp.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# 11: GET /feedback/summary
# ══════════════════════════════════════════════════════════════════════════════

def test_feedback_summary_has_all_modules():
    """Test 11 — summary endpoint returns all three modules."""
    resp = client.get("/feedback/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "feedback_summary" in data
    summary = data["feedback_summary"]
    assert "return" in summary
    assert "fraud" in summary
    assert "chargeback" in summary


# ══════════════════════════════════════════════════════════════════════════════
# 12-14: is_correct logic
# ══════════════════════════════════════════════════════════════════════════════

def test_is_correct_true_positive():
    """Test 12 — model predicted High and outcome was abusive → is_correct=1."""
    resp = client.post("/feedback/return", json={
        "original_id":     "RET-010",
        "risk_score":      0.90,
        "predicted_label": "High",
        "actual_outcome":  "abusive",
    })
    assert resp.status_code == 200
    assert resp.json()["is_correct"] == 1


def test_is_correct_false_positive():
    """Test 13 — model predicted High but outcome was legitimate → is_correct=0."""
    resp = client.post("/feedback/return", json={
        "original_id":     "RET-011",
        "risk_score":      0.75,
        "predicted_label": "High",
        "actual_outcome":  "legitimate",
    })
    assert resp.status_code == 200
    assert resp.json()["is_correct"] == 0


def test_is_correct_true_negative():
    """Test 14 — model predicted Low and outcome was legitimate → is_correct=1."""
    resp = client.post("/feedback/return", json={
        "original_id":     "RET-012",
        "risk_score":      0.05,
        "predicted_label": "Low",
        "actual_outcome":  "legitimate",
    })
    assert resp.status_code == 200
    assert resp.json()["is_correct"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 15-18: feedback_store unit tests
# ══════════════════════════════════════════════════════════════════════════════

def test_feedback_store_write_and_read():
    """Test 15 — write a record and read it back correctly."""
    write_feedback("fraud", "TX-UNIT-01", 0.93, "High", "fraud")
    records = read_feedback("fraud")
    assert len(records) >= 1
    last = records[-1]
    assert last["original_id"] == "TX-UNIT-01"
    assert last["actual_outcome"] == "fraud"
    assert last["module"] == "fraud"


def test_feedback_store_rejects_unknown_module():
    """Test 16 — writing to an unknown module raises ValueError."""
    with pytest.raises(ValueError, match="module must be one of"):
        write_feedback("unknown_module", "X-001", 0.5, "High", "fraud")


def test_feedback_store_rejects_invalid_outcome():
    """Test 17 — writing an invalid outcome raises ValueError."""
    with pytest.raises(ValueError, match="actual_outcome"):
        write_feedback("fraud", "TX-BAD", 0.5, "High", "maybe")


def test_feedback_summary_counts():
    """Test 18 — feedback_summary counts records correctly."""
    write_feedback("fraud", "TX-S1", 0.9, "High",   "fraud")       # correct
    write_feedback("fraud", "TX-S2", 0.8, "High",   "legitimate")  # wrong
    write_feedback("fraud", "TX-S3", 0.1, "Low",    "legitimate")  # correct

    summary = feedback_summary()
    fraud_summary = summary["fraud"]
    assert fraud_summary["total"] >= 3
    # At least 2 of our 3 written records are correct
    assert fraud_summary["correct"] >= 2


# ══════════════════════════════════════════════════════════════════════════════
# 19-20: retrain.py script tests
# ══════════════════════════════════════════════════════════════════════════════

def test_retrain_dry_run():
    """Test 19 — retrain.py --dry-run completes without error."""
    result = subprocess.run(
        [sys.executable, "src/retrain.py", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"dry-run failed:\n{result.stdout}\n{result.stderr}"
    assert "dry-run" in result.stdout.lower()


def test_retrain_creates_versioned_files():
    """Test 20 — retrain.py creates timestamped files in models/versions/."""
    versions_dir = os.path.join(ROOT, "models", "versions")
    files_before = set(os.listdir(versions_dir)) if os.path.exists(versions_dir) else set()

    result = subprocess.run(
        [sys.executable, "src/retrain.py", "--module", "fraud"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"retrain failed:\n{result.stdout}\n{result.stderr}"

    files_after = set(os.listdir(versions_dir)) if os.path.exists(versions_dir) else set()
    new_files   = files_after - files_before

    # At least one new versioned file should have been created
    assert len(new_files) >= 1
    # The new file should be a .pkl with the fraud model name
    fraud_versions = [f for f in new_files if "fraud_risk_model_v_" in f and f.endswith(".pkl")]
    assert len(fraud_versions) >= 1, f"No fraud versioned file found. New files: {new_files}"
