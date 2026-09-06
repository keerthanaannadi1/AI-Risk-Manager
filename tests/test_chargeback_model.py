"""
tests/test_chargeback_model.py
------------------------------
Automated tests for Phase 3: Chargeback Evidence Responder.

What is tested:
  1.  Model file exists        — chargeback_model.pkl is present on disk
  2.  Model loads cleanly      — pickle.load() raises no exceptions
  3.  Required keys present    — model, threshold, metrics, feature_columns
  4.  Threshold is valid        — between 0.0 and 1.0
  5.  Feature columns match    — consistent with chargeback_features.py
  6.  Precision target         — precision >= 0.75 on held-out test set
  7.  Recall target            — recall >= 0.75 on held-out test set
  8.  AUPRC target             — AUPRC >= 0.75 on held-out test set
  9.  Score range              — all predicted probabilities between 0.0 and 1.0
  10. API health check         — GET / returns status ok
  11. All 3 models in health   — GET / shows all three models loaded
  12. Chargeback model info    — GET /model/chargeback/info returns 200
  13. Model info fields        — all expected metric keys are present
  14. Model info precision     — reported precision meets target
  15. Model info recall        — reported recall meets target
  16. Strong case scores Strong — clear winning dispute → Strong winability
  17. Weak case scores Weak    — clear losing dispute → Weak winability
  18. Response has all fields  — all keys present in analyze response
  19. Score between 0 and 1   — winability_score is a valid probability
  20. Recommendation not empty — recommendation field is populated
  21. Evidence checklist built — evidence_present + evidence_missing both returned
  22. Evidence score matches   — evidence_score == len(evidence_present)
  23. Invalid payment method   — HTTP 422
  24. Invalid dispute reason   — HTTP 422
  25. Invalid product category — HTTP 422
  26. Missing required field   — HTTP 422
  27. Negative order value     — HTTP 422
  28. Evidence score > 7       — HTTP 422

How the test set is reconstructed (no new data needed):
  - Load the same chargebacks.csv used during training
  - Re-split with the same seed (random_state=42) used in chargeback_train.py
  - This gives the same held-out test rows every time — deterministic

Run with:
    pytest tests/test_chargeback_model.py -v
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import pytest

from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, average_precision_score

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.chargeback_features import build_features
from api.main import app

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(ROOT, "data", "raw", "chargebacks.csv")
MODEL_PATH = os.path.join(ROOT, "models", "chargeback_model.pkl")

# ── Minimum targets ────────────────────────────────────────────────────────────
MIN_PRECISION = 0.75
MIN_RECALL    = 0.75
MIN_AUPRC     = 0.75



# Fixtures — loaded once per session, shared across all tests


@pytest.fixture(scope="session")
def model_obj():
    """Load the saved chargeback model package from disk once for all tests."""
    assert os.path.exists(MODEL_PATH), (
        f"Model file not found at {MODEL_PATH}. "
        "Run `python src/chargeback_train.py` first."
    )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def test_set():
    """
    Reconstruct the held-out test set using the same split as chargeback_train.py.
    Same seed (42) = same rows every time — deterministic and reproducible.

    chargeback_train.py split:
      Step 1: 10% test  (test_size=0.10, random_state=42)
      Step 2: 20% of remaining for validation (test_size=0.2222, random_state=42)
    We only need the test set here.
    """
    assert os.path.exists(DATA_PATH), (
        f"Data file not found at {DATA_PATH}. "
        "Run `python data/generate_chargeback_data.py` first."
    )
    df = pd.read_csv(DATA_PATH)
    X, y = build_features(df)

    # Mirrors Step 1 from chargeback_train.py exactly
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42
    )
    return X_test, y_test


@pytest.fixture(scope="session")
def api_client():
    """Create a FastAPI test client — no real server needed."""
    return TestClient(app)


# Test payloads


# Clear winning dispute:
#   - Delivery confirmed + customer signed
#   - OTP used, IP logs available, order confirmation sent
#   - Late dispute (customer waited 45 days), not a quick dispute
#   - Old account, many orders, no prior disputes
#   - Easy dispute type (not_as_described)
#   - High evidence score (6 out of 7)
STRONG_WIN_PAYLOAD = {
    "order_value":               3500.0,
    "product_category":          "electronics",
    "payment_method":            "card",
    "days_to_dispute":           45,
    "delivery_days":             3,
    "dispute_reason":            "not_as_described",
    "delivery_confirmed":        True,
    "customer_signed_delivery":  True,
    "otp_used":                  True,
    "ip_logs_available":         True,
    "order_confirmation_sent":   True,
    "refund_issued":             False,
    "customer_contacted_support": True,
    "customer_account_age_days": 1200,
    "customer_total_orders":     60,
    "customer_past_disputes":    0,
    "customer_avg_order_value":  2500.0,
    "merchant_chargeback_rate":  0.02,
    "is_quick_dispute":          False,
    "is_late_dispute":           True,
    "is_high_value":             False,
    "is_first_order":            False,
    "is_new_account":            False,
    "is_serial_disputer":        False,
    "is_cod":                    False,
    "evidence_score":            6,
    "dispute_difficulty":        0.3,
    "order_value_normalized":    1.4,
}

# Clear losing dispute:
#   - No delivery confirmation, no signature
#   - No OTP, no IP logs
#   - Quick dispute (2 days after transaction)
#   - New account, first order, no support contact
#   - Hardest dispute type (not_authorized — bank sides with customer)
#   - Zero evidence score
WEAK_LOSS_PAYLOAD = {
    "order_value":               12000.0,
    "product_category":          "luxury",
    "payment_method":            "card",
    "days_to_dispute":           2,
    "delivery_days":             5,
    "dispute_reason":            "not_authorized",
    "delivery_confirmed":        False,
    "customer_signed_delivery":  False,
    "otp_used":                  False,
    "ip_logs_available":         False,
    "order_confirmation_sent":   False,
    "refund_issued":             False,
    "customer_contacted_support": False,
    "customer_account_age_days": 10,
    "customer_total_orders":     1,
    "customer_past_disputes":    3,
    "customer_avg_order_value":  500.0,
    "merchant_chargeback_rate":  0.12,
    "is_quick_dispute":          True,
    "is_late_dispute":           False,
    "is_high_value":             True,
    "is_first_order":            True,
    "is_new_account":            True,
    "is_serial_disputer":        True,
    "is_cod":                    False,
    "evidence_score":            0,
    "dispute_difficulty":        0.9,
    "order_value_normalized":    24.0,
}



# Group 1 — Model loading


class TestChargebackModelLoading:

    def test_model_file_exists(self):
        """The chargeback .pkl file must exist before anything else can work."""
        assert os.path.exists(MODEL_PATH), (
            f"Model not found: {MODEL_PATH}. "
            "Run `python src/chargeback_train.py` first."
        )

    def test_model_loads_without_error(self, model_obj):
        """Loading the .pkl should not raise any exception."""
        assert model_obj is not None

    def test_model_has_required_keys(self, model_obj):
        """The saved object must contain model, threshold, metrics, feature_columns."""
        for key in ["model", "threshold", "metrics", "feature_columns"]:
            assert key in model_obj, (
                f"Missing key '{key}' in saved model object. "
                "Retrain using src/chargeback_train.py."
            )

    def test_threshold_is_valid(self, model_obj):
        """Decision threshold must be a probability strictly between 0 and 1."""
        t = model_obj["threshold"]
        assert 0.0 < t < 1.0, (
            f"Threshold {t} is out of valid range (0, 1). "
            "Something went wrong during threshold tuning."
        )

    def test_feature_columns_match(self, model_obj):
        """Feature columns saved with the model must match chargeback_features.py."""
        from src.chargeback_features import FEATURE_COLUMNS
        assert model_obj["feature_columns"] == FEATURE_COLUMNS, (
            "Feature columns in saved model don't match FEATURE_COLUMNS in "
            "chargeback_features.py. Retrain the model after any feature changes."
        )



# Group 2 — Model metrics on held-out test set


class TestChargebackModelMetrics:

    def test_precision_meets_target(self, model_obj, test_set):
        """
        Precision on the held-out test set must be >= 0.75.
        Low precision means we're telling merchants to fight disputes they'll lose —
        wasted effort and dispute fees.
        """
        model          = model_obj["model"]
        threshold      = model_obj["threshold"]
        X_test, y_test = test_set

        proba     = model.predict_proba(X_test)[:, 1]
        preds     = (proba >= threshold).astype(int)
        precision = precision_score(y_test, preds, zero_division=0)

        assert precision >= MIN_PRECISION, (
            f"Precision {precision:.4f} is below the minimum target {MIN_PRECISION}. "
            "Retrain or tune the threshold."
        )

    def test_recall_meets_target(self, model_obj, test_set):
        """
        Recall on the held-out test set must be >= 0.75.
        Low recall means we're missing winnable disputes — merchant loses money
        they could have recovered.
        """
        model          = model_obj["model"]
        threshold      = model_obj["threshold"]
        X_test, y_test = test_set

        proba  = model.predict_proba(X_test)[:, 1]
        preds  = (proba >= threshold).astype(int)
        recall = recall_score(y_test, preds, zero_division=0)

        assert recall >= MIN_RECALL, (
            f"Recall {recall:.4f} is below the minimum target {MIN_RECALL}. "
            "Retrain or adjust the decision threshold."
        )

    def test_auprc_meets_target(self, model_obj, test_set):
        """
        AUPRC on the held-out test set must be >= 0.75.
        AUPRC is more informative than AUC-ROC for binary classification tasks.
        """
        model          = model_obj["model"]
        X_test, y_test = test_set

        proba = model.predict_proba(X_test)[:, 1]
        auprc = average_precision_score(y_test, proba)

        assert auprc >= MIN_AUPRC, (
            f"AUPRC {auprc:.4f} is below the minimum target {MIN_AUPRC}. "
            "Retrain or tune the model."
        )

    def test_score_range_is_valid(self, model_obj, test_set):
        """Every predicted probability must be in [0.0, 1.0] — no exceptions."""
        model     = model_obj["model"]
        X_test, _ = test_set

        proba = model.predict_proba(X_test)[:, 1]
        assert np.all(proba >= 0.0), "Some winability scores are below 0.0"
        assert np.all(proba <= 1.0), "Some winability scores are above 1.0"


# Group 3 — API health check and model info


class TestAPIHealth:

    def test_health_check_returns_ok(self, api_client):
        """GET / must return HTTP 200 with status ok."""
        response = api_client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_check_shows_all_three_models(self, api_client):
        """GET / must report all three models as loaded."""
        response = api_client.get("/")
        data     = response.json()
        assert "models_loaded" in data, "Health check is missing 'models_loaded' key"
        models = data["models_loaded"]
        assert "return_risk_scorer"       in models, "Return model missing from health check"
        assert "fraud_transaction_scorer" in models, "Fraud model missing from health check"
        assert "chargeback_responder"     in models, "Chargeback model missing from health check"


class TestChargebackModelInfo:

    def test_chargeback_model_info_returns_200(self, api_client):
        """GET /model/chargeback/info must return HTTP 200."""
        response = api_client.get("/model/chargeback/info")
        assert response.status_code == 200

    def test_chargeback_model_info_has_required_fields(self, api_client):
        """GET /model/chargeback/info must contain all expected metric fields."""
        response = api_client.get("/model/chargeback/info")
        data     = response.json()
        for field in ["model_name", "threshold", "precision", "recall",
                      "f1", "auc_roc", "auprc", "confusion_matrix"]:
            assert field in data, (
                f"Field '{field}' missing from /model/chargeback/info response"
            )

    def test_chargeback_model_info_precision_meets_target(self, api_client):
        """Precision reported by the info endpoint must meet the minimum target."""
        response = api_client.get("/model/chargeback/info")
        precision = response.json()["precision"]
        assert precision >= MIN_PRECISION, (
            f"Reported precision {precision} is below minimum target {MIN_PRECISION}"
        )

    def test_chargeback_model_info_recall_meets_target(self, api_client):
        """Recall reported by the info endpoint must meet the minimum target."""
        response = api_client.get("/model/chargeback/info")
        recall = response.json()["recall"]
        assert recall >= MIN_RECALL, (
            f"Reported recall {recall} is below minimum target {MIN_RECALL}"
        )

    def test_confusion_matrix_has_all_keys(self, api_client):
        """Confusion matrix in model info must have all four quadrant keys."""
        response = api_client.get("/model/chargeback/info")
        cm       = response.json()["confusion_matrix"]
        for key in ["true_negative", "false_positive", "false_negative", "true_positive"]:
            assert key in cm, f"Confusion matrix missing key: '{key}'"


# Group 4 — Chargeback analysis endpoint

class TestChargebackAnalysisEndpoint:

    def test_strong_win_case_scores_strong(self, api_client):
        """
        A dispute with strong evidence and late filing must be labelled Strong.
        This is the core use case: merchant has all proof, should fight it.
        """
        response = api_client.post("/chargeback/analyze", json=STRONG_WIN_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["winability_label"] == "Strong", (
            f"Expected Strong for clear winning dispute, "
            f"got {data['winability_label']} (score={data['winability_score']})"
        )

    def test_weak_loss_case_scores_weak(self, api_client):
        """
        A dispute with no evidence and quick filing must be labelled Weak.
        Merchant should settle rather than waste time fighting an unwinnable dispute.
        """
        response = api_client.post("/chargeback/analyze", json=WEAK_LOSS_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["winability_label"] == "Weak", (
            f"Expected Weak for clear losing dispute, "
            f"got {data['winability_label']} (score={data['winability_score']})"
        )

    def test_response_has_all_required_fields(self, api_client):
        """POST /chargeback/analyze response must contain all expected keys."""
        response = api_client.post("/chargeback/analyze", json=STRONG_WIN_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        for field in [
            "winability_score", "winability_label", "recommendation",
            "evidence_score", "evidence_present", "evidence_missing",
            "model_name", "threshold_used", "scored_at",
        ]:
            assert field in data, (
                f"Field '{field}' missing from /chargeback/analyze response"
            )

    def test_winability_score_is_between_0_and_1(self, api_client):
        """Winability score must be a valid probability for both payloads."""
        for payload in [STRONG_WIN_PAYLOAD, WEAK_LOSS_PAYLOAD]:
            response = api_client.post("/chargeback/analyze", json=payload)
            score    = response.json()["winability_score"]
            assert 0.0 <= score <= 1.0, f"Winability score out of range: {score}"

    def test_recommendation_is_not_empty(self, api_client):
        """Recommendation must never be blank — merchant always needs guidance."""
        for payload in [STRONG_WIN_PAYLOAD, WEAK_LOSS_PAYLOAD]:
            response = api_client.post("/chargeback/analyze", json=payload)
            rec = response.json()["recommendation"].strip()
            assert rec != "", "Recommendation field is empty"

    def test_evidence_present_is_a_list(self, api_client):
        """evidence_present must be a list of strings."""
        response = api_client.post("/chargeback/analyze", json=STRONG_WIN_PAYLOAD)
        assert isinstance(response.json()["evidence_present"], list)

    def test_evidence_missing_is_a_list(self, api_client):
        """evidence_missing must be a list of strings."""
        response = api_client.post("/chargeback/analyze", json=STRONG_WIN_PAYLOAD)
        assert isinstance(response.json()["evidence_missing"], list)

    def test_evidence_score_matches_evidence_present_count(self, api_client):
        """
        evidence_score must equal len(evidence_present).
        These two fields must always be consistent with each other.
        """
        response = api_client.post("/chargeback/analyze", json=STRONG_WIN_PAYLOAD)
        data     = response.json()
        assert data["evidence_score"] == len(data["evidence_present"]), (
            f"evidence_score={data['evidence_score']} but "
            f"len(evidence_present)={len(data['evidence_present'])}. "
            "These must always match."
        )

    def test_evidence_checklist_totals_seven(self, api_client):
        """
        evidence_present + evidence_missing must always sum to exactly 7.
        There are 7 evidence fields — all must be accounted for.
        """
        for payload in [STRONG_WIN_PAYLOAD, WEAK_LOSS_PAYLOAD]:
            response = api_client.post("/chargeback/analyze", json=payload)
            data     = response.json()
            total    = len(data["evidence_present"]) + len(data["evidence_missing"])
            assert total == 7, (
                f"Evidence checklist total is {total}, expected 7. "
                "All 7 evidence fields must appear in either present or missing."
            )

    def test_strong_win_has_no_missing_evidence(self, api_client):
        """
        The strong win payload has all 7 evidence fields set to True,
        so evidence_missing must be empty.
        """
        # Ensure all evidence booleans are True
        payload = {**STRONG_WIN_PAYLOAD, "refund_issued": True}
        response = api_client.post("/chargeback/analyze", json=payload)
        data     = response.json()
        assert data["evidence_missing"] == [], (
            f"Expected no missing evidence, but got: {data['evidence_missing']}"
        )

    def test_weak_loss_has_no_present_evidence(self, api_client):
        """
        The weak loss payload has all evidence booleans set to False,
        so evidence_present must be empty.
        """
        response = api_client.post("/chargeback/analyze", json=WEAK_LOSS_PAYLOAD)
        data     = response.json()
        assert data["evidence_present"] == [], (
            f"Expected no present evidence, but got: {data['evidence_present']}"
        )

    def test_scored_at_is_iso_timestamp(self, api_client):
        """scored_at must be a parseable ISO 8601 timestamp string."""
        from datetime import datetime
        response  = api_client.post("/chargeback/analyze", json=STRONG_WIN_PAYLOAD)
        scored_at = response.json()["scored_at"]
        # Will raise ValueError if not a valid ISO timestamp
        parsed = datetime.fromisoformat(scored_at.replace("Z", "+00:00"))
        assert parsed is not None



# Group 5 — Input validation (all must return 422, not 500)


class TestChargebackInputValidation:

    def test_invalid_payment_method_returns_422(self, api_client):
        """An unrecognised payment_method must be rejected with HTTP 422."""
        bad = {**STRONG_WIN_PAYLOAD, "payment_method": "crypto"}
        response = api_client.post("/chargeback/analyze", json=bad)
        assert response.status_code == 422, (
            f"Expected 422 for invalid payment_method, got {response.status_code}"
        )

    def test_invalid_dispute_reason_returns_422(self, api_client):
        """An unrecognised dispute_reason must be rejected with HTTP 422."""
        bad = {**STRONG_WIN_PAYLOAD, "dispute_reason": "i_changed_my_mind"}
        response = api_client.post("/chargeback/analyze", json=bad)
        assert response.status_code == 422, (
            f"Expected 422 for invalid dispute_reason, got {response.status_code}"
        )

    def test_invalid_product_category_returns_422(self, api_client):
        """An unrecognised product_category must be rejected with HTTP 422."""
        bad = {**STRONG_WIN_PAYLOAD, "product_category": "spaceship"}
        response = api_client.post("/chargeback/analyze", json=bad)
        assert response.status_code == 422, (
            f"Expected 422 for invalid product_category, got {response.status_code}"
        )

    def test_missing_required_field_returns_422(self, api_client):
        """Omitting a required field must return HTTP 422."""
        incomplete = {k: v for k, v in STRONG_WIN_PAYLOAD.items() if k != "order_value"}
        response   = api_client.post("/chargeback/analyze", json=incomplete)
        assert response.status_code == 422, (
            f"Expected 422 for missing field, got {response.status_code}"
        )

    def test_negative_order_value_returns_422(self, api_client):
        """Negative order_value must be rejected with HTTP 422."""
        bad = {**STRONG_WIN_PAYLOAD, "order_value": -100.0}
        response = api_client.post("/chargeback/analyze", json=bad)
        assert response.status_code == 422, (
            f"Expected 422 for negative order_value, got {response.status_code}"
        )

    def test_negative_days_to_dispute_returns_422(self, api_client):
        """Negative days_to_dispute is impossible and must be rejected."""
        bad = {**STRONG_WIN_PAYLOAD, "days_to_dispute": -1}
        response = api_client.post("/chargeback/analyze", json=bad)
        assert response.status_code == 422, (
            f"Expected 422 for negative days_to_dispute, got {response.status_code}"
        )

    def test_negative_customer_account_age_returns_422(self, api_client):
        """Negative customer_account_age_days is impossible and must be rejected."""
        bad = {**STRONG_WIN_PAYLOAD, "customer_account_age_days": -5}
        response = api_client.post("/chargeback/analyze", json=bad)
        assert response.status_code == 422, (
            f"Expected 422 for negative customer_account_age_days, "
            f"got {response.status_code}"
        )
