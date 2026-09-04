"""
tests/test_fraud_model.py
-------------------------
Automated tests for Phase 2: Fraud Transaction Scorer.

What is tested:
  1.  Model loading          — saved .pkl file loads without errors
  2.  Model has required keys — model, threshold, metrics, feature_columns
  3.  Threshold is valid      — between 0 and 1
  4.  Feature columns match   — consistent with fraud_features.py
  5.  Precision target        — >= 0.80 on held-out test set
  6.  Recall target           — >= 0.70 on held-out test set
  7.  AUPRC target            — >= 0.75 on held-out test set
  8.  Score range             — all predictions between 0.0 and 1.0
  9.  API health check        — GET / returns status ok
  10. Both models in health   — GET / shows both models loaded
  11. Fraud model info        — GET /model/fraud/info returns 200
  12. Fraud model info fields — all metric fields present
  13. Fraud model precision   — reported precision meets target
  14. Fraud model recall      — reported recall meets target
  15. High-risk scores High   — clearly fraudulent profile → High
  16. Low-risk scores Low     — clearly legitimate profile → Low
  17. Response has all fields — all required keys in response
  18. Score between 0 and 1   — valid probability in response
  19. Recommendation not empty— recommendation field is populated
  20. Invalid payment method  — HTTP 422
  21. Invalid category        — HTTP 422
  22. Missing required field  — HTTP 422
  23. Negative order value    — HTTP 422

Run with:
    pytest tests/test_fraud_model.py -v
"""

import os
import sys
import pickle

import numpy as np
import pytest

from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, average_precision_score

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
from src.fraud_features import build_features
from api.main import app

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(ROOT, "data", "raw", "transactions.csv")
MODEL_PATH = os.path.join(ROOT, "models", "fraud_risk_model.pkl")

# ── Minimum targets ────────────────────────────────────────────────────────────
MIN_PRECISION = 0.80
MIN_RECALL    = 0.70
MIN_AUPRC     = 0.75


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def model_obj():
    """Load the saved fraud model package from disk once for all tests."""
    assert os.path.exists(MODEL_PATH), (
        f"Model file not found at {MODEL_PATH}. "
        "Run `python src/fraud_train.py` first."
    )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def test_set():
    """
    Reconstruct the held-out test set using the same split as fraud_train.py.
    Same seed (42) = same 1,000 test rows every time.
    """
    assert os.path.exists(DATA_PATH), (
        f"Data file not found at {DATA_PATH}. "
        "Run `python data/generate_fraud_data.py` first."
    )
    df = pd.read_csv(DATA_PATH)
    X, y = build_features(df)

    # Mirrors the split in fraud_train.py exactly
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42
    )
    return X_test, y_test


@pytest.fixture(scope="session")
def api_client():
    """Create a FastAPI test client (no real server needed)."""
    return TestClient(app)


# ══════════════════════════════════════════════════════════════════════════════
# Test payloads
# ══════════════════════════════════════════════════════════════════════════════

# Clearly fraudulent: 3am, new device, new account, high-value electronics,
# card payment, 5 failed attempts, VPN, address mismatch, 6 orders/hr
HIGH_RISK_PAYLOAD = {
    "order_value":               18000.0,
    "product_category":          "electronics",
    "payment_method":            "card",
    "hour_of_day":               3,
    "day_of_week":               2,
    "failed_attempts":           5,
    "is_vpn":                    True,
    "customer_account_age_days": 12,
    "customer_total_past_orders": 2,
    "customer_avg_order_value":  1500.0,
    "customer_past_fraud_flags": 0,
    "customer_orders_1h":        6,
    "customer_orders_24h":       8,
    "merchant_fraud_rate":       0.08,
    "is_night_transaction":      True,
    "is_weekend":                False,
    "is_new_device":             True,
    "is_different_city":         True,
    "is_high_value":             True,
    "is_first_order":            False,
    "is_new_account":            True,
    "address_mismatch":          True,
    "multiple_failed_attempts":  True,
    "high_velocity_1h":          True,
    "order_value_normalized":    12.0,
    "new_account_high_value_card": True,
    "payment_risk_score":        0.9,
    "category_risk_score":       0.8,
}

# Clearly legitimate: 2pm, known device, old account, low-value books,
# UPI, 0 failures, no VPN, addresses match, 1 order/hr
LOW_RISK_PAYLOAD = {
    "order_value":               800.0,
    "product_category":          "books",
    "payment_method":            "upi",
    "hour_of_day":               14,
    "day_of_week":               1,
    "failed_attempts":           0,
    "is_vpn":                    False,
    "customer_account_age_days": 900,
    "customer_total_past_orders": 45,
    "customer_avg_order_value":  1200.0,
    "customer_past_fraud_flags": 0,
    "customer_orders_1h":        1,
    "customer_orders_24h":       2,
    "merchant_fraud_rate":       0.03,
    "is_night_transaction":      False,
    "is_weekend":                False,
    "is_new_device":             False,
    "is_different_city":         False,
    "is_high_value":             False,
    "is_first_order":            False,
    "is_new_account":            False,
    "address_mismatch":          False,
    "multiple_failed_attempts":  False,
    "high_velocity_1h":          False,
    "order_value_normalized":    0.67,
    "new_account_high_value_card": False,
    "payment_risk_score":        0.3,
    "category_risk_score":       0.1,
}


# ══════════════════════════════════════════════════════════════════════════════
# Group 1 — Model loading
# ══════════════════════════════════════════════════════════════════════════════

class TestFraudModelLoading:

    def test_model_file_exists(self):
        """The fraud .pkl file must exist before anything else can work."""
        assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"

    def test_model_loads_without_error(self, model_obj):
        """Loading the .pkl should not raise any exception."""
        assert model_obj is not None

    def test_model_has_required_keys(self, model_obj):
        """The saved object must contain model, threshold, metrics, feature_columns."""
        for key in ["model", "threshold", "metrics", "feature_columns"]:
            assert key in model_obj, f"Missing key in model_obj: '{key}'"

    def test_threshold_is_valid(self, model_obj):
        """Threshold must be a probability between 0 and 1."""
        t = model_obj["threshold"]
        assert 0.0 < t < 1.0, f"Threshold out of range: {t}"

    def test_feature_columns_match(self, model_obj):
        """Feature columns saved with model must match fraud_features.py."""
        from src.fraud_features import FEATURE_COLUMNS
        assert model_obj["feature_columns"] == FEATURE_COLUMNS, (
            "Feature columns in saved model don't match FEATURE_COLUMNS in "
            "fraud_features.py. Retrain the model after changing features."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Group 2 — Model metrics on held-out test set
# ══════════════════════════════════════════════════════════════════════════════

class TestFraudModelMetrics:

    def test_precision_meets_target(self, model_obj, test_set):
        """Precision on the held-out test set must be >= 0.80."""
        model     = model_obj["model"]
        threshold = model_obj["threshold"]
        X_test, y_test = test_set

        proba     = model.predict_proba(X_test)[:, 1]
        preds     = (proba >= threshold).astype(int)
        precision = precision_score(y_test, preds, zero_division=0)

        assert precision >= MIN_PRECISION, (
            f"Precision {precision:.4f} is below target {MIN_PRECISION}. "
            "Retrain or tune the model."
        )

    def test_recall_meets_target(self, model_obj, test_set):
        """Recall on the held-out test set must be >= 0.70."""
        model     = model_obj["model"]
        threshold = model_obj["threshold"]
        X_test, y_test = test_set

        proba  = model.predict_proba(X_test)[:, 1]
        preds  = (proba >= threshold).astype(int)
        recall = recall_score(y_test, preds, zero_division=0)

        assert recall >= MIN_RECALL, (
            f"Recall {recall:.4f} is below target {MIN_RECALL}. "
            "Retrain or tune the model."
        )

    def test_auprc_meets_target(self, model_obj, test_set):
        """AUPRC on the held-out test set must be >= 0.75."""
        model  = model_obj["model"]
        X_test, y_test = test_set

        proba = model.predict_proba(X_test)[:, 1]
        auprc = average_precision_score(y_test, proba)

        assert auprc >= MIN_AUPRC, (
            f"AUPRC {auprc:.4f} is below target {MIN_AUPRC}. "
            "Retrain or tune the model."
        )

    def test_score_range_is_valid(self, model_obj, test_set):
        """Every predicted probability must be between 0.0 and 1.0."""
        model  = model_obj["model"]
        X_test, _ = test_set

        proba = model.predict_proba(X_test)[:, 1]
        assert np.all(proba >= 0.0), "Some scores are below 0.0"
        assert np.all(proba <= 1.0), "Some scores are above 1.0"


# ══════════════════════════════════════════════════════════════════════════════
# Group 3 — API health and model info
# ══════════════════════════════════════════════════════════════════════════════

class TestAPIHealth:

    def test_health_check_returns_ok(self, api_client):
        """GET / must return status ok."""
        response = api_client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_check_shows_both_models(self, api_client):
        """GET / must show both return and fraud models loaded."""
        response = api_client.get("/")
        data     = response.json()
        assert "models_loaded" in data, "Health check missing 'models_loaded' key"
        models = data["models_loaded"]
        assert "return_risk_scorer" in models,       "Return model not reported in health check"
        assert "fraud_transaction_scorer" in models, "Fraud model not reported in health check"

    def test_fraud_model_info_returns_200(self, api_client):
        """GET /model/fraud/info must return HTTP 200."""
        response = api_client.get("/model/fraud/info")
        assert response.status_code == 200

    def test_fraud_model_info_has_required_fields(self, api_client):
        """GET /model/fraud/info must contain all expected metric fields."""
        response = api_client.get("/model/fraud/info")
        data     = response.json()
        for field in ["model_name", "threshold", "precision", "recall",
                      "f1", "auc_roc", "auprc"]:
            assert field in data, f"Missing field in /model/fraud/info: '{field}'"

    def test_fraud_model_info_precision_meets_target(self, api_client):
        """Precision reported by /model/fraud/info must meet minimum target."""
        response = api_client.get("/model/fraud/info")
        assert response.json()["precision"] >= MIN_PRECISION

    def test_fraud_model_info_recall_meets_target(self, api_client):
        """Recall reported by /model/fraud/info must meet minimum target."""
        response = api_client.get("/model/fraud/info")
        assert response.json()["recall"] >= MIN_RECALL


# ══════════════════════════════════════════════════════════════════════════════
# Group 4 — Fraud scoring endpoint
# ══════════════════════════════════════════════════════════════════════════════

class TestFraudScoringEndpoint:

    def test_high_risk_profile_scores_high(self, api_client):
        """A clearly fraudulent transaction must be labelled High risk."""
        response = api_client.post("/score/transaction", json=HIGH_RISK_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_label"] == "High", (
            f"Expected High risk for fraudulent profile, got {data['risk_label']} "
            f"(score={data['risk_score']})"
        )

    def test_low_risk_profile_scores_low(self, api_client):
        """A clearly legitimate transaction must be labelled Low risk."""
        response = api_client.post("/score/transaction", json=LOW_RISK_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_label"] == "Low", (
            f"Expected Low risk for legitimate profile, got {data['risk_label']} "
            f"(score={data['risk_score']})"
        )

    def test_response_has_all_fields(self, api_client):
        """POST /score/transaction response must contain all required fields."""
        response = api_client.post("/score/transaction", json=HIGH_RISK_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        for field in ["risk_score", "risk_label", "recommendation",
                      "model_name", "threshold_used", "scored_at"]:
            assert field in data, f"Missing field in scoring response: '{field}'"

    def test_risk_score_is_between_0_and_1(self, api_client):
        """Risk score must always be a valid probability."""
        for payload in [HIGH_RISK_PAYLOAD, LOW_RISK_PAYLOAD]:
            response = api_client.post("/score/transaction", json=payload)
            score    = response.json()["risk_score"]
            assert 0.0 <= score <= 1.0, f"Risk score out of range: {score}"

    def test_recommendation_is_not_empty(self, api_client):
        """Recommendation field must never be blank."""
        response = api_client.post("/score/transaction", json=HIGH_RISK_PAYLOAD)
        assert response.json()["recommendation"].strip() != ""

    def test_invalid_payment_method_returns_422(self, api_client):
        """Sending an invalid payment_method must return HTTP 422."""
        bad_payload = {**HIGH_RISK_PAYLOAD, "payment_method": "bitcoin"}
        response    = api_client.post("/score/transaction", json=bad_payload)
        assert response.status_code == 422, (
            f"Expected 422 for invalid payment_method, got {response.status_code}"
        )

    def test_invalid_product_category_returns_422(self, api_client):
        """Sending an invalid product_category must return HTTP 422."""
        bad_payload = {**HIGH_RISK_PAYLOAD, "product_category": "spaceship"}
        response    = api_client.post("/score/transaction", json=bad_payload)
        assert response.status_code == 422

    def test_missing_required_field_returns_422(self, api_client):
        """Omitting a required field must return HTTP 422."""
        incomplete = {k: v for k, v in HIGH_RISK_PAYLOAD.items() if k != "order_value"}
        response   = api_client.post("/score/transaction", json=incomplete)
        assert response.status_code == 422

    def test_negative_order_value_returns_422(self, api_client):
        """Negative order value must be rejected with HTTP 422."""
        bad_payload = {**HIGH_RISK_PAYLOAD, "order_value": -100.0}
        response    = api_client.post("/score/transaction", json=bad_payload)
        assert response.status_code == 422

    def test_invalid_hour_of_day_returns_422(self, api_client):
        """Hour of day outside 0-23 must be rejected with HTTP 422."""
        bad_payload = {**HIGH_RISK_PAYLOAD, "hour_of_day": 25}
        response    = api_client.post("/score/transaction", json=bad_payload)
        assert response.status_code == 422
