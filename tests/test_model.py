"""
tests/test_model.py
-------------------
Automated tests for Phase 1: Return Risk Scorer.

What is tested:
  1. Model loading        — saved .pkl file loads without errors
  2. Precision target     — precision >= 0.70 on the held-out test set
  3. Recall target        — recall >= 0.65 on the held-out test set
  4. AUPRC target         — AUPRC >= 0.75 on the held-out test set
  5. Score range          — every prediction is a probability between 0.0 and 1.0
  6. API health check     — GET / returns status ok
  7. API model info       — GET /model/info returns correct metric keys
  8. High-risk scoring    — a clearly abusive profile scores as High
  9. Low-risk scoring     — a clearly legitimate profile scores as Low
  10. Invalid input       — bad field value returns HTTP 422, not a crash

How the test set is reconstructed (no new data needed):
  - Load the same returns.csv used during training
  - Re-split with the exact same seed (random_state=42) used in train.py
  - This gives the same 1,000 test rows every time — deterministic

Run with:
    pytest tests/test_model.py -v
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

from src.features import build_features
from api.main import app

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(ROOT, "data", "raw", "returns.csv")
MODEL_PATH = os.path.join(ROOT, "models", "return_risk_model.pkl")

# ── Minimum targets (from docs/evaluation-strategy.md) ────────────────────────
MIN_PRECISION = 0.70
MIN_RECALL    = 0.65
MIN_AUPRC     = 0.75

# ── Decision threshold buckets (must match api/main.py) ───────────────────────
HIGH_RISK_THRESHOLD   = 0.70   # score >= this → High
LOW_RISK_THRESHOLD    = 0.40   # score <  this → Low


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures — shared setup loaded once for the whole test session
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def model_obj():
    """Load the saved model package from disk once for all tests."""
    assert os.path.exists(MODEL_PATH), (
        f"Model file not found at {MODEL_PATH}. Run `python src/train.py` first."
    )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def test_set():
    """
    Reconstruct the held-out test set using the same split as train.py.
    Same seed (42) = same 1,000 rows every time.
    """
    assert os.path.exists(DATA_PATH), (
        f"Data file not found at {DATA_PATH}. Run `python data/generate_data.py` first."
    )
    df = pd.read_csv(DATA_PATH)
    X, y = build_features(df)

    # Step 1: carve out 10% test set  (mirrors train.py)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42
    )
    return X_test, y_test


@pytest.fixture(scope="session")
def api_client():
    """Create a FastAPI test client (no real server needed)."""
    return TestClient(app)


# ══════════════════════════════════════════════════════════════════════════════
# Group 1 — Model tests
# ══════════════════════════════════════════════════════════════════════════════

class TestModelLoading:

    def test_model_file_exists(self):
        """The .pkl file must exist before anything else can work."""
        assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"

    def test_model_loads_without_error(self, model_obj):
        """Loading the .pkl should not raise any exception."""
        assert model_obj is not None

    def test_model_has_required_keys(self, model_obj):
        """The saved object must contain model, threshold, metrics, feature_columns."""
        required_keys = ["model", "threshold", "metrics", "feature_columns"]
        for key in required_keys:
            assert key in model_obj, f"Missing key in model_obj: '{key}'"

    def test_threshold_is_valid(self, model_obj):
        """Threshold must be a probability between 0 and 1."""
        t = model_obj["threshold"]
        assert 0.0 < t < 1.0, f"Threshold out of range: {t}"

    def test_feature_columns_match(self, model_obj):
        """Feature columns saved with the model must match what features.py defines."""
        from src.features import FEATURE_COLUMNS
        assert model_obj["feature_columns"] == FEATURE_COLUMNS, (
            "Feature columns in saved model don't match FEATURE_COLUMNS in features.py. "
            "Retrain the model after changing features."
        )


class TestModelMetrics:

    def test_precision_meets_target(self, model_obj, test_set):
        """Precision on the held-out test set must be >= 0.70."""
        model     = model_obj["model"]
        threshold = model_obj["threshold"]
        X_test, y_test = test_set

        proba  = model.predict_proba(X_test)[:, 1]
        preds  = (proba >= threshold).astype(int)
        precision = precision_score(y_test, preds, zero_division=0)

        assert precision >= MIN_PRECISION, (
            f"Precision {precision:.4f} is below minimum target {MIN_PRECISION}. "
            "Retrain or tune the model."
        )

    def test_recall_meets_target(self, model_obj, test_set):
        """Recall on the held-out test set must be >= 0.65."""
        model     = model_obj["model"]
        threshold = model_obj["threshold"]
        X_test, y_test = test_set

        proba  = model.predict_proba(X_test)[:, 1]
        preds  = (proba >= threshold).astype(int)
        recall = recall_score(y_test, preds, zero_division=0)

        assert recall >= MIN_RECALL, (
            f"Recall {recall:.4f} is below minimum target {MIN_RECALL}. "
            "Retrain or tune the model."
        )

    def test_auprc_meets_target(self, model_obj, test_set):
        """AUPRC on the held-out test set must be >= 0.75."""
        model  = model_obj["model"]
        X_test, y_test = test_set

        proba = model.predict_proba(X_test)[:, 1]
        auprc = average_precision_score(y_test, proba)

        assert auprc >= MIN_AUPRC, (
            f"AUPRC {auprc:.4f} is below minimum target {MIN_AUPRC}. "
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
# Group 2 — API tests
# ══════════════════════════════════════════════════════════════════════════════

# ── Reusable test payloads ─────────────────────────────────────────────────────

# Clearly abusive: serial returner, no images, no support, high value, COD,
# near deadline, new account, high-risk category
HIGH_RISK_PAYLOAD = {
    "customer_total_orders":     10,
    "customer_total_returns":    9,        # 90% return rate
    "customer_account_age_days": 20,       # new account
    "customer_returns_30d":      4,
    "customer_orders_30d":       5,
    "customer_avg_order_value":  2000.0,
    "order_value":               12000.0,  # high value
    "payment_method":            "cod",
    "days_to_return":            29,       # near deadline
    "return_reason":             "changed_mind",
    "support_contacted":         False,
    "images_submitted":          False,    # no images on high-value item
    "product_category":          "electronics",
    "merchant_return_rate":      0.15,
    "return_rate_lifetime":      0.90,
    "return_rate_30d":           0.80,
    "is_near_deadline":          True,
    "is_same_day_return":        False,
    "is_first_order":            False,
    "is_new_account":            True,
    "no_images_high_value":      True,
    "no_support_contact":        True,
    "category_risk_score":       0.8,
    "return_reason_risk":        0.7,
    "order_value_normalized":    6.0,
}

# Clearly legitimate: long history, low return rate, images submitted,
# contacted support, normal order value
LOW_RISK_PAYLOAD = {
    "customer_total_orders":     80,
    "customer_total_returns":    4,         # 5% return rate
    "customer_account_age_days": 900,       # old account
    "customer_returns_30d":      0,
    "customer_orders_30d":       3,
    "customer_avg_order_value":  1500.0,
    "order_value":               800.0,     # normal value
    "payment_method":            "upi",
    "days_to_return":            5,         # returned early, not near deadline
    "return_reason":             "defective",
    "support_contacted":         True,
    "images_submitted":          True,
    "product_category":          "books",
    "merchant_return_rate":      0.06,
    "return_rate_lifetime":      0.05,
    "return_rate_30d":           0.00,
    "is_near_deadline":          False,
    "is_same_day_return":        False,
    "is_first_order":            False,
    "is_new_account":            False,
    "no_images_high_value":      False,
    "no_support_contact":        False,
    "category_risk_score":       0.2,
    "return_reason_risk":        0.3,
    "order_value_normalized":    0.53,
}


class TestAPIHealth:

    def test_health_check_returns_ok(self, api_client):
        """GET / must return status ok."""
        response = api_client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_check_has_model_name(self, api_client):
        """GET / must report which models are loaded."""
        response = api_client.get("/")
        data = response.json()
        assert "models_loaded" in data
        assert "return_risk_scorer" in data["models_loaded"]


class TestAPIModelInfo:

    def test_model_info_returns_200(self, api_client):
        """GET /model/return/info must return HTTP 200."""
        response = api_client.get("/model/return/info")
        assert response.status_code == 200

    def test_model_info_has_required_fields(self, api_client):
        """GET /model/return/info must contain all expected metric fields."""
        response  = api_client.get("/model/return/info")
        data      = response.json()
        required  = ["model_name", "threshold", "precision", "recall", "f1", "auc_roc", "auprc"]
        for field in required:
            assert field in data, f"Missing field in /model/return/info response: '{field}'"

    def test_model_info_precision_meets_target(self, api_client):
        """Precision reported by /model/return/info must meet the minimum target."""
        response = api_client.get("/model/return/info")
        assert response.json()["precision"] >= MIN_PRECISION

    def test_model_info_recall_meets_target(self, api_client):
        """Recall reported by /model/return/info must meet the minimum target."""
        response = api_client.get("/model/return/info")
        assert response.json()["recall"] >= MIN_RECALL


class TestAPIScoringEndpoint:

    def test_high_risk_profile_scores_high(self, api_client):
        """A clearly abusive return profile must be labelled High risk."""
        response = api_client.post("/score/return", json=HIGH_RISK_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_label"] == "High", (
            f"Expected High risk for abusive profile, got {data['risk_label']} "
            f"(score={data['risk_score']})"
        )

    def test_low_risk_profile_scores_low(self, api_client):
        """A clearly legitimate return profile must be labelled Low risk."""
        response = api_client.post("/score/return", json=LOW_RISK_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_label"] == "Low", (
            f"Expected Low risk for legitimate profile, got {data['risk_label']} "
            f"(score={data['risk_score']})"
        )

    def test_response_has_all_fields(self, api_client):
        """POST /score/return response must contain all required fields."""
        response = api_client.post("/score/return", json=HIGH_RISK_PAYLOAD)
        assert response.status_code == 200
        data     = response.json()
        required = ["risk_score", "risk_label", "recommendation", "model_name",
                    "threshold_used", "scored_at"]
        for field in required:
            assert field in data, f"Missing field in scoring response: '{field}'"

    def test_risk_score_is_between_0_and_1(self, api_client):
        """Risk score in the response must always be a valid probability."""
        for payload in [HIGH_RISK_PAYLOAD, LOW_RISK_PAYLOAD]:
            response = api_client.post("/score/return", json=payload)
            score    = response.json()["risk_score"]
            assert 0.0 <= score <= 1.0, f"Risk score out of range: {score}"

    def test_recommendation_is_not_empty(self, api_client):
        """Recommendation field must never be blank."""
        response = api_client.post("/score/return", json=HIGH_RISK_PAYLOAD)
        assert response.json()["recommendation"].strip() != ""

    def test_invalid_payment_method_returns_422(self, api_client):
        """Sending an invalid payment_method must return HTTP 422, not a crash."""
        bad_payload = {**HIGH_RISK_PAYLOAD, "payment_method": "bitcoin"}
        response    = api_client.post("/score/return", json=bad_payload)
        assert response.status_code == 422, (
            f"Expected 422 for invalid payment_method, got {response.status_code}"
        )

    def test_invalid_return_reason_returns_422(self, api_client):
        """Sending an invalid return_reason must return HTTP 422."""
        bad_payload = {**HIGH_RISK_PAYLOAD, "return_reason": "i_felt_like_it"}
        response    = api_client.post("/score/return", json=bad_payload)
        assert response.status_code == 422

    def test_missing_required_field_returns_422(self, api_client):
        """Omitting a required field must return HTTP 422."""
        incomplete = {k: v for k, v in HIGH_RISK_PAYLOAD.items() if k != "order_value"}
        response   = api_client.post("/score/return", json=incomplete)
        assert response.status_code == 422

    def test_negative_order_value_returns_422(self, api_client):
        """Negative order value must be rejected with HTTP 422."""
        bad_payload = {**HIGH_RISK_PAYLOAD, "order_value": -500.0}
        response    = api_client.post("/score/return", json=bad_payload)
        assert response.status_code == 422
