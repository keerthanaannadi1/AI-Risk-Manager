"""
api/main.py
-----------
FastAPI application for the AI Risk Manager.

Endpoints:
  GET  /                        → Health check (all three models)
  GET  /model/return/info       → Return risk model metadata and metrics
  GET  /model/fraud/info        → Fraud model metadata and metrics
  GET  /model/chargeback/info   → Chargeback model metadata and metrics
  POST /score/return            → Score a return request (Phase 1)
  POST /score/transaction       → Score a payment transaction for fraud (Phase 2)
  POST /chargeback/analyze      → Analyze a chargeback dispute and return evidence report (Phase 3)

Usage:
    uvicorn api.main:app --reload
    # Then visit: http://127.0.0.1:8000/docs
"""

import os
import sys
import pickle
from datetime import datetime, timezone
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Phase 1 encodings
from src.features import (
    CATEGORY_ENCODING   as RETURN_CATEGORY_ENCODING,
    REASON_ENCODING     as RETURN_REASON_ENCODING,
    PAYMENT_ENCODING    as RETURN_PAYMENT_ENCODING,
    FEATURE_COLUMNS     as RETURN_FEATURE_COLUMNS,
)

# Phase 2 encodings
from src.fraud_features import (
    CATEGORY_ENCODING   as FRAUD_CATEGORY_ENCODING,
    PAYMENT_ENCODING    as FRAUD_PAYMENT_ENCODING,
    FEATURE_COLUMNS     as FRAUD_FEATURE_COLUMNS,
)

# Phase 3 encodings
from src.chargeback_features import (
    CATEGORY_ENCODING       as CB_CATEGORY_ENCODING,
    PAYMENT_ENCODING        as CB_PAYMENT_ENCODING,
    DISPUTE_REASON_ENCODING as CB_REASON_ENCODING,
    FEATURE_COLUMNS         as CB_FEATURE_COLUMNS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Model loading — all three models loaded once at server startup
# ══════════════════════════════════════════════════════════════════════════════

def _load_pkl(path: str, train_cmd: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}. Run `{train_cmd}` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# Phase 1 — Return Risk Scorer
return_model_obj  = _load_pkl(
    os.path.join(ROOT, "models", "return_risk_model.pkl"),
    "python src/train.py",
)
RETURN_MODEL      = return_model_obj["model"]
RETURN_THRESHOLD  = return_model_obj["threshold"]
RETURN_MODEL_NAME = return_model_obj["model_name"]
RETURN_METRICS    = return_model_obj["metrics"]

# Phase 2 — Fraud Transaction Scorer
fraud_model_obj  = _load_pkl(
    os.path.join(ROOT, "models", "fraud_risk_model.pkl"),
    "python src/fraud_train.py",
)
FRAUD_MODEL      = fraud_model_obj["model"]
FRAUD_THRESHOLD  = fraud_model_obj["threshold"]
FRAUD_MODEL_NAME = fraud_model_obj["model_name"]
FRAUD_METRICS    = fraud_model_obj["metrics"]

# Phase 3 — Chargeback Evidence Responder
cb_model_obj  = _load_pkl(
    os.path.join(ROOT, "models", "chargeback_model.pkl"),
    "python src/chargeback_train.py",
)
CB_MODEL      = cb_model_obj["model"]
CB_THRESHOLD  = cb_model_obj["threshold"]
CB_MODEL_NAME = cb_model_obj["model_name"]
CB_METRICS    = cb_model_obj["metrics"]


# ══════════════════════════════════════════════════════════════════════════════
# Risk label helpers
# ══════════════════════════════════════════════════════════════════════════════

def score_to_label(score: float) -> str:
    """Convert a probability to Low / Medium / High."""
    if score < 0.40:
        return "Low"
    elif score < 0.70:
        return "Medium"
    else:
        return "High"


def score_to_winability(score: float) -> str:
    """Convert a winability probability to Weak / Moderate / Strong."""
    if score < 0.40:
        return "Weak"
    elif score < 0.70:
        return "Moderate"
    else:
        return "Strong"


# Recommendations for return abuse (Phase 1)
RETURN_RECOMMENDATIONS = {
    "Low":    "Auto-approve the return. No further action needed.",
    "Medium": "Flag for manual review. Request supporting images if not provided.",
    "High":   "Hold return. Manual review required before processing.",
}

# Recommendations for fraud (Phase 2)
FRAUD_RECOMMENDATIONS = {
    "Low":    "Approve transaction. No further action needed.",
    "Medium": "Flag for review. Consider step-up authentication (OTP/2FA).",
    "High":   "Block transaction. Manual review required before processing.",
}

# Recommendations for chargeback winability (Phase 3)
CB_RECOMMENDATIONS = {
    "Strong":   "Strong case. Submit all available evidence immediately. Likely to win.",
    "Moderate": "Moderate case. Collect any missing evidence before submitting.",
    "Weak":     "Weak case. Consider settling with the customer to avoid chargeback fee.",
}

# Evidence field labels — human-readable names for each evidence boolean
EVIDENCE_LABELS = {
    "delivery_confirmed":       "Delivery confirmed (tracking number available)",
    "customer_signed_delivery": "Customer signed for delivery",
    "otp_used":                 "OTP / 2FA used during purchase",
    "ip_logs_available":        "IP address and device logs available",
    "order_confirmation_sent":  "Order confirmation sent to customer email/SMS",
    "refund_issued":            "Refund already issued to customer",
    "customer_contacted_support": "Customer contacted support before disputing",
}


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AI Risk Manager",
    description=(
        "ML-powered risk scoring for merchants on payment platforms. "
        "Detects return abuse (Phase 1), payment fraud (Phase 2), "
        "and chargeback winability (Phase 3). "
        "Defense-only system — scores and recommends, never blocks autonomously."
    ),
    version="3.0.0",
)


# ══════════════════════════════════════════════════════════════════════════════
# Shared response schemas
# ══════════════════════════════════════════════════════════════════════════════

class ScoreResponse(BaseModel):
    risk_score:     float = Field(..., description="Model confidence of risk (0.0 to 1.0)")
    risk_label:     str   = Field(..., description="Risk classification: Low, Medium, or High")
    recommendation: str   = Field(..., description="Suggested action for the merchant")
    model_name:     str   = Field(..., description="Name of the model that produced this score")
    threshold_used: float = Field(..., description="Decision threshold used for this prediction")
    scored_at:      str   = Field(..., description="ISO timestamp of when the score was produced")


class ModelInfoResponse(BaseModel):
    model_name:       str   = Field(..., description="Name of the trained model")
    threshold:        float = Field(..., description="Decision threshold")
    precision:        float
    recall:           float
    f1:               float
    auc_roc:          float
    auprc:            float
    confusion_matrix: dict


class ChargebackAnalysisResponse(BaseModel):
    winability_score:    float       = Field(..., description="Model confidence merchant will win (0.0 to 1.0)")
    winability_label:    str         = Field(..., description="Weak, Moderate, or Strong")
    recommendation:      str         = Field(..., description="What the merchant should do")
    evidence_score:      int         = Field(..., description="Count of evidence pieces available (0 to 7)")
    evidence_present:    List[str]   = Field(..., description="Evidence pieces that are available")
    evidence_missing:    List[str]   = Field(..., description="Evidence pieces that are missing")
    model_name:          str         = Field(..., description="Name of the model")
    threshold_used:      float       = Field(..., description="Decision threshold")
    scored_at:           str         = Field(..., description="ISO timestamp")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Return Risk Scorer request schema
# ══════════════════════════════════════════════════════════════════════════════

class ReturnRequest(BaseModel):
    customer_total_orders:     int   = Field(..., ge=0)
    customer_total_returns:    int   = Field(..., ge=0)
    customer_account_age_days: int   = Field(..., ge=0)
    customer_returns_30d:      int   = Field(..., ge=0)
    customer_orders_30d:       int   = Field(..., ge=0)
    customer_avg_order_value:  float = Field(..., ge=0)
    order_value:               float = Field(..., ge=0)
    payment_method:            str   = Field(..., description="cod, wallet, netbanking, upi, card")
    days_to_return:            int   = Field(..., ge=0)
    return_reason:             str   = Field(..., description="defective, wrong_item, quality_issue, changed_mind, not_needed")
    support_contacted:         bool
    images_submitted:          bool
    product_category:          str   = Field(..., description="books, home, apparel, electronics, luxury")
    merchant_return_rate:      float = Field(..., ge=0, le=1)
    return_rate_lifetime:      float = Field(..., ge=0, le=1)
    return_rate_30d:           float = Field(..., ge=0, le=1)
    is_near_deadline:          bool
    is_same_day_return:        bool
    is_first_order:            bool
    is_new_account:            bool
    no_images_high_value:      bool
    no_support_contact:        bool
    category_risk_score:       float = Field(..., ge=0, le=1)
    return_reason_risk:        float = Field(..., ge=0, le=1)
    order_value_normalized:    float = Field(..., ge=0)

    @field_validator("payment_method")
    @classmethod
    def validate_payment_method(cls, v):
        valid = list(RETURN_PAYMENT_ENCODING.keys())
        if v.lower() not in valid:
            raise ValueError(f"payment_method must be one of: {valid}")
        return v.lower()

    @field_validator("return_reason")
    @classmethod
    def validate_return_reason(cls, v):
        valid = list(RETURN_REASON_ENCODING.keys())
        if v.lower() not in valid:
            raise ValueError(f"return_reason must be one of: {valid}")
        return v.lower()

    @field_validator("product_category")
    @classmethod
    def validate_product_category(cls, v):
        valid = list(RETURN_CATEGORY_ENCODING.keys())
        if v.lower() not in valid:
            raise ValueError(f"product_category must be one of: {valid}")
        return v.lower()

    model_config = {"json_schema_extra": {"example": {
        "customer_total_orders": 12, "customer_total_returns": 8,
        "customer_account_age_days": 45, "customer_returns_30d": 3,
        "customer_orders_30d": 4, "customer_avg_order_value": 1800.0,
        "order_value": 3500.0, "payment_method": "cod",
        "days_to_return": 28, "return_reason": "changed_mind",
        "support_contacted": False, "images_submitted": False,
        "product_category": "electronics", "merchant_return_rate": 0.12,
        "return_rate_lifetime": 0.67, "return_rate_30d": 0.75,
        "is_near_deadline": True, "is_same_day_return": False,
        "is_first_order": False, "is_new_account": False,
        "no_images_high_value": True, "no_support_contact": True,
        "category_risk_score": 0.8, "return_reason_risk": 0.9,
        "order_value_normalized": 1.94,
    }}}


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Fraud Transaction Scorer request schema
# ══════════════════════════════════════════════════════════════════════════════

class TransactionRequest(BaseModel):
    order_value:               float = Field(..., ge=0)
    product_category:          str   = Field(..., description="electronics, apparel, books, home, luxury, gift_cards")
    payment_method:            str   = Field(..., description="card, upi, netbanking, wallet, cod")
    hour_of_day:               int   = Field(..., ge=0, le=23)
    day_of_week:               int   = Field(..., ge=0, le=6)
    failed_attempts:           int   = Field(..., ge=0)
    is_vpn:                    bool
    customer_account_age_days: int   = Field(..., ge=0)
    customer_total_past_orders: int  = Field(..., ge=0)
    customer_avg_order_value:  float = Field(..., ge=0)
    customer_past_fraud_flags: int   = Field(..., ge=0)
    customer_orders_1h:        int   = Field(..., ge=0)
    customer_orders_24h:       int   = Field(..., ge=0)
    merchant_fraud_rate:       float = Field(..., ge=0, le=1)
    is_night_transaction:      bool
    is_weekend:                bool
    is_new_device:             bool
    is_different_city:         bool
    is_high_value:             bool
    is_first_order:            bool
    is_new_account:            bool
    address_mismatch:          bool
    multiple_failed_attempts:  bool
    high_velocity_1h:          bool
    order_value_normalized:    float = Field(..., ge=0)
    new_account_high_value_card: bool
    payment_risk_score:        float = Field(..., ge=0, le=1)
    category_risk_score:       float = Field(..., ge=0, le=1)

    @field_validator("payment_method")
    @classmethod
    def validate_payment_method(cls, v):
        valid = list(FRAUD_PAYMENT_ENCODING.keys())
        if v.lower() not in valid:
            raise ValueError(f"payment_method must be one of: {valid}")
        return v.lower()

    @field_validator("product_category")
    @classmethod
    def validate_product_category(cls, v):
        valid = list(FRAUD_CATEGORY_ENCODING.keys())
        if v.lower() not in valid:
            raise ValueError(f"product_category must be one of: {valid}")
        return v.lower()

    model_config = {"json_schema_extra": {"example": {
        "order_value": 18000.0, "product_category": "electronics",
        "payment_method": "card", "hour_of_day": 3, "day_of_week": 2,
        "failed_attempts": 5, "is_vpn": True,
        "customer_account_age_days": 12, "customer_total_past_orders": 2,
        "customer_avg_order_value": 1500.0, "customer_past_fraud_flags": 0,
        "customer_orders_1h": 6, "customer_orders_24h": 8,
        "merchant_fraud_rate": 0.08,
        "is_night_transaction": True, "is_weekend": False,
        "is_new_device": True, "is_different_city": True,
        "is_high_value": True, "is_first_order": False,
        "is_new_account": True, "address_mismatch": True,
        "multiple_failed_attempts": True, "high_velocity_1h": True,
        "order_value_normalized": 12.0, "new_account_high_value_card": True,
        "payment_risk_score": 0.9, "category_risk_score": 0.8,
    }}}


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Chargeback Evidence Responder request schema
# ══════════════════════════════════════════════════════════════════════════════

class ChargebackRequest(BaseModel):
    # Transaction details
    order_value:      float = Field(..., ge=0, description="Order value in INR")
    product_category: str   = Field(..., description="electronics, apparel, books, home, luxury, gift_cards")
    payment_method:   str   = Field(..., description="card, upi, netbanking, wallet, cod")
    days_to_dispute:  int   = Field(..., ge=0, description="Days between transaction and chargeback dispute")
    delivery_days:    int   = Field(..., ge=0, description="Days between order and delivery")
    dispute_reason:   str   = Field(..., description="not_received, not_authorized, not_as_described, duplicate_charge, credit_not_processed")

    # Evidence available
    delivery_confirmed:       bool = Field(..., description="Is delivery confirmed with tracking?")
    customer_signed_delivery: bool = Field(..., description="Did customer sign for delivery?")
    otp_used:                 bool = Field(..., description="Was OTP/2FA used during purchase?")
    ip_logs_available:        bool = Field(..., description="Are IP address and device logs available?")
    order_confirmation_sent:  bool = Field(..., description="Was order confirmation sent to customer?")
    refund_issued:            bool = Field(..., description="Has a refund already been issued?")

    # Customer behaviour
    customer_contacted_support: bool  = Field(..., description="Did customer contact support before disputing?")
    customer_account_age_days:  int   = Field(..., ge=0)
    customer_total_orders:      int   = Field(..., ge=0)
    customer_past_disputes:     int   = Field(..., ge=0)
    customer_avg_order_value:   float = Field(..., ge=0)

    # Merchant
    merchant_chargeback_rate: float = Field(..., ge=0, le=1)

    # Engineered signals
    is_quick_dispute:       bool  = Field(..., description="Disputed within 3 days of transaction?")
    is_late_dispute:        bool  = Field(..., description="Disputed after 60 days?")
    is_high_value:          bool  = Field(..., description="Order value above Rs.5000?")
    is_first_order:         bool  = Field(..., description="Customer's first order?")
    is_new_account:         bool  = Field(..., description="Account less than 30 days old?")
    is_serial_disputer:     bool  = Field(..., description="Customer has 2+ past disputes?")
    is_cod:                 bool  = Field(..., description="Payment was cash on delivery?")
    evidence_score:         int   = Field(..., ge=0, le=7, description="Count of evidence pieces available (0-7)")
    dispute_difficulty:     float = Field(..., ge=0, le=1, description="Difficulty score of this dispute type (0.0 to 1.0)")
    order_value_normalized: float = Field(..., ge=0, description="Order value divided by customer avg order value")

    @field_validator("payment_method")
    @classmethod
    def validate_payment_method(cls, v):
        valid = list(CB_PAYMENT_ENCODING.keys())
        if v.lower() not in valid:
            raise ValueError(f"payment_method must be one of: {valid}")
        return v.lower()

    @field_validator("product_category")
    @classmethod
    def validate_product_category(cls, v):
        valid = list(CB_CATEGORY_ENCODING.keys())
        if v.lower() not in valid:
            raise ValueError(f"product_category must be one of: {valid}")
        return v.lower()

    @field_validator("dispute_reason")
    @classmethod
    def validate_dispute_reason(cls, v):
        valid = list(CB_REASON_ENCODING.keys())
        if v.lower() not in valid:
            raise ValueError(f"dispute_reason must be one of: {valid}")
        return v.lower()

    model_config = {"json_schema_extra": {"example": {
        "order_value": 8500.0, "product_category": "electronics",
        "payment_method": "card", "days_to_dispute": 5,
        "delivery_days": 3, "dispute_reason": "not_received",
        "delivery_confirmed": True, "customer_signed_delivery": False,
        "otp_used": True, "ip_logs_available": True,
        "order_confirmation_sent": True, "refund_issued": False,
        "customer_contacted_support": False,
        "customer_account_age_days": 400, "customer_total_orders": 15,
        "customer_past_disputes": 0, "customer_avg_order_value": 3000.0,
        "merchant_chargeback_rate": 0.04,
        "is_quick_dispute": True, "is_late_dispute": False,
        "is_high_value": True, "is_first_order": False,
        "is_new_account": False, "is_serial_disputer": False,
        "is_cod": False, "evidence_score": 4,
        "dispute_difficulty": 0.5, "order_value_normalized": 2.83,
    }}}


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", summary="Health check")
def root():
    """Check that the API is running and all three models are loaded."""
    return {
        "status": "ok",
        "models_loaded": {
            "return_risk_scorer":        RETURN_MODEL_NAME,
            "fraud_transaction_scorer":  FRAUD_MODEL_NAME,
            "chargeback_responder":      CB_MODEL_NAME,
        },
        "message": "AI Risk Manager is live. All three models ready.",
    }


@app.get("/model/return/info", response_model=ModelInfoResponse,
         summary="Return risk model metadata and metrics")
def return_model_info():
    return ModelInfoResponse(
        model_name=RETURN_MODEL_NAME, threshold=RETURN_THRESHOLD,
        precision=round(RETURN_METRICS["precision"], 4),
        recall=round(RETURN_METRICS["recall"], 4),
        f1=round(RETURN_METRICS["f1"], 4),
        auc_roc=round(RETURN_METRICS["auc_roc"], 4),
        auprc=round(RETURN_METRICS["auprc"], 4),
        confusion_matrix={
            "true_negative":  int(RETURN_METRICS["tn"]),
            "false_positive": int(RETURN_METRICS["fp"]),
            "false_negative": int(RETURN_METRICS["fn"]),
            "true_positive":  int(RETURN_METRICS["tp"]),
        },
    )


@app.get("/model/fraud/info", response_model=ModelInfoResponse,
         summary="Fraud model metadata and metrics")
def fraud_model_info():
    return ModelInfoResponse(
        model_name=FRAUD_MODEL_NAME, threshold=FRAUD_THRESHOLD,
        precision=round(FRAUD_METRICS["precision"], 4),
        recall=round(FRAUD_METRICS["recall"], 4),
        f1=round(FRAUD_METRICS["f1"], 4),
        auc_roc=round(FRAUD_METRICS["auc_roc"], 4),
        auprc=round(FRAUD_METRICS["auprc"], 4),
        confusion_matrix={
            "true_negative":  int(FRAUD_METRICS["tn"]),
            "false_positive": int(FRAUD_METRICS["fp"]),
            "false_negative": int(FRAUD_METRICS["fn"]),
            "true_positive":  int(FRAUD_METRICS["tp"]),
        },
    )


@app.get("/model/chargeback/info", response_model=ModelInfoResponse,
         summary="Chargeback model metadata and metrics")
def chargeback_model_info():
    return ModelInfoResponse(
        model_name=CB_MODEL_NAME, threshold=CB_THRESHOLD,
        precision=round(CB_METRICS["precision"], 4),
        recall=round(CB_METRICS["recall"], 4),
        f1=round(CB_METRICS["f1"], 4),
        auc_roc=round(CB_METRICS["auc_roc"], 4),
        auprc=round(CB_METRICS["auprc"], 4),
        confusion_matrix={
            "true_negative":  int(CB_METRICS["tn"]),
            "false_positive": int(CB_METRICS["fp"]),
            "false_negative": int(CB_METRICS["fn"]),
            "true_positive":  int(CB_METRICS["tp"]),
        },
    )


@app.post("/score/return", response_model=ScoreResponse,
          summary="Score a return request (Phase 1)")
def score_return(request: ReturnRequest):
    """Score a return request as Low / Medium / High risk."""
    try:
        row = {
            "customer_total_orders":     request.customer_total_orders,
            "customer_total_returns":    request.customer_total_returns,
            "customer_account_age_days": request.customer_account_age_days,
            "customer_returns_30d":      request.customer_returns_30d,
            "customer_orders_30d":       request.customer_orders_30d,
            "customer_avg_order_value":  request.customer_avg_order_value,
            "order_value":               request.order_value,
            "payment_method":            RETURN_PAYMENT_ENCODING[request.payment_method],
            "days_to_return":            request.days_to_return,
            "return_reason":             RETURN_REASON_ENCODING[request.return_reason],
            "support_contacted":         int(request.support_contacted),
            "images_submitted":          int(request.images_submitted),
            "product_category":          RETURN_CATEGORY_ENCODING[request.product_category],
            "merchant_return_rate":      request.merchant_return_rate,
            "return_rate_lifetime":      request.return_rate_lifetime,
            "return_rate_30d":           request.return_rate_30d,
            "is_near_deadline":          int(request.is_near_deadline),
            "is_same_day_return":        int(request.is_same_day_return),
            "is_first_order":            int(request.is_first_order),
            "is_new_account":            int(request.is_new_account),
            "no_images_high_value":      int(request.no_images_high_value),
            "no_support_contact":        int(request.no_support_contact),
            "category_risk_score":       request.category_risk_score,
            "return_reason_risk":        request.return_reason_risk,
            "order_value_normalized":    request.order_value_normalized,
        }
        X          = pd.DataFrame([row])[RETURN_FEATURE_COLUMNS]
        risk_score = float(RETURN_MODEL.predict_proba(X)[0][1])
        risk_label = score_to_label(risk_score)
        return ScoreResponse(
            risk_score=round(risk_score, 4), risk_label=risk_label,
            recommendation=RETURN_RECOMMENDATIONS[risk_label],
            model_name=RETURN_MODEL_NAME, threshold_used=RETURN_THRESHOLD,
            scored_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.post("/score/transaction", response_model=ScoreResponse,
          summary="Score a payment transaction for fraud (Phase 2)")
def score_transaction(request: TransactionRequest):
    """Score a payment transaction as Low / Medium / High fraud risk."""
    try:
        row = {
            "order_value":               request.order_value,
            "product_category":          FRAUD_CATEGORY_ENCODING[request.product_category],
            "payment_method":            FRAUD_PAYMENT_ENCODING[request.payment_method],
            "hour_of_day":               request.hour_of_day,
            "day_of_week":               request.day_of_week,
            "failed_attempts":           request.failed_attempts,
            "is_vpn":                    int(request.is_vpn),
            "customer_account_age_days": request.customer_account_age_days,
            "customer_total_past_orders": request.customer_total_past_orders,
            "customer_avg_order_value":  request.customer_avg_order_value,
            "customer_past_fraud_flags": request.customer_past_fraud_flags,
            "customer_orders_1h":        request.customer_orders_1h,
            "customer_orders_24h":       request.customer_orders_24h,
            "merchant_fraud_rate":       request.merchant_fraud_rate,
            "is_night_transaction":      int(request.is_night_transaction),
            "is_weekend":                int(request.is_weekend),
            "is_new_device":             int(request.is_new_device),
            "is_different_city":         int(request.is_different_city),
            "is_high_value":             int(request.is_high_value),
            "is_first_order":            int(request.is_first_order),
            "is_new_account":            int(request.is_new_account),
            "address_mismatch":          int(request.address_mismatch),
            "multiple_failed_attempts":  int(request.multiple_failed_attempts),
            "high_velocity_1h":          int(request.high_velocity_1h),
            "order_value_normalized":    request.order_value_normalized,
            "new_account_high_value_card": int(request.new_account_high_value_card),
            "payment_risk_score":        request.payment_risk_score,
            "category_risk_score":       request.category_risk_score,
        }
        X          = pd.DataFrame([row])[FRAUD_FEATURE_COLUMNS]
        risk_score = float(FRAUD_MODEL.predict_proba(X)[0][1])
        risk_label = score_to_label(risk_score)
        return ScoreResponse(
            risk_score=round(risk_score, 4), risk_label=risk_label,
            recommendation=FRAUD_RECOMMENDATIONS[risk_label],
            model_name=FRAUD_MODEL_NAME, threshold_used=FRAUD_THRESHOLD,
            scored_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.post("/chargeback/analyze", response_model=ChargebackAnalysisResponse,
          summary="Analyze a chargeback dispute and return evidence report (Phase 3)")
def analyze_chargeback(request: ChargebackRequest):
    """
    Analyze a chargeback dispute and return:
    - Winability score and label (Weak / Moderate / Strong)
    - Recommendation (fight, collect more evidence, or settle)
    - Evidence checklist (what's present and what's missing)

    Call this endpoint when a bank raises a chargeback dispute.
    Use the evidence checklist to decide what to submit to the bank.
    """
    try:
        row = {
            "order_value":               request.order_value,
            "product_category":          CB_CATEGORY_ENCODING[request.product_category],
            "payment_method":            CB_PAYMENT_ENCODING[request.payment_method],
            "days_to_dispute":           request.days_to_dispute,
            "delivery_days":             request.delivery_days,
            "dispute_reason":            CB_REASON_ENCODING[request.dispute_reason],
            "delivery_confirmed":        int(request.delivery_confirmed),
            "customer_signed_delivery":  int(request.customer_signed_delivery),
            "otp_used":                  int(request.otp_used),
            "ip_logs_available":         int(request.ip_logs_available),
            "order_confirmation_sent":   int(request.order_confirmation_sent),
            "refund_issued":             int(request.refund_issued),
            "customer_contacted_support": int(request.customer_contacted_support),
            "customer_account_age_days": request.customer_account_age_days,
            "customer_total_orders":     request.customer_total_orders,
            "customer_past_disputes":    request.customer_past_disputes,
            "customer_avg_order_value":  request.customer_avg_order_value,
            "merchant_chargeback_rate":  request.merchant_chargeback_rate,
            "is_quick_dispute":          int(request.is_quick_dispute),
            "is_late_dispute":           int(request.is_late_dispute),
            "is_high_value":             int(request.is_high_value),
            "is_first_order":            int(request.is_first_order),
            "is_new_account":            int(request.is_new_account),
            "is_serial_disputer":        int(request.is_serial_disputer),
            "is_cod":                    int(request.is_cod),
            "evidence_score":            request.evidence_score,
            "dispute_difficulty":        request.dispute_difficulty,
            "order_value_normalized":    request.order_value_normalized,
        }

        X                = pd.DataFrame([row])[CB_FEATURE_COLUMNS]
        winability_score = float(CB_MODEL.predict_proba(X)[0][1])
        winability_label = score_to_winability(winability_score)
        recommendation   = CB_RECOMMENDATIONS[winability_label]

        # ── Build evidence checklist ───────────────────────────────────────────
        evidence_flags = {
            "delivery_confirmed":         request.delivery_confirmed,
            "customer_signed_delivery":   request.customer_signed_delivery,
            "otp_used":                   request.otp_used,
            "ip_logs_available":          request.ip_logs_available,
            "order_confirmation_sent":    request.order_confirmation_sent,
            "refund_issued":              request.refund_issued,
            "customer_contacted_support": request.customer_contacted_support,
        }

        evidence_present = [
            EVIDENCE_LABELS[k] for k, v in evidence_flags.items() if v
        ]
        evidence_missing = [
            EVIDENCE_LABELS[k] for k, v in evidence_flags.items() if not v
        ]

        return ChargebackAnalysisResponse(
            winability_score=round(winability_score, 4),
            winability_label=winability_label,
            recommendation=recommendation,
            evidence_score=len(evidence_present),
            evidence_present=evidence_present,
            evidence_missing=evidence_missing,
            model_name=CB_MODEL_NAME,
            threshold_used=CB_THRESHOLD,
            scored_at=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
