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

import logging
import os
import sys
import pickle
import time
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai_risk_manager")

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

# Feedback & retraining pipeline
from data.outcomes.feedback_store import (
    write_feedback,
    read_feedback,
    feedback_summary,
    VALID_OUTCOMES,
)



# API key authentication

API_KEY        = os.environ.get("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

if not API_KEY:
    logger.warning(
        "API_KEY environment variable is not set. "
        "Server is running in open mode — all requests will be accepted. "
        "Set API_KEY before deploying."
    )


def verify_api_key(key: Optional[str] = Security(api_key_header)) -> None:
    """Reject requests with a missing or wrong API key (when API_KEY is set)."""
    if not API_KEY:
        return  # open mode — no key required
    if key != API_KEY:
        logger.warning("Rejected request — invalid or missing API key")
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")



# Model loading — all three models loaded once at server startup


def _resolve_model_path(env_var: str, default_relative: str) -> str:
    """Return model path from env variable, or fall back to default."""
    return os.environ.get(env_var, os.path.join(ROOT, default_relative))


def _load_pkl(path: str, train_cmd: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}. Run `{train_cmd}` first."
        )
    logger.info(f"Loading model from {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# Phase 1 — Return Risk Scorer
return_model_obj  = _load_pkl(
    _resolve_model_path("RETURN_MODEL_PATH", "models/return_risk_model.pkl"),
    "python src/train.py",
)
RETURN_MODEL      = return_model_obj["model"]
RETURN_THRESHOLD  = return_model_obj["threshold"]
RETURN_MODEL_NAME = return_model_obj["model_name"]
RETURN_METRICS    = return_model_obj["metrics"]
logger.info(f"Return risk model loaded: {RETURN_MODEL_NAME}  threshold={RETURN_THRESHOLD}")

# Phase 2 — Fraud Transaction Scorer
fraud_model_obj  = _load_pkl(
    _resolve_model_path("FRAUD_MODEL_PATH", "models/fraud_risk_model.pkl"),
    "python src/fraud_train.py",
)
FRAUD_MODEL      = fraud_model_obj["model"]
FRAUD_THRESHOLD  = fraud_model_obj["threshold"]
FRAUD_MODEL_NAME = fraud_model_obj["model_name"]
FRAUD_METRICS    = fraud_model_obj["metrics"]
logger.info(f"Fraud model loaded:       {FRAUD_MODEL_NAME}  threshold={FRAUD_THRESHOLD}")

# Phase 3 — Chargeback Evidence Responder
cb_model_obj  = _load_pkl(
    _resolve_model_path("CHARGEBACK_MODEL_PATH", "models/chargeback_model.pkl"),
    "python src/chargeback_train.py",
)
CB_MODEL      = cb_model_obj["model"]
CB_THRESHOLD  = cb_model_obj["threshold"]
CB_MODEL_NAME = cb_model_obj["model_name"]
CB_METRICS    = cb_model_obj["metrics"]
logger.info(f"Chargeback model loaded:  {CB_MODEL_NAME}  threshold={CB_THRESHOLD}")



# Risk label helpers


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



# FastAPI app


app = FastAPI(
    title="AI Risk Manager",
    description=(
        "ML-powered risk scoring for merchants on payment platforms.\n\n"
        "## What This API Does\n"
        "Detects three types of merchant losses:\n"
        "- **Return Abuse** — wardrobing, swap fraud, serial returners\n"
        "- **Payment Fraud** — stolen cards, account takeover, velocity abuse\n"
        "- **Chargebacks** — scores dispute winability and provides evidence checklist\n\n"
        "## Design Philosophy\n"
        "- **Defense-only** — scores and recommends, never blocks autonomously\n"
        "- **Honest evaluation** — precision, recall, AUPRC on held-out test sets\n"
        "- **No accuracy metric** — useless on imbalanced fraud data\n\n"
        "## Authentication\n"
        "Set `API_KEY` environment variable to enable authentication.\n"
        "If not set, server runs in open mode (no key required).\n\n"
        "## Quick Start\n"
        "1. Send a request to any scoring endpoint\n"
        "2. Receive risk score, label, and recommendation\n"
        "3. Use the recommendation to decide next action"
    ),
    version="3.0.0",
    openapi_tags=[
        {
            "name": "Health",
            "description": "System health checks and model status",
        },
        {
            "name": "Return Risk",
            "description": "Score return requests for abuse risk (Phase 1)",
        },
        {
            "name": "Fraud Detection",
            "description": "Score payment transactions for fraud risk (Phase 2)",
        },
        {
            "name": "Chargeback Analysis",
            "description": "Analyze chargeback disputes and evidence (Phase 3)",
        },
        {
            "name": "Model Info",
            "description": "View model metrics and thresholds",
        },
        {
            "name": "Feedback",
            "description": "Submit confirmed outcomes for retraining",
        },
    ],
)


# ── Request logging middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming request with method, path, status code, and latency."""
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method}  {request.url.path}  "
        f"status={response.status_code}  latency={latency_ms:.1f}ms"
    )
    return response



# Shared response schemas


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



# Phase 1 — Return Risk Scorer request schema


# Risk score lookup tables (same values used during data generation)
RETURN_CATEGORY_RISK = {
    "electronics": 0.8,
    "apparel":     0.6,
    "books":       0.2,
    "home":        0.4,
    "luxury":      0.9,
}

RETURN_REASON_RISK = {
    "defective":     0.3,
    "wrong_item":    0.4,
    "not_needed":    0.8,
    "quality_issue": 0.5,
    "changed_mind":  0.7,
}

# Order value above this is considered "high value" for no_images_high_value flag
HIGH_VALUE_THRESHOLD = 1800.0
# Account younger than this (days) is considered "new"
NEW_ACCOUNT_THRESHOLD = 30
# Return window (days) — return within last 3 days of this is "near deadline"
RETURN_WINDOW_DAYS = 30


class ReturnRequest(BaseModel):
    # Raw fields only — the API computes all derived features internally
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
    }}}



# Phase 2 — Fraud Transaction Scorer request schema


# Risk score lookup tables for fraud (same values used during data generation)
FRAUD_CATEGORY_RISK = {
    "electronics": 0.8,
    "gift_cards":  0.9,
    "luxury":      0.85,
    "apparel":     0.4,
    "books":       0.1,
    "home":        0.3,
}

FRAUD_PAYMENT_RISK = {
    "card":       0.9,
    "upi":        0.3,
    "netbanking": 0.2,
    "wallet":     0.4,
    "cod":        0.5,
}

# Thresholds for fraud feature computation
FRAUD_HIGH_VALUE_THRESHOLD    = 10000.0  # order above this is high value
FRAUD_NEW_ACCOUNT_THRESHOLD   = 30       # account age in days
FRAUD_NIGHT_HOURS             = (0, 6)   # hours considered night (0am to 6am)
FRAUD_HIGH_VELOCITY_THRESHOLD = 3        # orders in 1h to flag high velocity
FRAUD_MULTIPLE_FAILS          = 2        # failed attempts to flag as multiple


class TransactionRequest(BaseModel):
    # Raw fields only — the API computes all derived features internally
    order_value:               float = Field(..., ge=0)
    product_category:          str   = Field(..., description="electronics, apparel, books, home, luxury, gift_cards")
    payment_method:            str   = Field(..., description="card, upi, netbanking, wallet, cod")
    hour_of_day:               int   = Field(..., ge=0, le=23)
    day_of_week:               int   = Field(..., ge=0, le=6, description="0=Monday, 6=Sunday")
    failed_attempts:           int   = Field(..., ge=0)
    is_vpn:                    bool
    customer_account_age_days: int   = Field(..., ge=0)
    customer_total_past_orders: int  = Field(..., ge=0)
    customer_avg_order_value:  float = Field(..., ge=0)
    customer_past_fraud_flags: int   = Field(..., ge=0)
    customer_orders_1h:        int   = Field(..., ge=0)
    customer_orders_24h:       int   = Field(..., ge=0)
    merchant_fraud_rate:       float = Field(..., ge=0, le=1)
    is_new_device:             bool
    is_different_city:         bool
    address_mismatch:          bool

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
        "is_new_device": True, "is_different_city": True,
        "address_mismatch": True,
    }}}



# Phase 3 — Chargeback Evidence Responder request schema


# Dispute difficulty lookup (same values used during data generation)
CB_DISPUTE_DIFFICULTY = {
    "not_received":          0.5,
    "not_authorized":        0.8,
    "not_as_described":      0.6,
    "duplicate_charge":      0.2,
    "credit_not_processed":  0.4,
}

# Thresholds for chargeback feature computation
CB_HIGH_VALUE_THRESHOLD   = 5000.0  # order above this is high value
CB_NEW_ACCOUNT_THRESHOLD  = 30      # account age in days
CB_QUICK_DISPUTE_DAYS     = 3       # disputed within this many days = quick
CB_LATE_DISPUTE_DAYS      = 60      # disputed after this many days = late
CB_SERIAL_DISPUTER_COUNT  = 2       # past disputes >= this = serial disputer


class ChargebackRequest(BaseModel):
    # Raw fields only — the API computes all derived features internally

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
    }}}



# Routes

@app.get("/", summary="Health check", tags=["Health"])
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
         summary="Return risk model metadata and metrics", tags=["Model Info"])
def return_model_info(_: None = Security(verify_api_key)):
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
         summary="Fraud model metadata and metrics", tags=["Model Info"])
def fraud_model_info(_: None = Security(verify_api_key)):
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
         summary="Chargeback model metadata and metrics", tags=["Model Info"])
def chargeback_model_info(_: None = Security(verify_api_key)):
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
          summary="Score a return request (Phase 1)", tags=["Return Risk"])
def score_return(request: ReturnRequest, _: None = Security(verify_api_key)):
    """Score a return request as Low / Medium / High risk."""
    try:
        # ── Compute derived features from raw inputs ───────────────────────────
        total_orders  = request.customer_total_orders
        total_returns = request.customer_total_returns

        return_rate_lifetime = (
            total_returns / total_orders if total_orders > 0 else 0.0
        )
        return_rate_30d = (
            request.customer_returns_30d / request.customer_orders_30d
            if request.customer_orders_30d > 0 else 0.0
        )
        is_near_deadline    = request.days_to_return >= (RETURN_WINDOW_DAYS - 3)
        is_same_day_return  = request.days_to_return == 0
        is_first_order      = total_orders <= 1
        is_new_account      = request.customer_account_age_days < NEW_ACCOUNT_THRESHOLD
        no_images_high_value = (
            not request.images_submitted and request.order_value > HIGH_VALUE_THRESHOLD
        )
        no_support_contact   = not request.support_contacted
        category_risk_score  = RETURN_CATEGORY_RISK[request.product_category]
        return_reason_risk   = RETURN_REASON_RISK[request.return_reason]
        order_value_normalized = (
            request.order_value / request.customer_avg_order_value
            if request.customer_avg_order_value > 0 else 1.0
        )

        row = {
            "customer_total_orders":     total_orders,
            "customer_total_returns":    total_returns,
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
            "return_rate_lifetime":      return_rate_lifetime,
            "return_rate_30d":           return_rate_30d,
            "is_near_deadline":          int(is_near_deadline),
            "is_same_day_return":        int(is_same_day_return),
            "is_first_order":            int(is_first_order),
            "is_new_account":            int(is_new_account),
            "no_images_high_value":      int(no_images_high_value),
            "no_support_contact":        int(no_support_contact),
            "category_risk_score":       category_risk_score,
            "return_reason_risk":        return_reason_risk,
            "order_value_normalized":    order_value_normalized,
        }
        X          = pd.DataFrame([row])[RETURN_FEATURE_COLUMNS]
        risk_score = float(RETURN_MODEL.predict_proba(X)[0][1])
        risk_label = score_to_label(risk_score)
        logger.info(f"score/return  score={risk_score:.4f}  label={risk_label}")
        return ScoreResponse(
            risk_score=round(risk_score, 4), risk_label=risk_label,
            recommendation=RETURN_RECOMMENDATIONS[risk_label],
            model_name=RETURN_MODEL_NAME, threshold_used=RETURN_THRESHOLD,
            scored_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"score/return failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.post("/score/transaction", response_model=ScoreResponse,
          summary="Score a payment transaction for fraud (Phase 2)", tags=["Fraud Detection"])
def score_transaction(request: TransactionRequest, _: None = Security(verify_api_key)):
    """Score a payment transaction as Low / Medium / High fraud risk."""
    try:
        # ── Compute derived features from raw inputs ───────────────────────────
        is_night_transaction     = FRAUD_NIGHT_HOURS[0] <= request.hour_of_day < FRAUD_NIGHT_HOURS[1]
        is_weekend               = request.day_of_week >= 5
        is_high_value            = request.order_value > FRAUD_HIGH_VALUE_THRESHOLD
        is_first_order           = request.customer_total_past_orders == 0
        is_new_account           = request.customer_account_age_days < FRAUD_NEW_ACCOUNT_THRESHOLD
        multiple_failed_attempts = request.failed_attempts >= FRAUD_MULTIPLE_FAILS
        high_velocity_1h         = request.customer_orders_1h >= FRAUD_HIGH_VELOCITY_THRESHOLD
        order_value_normalized   = (
            request.order_value / request.customer_avg_order_value
            if request.customer_avg_order_value > 0 else 1.0
        )
        new_account_high_value_card = (
            is_new_account and is_high_value and request.payment_method == "card"
        )
        payment_risk_score  = FRAUD_PAYMENT_RISK[request.payment_method]
        category_risk_score = FRAUD_CATEGORY_RISK[request.product_category]

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
            "is_night_transaction":      int(is_night_transaction),
            "is_weekend":                int(is_weekend),
            "is_new_device":             int(request.is_new_device),
            "is_different_city":         int(request.is_different_city),
            "is_high_value":             int(is_high_value),
            "is_first_order":            int(is_first_order),
            "is_new_account":            int(is_new_account),
            "address_mismatch":          int(request.address_mismatch),
            "multiple_failed_attempts":  int(multiple_failed_attempts),
            "high_velocity_1h":          int(high_velocity_1h),
            "order_value_normalized":    order_value_normalized,
            "new_account_high_value_card": int(new_account_high_value_card),
            "payment_risk_score":        payment_risk_score,
            "category_risk_score":       category_risk_score,
        }
        X          = pd.DataFrame([row])[FRAUD_FEATURE_COLUMNS]
        risk_score = float(FRAUD_MODEL.predict_proba(X)[0][1])
        risk_label = score_to_label(risk_score)
        logger.info(f"score/transaction  score={risk_score:.4f}  label={risk_label}")
        return ScoreResponse(
            risk_score=round(risk_score, 4), risk_label=risk_label,
            recommendation=FRAUD_RECOMMENDATIONS[risk_label],
            model_name=FRAUD_MODEL_NAME, threshold_used=FRAUD_THRESHOLD,
            scored_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"score/transaction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.post("/chargeback/analyze", response_model=ChargebackAnalysisResponse,
          summary="Analyze a chargeback dispute and return evidence report (Phase 3)", tags=["Chargeback Analysis"])
def analyze_chargeback(request: ChargebackRequest, _: None = Security(verify_api_key)):
    """
    Analyze a chargeback dispute and return:
    - Winability score and label (Weak / Moderate / Strong)
    - Recommendation (fight, collect more evidence, or settle)
    - Evidence checklist (what's present and what's missing)

    Call this endpoint when a bank raises a chargeback dispute.
    Use the evidence checklist to decide what to submit to the bank.
    """
    try:
        # ── Compute derived features from raw inputs ───────────────────────────
        evidence_flags = {
            "delivery_confirmed":         request.delivery_confirmed,
            "customer_signed_delivery":   request.customer_signed_delivery,
            "otp_used":                   request.otp_used,
            "ip_logs_available":          request.ip_logs_available,
            "order_confirmation_sent":    request.order_confirmation_sent,
            "refund_issued":              request.refund_issued,
            "customer_contacted_support": request.customer_contacted_support,
        }
        evidence_score  = sum(1 for v in evidence_flags.values() if v)
        is_quick_dispute = request.days_to_dispute <= CB_QUICK_DISPUTE_DAYS
        is_late_dispute  = request.days_to_dispute > CB_LATE_DISPUTE_DAYS
        is_high_value    = request.order_value > CB_HIGH_VALUE_THRESHOLD
        is_first_order   = request.customer_total_orders <= 1
        is_new_account   = request.customer_account_age_days < CB_NEW_ACCOUNT_THRESHOLD
        is_serial_disputer = request.customer_past_disputes >= CB_SERIAL_DISPUTER_COUNT
        is_cod           = request.payment_method == "cod"
        dispute_difficulty = CB_DISPUTE_DIFFICULTY[request.dispute_reason]
        order_value_normalized = (
            request.order_value / request.customer_avg_order_value
            if request.customer_avg_order_value > 0 else 1.0
        )

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
            "is_quick_dispute":          int(is_quick_dispute),
            "is_late_dispute":           int(is_late_dispute),
            "is_high_value":             int(is_high_value),
            "is_first_order":            int(is_first_order),
            "is_new_account":            int(is_new_account),
            "is_serial_disputer":        int(is_serial_disputer),
            "is_cod":                    int(is_cod),
            "evidence_score":            evidence_score,
            "dispute_difficulty":        dispute_difficulty,
            "order_value_normalized":    order_value_normalized,
        }

        X                = pd.DataFrame([row])[CB_FEATURE_COLUMNS]
        winability_score = float(CB_MODEL.predict_proba(X)[0][1])
        winability_label = score_to_winability(winability_score)
        recommendation   = CB_RECOMMENDATIONS[winability_label]

        # ── Build evidence checklist ───────────────────────────────────────────
        evidence_present = [
            EVIDENCE_LABELS[k] for k, v in evidence_flags.items() if v
        ]
        evidence_missing = [
            EVIDENCE_LABELS[k] for k, v in evidence_flags.items() if not v
        ]

        logger.info(
            f"chargeback/analyze  score={winability_score:.4f}  "
            f"label={winability_label}  evidence={evidence_score}/7"
        )
        return ChargebackAnalysisResponse(
            winability_score=round(winability_score, 4),
            winability_label=winability_label,
            recommendation=recommendation,
            evidence_score=evidence_score,
            evidence_present=evidence_present,
            evidence_missing=evidence_missing,
            model_name=CB_MODEL_NAME,
            threshold_used=CB_THRESHOLD,
            scored_at=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.error(f"chargeback/analyze failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Feedback & Retraining Pipeline
# ══════════════════════════════════════════════════════════════════════════════

# ── Feedback request schemas ──────────────────────────────────────────────────

class ReturnFeedbackRequest(BaseModel):
    original_id:     str   = Field(..., description="The return/order ID from the original request")
    risk_score:      float = Field(..., ge=0, le=1, description="The score the model gave at prediction time")
    predicted_label: str   = Field(..., description="The label the model returned: Low, Medium, or High")
    actual_outcome:  str   = Field(..., description="What actually happened: abusive or legitimate")

    @field_validator("actual_outcome")
    @classmethod
    def validate_outcome(cls, v):
        valid = VALID_OUTCOMES["return"]
        if v.lower() not in valid:
            raise ValueError(f"actual_outcome must be one of: {valid}")
        return v.lower()

    @field_validator("predicted_label")
    @classmethod
    def validate_label(cls, v):
        valid = ["Low", "Medium", "High"]
        if v not in valid:
            raise ValueError(f"predicted_label must be one of: {valid}")
        return v

    model_config = {"json_schema_extra": {"example": {
        "original_id": "RET-20260901-001",
        "risk_score": 0.87,
        "predicted_label": "High",
        "actual_outcome": "abusive",
    }}}


class FraudFeedbackRequest(BaseModel):
    original_id:     str   = Field(..., description="The transaction ID from the original request")
    risk_score:      float = Field(..., ge=0, le=1, description="The score the model gave at prediction time")
    predicted_label: str   = Field(..., description="The label the model returned: Low, Medium, or High")
    actual_outcome:  str   = Field(..., description="What actually happened: fraud or legitimate")

    @field_validator("actual_outcome")
    @classmethod
    def validate_outcome(cls, v):
        valid = VALID_OUTCOMES["fraud"]
        if v.lower() not in valid:
            raise ValueError(f"actual_outcome must be one of: {valid}")
        return v.lower()

    @field_validator("predicted_label")
    @classmethod
    def validate_label(cls, v):
        valid = ["Low", "Medium", "High"]
        if v not in valid:
            raise ValueError(f"predicted_label must be one of: {valid}")
        return v

    model_config = {"json_schema_extra": {"example": {
        "original_id": "TX-20260901-001",
        "risk_score": 0.95,
        "predicted_label": "High",
        "actual_outcome": "fraud",
    }}}


class ChargebackFeedbackRequest(BaseModel):
    original_id:     str   = Field(..., description="The dispute ID from the original request")
    risk_score:      float = Field(..., ge=0, le=1, description="The winability score the model gave")
    predicted_label: str   = Field(..., description="The label the model returned: Weak, Moderate, or Strong")
    actual_outcome:  str   = Field(..., description="What actually happened: won or lost")

    @field_validator("actual_outcome")
    @classmethod
    def validate_outcome(cls, v):
        valid = VALID_OUTCOMES["chargeback"]
        if v.lower() not in valid:
            raise ValueError(f"actual_outcome must be one of: {valid}")
        return v.lower()

    @field_validator("predicted_label")
    @classmethod
    def validate_label(cls, v):
        valid = ["Weak", "Moderate", "Strong"]
        if v not in valid:
            raise ValueError(f"predicted_label must be one of: {valid}")
        return v

    model_config = {"json_schema_extra": {"example": {
        "original_id": "CB-20260901-001",
        "risk_score": 0.91,
        "predicted_label": "Strong",
        "actual_outcome": "won",
    }}}


# ── Feedback response schema ───────────────────────────────────────────────────

class FeedbackResponse(BaseModel):
    record_id:       str   = Field(..., description="Unique ID for this feedback record")
    original_id:     str   = Field(..., description="The original transaction/return/dispute ID")
    module:          str   = Field(..., description="Which module this feedback is for")
    risk_score:      float = Field(..., description="The model score at prediction time")
    predicted_label: str   = Field(..., description="What the model predicted")
    actual_outcome:  str   = Field(..., description="What actually happened")
    is_correct:      int   = Field(..., description="1 if model was correct, 0 if wrong")
    submitted_at:    str   = Field(..., description="When this feedback was submitted")
    message:         str   = Field(..., description="Confirmation message")


# ── Feedback endpoints ─────────────────────────────────────────────────────────

@app.post("/feedback/return", response_model=FeedbackResponse,
          summary="Submit confirmed outcome for a return request (Phase 1)", tags=["Feedback"])
def feedback_return(request: ReturnFeedbackRequest, _: None = Security(verify_api_key)):
    """
    Submit a confirmed outcome for a previously scored return request.

    Call this endpoint when the actual outcome of a return is known —
    e.g. after a manual review confirms the return was abusive, or after
    the item arrives back and is confirmed legitimate.

    This feedback is stored and used in the next model retraining cycle.
    """
    try:
        record = write_feedback(
            module          = "return",
            original_id     = request.original_id,
            risk_score      = request.risk_score,
            predicted_label = request.predicted_label,
            actual_outcome  = request.actual_outcome,
        )
        correct_str = "correct" if record["is_correct"] else "incorrect"
        logger.info(
            f"feedback/return  id={request.original_id}  "
            f"predicted={request.predicted_label}  actual={request.actual_outcome}  "
            f"correct={record['is_correct']}"
        )
        return FeedbackResponse(
            **record,
            message=f"Feedback recorded. Model prediction was {correct_str}.",
        )
    except Exception as e:
        logger.error(f"feedback/return failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@app.post("/feedback/transaction", response_model=FeedbackResponse,
          summary="Submit confirmed outcome for a fraud transaction (Phase 2)", tags=["Feedback"])
def feedback_transaction(request: FraudFeedbackRequest, _: None = Security(verify_api_key)):
    """
    Submit a confirmed outcome for a previously scored transaction.

    Call this endpoint when fraud is confirmed (e.g. chargeback received,
    bank confirms stolen card) or when a flagged transaction is cleared as
    legitimate after manual review.

    This feedback is stored and used in the next model retraining cycle.
    """
    try:
        record = write_feedback(
            module          = "fraud",
            original_id     = request.original_id,
            risk_score      = request.risk_score,
            predicted_label = request.predicted_label,
            actual_outcome  = request.actual_outcome,
        )
        correct_str = "correct" if record["is_correct"] else "incorrect"
        logger.info(
            f"feedback/transaction  id={request.original_id}  "
            f"predicted={request.predicted_label}  actual={request.actual_outcome}  "
            f"correct={record['is_correct']}"
        )
        return FeedbackResponse(
            **record,
            message=f"Feedback recorded. Model prediction was {correct_str}.",
        )
    except Exception as e:
        logger.error(f"feedback/transaction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@app.post("/feedback/chargeback", response_model=FeedbackResponse,
          summary="Submit confirmed outcome for a chargeback dispute (Phase 3)", tags=["Feedback"])
def feedback_chargeback(request: ChargebackFeedbackRequest, _: None = Security(verify_api_key)):
    """
    Submit a confirmed outcome for a previously analyzed chargeback dispute.

    Call this endpoint after the bank resolves the dispute — either the
    merchant won (chargeback reversed) or lost (chargeback confirmed).

    This feedback is stored and used in the next model retraining cycle.
    """
    try:
        record = write_feedback(
            module          = "chargeback",
            original_id     = request.original_id,
            risk_score      = request.risk_score,
            predicted_label = request.predicted_label,
            actual_outcome  = request.actual_outcome,
        )
        correct_str = "correct" if record["is_correct"] else "incorrect"
        logger.info(
            f"feedback/chargeback  id={request.original_id}  "
            f"predicted={request.predicted_label}  actual={request.actual_outcome}  "
            f"correct={record['is_correct']}"
        )
        return FeedbackResponse(
            **record,
            message=f"Feedback recorded. Model prediction was {correct_str}.",
        )
    except Exception as e:
        logger.error(f"feedback/chargeback failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@app.get("/feedback/summary",
         summary="Summary of all feedback collected so far", tags=["Feedback"])
def get_feedback_summary(_: None = Security(verify_api_key)):
    """
    Returns how much feedback has been collected per module and
    how accurate the model has been on confirmed outcomes.
    Useful for deciding when to trigger a retraining run.
    """
    try:
        summary = feedback_summary()
        return {
            "feedback_summary": summary,
            "message": (
                "Run `python src/retrain.py` to retrain models using this feedback."
            ),
        }
    except Exception as e:
        logger.error(f"feedback/summary failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")
