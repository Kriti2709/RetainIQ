from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from api.schemas import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
)

# =====================================================
# App
# =====================================================

app = FastAPI(
    title="RetainIQ API",
    description="B2B SaaS Churn Intelligence Platform",
    version="1.0.0",
)

# =====================================================
# Load Model
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "artifacts"
    / "churn_model.pkl"
)

model = joblib.load(MODEL_PATH)

# =====================================================
# Routes
# =====================================================


@app.get("/")
def root():
    return {
        "service": "RetainIQ",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post(
    "/predict",
    response_model=ChurnPredictionResponse
)
def predict(
    request: ChurnPredictionRequest
):
    data = pd.DataFrame(
        [
            {
                "plan_tier": request.plan_tier,
                "monthly_price": request.monthly_price,
                "dau_wau_ratio": request.dau_wau_ratio,
                "seat_utilization": request.seat_utilization,
                "ticket_velocity": request.ticket_velocity,
                "avg_resolution_days": request.avg_resolution_days,
                "feature_adoption_score": request.feature_adoption_score,
                "total_events": request.total_events,
                "workspace_age_days": request.workspace_age_days,
            }
        ]
    )

    probability = float(
        model.predict_proba(data)[0][1]
    )

    if probability >= 0.75:
        risk = "HIGH"
    elif probability >= 0.40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return ChurnPredictionResponse(
        churn_probability=round(probability, 4),
        risk_level=risk,
    )