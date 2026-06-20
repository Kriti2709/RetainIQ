from pydantic import BaseModel


class ChurnPredictionRequest(BaseModel):
    plan_tier: int
    monthly_price: float
    dau_wau_ratio: float
    seat_utilization: float
    ticket_velocity: float
    avg_resolution_days: float
    feature_adoption_score: float
    total_events: float
    workspace_age_days: float


class ChurnPredictionResponse(BaseModel):
    churn_probability: float
    risk_level: str