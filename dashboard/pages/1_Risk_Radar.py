from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import joblib
import pandas as pd
import streamlit as st

from models.intervention_engine import get_intervention

st.set_page_config(layout="wide")
from models.intervention_engine import get_intervention

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "training_dataset.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "artifacts"
    / "churn_model.pkl"
)

df = pd.read_csv(DATA_FILE)

model = joblib.load(MODEL_FILE)

model_df = df.copy()

model_df["plan_tier"] = (
    model_df["plan_tier"]
    .astype("category")
    .cat.codes
)

X = model_df.drop(
    columns=[
        "workspace_id",
        "churned"
    ]
)

df["churn_probability"] = (
    model.predict_proba(X)[:, 1]
)

results = []

for _, row in df.iterrows():

    intervention = get_intervention(
        churn_probability=row["churn_probability"],
        plan_tier=row["plan_tier"],
        ticket_velocity=row["ticket_velocity"]
    )

    results.append(
        {
            "workspace_id": row["workspace_id"],
            "churn_probability":
                round(row["churn_probability"], 4),
            "risk_level":
                intervention["risk_level"],
            "recommended_action":
                intervention["recommended_action"]
        }
    )

risk_df = pd.DataFrame(results)

risk_df = risk_df.sort_values(
    by="churn_probability",
    ascending=False
)

st.title("🚨 Risk Radar")

st.metric(
    "High Risk Accounts",
    len(
        risk_df[
            risk_df["risk_level"] == "HIGH"
        ]
    )
)

st.dataframe(
    risk_df.head(25),
    use_container_width=True
)