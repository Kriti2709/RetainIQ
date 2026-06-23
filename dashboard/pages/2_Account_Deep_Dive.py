from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import joblib
import pandas as pd
import streamlit as st

from models.intervention_engine import get_intervention

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

WATERFALL_PLOT = (
    BASE_DIR
    / "outputs"
    / "shap_waterfall.png"
)

df = pd.read_csv(DATA_FILE)

model = joblib.load(MODEL_FILE)

st.title("🔍 Account Deep Dive")

workspace = st.selectbox(
    "Select Workspace",
    sorted(df["workspace_id"])
)

row = df[
    df["workspace_id"] == workspace
].iloc[0]

model_row = row.copy()

plan_mapping = {
    "Business": 0,
    "Enterprise": 1,
    "Growth": 2,
    "Starter": 3
}

model_row["plan_tier"] = (
    plan_mapping[
        model_row["plan_tier"]
    ]
)

X = pd.DataFrame(
    [
        model_row.drop(
            [
                "workspace_id",
                "churned"
            ]
        )
    ]
)

probability = float(
    model.predict_proba(X)[0][1]
)

intervention = get_intervention(
    churn_probability=probability,
    plan_tier=row["plan_tier"],
    ticket_velocity=row["ticket_velocity"]
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

with col2:
    st.metric(
        "Risk Level",
        intervention["risk_level"]
    )

with col3:
    st.metric(
        "Recommended Action",
        intervention["recommended_action"]
    )

st.subheader("Reason")

st.info(
    intervention["reason"]
)

st.subheader("SHAP Explanation")

st.image(
    str(WATERFALL_PLOT),
    use_container_width=True
)

st.subheader("Workspace Details")

st.dataframe(
    pd.DataFrame(
        row.to_dict(),
        index=["Value"]
    ).T
)