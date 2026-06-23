from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent.parent

df = pd.read_csv(
    BASE_DIR /
    "data" /
    "processed" /
    "training_dataset.csv"
)

st.title("📊 Cohort Analytics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Accounts",
        len(df)
    )

with col2:
    st.metric(
        "Churn Rate",
        f"{df['churned'].mean()*100:.2f}%"
    )

with col3:
    st.metric(
        "Active Accounts",
        len(df[df["churned"] == 0])
    )

cohort = (
    df.groupby("plan_tier")
    ["churned"]
    .mean()
    .reset_index()
)

cohort["churn_rate"] = (
    cohort["churned"] * 100
)

fig = px.bar(
    cohort,
    x="plan_tier",
    y="churn_rate",
    title="Churn Rate by Plan Tier"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.dataframe(
    cohort,
    use_container_width=True
)