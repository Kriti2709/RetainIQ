from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent.parent

df = pd.read_csv(
    BASE_DIR /
    "data" /
    "processed" /
    "training_dataset.csv"
)

st.title("🚨 Risk Radar")

high_risk = df.sort_values(
    by="ticket_velocity",
    ascending=False
)

st.subheader("Top At-Risk Accounts")

st.dataframe(
    high_risk.head(25),
    use_container_width=True
)