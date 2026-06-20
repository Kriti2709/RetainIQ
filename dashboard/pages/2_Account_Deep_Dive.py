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

st.title("🔍 Account Deep Dive")

workspace = st.selectbox(
    "Select Workspace",
    sorted(df["workspace_id"].unique())
)

row = df[
    df["workspace_id"] == workspace
].iloc[0]

st.subheader(f"Workspace {workspace}")

st.json(
    row.to_dict()
)