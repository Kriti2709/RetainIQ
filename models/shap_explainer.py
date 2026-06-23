"""
RetainIQ SHAP Explainability
"""

from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR /
    "data" /
    "processed" /
    "training_dataset.csv"
)

MODEL_FILE = (
    BASE_DIR /
    "models" /
    "artifacts" /
    "churn_model.pkl"
)

OUTPUT_DIR = (
    BASE_DIR /
    "outputs"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

# ==========================================
# Load Data
# ==========================================

print("Loading data...")

df = pd.read_csv(DATA_FILE)

df["plan_tier"] = (
    df["plan_tier"]
    .astype("category")
    .cat.codes
)

X = df.drop(
    columns=[
        "workspace_id",
        "churned"
    ]
)

# ==========================================
# Load Model
# ==========================================

print("Loading model...")

model = joblib.load(MODEL_FILE)

# ==========================================
# SHAP Explainer
# ==========================================

print("Computing SHAP values...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

# ==========================================
# Global Summary Plot
# ==========================================

print("Generating summary plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.tight_layout()

summary_path = (
    OUTPUT_DIR /
    "shap_summary.png"
)

plt.savefig(
    summary_path,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# Single Account Explanation
# ==========================================

sample_idx = 0

waterfall_path = (
    OUTPUT_DIR /
    "shap_waterfall.png"
)

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X.iloc[sample_idx],
        feature_names=X.columns
    ),
    show=False
)

plt.savefig(
    waterfall_path,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# Finish
# ==========================================

print("\nDone")

print(f"Summary Plot: {summary_path}")
print(f"Waterfall Plot: {waterfall_path}")