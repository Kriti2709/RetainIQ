"""
RetainIQ Model Evaluation
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

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

# ==========================================
# Load
# ==========================================

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

y = df["churned"]

model = joblib.load(
    MODEL_FILE
)

# ==========================================
# Predict
# ==========================================

preds = model.predict(X)

probs = model.predict_proba(
    X
)[:, 1]

# ==========================================
# Metrics
# ==========================================

auc = roc_auc_score(
    y,
    probs
)

print("\nROC-AUC")
print(auc)

print("\nConfusion Matrix")
print(
    confusion_matrix(
        y,
        preds
    )
)

print("\nClassification Report")
print(
    classification_report(
        y,
        preds
    )
)