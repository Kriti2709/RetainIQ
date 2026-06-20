"""
RetainIQ Churn Model Training
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

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

ARTIFACT_DIR = (
    BASE_DIR /
    "models" /
    "artifacts"
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# Load Dataset
# ==========================================

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

# ==========================================
# Encode Categories
# ==========================================

df["plan_tier"] = (
    df["plan_tier"]
    .astype("category")
    .cat.codes
)

# ==========================================
# Features / Target
# ==========================================

X = df.drop(
    columns=[
        "workspace_id",
        "churned"
    ]
)

y = df["churned"]

# ==========================================
# Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows: {len(X_test)}"
)

# ==========================================
# Train
# ==========================================

print("Training XGBoost model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(
    X_train,
    y_train
)

# ==========================================
# Evaluate
# ==========================================

pred_probs = model.predict_proba(
    X_test
)[:, 1]

auc = roc_auc_score(
    y_test,
    pred_probs
)

print(
    f"\nROC-AUC: {auc:.4f}"
)

# ==========================================
# Save Model
# ==========================================

model_path = (
    ARTIFACT_DIR /
    "churn_model.pkl"
)

joblib.dump(
    model,
    model_path
)

print(
    f"Model saved: {model_path}"
)

# ==========================================
# Feature Importance
# ==========================================

importance_df = pd.DataFrame(
    {
        "feature": X.columns,
        "importance": model.feature_importances_
    }
)

importance_df = (
    importance_df
    .sort_values(
        by="importance",
        ascending=False
    )
)

importance_path = (
    ARTIFACT_DIR /
    "feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)

print(
    "\nTop Features:"
)

print(
    importance_df.head(10)
)

print(
    f"\nSaved: {importance_path}"
)