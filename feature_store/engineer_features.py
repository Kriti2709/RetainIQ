"""
RetainIQ Feature Engineering Pipeline
-------------------------------------

Input:
    data/raw/workspaces.csv
    data/raw/user_events.csv
    data/raw/subscriptions.csv
    data/raw/support_tickets.csv

Output:
    data/processed/training_dataset.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Load Data
# =====================================================

print("Loading datasets...")

workspaces = pd.read_csv(RAW_DIR / "workspaces.csv")
events = pd.read_csv(RAW_DIR / "user_events.csv")
subscriptions = pd.read_csv(RAW_DIR / "subscriptions.csv")
tickets = pd.read_csv(RAW_DIR / "support_tickets.csv")

# =====================================================
# Datetime Conversion
# =====================================================

events["event_timestamp"] = pd.to_datetime(
    events["event_timestamp"]
)

workspaces["created_at"] = pd.to_datetime(
    workspaces["created_at"]
)

tickets["created_at"] = pd.to_datetime(
    tickets["created_at"]
)

# =====================================================
# DAU / WAU Ratio
# =====================================================

print("Calculating DAU/WAU ratio...")

events["event_date"] = events["event_timestamp"].dt.date

daily_activity = (
    events.groupby(["workspace_id", "event_date"])
    .size()
    .reset_index(name="daily_events")
)

avg_daily_events = (
    daily_activity.groupby("workspace_id")
    ["daily_events"]
    .mean()
)

events["week"] = (
    events["event_timestamp"]
    .dt.isocalendar()
    .week
)

weekly_activity = (
    events.groupby(["workspace_id", "week"])
    .size()
    .reset_index(name="weekly_events")
)

avg_weekly_events = (
    weekly_activity.groupby("workspace_id")
    ["weekly_events"]
    .mean()
)

dau_wau_ratio = (
    avg_daily_events /
    avg_weekly_events
).fillna(0)

# =====================================================
# Seat Utilization
# =====================================================

print("Calculating seat utilization...")

event_counts = (
    events.groupby("workspace_id")
    .size()
    .rename("total_events")
)

seat_utilization = (
    event_counts /
    workspaces.set_index("workspace_id")
    ["seat_count"]
)

seat_utilization = seat_utilization.clip(upper=10)

# =====================================================
# Ticket Velocity
# =====================================================

print("Calculating ticket velocity...")

ticket_velocity = (
    tickets.groupby("workspace_id")
    .size()
    .rename("ticket_velocity")
)

avg_resolution = (
    tickets.groupby("workspace_id")
    ["resolution_days"]
    .mean()
    .rename("avg_resolution_days")
)

# =====================================================
# Feature Adoption Score
# =====================================================

print("Calculating feature adoption score...")

feature_adoption = (
    events.groupby("workspace_id")
    ["event_type"]
    .nunique()
    .rename("feature_adoption_score")
)

# =====================================================
# Workspace Age
# =====================================================

today = pd.Timestamp("2025-07-01")

workspace_age = (
    today -
    workspaces.set_index("workspace_id")["created_at"]
)

workspace_age_days = (
    workspace_age.dt.days
    .rename("workspace_age_days")
)

# =====================================================
# Build Feature Dataset
# =====================================================

print("Building training dataset...")

dataset = pd.DataFrame({
    "workspace_id": workspaces["workspace_id"]
})

dataset = dataset.merge(
    subscriptions[
        [
            "workspace_id",
            "churned",
            "plan_tier",
            "monthly_price"
        ]
    ],
    on="workspace_id",
    how="left"
)

dataset["dau_wau_ratio"] = (
    dataset["workspace_id"]
    .map(dau_wau_ratio)
)

dataset["seat_utilization"] = (
    dataset["workspace_id"]
    .map(seat_utilization)
)

dataset["ticket_velocity"] = (
    dataset["workspace_id"]
    .map(ticket_velocity)
)

dataset["avg_resolution_days"] = (
    dataset["workspace_id"]
    .map(avg_resolution)
)

dataset["feature_adoption_score"] = (
    dataset["workspace_id"]
    .map(feature_adoption)
)

dataset["total_events"] = (
    dataset["workspace_id"]
    .map(event_counts)
)

dataset["workspace_age_days"] = (
    dataset["workspace_id"]
    .map(workspace_age_days)
)

# =====================================================
# Missing Values
# =====================================================

dataset.fillna(0, inplace=True)

# =====================================================
# Save Dataset
# =====================================================

output_file = (
    PROCESSED_DIR /
    "training_dataset.csv"
)

dataset.to_csv(
    output_file,
    index=False
)

# =====================================================
# Summary
# =====================================================

print("\nFeature Engineering Complete")
print("-" * 40)

print(f"Rows: {len(dataset):,}")
print(f"Columns: {len(dataset.columns)}")

print("\nSaved to:")
print(output_file)

print("\nPreview:")
print(dataset.head())