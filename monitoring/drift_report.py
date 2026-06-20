from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

baseline = pd.read_csv(
    BASE_DIR /
    "data" /
    "processed" /
    "training_dataset.csv"
)

current = baseline.sample(
    frac=1,
    random_state=42
).copy()

# Simulate drift
current["ticket_velocity"] *= 1.5

numeric_cols = [
    "dau_wau_ratio",
    "seat_utilization",
    "ticket_velocity",
    "avg_resolution_days",
    "total_events",
    "workspace_age_days"
]

results = []

for col in numeric_cols:

    baseline_mean = baseline[col].mean()
    current_mean = current[col].mean()

    drift_pct = abs(
        current_mean - baseline_mean
    ) / baseline_mean * 100

    results.append(
        {
            "feature": col,
            "baseline_mean": round(baseline_mean, 2),
            "current_mean": round(current_mean, 2),
            "drift_percent": round(drift_pct, 2),
        }
    )

report = pd.DataFrame(results)

print("\nDRIFT REPORT")
print(report)

report.to_csv(
    BASE_DIR /
    "monitoring" /
    "drift_report.csv",
    index=False
)