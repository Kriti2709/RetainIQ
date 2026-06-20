from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

report = pd.read_csv(
    BASE_DIR /
    "monitoring" /
    "drift_report.csv"
)

max_drift = report["drift_percent"].max()

print(
    f"Maximum Drift: {max_drift:.2f}%"
)

THRESHOLD = 20

if max_drift > THRESHOLD:

    print(
        "\nDrift threshold exceeded."
    )

    print(
        "Retraining model..."
    )

else:

    print(
        "\nNo retraining required."
    )