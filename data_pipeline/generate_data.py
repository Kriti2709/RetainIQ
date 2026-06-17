"""
RetainIQ Synthetic Data Generator
---------------------------------

Generates four tables:

1. workspaces.csv
2. user_events.csv
3. subscriptions.csv
4. support_tickets.csv

Output folder:
data/raw/

Author: RetainIQ
"""

from pathlib import Path
from faker import Faker
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# =====================================================
# Configuration
# =====================================================

SEED = 42

NUM_WORKSPACES = 1000
MIN_USERS = 5
MAX_USERS = 250

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 6, 30)

fake = Faker()
Faker.seed(SEED)

random.seed(SEED)
np.random.seed(SEED)

# =====================================================
# Output Directories
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Helpers
# =====================================================


def random_date(start, end):
    """
    Generate random datetime between start and end.
    """
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


# =====================================================
# Generate Workspaces
# =====================================================

print("Generating workspaces...")

workspace_rows = []

plan_types = ["Starter", "Growth", "Business", "Enterprise"]

for workspace_id in range(1, NUM_WORKSPACES + 1):

    created_at = random_date(START_DATE, END_DATE)

    company_size = random.choice(
        ["1-10", "11-50", "51-200", "201-1000"]
    )

    seat_count = random.randint(MIN_USERS, MAX_USERS)

    plan = np.random.choice(
        plan_types,
        p=[0.35, 0.35, 0.20, 0.10]
    )

    workspace_rows.append(
        {
            "workspace_id": workspace_id,
            "workspace_name": fake.company(),
            "industry": fake.random_element(
                elements=(
                    "Technology",
                    "Finance",
                    "Healthcare",
                    "Education",
                    "Retail",
                    "Manufacturing",
                )
            ),
            "company_size": company_size,
            "seat_count": seat_count,
            "plan_tier": plan,
            "created_at": created_at.date(),
        }
    )

workspaces = pd.DataFrame(workspace_rows)

# =====================================================
# Generate Subscriptions
# =====================================================

print("Generating subscriptions...")

subscription_rows = []

for _, row in workspaces.iterrows():

    plan = row["plan_tier"]

    if plan == "Starter":
        monthly_price = 49
    elif plan == "Growth":
        monthly_price = 199
    elif plan == "Business":
        monthly_price = 499
    else:
        monthly_price = 999

    renewal_date = random_date(
        datetime(2025, 7, 1),
        datetime(2025, 12, 31)
    )

    # realistic churn distribution
    churn_probability = np.random.beta(2, 8)

    churned = int(churn_probability > 0.35)

    subscription_rows.append(
        {
            "subscription_id": row["workspace_id"],
            "workspace_id": row["workspace_id"],
            "plan_tier": plan,
            "monthly_price": monthly_price,
            "renewal_date": renewal_date.date(),
            "churned": churned,
        }
    )

subscriptions = pd.DataFrame(subscription_rows)

# =====================================================
# Generate User Events
# =====================================================

print("Generating user events...")

event_types = [
    "login",
    "task_created",
    "task_completed",
    "comment_added",
    "file_uploaded",
    "dashboard_view",
]

event_rows = []

for _, workspace in workspaces.iterrows():

    workspace_id = workspace["workspace_id"]

    churn_label = subscriptions.loc[
        subscriptions["workspace_id"] == workspace_id,
        "churned"
    ].values[0]

    base_activity = np.random.randint(150, 800)

    # churned customers have lower activity
    if churn_label == 1:
        base_activity = int(base_activity * 0.40)

    for _ in range(base_activity):

        event_date = random_date(
            datetime(2025, 1, 1),
            datetime(2025, 6, 30)
        )

        event_rows.append(
            {
                "event_id": len(event_rows) + 1,
                "workspace_id": workspace_id,
                "event_type": random.choice(event_types),
                "event_timestamp": event_date,
            }
        )

user_events = pd.DataFrame(event_rows)

# =====================================================
# Generate Support Tickets
# =====================================================

print("Generating support tickets...")

ticket_rows = []

ticket_types = [
    "Bug",
    "Feature Request",
    "Billing",
    "Technical Issue",
]

ticket_statuses = [
    "Open",
    "Resolved",
    "Closed",
]

for _, workspace in workspaces.iterrows():

    workspace_id = workspace["workspace_id"]

    churn_label = subscriptions.loc[
        subscriptions["workspace_id"] == workspace_id,
        "churned"
    ].values[0]

    ticket_count = np.random.poisson(6)

    # churned customers tend to raise more tickets
    if churn_label == 1:
        ticket_count += np.random.randint(4, 12)

    for _ in range(ticket_count):

        created_date = random_date(
            datetime(2025, 1, 1),
            datetime(2025, 6, 30)
        )

        resolution_days = np.random.randint(1, 14)

        ticket_rows.append(
            {
                "ticket_id": len(ticket_rows) + 1,
                "workspace_id": workspace_id,
                "ticket_type": random.choice(ticket_types),
                "status": random.choice(ticket_statuses),
                "created_at": created_date,
                "resolution_days": resolution_days,
            }
        )

support_tickets = pd.DataFrame(ticket_rows)

# =====================================================
# Save Files
# =====================================================

print("Saving CSV files...")

workspaces.to_csv(
    OUTPUT_DIR / "workspaces.csv",
    index=False
)

subscriptions.to_csv(
    OUTPUT_DIR / "subscriptions.csv",
    index=False
)

user_events.to_csv(
    OUTPUT_DIR / "user_events.csv",
    index=False
)

support_tickets.to_csv(
    OUTPUT_DIR / "support_tickets.csv",
    index=False
)

# =====================================================
# Summary
# =====================================================

print("\nGeneration Complete")
print("-" * 40)

print(f"Workspaces: {len(workspaces):,}")
print(f"Subscriptions: {len(subscriptions):,}")
print(f"User Events: {len(user_events):,}")
print(f"Support Tickets: {len(support_tickets):,}")

print("\nSaved to:")
print(OUTPUT_DIR)