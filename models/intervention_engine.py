"""
RetainIQ Intervention Engine
"""

from typing import Dict


def get_intervention(
    churn_probability: float,
    plan_tier: str,
    ticket_velocity: float,
) -> Dict:
    """
    Returns retention recommendation.
    """

    if churn_probability >= 0.75:

        if plan_tier == "Enterprise":

            return {
                "risk_level": "HIGH",
                "recommended_action": "CSM Outreach",
                "reason":
                    "High-value enterprise customer at risk."
            }

        elif plan_tier == "Business":

            return {
                "risk_level": "HIGH",
                "recommended_action": "Executive Check-in",
                "reason":
                    "Business account showing churn signals."
            }

        else:

            return {
                "risk_level": "HIGH",
                "recommended_action": "Discount Offer",
                "reason":
                    "Price-sensitive account with high churn risk."
            }

    elif churn_probability >= 0.40:

        if ticket_velocity > 8:

            return {
                "risk_level": "MEDIUM",
                "recommended_action": "Support Escalation",
                "reason":
                    "High support burden detected."
            }

        return {
            "risk_level": "MEDIUM",
            "recommended_action": "Feature Nudge",
            "reason":
                "Increase feature adoption and engagement."
        }

    else:

        return {
            "risk_level": "LOW",
            "recommended_action": "No Action",
            "reason":
                "Account appears healthy."
        }


if __name__ == "__main__":

    result = get_intervention(
        churn_probability=0.87,
        plan_tier="Enterprise",
        ticket_velocity=12
    )

    print(result)