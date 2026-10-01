"""Turning probabilities into money.

A churn model is a ranking of who to contact, not a set of answers. The operating
point comes from net value, not from maximising F1, because the two costs are
decided by the business, not by the metric.

Every figure here that we cannot measure from the data -- the cost of a retention
offer, the share of contacted customers who would have churned anyway -- is passed
in explicitly and must be stated in the README.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostAssumptions:
    monthly_revenue_per_customer: float
    retention_offer_cost: float
    """Cost to make the offer, e.g. a discount or a service credit."""
    baseline_contacts: int
    """Size of the untargeted campaign this model replaces."""
    cost_per_contact_no_offer: float
    """What a churned customer costs us with no intervention at all."""


def net_value(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    costs: CostAssumptions,
) -> dict[str, float]:
    """Net expected value of running the campaign at `threshold`, versus doing nothing.

    Value of flagging ONE customer =
        P(churn | this customer) * cost_if_churned  -  retention_offer_cost

    Each flagged customer is valued at their *own* predicted probability, not at the
    population churn rate. This is the whole point of the exercise: with a flat
    rate every flagged customer contributes an identical amount, so net value rises
    monotonically as you contact more people and the optimal threshold collapses to
    "contact everyone". Using the per-customer probability puts a real break-even
    point in the curve at

        p* = retention_offer_cost / cost_if_churned

    above which contacting someone pays for itself and below which it destroys
    value. A model that cannot separate customers above and below p* cannot be made
    profitable by thresholding at all.

    `y_true` is used only for the reported precision/recall, never for the money.
    At the moment the retention team presses send, the outcome is unknown; valuing
    a contact by the label would be lookahead leakage and would flatter every
    threshold equally.
    """
    flagged = proba >= threshold
    n_flagged = int(flagged.sum())
    if n_flagged == 0:
        return {"threshold": threshold, "n_flagged": 0, "net_value": 0.0, "recall": 0.0}

    revenue_lost_if_churned = costs.monthly_revenue_per_customer + costs.cost_per_contact_no_offer
    mean_flagged_probability = float(proba[flagged].mean())

    # Break-even probability: below this, the offer costs more than it saves.
    break_even = costs.retention_offer_cost / revenue_lost_if_churned

    saved = mean_flagged_probability * revenue_lost_if_churned - costs.retention_offer_cost
    total_net = n_flagged * saved

    y_flagged = y_true[flagged]
    recall = float(y_flagged.sum() / y_true.sum()) if y_true.sum() else 0.0

    return {
        "threshold": threshold,
        "n_flagged": n_flagged,
        "mean_flagged_probability": mean_flagged_probability,
        "break_even_probability": break_even,
        "precision_at_threshold": float(y_flagged.mean()),
        "recall": recall,
        "net_value": float(total_net),
        "net_value_per_contact": float(total_net / n_flagged),
    }


def net_value_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    costs: CostAssumptions,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """net_value at many thresholds, so the curve can be plotted and the peak found."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    rows = [net_value(y_true, proba, t, costs) for t in thresholds]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def best_threshold(curve: pd.DataFrame) -> float:
    """Threshold maximising net value, not F1 and not accuracy."""
    if curve.empty:
        raise ValueError("empty curve")
    return float(curve.loc[curve["net_value"].idxmax(), "threshold"])
