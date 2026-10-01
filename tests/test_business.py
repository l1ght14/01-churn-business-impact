"""Business-value maths. Pure functions, no I/O."""

import numpy as np
import pandas as pd
import pytest

from src.churn.business import CostAssumptions, best_threshold, net_value, net_value_curve

COSTS = CostAssumptions(
    monthly_revenue_per_customer=70.0,
    retention_offer_cost=15.0,
    baseline_contacts=1000,
    cost_per_contact_no_offer=200.0,
)


def test_no_flags_means_no_value():
    out = net_value(np.array([1, 0]), np.array([0.1, 0.2]), threshold=0.9, costs=COSTS)
    assert out["n_flagged"] == 0
    assert out["net_value"] == 0.0


def test_contacting_a_low_probability_customer_is_net_negative():
    # Break-even is offer_cost / (revenue + cost_if_churned) = 15 / 270 = 0.0556.
    # A contact on someone predicted at 0.05 loses money: the expected saving is
    # 0.05 * 270 = 13.50 against a 15.00 offer cost.
    out = net_value(np.array([0, 0, 1]), np.array([0.05, 0.05, 0.05]), 0.0, COSTS)
    assert out["break_even_probability"] == pytest.approx(15 / 270)
    assert out["net_value"] < 0


def test_contacting_a_high_probability_customer_is_net_positive():
    # 0.9 * 270 = 243 against a 15 offer, across 3 contacts.
    out = net_value(np.array([1, 1, 1]), np.array([0.9, 0.9, 0.9]), 0.0, COSTS)
    assert out["net_value"] == pytest.approx(3 * (0.9 * 270 - 15))


def test_value_uses_each_customers_own_probability_not_the_population_rate():
    # The bug this guards: valuing every contact at the population churn rate makes
    # net value rise monotonically with the number of contacts, collapsing the
    # optimal threshold to "contact everyone". Mixing a certain churner with a
    # certain non-churner must therefore average out near zero, not come out as
    # two profitable contacts.
    out = net_value(np.array([1, 0]), np.array([0.9, 0.0]), 0.0, COSTS)
    assert out["mean_flagged_probability"] == pytest.approx(0.45)
    # Total across both contacts: 2 * (0.45 * 270 - 15) = 213. Under the old
    # flat-population-rate version this came out at 2 * (0.5 * 270 - 15) = 240.
    assert out["net_value"] == pytest.approx(213.0)


def test_threshold_above_all_scores_flags_nothing():
    proba = np.array([0.1, 0.2, 0.3, 0.95])
    y = np.array([0, 1, 0, 1])
    assert net_value(y, proba, 0.99, COSTS)["n_flagged"] == 0


def test_recall_matches_hand_computation():
    # 2 of 3 positives are above threshold 0.5.
    y = np.array([1, 0, 1, 0, 1])
    proba = np.array([0.9, 0.1, 0.8, 0.2, 0.3])
    out = net_value(y, proba, 0.5, COSTS)
    assert out["n_flagged"] == 2
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["precision_at_threshold"] == pytest.approx(1.0)


def test_raising_the_threshold_shrinks_the_campaign():
    proba = np.linspace(0.0, 1.0, 500)
    y = (proba > 0.5).astype(int)
    loose = net_value(y, proba, 0.2, COSTS)
    tight = net_value(y, proba, 0.8, COSTS)
    assert tight["n_flagged"] < loose["n_flagged"]


def test_net_value_is_negative_when_the_offer_costs_more_than_saving():
    expensive = CostAssumptions(
        monthly_revenue_per_customer=1.0,
        retention_offer_cost=1000.0,
        baseline_contacts=10,
        cost_per_contact_no_offer=0.0,
    )
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.9, 0.9, 0.9])
    assert net_value(y, proba, 0.5, expensive)["net_value"] < 0


def test_best_threshold_lands_above_the_break_even_probability():
    # The economically meaningful assertion: the chosen operating point must be
    # above break-even, otherwise every contact in it destroys value.
    rng = np.random.default_rng(0)
    proba = rng.uniform(0, 1, 2000)
    y = (rng.random(2000) < proba).astype(int)
    curve = net_value_curve(y, proba, COSTS)
    thr = best_threshold(curve)
    break_even = COSTS.retention_offer_cost / (
        COSTS.monthly_revenue_per_customer + COSTS.cost_per_contact_no_offer
    )
    assert thr > break_even
    assert curve["net_value"].max() > 0


def test_curve_covers_every_threshold_requested():
    thresholds = np.array([0.2, 0.5, 0.8])
    curve = net_value_curve(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.7, 0.3]), COSTS, thresholds)
    assert list(curve["threshold"]) == [0.2, 0.5, 0.8]


def test_best_threshold_ignores_accuracy_and_picks_the_money_peak():
    curve = pd.DataFrame(
        {
            "threshold": [0.1, 0.5, 0.9],
            "net_value": [-50.0, 400.0, 10.0],
        }
    )
    assert best_threshold(curve) == 0.5


def test_empty_curve_is_an_error_not_a_crash():
    with pytest.raises(ValueError):
        best_threshold(pd.DataFrame(columns=["threshold", "net_value"]))
