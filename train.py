"""Train, compare, pick a threshold by money, persist.

    python train.py

Prints a comparison table, writes figures to artifacts/, and dumps the winning
pipeline to artifacts/model.joblib.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.churn.business import CostAssumptions, best_threshold, net_value, net_value_curve
from src.churn.data import RANDOM_STATE, clean, load_raw, train_test_split_df
from src.churn.pipeline import build_pipeline

DATA_DIR = Path("data")
ARTIFACTS = Path("artifacts")
MODELS = ["logreg", "rf", "xgb"]

# Stated assumptions, not measured from this dataset. They are the whole ballgame:
# change them and the optimal threshold moves. Documented here and in the README.
#
# retention_offer_cost is the load-bearing one. A 10-20% inbound discount on a $70
# monthly plan, given for six months, is $42-84 before any contact cost, so $45 is
# the central case. An earlier draft used $15, which is not a real offer: at $15 the
# break-even probability is 0.056 and the "optimal" campaign contacts 75% of the
# customer base, which is not a targeted retention programme. The sensitivity sweep
# at the end of this run is what makes that visible instead of a hidden judgement.
COSTS = CostAssumptions(
    monthly_revenue_per_customer=70.0,   # median MonthlyCharges in the dataset
    retention_offer_cost=45.0,           # ~15% discount x 6 months, plus contact cost
    baseline_contacts=7043,              # every customer, i.e. the untargeted campaign
    cost_per_contact_no_offer=200.0,     # acquisition + 12mo lost margin
)


def evaluate(name: str, y_true, proba) -> dict:
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)

    df = clean(load_raw(DATA_DIR))
    X_train, X_test, y_train, y_test = train_test_split_df(df)
    print(f"{len(df)} rows | train {len(X_train)} | test {len(X_test)}")
    print(f"churn rate: train {y_train.mean():.3f} | test {y_test.mean():.3f}")

    # The number that justifies everything else. Accuracy here is meaningless.
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    print(f"\nBaseline (predict 'no churn' always): accuracy={dummy.score(X_test, y_test):.3f}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = [evaluate("dummy", y_test, dummy.predict_proba(X_test)[:, 1])]

    fitted = {}
    for name in MODELS:
        pipe = build_pipeline(name, X_train)
        # Cross-validated probabilities, not cross_val_score: we need out-of-fold
        # predictions to draw the PR curve and to pick the threshold without ever
        # touching the test set.
        oof = cross_val_predict(
            pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        rows.append(evaluate(f"{name} (oof)", y_train, oof))

        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        rows.append(evaluate(f"{name} (test)", y_test, pipe.predict_proba(X_test)[:, 1]))

    results = pd.DataFrame(rows)
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    results.to_csv(ARTIFACTS / "model_comparison.csv", index=False)

    # Threshold chosen on out-of-fold training predictions, so the test set stays clean
    # and the threshold is not fitted to the data it is later scored on.
    best = max(
        MODELS,
        key=lambda n: average_precision_score(
            y_train, cross_val_predict(
                build_pipeline(n, X_train), X_train, y_train, cv=cv,
                method="predict_proba", n_jobs=-1,
            )[:, 1],
        ),
    )
    oof_best = cross_val_predict(
        build_pipeline(best, X_train), X_train, y_train, cv=cv,
        method="predict_proba", n_jobs=-1,
    )[:, 1]

    curve = net_value_curve(y_train, oof_best, COSTS)
    thr = best_threshold(curve)
    print(f"\nwinning model: {best} | cost-optimal threshold: {thr:.3f}")
    print(curve.loc[curve["threshold"].round(3) == round(thr, 3)].to_string(index=False))

    # Honest reporting: the test set is scored once, at the threshold already fixed.
    test_proba = fitted[best].predict_proba(X_test)[:, 1]
    final = net_value(y_test, test_proba, thr, COSTS)
    print("\ntest set, at the fixed threshold:")
    print(json.dumps(final, indent=2))

    joblib.dump(
        {"model": fitted[best], "threshold": thr, "costs": COSTS}, ARTIFACTS / "model.joblib"
    )

    _plot(curve, test_proba, y_test, best, thr)
    _sensitivity(y_train, oof_best)

    json.dump(
        {"model": best, "threshold": thr, "test": final,
         "results": results.to_dict("records")},
        (ARTIFACTS / "results.json").open("w"), indent=2, default=str,
    )
    print(f"\nwrote {ARTIFACTS}/model.joblib, results.json, figures")


def _sensitivity(y_true, proba) -> None:
    """How the chosen campaign changes as the offer gets more or less expensive.

    This table is the argument of the project. The threshold is not a property of
    the model -- it is a property of the ratio between what an offer costs and what
    a churn costs. A model with a fixed ROC-AUC supports a wide range of very
    different campaigns depending on which side of break-even you sit, so the
    honest deliverable is the whole curve, not one number.

    Read it as: as the offer gets dearer, the break-even probability rises and the
    campaign must shrink or it destroys value. Below a certain price the correct
    answer stops being a model problem and becomes a "contact everyone" decision,
    which is worth saying out loud rather than hiding behind a threshold.
    """
    rows = []
    for offer_cost in [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]:
        costs = replace(COSTS, retention_offer_cost=offer_cost)
        curve = net_value_curve(y_true, proba, costs)
        thr = best_threshold(curve)
        row = curve.loc[curve["threshold"] == thr].iloc[0]
        rows.append(
            {
                "offer_cost": offer_cost,
                "break_even_p": round(row["break_even_probability"], 4),
                "threshold": thr,
                "n_flagged": int(row["n_flagged"]),
                "pct_of_base": f"{row['n_flagged'] / len(y_true):.0%}",
                "recall": round(row["recall"], 3),
                "net_value": round(row["net_value"], 0),
            }
        )
    sweep = pd.DataFrame(rows)
    print("\n--- offer-cost sensitivity (out-of-fold) --------------------")
    print(sweep.to_string(index=False))
    sweep.to_csv(ARTIFACTS / "sensitivity.csv", index=False)
    if sweep.iloc[-1]["net_value"] < 0:
        print("  note: at the highest offer cost no campaign is profitable at all --")
        print("        the correct business answer there is to improve the offer, not the model.")


def _plot(curve, test_proba, y_test, best, thr) -> None:
    from sklearn.metrics import PrecisionRecallDisplay

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    PrecisionRecallDisplay.from_predictions(y_test, test_proba, ax=ax1)
    ax1.set_title(f"PR curve - {best} (test set)")

    ax2.plot(curve["threshold"], curve["net_value"], color="#c0392b")
    ax2.axvline(thr, ls="--", color="#2c3e50", label=f"chosen: {thr:.3f}")
    ax2.set_xlabel("threshold")
    ax2.set_ylabel("net value ($)")
    ax2.set_title("Net value vs threshold")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(ARTIFACTS / "curves.png", dpi=150)
    print(f"wrote {ARTIFACTS}/curves.png")


if __name__ == "__main__":
    main()
