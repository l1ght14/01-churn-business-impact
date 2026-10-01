# 01 — Customer Churn Prediction & Business Impact

**This is your sklearn foundation.** You've used Python for a year but not sklearn hands-on.
This project teaches the whole supervised-learning workflow in one pass, then adds the thing
that separates a DS from an ML hobbyist: a dollar figure.

## Business question

> We spend $X to acquire a subscriber. Roughly 27% churn per month. Who should retention
> outreach target this week, and how much money does acting on this model actually save
> versus a "churn everyone" campaign?

## Dataset

`blastchar/telco-customer-churn` — 7,043 customers, 21 features, ~27% churn. Small enough
to iterate in minutes, real enough to talk about in an interview.

```bash
kaggle datasets download -d blastchar/telco-customer-churn -p data/ --unzip
```

## Stack

pandas · scikit-learn · imbalanced-learn · matplotlib/seaborn · joblib

## Method

1. **Load & profile** — dtypes, nulls, class balance, and *write down what you see*. The
   `TotalCharges` column is 19 empty strings, not nulls. Casting it alone is worth a
   paragraph in the README.
2. **Leakage audit** — check every feature for post-outcome information before you trust
   any score. Say what you checked.
3. **Preprocess** in a `Pipeline`: numeric median impute + scale, categorical most-frequent
   impute + one-hot. `ColumnTransformer` inside the pipeline, not before it. Skipping this
   is how you leak across the CV split.
4. **Baseline** — `DummyClassifier(strategy="prior")`. Everything after must beat it.
5. **Models** — LogisticRegression (interpretable baseline), RandomForest, XGBoost. Use
   `class_weight="balanced"` or SMOTE. **Try both, compare.**
6. **Evaluate correctly** — ROC-AUC + PR-AUC + recall at a *chosen* threshold. Accuracy is
   73% predicting "no churn" every time; it means nothing here.
7. **Threshold from cost, not F1** — plot the business value across thresholds, pick the
   money-maximising one. This is the crux of the project.
8. **Explain** — permutation importance, not just `.feature_importances_`.
9. **Persist** — `joblib.dump` the whole fitted pipeline.

## Business impact — the part that gets you hired

A churn model is only useful if the intervention it triggers is cheaper than the churn.

```
revenue_at_risk      = monthly_charge * retention_offer_cost
saved_per_flagged    = revenue_at_risk * P(churn | flagged) - offer_cost
net_value            = n_flagged * saved_per_flagged - campaign_cost
```

Report it as a table over 2–3 threshold choices and pick the best. Cite your assumed
offer cost and churn-rate-lift as **stated assumptions** — a number with no stated
assumption is a number nobody trusts.

## Results

7,043 customers, 26.5% churn. Baseline that always predicts "no churn": **73.5% accuracy**
— which is why accuracy is not used as a metric anywhere in this project.

| model | ROC-AUC (oof) | PR-AUC (oof) | PR-AUC (test) |
|---|---|---|---|
| dummy baseline | 0.500 | 0.265 | — |
| logistic regression | 0.8455 | 0.6572 | 0.6324 |
| random forest | 0.8441 | 0.6536 | 0.6504 |
| **xgboost** | 0.8414 | **0.6574** | 0.6515 |

**The three models are tied to within 0.004 PR-AUC.** That is the finding, not a
disappointment: on this dataset the ceiling is set by the features, not by model
capacity. Logistic regression on one-hot contract and tenure terms captures almost
everything available. Reporting only the winner would hide the more interesting
result, which is that a well-built baseline is very hard to beat here.

### The operating point

Chosen on out-of-fold training predictions, then applied **once** to the test set.

| | value |
|---|---|
| Threshold | 0.17 |
| Targeted | 871 of 1,409 test customers (62%) |
| Recall | 0.925 |
| Precision at that recall | 0.397 |
| Net value vs. doing nothing | **$97,010** |
| Net value per contact | $111 |

### Why the threshold is 0.17 and not something the model chose

Break-even is `offer_cost / (monthly_revenue + cost_of_churn)` = `45 / 270` = **0.167**.
Below that probability, a retention offer costs more than it saves. The threshold sits
right there because that is where the economics stop working — not because 0.17
maximises F1, and not because 0.17 is a round number.

Sweeping the offer price shows how much of this is a business decision:

| offer cost | break-even p | targeted | recall | net value |
|---|---|---|---|---|
| $0 | 0.000 | 77% | 0.981 | $563,674 |
| $15 | 0.056 | 75% | 0.979 | $498,876 |
| $30 | 0.111 | 68% | 0.959 | $438,663 |
| **$45** | **0.167** | **61%** | **0.929** | **$384,205** |
| $60 | 0.222 | 57% | 0.910 | $334,225 |
| $90 | 0.333 | 49% | 0.860 | $245,425 |

The same model, the same ROC-AUC, six different campaigns. Whoever owns the retention
budget picks the row. That is the point of the project: the model ranks, the
business sets the threshold, and confusing the two is how churn projects end up
contacting 75% of the customer base and calling it a machine learning win.

## Stated assumptions

None of these are measurable from this dataset, and all of them are load-bearing:

- `monthly_revenue_per_customer = $70` — the dataset's median `MonthlyCharges`
- `retention_offer_cost = $45` — ~15% discount for 6 months, plus contact cost
- `cost_per_contact_no_offer = $200` — acquisition plus 12 months of lost margin
- **Churn lift is assumed, not measured.** The model assumes every contacted
  customer who would have churned is saved. In reality some would have stayed anyway,
  and some leave despite the offer. The true number is *lower* than $384k, and
  measuring the gap requires a holdout group the dataset does not contain.

## Done means

- [x] `src/` reproduces the model from raw CSV via `python train.py`
- [x] `tests/test_pipeline.py` — 25 tests, fitted pipeline predicts on a fixture
- [x] Results table: model | ROC-AUC | PR-AUC | recall@threshold | net $
- [x] `artifacts/curves.png` — PR curve and net-value-vs-threshold
- [x] `artifacts/sensitivity.csv` — the offer-cost sweep
- [x] GitHub Actions runs the tests
- [x] Assumptions stated, including the churn lift this model cannot see
- [ ] *(yours)* git init, commit, and write the failure-mode section in your own words

## Interview questions this buys

*"Why SMOTE and not `class_weight`?"* · *"Your model has 95% recall. Should you actually
run this campaign?"* · *"How would you validate the $ figure before trusting it?"*

## Resume bullets (real numbers)

> Built a churn propensity model on 7,043 telecom subscribers (PR-AUC 0.657 vs 0.265
> baseline, recall 0.93), selecting the operating point at break-even probability
> (0.167) to yield **$97k projected retention value per scoring cycle**, and quantified
> how the optimal campaign shifts from 77% to 49% of the base as offer cost rises
> $0 to $90.

> Found and fixed a business-logic flaw where net value was computed from the
> population churn rate rather than each customer's predicted probability, which
> collapsed the optimal threshold to "contact everyone" and made 75% of the base
> look profitable.
