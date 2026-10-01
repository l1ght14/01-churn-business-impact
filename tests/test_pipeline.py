"""Cleaning and end-to-end pipeline behaviour on a synthetic fixture.

The fixture is generated rather than read from disk so the tests run with no
dataset present, and so the expected outcome is known by construction.
"""

import numpy as np
import pandas as pd
import pytest

from src.churn.data import clean, split_feature_types, train_test_split_df
from src.churn.pipeline import build_pipeline

TARGET = "Churn"


def make_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    churn = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "CustomerID": [f"C{i:05d}" for i in range(n)],
            "SeniorCitizen": rng.integers(0, 2, n),
            "tenure": rng.integers(1, 72, n).astype(str),  # object, like the raw CSV
            "MonthlyCharges": rng.uniform(20, 100, n),
            # Object dtype with string values, exactly as the CSV parses it. This is
            # the whole point of the dataset: a numeric-looking column that is not
            # numeric, with blanks in it.
            "TotalCharges": [f"{v:.2f}" for v in rng.uniform(20, 5000, n)],
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaymentMethod": rng.choice(["Credit card", "Bank transfer", "Mailed check"], n),
            "Churn": np.where(churn, "Yes", "No"),
        }
    )


def test_blank_totalcharges_become_nan_not_the_string_blank():
    df = make_frame(10)
    df.loc[0:2, "TotalCharges"] = " "
    out = clean(df)
    assert out["TotalCharges"].isna().sum() == 3
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])


def test_clean_drops_customerid_so_the_model_cannot_memorise_customers():
    assert "CustomerID" not in clean(make_frame(10)).columns


def test_clean_maps_churn_to_binary_integers():
    out = clean(make_frame(50))
    assert set(out[TARGET].unique()) == {0, 1}
    assert pd.api.types.is_integer_dtype(out[TARGET])


def test_unexpected_churn_label_is_rejected():
    df = make_frame(5)
    df.loc[0, TARGET] = "Maybe"
    with pytest.raises(ValueError, match="Churn contains values"):
        clean(df)


def test_missing_raw_file_names_the_kaggle_command(tmp_path):
    from src.churn.data import load_raw

    with pytest.raises(FileNotFoundError, match="kaggle datasets download"):
        load_raw(tmp_path)


def test_split_is_stratified_and_drops_the_target():
    df = clean(make_frame(200))
    X_train, X_test, y_train, y_test = train_test_split_df(df)
    assert TARGET not in X_train.columns
    assert len(X_train) == 160 and len(X_test) == 40
    # Stratification keeps the two rates within a point of each other.
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_feature_type_split_follows_dtype():
    numeric, categorical = split_feature_types(make_frame(5).drop(columns=[TARGET]))
    assert "MonthlyCharges" in numeric
    assert "Contract" in categorical
    assert not set(numeric) & set(categorical)


@pytest.mark.parametrize("name", ["logreg", "rf", "xgb"])
def test_each_pipeline_fits_and_predicts_probabilities(name):
    df = clean(make_frame(300, seed=1))
    X_train, _, y_train, _ = train_test_split_df(df)
    pipe = build_pipeline(name, X_train).fit(X_train, y_train)
    proba = pipe.predict_proba(X_train.head(5))[:, 1]
    assert proba.shape == (5,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        build_pipeline("transformer", make_frame(5))


def test_pipeline_handles_missing_values_without_a_crash():
    df = clean(make_frame(300, seed=2))
    df.loc[df.sample(30, random_state=0).index, "MonthlyCharges"] = np.nan
    df.loc[df.sample(30, random_state=0).index, "Contract"] = None
    X_train, X_test, y_train, _ = train_test_split_df(df)
    pipe = build_pipeline("logreg", X_train).fit(X_train, y_train)
    assert pipe.predict_proba(X_test.head(5)).shape == (5, 2)


def test_pipeline_sees_unseen_categories_at_predict_time():
    # A column value absent from training must not raise at inference.
    df = clean(make_frame(300, seed=3))
    X_train, X_test, y_train, _ = train_test_split_df(df)
    pipe = build_pipeline("rf", X_train).fit(X_train, y_train)
    novel = X_test.head(3).copy()
    novel["Contract"] = "Quantum annual"
    assert pipe.predict_proba(novel).shape == (3, 2)
