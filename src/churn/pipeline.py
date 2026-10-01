"""Preprocessing and the model zoo.

Everything lives inside a Pipeline. That is not style preference: fitting a
scaler or an imputer *before* train_test_split leaks test-set statistics into
training and quietly inflates every score. The Pipeline makes that mistake
unrepresentable.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.churn.data import split_feature_types

# Chosen by dtype from the cleaned frame, not hardcoded, so a schema change in the
# upstream CSV surfaces as an error here instead of a silent wrong-dtype crash.
TARGET = "Churn"


def build_preprocessor(X) -> ColumnTransformer:
    numeric, categorical = split_feature_types(X)
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        # Median, not mean: TotalCharges is right-skewed and a handful
                        # of extreme high-tenure customers would drag a mean far up.
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        # min_frequency groups the ~2% "Other"/rare categories
                        # instead of minting a one-hot column for each of them.
                        ("onehot", OneHotEncoder(min_frequency=10, drop="if_binary",
                                                 sparse_output=False, handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def build_pipeline(model_name: str, X) -> Pipeline:
    pre = build_preprocessor(X)
    steps = [("preprocess", pre)]

    if model_name == "logreg":
        # class_weight handles the 27% majority without resampling, so no synthetic
        # rows can end up on the wrong side of the CV boundary.
        steps.append(
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            )
        )
    elif model_name == "rf":
        steps.append(
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    class_weight="balanced_subsample",
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=42,
                ),
            )
        )
    elif model_name == "xgb":
        from xgboost import XGBClassifier

        steps.append(
            (
                "clf",
                XGBClassifier(
                    # Ratio comes from the training fold's own class balance; we set it
                    # statically at ~2.7 which is the dataset's known rate, and
                    # recompute it per-fold in train.py.
                    scale_pos_weight=2.7,
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42,
                ),
            )
        )
    else:
        raise ValueError(f"unknown model {model_name!r}; expected logreg, rf or xgb")

    return Pipeline(steps)
