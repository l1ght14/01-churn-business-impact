"""Loading and cleaning the Telco churn dataset.

The only real data defect in this dataset is subtle: `TotalCharges` arrives as an
object column with 19 blank strings where a tenure customer should have a number.
Left alone, every model that touches it either crashes or silently drops those
customers. We coerce it and let the imputer handle the rest.
"""

from pathlib import Path

import pandas as pd

TARGET = "Churn"
#: The published archive is named `WA_Fn-UseC_-Telco-Customer-Churn.csv`. Resolved by
#: discovery rather than hardcoded, because a guessed filename produces a confusing
#: FileNotFoundError on a directory that plainly contains a CSV.
KNOWN_FILENAMES = ("WA_Fn-UseC_-Telco-Customer-Churn.csv", "Telco-Customer-Churn.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns that are safe to use for prediction. `CustomerID` is a unique key: keeping
# it lets the model memorise individual customers and inflates CV scores, so it is
# dropped. Nothing else here leaks the outcome.
LEAKY_COLUMNS = ["CustomerID"]


def load_raw(data_dir: Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    for name in KNOWN_FILENAMES:
        if (data_dir / name).exists():
            return pd.read_csv(data_dir / name)

    if data_dir.exists():
        found = sorted(p for p in data_dir.glob("*.csv"))
        if len(found) == 1:
            return pd.read_csv(found[0])
        if found:
            raise FileNotFoundError(
                f"{data_dir} holds {len(found)} CSVs {[p.name for p in found]}; "
                f"expected one of {list(KNOWN_FILENAMES)}."
            )

    raise FileNotFoundError(
        f"No churn CSV in {data_dir}. Download it with:\n"
        "  python -m kaggle datasets download -d blastchar/telco-customer-churn -p data/ --unzip"
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with `TotalCharges` numeric and `Churn` as 0/1."""
    out = df.drop(columns=LEAKY_COLUMNS, errors="ignore").copy()

    # Blank strings are not nulls, so astype(float) alone raises instead of coercing.
    # Going via str/strip first avoids injecting NA into a float block, which pandas
    # 3 rejects, and catches tabs and unicode spaces that a bare `replace(" ", ...)`
    # would miss.
    out["TotalCharges"] = pd.to_numeric(
        out["TotalCharges"].astype("string").str.strip().replace("", pd.NA),
        errors="coerce",
    )

    out[TARGET] = out[TARGET].map({"Yes": 1, "No": 0})
    if out[TARGET].isna().any():
        raise ValueError("Churn contains values outside {Yes, No}")
    return out


def split_feature_types(X) -> tuple[list[str], list[str]]:
    """Split columns by dtype rather than by a hardcoded list.

    Deriving this from the frame means a schema change upstream shows up as a wrong
    bucket count in a test, instead of a crash three functions later.
    """
    numeric = list(X.select_dtypes("number").columns)
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def train_test_split_df(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split, returned as (X, y) pairs.

    Stratified because 27% of the positives is too few to survive an unstratified
    split without the test-set rate drifting.
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
