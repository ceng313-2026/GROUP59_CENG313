"""Data loading and splitting helpers for the Hull Tactical Market Prediction data."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ID_COL = "date_id"
TARGET_COLS = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]
LAGGED_COLS = [
    "lagged_forward_returns",
    "lagged_risk_free_rate",
    "lagged_market_forward_excess_returns",
]
EXTRA_TEST_COLS = ["is_scored"]

FEATURE_GROUPS = {
    "M": "Market dynamics / technical",
    "E": "Macro-economic",
    "I": "Interest rates",
    "P": "Price / valuation",
    "V": "Volatility",
    "S": "Sentiment",
    "MOM": "Momentum",
    "D": "Binary / dummy",
}

_GROUP_PATTERN = re.compile(r"^(MOM|M|E|I|P|V|S|D)\d+$")


def load_train(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load train.csv sorted by date_id."""
    df = pd.read_csv(Path(data_dir) / "train.csv")
    return df.sort_values(ID_COL).reset_index(drop=True)


def load_test(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load test.csv sorted by date_id."""
    df = pd.read_csv(Path(data_dir) / "test.csv")
    return df.sort_values(ID_COL).reset_index(drop=True)


def feature_group(col: str) -> str | None:
    """Return the feature-group prefix of a column (e.g. 'V12' -> 'V'), or None."""
    m = _GROUP_PATTERN.match(col)
    return m.group(1) if m else None


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns that belong to one of the anonymised feature groups."""
    return [c for c in df.columns if feature_group(c) is not None]


def group_features(df: pd.DataFrame) -> dict[str, list[str]]:
    """Map each feature-group prefix to its list of columns."""
    groups: dict[str, list[str]] = {g: [] for g in FEATURE_GROUPS}
    for c in get_feature_columns(df):
        groups[feature_group(c)].append(c)
    return {g: cols for g, cols in groups.items() if cols}


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature missing ratio and the first date_id where the feature is available."""
    rows = []
    for c in get_feature_columns(df):
        notna = df[c].notna()
        rows.append({
            "feature": c,
            "group": feature_group(c),
            "missing_ratio": 1 - notna.mean(),
            "first_valid_date_id": df.loc[notna, ID_COL].min() if notna.any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["group", "feature"]).reset_index(drop=True)


def walk_forward_splits(
    n_samples: int,
    n_folds: int = 5,
    val_size: int = 252,
    gap: int = 0,
    min_train_size: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward splits over time-ordered rows.

    The last `n_folds * val_size` rows are cut into consecutive validation blocks.
    Each fold trains on all rows before its block (minus `gap` rows) and
    validates on the block.
    """
    start = n_samples - n_folds * val_size
    if start <= 0:
        raise ValueError("not enough rows for the requested folds")
    if min_train_size is not None and start - gap < min_train_size:
        raise ValueError("first fold's training set is smaller than min_train_size")

    for k in range(n_folds):
        val_start = start + k * val_size
        val_end = val_start + val_size
        train_idx = np.arange(0, val_start - gap)
        val_idx = np.arange(val_start, val_end)
        yield train_idx, val_idx


EARLY_PERIOD_END = 1006


def duplicate_summary(df: pd.DataFrame) -> pd.Series:
    """Count duplicated days, duplicated rows and identical feature columns."""
    features = get_feature_columns(df)
    other_cols = [c for c in df.columns if c != ID_COL]
    identical = [
        (a, b)
        for i, a in enumerate(features)
        for b in features[i + 1:]
        if df[a].equals(df[b])
    ]
    return pd.Series({
        "duplicated_date_ids": int(df[ID_COL].duplicated().sum()),
        "duplicated_rows_ignoring_date_id": int(df.duplicated(subset=other_cols).sum()),
        "identical_feature_pairs": ", ".join(f"{a}={b}" for a, b in identical) or "none",
    })


def preprocess(df: pd.DataFrame, start_date_id: int = EARLY_PERIOD_END) -> tuple[pd.DataFrame, list[str]]:
    """Prepare the training data for modelling.

    - sorts the rows by `date_id` (time order is required for walk-forward validation),
    - drops the early rows before `start_date_id`, where only the binary `D*` features exist,
    - keeps missing values: imputation and scaling are fitted inside each model on the
      training rows only, so no information from the validation period leaks into them.

    Returns the prepared frame (with a fresh index) and the list of feature columns.
    """
    out = df.sort_values(ID_COL)
    out = out[out[ID_COL] >= start_date_id].reset_index(drop=True)
    return out, get_feature_columns(out)


def holdout_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological hold-out split: the last `test_size` share of the days is the test set.

    `shuffle=False` keeps the time order, so the model never sees days that come after
    the test period starts.
    """
    df = df.sort_values(ID_COL)
    train_part, test_part = train_test_split(df, test_size=test_size, shuffle=False)
    return train_part, test_part