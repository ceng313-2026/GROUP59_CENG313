"""Models for the Hull Tactical Market Prediction project.

Every model follows the same interface so that baselines and the proposed
solution can be evaluated with the same walk-forward loop:

    model.fit(X_train, y_train)
    allocation = model.predict_allocation(X_val)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .utils import adjusted_sharpe, clip_allocation


class BaseAllocationModel:
    """Common interface for all models."""

    name = "base"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseAllocationModel":
        return self

    def predict_allocation(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class ConstantAllocation(BaseAllocationModel):
    """Always holds the same position (1.0 = buy-and-hold)."""

    def __init__(self, value: float = 1.0):
        self.value = float(value)
        self.name = f"constant_{self.value:g}"

    def predict_allocation(self, X: pd.DataFrame) -> np.ndarray:
        return clip_allocation(np.full(len(X), self.value))


def signal_to_allocation(pred, pred_std: float, strength: float = 0.5, center: float = 1.0) -> np.ndarray:
    """Map a predicted excess return to an allocation in [0, 2].

    The prediction is standardised by `pred_std` (the spread of the model's
    predictions on its training data), so `strength` has the same meaning
    for every model: allocation = center + strength * pred / pred_std.
    """
    pred = np.asarray(pred, dtype=float)
    if pred_std <= 0 or not np.isfinite(pred_std):
        return clip_allocation(np.full(len(pred), center))
    return clip_allocation(center + strength * pred / pred_std)


class RegressionAllocationModel(BaseAllocationModel):
    """Predicts next-day excess return with a regressor, then maps it to an allocation."""

    def __init__(self, estimator, name: str, strength: float = 0.5):
        self.estimator = estimator
        self.name = name
        self.strength = strength
        self.pred_std_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegressionAllocationModel":
        self.estimator.fit(X, y)
        self.pred_std_ = float(np.std(self.estimator.predict(X)))
        return self

    def predict_returns(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=float)

    def predict_allocation(self, X: pd.DataFrame) -> np.ndarray:
        return signal_to_allocation(self.predict_returns(X), self.pred_std_, self.strength)


def make_ridge(alpha: float = 100.0, strength: float = 0.5) -> RegressionAllocationModel:
    """Ridge regression baseline with median imputation and standardisation."""
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=alpha),
    )
    return RegressionAllocationModel(pipe, name=f"ridge_a{alpha:g}", strength=strength)


def make_lightgbm(strength: float = 0.5, **params) -> RegressionAllocationModel:
    """LightGBM baseline. LightGBM handles missing values natively."""
    from lightgbm import LGBMRegressor

    defaults = dict(
        n_estimators=300,
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    defaults.update(params)
    return RegressionAllocationModel(LGBMRegressor(**defaults), name="lightgbm", strength=strength)


def walk_forward_evaluate(
    model_factory,
    df: pd.DataFrame,
    features: list[str],
    splits,
    target: str = "market_forward_excess_returns",
) -> dict:
    """Train a fresh model on each fold and score it with the competition metric.

    `model_factory` is a function that returns a new, unfitted model.
    Returns the model name, per-fold scores and the out-of-fold allocations.
    """
    scores = []
    oof = pd.Series(np.nan, index=df.index, name="allocation")
    name = None

    for train_idx, val_idx in splits:
        model = model_factory()
        name = model.name
        train, val = df.iloc[train_idx], df.iloc[val_idx]

        model.fit(train[features], train[target])
        alloc = model.predict_allocation(val[features])

        oof.iloc[val_idx] = alloc
        scores.append(adjusted_sharpe(alloc, val["forward_returns"], val["risk_free_rate"]))

    return {"name": name, "scores": scores, "allocations": oof}