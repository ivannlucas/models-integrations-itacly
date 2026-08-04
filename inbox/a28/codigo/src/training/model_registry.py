"""Model builders for baseline and neuroevolution training."""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(model_family: str, hyperparameters: dict[str, Any]):
    """Instantiate a supported regressor."""
    family = model_family.lower()
    if family == "dummy":
        return DummyRegressor(**hyperparameters)
    if family == "random_forest":
        return RandomForestRegressor(**hyperparameters)
    if family == "gradient_boosting":
        return GradientBoostingRegressor(**hyperparameters)
    if family == "linear_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression(**hyperparameters)),
            ]
        )
    if family == "ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(**hyperparameters)),
            ]
        )
    if family == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError("xgboost is not installed. Add it to requirements before using model_family=xgboost.") from exc
        return XGBRegressor(**hyperparameters)
    if family == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError("lightgbm is not installed. Add it to requirements before using model_family=lightgbm.") from exc
        return LGBMRegressor(**hyperparameters)
    if family == "neuroevolution":
        from .neuroevolution import NeuroevolutionRegressor

        return NeuroevolutionRegressor(**hyperparameters)
    raise ValueError(f"Unsupported model family: {model_family}")


def build_classifier(model_family: str, hyperparameters: dict[str, Any]):
    """Instantiate a supported classifier for the two-stage procurement policy."""
    family = model_family.lower()
    if family == "dummy":
        return DummyClassifier(**hyperparameters)
    if family == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(**hyperparameters)),
            ]
        )
    raise ValueError(f"Unsupported classifier family: {model_family}")
