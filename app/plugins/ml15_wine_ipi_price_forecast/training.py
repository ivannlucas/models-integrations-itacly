"""Retraining logic for ml15 — fits a fresh Pipeline(StandardScaler + Ridge(alpha)) from scratch.

Faithful port of src/training/trainer.py::train_model + src/get_stats/get_stats.py::
_regression_metrics from inbox/a15/codigo/ (see inbox/a15/manifest.yaml). Deliberately restricted
to the FIXED 16 feature_columns of the delivered artifact, instead of the original trainer.py's
auto-selection of "any numeric column not in non_predictor_columns/target_prefixes" — exposing
that auto-selection over a generic /train endpoint would let a caller's CSV silently smuggle in
extra numeric columns as model features (see inbox/a15/manifest.yaml known_issues).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.plugins.ml15_wine_ipi_price_forecast.constants import (
    ALPHA,
    ANCHOR_COLUMN,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_HOLDOUT_FRACTION,
)

logger = logging.getLogger(__name__)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_anchor: np.ndarray) -> dict[str, float]:
    """Port of src/get_stats/get_stats.py::_regression_metrics (rmse/mae/mape_pct/r2/mda_pct)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    with np.errstate(divide="ignore", invalid="ignore"):
        mape_arr = np.abs((y_true - y_pred) / y_true)
    mape_arr = mape_arr[np.isfinite(mape_arr)]
    mape_pct = float(np.mean(mape_arr)) * 100.0 if len(mape_arr) else float("nan")

    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")

    real_delta = y_true - y_anchor
    pred_delta = y_pred - y_anchor
    valid = np.isfinite(real_delta) & np.isfinite(pred_delta)
    mda_pct = (
        float(np.mean(np.sign(real_delta[valid]) == np.sign(pred_delta[valid]))) * 100.0
        if valid.any() else float("nan")
    )

    return {"rmse": rmse, "mae": mae, "mape_pct": mape_pct, "r2": r2, "mda_pct": mda_pct}


def train_model(
    df: pd.DataFrame, *, alpha: float = ALPHA, test_fraction: float = TEST_HOLDOUT_FRACTION,
) -> dict:
    """Fit a fresh StandardScaler+Ridge(alpha) pipeline on *df*.

    Uses a chronological tail holdout (sorted by 'date' if present, else by row order) to
    report test metrics — the AI team's original fixed train/val/test date cuts (2016-06 /
    2023-12 / 2024-12 / 2026-01) don't generalize to an arbitrary retraining CSV.

    Raises ValueError if required columns are missing or too few rows survive to train/test.
    """
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"El CSV de entrenamiento no trae las columnas requeridas: {missing}")

    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", kind="mergesort")
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)

    n = len(df)
    n_test = max(1, int(round(n * test_fraction)))
    if n - n_test < 2:
        raise ValueError(
            f"Histórico insuficiente para entrenar: {n} fila(s) útil(es) no permiten separar un "
            f"holdout de test ({n_test}) manteniendo al menos 2 filas de entrenamiento."
        )

    n_train = n - n_test
    train_df, test_df = df.iloc[:n_train], df.iloc[n_train:]

    model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=float(alpha)))])
    model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN].to_numpy(dtype=float))

    y_pred_test = np.asarray(model.predict(test_df[FEATURE_COLUMNS]), dtype=float)
    metrics = _regression_metrics(
        y_true=test_df[TARGET_COLUMN].to_numpy(dtype=float),
        y_pred=y_pred_test,
        y_anchor=test_df[ANCHOR_COLUMN].to_numpy(dtype=float),
    )

    logger.info(
        "ml15 train_model() done — n_train=%d n_test=%d rmse=%.4f mae=%.4f r2=%.4f",
        len(train_df), len(test_df), metrics["rmse"], metrics["mae"], metrics["r2"],
    )
    return {
        "model": model,
        "metrics": metrics,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
