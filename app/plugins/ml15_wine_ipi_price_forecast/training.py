"""Retraining logic for ml15 — the AI team's production training, ported faithfully.

Port of inbox/a15/codigo/:
- src/data_processing/features.py::_generate_temporal_splits — frozen national splits by origin
  date: train = [first, val_start), validation = [val_start, val_end], test = [test_start, ...).
- src/training/trainer.py::train_model — refit of the deployed Pipeline(StandardScaler,
  Ridge(alpha=25.0)) on the train split, keeping only rows with target and anchor and whose
  target_date (origin + 6 months) falls inside the training period (target_date <= train_end);
  feature NaN/inf are filled with 0.0 exactly like _build_feature_matrix.
- src/get_stats/get_stats.py — test evaluation (rmse, mae, mape_pct, r2, mda_pct) against the
  persistence baseline (y_pred = y_anchor), on test rows with an observed target.

The original dates are fixed in config.yaml (train_end 2023-12, validation 2024-01..2024-12,
test 2025-01..2026-01). A retraining CSV covers other months, so the same structure is anchored
to the CSV's last origin month: test = its last TEST_MONTHS origin months, validation = the
VALIDATION_MONTHS before that, train_end = the month before validation starts. On the delivered
data/processed/model_ready_panel.csv (origins 2016-06..2026-01) this is exactly the original
split, and train_model reproduces horizon_6_ridge.pkl and final_evaluation.json bit for bit
(tests/unit/test_ml15_training.py, outputs/a15/verification_report.md).

One deliberate deviation: the 16 feature_columns of the delivered artifact are fixed, instead of
trainer.py's "every numeric column not excluded by feature_policy" — over a generic /train
endpoint that auto-selection would turn any extra numeric column of the caller's CSV into a model
feature (inbox/a15/manifest.yaml known_issues). On the delivered panels both select the same 16
columns in the same order.
"""
from __future__ import annotations

import logging
import time

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
    HORIZON,
    TARGET_COLUMN,
    TEST_MONTHS,
    VALIDATION_MONTHS,
)

logger = logging.getLogger(__name__)


def add_origin_date(df: pd.DataFrame) -> pd.DataFrame:
    """Port of trainer.py::_add_origin_date: origin_date, else date, else year+month (month start)."""
    out = df.copy()
    if "origin_date" in out.columns:
        out["origin_date"] = pd.to_datetime(out["origin_date"], errors="coerce")
    elif "date" in out.columns:
        out["origin_date"] = pd.to_datetime(out["date"], errors="coerce")
    else:
        missing = [c for c in ("year", "month") if c not in out.columns]
        if missing:
            raise ValueError(
                "El CSV de entrenamiento necesita 'origin_date', 'date' o 'year'+'month' para "
                f"ordenar el histórico; faltan: {missing}"
            )
        out["origin_date"] = pd.to_datetime(
            out["year"].astype(int).astype(str) + "-" + out["month"].astype(int).astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
    invalid = int(out["origin_date"].isna().sum())
    if invalid:
        raise ValueError(f"El CSV de entrenamiento tiene {invalid} fila(s) con fecha de origen inválida.")
    out["origin_date"] = out["origin_date"].dt.to_period("M").dt.to_timestamp()
    return out.sort_values("origin_date", kind="mergesort").reset_index(drop=True)


def split_dates(last_origin: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """Original config.yaml split structure anchored to the CSV's last origin month."""
    test_start = last_origin - pd.DateOffset(months=TEST_MONTHS - 1)
    val_start = test_start - pd.DateOffset(months=VALIDATION_MONTHS)
    return {
        "train_end": val_start - pd.DateOffset(months=1),
        "val_start": val_start,
        "test_start": test_start,
        "test_end": last_origin,
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_anchor: np.ndarray) -> dict[str, float | None]:
    """Port of get_stats.py::_regression_metrics (None where the original yields NaN)."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape_arr = np.abs((y_true - y_pred) / y_true)
    mape_arr = mape_arr[np.isfinite(mape_arr)]
    mape = float(np.mean(mape_arr)) if len(mape_arr) else float("nan")
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    real_delta, pred_delta = y_true - y_anchor, y_pred - y_anchor
    valid = np.isfinite(real_delta) & np.isfinite(pred_delta)
    mda = float(np.mean(np.sign(real_delta[valid]) == np.sign(pred_delta[valid]))) if valid.any() else float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "mape_pct": mape * 100.0 if np.isfinite(mape) else None,
        "r2": r2 if np.isfinite(r2) else None,
        "mda_pct": mda * 100.0 if np.isfinite(mda) else None,
        "n_observations": int(len(y_true)),
    }


def _reduction_pct(model_value: float | None, baseline_value: float | None) -> float | None:
    """Port of get_stats.py::_safe_reduction_pct."""
    if model_value is None or baseline_value is None or not baseline_value:
        return None
    return float((baseline_value - model_value) / baseline_value * 100.0)


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Port of trainer.py/predictor.py::_build_feature_matrix."""
    return df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def train_model(df: pd.DataFrame, *, alpha: float = ALPHA) -> dict:
    """Fit the deployed Pipeline on the train split and evaluate it on the test split.

    Raises ValueError for a CSV without the required columns, or too short to leave training
    rows before the validation and test windows.
    """
    started = time.perf_counter()
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN, ANCHOR_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"El CSV de entrenamiento no trae las columnas requeridas: {missing}")

    data = add_origin_date(df)
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data[ANCHOR_COLUMN] = pd.to_numeric(data[ANCHOR_COLUMN], errors="coerce")
    dates = split_dates(data["origin_date"].max())

    train = data[data["origin_date"] < dates["val_start"]].copy()
    train["target_date"] = train["origin_date"] + pd.DateOffset(months=HORIZON)
    train = train.dropna(subset=[TARGET_COLUMN, ANCHOR_COLUMN])
    train = train[train["target_date"] <= dates["train_end"]]
    if len(train) < 2:
        raise ValueError(
            f"Histórico insuficiente: quedan {len(train)} fila(s) de entrenamiento. El procedimiento "
            f"original reserva los últimos {TEST_MONTHS} meses de origen para test y los "
            f"{VALIDATION_MONTHS} anteriores para validación, y solo entrena con filas cuyo objetivo "
            f"(origen + {HORIZON} meses) cae antes de la validación (hasta {dates['train_end'].date()})."
        )

    model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=float(alpha)))])
    model.fit(_feature_matrix(train), train[TARGET_COLUMN].to_numpy(dtype=float))

    test = data[data["origin_date"] >= dates["test_start"]].dropna(subset=[ANCHOR_COLUMN])
    evaluable = test.dropna(subset=[TARGET_COLUMN])
    metrics: dict[str, float | None] = dict.fromkeys(("rmse", "mae", "mape_pct", "r2", "mda_pct"))
    baseline: dict[str, float | None] = dict.fromkeys(("rmse", "mae", "mape_pct", "r2", "mda_pct"))
    if not evaluable.empty:
        y_true = evaluable[TARGET_COLUMN].to_numpy(dtype=float)
        y_anchor = evaluable[ANCHOR_COLUMN].to_numpy(dtype=float)
        y_pred = np.asarray(model.predict(_feature_matrix(evaluable)), dtype=float)
        metrics = regression_metrics(y_true, y_pred, y_anchor)
        baseline = regression_metrics(y_true, y_anchor, y_anchor)

    result = {
        "model": model,
        "metrics": {k: metrics[k] for k in ("rmse", "mae", "mape_pct", "r2", "mda_pct")},
        "baseline_persistence": {k: baseline[k] for k in ("rmse", "mae", "mape_pct", "r2", "mda_pct")},
        "rmse_reduction_pct": _reduction_pct(metrics["rmse"], baseline["rmse"]),
        "mape_reduction_pct": _reduction_pct(metrics["mape_pct"], baseline["mape_pct"]),
        "n_train": int(len(train)),
        "n_test": int(len(evaluable)),
        "n_test_rows": int(len(test)),
        "train_origin_start": train["origin_date"].min(),
        "train_origin_end": train["origin_date"].max(),
        "train_end_date": dates["train_end"],
        "test_start_date": dates["test_start"],
        "test_end_date": dates["test_end"],
        "training_time_s": round(time.perf_counter() - started, 3),
    }
    logger.info(
        "ml15 train_model() done — n_train=%d (origins %s..%s) n_test=%d/%d rmse=%s",
        result["n_train"], result["train_origin_start"].date(), result["train_origin_end"].date(),
        result["n_test"], result["n_test_rows"], metrics["rmse"],
    )
    return result
