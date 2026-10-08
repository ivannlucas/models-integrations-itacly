"""ml18 /train — port of the AI team's training procedure (src/main.py::train() +
src/training/trainer.py + src/data_processing/preprocess.py, config/config.yaml).

Same steps, same order, same hyperparameters (constants.TRAIN_*):

1. Read the modelling CSV (``;``-separated, the format of
   data/processed/model_ready/dataset_modelado_desde_2008.csv) and resolve the target column.
2. Feature engineering shared with inference: time features, spatial lags of neighbouring
   CCAA, own-price alias, forward-fill + median imputation of features and target.
3. Temporal cut-offs over the unique dates (70 / 15 / 15), StandardScaler for X and y fitted on
   the train dates only.
4. Every 12-month window per (CCAA, Producto), split by target date.
5. GRU(96) → Dropout(0.2) → Dense(32, relu) → Dense(1), Adam(1e-3), MSE; 60 epochs, batch 32,
   EarlyStopping(val_loss, patience 8, restore_best_weights).
6. MAE / RMSE / MAPE / R² on train, val and test in euros (inverse-scaled).

Plugin adaptations, all deliberate:

* The data comes from /train's ``data_path`` instead of config.yaml's fixed ``dataset_path``.
  That fixed path was the reason the plugin used to declare training unsupported; the procedure
  itself has nothing tied to that file.
* Seeds via ``keras.utils.set_random_seed(42)`` (python, numpy, TF), as scripts/_seed.py does.
  ``TF_DETERMINISTIC_OPS`` is not set: it is process-wide and this is a shared server.
* The model is not written to models/artifacts/: the caller uploads it to MLflow.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from app.plugins.ml18_meat_spatial_price_forecast import preprocessing
from app.plugins.ml18_meat_spatial_price_forecast.constants import (
    BEST_CONFIG_FILENAME,
    CCAA_COL,
    DATE_COL,
    DENSE_UNITS,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    FEATURE_COLUMNS,
    HIDDEN_UNITS,
    LEARNING_RATE,
    LOOKBACK,
    LOSS,
    METRICS_FILENAME,
    MODEL_FILENAME,
    PRODUCTO_COL,
    TARGET_COL,
    TRAIN_BASE_REQUIRED_COLS,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_RATIO,
    TRAIN_SEED,
    VAL_RATIO,
    X_SCALER_FILENAME,
    Y_SCALER_FILENAME,
)

logger = logging.getLogger(__name__)

TARGET_SCALED = "TARGET_SCALED"


@dataclass
class TrainingResult:
    """A retrained GRU with its scalers and the train/val/test metrics of the original."""

    model: Any
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    metrics: dict


# ── data ──────────────────────────────────────────────────────────────────────

def load_training_frame(csv_path: str) -> pd.DataFrame:
    """load_processed_dataset + the column checks of src/main.py::train()."""
    df = pd.read_csv(csv_path, sep=";")
    if not preprocessing.resolve_target_column(df):
        raise ValueError(
            f"No se pudo resolver una columna válida para el target '{TARGET_COL}'. El CSV de "
            "entrenamiento debe traer el precio observado (separador ';')."
        )
    missing = [c for c in TRAIN_BASE_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas en el CSV de entrenamiento: {missing}. Formato esperado: el de "
            "dataset_modelado_desde_2008.csv del equipo de IA, con separador ';'."
        )
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if df[DATE_COL].isna().any():
        raise ValueError(f"La columna '{DATE_COL}' tiene valores no parseables (formato YYYY-MM-DD).")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering of src/main.py::train(), in the same order."""
    df = preprocessing.add_time_features(df)
    df = preprocessing.add_spatial_lags(df)
    df = preprocessing.add_own_price_feature(df)
    df = preprocessing.fill_numeric(df, list(FEATURE_COLUMNS) + [TARGET_COL])
    if df[TARGET_COL].isna().all():
        raise ValueError(f"La columna target '{TARGET_COL}' sigue completamente vacía tras la imputación.")
    return df


def temporal_cutoffs(df: pd.DataFrame) -> tuple[np.datetime64, np.datetime64]:
    """preprocess.py::temporal_cutoffs — cut-offs over the sorted unique dates."""
    dates = np.array(sorted(df[DATE_COL].unique()))
    n_dates = len(dates)
    train_idx = int(n_dates * TRAIN_RATIO) - 1
    val_idx = int(n_dates * (TRAIN_RATIO + VAL_RATIO)) - 1
    if train_idx < 0 or val_idx <= train_idx or val_idx >= n_dates - 1:
        raise ValueError(
            f"El CSV tiene {n_dates} fechas distintas: no alcanza para el split temporal "
            "70/15/15 con validación y test no vacíos."
        )
    return dates[train_idx], dates[val_idx]


def fit_scaler_train_only(df: pd.DataFrame, cols: list[str], train_end) -> StandardScaler:
    return StandardScaler().fit(df.loc[df[DATE_COL] <= train_end, cols])


def build_training_sequences(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """preprocess.py::build_sequences — every LOOKBACK window per (CCAA, Producto)."""
    x_list, y_list, dates_list = [], [], []
    work = df.sort_values([CCAA_COL, PRODUCTO_COL, DATE_COL])
    for _, g in work.groupby([CCAA_COL, PRODUCTO_COL]):
        g = g.reset_index(drop=True)
        if len(g) <= LOOKBACK:
            continue
        vals = g[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        tgt = g[TARGET_SCALED].to_numpy(dtype=np.float32)
        dates = g[DATE_COL].to_numpy(dtype="datetime64[ns]")
        for i in range(LOOKBACK, len(g)):
            x_list.append(vals[i - LOOKBACK:i])
            y_list.append(tgt[i])
            dates_list.append(dates[i])
    if not x_list:
        raise ValueError(
            f"No se generó ninguna secuencia: cada combinación CCAA-Producto necesita más de "
            f"{LOOKBACK} meses."
        )
    return (np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.float32),
            np.array(dates_list, dtype="datetime64[ns]"))


def split_by_date(x, y, dates, train_end, val_end):
    """preprocess.py::split_by_date."""
    train_end, val_end = np.datetime64(train_end), np.datetime64(val_end)
    train, test = dates <= train_end, dates > val_end
    val = ~train & ~test
    parts = (x[train], y[train], x[val], y[val], x[test], y[test])
    if min(len(parts[0]), len(parts[2]), len(parts[4])) == 0:
        raise ValueError(
            f"Split temporal sin muestras suficientes: train={len(parts[0])}, val={len(parts[2])}, "
            f"test={len(parts[4])} ventanas de {LOOKBACK} meses."
        )
    return parts


# ── model ─────────────────────────────────────────────────────────────────────

def build_gru_model(input_shape: tuple[int, int]):
    """trainer.py::build_gru_model with config.yaml training.model."""
    import tensorflow as tf  # pylint: disable=import-outside-toplevel
    from tensorflow.keras import layers, models  # pylint: disable=import-outside-toplevel

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(HIDDEN_UNITS, return_sequences=False),
        layers.Dropout(DROPOUT),
        layers.Dense(DENSE_UNITS, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse" if LOSS == "mse" else tf.keras.losses.Huber(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                 tf.keras.metrics.MeanAbsolutePercentageError(name="mape")],
    )
    return model


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    """trainer.py::regression_metrics (MAPE with the same eps), keys prefixed per split."""
    return {
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_mape_pct": _mape(y_true, y_pred),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
    }


def train_gru(
    csv_path: str,
    *,
    epochs: int = TRAIN_EPOCHS,
    on_epoch: Callable[[int, dict], None] | None = None,
) -> TrainingResult:
    """Run the delivered training procedure on *csv_path* and return the new model."""
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    start = time.time()
    tf.keras.utils.set_random_seed(TRAIN_SEED)

    df = engineer_features(load_training_frame(csv_path))
    train_end, val_end = temporal_cutoffs(df)
    x_scaler = fit_scaler_train_only(df, list(FEATURE_COLUMNS), train_end)
    df[list(FEATURE_COLUMNS)] = x_scaler.transform(df[list(FEATURE_COLUMNS)])
    y_scaler = fit_scaler_train_only(df, [TARGET_COL], train_end)
    df[TARGET_SCALED] = y_scaler.transform(df[[TARGET_COL]])
    if not np.isfinite(df[TARGET_SCALED].to_numpy(dtype=np.float64)).all():
        raise ValueError("El target escalado contiene NaN/Inf: revisa el precio del CSV.")

    x, y, dates = build_training_sequences(df)
    x_train, y_train, x_val, y_val, x_test, y_test = split_by_date(x, y, dates, train_end, val_end)
    logger.info("ml18 train: X_train=%s X_val=%s X_test=%s", x_train.shape, x_val.shape, x_test.shape)

    model = build_gru_model((x_train.shape[1], x_train.shape[2]))
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True,
    )]
    if on_epoch is not None:
        callbacks.append(tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: on_epoch(epoch, {k: float(v) for k, v in (logs or {}).items()}),
        ))
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
        batch_size=TRAIN_BATCH_SIZE, verbose=0, callbacks=callbacks,
    )

    metrics: dict = {}
    for prefix, xs, ys in (("train", x_train, y_train), ("val", x_val, y_val), ("test", x_test, y_test)):
        pred = y_scaler.inverse_transform(model.predict(xs, verbose=0).reshape(-1, 1)).ravel()
        real = y_scaler.inverse_transform(ys.reshape(-1, 1)).ravel()
        metrics.update(regression_metrics(real, pred, prefix))
    val_losses = history.history["val_loss"]
    metrics.update({
        "n_train": int(len(x_train)), "n_val": int(len(x_val)), "n_test": int(len(x_test)),
        "epochs_run": len(val_losses), "best_epoch": int(np.argmin(val_losses)) + 1,
        "training_time_s": round(time.time() - start, 1),
    })
    logger.info("ml18 training done: %s", metrics)
    return TrainingResult(model, x_scaler, y_scaler, metrics)


# ── artifacts ─────────────────────────────────────────────────────────────────

def _scaler_to_json(scaler: StandardScaler, path: Path) -> None:
    """preprocess.py::save_scaler_to_json."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                   "var": scaler.var_.tolist()}, fh, indent=2)


def save_training_artifacts(artifact_dir: Path, result: TrainingResult) -> None:
    """trainer.py::save_model_artifacts — same files the base artifact and the loader use."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model_json": result.model.to_json(), "weights": result.model.get_weights()},
                artifact_dir / MODEL_FILENAME)
    _scaler_to_json(result.x_scaler, artifact_dir / X_SCALER_FILENAME)
    _scaler_to_json(result.y_scaler, artifact_dir / Y_SCALER_FILENAME)
    with open(artifact_dir / BEST_CONFIG_FILENAME, "w", encoding="utf-8") as fh:
        json.dump({"type": "GRU", "units": HIDDEN_UNITS, "dropout": DROPOUT, "dense_units": DENSE_UNITS,
                   "learning_rate": LEARNING_RATE, "loss": LOSS}, fh, indent=2)
    with open(artifact_dir / METRICS_FILENAME, "w", encoding="utf-8") as fh:
        json.dump({**result.metrics, "evaluated_on": "train+val+test_splits"}, fh, indent=2)
