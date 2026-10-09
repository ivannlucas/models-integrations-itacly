"""ml14 /train — port of the GRU part of src/training/compare_models.py::run_comparison
(inbox/a14/codigo, config/config.py defaults).

Same steps and hyperparameters as the delivered code:

1. 16-week horizon target and the Drift baseline (prepare_horizon_dataset); the GRU learns the
   residual over the Drift baseline (target_mode "drift_residual_h").
2. Temporal split 80/20 by date (split_temporal); StandardScaler for X and for the residual,
   both fitted on train.
3. 12-week sequences; the test ones bridge the last 11 train weeks
   (build_train_sequences / build_test_sequences_with_bridge).
4. GRU(64, 2 layers, dropout 0.3), Adam(1e-3, weight_decay 1e-5), ReduceLROnPlateau(0.5, 10),
   grad-clip 1.0, MSE, up to 150 epochs, batch 32, patience 20 (train_rnn).
5. Seeds 42, 43, 44 (train_rnn_multi_seed): the metrics are their mean ± std and the model kept
   is always seed 42's, never the best one on test.

Not ported: XGBoost and LSTM. They are the original's model comparison, which picked GRU; /train
refits the deployed architecture (same rule as ml23 and ml30).

Faithful to a known weakness of the original: train_rnn computes its "val_loss" on the TEST
sequences, so early stopping and the restored best weights are chosen on test, and the test
metrics are optimistic. It is reproduced, not fixed, so the retrained model is comparable with
the delivered one; see inbox/a14/manifest.yaml known_issues.

Input: the AI team's training dataset (FECHA + the 39 engineered columns, as
data/processed/final_dataset_for_modeling.csv) or the raw weekly series /predict accepts
(date + 6 series). Raw input goes through the same feature engineering as inference, which drops
the first 12 weeks (lag warm-up) that the delivered dataset keeps.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from app.plugins.ml14_wine_phyto_price_forecast.constants import (
    DATE_COL,
    DRIFT_COL,
    HIDDEN_SIZE,
    HORIZON_WEEKS,
    RAW_VALUE_COLS,
    REFERENCE_DATE_COL,
    SEQ_LEN,
    TARGET_SERIES_COL,
    TEST_RATIO,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_N_SEEDS,
    TRAIN_PATIENCE,
    TRAIN_SCHEDULER_FACTOR,
    TRAIN_SCHEDULER_PATIENCE,
    TRAIN_SEED_BASE,
    TRAIN_WEIGHT_DECAY,
)
from app.plugins.ml14_wine_phyto_price_forecast.feature_engineering import build_modeling_features
from app.plugins.ml14_wine_phyto_price_forecast.rnn_models import GRUModel

TARGET_COL = "target_h"
CURRENT_PRICE_COL = "current_price"
DRIFT_BASELINE_COL = "drift_baseline_h"
DELTA_TARGET_COL = "target_delta_h"
RESIDUAL_TARGET_COL = "target_residual_h"
EXCLUDE_COLS = {REFERENCE_DATE_COL, TARGET_COL, DELTA_TARGET_COL, RESIDUAL_TARGET_COL,
                CURRENT_PRICE_COL, DRIFT_BASELINE_COL}


@dataclass
class TrainingResult:
    """The seed-42 GRU, what inference needs to use it, and the original's metrics."""

    model: nn.Module
    feature_columns: list[str]
    input_scaler: StandardScaler
    target_scaler: StandardScaler
    metrics: dict


# ── data ──────────────────────────────────────────────────────────────────────

def load_training_frame(csv_path: str, expected_features: list[str]) -> pd.DataFrame:
    """Return the modelling frame (FECHA + features), from either accepted format."""
    df = pd.read_csv(csv_path)
    if REFERENCE_DATE_COL in df.columns:
        missing = [c for c in expected_features if c not in df.columns]
        if missing:
            raise ValueError(
                f"El CSV de entrenamiento (formato final_dataset_for_modeling) no trae las columnas "
                f"{missing}."
            )
    elif {DATE_COL, *RAW_VALUE_COLS} <= set(df.columns):
        df = build_modeling_features(df.to_dict("records")).rename(columns={DATE_COL: REFERENCE_DATE_COL})
    else:
        raise ValueError(
            "El CSV de entrenamiento debe ser el dataset de modelado del equipo de IA (FECHA + "
            f"features) o las series semanales en bruto ({DATE_COL} + {list(RAW_VALUE_COLS)})."
        )
    df[REFERENCE_DATE_COL] = pd.to_datetime(df[REFERENCE_DATE_COL], errors="coerce")
    if df[REFERENCE_DATE_COL].isna().any():
        raise ValueError(f"La columna '{REFERENCE_DATE_COL}' tiene fechas no parseables.")
    return df.sort_values(REFERENCE_DATE_COL).reset_index(drop=True)[[REFERENCE_DATE_COL, *expected_features]]


def prepare_horizon_dataset(df: pd.DataFrame, horizon: int = HORIZON_WEEKS) -> pd.DataFrame:
    """compare_models.py::prepare_horizon_dataset."""
    result = df.copy()
    result[TARGET_COL] = result[TARGET_SERIES_COL].shift(-horizon)
    result[CURRENT_PRICE_COL] = result[TARGET_SERIES_COL]
    result[DRIFT_BASELINE_COL] = result[CURRENT_PRICE_COL] + result[DRIFT_COL] * (horizon / 4.0)
    result[DELTA_TARGET_COL] = result[TARGET_COL] - result[CURRENT_PRICE_COL]
    result[RESIDUAL_TARGET_COL] = result[TARGET_COL] - result[DRIFT_BASELINE_COL]
    return result.dropna(subset=[TARGET_COL]).copy()


def split_temporal(df: pd.DataFrame, test_ratio: float = TEST_RATIO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """compare_models.py::split_temporal."""
    fechas = sorted(df[REFERENCE_DATE_COL].unique())
    n_test = max(1, int(len(fechas) * test_ratio))
    fecha_corte = fechas[-n_test]
    return (df[df[REFERENCE_DATE_COL] < fecha_corte].copy(),
            df[df[REFERENCE_DATE_COL] >= fecha_corte].copy())


def build_train_sequences(df, feature_cols, scaler):
    """compare_models.py::build_train_sequences."""
    x_scaled = scaler.transform(df[feature_cols].values)
    y_raw = df[RESIDUAL_TARGET_COL].values
    xs = [x_scaled[i:i + SEQ_LEN] for i in range(len(x_scaled) - SEQ_LEN + 1)]
    ys = [y_raw[i + SEQ_LEN - 1] for i in range(len(x_scaled) - SEQ_LEN + 1)]
    return np.array(xs), np.array(ys)


def build_test_sequences_with_bridge(train_df, test_df, feature_cols, scaler):
    """compare_models.py::build_test_sequences_with_bridge (X, residual, current, drift, future)."""
    bridge = train_df.tail(SEQ_LEN - 1)
    combined = pd.concat([bridge, test_df], ignore_index=True)
    x_scaled = scaler.transform(combined[feature_cols].values)
    cols = [combined[c].values for c in (RESIDUAL_TARGET_COL, CURRENT_PRICE_COL, DRIFT_BASELINE_COL, TARGET_COL)]
    out = [[], [], [], [], []]
    for i in range(len(x_scaled) - SEQ_LEN + 1):
        target_idx = i + SEQ_LEN - 1
        if target_idx >= len(bridge):
            out[0].append(x_scaled[i:i + SEQ_LEN])
            for k, col in enumerate(cols, start=1):
                out[k].append(col[target_idx])
    return tuple(np.array(a) for a in out)


# ── metrics ───────────────────────────────────────────────────────────────────

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """compare_models.py::calc_metrics."""
    mask = y_true != 0
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": round(float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100), 2),
        "R2": float(r2_score(y_true, y_pred)),
    }


def calc_skill_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> dict:
    """compare_models.py::calc_skill_metrics."""
    delta_true, delta_pred = y_true - y_naive, y_pred - y_naive
    nonzero = delta_true != 0
    direction = 50.0 if nonzero.sum() == 0 else float(
        np.mean(np.sign(delta_pred[nonzero]) == np.sign(delta_true[nonzero])) * 100)
    mse_model, mse_naive = float(np.mean((y_true - y_pred) ** 2)), float(np.mean((y_true - y_naive) ** 2))
    return {"direction_acc_%": round(direction, 1),
            "skill_score": round(1.0 - mse_model / mse_naive if mse_naive > 0 else 0.0, 4)}


# ── GRU ───────────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_rnn(model, train_x, train_y, test_x, test_y, test_drift, test_future, target_scaler, *,
              epochs: int, device: torch.device):
    """compare_models.py::train_rnn (early stopping on the test sequences, as delivered)."""
    model = model.to(device)
    train_ds = TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
    train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_t = torch.tensor(test_x, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=TRAIN_SCHEDULER_FACTOR, patience=TRAIN_SCHEDULER_PATIENCE)
    criterion = nn.MSELoss()

    best_val, best_state, wait, epochs_run = float("inf"), None, 0, 0
    for epochs_run in range(1, epochs + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean((model(test_t).cpu().numpy() - test_y) ** 2))
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val, best_state, wait = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= TRAIN_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds_scaled = model(test_t).cpu().numpy().reshape(-1, 1)
    preds = test_drift + target_scaler.inverse_transform(preds_scaled).ravel()
    return calc_metrics(test_future, preds), preds, epochs_run


def train_gru(csv_path: str, expected_features: list[str], *, epochs: int = TRAIN_EPOCHS) -> TrainingResult:
    """Run the delivered GRU training on *csv_path* (seed-42 model + mean ± std metrics)."""
    start = time.time()
    device = torch.device("cpu")   # determinism; the delivered run used CUDA if available
    df = prepare_horizon_dataset(load_training_frame(csv_path, expected_features))
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    train_df, test_df = split_temporal(df)
    if len(train_df) < SEQ_LEN or len(test_df) == 0:
        raise ValueError(
            f"Histórico insuficiente para el split temporal 80/20 con secuencias de {SEQ_LEN} semanas "
            f"(train={len(train_df)}, test={len(test_df)} filas con objetivo a {HORIZON_WEEKS} semanas)."
        )

    scaler = StandardScaler().fit(train_df[feature_cols].values)
    target_scaler = StandardScaler().fit(train_df[[RESIDUAL_TARGET_COL]].values)
    train_x, train_y = build_train_sequences(train_df, feature_cols, scaler)
    test_x, test_y, test_cp, test_drift, test_future = build_test_sequences_with_bridge(
        train_df, test_df, feature_cols, scaler)
    train_y = target_scaler.transform(train_y.reshape(-1, 1)).ravel()
    test_y = target_scaler.transform(test_y.reshape(-1, 1)).ravel()

    per_seed, canonical = [], None
    for seed in range(TRAIN_SEED_BASE, TRAIN_SEED_BASE + TRAIN_N_SEEDS):
        _set_seed(seed)
        model = GRUModel(len(feature_cols), hidden_size=HIDDEN_SIZE)
        metrics, preds, epochs_run = train_rnn(
            model, train_x, train_y, test_x, test_y, test_drift, test_future, target_scaler,
            epochs=epochs, device=device)
        per_seed.append(metrics)
        if seed == TRAIN_SEED_BASE:
            canonical = (model.cpu(), preds, epochs_run)

    model, preds, epochs_run = canonical
    out = {}
    for key, name in (("MAE", "mae"), ("RMSE", "rmse"), ("MAPE_%", "mape_pct"), ("R2", "r2")):
        vals = [m[key] for m in per_seed]
        out[name], out[f"{name}_std"] = round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)
    skill = calc_skill_metrics(test_future, preds, test_cp)
    drift = calc_metrics(test_df[TARGET_COL].values, test_df[DRIFT_BASELINE_COL].values)
    naive = calc_metrics(test_df[TARGET_COL].values, test_df[CURRENT_PRICE_COL].values)
    out.update({
        "direction_acc_pct": skill["direction_acc_%"], "skill_score": skill["skill_score"],
        "drift_rmse": drift["RMSE"], "naive_rmse": naive["RMSE"],
        "beats_drift": out["rmse"] < drift["RMSE"],
        "n_train": int(len(train_x)), "n_test": int(len(test_x)),
        "epochs_run_seed_42": int(epochs_run), "n_seeds": TRAIN_N_SEEDS,
        "training_time_s": round(time.time() - start, 1),
    })
    return TrainingResult(model, feature_cols, scaler, target_scaler, out)
