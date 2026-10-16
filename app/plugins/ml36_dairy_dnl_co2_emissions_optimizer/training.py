"""Retraining logic for ml36 — the AI team's production training, ported faithfully.

Port of inbox/a36/codigo/:
- src/data_processing/split_normalization.py::run_split_normalization — split by cycle
  (``cycle_id`` or cycles rebuilt from ``t_ciclo`` resets), balanced by season when a season,
  month or date column exists, 70/15/15 per season with ``np.random.default_rng(42)``;
  StandardScaler on the features and MinMaxScaler on [T_out, CO2_emissions], both fitted on
  train only.
- src/training/trainer.py::run_training_pipeline with ``use_precomputed_best_config: true``
  (config.yaml) — a NEW DynamicMLP (1 x 64, ReLU, dropout 0.1) trained from scratch with seed
  ``training_model1.seed + final_model.seed_offset`` (1 + 500), Adam lr 1e-4, MSELoss,
  batch 128 (shuffled with a generator seeded the same), up to 400 epochs with early stopping
  (patience 30) on the full-batch validation loss, best weights restored; metrics on the test
  split in real units (mse, mae, r2 per target).

On the delivered data/processed/df_pasteurizacion_master.csv the split equals the delivered
train/val/test_raw.csv and the retrained model reproduces MLP_Final_Best_From_Search.pt and
models/metrics/final_test_metrics.json (tests/unit/test_ml36_training.py,
outputs/a36/verification_report.md).

Deliberate deviations:
- The 11 model inputs are the fixed FEATURE_ORDER, instead of split_normalization's "every
  column that is not a target, cycle, season, month or date column" — over a generic /train
  endpoint that would turn any extra column of the caller's CSV into a model input. On the
  delivered master dataset both give the same 11 columns in the same order. The 3 derived
  features are computed with the original formulas when the CSV does not bring them.
- Training always runs on CPU (the delivered model was; the README states the project is
  CPU-only) and ``torch.use_deterministic_algorithms`` is not set, because it is process-global
  and this is a shared server.
- If the last training batch would hold a single sample, it is dropped: BatchNorm cannot train
  on it and the original would crash.
- The global GA policy (ga_policy_global.json) is not rebuilt: optimize mode keeps the policy
  of the base artifacts (rebuilding it is the separate ``optimize_policy`` step, a 120 x 100 GA
  over the whole dataset).
"""
from __future__ import annotations

import copy
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.constants import (
    BASE_FEATURES,
    CYCLE_COL_CANDIDATES,
    DATE_COL_CANDIDATES,
    FEATURE_ORDER,
    MODEL_FILENAME,
    MONTH_COL_CANDIDATES,
    SEASON_COL_CANDIDATES,
    SPLIT_SEED,
    TARGETS,
    TRAIN_ARCHITECTURE,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_PATIENCE,
    TRAIN_SEED,
    TRAIN_SEED_OFFSET,
)
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.model_loader import build_model_from_config
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.preprocessing import add_derived_features

logger = logging.getLogger(__name__)

DERIVED_FEATURES = [f for f in FEATURE_ORDER if f not in BASE_FEATURES]


# ── Data and split (split_normalization.py) ───────────────────────────────────

def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the training CSV and complete the 3 derived features when absent."""
    missing = [c for c in BASE_FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"El CSV de entrenamiento no trae las columnas requeridas: {missing}")
    out = df.copy()
    absent = [c for c in DERIVED_FEATURES if c not in out.columns]
    if absent:
        out[absent] = add_derived_features(out)[absent]
    if out[FEATURE_ORDER + TARGETS].isna().any().any():
        raise ValueError("El CSV de entrenamiento tiene valores vacíos en las features o los targets.")
    return out


def detect_cycle_ids(df: pd.DataFrame) -> pd.Series:
    """_detect_cycle_ids: cycle column if present, else a new cycle on every t_ciclo reset."""
    cycle_col = next((c for c in CYCLE_COL_CANDIDATES if c in df.columns), None)
    if cycle_col is not None:
        return df[cycle_col].copy()
    return (df["t_ciclo"].diff().fillna(0) < 0).cumsum().astype(int)


def infer_season_label(df: pd.DataFrame) -> pd.Series:
    """_infer_season_label: season column, else from month or date, else 'unknown'."""
    season_col = next((c for c in SEASON_COL_CANDIDATES if c in df.columns), None)
    month_col = next((c for c in MONTH_COL_CANDIDATES if c in df.columns), None)
    date_col = next((c for c in DATE_COL_CANDIDATES if c in df.columns), None)
    bins, labels = [0, 2, 5, 8, 11, 12], ["winter", "spring", "summer", "autumn", "winter"]
    if season_col is not None:
        return df[season_col].astype(str)
    if month_col is not None:
        month = pd.to_numeric(df[month_col], errors="coerce")
    elif date_col is not None:
        month = pd.to_datetime(df[date_col], errors="coerce").dt.month
    else:
        return pd.Series(["unknown"] * len(df), index=df.index)
    return pd.cut(month, bins=bins, labels=labels, include_lowest=True, ordered=False).astype(str)


def split_cycles(cycle_meta: pd.DataFrame, seed: int = SPLIT_SEED) -> tuple[list, list, list]:
    """_split_cycles: 70/15/15 of the cycles of each season, shuffled with default_rng(seed)."""
    rng = np.random.default_rng(seed)
    train_cycles: list = []
    val_cycles: list = []
    test_cycles: list = []
    for _, group in cycle_meta.groupby("season_label"):
        group_cycles = group["cycle_id_for_split"].to_numpy(copy=True)
        rng.shuffle(group_cycles)
        n = len(group_cycles)
        if n == 1:
            train_cycles.extend(group_cycles)
            continue
        n_train = max(1, int(n * 0.70))
        n_temp = n - n_train
        if n_temp == 1:
            train_cycles.extend(group_cycles[:n_train])
            val_cycles.extend(group_cycles[n_train:n_train + 1])
            continue
        n_val = max(1, int(n_temp * 0.50))
        n_test = n_temp - n_val
        if n_test == 0:
            n_val = n_temp - 1
        train_cycles.extend(group_cycles[:n_train])
        val_cycles.extend(group_cycles[n_train:n_train + n_val])
        test_cycles.extend(group_cycles[n_train + n_val:])

    if len(val_cycles) == 0 and len(test_cycles) > 1:
        val_cycles.append(test_cycles.pop())
    if len(test_cycles) == 0 and len(val_cycles) > 1:
        test_cycles.append(val_cycles.pop())
    if len(val_cycles) == 0 and len(train_cycles) > 2:
        val_cycles.append(train_cycles.pop())
    if len(test_cycles) == 0 and len(train_cycles) > 2:
        test_cycles.append(train_cycles.pop())
    return train_cycles, val_cycles, test_cycles


def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train / validation / test rows of *df*, whole cycles together (run_split_normalization)."""
    cycle_ids = detect_cycle_ids(df)
    work = pd.DataFrame({"cycle_id_for_split": cycle_ids, "season_label": infer_season_label(df)})
    cycle_meta = (
        work.groupby("cycle_id_for_split")["season_label"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown")
        .reset_index()
    )
    if len(cycle_meta) < 3:
        raise ValueError(
            f"Se requieren al menos 3 ciclos para separar train/validación/test y solo hay {len(cycle_meta)} "
            "(columna cycle_id, o ciclos reconstruidos a partir de los reinicios de t_ciclo)."
        )
    train_c, val_c, test_c = split_cycles(cycle_meta)
    parts = tuple(df[cycle_ids.isin(c)].copy() for c in (train_c, val_c, test_c))
    if any(part.empty for part in parts):
        raise ValueError("No hay ciclos suficientes para que train, validación y test tengan datos.")
    return parts


# ── Training (trainer.py) ─────────────────────────────────────────────────────

def _set_training_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _real_unit_metrics(model, x_t, y_t, scaler_y) -> dict[str, dict[str, float]]:
    """_evaluate_real_units: mse/mae/r2 per target after inverse-scaling truth and prediction."""
    model.eval()
    with torch.no_grad():
        y_pred = scaler_y.inverse_transform(model(x_t).numpy())
    y_true = scaler_y.inverse_transform(y_t.numpy())
    return {
        col: {
            "mse": float(mean_squared_error(y_true[:, i], y_pred[:, i])),
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
        }
        for i, col in enumerate(TARGETS)
    }


def train_model(df: pd.DataFrame) -> dict:
    """Split, fit the scalers and train a new MLP; returns the model, its scalers and test metrics."""
    started = time.perf_counter()
    data = prepare_frame(df)
    train_df, val_df, test_df = split_frame(data)

    scaler_x = StandardScaler().fit(train_df[FEATURE_ORDER])
    scaler_y = MinMaxScaler().fit(train_df[TARGETS])

    def tensors(part: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        x = scaler_x.transform(part[FEATURE_ORDER])
        y = scaler_y.transform(part[TARGETS])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    x_train, y_train = tensors(train_df)
    x_val, y_val = tensors(val_df)
    x_test, y_test = tensors(test_df)

    config = {
        "model_name": MODEL_FILENAME.removesuffix(".pt"),
        "input_size": len(FEATURE_ORDER),
        "output_size": len(TARGETS),
        **TRAIN_ARCHITECTURE,
        "lr": TRAIN_LR,
        "epochs": TRAIN_EPOCHS,
        "patience": TRAIN_PATIENCE,
        "best_source": "notebook_best_config_1_layer",
    }

    final_seed = TRAIN_SEED + TRAIN_SEED_OFFSET
    _set_training_seed(final_seed)
    generator = torch.Generator()
    generator.manual_seed(final_seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0,
        generator=generator, drop_last=len(x_train) % TRAIN_BATCH_SIZE == 1 and len(x_train) > 1,
    )
    model = build_model_from_config(config)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    early_stop_count = 0
    epochs_run = 0
    for epoch in range(TRAIN_EPOCHS):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val), y_val).item())
        epochs_run = epoch + 1
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count >= TRAIN_PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()

    result = {
        "model": model,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "config": config,
        "metrics": _real_unit_metrics(model, x_test, y_test, scaler_y),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "epochs_run": epochs_run,
        "best_val_loss": best_loss,
        "training_time_s": round(time.perf_counter() - started, 1),
    }
    logger.info(
        "ml36 train_model() done — train/val/test=%d/%d/%d epochs=%d best_val_loss=%.6g test=%s",
        result["n_train"], result["n_val"], result["n_test"], epochs_run, best_loss, result["metrics"],
    )
    return result
