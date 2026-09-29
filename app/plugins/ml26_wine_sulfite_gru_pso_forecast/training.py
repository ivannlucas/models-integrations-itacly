"""Fine-tuning of the served GRU-PSO model on a user CSV (manifest training block).

Reuses the AI team's recipe from src/training/models/train_sequence.py::train_sequence_with_config
(AdamW, SmoothL1Loss on z-scored targets, ReduceLROnPlateau on validation RMSE, gradient clipping,
early stopping) and their data preparation (hold-out split by lot_id with seed 42, windows of 24
with stride 2, target taken at the last step of each window). Differences, documented in
manifest known_issues KI-04: starts from the served weights (fine-tuning, not from scratch), keeps
the served normalization statistics, and does not re-run the PSO hyperparameter search.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.plugins.ml26_wine_sulfite_gru_pso_forecast._vendor.feature_engineering import (
    build_feature_frame,
)
from app.plugins.ml26_wine_sulfite_gru_pso_forecast._vendor.gru_model import build_gru_from_config
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.constants import (
    PIPELINE_CONFIG,
    SPLIT_SEED,
    STRIDE,
    TRAINING_SEED,
    WINDOW,
)
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.model_loader import LoadedModel
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.postprocessing import constrain_predictions

logger = logging.getLogger(__name__)


@dataclass
class FineTuneResult:
    """Outcome of a fine-tuning run."""

    model: LoadedModel
    n_lots: dict[str, int]
    n_windows: dict[str, int]
    epochs_run: int
    best_epoch: int
    val_report: dict[str, Any]
    test_report: dict[str, Any] | None


def set_global_seed(seed: int) -> None:
    """Seed python/numpy/torch like src/utils/common.py::set_global_seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_lot_ids(frame: pd.DataFrame, seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    """Hold-out split by lot_id (src/data_processing/preprocessing/splitters.py)."""
    unique_ids = sorted(frame["lot_id"].unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)
    n_total = len(unique_ids)
    n_train = int(n_total * float(PIPELINE_CONFIG["data"]["train_split"]))
    n_val = int(n_total * float(PIPELINE_CONFIG["data"]["val_split"]))
    return {
        "train": unique_ids[:n_train],
        "val": unique_ids[n_train:n_train + n_val],
        "test": unique_ids[n_train + n_val:],
    }


def build_windows(
    frame: pd.DataFrame, feature_names: list[str], target_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows per lot (dataset_builder._windows_for_split)."""
    x_windows: list[np.ndarray] = []
    y_windows: list[np.ndarray] = []
    for _, lot_frame in frame.groupby("lot_id", sort=False):
        lot_frame = lot_frame.sort_values("timestamp_index")
        if len(lot_frame) < WINDOW:
            continue
        values = lot_frame[feature_names].to_numpy(dtype=np.float32)
        targets = lot_frame[target_names].to_numpy(dtype=np.float32)
        for start in range(0, len(lot_frame) - WINDOW + 1, STRIDE):
            end = start + WINDOW
            x_windows.append(values[start:end])
            y_windows.append(targets[end - 1])
    if not x_windows:
        return np.empty((0, WINDOW, len(feature_names)), dtype=np.float32), np.empty(
            (0, len(target_names)), dtype=np.float32
        )
    return np.stack(x_windows), np.stack(y_windows)


def regression_report(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]
) -> dict[str, Any]:
    """MAE/RMSE per target + overall (src/training/evaluation/metrics.py::regression_report)."""
    report: dict[str, Any] = {}
    for index, target in enumerate(target_names):
        err = y_pred[:, index] - y_true[:, index]
        report[target] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
        }
    report["overall_mae"] = float(np.mean(np.abs(y_pred - y_true)))
    report["overall_rmse"] = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return report


def _predict_denorm(
    network: nn.Module, x_norm: np.ndarray, base: LoadedModel, batch_size: int
) -> np.ndarray:
    network.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(x_norm), batch_size):
            outs.append(network(torch.from_numpy(x_norm[start:start + batch_size])).numpy())
    preds = np.concatenate(outs) * base.y_std + base.y_mean
    return constrain_predictions(preds, base.target_names)


def prepare_training_data(
    raw_df: pd.DataFrame, base: LoadedModel
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Feature engineering with targets + split + windows. Returns {split: (X, y, n_lots)}."""
    feature_frame, feature_columns, target_columns = build_feature_frame(
        raw_df, PIPELINE_CONFIG, include_targets=True
    )
    if list(target_columns) != list(base.target_names):
        raise ValueError(
            f"Targets derivados {target_columns} no coinciden con los del modelo "
            f"{base.target_names}"
        )
    missing = [c for c in base.feature_names if c not in feature_columns]
    for (
        column
    ) in missing:  # e.g. a stage never seen in the user CSV -> its one-hot column is absent
        feature_frame[column] = 0.0
    split_ids = split_lot_ids(feature_frame)
    data = {}
    for split, ids in split_ids.items():
        frame = feature_frame[feature_frame["lot_id"].isin(ids)]
        x, y = build_windows(frame, base.feature_names, base.target_names)
        data[split] = (x, y, len(ids))
    return data


def fine_tune(  # pylint: disable=too-many-locals,too-many-statements
    base: LoadedModel,
    data: dict[str, tuple[np.ndarray, np.ndarray, int]],
    hyperparams: dict[str, Any],
    on_epoch=None,
) -> FineTuneResult:
    """Fine-tune a CLONE of base.network; base is never mutated."""
    x_train, y_train, _ = data["train"]
    x_val, y_val, _ = data["val"]
    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError(
            "No hay ventanas suficientes para entrenar/validar: se necesitan >= 7 lotes con >= "
            f"{WINDOW} pasos operativos con target a 72 h (train={len(x_train)}, "
            f"val={len(x_val)} ventanas)."
        )
    set_global_seed(TRAINING_SEED)
    batch_size = int(hyperparams["batch_size"])

    def norm_x(x: np.ndarray) -> np.ndarray:
        return ((x - base.mean) / base.std).astype(np.float32)

    def norm_y(y: np.ndarray) -> np.ndarray:
        return ((y - base.y_mean) / base.y_std).astype(np.float32)

    network = build_gru_from_config(
        base.config, input_dim=len(base.feature_names), output_dim=len(base.target_names)
    )
    network.load_state_dict(copy.deepcopy(base.network.state_dict()))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(norm_x(x_train)), torch.from_numpy(norm_y(y_train))),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(hyperparams["learning_rate"]),
        weight_decay=float(hyperparams["weight_decay"]),
    )
    criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(hyperparams["lr_decay_factor"]),
        patience=int(hyperparams["lr_patience"]),
    )
    x_val_norm = norm_x(x_val)
    best_state = copy.deepcopy(network.state_dict())
    best_rmse, best_epoch, epochs_run = float("inf"), 0, 0
    patience_left = int(hyperparams["patience"])
    for epoch in range(1, int(hyperparams["epochs"]) + 1):
        network.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(network(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                network.parameters(), max_norm=float(hyperparams["gradient_clip"])
            )
            optimizer.step()
            losses.append(loss.item())
        val_pred = _predict_denorm(network, x_val_norm, base, batch_size)
        val_rmse = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))
        scheduler.step(val_rmse)
        epochs_run = epoch
        train_loss = float(np.mean(losses)) if losses else 0.0
        logger.info(
            "ml26 fine-tune epoch=%d train_loss=%.6f val_rmse=%.6f", epoch, train_loss, val_rmse
        )
        if on_epoch:
            on_epoch(epoch, train_loss, val_rmse)
        if val_rmse + 1e-6 < best_rmse:
            best_rmse, best_epoch = val_rmse, epoch
            best_state = copy.deepcopy(network.state_dict())
            patience_left = int(hyperparams["patience"])
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("ml26 fine-tune early stopping at epoch %d", epoch)
                break

    network.load_state_dict(best_state)
    network.eval()
    tuned = replace(base, network=network, source="fine_tuned")
    val_report = regression_report(
        y_val, _predict_denorm(network, x_val_norm, base, batch_size), base.target_names
    )
    x_test, y_test, _ = data["test"]
    test_report = None
    if len(x_test):
        test_report = regression_report(
            y_test, _predict_denorm(network, norm_x(x_test), base, batch_size), base.target_names
        )
    return FineTuneResult(
        model=tuned,
        n_lots={k: v[2] for k, v in data.items()},
        n_windows={k: len(v[0]) for k, v in data.items()},
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        val_report=val_report,
        test_report=test_report,
    )
