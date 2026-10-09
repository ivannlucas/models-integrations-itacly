"""ml26 /train — port of the final training of the AI team's GRU-PSO
(src/training/models/train_sequence_pso.py → train_sequence_with_config in
src/training/models/train_sequence.py, inbox/a26/codigo).

Same steps as the delivered code:

1. Feature engineering with the two 72 h targets, hold-out split by lot_id (70/15/15, seed 42) and
   windows of 24 steps with stride 2 (dataset_builder). Checked against the delivered
   data/splits/*_seq_{X,y}.npy: identical (difference 0).
2. z-score of X (per feature over train windows) and of y, both fitted on the user's train split.
3. A NEW GRUMultiTaskRegressor with the final PSO configuration (v2.5 Tabla 13, the served
   model's config), seed 7 set right before training as train_sequence_pso does.
4. AdamW, SmoothL1Loss on z-scored targets, ReduceLROnPlateau on validation RMSE, grad-clip 1.0,
   up to 40 epochs, early stopping with patience 8 on validation RMSE; best weights restored.
5. MAE / RMSE / MAPE per target and overall on validation (best epoch) and test, with the risk
   clipped to [0, 1] as in the original.

Not ported: the PSO hyperparameter search (4 particles x 3 iterations x 2 repeats = 24 trainings of
14 epochs): it selects the configuration, which /train reuses. Same rule as ml14 and ml23.

The integration of 2026-10 shipped a different procedure (fine-tuning of the served weights with the
served normalization), which the AI team never defined; replaced on 2026-10-09.
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
class TrainResult:
    """Outcome of train_with_config: the new model and the original's metrics."""

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
    # C-contiguous like the .npy splits the original trains from: float32 means are accumulated in
    # memory order, so a strided copy of the same values gives a mean that differs in the last bit
    # and, after 25 epochs, a different model.
    return np.ascontiguousarray(np.stack(x_windows)), np.ascontiguousarray(np.stack(y_windows))


def regression_report(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]
) -> dict[str, Any]:
    """src/training/evaluation/metrics.py::regression_report (MAE, RMSE, MAPE per target + overall)."""
    report: dict[str, Any] = {}
    for index, target in enumerate(target_names):
        err = y_pred[:, index] - y_true[:, index]
        report[target] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mape": float(np.mean(np.abs(err / np.maximum(np.abs(y_true[:, index]), 1e-6)))),
        }
    report["overall_mae"] = float(np.mean(np.abs(y_pred - y_true)))
    report["overall_rmse"] = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return report


def _evaluate(network: nn.Module, loader: DataLoader, y_mean, y_std, target_names) -> tuple[np.ndarray, np.ndarray]:
    """train_sequence.py::_evaluate: de-normalized, clipped predictions and de-normalized targets."""
    network.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(network(xb).numpy())
            targets.append(yb.numpy())
    preds_np = np.concatenate(preds) * y_std + y_mean
    targets_np = np.concatenate(targets) * y_std + y_mean
    return constrain_predictions(preds_np, target_names), targets_np


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


def train_with_config(  # pylint: disable=too-many-locals,too-many-statements
    base: LoadedModel,
    data: dict[str, tuple[np.ndarray, np.ndarray, int]],
    model_cfg: dict[str, Any],
    on_epoch=None,
) -> TrainResult:
    """train_sequence.py::train_sequence_with_config on the user's windows (seed 7, from scratch).

    *base* only provides the feature/target contract and the final configuration; its weights and
    normalization are not used, and it is never mutated.
    """
    x_train, y_train, _ = data["train"]
    x_val, y_val, _ = data["val"]
    x_test, y_test, _ = data["test"]
    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError(
            "No hay ventanas suficientes para entrenar/validar: se necesitan >= 7 lotes con >= "
            f"{WINDOW} pasos operativos con target a 72 h (train={len(x_train)}, "
            f"val={len(x_val)} ventanas)."
        )
    set_global_seed(TRAINING_SEED)   # train_sequence_pso: final_seed = training.seed, right before

    # _normalize_arrays / _normalize_targets
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-6

    def norm_x(x):
        return ((x - mean) / std).astype(np.float32)

    def norm_y(y):
        return ((y - y_mean) / y_std).astype(np.float32)

    batch_size = int(model_cfg["batch_size"])

    def loader(x, y, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(norm_x(x)), torch.from_numpy(norm_y(y))),
                          batch_size=batch_size, shuffle=shuffle)

    train_loader = loader(x_train, y_train, True)
    val_loader = loader(x_val, y_val, False)
    test_loader = loader(x_test, y_test, False) if len(x_test) else None

    network = build_gru_from_config(model_cfg, input_dim=x_train.shape[-1], output_dim=y_train.shape[-1])
    optimizer = torch.optim.AdamW(network.parameters(), lr=float(model_cfg["learning_rate"]),
                                  weight_decay=float(model_cfg.get("weight_decay", 0.0)))
    criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(model_cfg.get("lr_decay_factor", 0.5)),
        patience=int(model_cfg.get("lr_patience", 2)))
    gradient_clip = float(model_cfg.get("gradient_clip", 1.0))

    best_state = copy.deepcopy(network.state_dict())
    best_rmse, best_epoch, epochs_run = float("inf"), 0, 0
    best_val = (np.empty((0, y_train.shape[1])), np.empty((0, y_train.shape[1])))
    patience_left = int(model_cfg["patience"])
    for epoch in range(1, int(model_cfg["epochs"]) + 1):
        network.train()
        losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(network(xb), yb)
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=gradient_clip)
            optimizer.step()
            losses.append(loss.item())
        val_pred, val_true = _evaluate(network, val_loader, y_mean, y_std, base.target_names)
        val_rmse = float(np.sqrt(np.mean((val_true - val_pred) ** 2)))
        scheduler.step(val_rmse)
        epochs_run = epoch
        train_loss = float(np.mean(losses)) if losses else 0.0
        logger.info("ml26 train epoch=%d train_loss=%.6f val_rmse=%.6f", epoch, train_loss, val_rmse)
        if on_epoch:
            on_epoch(epoch, train_loss, val_rmse)
        if val_rmse + 1e-6 < best_rmse:
            best_rmse, best_epoch = val_rmse, epoch
            best_state = copy.deepcopy(network.state_dict())
            best_val = (val_pred, val_true)
            patience_left = int(model_cfg["patience"])
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("ml26 train: early stopping at epoch %d", epoch)
                break

    network.load_state_dict(best_state)
    network.eval()
    test_report = None
    if test_loader is not None:
        test_pred, test_true = _evaluate(network, test_loader, y_mean, y_std, base.target_names)
        test_report = regression_report(test_true, test_pred, base.target_names)
    trained = replace(
        base, network=network, mean=mean.squeeze(0).astype(np.float32), std=std.squeeze(0).astype(np.float32),
        y_mean=y_mean.squeeze(0).astype(np.float32), y_std=y_std.squeeze(0).astype(np.float32),
        config=dict(model_cfg), source="retrained",
    )
    return TrainResult(
        model=trained,
        n_lots={k: v[2] for k, v in data.items()},
        n_windows={k: len(v[0]) for k, v in data.items()},
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        val_report=regression_report(best_val[1], best_val[0], base.target_names),
        test_report=test_report,
    )
