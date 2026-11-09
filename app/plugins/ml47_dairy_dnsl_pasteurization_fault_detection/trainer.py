"""ml47 /train — ports of the AI team's two training procedures.

Source: inbox/a47/codigo (config/config.yaml, src/).

* ``train_full`` = ``main.py train``: src/data_processing/preprocess.py (split by Cycle_ID,
  digital twin, noise augmentation, feature engineering, scaler) + src/training/trainer.py
  (Physics-Guided Loss with the lambda curriculum, early stopping on validation loss).
* ``fine_tune`` = ``main.py fine_tune`` (the /train default): src/fine_tuning/fine_tuner.py
  (frozen CNN backbone, only the 4 classification heads recalibrated with CrossEntropy,
  LR = learning_rate / 10).

Plugin adaptations, all deliberate:

* /train receives ONE CSV. Both modes split it by Cycle_ID with the original 70/15/15 split
  (split_data) and report metrics on the test part — the original fine_tune takes separate
  train/val CSVs and reports only the validation loss.
* The digital twin (TS1/TS2 shifted to a 65 °C baseline) is applied in ``full`` only when
  APPLY_DIGITAL_TWIN is on — the same switch inference uses. The original always applies it
  because its data is the UCI lab bench; with real plant data it must stay off.
* Augmented copies get Cycle_ID = max(Cycle_ID) + 1 + i instead of the original + 50000, which
  could collide with real plant cycle ids. The noise itself is identical.
* Seeds are set per call (python, numpy, torch, DataLoader generator) instead of
  ``torch.use_deterministic_algorithms``, which is process-global and this is a shared server.
"""
from __future__ import annotations

import copy
import logging
import random
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.plugins.ml47_dairy_dnsl_pasteurization_fault_detection.constants import (
    APPLY_DIGITAL_TWIN,
    FEATURE_COLUMNS_FILENAME,
    FINE_TUNE_HYPERPARAMS,
    LEGACY_TARGET_COLUMNS,
    MODEL_FILENAME,
    N_CLASSES,
    NOISE_LEVEL,
    RANDOM_STATE,
    SCALER_FILENAME,
    SENSOR_COLUMNS,
    SPLIT_TEST_SIZE_1,
    SPLIT_TEST_SIZE_2,
    TARGET_COLUMNS,
    TRAIN_HYPERPARAMS,
    TS1_MEAN_FILENAME,
)
from app.plugins.ml47_dairy_dnsl_pasteurization_fault_detection.model_loader import (
    CNN_Pasteurizer,
    PhysicsGuidedLoss,
    safe_device,
)
from app.plugins.ml47_dairy_dnsl_pasteurization_fault_detection.preprocessing import (
    apply_digital_twin,
    engineer_features,
    pad_or_truncate,
)

logger = logging.getLogger(__name__)

ID_COLUMNS = ["Cycle_ID", "Time_Segundos"]


@dataclass
class TrainingResult:
    """A retrained model plus everything inference needs to use it, and its hold-out metrics."""

    model: nn.Module
    scaler: Any
    feature_cols: list[str]
    ts1_mean_train: float
    metrics: dict


# ── Data ──────────────────────────────────────────────────────────────────────

def load_training_frame(csv_path: str) -> pd.DataFrame:
    """Read a labelled 10 Hz cycle CSV and keep the raw columns the procedures start from.

    Targets: Target_Fouling/Valvula/Bomba/Acumulador (features.cols_targets) or the short names
    the previous plugin trainer required. Precomputed feature columns are ignored and recomputed,
    as fine_tuner.py does.
    """
    df = pd.read_csv(csv_path)
    if "Time_Segundos" not in df.columns and "Time" in df.columns:
        df = df.rename(columns={"Time": "Time_Segundos"})
    if not set(TARGET_COLUMNS) <= set(df.columns):
        if set(LEGACY_TARGET_COLUMNS) <= set(df.columns):
            df = df.rename(columns=dict(zip(LEGACY_TARGET_COLUMNS, TARGET_COLUMNS)))
        else:
            raise ValueError(
                f"El CSV de entrenamiento debe traer las etiquetas {TARGET_COLUMNS} "
                f"(0=Sano, 1=Warning, 2=Crítico por ciclo). Faltan: "
                f"{[c for c in TARGET_COLUMNS if c not in df.columns]}"
            )
    missing = [c for c in ID_COLUMNS + SENSOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"El CSV de entrenamiento no trae las columnas obligatorias: {missing}")
    df = df[ID_COLUMNS + SENSOR_COLUMNS + TARGET_COLUMNS].copy()
    df[TARGET_COLUMNS] = df[TARGET_COLUMNS].astype(int)
    return df


def split_cycle_ids(cycle_ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """split_data(): 70% train / 15% val / 15% test by Cycle_ID, random_state=42."""
    ids = np.sort(np.unique(np.asarray(cycle_ids)))
    try:
        train, temp = train_test_split(
            ids, test_size=SPLIT_TEST_SIZE_1, random_state=RANDOM_STATE, shuffle=True
        )
        val, test = train_test_split(
            temp, test_size=SPLIT_TEST_SIZE_2, random_state=RANDOM_STATE, shuffle=True
        )
    except ValueError as exc:
        raise ValueError(
            "No hay suficientes ciclos para separar train/validación/test "
            "(70/15/15 por Cycle_ID): "
            f"se recibieron {len(ids)} ciclos distintos."
        ) from exc
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(
            "No hay suficientes ciclos para separar train/validación/test "
            "(70/15/15 por Cycle_ID): "
            f"se recibieron {len(ids)} ciclos distintos."
        )
    return train, val, test


def engineer_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """feature_engineering(): rolling mean/std (5) and lag-1, computed inside each cycle."""
    parts = [engineer_features(group) for _, group in df.groupby("Cycle_ID", sort=False)]
    return pd.concat(parts, ignore_index=True)


def feature_columns(df_feat: pd.DataFrame) -> list[str]:
    """Model input columns, in the order of the delivered feature_columns.pkl."""
    cols = (SENSOR_COLUMNS + [f"{c}_rmean" for c in SENSOR_COLUMNS]
            + [f"{c}_rstd" for c in SENSOR_COLUMNS] + [f"{c}_lag1" for c in SENSOR_COLUMNS])
    missing = [c for c in cols if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Faltan columnas derivadas: {missing}")
    return cols


def _scale(scaler: Any, frame: pd.DataFrame) -> np.ndarray:
    data = frame if hasattr(scaler, "feature_names_in_") else frame.to_numpy()
    return scaler.transform(data)


def build_tensors(df_feat: pd.DataFrame, cycle_ids, feature_cols: list[str], scaler: Any):
    """create_tensors(): one (features x 600) window per cycle and its 4 labels (first row).

    Cycles go in ascending Cycle_ID order, as in the original (feature_engineering's groupby
    sorts df_final, and fine_tuner groups by Cycle_ID), so the DataLoader sees the same batches.
    """
    # dict(groupby) is not equivalent: dict() sees GroupBy.keys (a str) and raises TypeError.
    groups = {  # pylint: disable=unnecessary-comprehension
        cid: g for cid, g in df_feat.groupby("Cycle_ID", sort=False)
    }
    x_list, y_list = [], []
    for cid in np.sort(np.asarray(cycle_ids)):
        group = groups[cid]
        x_list.append(pad_or_truncate(_scale(scaler, group[feature_cols])))
        y_list.append(group[TARGET_COLUMNS].iloc[0].to_numpy(dtype=int))
    x = torch.tensor(np.array(x_list), dtype=torch.float32).permute(0, 2, 1)
    y = torch.tensor(np.array(y_list), dtype=torch.long)
    return x, y


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_test_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Same formulas as models/metrics/test_metrics_confusion.csv and the memoria (Tabla 6).

    exact_match = all 4 components right at once; accuracy = mean of the 4 per-component
    accuracies; precision/recall/f1_macro = mean over the 12 component x class values
    (labels 0/1/2, zero_division=0).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    precision, recall, f1, accuracy = [], [], [], []
    for i in range(y_true.shape[1]):
        p, r, f, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], labels=[0, 1, 2], zero_division=0,
        )
        precision.extend(p)
        recall.extend(r)
        f1.extend(f)
        accuracy.append(float(np.mean(y_true[:, i] == y_pred[:, i])))
    return {
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "accuracy": float(np.mean(accuracy)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
    }


# ── Shared helpers ────────────────────────────────────────────────────────────

def _seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return safe_device()


def _loader(x, y, batch_size: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(RANDOM_STATE) if shuffle else None
    return DataLoader(
        TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, generator=generator
    )


def _ce_sum(criterion, preds, labels):
    return sum(criterion(preds[i], labels[:, i]) for i in range(4))


def _predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(x), batch_size=256):
            preds = model(batch.to(device))
            out.append(torch.stack([p.argmax(1) for p in preds], dim=1).cpu().numpy())
    return np.vstack(out)


def _validation_loss(model, loader, device) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            total += _ce_sum(criterion, model(inputs), labels).item()
    return total / len(loader)


# ── mode="full" ───────────────────────────────────────────────────────────────

def _augment_train_cycles(df: pd.DataFrame, train_ids: np.ndarray):
    """apply_digital_twin_and_augment(), noise part: a noisy copy of every train cycle."""
    noisy = df[df["Cycle_ID"].isin(train_ids)].copy()
    first_new_id = int(df["Cycle_ID"].max()) + 1
    # Ascending, like the original + 50000: copies keep the relative order of their source cycles.
    id_map = {cid: first_new_id + i for i, cid in enumerate(np.sort(train_ids))}
    noisy["Cycle_ID"] = noisy["Cycle_ID"].map(id_map)
    for column in SENSOR_COLUMNS:
        std_dev = noisy[column].std()
        noisy[column] += np.random.normal(0, std_dev * NOISE_LEVEL, size=len(noisy))
    return (
        pd.concat([df, noisy], ignore_index=True),
        np.concatenate([train_ids, list(id_map.values())]),
    )


def train_full(
    csv_path: str, apply_twin: bool = APPLY_DIGITAL_TWIN
) -> TrainingResult:
    """Retrain the DNSL CNN from scratch (main.py train).

    Uses the original Optuna hyperparameters.
    """
    _seed_everything()
    start = time.time()
    device = _device()

    data = _prepare_full_data(csv_path, apply_twin)
    fit = _fit_full(data, device)
    metrics = _test_metrics(fit["model"], data["test"], device)
    train_ids, val_ids, test_ids = data["ids"]
    metrics.update({
        "n_train": len(train_ids), "n_train_augmented": len(data["train_ids_aug"]),
        "n_val": len(val_ids), "n_test": len(test_ids), "epochs_run": fit["epoch"] + 1,
        "best_val_loss": float(fit["best_val_loss"]),
        "training_time_s": round(time.time() - start, 1),
    })
    logger.info("ml47 full training done: %s", metrics)
    return TrainingResult(
        fit["model"].cpu(), data["scaler"], data["cols"], data["ts1_mean_train"], metrics
    )


def _test_metrics(model: nn.Module, test: tuple, device: torch.device) -> dict:
    """compute_test_metrics() of ``model`` on the hold-out ``test`` = (x, y) tensors."""
    x_test, y_test = test
    return compute_test_metrics(y_test.numpy(), _predict(model, x_test, device))


def _prepare_full_data(csv_path: str, apply_twin: bool) -> dict:
    """Load, split by Cycle_ID, twin + augment, engineer features, fit the scaler, build tensors."""
    df = load_training_frame(csv_path)
    train_ids, val_ids, test_ids = split_cycle_ids(df["Cycle_ID"].unique())
    ts1_mean_train = float(df.loc[df["Cycle_ID"].isin(train_ids), "TS1"].mean())
    if apply_twin:
        df = apply_digital_twin(df, ts1_mean_train)
    df_aug, train_ids_aug = _augment_train_cycles(df, train_ids)

    df_feat = engineer_cycle_features(df_aug)
    cols = feature_columns(df_feat)
    scaler = StandardScaler().fit(df_feat.loc[df_feat["Cycle_ID"].isin(train_ids_aug), cols])
    return {
        "ids": (train_ids, val_ids, test_ids),
        "train_ids_aug": train_ids_aug,
        "ts1_mean_train": ts1_mean_train,
        "cols": cols,
        "scaler": scaler,
        "train": build_tensors(df_feat, train_ids_aug, cols, scaler),
        "val": build_tensors(df_feat, val_ids, cols, scaler),
        "test": build_tensors(df_feat, test_ids, cols, scaler),
    }


def _curriculum_lambda(epoch: int, hp: dict) -> float:
    """Physics weight: data only, then a linear ramp, then max_lambda."""
    if epoch < hp["warmup_epochs"]:
        return 0.0
    if epoch < hp["warmup_epochs"] + hp["ramp_up_epochs"]:
        return (epoch - hp["warmup_epochs"]) / hp["ramp_up_epochs"] * hp["max_lambda"]
    return hp["max_lambda"]


def _physics_loss(criterion, preds, labels, inputs):
    """Total PhysicsGuidedLoss of a batch (first element of what the criterion returns)."""
    loss, _, _, _ = criterion(preds, labels, inputs)
    return loss


def _ce_loss(criterion, preds, labels, _inputs):
    """Sum of the 4 heads' cross-entropy of a batch (the inputs are not used)."""
    return _ce_sum(criterion, preds, labels)


def _train_epoch(model, loader, optimizer, loss_fn, device) -> None:
    """One optimisation pass over ``loader``; ``loss_fn(preds, labels, inputs)`` is the loss."""
    model.train()
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), labels, inputs)
        loss.backward()
        optimizer.step()


def _fit_full(data: dict, device: torch.device) -> dict:
    """Train a new CNN with the physics-guided loss curriculum and early stopping."""
    hp = TRAIN_HYPERPARAMS
    train_loader = _loader(*data["train"], hp["batch_size"], shuffle=True)
    val_loader = _loader(*data["val"], hp["batch_size"], shuffle=False)

    model = CNN_Pasteurizer(
        n_sensors=len(data["cols"]), n_classes=N_CLASSES, dropout_prob=hp["dropout_rate"]
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hp["learning_rate"])
    criterion = PhysicsGuidedLoss(
        feature_cols=data["cols"], scaler=data["scaler"], lambda_phys=1.0
    ).to(device)

    best_val_loss, best_weights, counter, epoch = (
        float("inf"), copy.deepcopy(model.state_dict()), 0, 0
    )
    for epoch in range(hp["epochs"]):
        # Curriculum: data only, then a linear ramp of the physics weight, then max_lambda.
        criterion.lambda_phys = _curriculum_lambda(epoch, hp)

        _train_epoch(
            model, train_loader, optimizer,
            partial(_physics_loss, criterion), device,
        )

        val_loss = _validation_loss(model, val_loader, device)
        if val_loss < best_val_loss:
            best_val_loss, best_weights, counter = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            counter += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("ml47 full Ep %03d/%d | lambda %.2f | val loss %.4f",
                        epoch + 1, hp["epochs"], criterion.lambda_phys, val_loss)
        if counter >= hp["patience"] and epoch > hp["warmup_epochs"]:
            logger.info("ml47 full: early stopping at epoch %d", epoch + 1)
            break

    model.load_state_dict(best_weights)
    return {"model": model, "best_val_loss": best_val_loss, "epoch": epoch}


# ── mode="fine_tune" ──────────────────────────────────────────────────────────

def fine_tune(
    csv_path: str, *, base_model: nn.Module, scaler: Any, feature_cols: list[str],
    ts1_mean_train: float = 0.0,
) -> TrainingResult:
    """Recalibrate the 4 heads of the served model on plant cycles (main.py fine_tune).

    The base model is deep-copied, never mutated; the fitted scaler and feature list are reused
    as-is (no re-fit, as in fine_tuner.py), so they travel unchanged with the result.
    """
    _seed_everything()
    start = time.time()
    device = _device()

    data = _prepare_fine_tune_data(csv_path, feature_cols, scaler)
    fit = _fit_fine_tune(base_model, data, device)
    for param in fit["model"].parameters():
        param.requires_grad = True
    metrics = _test_metrics(fit["model"], data["test"], device)
    metrics.update({
        "n_train": len(data["ids"][0]), "n_val": len(data["ids"][1]),
        "n_test": len(data["ids"][2]),
        "epochs_run": fit["epoch"] + 1, "best_val_loss": float(fit["best_val_loss"]),
        "training_time_s": round(time.time() - start, 1), "learning_rate": fit["lr"],
    })
    logger.info("ml47 fine-tuning done: %s", metrics)
    return TrainingResult(fit["model"].cpu(), scaler, feature_cols, ts1_mean_train, metrics)


def _prepare_fine_tune_data(csv_path: str, feature_cols: list[str], scaler: Any) -> dict:
    """Load, split by Cycle_ID and build tensors with the served scaler/features (no re-fit)."""
    df = load_training_frame(csv_path)
    train_ids, val_ids, test_ids = split_cycle_ids(df["Cycle_ID"].unique())
    df_feat = engineer_cycle_features(df)
    return {
        "ids": (train_ids, val_ids, test_ids),
        "train": build_tensors(df_feat, train_ids, feature_cols, scaler),
        "val": build_tensors(df_feat, val_ids, feature_cols, scaler),
        "test": build_tensors(df_feat, test_ids, feature_cols, scaler),
    }


def _freeze_all_but_heads(model: nn.Module) -> None:
    """Only dropout_final and the 4 heads stay trainable (fine_tuner.py)."""
    for param in model.parameters():
        param.requires_grad = False
    for layer in (model.dropout_final, model.head_fouling, model.head_valvula, model.head_bomba,
                  model.head_acumulador):
        for param in layer.parameters():
            param.requires_grad = True


def _fine_tune_loaders(data: dict, max_batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Train (shuffled) and validation loaders; the batch never exceeds the train cycles."""
    batch_size = min(max_batch_size, len(data["train"][0]))
    return (
        _loader(*data["train"], batch_size, shuffle=True),
        _loader(*data["val"], batch_size, shuffle=False),
    )


def _fit_fine_tune(base_model: nn.Module, data: dict, device: torch.device) -> dict:
    """Train the 4 heads of a deep copy of ``base_model`` with early stopping."""
    hp = FINE_TUNE_HYPERPARAMS
    train_loader, val_loader = _fine_tune_loaders(data, hp["batch_size"])

    model = copy.deepcopy(base_model).to(device)
    _freeze_all_but_heads(model)

    lr = TRAIN_HYPERPARAMS["learning_rate"] / hp["lr_divisor"]
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss, best_weights, counter, epoch = (
        float("inf"), copy.deepcopy(model.state_dict()), 0, 0
    )
    for epoch in range(hp["epochs"]):
        _train_epoch(
            model, train_loader, optimizer,
            partial(_ce_loss, criterion), device,
        )

        val_loss = _validation_loss(model, val_loader, device)
        if val_loss < best_val_loss:
            best_val_loss, best_weights, counter = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            counter += 1
            if counter >= hp["patience"]:
                logger.info("ml47 fine_tune: early stopping at epoch %d", epoch + 1)
                break

    model.load_state_dict(best_weights)
    return {"model": model, "lr": lr, "best_val_loss": best_val_loss, "epoch": epoch}


# ── Artifacts ─────────────────────────────────────────────────────────────────

def save_training_artifacts(artifact_dir: Path, result: TrainingResult) -> None:
    """Write the 4 files mlflow_utils.download_user_model_from_mlflow reads back."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), artifact_dir / MODEL_FILENAME)
    joblib.dump(result.scaler, artifact_dir / SCALER_FILENAME)
    joblib.dump(result.feature_cols, artifact_dir / FEATURE_COLUMNS_FILENAME)
    joblib.dump(result.ts1_mean_train, artifact_dir / TS1_MEAN_FILENAME)
