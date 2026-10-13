"""Reentrenamiento del DNSL (entregable 47/48) con el pipeline original del equipo de IA.

Replica ``src/data_processing/preprocess.py`` + ``src/training/trainer.py``: split por Cycle_ID 70/15/15
(seed 42), gemelo térmico calculado sobre train, data augmentation por ruido solo en train, feature
engineering (_rmean/_rstd/_lag1), escalado ajustado en train y entrenamiento con PhysicsGuidedLoss
(curriculum de lambda) y early stopping sobre la pérdida de validación. Entrena un modelo NUEVO: el
modelo cargado en el plugin nunca se muta.
"""
from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from app.plugins.m48_dnsl_fallas_maquinaria_pasteurizado.constants import (
    N_CLASSES,
    SENSOR_COLUMNS,
    SHAP_BACKGROUND_FILENAME,
    SHAP_BACKGROUND_META_FILENAME,
    SHAP_N_BACKGROUND,
    TARGET_COLUMNS,
    TRAIN_HYPERPARAMS as HP,
)
from app.plugins.m48_dnsl_fallas_maquinaria_pasteurizado.model_loader import (
    CNN_Pasteurizer,
    PhysicsGuidedLoss,
)
from app.plugins.m48_dnsl_fallas_maquinaria_pasteurizado.preprocessing import (
    engineer_features,
    pad_or_truncate,
)

logger = logging.getLogger(__name__)

AUGMENTED_ID_OFFSET = 50000
MIN_CYCLES = 10
REQUIRED_COLUMNS = ["Cycle_ID", "Time_Segundos", *SENSOR_COLUMNS, *TARGET_COLUMNS]


def _load_training_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Required: {REQUIRED_COLUMNS}")
    if df["Cycle_ID"].nunique() < MIN_CYCLES:
        raise ValueError(
            f"At least {MIN_CYCLES} distinct Cycle_ID values are needed for the 70/15/15 split; "
            f"got {df['Cycle_ID'].nunique()}"
        )
    bad = df[TARGET_COLUMNS].isin([0, 1, 2]).all(axis=1)
    if not bad.all():
        raise ValueError(f"Target columns {TARGET_COLUMNS} must only contain 0, 1 or 2")
    return df


def _resample_10hz(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Time_Segundos"] = df["Time_Segundos"].round(1)
    grouped = df.groupby(["Cycle_ID", "Time_Segundos"])[SENSOR_COLUMNS].mean().reset_index()
    targets = df.groupby("Cycle_ID")[TARGET_COLUMNS].first().reset_index()
    return grouped.merge(targets, on="Cycle_ID")


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby("Cycle_ID", sort=False):
        parts.append(engineer_features(g))
    out = pd.concat(parts, ignore_index=True)
    new_cols = [c for c in out.columns if c.endswith(("_rmean", "_rstd", "_lag1"))]
    return out[["Cycle_ID", "Time_Segundos", *SENSOR_COLUMNS, *new_cols, *TARGET_COLUMNS]]


def _to_tensors(df: pd.DataFrame, cycles, feature_cols: list[str], scaler):
    sub = df[df["Cycle_ID"].isin(cycles)]
    xs, ys = [], []
    for _, g in sub.groupby("Cycle_ID", sort=True):
        xs.append(pad_or_truncate(scaler.transform(g[feature_cols])))
        ys.append(g[TARGET_COLUMNS].iloc[0].values)
    X = torch.tensor(np.array(xs), dtype=torch.float32).permute(0, 2, 1)
    y = torch.tensor(np.array(ys), dtype=torch.long)
    return X, y, sorted(sub["Cycle_ID"].unique().tolist())


def _exact_match(logits, labels) -> torch.Tensor:
    match = torch.ones(labels.size(0), dtype=torch.bool, device=labels.device)
    for i in range(4):
        match &= logits[i].argmax(1) == labels[:, i]
    return match


def _test_metrics(model, X_test, y_test, device) -> dict:
    model.eval()
    with torch.no_grad():
        out = model(X_test.to(device))
    y_pred = torch.stack([o.argmax(1) for o in out], dim=1).cpu().numpy()
    y_true = y_test.numpy()
    exact = float(np.mean(np.all(y_pred == y_true, axis=1)))
    acc, f1, rec = [], [], []
    for i in range(4):
        acc.append(float(np.mean(y_pred[:, i] == y_true[:, i])))
        _, r, f, _ = precision_recall_fscore_support(y_true[:, i], y_pred[:, i], labels=[0, 1, 2],
                                                     average="macro", zero_division=0)
        f1.append(float(f))
        rec.append(float(r))
    return {
        "exact_match": exact,
        "accuracy": float(np.mean(acc)),
        "f1_macro": float(np.mean(f1)),
        "recall_macro": float(np.mean(rec)),
    }


def train_model_from_csv(csv_path: str):
    """Entrena un DNSL nuevo. Devuelve (model, scaler, feature_cols, ts1_mean_train, metrics, shap_background)."""
    seed = HP["random_state"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    df_10hz = _resample_10hz(_load_training_csv(csv_path))

    cycles = df_10hz["Cycle_ID"].unique()
    train_cycles, temp_cycles = train_test_split(cycles, test_size=HP["test_size_1"], random_state=seed, shuffle=True)
    val_cycles, test_cycles = train_test_split(temp_cycles, test_size=HP["test_size_2"], random_state=seed, shuffle=True)

    # Gemelo térmico: media de TS1 en train; offset a 65 °C aplicado a todos los ciclos (como en el original).
    ts1_mean_train = float(df_10hz.loc[df_10hz["Cycle_ID"].isin(train_cycles), "TS1"].mean())
    offset = 65.0 - ts1_mean_train
    df_thermo = df_10hz.copy()
    df_thermo["TS1"] += offset
    df_thermo["TS2"] += offset

    # Data augmentation (solo train): copia con ruido gaussiano proporcional a la desviación típica.
    df_noisy = df_thermo[df_thermo["Cycle_ID"].isin(train_cycles)].copy()
    df_noisy["Cycle_ID"] = df_noisy["Cycle_ID"] + AUGMENTED_ID_OFFSET
    rng = np.random.default_rng(seed)
    for col in SENSOR_COLUMNS:
        df_noisy[col] += rng.normal(0, df_noisy[col].std() * HP["noise_level"], size=len(df_noisy))
    df_aug = pd.concat([df_thermo, df_noisy], ignore_index=True)
    train_cycles_aug = np.concatenate([train_cycles, df_noisy["Cycle_ID"].unique()])

    df_final = _feature_engineering(df_aug)
    feature_cols = [c for c in df_final.columns if c not in ["Cycle_ID", "Time_Segundos", *TARGET_COLUMNS]]

    scaler = StandardScaler().fit(df_final.loc[df_final["Cycle_ID"].isin(train_cycles_aug), feature_cols])
    X_train, y_train, _ = _to_tensors(df_final, train_cycles_aug, feature_cols, scaler)
    X_val, y_val, _ = _to_tensors(df_final, val_cycles, feature_cols, scaler)
    X_test, y_test, test_ids = _to_tensors(df_final, test_cycles, feature_cols, scaler)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=HP["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=HP["batch_size"], shuffle=False)

    model = CNN_Pasteurizer(n_sensors=len(feature_cols), n_classes=N_CLASSES,
                            dropout_prob=HP["dropout_rate"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=HP["learning_rate"])
    criterion = PhysicsGuidedLoss(feature_cols=feature_cols, scaler=scaler, lambda_phys=1.0).to(device)
    criterion_val = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_wts = copy.deepcopy(model.state_dict())
    counter = 0
    epochs_run = 0
    start = time.time()

    for epoch in range(HP["epochs"]):
        warm, ramp = HP["warmup_epochs"], HP["ramp_up_epochs"]
        if epoch < warm:
            criterion.lambda_phys = 0.0
        elif epoch < warm + ramp:
            criterion.lambda_phys = (epoch - warm) / ramp * HP["max_lambda"]
        else:
            criterion.lambda_phys = HP["max_lambda"]

        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss, *_ = criterion(model(inputs), labels, inputs)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                out = model(inputs)
                val_loss += sum(criterion_val(out[i], labels[:, i]) for i in range(4)).item()
        avg_val_loss = val_loss / len(val_loader)
        epochs_run = epoch + 1

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
        if epochs_run % 5 == 0 or epoch == 0:
            logger.info("Epoch %03d/%d | lambda %.2f | val loss %.4f", epochs_run, HP["epochs"],
                        criterion.lambda_phys, avg_val_loss)
        if counter >= HP["patience"] and epoch > warm:
            logger.info("Early stopping at epoch %d", epochs_run)
            break

    elapsed = time.time() - start
    model.load_state_dict(best_wts)
    model = model.cpu().eval()

    metrics = _test_metrics(model, X_test, y_test, torch.device("cpu"))
    metrics.update({
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "training_time_s": elapsed,
        "best_val_loss": float(best_val_loss),
        "epochs_run": epochs_run,
    })
    logger.info("Test metrics: %s", metrics)

    # Fondo SHAP del modelo reentrenado: primeros 50 ciclos de test (orden por Cycle_ID), como el original.
    n_bg = min(SHAP_N_BACKGROUND, X_test.shape[0])
    shap_background = {"array": X_test[:n_bg].numpy(), "cycle_ids": test_ids[:n_bg]}
    return model, scaler, feature_cols, ts1_mean_train, metrics, shap_background


def save_training_artifacts(artifact_dir: Path, model: nn.Module, scaler, feature_cols: list[str],
                            ts1_mean_train: float, shap_background: dict | None = None):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "neurosymbolic_cnn.pth")
    joblib.dump(scaler, artifact_dir / "scaler_cnn_dns.pkl")
    joblib.dump(feature_cols, artifact_dir / "feature_columns.pkl")
    joblib.dump(ts1_mean_train, artifact_dir / "ts1_mean_train.pkl")
    if shap_background is not None:
        np.save(artifact_dir / SHAP_BACKGROUND_FILENAME, shap_background["array"])
        (artifact_dir / SHAP_BACKGROUND_META_FILENAME).write_text(
            json.dumps({"cycle_ids": shap_background["cycle_ids"], "source": "user retraining test split"}),
            encoding="utf-8",
        )
    logger.info("Training artifacts saved to %s", artifact_dir)
