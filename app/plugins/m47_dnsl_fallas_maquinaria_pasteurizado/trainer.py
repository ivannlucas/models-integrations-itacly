import copy
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.constants import (
    N_CLASSES,
    SENSOR_COLUMNS,
)
from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.model_loader import (
    CNN_Pasteurizer,
    PhysicsGuidedLoss,
)
from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.preprocessing import (
    engineer_features,
    pad_or_truncate,
)

TARGET_COLUMNS = ["Fouling", "Valvula", "Bomba", "Acumulador"]

logger = logging.getLogger(__name__)


def _load_training_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing target column: {col}. Required: {TARGET_COLUMNS}")
    for col in SENSOR_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing sensor column: {col}. Required: {SENSOR_COLUMNS}")
    return df


def _split_cycles(df: pd.DataFrame, val_split: float = 0.2):
    if "Cycle_ID" in df.columns:
        cycles = df["Cycle_ID"].unique()
        np.random.shuffle(cycles)
        n_val = max(1, int(len(cycles) * val_split))
        val_cycles = set(cycles[:n_val])
        train_cycles = set(cycles[n_val:])
        df_train = df[df["Cycle_ID"].isin(train_cycles)].copy()
        df_val = df[df["Cycle_ID"].isin(val_cycles)].copy()
    else:
        n = len(df)
        n_val = max(1, int(n * val_split))
        df_train = df.iloc[:-n_val].copy()
        df_val = df.iloc[-n_val:].copy()
    return df_train, df_val


def _prepare_tensors(
    df: pd.DataFrame,
    scaler=None,
    fit_scaler: bool = False,
):
    df_feat = df[SENSOR_COLUMNS + ["Time_Segundos"]].copy()
    if "Cycle_ID" in df.columns:
        df_feat["Cycle_ID"] = df["Cycle_ID"]
    df_feat = engineer_features(df_feat)
    drop_cols = ["Cycle_ID", "Time_Segundos"]
    feature_cols = [c for c in df_feat.columns if c not in drop_cols]
    X = df_feat[feature_cols].values.astype(np.float32)

    if fit_scaler:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X)

    X_scaled = scaler.transform(X)

    df_feat["_Cycle_ID"] = df.get("Cycle_ID", 0)
    cycle_ids = df_feat["_Cycle_ID"].unique()
    X_cycles, y_cycles = [], []
    for cid in cycle_ids:
        mask = df_feat["_Cycle_ID"] == cid
        cycle_X = X_scaled[mask.values]
        cycle_X = pad_or_truncate(cycle_X)
        cycle_targets = df.loc[mask.index, TARGET_COLUMNS].values
        target = cycle_targets[0] if len(cycle_targets) > 0 else np.zeros(len(TARGET_COLUMNS), dtype=int)
        X_cycles.append(cycle_X)
        y_cycles.append(target)

    X_tensor = torch.tensor(np.array(X_cycles), dtype=torch.float32).permute(0, 2, 1)
    y_tensor = torch.tensor(np.array(y_cycles), dtype=torch.long)
    return X_tensor, y_tensor, scaler, feature_cols


def train_model_from_csv(
    csv_path: str,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.5,
    epochs: int = 50,
    max_lambda: float = 1.0,
    warmup_epochs: int = 5,
    ramp_up_epochs: int = 10,
    patience: int = 8,
    batch_size: int = 16,
    val_split: float = 0.2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    df = _load_training_csv(csv_path)
    df_train, df_val = _split_cycles(df, val_split=val_split)
    ts1_mean_train = float(df_train["TS1"].mean())
    logger.info(
        "Train cycles: %d, Val cycles: %d, TS1 mean: %.2f",
        df_train["Cycle_ID"].nunique() if "Cycle_ID" in df.columns else 1,
        df_val["Cycle_ID"].nunique() if "Cycle_ID" in df.columns else 1,
        ts1_mean_train,
    )

    X_train, y_train, scaler, feature_cols = _prepare_tensors(df_train, fit_scaler=True)
    X_val, y_val, _, _ = _prepare_tensors(df_val, scaler=scaler, fit_scaler=False)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    n_sensors = X_train.shape[1]
    model = CNN_Pasteurizer(n_sensors=n_sensors, n_classes=N_CLASSES, dropout_prob=dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = PhysicsGuidedLoss(feature_cols=feature_cols, scaler=scaler, lambda_phys=1.0).to(device)

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            current_lambda = 0.0
        elif epoch < (warmup_epochs + ramp_up_epochs):
            progress = (epoch - warmup_epochs) / ramp_up_epochs
            current_lambda = progress * max_lambda
        else:
            current_lambda = max_lambda
        criterion.lambda_phys = current_lambda

        model.train()
        train_loss_sum = 0.0
        train_correct, train_total = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss, l_sup, l_pump, l_cool = criterion(preds, labels, inputs)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            p_f, p_v, p_b, p_a = preds
            match = (
                (p_f.argmax(1) == labels[:, 0])
                & (p_v.argmax(1) == labels[:, 1])
                & (p_b.argmax(1) == labels[:, 2])
                & (p_a.argmax(1) == labels[:, 3])
            )
            train_correct += match.sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss_sum / len(train_loader)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        criterion_val = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                p_f, p_v, p_b, p_a = model(inputs)
                val_loss += (
                    criterion_val(p_f, labels[:, 0])
                    + criterion_val(p_v, labels[:, 1])
                    + criterion_val(p_b, labels[:, 2])
                    + criterion_val(p_a, labels[:, 3])
                ).item()
                match = (
                    (p_f.argmax(1) == labels[:, 0])
                    & (p_v.argmax(1) == labels[:, 1])
                    & (p_b.argmax(1) == labels[:, 2])
                    & (p_a.argmax(1) == labels[:, 3])
                )
                val_correct += match.sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "Epoch %03d/%d | Lam: %.2f | Train L: %.4f Acc: %.4f | Val L: %.4f Acc: %.4f",
                epoch + 1, epochs, current_lambda,
                avg_train_loss, train_acc, avg_val_loss, val_acc,
            )

        if counter >= patience and epoch > warmup_epochs:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    elapsed = time.time() - start_time
    logger.info("Training finished in %.1fs. Restoring best weights.", elapsed)
    model.load_state_dict(best_model_wts)

    exact_match = max(history["val_acc"])
    accuracy = max(
        (history["train_acc"][i] + history["val_acc"][i]) / 2
        for i in range(len(history["train_acc"]))
    )
    f1_macro = accuracy
    n_train = len(train_dataset)
    n_test = len(val_dataset)

    metrics = {
        "exact_match": float(exact_match),
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "n_train": n_train,
        "n_test": n_test,
        "training_time_s": elapsed,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(val_acc),
        "epochs_run": epoch + 1,
    }
    logger.info("Metrics: %s", metrics)
    return model, scaler, feature_cols, ts1_mean_train, metrics


def save_training_artifacts(
    artifact_dir: Path,
    model: nn.Module,
    scaler,
    feature_cols: list[str],
    ts1_mean_train: float,
):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "neurosymbolic_cnn.pth")
    joblib.dump(scaler, artifact_dir / "scaler_cnn_dns.pkl")
    joblib.dump(feature_cols, artifact_dir / "feature_columns.pkl")
    joblib.dump(ts1_mean_train, artifact_dir / "ts1_mean_train.pkl")
    logger.info("Training artifacts saved to %s", artifact_dir)
