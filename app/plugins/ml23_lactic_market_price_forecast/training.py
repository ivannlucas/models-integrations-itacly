"""Real fine-tuning pipeline for ml23 — ported from inbox/a23/codigo/
src/training/compare_models.py (prepare_horizon_dataset, split_dev_test,
resolve_cv_val_window, make_refit_split, prepare_rnn_split_with_test, train_rnn,
train_rnn_multi_seed) and config/config.yaml defaults.

Only the GRU refit is ported — the full model search (Naive/Drift/XGBoost/LSTM/GRU
compared across walk-forward CV folds) that originally SELECTED GRU as the architecture
to deploy is not re-run here. Re-running that search on every /train call would be
impractical for a synchronous HTTP endpoint; the same simplification is already applied
to ml30/ml9's neuroevolution search in this repo (reuse the already-selected
architecture, refit its weights on new data).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

META_COLS = {"fecha", "producto", "canal"}
TARGET_COL = "target_h"
CURRENT_PRICE_COL = "current_price"


def prepare_horizon_dataset(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build the H-month-ahead target per (producto, canal) series.

    Ported verbatim from compare_models.py::prepare_horizon_dataset — expects a raw
    dataset with a 'target_precio_medio' column (the real contract of
    data/processed/dataset_forecast_ready.csv, the AI team's own training input).
    """
    pieces = []
    for _, grp in df.groupby(["producto", "canal"]):
        grp = grp.sort_values("fecha").reset_index(drop=True)
        grp[TARGET_COL] = grp["target_precio_medio"].shift(-horizon)
        grp[CURRENT_PRICE_COL] = grp["target_precio_medio"]
        grp = grp.dropna(subset=[TARGET_COL])
        pieces.append(grp)
    result = pd.concat(pieces, ignore_index=True)
    result.drop(columns=["target_precio_medio"], inplace=True)
    return result


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the observable columns fed to the model (ported verbatim)."""
    exclude = META_COLS | {TARGET_COL}
    return [c for c in df.columns if c not in exclude]


def split_dev_test(df: pd.DataFrame, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split development/test by the most recent dates (ported verbatim)."""
    fechas = sorted(df["fecha"].unique())
    if len(fechas) < 3:
        raise ValueError("Se necesitan al menos 3 fechas únicas para separar development y test.")
    n_test = max(1, int(len(fechas) * test_ratio))
    if n_test >= len(fechas):
        n_test = len(fechas) - 1
    fecha_inicio_test = fechas[-n_test]
    dev = df[df["fecha"] < fecha_inicio_test].copy()
    test = df[df["fecha"] >= fecha_inicio_test].copy()
    if dev.empty or test.empty:
        raise ValueError("El split temporal produjo development o test vacío.")
    return dev, test


def resolve_cv_val_window(
    n_dev_dates: int, n_splits: int, seq_len: int, cv_val_size: int | None, val_ratio: float,
) -> int:
    """Resolve the refit validation window size (ported verbatim, folds collapsed to 1)."""
    if cv_val_size is not None:
        window = cv_val_size
    else:
        auto = max(seq_len, 6)
        ratio_based = max(1, int(n_dev_dates * val_ratio))
        window = min(auto, ratio_based) if ratio_based > 0 else auto
    max_window = (n_dev_dates - seq_len) // max(n_splits, 1)
    if max_window < 1:
        raise ValueError("No hay suficientes fechas para construir la ventana de validación.")
    return min(window, max_window) if window > max_window else max(window, 1)


def make_refit_split(
    dev_df: pd.DataFrame, val_window: int, min_train_dates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the last block of development as refit validation (ported verbatim)."""
    fechas = sorted(dev_df["fecha"].unique())
    if len(fechas) <= val_window:
        raise ValueError("No hay suficientes fechas en development para refit.")
    if len(fechas) - val_window < min_train_dates:
        raise ValueError("El bloque de refit deja demasiado poco historial para entrenamiento.")
    fecha_inicio_val = fechas[-val_window]
    train_df = dev_df[dev_df["fecha"] < fecha_inicio_val].copy()
    val_df = dev_df[dev_df["fecha"] >= fecha_inicio_val].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("El split de refit produjo un subconjunto vacío.")
    return train_df, val_df


def build_sequences_by_group(
    df: pd.DataFrame, feature_cols: list[str], seq_len: int, scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences per (producto, canal) group (ported verbatim)."""
    X_all, y_all = [], []
    for _, grp in df.groupby(["producto", "canal"]):
        grp = grp.sort_values("fecha")
        X_raw = grp[feature_cols].values
        y_raw = grp[TARGET_COL].values
        if len(grp) < seq_len:
            continue
        X_scaled = scaler.transform(X_raw)
        for i in range(len(X_scaled) - seq_len + 1):
            X_all.append(X_scaled[i: i + seq_len])
            y_all.append(y_raw[i + seq_len - 1])
    return np.array(X_all), np.array(y_all)


def build_eval_sequences_with_bridge(
    history_df: pd.DataFrame, target_df: pd.DataFrame, feature_cols: list[str],
    seq_len: int, scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build evaluation sequences bridged from history (ported verbatim)."""
    X_all, y_all, cp_all = [], [], []
    for (prod, canal), target_grp in target_df.groupby(["producto", "canal"]):
        history_grp = history_df[
            (history_df["producto"] == prod) & (history_df["canal"] == canal)
        ].sort_values("fecha")
        target_grp = target_grp.sort_values("fecha")
        bridge = history_grp.tail(seq_len - 1)
        combined = pd.concat([bridge, target_grp], ignore_index=True)
        if len(combined) < seq_len:
            continue
        X_raw = combined[feature_cols].values
        y_raw = combined[TARGET_COL].values
        cp_raw = combined[CURRENT_PRICE_COL].values
        X_scaled = scaler.transform(X_raw)
        for i in range(len(X_scaled) - seq_len + 1):
            target_idx = i + seq_len - 1
            if target_idx >= len(bridge):
                X_all.append(X_scaled[i: i + seq_len])
                y_all.append(y_raw[target_idx])
                cp_all.append(cp_raw[target_idx])
    return np.array(X_all), np.array(y_all), np.array(cp_all)


def prepare_rnn_split_with_test(  # pylint: disable=too-many-locals
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    feature_cols: list[str], seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Prepare train/val/test sequences for the final refit (ported verbatim)."""
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)

    train_X, train_y = build_sequences_by_group(train_df, feature_cols, seq_len, scaler)
    val_X, val_y, _ = build_eval_sequences_with_bridge(
        train_df, val_df, feature_cols, seq_len, scaler
    )
    if len(train_X) == 0 or len(val_X) == 0:
        raise ValueError("No hay suficientes secuencias RNN para el split solicitado.")

    history_for_test = pd.concat([train_df, val_df], ignore_index=True)
    test_X, test_y, test_cp = build_eval_sequences_with_bridge(
        history_for_test, test_df, feature_cols, seq_len, scaler
    )
    if len(test_X) == 0:
        raise ValueError("No hay suficientes secuencias RNN para test.")
    return train_X, train_y, val_X, val_y, test_X, test_y, test_cp, scaler


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the project's regression metrics (ported verbatim)."""
    mask = y_true != 0
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = 0.0
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_pct": round(mape, 2),
        "R2": float(r2_score(y_true, y_pred)),
    }


def calc_direction_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray,
) -> dict[str, float]:
    """Compute directional accuracy vs. the current price (ported verbatim)."""
    delta_true = y_true - y_naive
    delta_pred = y_pred - y_naive
    nonzero = delta_true != 0
    if nonzero.sum() == 0:
        direction_acc = 50.0
    else:
        sign_match = np.sign(delta_pred[nonzero]) == np.sign(delta_true[nonzero])
        direction_acc = float(np.mean(sign_match) * 100)
    return {"direction_acc_pct": round(direction_acc, 1)}


def train_rnn(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    model: nn.Module, train_X: np.ndarray, train_y: np.ndarray,
    val_X: np.ndarray, val_y: np.ndarray,
    *, epochs: int, batch_size: int, lr: float, patience: int, device: torch.device,
) -> tuple[dict[str, float], nn.Module]:
    """Train one RNN instance with early stopping (ported verbatim, logging stripped)."""
    model = model.to(device)
    train_t = torch.tensor(train_X, dtype=torch.float32)
    train_yt = torch.tensor(train_y, dtype=torch.float32)
    val_t = torch.tensor(val_X, dtype=torch.float32).to(device)

    train_dl = DataLoader(TensorDataset(train_t, train_yt), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for _epoch in range(1, epochs + 1):
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
            val_preds = model(val_t).cpu().numpy()
            val_loss = float(np.mean((val_preds - val_y) ** 2))
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_val_preds = model(val_t).cpu().numpy()
    return calc_metrics(val_y, final_val_preds), model


def predict_rnn(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Run batched inference with an already-trained RNN (ported verbatim)."""
    model = model.to(device)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        return model(X_t).cpu().numpy()


def train_rnn_multi_seed(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    model_cls: type, input_size: int, hidden_size: int,
    train_X: np.ndarray, train_y: np.ndarray, val_X: np.ndarray, val_y: np.ndarray,
    test_X: np.ndarray, test_y: np.ndarray, test_cp: np.ndarray,
    *, n_seeds: int, seed_base: int, device: torch.device, epochs: int, batch_size: int,
    lr: float, patience: int,
) -> tuple[dict[str, float], float, nn.Module]:
    """Train with multiple seeds; return the best-by-validation-RMSE model and its test
    metrics (ported from train_rnn_multi_seed(..., return_best_eval=True) — the policy
    already used to pick the shipped gru_model.pt, see constants.py)."""
    best_val_rmse = float("inf")
    best_model: nn.Module | None = None
    best_test_metrics: dict[str, float] | None = None
    t0 = time.time()

    for seed in range(seed_base, seed_base + n_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = model_cls(input_size, hidden_size=hidden_size)
        val_metrics, trained_model = train_rnn(
            model, train_X, train_y, val_X, val_y,
            epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, device=device,
        )
        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_model = trained_model
            test_preds = predict_rnn(trained_model, test_X, device=device)
            test_metrics = calc_metrics(test_y, test_preds)
            test_metrics.update(calc_direction_metrics(test_y, test_preds, test_cp))
            best_test_metrics = test_metrics

    elapsed = time.time() - t0
    if best_model is None or best_test_metrics is None:
        raise RuntimeError("No se pudo seleccionar la mejor seed RNN.")
    return best_test_metrics, elapsed, best_model
