"""
GRU and Ensemble training workflow for experimental wine price forecasting.

This module implements the experimental training workflow:
- Build technical features and target from the cleaned price series.
- Split the data into train+validation and test.
- Run walk-forward (time series) validation on the train+validation block
  to compare GRU and GRU+Logistic ensemble using AUC.
- Select the best model type based on validation AUC.

This is for experimentation only. Production uses tabular models from ML_models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf

from modules.common.config import MODEL_CONFIG
from modules.common.data import load_raw_data, clean_and_index_data
from modules.common.features import (
    generate_technical_features,
    fit_scaler,
    transform_features,
)
from modules.ML_models.models import build_logistic_model
from .gru_models import build_gru_model, GRULogisticEnsemble, EnsembleWeights
from .gru_features import create_sequences


@dataclass
class FoldMetrics:
    """Container for metrics computed on a single validation fold."""
    auc: float
    accuracy: float
    f1: float
    precision: float
    recall: float


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> FoldMetrics:
    """Compute standard binary classification metrics from probabilities."""
    if len(np.unique(y_true)) == 1:
        auc = 0.5
    else:
        auc = roc_auc_score(y_true, y_prob)

    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)

    return FoldMetrics(auc=auc, accuracy=acc, f1=f1, precision=prec, recall=rec)


def _aggregate_metrics(metrics_list: List[FoldMetrics]) -> Dict[str, float]:
    """Aggregate a list of FoldMetrics into mean values per metric."""
    if not metrics_list:
        raise ValueError("No metrics to aggregate.")

    auc_vals = [m.auc for m in metrics_list]
    acc_vals = [m.accuracy for m in metrics_list]
    f1_vals = [m.f1 for m in metrics_list]
    prec_vals = [m.precision for m in metrics_list]
    rec_vals = [m.recall for m in metrics_list]

    return {
        "auc_mean": float(np.mean(auc_vals)),
        "auc_std": float(np.std(auc_vals)),
        "accuracy_mean": float(np.mean(acc_vals)),
        "f1_mean": float(np.mean(f1_vals)),
        "precision_mean": float(np.mean(prec_vals)),
        "recall_mean": float(np.mean(rec_vals)),
    }


def _split_trainval_test(
    df_features: pd.DataFrame,
    test_weeks: int = 24,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the feature DataFrame into train+validation and test sets.
    """
    n = len(df_features)
    if n == 0:
        raise ValueError("No data available for splitting.")

    if n <= test_weeks:
        df_trainval = df_features.copy()
        df_test = df_features.iloc[0:0]
        return df_trainval, df_test

    df_trainval = df_features.iloc[:-test_weeks].copy()
    df_test = df_features.iloc[-test_weeks:].copy()
    return df_trainval, df_test


def train_gru_with_walk_forward(
    df_price: pd.DataFrame,
    n_splits: int = 5,
) -> Dict:
    """
    Run walk-forward validation for GRU and GRU+Logistic ensemble.

    Steps:
    - Build technical features and target.
    - Exclude the last 24 weeks (reserved for test).
    - Run TimeSeriesSplit on the remaining data to train and evaluate
      GRU and GRU+Logistic ensemble.

    Returns:
        A dictionary with:
            - "metrics_gru": aggregated metrics for GRU.
            - "metrics_ensemble": aggregated metrics for ensemble.
            - "best_model_type": one of {"gru", "ensemble"}.
    """
    df_features = generate_technical_features(
        df_price,
        target_window=MODEL_CONFIG.target_window,
        return_threshold=MODEL_CONFIG.return_threshold,
    )

    df_trainval, _ = _split_trainval_test(df_features, test_weeks=MODEL_CONFIG.test_size)

    feature_cols = ["logret", "distsma12", "rsi14", "bollingerpos", "weeksin", "weekcos"]
    X_df = df_trainval[feature_cols]
    y = df_trainval["target"].values.astype(int)

    tscv = TimeSeriesSplit(
        n_splits=MODEL_CONFIG.n_folds,
        gap=MODEL_CONFIG.gap
    )

    gru_metrics: List[FoldMetrics] = []
    ensemble_metrics: List[FoldMetrics] = []

    splits = list(tscv.split(X_df))
    n_features = len(feature_cols)

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train_df, X_val_df = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = fit_scaler(X_train_df)
        X_train_scaled = transform_features(scaler, X_train_df)
        X_val_scaled = transform_features(scaler, X_val_df)

        # Create sequences for GRU
        X_seq_train, y_seq_train = create_sequences(
            X_train_scaled, y_train, lookback=MODEL_CONFIG.lookback
        )
        X_seq_val, y_seq_val = create_sequences(
            X_val_scaled, y_val, lookback=MODEL_CONFIG.lookback
        )

        # Train GRU
        gru_model = build_gru_model(input_shape=(MODEL_CONFIG.lookback, n_features))

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=MODEL_CONFIG.gru_patience,
            restore_best_weights=True,
            verbose=0,
        )

        gru_model.fit(
            X_seq_train,
            y_seq_train,
            validation_data=(X_seq_val, y_seq_val),
            epochs=MODEL_CONFIG.gru_epochs,
            batch_size=MODEL_CONFIG.gru_batch_size,
            callbacks=[early_stopping],
            verbose=0,
        )

        p_gru_val = gru_model.predict(X_seq_val, verbose=0).reshape(-1)
        gru_metrics.append(_compute_metrics(y_seq_val, p_gru_val))

        # Train Logistic Regression for ensemble
        logreg = build_logistic_model()
        logreg.fit(X_train_scaled, y_train)

        # Align tabular data with sequences
        X_tab_val_aligned = X_val_scaled[-len(y_seq_val):]

        # Create ensemble
        ensemble = GRULogisticEnsemble(gru_model, logreg, weights=None)
        p_ens_val = ensemble.predict_proba(
            X_tabular=X_tab_val_aligned, X_seq=X_seq_val
        )
        ensemble_metrics.append(_compute_metrics(y_seq_val, p_ens_val))

    metrics_gru = _aggregate_metrics(gru_metrics)
    metrics_ensemble = _aggregate_metrics(ensemble_metrics)

    # Select best model type based on compound metric
    def calculate_smart_score(m: Dict[str, float]) -> float:
        """
        Calculates a score that rewards performance and penalizes instability.
        Formula: 50% AUC + 30% F1 + 20% Stability
        """
        stability_penalty = min(m["auc_std"] / 0.15, 1.0)
        stability_score = 1.0 - stability_penalty

        score = (0.5 * m["auc_mean"]) + \
                (0.3 * m["f1_mean"]) + \
                (0.2 * stability_score)

        return score

    # Calculate scores
    scores = {
        "gru": calculate_smart_score(metrics_gru),
        "ensemble": calculate_smart_score(metrics_ensemble),
    }

    best_model_type = max(scores, key=scores.get)

    # Print for debugging
    print("\n[GRU Model Selection] Final Scores (AUC + F1 + Stability):")
    for model, score in scores.items():
        print(f"  - {model}: {score:.4f} (AUC: {locals()[f'metrics_{model}']['auc_mean']:.4f})")

    return {
        "metrics_gru": metrics_gru,
        "metrics_ensemble": metrics_ensemble,
        "best_model_type": best_model_type,
    }
