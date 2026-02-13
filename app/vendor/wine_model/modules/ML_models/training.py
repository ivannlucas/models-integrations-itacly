"""
Training and validation routines for tabular wine price forecasting models.

This module implements the training workflow:
- Build technical features and target from the cleaned price series.
- Split the data into train+validation and test (last 24 weeks).
- Run walk-forward (time series) validation on the train+validation block
  to compare Logistic Regression and XGBoost using AUC.
- Select the best model type based on validation AUC.
- Train final production models on all train+validation data and save
  the artifacts (models, scaler, feature schema and model type config).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import json
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
from sklearn.preprocessing import StandardScaler
import joblib

from modules.common.config import MODEL_CONFIG, MODELS_PROD_DIR
from modules.common.data import load_raw_data, clean_and_index_data
from modules.common.features import (
    generate_technical_features,
    fit_scaler,
    transform_features,
)
from .models import build_logistic_model, build_xgboost_model


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


def train_with_walk_forward(
    df_price: pd.DataFrame,
    n_splits: int = 5,
) -> Dict:
    """
    Run walk-forward validation on the given price series.

    Steps:
    - Build technical features and target.
    - Exclude the last 24 weeks (reserved for test).
    - Run TimeSeriesSplit on the remaining data to train and evaluate
      Logistic Regression and XGBoost.

    Returns:
        A dictionary with:
            - "metrics_logreg": aggregated metrics for Logistic Regression.
            - "metrics_xgboost": aggregated metrics for XGBoost.
            - "best_model_type": one of {"logreg", "xgboost"}.
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

    logreg_metrics: List[FoldMetrics] = []
    xgboost_metrics: List[FoldMetrics] = []

    splits = list(tscv.split(X_df))

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train_df, X_val_df = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Verificar que haya al menos ejemplos de ambas clases en train y val
        n_classes_train = len(np.unique(y_train))
        n_classes_val = len(np.unique(y_val))

        if n_classes_train < 2:
            print(f"⚠️  Fold {fold_idx}: Solo {n_classes_train} clase en train. Saltando fold.")
            continue

        if n_classes_val < 2:
            print(f"⚠️  Fold {fold_idx}: Solo {n_classes_val} clase en validación. Saltando fold.")
            continue

        scaler = fit_scaler(X_train_df)
        X_train_scaled = transform_features(scaler, X_train_df)
        X_val_scaled = transform_features(scaler, X_val_df)

        # Train Logistic Regression
        logreg = build_logistic_model()
        logreg.fit(X_train_scaled, y_train)
        p_logreg_val = logreg.predict_proba(X_val_scaled)[:, 1]
        logreg_metrics.append(_compute_metrics(y_val, p_logreg_val))

        # Train XGBoost
        xgb = build_xgboost_model()
        xgb.fit(X_train_scaled, y_train)
        p_xgb_val = xgb.predict_proba(X_val_scaled)[:, 1]
        xgboost_metrics.append(_compute_metrics(y_val, p_xgb_val))

    # Verificar que se entrenaron al menos algunos folds
    if len(logreg_metrics) == 0 or len(xgboost_metrics) == 0:
        raise ValueError(
            f"No se pudo entrenar ningún fold válido. "
            f"Umbral {MODEL_CONFIG.return_threshold} puede ser demasiado alto. "
            f"Folds intentados: {len(splits)}, válidos: {len(logreg_metrics)}"
        )

    if len(logreg_metrics) < len(splits) // 2:
        print(f"⚠️  ADVERTENCIA: Solo se entrenaron {len(logreg_metrics)} de {len(splits)} folds.")
        print(f"   El umbral {MODEL_CONFIG.return_threshold} puede ser demasiado alto para este dataset.")

    metrics_logreg = _aggregate_metrics(logreg_metrics)
    metrics_xgboost = _aggregate_metrics(xgboost_metrics)

    # Select best model type based on a compound metric
    def calculate_smart_score(m: Dict[str, float]) -> float:
        """
        Calculates a score that rewards performance and penalizes instability.
        Formula: 50% AUC + 30% F1 + 20% Stability
        """
        # Stability penalty
        stability_penalty = min(m["auc_std"] / 0.15, 1.0)
        stability_score = 1.0 - stability_penalty

        # Weighted Score
        score = (0.5 * m["auc_mean"]) + \
                (0.3 * m["f1_mean"]) + \
                (0.2 * stability_score)

        return score

    # Calculate scores for all candidates
    scores = {
        "logreg": calculate_smart_score(metrics_logreg),
        "xgboost": calculate_smart_score(metrics_xgboost),
    }

    # Select the winner based on the Composite Score
    best_model_type = max(scores, key=scores.get)

    # Print for debugging
    print("\n[Model Selection] Final Scores (AUC + F1 + Stability):")
    for model, score in scores.items():
        print(f"  - {model}: {score:.4f} (AUC: {locals()[f'metrics_{model}']['auc_mean']:.4f})")

    return {
        "metrics_logreg": metrics_logreg,
        "metrics_xgboost": metrics_xgboost,
        "best_model_type": best_model_type,
    }


def train_final_models(df_price: pd.DataFrame, best_model_type: str = "logreg") -> Dict:
    """
    Train final production models on all train+validation data and save artifacts.

    Args:
        df_price: Price DataFrame with date index.
        best_model_type: Model type to save ('logreg' or 'xgboost').

    This function:
    - Builds features on the full series.
    - Excludes the last 24 weeks (reserved for test).
    - Fits a scaler on all train+validation features.
    - Trains the selected model (Logistic Regression or XGBoost).
    - Saves:
        - scaler.pkl
        - feature_schema.json
        - ml_model.pkl (selected model)
        - model_config.json with model_type info.
    """
    df_features = generate_technical_features(
        df_price,
        target_window=MODEL_CONFIG.target_window,
        return_threshold=MODEL_CONFIG.return_threshold,
    )

    df_trainval, _ = _split_trainval_test(df_features, test_weeks=MODEL_CONFIG.test_size)

    feature_cols = ["logret", "distsma12", "rsi14", "bollingerpos", "weeksin", "weekcos"]
    X_trainval_df = df_trainval[feature_cols]
    y_trainval = df_trainval["target"].values.astype(int)

    scaler = fit_scaler(X_trainval_df)
    X_trainval_scaled = transform_features(scaler, X_trainval_df)

    MODELS_PROD_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = MODELS_PROD_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    schema_path = MODELS_PROD_DIR / "feature_schema.json"
    with schema_path.open("w", encoding="utf-8") as f:
        json.dump({"feature_columns": feature_cols}, f, indent=2)

    # Train final model based on best_model_type
    if best_model_type == "logreg":
        final_model = build_logistic_model()
    elif best_model_type == "xgboost":
        final_model = build_xgboost_model()
    else:
        # Default to logistic regression
        final_model = build_logistic_model()

    final_model.fit(X_trainval_scaled, y_trainval)

    ml_model_path = MODELS_PROD_DIR / "ml_model.pkl"
    joblib.dump(final_model, ml_model_path)

    # Save model configuration
    model_config: Dict = {"model_type": best_model_type}
    model_config_path = MODELS_PROD_DIR / "model_config.json"
    with model_config_path.open("w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    return {"model_type": best_model_type, "artifacts_saved": True}
