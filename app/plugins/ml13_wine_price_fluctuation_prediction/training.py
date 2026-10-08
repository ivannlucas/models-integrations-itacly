"""Retraining logic for ml13 — port of the AI team's modules/ML_models/training.py
(train_with_walk_forward + train_final_models) and inference.py::evaluate_on_last_weeks metrics.

Trains fresh scaler/model objects in memory and returns them; persisting is the plugin's job, so
the served fixed artifacts are never mutated. The original procedure retrains from scratch and
re-selects between LogisticRegression and XGBoost by Smart Score — it is not incremental
fine-tuning (see inbox/a13/manifest.yaml training.note).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.plugins.ml13_wine_price_fluctuation_prediction.constants import (
    DECISION_THRESHOLD,
    FEATURE_COLUMNS,
    GAP,
    LOGREG_PARAMS,
    N_FOLDS,
    RANDOM_SEED,
    RETURN_THRESHOLD,
    SMART_SCORE_WEIGHTS,
    STABILITY_STD_CAP,
    TARGET_WINDOW,
    TEST_SIZE,
    XGB_PARAMS,
)
from app.plugins.ml13_wine_price_fluctuation_prediction.preprocessing import (
    generate_technical_features,
)

logger = logging.getLogger(__name__)


def build_logistic_model() -> LogisticRegression:
    """models.py::build_logistic_model."""
    return LogisticRegression(**LOGREG_PARAMS, random_state=RANDOM_SEED)


def build_xgboost_model(scale_pos_weight: float | None = None) -> XGBClassifier:
    """models.py::build_xgboost_model."""
    return XGBClassifier(random_state=RANDOM_SEED, scale_pos_weight=scale_pos_weight, **XGB_PARAMS)


def _scale_pos_weight(y: np.ndarray) -> float:
    n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())
    return n_neg / n_pos if n_pos > 0 else 1.0


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """training.py::_compute_metrics / inference.py::_compute_test_metrics (threshold 0.5)."""
    auc = 0.5 if len(np.unique(y_true)) == 1 else roc_auc_score(y_true, y_prob)
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    return {
        "auc": float(auc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
    }


def adaptive_walk_forward_splits(n_samples: int, y: np.ndarray, n_splits: int, gap: int):
    """training.py::get_adaptive_walk_forward_splits — expands a single-class validation block
    week by week until it contains both classes."""
    if n_samples <= n_splits + gap:
        raise ValueError(
            f"Insufficient data: {n_samples} samples for {n_splits} folds with gap={gap}"
        )
    val_size_base = (n_samples - gap) // (n_splits + 1)
    if val_size_base < 1:
        raise ValueError(
            f"val_size_base={val_size_base} is too small. "
            f"Data: {n_samples}, splits: {n_splits}, gap: {gap}"
        )

    for fold_idx in range(n_splits):
        train_end = (fold_idx + 1) * val_size_base
        val_start = train_end + gap
        val_end = min(val_start + val_size_base, n_samples)
        if val_start >= n_samples:
            break
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        classes = np.unique(y[val_idx])
        if len(classes) >= 2:
            yield train_idx, val_idx
            continue
        missing_class = 0 if 1 in classes else 1
        while val_end < n_samples:
            val_end += 1
            val_idx = np.arange(val_start, val_end)
            if missing_class in y[val_idx]:
                logger.info("Fold %d validation block expanded to week %d", fold_idx + 1, val_end)
                break
        if len(np.unique(y[val_idx])) < 2:
            raise ValueError(
                f"Fold {fold_idx + 1}: Could not find class {missing_class} in available range. "
                "Consider reducing the return threshold or the number of folds."
            )
        yield train_idx, val_idx


def _aggregate(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    auc_vals = [m["auc"] for m in metrics_list]
    return {
        "auc_mean": float(np.mean(auc_vals)),
        "auc_std": float(np.std(auc_vals, ddof=1)),
        "accuracy_mean": float(np.mean([m["accuracy"] for m in metrics_list])),
        "f1_mean": float(np.mean([m["f1"] for m in metrics_list])),
        "precision_mean": float(np.mean([m["precision"] for m in metrics_list])),
        "recall_mean": float(np.mean([m["recall"] for m in metrics_list])),
    }


def smart_score(m: dict[str, float]) -> float:
    """0.5*AUC_mean + 0.3*F1_mean + 0.2*(1 - min(auc_std/0.15, 1))."""
    stability = 1.0 - min(m["auc_std"] / STABILITY_STD_CAP, 1.0)
    return (
        SMART_SCORE_WEIGHTS["auc"] * m["auc_mean"]
        + SMART_SCORE_WEIGHTS["f1"] * m["f1_mean"]
        + SMART_SCORE_WEIGHTS["stability"] * stability
    )


def _split_trainval_test(df_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df_features) <= TEST_SIZE:
        raise ValueError(
            f"Solo hay {len(df_features)} semanas con features+target: se necesitan más de "
            f"test_size={TEST_SIZE} para reservar el hold-out."
        )
    return df_features.iloc[:-TEST_SIZE].copy(), df_features.iloc[-TEST_SIZE:].copy()


def walk_forward_cv(df_trainval: pd.DataFrame) -> dict:  # pylint: disable=too-many-locals
    """training.py::train_with_walk_forward on the train+val block."""
    x_df = df_trainval[FEATURE_COLUMNS]
    y = df_trainval["target"].values.astype(int)
    per_model: dict[str, list[dict[str, float]]] = {"logreg": [], "xgboost": []}

    splits = adaptive_walk_forward_splits(len(x_df), y, N_FOLDS, GAP)
    for fold, (tr, va) in enumerate(splits, start=1):
        y_tr, y_va = y[tr], y[va]
        if len(np.unique(y_tr)) < 2:
            logger.info("Fold %d: una sola clase en train, se omite", fold)
            continue
        scaler = StandardScaler().fit(x_df.iloc[tr].values)
        x_tr = scaler.transform(x_df.iloc[tr].values)
        x_va = scaler.transform(x_df.iloc[va].values)

        logreg = build_logistic_model().fit(x_tr, y_tr)
        per_model["logreg"].append(compute_metrics(y_va, logreg.predict_proba(x_va)[:, 1]))
        xgb = build_xgboost_model(scale_pos_weight=_scale_pos_weight(y_tr)).fit(x_tr, y_tr)
        per_model["xgboost"].append(compute_metrics(y_va, xgb.predict_proba(x_va)[:, 1]))

    if not per_model["logreg"] or not per_model["xgboost"]:
        raise ValueError(
            f"Could not train any valid fold. Threshold {RETURN_THRESHOLD} may be too high."
        )

    metrics = {name: _aggregate(vals) for name, vals in per_model.items()}
    scores = {name: smart_score(m) for name, m in metrics.items()}
    return {"metrics": metrics, "scores": scores, "best_model_type": max(scores, key=scores.get)}


def train_models(df_price: pd.DataFrame) -> dict:
    """Full original procedure on a cleaned, date-indexed price series.

    Returns fresh scaler/model, CV summary and hold-out (last 24 weeks) metrics.
    """
    df_features = generate_technical_features(df_price, TARGET_WINDOW, RETURN_THRESHOLD)
    df_trainval, df_test = _split_trainval_test(df_features)
    cv = walk_forward_cv(df_trainval)
    best = cv["best_model_type"]

    x_trainval = df_trainval[FEATURE_COLUMNS]
    y_trainval = df_trainval["target"].values.astype(int)
    scaler = StandardScaler().fit(x_trainval.values)
    if best == "xgboost":
        model = build_xgboost_model(scale_pos_weight=_scale_pos_weight(y_trainval))
    else:
        model = build_logistic_model()
    model.fit(scaler.transform(x_trainval.values), y_trainval)

    y_test = df_test["target"].values.astype(int)
    y_prob = model.predict_proba(scaler.transform(df_test[FEATURE_COLUMNS].values))[:, 1]
    test_metrics = compute_metrics(y_test, y_prob)
    logger.info(
        "ml13 train_models() done — best=%s n_trainval=%d n_test=%d test_auc=%.4f",
        best, len(df_trainval), len(df_test), test_metrics["auc"],
    )
    return {
        "model": model,
        "scaler": scaler,
        "model_type": best,
        "cv": cv,
        "test_metrics": test_metrics,
        "n_trainval": int(len(df_trainval)),
        "n_test": int(len(df_test)),
        "test_period": (str(df_test.index[0].date()), str(df_test.index[-1].date())),
    }
