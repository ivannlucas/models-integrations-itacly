"""Retraining logic for ml40 — faithful port of src/training/{refrig,aireado}_trainer.py.

Reproduces the AI team's original training procedure (stratified 80/20 split by run_id,
per-system hyperparameters from config.yaml, expert sample weights and StandardScaler for
refrigeration, no scaler for aeration) on a user-provided labeled CSV. Hold-out metrics are
computed with the FULL production pipeline (RF + neurosymbolic rules + per-run vote) so they
are comparable with the memoria's Tabla 4.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis import postprocessing
from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.constants import (
    MODEL_PARAMS,
    REFRIGERACION_BINARY_COLS,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


def _split_by_run(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 stratified split over whole cycles, as both original trainers do."""
    run_labels = df.groupby("run_id")[target].first()
    train_runs, test_runs = train_test_split(
        run_labels.index, test_size=0.2, random_state=42, stratify=run_labels,
    )
    return df[df["run_id"].isin(train_runs)].copy(), df[df["run_id"].isin(test_runs)].copy()


def _compute_sample_weights(df: pd.DataFrame, target_col: str = "fault_numeric") -> np.ndarray:
    """Balanced class weights x expert pressure multipliers (refrig_trainer.py)."""
    classes = np.unique(df[target_col])
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=df[target_col])
    weights = df[target_col].map(dict(zip(classes, class_weights))).values.copy()

    p_dis_threshold = df["mean_P_dis_bar"].quantile(0.90)
    early_error_threshold = df["early_P_dis_error"].quantile(0.90)
    weights *= np.where(df["early_P_dis_error"] > early_error_threshold, 1.5, 1.0)
    weights *= np.where(df["mean_P_dis_bar"] > p_dis_threshold, 1.2, 1.0)
    return weights


def _train_refrigeration(df: pd.DataFrame) -> tuple[RandomForestClassifier, StandardScaler, pd.DataFrame, dict]:
    """Port of train_refrigeration: returns (model, scaler, test_df, train_stats)."""
    df = df.copy()
    df["fault_numeric"] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=["fault_numeric"]).copy()
    if len(df) == 0:
        raise ValueError(f"No hay datos para entrenar en la columna {TARGET_COLUMN}")
    df["fault_numeric"] = df["fault_numeric"].astype(int)

    train_df, test_df = _split_by_run(df, "fault_numeric")

    drop_cols = ["fault_numeric", "fault", "run_id", "fault_id", "time_min",
                 "T_cond_sat", "T_cab_meas", "P_suc_bar"]
    x_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y_train = train_df["fault_numeric"]

    sample_weights = _compute_sample_weights(train_df)

    numerical_cols = [c for c in x_train.columns if c not in REFRIGERACION_BINARY_COLS]
    scaler = StandardScaler()
    x_train_scaled = x_train.copy()
    x_train_scaled[numerical_cols] = scaler.fit_transform(x_train[numerical_cols])

    model = RandomForestClassifier(**MODEL_PARAMS["refrigeracion"])
    model.fit(x_train_scaled, y_train, sample_weight=sample_weights)

    stats = {"mean": x_train.mean().to_dict(), "std": x_train.std().to_dict()}
    return model, scaler, test_df, stats


def _train_aireado(df: pd.DataFrame) -> tuple[RandomForestClassifier, None, pd.DataFrame, dict]:
    """Port of train_aireado: returns (model, None, test_df, train_stats)."""
    features = [col for col in df.columns if col not in ["run_id", "time_min", "fault_id", "fault"]]
    train_df, test_df = _split_by_run(df, TARGET_COLUMN)

    x_train = train_df[features]
    y_train = train_df[TARGET_COLUMN]

    model = RandomForestClassifier(**MODEL_PARAMS["aireado"])
    model.fit(x_train, y_train)

    stats = {"mean": x_train.mean().to_dict(), "std": x_train.std().to_dict()}
    return model, None, test_df, stats


def train_system(engineered_df: pd.DataFrame, system: str, thresholds: dict) -> dict:
    """Train one subsystem from an engineered, labeled frame and evaluate the full pipeline.

    Returns {model, scaler, stats, metrics, n_samples, n_runs_train, n_runs_test}.
    """
    if system == "refrigeracion":
        model, scaler, test_df, stats = _train_refrigeration(engineered_df)
    else:
        model, scaler, test_df, stats = _train_aireado(engineered_df)

    test_df = test_df.reset_index(drop=True)
    y_true = pd.to_numeric(test_df[TARGET_COLUMN], errors="coerce").astype(int)
    y_ml, _ = postprocessing.run_inference(model, scaler, test_df, system)
    y_ns = postprocessing.apply_neurosymbolic_rules(test_df, y_ml, system, thresholds)
    y_final = postprocessing.apply_run_voting(test_df, y_ns)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_final)),
        "f1_macro": float(f1_score(y_true, y_final, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_final, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_final, average="macro", zero_division=0)),
    }
    n_runs_test = int(test_df["run_id"].nunique())
    n_runs_train = int(engineered_df["run_id"].nunique()) - n_runs_test
    logger.info(
        "ml40 train_system(%s) — n_train_rows=%d n_runs_train=%d n_runs_test=%d f1_macro=%.4f",
        system, len(engineered_df) - len(test_df), n_runs_train, n_runs_test, metrics["f1_macro"],
    )
    return {
        "model": model,
        "scaler": scaler,
        "stats": stats,
        "metrics": metrics,
        "n_samples": int(len(engineered_df) - len(test_df)),
        "n_runs_train": n_runs_train,
        "n_runs_test": n_runs_test,
    }
