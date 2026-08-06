"""Scaling, inference and response formatting for ml43."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection._vendor.preprocess import (
    STATS_CREATION,
    stats_windows,
)
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.constants import MODEL_ID, SENSOR_COLUMNS

logger = logging.getLogger(__name__)


def scale_sequences(X_arr: np.ndarray, scaler_x) -> np.ndarray:
    """Scale raw sequences [N,T,F] with the fitted StandardScaler, if any."""
    n_feat = X_arr.shape[-1]
    if scaler_x is None:
        return X_arr.astype(np.float32)
    return scaler_x.transform(X_arr.reshape(-1, n_feat)).reshape(X_arr.shape).astype(np.float32)


def compute_scaled_stats(X_arr: np.ndarray, scaler_num) -> np.ndarray:
    """Compute per-window statistics for the fuzzy branch and scale them, if a scaler exists."""
    df_stats = stats_windows(X_arr, feature_names=SENSOR_COLUMNS, stats_creation=STATS_CREATION, ddof=0)
    if scaler_num is None:
        return df_stats.values.astype(np.float32)
    return scaler_num.transform(df_stats.values).astype(np.float32)


def run_model(model, X_scaled: np.ndarray, stats_scaled: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Run the DNF model over scaled sequences/stats and return anomaly probabilities [N]."""
    x_tensor = torch.from_numpy(X_scaled)
    s_tensor = torch.from_numpy(stats_scaled)
    n_windows = x_tensor.shape[0]

    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            end = min(start + batch_size, n_windows)
            out = model(x_tensor[start:end], s_tensor[start:end])
            scores.extend(torch.sigmoid(out["anomaly_score"]).squeeze(-1).cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float32)


def run_xai(
    explainer,
    xai_background: np.ndarray | None,
    x_window: np.ndarray,
    s_stats: np.ndarray,
    threshold: float,
) -> tuple[dict | None, str | None]:
    """Run the CU44 DNFLExplainer with graceful degradation.

    Returns (final_report_dict, None) on success or (None, error_message) on failure.
    """
    if explainer is None or xai_background is None:
        return None, "explainer_not_initialized"
    try:
        result = explainer.explain(
            x_window=x_window,
            s_stats=s_stats,
            background_windows=xai_background,
            anomaly_threshold=threshold,
        )
        return result["final_report"], None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("XAI failed for window: %s", exc)
        return None, str(exc)


def format_batch_predictions(
    X_arr: np.ndarray,
    scores: np.ndarray,
    cycle_ids: list | None,
    threshold: float,
    explainer,
    xai_background,
    X_scaled: np.ndarray,
    stats_scaled: np.ndarray,
) -> list[dict[str, Any]]:
    """Build the per-window prediction records for a batch response, with XAI on anomalies."""
    predictions: list[dict[str, Any]] = []
    for i, score in enumerate(scores):
        is_anomaly = bool(score >= threshold)
        label = "Fallo" if is_anomaly else "No Fallo"
        cycle_id = cycle_ids[i] if cycle_ids else None

        xai_values = {col: float(np.mean(X_arr[i, :, j])) for j, col in enumerate(SENSOR_COLUMNS)}

        if is_anomaly:
            corrective_actions, xai_error = run_xai(
                explainer, xai_background, X_scaled[i], stats_scaled[i], threshold,
            )
        else:
            corrective_actions, xai_error = None, None

        predictions.append({
            "window_index": i + 1,
            "cycle_id": str(cycle_id) if cycle_id is not None else None,
            "predicted_anomaly_class": int(is_anomaly),
            "predicted_anomaly_label": label,
            "anomaly_probability": round(float(score), 6),
            "decision_threshold": threshold,
            "xai_feature_values": xai_values,
            "corrective_actions": corrective_actions,
            "xai_error": xai_error,
        })
    return predictions


def format_inline_response(
    score: float,
    threshold: float,
    features: dict,
    corrective_actions: dict | None,
    xai_error: str | None,
) -> dict:
    """Build the PredictInlineResponse payload dict for a single sensor snapshot."""
    label = "Fallo" if score >= threshold else "No Fallo"
    return {
        "model_id": MODEL_ID,
        "predicted_anomaly_class": int(score >= threshold),
        "predicted_anomaly_label": label,
        "anomaly_probability": round(float(score), 6),
        "decision_threshold": threshold,
        "xai_feature_values": {col: float(features[col]) for col in SENSOR_COLUMNS},
        "corrective_actions": corrective_actions,
        "xai_error": xai_error,
        "model_name": MODEL_ID,
    }
