"""Scaling, inference and response formatting for ml43-cereals-dnsl-anomaly-fault-detection."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection._vendor.preprocess import STATS_CREATION, stats_windows
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.constants import SENSOR_COLUMNS

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
    """Run the DNFLExplainer with graceful degradation.

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


def _stringify_timestamp(value: Any) -> str | None:
    """Convert a raw timestamp scalar (numpy/pandas/str) to a JSON-safe string, or None."""
    return None if value is None else str(value)


def format_batch_predictions(
    X_arr: np.ndarray,
    scores: np.ndarray,
    cycle_ids: list | None,
    threshold: float,
    explainer,
    xai_background,
    X_scaled: np.ndarray,
    stats_scaled: np.ndarray,
    window_timestamps: list | None = None,
) -> list[dict[str, Any]]:
    """Build the per-window prediction records for a batch response, with XAI on anomalies."""
    predictions: list[dict[str, Any]] = []
    for i, score in enumerate(scores):
        is_anomaly = bool(score >= threshold)
        # Feedback 3 (modelo 43-44, 09/09/2026): label was hardcoded in English —
        # ml45_cereals_dnsl_critical_point_detection/postprocessing.py already uses the
        # Spanish "Fallo"/"No Fallo" pair, mirrored here for consistency across models.
        label = "Fallo" if is_anomaly else "No Fallo"
        cycle_id = cycle_ids[i] if cycle_ids else None
        ts_init, ts_end = (None, None)
        if window_timestamps is not None and i < len(window_timestamps):
            ts_init, ts_end = window_timestamps[i]

        xai_values = {col: float(np.mean(X_arr[i, :, j])) for j, col in enumerate(SENSOR_COLUMNS)}

        # Feedback 3 (modelo 43-44 audit): XAI used to run only for anomalous windows
        # (is_anomaly), leaving every "No Fallo" row's XAI columns empty — not a failure,
        # run_xai was simply never called for them. _build_final_report already produces
        # a full, distinct report for every Estado_interpretativo (Normal, Normal con
        # señales, Alerta no confirmada, Anomalía confirmada), so it must run for every
        # window, not just the ones over the binary decision threshold.
        xai_result, xai_error = run_xai(
            explainer, xai_background, X_scaled[i], stats_scaled[i], threshold,
        )

        predictions.append({
            # 1-indexed (window 1..N), not the raw 0-based loop counter.
            "window_index": i + 1,
            "cycle_id": str(cycle_id) if cycle_id is not None else None,
            "timestamp_init": _stringify_timestamp(ts_init),
            "timestamp_end": _stringify_timestamp(ts_end),
            "predicted_anomaly_label": label,
            "anomaly_probability": round(float(score), 6),
            "decision_threshold": threshold,
            "xai_feature_values": xai_values,
            "xai_result": xai_result,
            "xai_error": xai_error,
        })
    return predictions
