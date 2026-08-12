"""Runs the DNFLExplainer (fuzzy + temporal SHAP + fusion + PCC) per window and formats the
result the same way the original src/predict/xai_predictor.py did: anomaly_probability in each
output row IS the explainer's fused ensemble probability, not a raw model.forward() pass —
there is no cheaper path in the original code, predict() always runs the full XAI pipeline.
"""
from __future__ import annotations

import logging

import numpy as np

from app.plugins.ml45_cereals_dnsl_critical_point_detection._vendor.xai import DNFLExplainer
from app.plugins.ml45_cereals_dnsl_critical_point_detection.constants import (
    PCC_CATALOG_RECORDS,
    PCC_MONITOR_POLICY,
    PCC_SUBSYSTEMS_CONFIG,
    SENSOR_COLUMNS,
    STATS_CREATION,
    XAI_N_BACKGROUND,
    XAI_TOP_RULES,
    XAI_TOP_VARIABLES,
)

logger = logging.getLogger(__name__)


def build_explainer(model, stats_columns: list[str], model_cfg: dict) -> DNFLExplainer:
    pcc_cfg = {
        "subsystems": PCC_SUBSYSTEMS_CONFIG,
        "catalog": PCC_CATALOG_RECORDS,
        "monitor_policy": PCC_MONITOR_POLICY,
    }
    return DNFLExplainer(
        model=model,
        feature_names_stats=stats_columns,
        feature_names_original=SENSOR_COLUMNS,
        pcc_cfg=pcc_cfg,
        stats_creation=STATS_CREATION,
        model_cfg=model_cfg,
    )


def explain_window(
    explainer: DNFLExplainer,
    x_window: np.ndarray,
    s_stats: np.ndarray,
    xai_background: np.ndarray,
    threshold: float,
    random_state: int = 42,
) -> dict:
    """Explain a single window and return the flat row dict used by predict_batch/predict_inline."""
    if xai_background is None:
        raise ValueError(
            "No hay background SHAP disponible (xai_background.npy). No se puede generar la "
            "capa XAI/PCC — ver known_issues del manifest."
        )

    result = explainer.explain(
        x_window=x_window,
        s_stats=s_stats,
        background_windows=xai_background,
        anomaly_threshold=threshold,
        n_background=XAI_N_BACKGROUND,
        random_state=random_state,
        top_rules=XAI_TOP_RULES,
        top_variables=XAI_TOP_VARIABLES,
    )

    anomaly_probability = float(result["prediction"]["anomaly_probability"])
    predicted_class = int(anomaly_probability >= threshold)

    row = {
        "predicted_anomaly_class": predicted_class,
        "predicted_anomaly_label": "Fallo" if predicted_class == 1 else "No Fallo",
        "anomaly_probability": round(anomaly_probability, 4),
        "decision_threshold": float(threshold),
    }
    row.update(result["final_report"])
    return row


def build_predictions(
    model,
    sequences_scaled: np.ndarray,
    stats_scaled: np.ndarray,
    stats_columns: list[str],
    timestamp_windows: list,
    entity_ids: list,
    xai_background: np.ndarray,
    threshold: float,
    model_cfg: dict,
) -> list[dict]:
    """Explain every window and return one row per window (predict_batch)."""
    explainer = build_explainer(model, stats_columns, model_cfg)

    predictions: list[dict] = []
    for idx in range(len(sequences_scaled)):
        try:
            row = explain_window(
                explainer,
                x_window=sequences_scaled[idx],
                s_stats=stats_scaled[idx],
                xai_background=xai_background,
                threshold=threshold,
            )
            row = {
                "window_index": idx + 1,
                **({"cycle_id": entity_ids[idx]} if entity_ids else {}),
                "timestamp_init": str(timestamp_windows[idx][0]),
                "timestamp_end": str(timestamp_windows[idx][1]),
                **row,
            }
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Error en ventana %s: %s", idx + 1, exc)
            row = {"window_index": idx + 1, "error": str(exc)}
        predictions.append(row)
    return predictions
