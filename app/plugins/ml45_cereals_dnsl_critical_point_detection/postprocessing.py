"""Runs the DNFLExplainer (fuzzy + temporal SHAP + fusion + PCC) per window and formats the
result the same way the original src/predict/xai_predictor.py did: anomaly_probability in each
output row IS the explainer's fused ensemble probability, not a raw model.forward() pass —
there is no cheaper path in the original code, predict() always runs the full XAI pipeline.
"""
from __future__ import annotations

import numpy as np

from app.domain.services.batch_rows import row_error_capture
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


def build_explainer(model, stats_columns: list[str], model_cfg: dict) -> DNFLExplainer:
    """Build the DNFL explainer (SHAP + fuzzy rules + PCC monitor) for *model*."""
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


_XAI_RANDOM_STATE = 42


def explain_window(
    explainer: DNFLExplainer,
    *,
    x_window: np.ndarray,
    s_stats: np.ndarray,
    xai_background: np.ndarray,
    threshold: float,
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
        random_state=_XAI_RANDOM_STATE,
        top_rules=XAI_TOP_RULES,
        top_variables=XAI_TOP_VARIABLES,
    )

    anomaly_probability = float(result["prediction"]["anomaly_probability"])
    predicted_class = int(anomaly_probability >= threshold)

    row = {
        "predicted_anomaly_label": "Fallo" if predicted_class == 1 else "No Fallo",
        "anomaly_probability": round(anomaly_probability, 4),
        "decision_threshold": float(threshold),
    }
    # final_report (DNFLExplainer._build_final_report, vendored — not modified here) also
    # carries "Probabilidad de anomalia"/"Umbral de detección de anomalias" (the same two
    # values already set above under their canonical English names) and "Margen respecto al
    # umbral" (anomaly_probability - decision_threshold, trivially derivable from the two
    # fields already in `row`) — dropped here so the output has one column per real value
    # instead of two names for the same number.
    final_report = dict(result["final_report"])
    for _redundant_key in (
        "Probabilidad de anomalia", "Umbral de detección de anomalias", "Margen respecto al umbral"
    ):
        final_report.pop(_redundant_key, None)
    row.update(final_report)
    return row


def build_predictions(ctx: dict, windows: tuple) -> list[dict]:
    """Explain every window and return one row per window (predict_batch).

    ``ctx`` holds model, model_cfg, xai_background and threshold; ``windows`` is
    (sequences_scaled, stats_scaled, stats_columns, timestamp_windows, entity_ids).
    """
    sequences_scaled, stats_scaled, stats_columns, timestamp_windows, entity_ids = windows
    xai_background, threshold = ctx["xai_background"], ctx["threshold"]
    explainer = build_explainer(ctx["model"], stats_columns, ctx["model_cfg"])

    predictions: list[dict] = []
    for idx, x_window in enumerate(sequences_scaled):
        with row_error_capture(predictions, {"window_index": idx + 1}, what=f"ventana {idx + 1}"):
            row = explain_window(
                explainer,
                x_window=x_window,
                s_stats=stats_scaled[idx],
                xai_background=xai_background,
                threshold=threshold,
            )
            predictions.append({
                "window_index": idx + 1,
                **({"cycle_id": entity_ids[idx]} if entity_ids else {}),
                "timestamp_init": str(timestamp_windows[idx][0]),
                "timestamp_end": str(timestamp_windows[idx][1]),
                **row,
            })
    return predictions
