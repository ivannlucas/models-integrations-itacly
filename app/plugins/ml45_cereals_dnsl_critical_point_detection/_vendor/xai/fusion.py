"""Vendored verbatim from inbox/a45/codigo/src/xai/fusion.py."""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional


def _safe_logit_from_prob(p: float, eps: float = 1e-6) -> float:
    """Convierte probabilidad en logit de forma estable."""
    p = float(np.clip(float(p), eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def _resolve_fuzzy_class(fuzzy_report: Dict[str, Any]) -> int:
    """Resuelve clase fuzzy tolerando variantes de claves."""
    if "predicted_class_fuzzy" in fuzzy_report:
        return int(fuzzy_report.get("predicted_class_fuzzy", 0))
    return int(fuzzy_report.get("predicted_class", 0))


def _resolve_fuzzy_prob(fuzzy_report: Dict[str, Any]) -> float:
    """Resuelve probabilidad fuzzy tolerando variantes de claves."""
    if "fuzzy_probability" in fuzzy_report:
        return float(fuzzy_report.get("fuzzy_probability", 0.0))
    return float(fuzzy_report.get("anomaly_probability", 0.0))


def _resolve_threshold(
    fuzzy_report: Dict[str, Any],
    lstm_report: Dict[str, Any],
) -> float:
    """Resuelve umbral de decisión priorizando fuzzy y luego temporal."""
    return float(
        fuzzy_report.get(
            "decision_threshold",
            lstm_report.get("decision_threshold", 0.5),
        )
    )


def _resolve_lambda_anomaly(model_cfg: Optional[Dict[str, Any]]) -> float:
    """Resuelve el peso real de fusión fuzzy/DL."""
    if not isinstance(model_cfg, dict):
        raise ValueError("model_cfg debe ser un diccionario.")

    fuzzy_cfg = model_cfg.get("fuzzy")
    if not isinstance(fuzzy_cfg, dict) or "lambda_anomaly" not in fuzzy_cfg:
        raise ValueError("model_cfg.fuzzy.lambda_anomaly es obligatorio.")

    return float(np.clip(float(fuzzy_cfg["lambda_anomaly"]), 0.0, 1.0))


def fuse_fuzzy_lstm_reports(
    fuzzy_report: Dict[str, Any],
    lstm_report: Dict[str, Any],
    model_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fusiona reportes fuzzy y temporal en una estructura unificada."""
    fuzzy_vars = [
        str(v)
        for v, _ in fuzzy_report.get("fuzzy_case_signature", {}).get("dominant_variables", [])
    ]
    lstm_vars = [
        str(v.get("feature"))
        for v in lstm_report.get("top_variables", [])
        if "feature" in v
    ]

    fuzzy_set = set(fuzzy_vars)
    lstm_set = set(lstm_vars)
    shared = sorted(list(fuzzy_set & lstm_set))
    union = sorted(list(fuzzy_set | lstm_set))
    agreement_score = (len(shared) / len(union)) if union else 0.0

    fuzzy_prob = _resolve_fuzzy_prob(fuzzy_report)
    fuzzy_class = _resolve_fuzzy_class(fuzzy_report)

    dl_prob = float(lstm_report.get("probability_dl", 0.0))
    dl_class = int(lstm_report.get("predicted_class_dl", 0))

    threshold = _resolve_threshold(fuzzy_report, lstm_report)
    lambda_anom = _resolve_lambda_anomaly(model_cfg)

    logit_fuzzy = float(fuzzy_report.get("fuzzy_logit", _safe_logit_from_prob(fuzzy_prob)))
    logit_dl = float(lstm_report.get("logit_dl", _safe_logit_from_prob(dl_prob)))

    ensemble_logit = (1.0 - lambda_anom) * logit_dl + lambda_anom * logit_fuzzy
    ensemble_prob = float(1.0 / (1.0 + np.exp(-ensemble_logit)))
    ensemble_class = int(ensemble_prob >= threshold)

    branch_agreement = bool(fuzzy_class == dl_class)

    fusion_summary = {
        "agreement_score": float(agreement_score),
        "shared_variables": shared,
        "union_variables": union,
        "branch_agreement": branch_agreement,
        "fuzzy_support_ratio": float(fuzzy_report.get("fuzzy_case_signature", {}).get("support_ratio", 0.0)),
        "lstm_dominant_span_shap": lstm_report.get("dominant_span_shap", "unknown"),
        "lstm_dominant_span_attention": lstm_report.get("dominant_span_attention", "unknown"),
        "lstm_dominant_patterns": lstm_report.get("dominant_patterns", []),
        "lambda_anomaly": float(lambda_anom),
    }

    prediction = {
        "fuzzy_prob": float(fuzzy_prob),
        "fuzzy_class": int(fuzzy_class),
        "fuzzy_logit": float(logit_fuzzy),
        "dl_prob": float(dl_prob),
        "dl_class": int(dl_class),
        "dl_logit": float(logit_dl),
        "ensemble_logit": float(ensemble_logit),
        "ensemble_prob": float(ensemble_prob),
        "ensemble_class": int(ensemble_class),
        "decision_threshold": float(threshold),
        "lambda_anomaly": float(lambda_anom),
    }

    return {
        "sample_idx": fuzzy_report.get("sample_idx", lstm_report.get("sample_idx")),
        "split": fuzzy_report.get("split", lstm_report.get("split")),
        "true_class": fuzzy_report.get("true_class", lstm_report.get("true_class")),
        "fuzzy": {
            "predicted_class_fuzzy": int(fuzzy_class),
            "fuzzy_probability": float(fuzzy_prob),
            "fuzzy_mode": fuzzy_report.get("fuzzy_mode", "unknown"),
            "support_ratio": float(fuzzy_report.get("fuzzy_case_signature", {}).get("support_ratio", 0.0)),
            "dominant_variables": fuzzy_report.get("fuzzy_case_signature", {}).get("dominant_variables", []),
            "dominant_stats": fuzzy_report.get("fuzzy_case_signature", {}).get("dominant_stats", []),
        },
        "lstm": {
            "predicted_class_dl": int(dl_class),
            "probability_dl": float(dl_prob),
            "dominant_span_shap": lstm_report.get("dominant_span_shap", "unknown"),
            "dominant_span_attention": lstm_report.get("dominant_span_attention", "unknown"),
            "dominant_patterns": lstm_report.get("dominant_patterns", []),
            "top_variables": lstm_report.get("top_variables", []),
            "top_timesteps": lstm_report.get("top_timesteps", []),
        },
        "agreement": {
            "score": float(agreement_score),
            "shared_variables": shared,
            "union_variables": union,
            "branch_agreement": branch_agreement,
        },
        "prediction": prediction,
        "fusion_summary": fusion_summary,
    }
