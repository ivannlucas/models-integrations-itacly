"""
Modulo de fusion: combinacion de explicaciones fuzzy y temporal.
"""

import numpy as np
from typing import Any, Dict, Optional


def _safe_logit_from_prob(p: float, eps: float = 1e-6) -> float:
    """Convierte probabilidad en logit de forma estable."""
    p = float(np.clip(float(p), eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def _resolve_lambda_anomaly(model_cfg: Optional[Dict[str, Any]]) -> float:
    """Resuelve el peso de fusion fuzzy/temporal."""
    if model_cfg is None:
        raise ValueError("model_cfg debe contener 'fuzzy.lambda_anomaly' para resolver el peso de fusion.")
    lam = model_cfg.get("fuzzy", {}).get("lambda_anomaly")
    if lam is None:
        raise ValueError("model_cfg debe contener 'fuzzy.lambda_anomaly' para resolver el peso de fusion.")
    return float(np.clip(float(lam), 0.0, 1.0))


def fuse_fuzzy_temporal_reports(
    fuzzy_report: Dict[str, Any],
    temporal_report: Dict[str, Any],
    model_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fusiona reportes fuzzy y temporal en una estructura unificada."""
    fuzzy_signature = fuzzy_report.get("case_signature", {})
    temporal_variables = temporal_report.get("top_variables", [])

    fuzzy_vars = [
        str(v)
        for v, _ in fuzzy_signature.get("dominant_variables", [])
    ]
    temporal_vars = [
        str(v.get("feature"))
        for v in temporal_variables
        if "feature" in v
    ]

    fuzzy_set = set(fuzzy_vars)
    temporal_set = set(temporal_vars)
    shared = sorted(list(fuzzy_set & temporal_set))
    union = sorted(list(fuzzy_set | temporal_set))
    agreement_score = (len(shared) / len(union)) if union else 0.0

    fuzzy_prob = float(fuzzy_report.get("probability", 0.0))
    fuzzy_class = int(fuzzy_report.get("predicted_class", 0))

    temporal_prob = float(temporal_report.get("probability", 0.0))
    temporal_class = int(temporal_report.get("predicted_class", 0))

    threshold = float(fuzzy_report.get("decision_threshold", temporal_report.get("decision_threshold", 0.5)))
    lambda_anom = _resolve_lambda_anomaly(model_cfg)

    logit_fuzzy = float(fuzzy_report.get("logit", _safe_logit_from_prob(fuzzy_prob)))
    logit_temporal = float(temporal_report.get("logit", _safe_logit_from_prob(temporal_prob)))

    ensemble_logit = (1.0 - lambda_anom) * logit_temporal + lambda_anom * logit_fuzzy
    ensemble_prob = float(1.0 / (1.0 + np.exp(-ensemble_logit)))
    ensemble_class = int(ensemble_prob >= threshold)

    prediction = {
        "decision_threshold": float(threshold),
        "lambda_anomaly": float(lambda_anom),
        "fuzzy": {
            "logit": float(logit_fuzzy),
            "probability": float(fuzzy_prob),
            "predicted_class": int(fuzzy_class),
        },
        "temporal": {
            "logit": float(logit_temporal),
            "probability": float(temporal_prob),
            "predicted_class": int(temporal_class),
        },
        "ensemble": {
            "logit": float(ensemble_logit),
            "probability": float(ensemble_prob),
            "predicted_class": int(ensemble_class),
        },
    }

    return {
        "sample_idx": fuzzy_report.get("sample_idx", temporal_report.get("sample_idx")),
        "split": fuzzy_report.get("split", temporal_report.get("split")),
        "true_class": fuzzy_report.get("true_class", temporal_report.get("true_class")),
        "fuzzy": {
            "support_ratio": float(fuzzy_signature.get("support_ratio", 0.0)),
            "dominant_variables": fuzzy_signature.get("dominant_variables", []),
            "dominant_stats": fuzzy_signature.get("dominant_stats", []),
        },
        "temporal": {
            "dominant_span_shap": temporal_report.get("dominant_span_shap", "unknown"),
            "dominant_span_attention": temporal_report.get("dominant_span_attention", "unknown"),
            "top_variables": temporal_variables,
        },
        "agreement": {
            "score": float(agreement_score),
            "shared_variables": shared,
            "union_variables": union,
            "branch_agreement": bool(fuzzy_class == temporal_class),
        },
        "prediction": prediction,
    }
