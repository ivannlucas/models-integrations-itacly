"""Hierarchical inference for ml38 — port of src/predict/predictor.py (run_inference and
run_inference_pipeline).

1. ``family_scaler`` → RandomForest ``predict`` → family.
2. If the family is ``healthy``: no severity ("sin fallo").
   If there is no severity model for it: no severity ("modelo no disponible").
   Otherwise the family's own scaler on the UNSCALED features → MLP logit → sigmoid →
   severity = int(prob > 0.5) (0 "leve", 1 "significativo"), confidence = prob.

The original scores severity row by row; here the rows of each family go through the MLP in one
call (same weights, same float32 input — differences are float rounding, < 1e-6). Only keras.ops
is used, so it works with the torch backend the server runs with (KI-04).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.plugins.ml38_wine_machinery_fault_detection.constants import (
    HEALTHY_FAMILY,
    LABEL_NO_FAULT,
    LABEL_NO_SEVERITY_MODEL,
    SEVERITY_LABELS,
    SEVERITY_THRESHOLD,
)
from app.plugins.ml38_wine_machinery_fault_detection.model_loader import ModelBundle


def severity_probabilities(model: Any, x_scaled: np.ndarray) -> np.ndarray:
    """sigmoid(logit) of a Keras severity MLP for a batch of scaled rows (float32 input)."""
    import keras  # pylint: disable=import-outside-toplevel  # see model_loader: lazy on purpose

    logits = model(x_scaled.astype(np.float32), training=False)
    return np.asarray(keras.ops.convert_to_numpy(keras.ops.sigmoid(logits))).reshape(-1)


def predict_matrix(bundle: ModelBundle, features: np.ndarray) -> list[dict[str, Any]]:
    """Predictions for an (n, 24) float matrix in FEATURE_NAMES order, one dict per row."""
    x = np.asarray(features, dtype=float)
    x_family = bundle.family_scaler.transform(x)
    families = bundle.family_model.predict(x_family)
    probas = bundle.family_model.predict_proba(x_family)
    classes = bundle.families

    results: list[dict[str, Any]] = []
    for fam, proba in zip(families, probas):
        fam = str(fam)
        results.append({
            "family": fam,
            "severity": None,
            "severity_label": (
                LABEL_NO_FAULT if fam == HEALTHY_FAMILY else LABEL_NO_SEVERITY_MODEL
            ),
            "severity_confidence": None,
            "family_probabilities": {c: float(p) for c, p in zip(classes, proba)},
        })

    _add_severity(results, bundle, x, families.astype(str))
    return results


def _add_severity(results: list[dict[str, Any]], bundle: ModelBundle, x: np.ndarray,
                  families: np.ndarray) -> None:
    """Fill severity fields in place for the rows whose family has a severity model."""
    for fam in sorted(set(families)):
        if fam == HEALTHY_FAMILY or fam not in bundle.severity_models:
            continue
        idx = np.flatnonzero(families == fam)
        x_sev = bundle.severity_scalers[fam].transform(x[idx])
        for i, prob in zip(idx, severity_probabilities(bundle.severity_models[fam], x_sev)):
            severity = int(prob > SEVERITY_THRESHOLD)
            results[i]["severity"] = severity
            results[i]["severity_label"] = SEVERITY_LABELS[severity]
            results[i]["severity_confidence"] = float(prob)
