"""Ensemble postprocessing for ml3 — soft voting + severity mean + treatment KB.

Replicates ``predecir_ensemble`` from inbox/a03 predictor.py: the class is the argmax of the
mean class probabilities across the three networks (Soft Voting); the severity is the arithmetic
mean of the three sigmoid regressions; the treatment comes from the editable knowledge base.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.plugins.ml3_wine_disease_pest_forecast.constants import (
    TREATMENT_KNOWLEDGE_BASE,
)

logger = logging.getLogger(__name__)


# El nombre X_tensor es el del código entregado (predictor.py::predecir_ensemble).
# pylint: disable=invalid-name,too-many-locals


def predict_ensemble(
    models: list, X_tensor: np.ndarray, label_encoder, class_names: list[str],
) -> dict:
    """Run the three networks on *X_tensor* and return the ensemble prediction dict.

    Returns ``{diagnostico_ia, confianza_clasificacion, grado_severidad, probabilidades_clases}``.
    """
    preds_clases_prob, preds_regresion = [], []
    for mod in models:
        p_class, p_reg = mod.predict(X_tensor, verbose=0)
        preds_clases_prob.append(p_class)
        preds_regresion.append(p_reg)

    media_probs = np.mean(preds_clases_prob, axis=0)
    clases_finales_idx = int(np.argmax(media_probs, axis=1)[0])
    confianza_final = float(np.max(media_probs, axis=1)[0])
    grado_final = float(np.mean(preds_regresion, axis=0).flatten()[0])
    prob_vector = media_probs[0]

    if label_encoder is not None:
        class_name = str(label_encoder.inverse_transform([clases_finales_idx])[0])
    else:
        class_name = class_names[clases_finales_idx]

    probabilidades = {str(name): float(prob_vector[i]) for i, name in enumerate(class_names)}

    return {
        "diagnostico_ia": class_name,
        "confianza_clasificacion": confianza_final,
        "grado_severidad": grado_final,
        "probabilidades_clases": probabilidades,
    }


def build_treatment(class_name: str) -> str:
    """Serialize the treatment protocol for *class_name* exactly like the delivered code."""
    trats_dict = TREATMENT_KNOWLEDGE_BASE.get(class_name, {"Aviso": "N/A"})
    return " | ".join(f"{k}: {v}" for k, v in trats_dict.items())


def raw_snapshot(window_df: pd.DataFrame) -> dict[str, Any] | None:
    """Snapshot the last raw row of the scored window for the XAI service."""
    if window_df is None or len(window_df) == 0:
        return None
    last = window_df.iloc[-1]
    raw_cols = ["Temp_Amb_C", "Hum_Rel_Pct", "Lluvia_mm", "Viento_kmh",
                "CO2_ppm", "VOC_ppb", "Hum_Suelo_Pct", "pH_Suelo"]
    snapshot: dict[str, Any] = {}
    for col in raw_cols:
        if col in window_df.columns:
            val = last[col]
            snapshot[col] = float(val) if not _is_missing(val) else None
    return snapshot


def _is_missing(value: Any) -> bool:
    return bool(pd.isna(value))
