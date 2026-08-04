from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd


def merge_runtime_config_with_bundle(cfg: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    """Uses the training snapshot for model semantics, keeping runtime paths and device."""
    merged = deepcopy(cfg)
    snapshot = bundle.get("config_snapshot")
    if not isinstance(snapshot, dict):
        return merged

    for section in ["data", "synthetic_dataset", "sequence"]:
        payload = snapshot.get(section)
        if isinstance(payload, dict):
            merged[section] = deepcopy(payload)

    training_snapshot = snapshot.get("training")
    if isinstance(training_snapshot, dict):
        merged["training"] = deepcopy(training_snapshot)

        # The execution environment may legitimately choose CPU or GPU without
        # changing the data representation used to train the saved model.
        runtime_training = cfg.get("training", {})
        for key in ("device", "fallback_to_cpu", "quick"):
            if key in runtime_training:
                merged["training"][key] = deepcopy(runtime_training[key])
    return merged


def align_feature_columns(
    X: pd.DataFrame,
    expected_feature_columns: list[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    if not expected_feature_columns:
        return X, X.columns.tolist()

    missing = [col for col in expected_feature_columns if col not in X.columns]
    extra = [col for col in X.columns if col not in expected_feature_columns]
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"faltan columnas esperadas: {missing}")
        if extra:
            parts.append(f"hay columnas nuevas no vistas por el modelo: {extra}")
        raise ValueError("No se puede alinear el dataset con el modelo guardado: " + " | ".join(parts))

    return X.loc[:, expected_feature_columns].copy(), list(expected_feature_columns)
