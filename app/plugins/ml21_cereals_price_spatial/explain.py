"""Causal-driver explanation for ml21 — mirrors predict_v1.py's importance/z-score logic.

Real feature importances extracted from the fitted sklearn/xgboost estimators, ranked by
|z-score| * importance against a reference distribution, so causal_drivers reflects the
actual model instead of a static placeholder.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def extract_feature_importance(model: Any, feature_names: list[str]) -> list[dict[str, float]]:
    """Pull per-feature importance out of a fitted estimator (tree, linear, or ensemble)."""
    importances: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_, dtype=float).ravel())
    elif hasattr(model, "estimators_"):
        parts: list[np.ndarray] = []
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                parts.append(np.asarray(est.feature_importances_, dtype=float))
            elif hasattr(est, "coef_"):
                parts.append(np.abs(np.asarray(est.coef_, dtype=float).ravel()))
        if parts:
            importances = np.mean(np.vstack(parts), axis=0)
    elif hasattr(model, "calibrated_classifiers_"):
        parts = []
        for cal in model.calibrated_classifiers_:
            est = getattr(cal, "estimator", None)
            if est is None:
                continue
            if hasattr(est, "feature_importances_"):
                parts.append(np.asarray(est.feature_importances_, dtype=float))
            elif hasattr(est, "coef_"):
                parts.append(np.abs(np.asarray(est.coef_, dtype=float).ravel()))
        if parts:
            importances = np.mean(np.vstack(parts), axis=0)

    if importances is None or len(importances) != len(feature_names):
        return []

    pairs = [
        {"feature": str(f), "importance": float(i)}
        for f, i in zip(feature_names, importances)
        if float(i) > 0
    ]
    pairs.sort(key=lambda x: x["importance"], reverse=True)
    return pairs


def feature_to_human(feature: str) -> str:
    """Render a raw engineered-feature name in a human-readable form for the decision card."""
    pretty = feature.replace("_", " ")
    pretty = pretty.replace("prepag1 urea 46", "Urea")
    pretty = pretty.replace("wheat intl eur", "Trigo Internacional")
    pretty = pretty.replace("corn intl eur", "Maiz Internacional")
    pretty = pretty.replace("eur usd", "EUR/USD")
    pretty = pretty.replace("idx ", "Indice ")
    return pretty


def top_causal_drivers(
    row_features: pd.DataFrame,
    feature_mean: pd.Series,
    feature_std: pd.Series,
    importance_records: list[dict[str, float]],
    top_n: int = 3,
) -> list[str]:
    """Rank the top-N features by |z-score| * importance and format them for the decision card."""
    if not importance_records:
        return ["No hay importancias disponibles para explicar esta prediccion."]

    row = row_features.iloc[0]
    sigma = feature_std.replace(0, np.nan)

    scored: list[tuple[str, float, float]] = []
    for rec in importance_records:
        f = rec["feature"]
        imp = float(rec["importance"])
        if f not in row.index:
            continue
        mu_f = float(feature_mean.get(f, 0.0))
        sigma_f = sigma.get(f, np.nan)
        z = (float(row[f]) - mu_f) / float(sigma_f) if pd.notna(sigma_f) else 0.0
        score = abs(z) * imp
        scored.append((f, score, z))

    if not scored:
        return ["No se pudieron calcular motores causales para este registro."]

    scored.sort(key=lambda x: x[1], reverse=True)
    out: list[str] = []
    for f, _, z in scored[:top_n]:
        direction = "al alza" if z >= 0 else "a la baja"
        out.append(f"{feature_to_human(f)} ({direction}, intensidad relativa z={z:.2f})")
    return out
