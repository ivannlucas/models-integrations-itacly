"""Inference execution and response formatting.

Port of src/predict/predictor.py::run_inference and save_predictions, adapted to return
JSON-serializable records instead of writing a CSV.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.plugins.ml16_meat_raw_material_price_alert.constants import TARGETS


def _clean_float(value: Any, ndigits: int = 4) -> float | None:
    """Round to ndigits and convert NaN/inf to None (JSON has no NaN)."""
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fv):
        return None
    return round(fv, ndigits)


def run_inference(
    models: dict,
    bagging_models: dict,
    thresholds: dict,
    x_flat_per_target: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Port of predictor.py::run_inference — proba/pred per target, with optional bagging range.

    Bagging is optional: if bagging_models[target] is empty (missing/failed artifact), the
    uncertainty range collapses to proba_low == proba_high == proba, matching the delivered
    predictor.py's graceful degradation.
    """
    results: dict[str, dict[str, np.ndarray]] = {}
    for target, model in models.items():
        x = x_flat_per_target[target]
        proba = model.predict_proba(x)[:, 1]
        th = float(thresholds[target])
        pred = (proba >= th).astype(int)

        bag_list = bagging_models.get(target) or []
        if bag_list:
            bag_probas = np.stack([bm.predict_proba(x)[:, 1] for bm in bag_list], axis=0)
            proba_low = np.minimum(np.percentile(bag_probas, 10, axis=0), proba)
            proba_high = np.maximum(np.percentile(bag_probas, 90, axis=0), proba)
        else:
            proba_low = proba.copy()
            proba_high = proba.copy()

        results[target] = {
            "proba": proba,
            "pred": pred,
            "threshold": th,
            "proba_low": proba_low,
            "proba_high": proba_high,
        }
    return results


def build_predictions_records(results: dict, fechas: pd.DatetimeIndex) -> list[dict[str, Any]]:
    """Build one JSON record per output month — mirrors predictor.py::save_predictions.

    'fecha' here is the TARGET month (t + horizon), matching the delivered predictor.py
    convention — NOT the input observation month used by trainer.py's predicciones_test.csv.
    See inbox/a16/manifest.yaml known_issues on this date-shift discrepancy.
    """
    n = len(fechas)
    records: list[dict[str, Any]] = []
    for i in range(n):
        record: dict[str, Any] = {"fecha": pd.Timestamp(fechas[i]).strftime("%Y-%m-%d")}
        for target in TARGETS:
            res = results[target]
            record[f"{target}_pred"] = int(res["pred"][i])
            record[f"{target}_proba"] = _clean_float(res["proba"][i])
            record[f"{target}_proba_low"] = _clean_float(res["proba_low"][i])
            record[f"{target}_proba_high"] = _clean_float(res["proba_high"][i])
        records.append(record)
    return records
