"""Inference core for ml37 — faithful port of src/predict/predictor.py::run_predict (the model
forward pass + decode, not the file I/O) and src/training/trainer.py::risk_from_probs /
sem_from_risk.
"""
from __future__ import annotations

import numpy as np

from app.plugins.ml37_wine_press_predictive_maintenance.constants import (
    RISK_GREEN_MAX,
    RISK_YELLOW_MAX,
    TARGET_NAMES,
)


def risk_from_probs(probs: np.ndarray) -> float:
    """Predictive uncertainty = 1 - max(softmax probability). NOT a physical risk/health
    signal — faithful to trainer.py::risk_from_probs's own docstring."""
    return float(1.0 - float(np.max(probs)))


def sem_from_risk(risk: float) -> str:
    """Uncertainty traffic light — faithful to trainer.py::sem_from_risk."""
    if risk <= RISK_GREEN_MAX:
        return "green"
    if risk <= RISK_YELLOW_MAX:
        return "yellow"
    return "red"


def run_inference(bundle: dict, X_scaled: np.ndarray) -> list[dict]:
    """Run the LSTM multitask model on a (N, 600, 40) scaled batch and decode every head.

    Returns one dict per row with: pred_<target> (real UCI code), risk_<target>, sem_<target>,
    p_<target>_<clase> (softmax probability per class) — same fields as predictions_all.parquet.
    """
    model = bundle["model"]
    class_mappings = bundle["class_mappings"]
    i2v = class_mappings["mappings_index_to_value"]
    class_order = class_mappings["class_order_by_target"]

    probs_all = model.predict(X_scaled, verbose=0)
    output_names = list(getattr(model, "output_names", TARGET_NAMES))
    pos = {name: i for i, name in enumerate(output_names)}

    n = X_scaled.shape[0]
    rows: list[dict] = []
    for row_i in range(n):
        row: dict = {}
        for tname in TARGET_NAMES:
            p = np.asarray(probs_all[pos[tname]][row_i])
            idx_pred = int(np.argmax(p))
            row[f"pred_{tname}"] = int(i2v[tname][str(idx_pred)])
            risk = risk_from_probs(p)
            row[f"risk_{tname}"] = round(risk, 4)
            row[f"sem_{tname}"] = sem_from_risk(risk)
            for j, cls_val in enumerate(class_order[tname]):
                row[f"p_{tname}_{cls_val}"] = round(float(p[j]), 6)
        rows.append(row)
    return rows


def model_predictions_for_rules(row: dict) -> dict[str, dict]:
    """Shape run_inference()'s row into symbolic_engine.py::check_model's expected input:
    {head: {"class": <pred>, "proba": <confianza de la clase ganadora>}}."""
    predictions = {}
    for tname in TARGET_NAMES:
        pred_class = row[f"pred_{tname}"]
        proba = 1.0 - row[f"risk_{tname}"]
        predictions[tname] = {"class": pred_class, "proba": proba}
    return predictions
