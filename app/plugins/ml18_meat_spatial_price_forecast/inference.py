"""Inference core for ml18 — shared by predict_inline/predict_batch.

Faithful port of src/predict/predictor.py::run_inference()/inverse_scale_predictions() from the
delivered code, restricted to the "forecast" mode (see inbox/a18/manifest.yaml known_issues).
"""
from __future__ import annotations

import numpy as np

from app.plugins.ml18_meat_spatial_price_forecast.preprocessing import build_modeling_panel, build_sequences


def run_inference(bundle: dict, rows: list[dict]) -> list[dict]:
    """Derive the modeling panel from raw rows and predict next-month price per group.

    Returns a list of {"CCAA", "Producto", "Fecha", "predicted_price"} — one per
    (CCAA, Producto) group with enough historical rows (see build_sequences).
    """
    panel = build_modeling_panel(rows)
    entries = build_sequences(panel)

    x_scaler = bundle["x_scaler"]
    y_scaler = bundle["y_scaler"]
    model = bundle["model"]

    windows = np.stack([e["window"] for e in entries])
    n_samples, lookback, n_features = windows.shape
    flat = windows.reshape(-1, n_features)
    flat_scaled = x_scaler.transform(flat)
    windows_scaled = flat_scaled.reshape(n_samples, lookback, n_features).astype(np.float32)

    y_pred_scaled = model.predict(windows_scaled, verbose=0).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    return [
        {
            "CCAA": e["ccaa"],
            "Producto": e["producto"],
            "Fecha": str(e["target_date"].date()),
            "predicted_price": round(float(pred), 6),
        }
        for e, pred in zip(entries, y_pred)
    ]
