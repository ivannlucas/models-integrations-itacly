"""Builds response dicts for predict_inline/predict_batch from raw inference scores."""
from __future__ import annotations


def build_inline_result(
    *, model_id: str, machine: str, machine_id: str, snr: str,
    mse_score: float, maha_score: float, threshold: float,
) -> dict:
    return {
        "model_id": model_id,
        "machine": machine,
        "machine_id": machine_id,
        "snr": snr,
        "mse_score": round(mse_score, 6),
        "maha_score": round(maha_score, 6),
        "predicted_label": int(maha_score >= threshold),
        "threshold_used": threshold,
    }
