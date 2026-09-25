"""Inference core for ml14 — shared by predict_inline/predict_batch.

Faithful port of src/predict/predictor.py::run_prediction()/_predict_with_rnn() from the
delivered code, restricted to the LSTM path only (the served, audited model — see
inbox/a14/manifest.yaml::model_status). Does NOT replicate:

- predictor.py::_pick_best_model() / --model auto: reads the stale/corrupted
  models/metrics/model_comparison.json (sha256 mismatch with what's declared for it — see
  manifest known_issues) and would pick GRU today, not LSTM. This plugin always serves LSTM.
- The "degradation_warning" re-evaluation: it re-derives its baseline RMSE/DA/skill from that
  same untrustworthy report file. Out of scope here for the same reason.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from app.plugins.ml14_wine_phyto_price_forecast.constants import (
    DATE_COL,
    DRIFT_COL,
    HORIZON_WEEKS,
    SEQ_LEN,
    TARGET_SERIES_COL,
)
from app.plugins.ml14_wine_phyto_price_forecast.feature_engineering import build_modeling_features

_GAP_WARNING_THRESHOLD_DAYS = 7


def _build_gap_warning(last_date: pd.Timestamp) -> str | None:
    execution_date = datetime.now(timezone.utc).date()
    gap_days = (execution_date - last_date.date()).days
    if gap_days <= _GAP_WARNING_THRESHOLD_DAYS:
        return None
    gap_weeks = gap_days / 7.0
    return (
        f"El histórico termina el {last_date.date()} pero la ejecución es el "
        f"{execution_date} ({gap_days} días / {gap_weeks:.1f} semanas de gap). "
        f"La ventana de entrada al modelo usa datos hasta {last_date.date()}, no hasta hoy. "
        "Para una predicción actualizada, aporta un histórico más reciente."
    )


def run_inference(bundle: dict, rows: list[dict]) -> dict:
    """Derive features from a raw weekly history and run the LSTM forecast.

    Returns a dict with predicted_price, current_price, drift_baseline, horizon_weeks,
    last_observed_date, prediction_date, gap_warning, n_rows_used and xai_feature_values
    (the last row's feature values, for the explainability service).
    """
    features_df = build_modeling_features(rows)
    feature_columns: list[str] = bundle["feature_columns"]

    seq_df = features_df[feature_columns].tail(SEQ_LEN)
    mean = np.asarray(bundle["input_scaler_mean"], dtype=np.float32)
    scale = np.asarray(bundle["input_scaler_scale"], dtype=np.float32)
    seq_scaled = (seq_df.to_numpy(dtype=np.float32) - mean) / scale

    model = bundle["model"]
    x = torch.tensor(seq_scaled[np.newaxis, ...], dtype=torch.float32)
    with torch.no_grad():
        residual_scaled = float(model(x).item())

    residual = residual_scaled * float(bundle["target_scaler_scale"]) + float(bundle["target_scaler_mean"])

    last_row = features_df.iloc[-1]
    last_date = pd.Timestamp(last_row[DATE_COL])
    current_price = float(last_row[TARGET_SERIES_COL])
    drift = float(last_row[DRIFT_COL])
    drift_baseline = float(current_price + drift * (HORIZON_WEEKS / 4.0))
    predicted_price = drift_baseline + residual

    prediction_date = last_date + pd.Timedelta(weeks=HORIZON_WEEKS)

    return {
        "predicted_price": round(float(predicted_price), 6),
        "current_price": round(current_price, 6),
        "drift_baseline": round(drift_baseline, 6),
        "horizon_weeks": HORIZON_WEEKS,
        "last_observed_date": str(last_date.date()),
        "prediction_date": str(prediction_date.date()),
        "gap_warning": _build_gap_warning(last_date),
        "n_rows_used": len(features_df),
        "xai_feature_values": {col: float(last_row[col]) for col in feature_columns},
    }
