"""Inference core for ml14 — shared by predict_inline/predict_batch.

Faithful port of src/predict/predictor.py::run_prediction()/_predict_with_rnn() from the
delivered (approved) code, restricted to the GRU path only — the model
predictor.py::_pick_best_model() actually selects by RMSE on the reported test split (see
inbox/a14/manifest.yaml). Does NOT replicate the "degradation_warning" re-evaluation (a
re-run of the historical test sequences against training-time baselines) — out of scope for
this plugin's minimal predict contract; see manifest known_issues.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from app.domain.services.exceptions import InsufficientDataError
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
    """Derive features from a raw weekly history and run the GRU forecast.

    Returns a dict with predicted_price, current_price, drift_baseline, horizon_weeks,
    last_observed_date, prediction_date, gap_warning, n_rows_used and xai_feature_values
    (the last row's feature values, for the explainability service).
    """
    features_df = build_modeling_features(rows)
    feature_columns: list[str] = bundle["feature_columns"]

    seq_df = features_df[feature_columns].tail(SEQ_LEN)
    mean = np.asarray(bundle["input_scaler_mean"], dtype=np.float32)
    scale = np.asarray(bundle["input_scaler_scale"], dtype=np.float32)
    # El histórico ya se validó como finito en float64 (feature_engineering.py), pero un valor
    # finito muy grande (p. ej. 1e250) desborda silenciosamente a +/-inf al convertir a float32
    # aquí -- numpy no lanza excepción, solo un RuntimeWarning. Sin esta comprobación, ese
    # desbordamiento llega al modelo y produce una predicción no finita que Pydantic serializa
    # como "predicted_price": null con HTTP 200, en vez de un error explícito.
    seq_scaled = (seq_df.to_numpy(dtype=np.float32) - mean) / scale
    if not np.isfinite(seq_scaled).all():
        raise InsufficientDataError(
            "El histórico contiene valores numéricos demasiado grandes: al menos uno desborda "
            "la precisión numérica del modelo (float32) aunque sea un valor finito válido en la "
            "entrada. Revisa las columnas numéricas del histórico aportado."
        )

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
    if not np.isfinite(predicted_price):
        # Red de seguridad adicional: si, pese al chequeo de arriba, el modelo o la
        # reconstrucción del precio produjeran NaN/inf por cualquier otra vía, lo rechazamos
        # aquí en vez de dejar que Pydantic lo serialice silenciosamente como
        # "predicted_price": null con HTTP 200.
        raise InsufficientDataError(
            "La predicción resultante no es un número finito (posible desbordamiento numérico "
            "en el histórico aportado). Revisa las columnas numéricas del histórico."
        )

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
