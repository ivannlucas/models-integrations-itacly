"""Inference core for ml19 — faithful port of src/predict/predict.py (predict_next_month,
get_row_from_dataset and the month-label helpers) from the delivered code, restricted to
reading the bundled reference dataset (see model_loader.py / manifest.yaml for why).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.domain.services.exceptions import DataContractError
from app.plugins.ml19_cereals_cost_forecast.constants import DATE_COL, MAPA_PUBLICATION_LAG_MONTHS


def shift_month_label(date_label: str, delta: int) -> str:
    """predict.py::_shift_month_label — calendar month 'date_label' shifted by 'delta' months."""
    year, month = map(int, date_label.split("-"))
    total = year * 12 + (month - 1) + delta
    year, month = divmod(total, 12)
    return f"{year:04d}-{month + 1:02d}"


def mapa_month_used(date_label: str, mapa_lag: int = MAPA_PUBLICATION_LAG_MONTHS) -> str:
    """predict.py::_mapa_month_used — most recent MAPA month feeding 'date_label's features."""
    return shift_month_label(date_label, -mapa_lag)


def get_row_from_dataset(
    dataset: pd.DataFrame, features: list[str], date_str: str | None = None,
) -> tuple[pd.Series, str]:
    """predict.py::get_row_from_dataset, against the bundled dataset (already loaded).

    Returns (row_series, date_label).
    """
    df = dataset.dropna(subset=features)

    if date_str is not None:
        mask = df[DATE_COL].dt.strftime("%Y-%m") == date_str
        if not mask.any():
            available = df[DATE_COL].dt.strftime("%Y-%m").tolist()
            raise DataContractError(
                f"Fecha '{date_str}' no disponible en el dataset de referencia empaquetado. "
                f"Rango disponible: {available[0]} -> {available[-1]}."
            )
        row = df.loc[mask].iloc[0]
        date_label = date_str
    else:
        row = df.iloc[-1]
        date_label = row[DATE_COL].strftime("%Y-%m")

    return row, date_label


def predict_next_month(
    row: pd.Series, artifacts: dict, horizons: list[int] | None = None,
) -> dict[int, dict]:
    """predict.py::predict_next_month — regression + classification + ensemble signal per
    horizon, verbatim from the delivered code (only input source changed, see get_row_from_dataset).
    """
    features = artifacts["features"]
    x = row[features].values

    avail_horizons = sorted(set(artifacts["reg"]) | set(artifacts["clf"]))
    if horizons is None:
        horizons = avail_horizons

    results: dict[int, dict] = {}

    for h in horizons:
        h_result: dict = {}

        if h in artifacts["reg"]:
            art = artifacts["reg"][h]
            mdl, sc, mname = art["model"], art["scaler"], art["model_name"]
            x_in = sc.transform(x.reshape(1, -1)) if sc is not None else x.reshape(1, -1)
            pred_reg = float(mdl.predict(x_in)[0])
            h_result["reg"] = {
                "model_name": mname,
                "pred": pred_reg,
                "signal": int(np.sign(pred_reg)) if pred_reg != 0 else 0,
                "signal_str": "LONG" if pred_reg > 0 else ("SHORT" if pred_reg < 0 else "FLAT"),
            }

        if h in artifacts["clf"]:
            art = artifacts["clf"][h]
            mdl, sc, mname = art["model"], art["scaler"], art["model_name"]
            x_in = sc.transform(x.reshape(1, -1)) if sc is not None else x.reshape(1, -1)
            pred_cls = int(mdl.predict(x_in)[0])
            prob_up = float(mdl.predict_proba(x_in)[0][1])
            h_result["clf"] = {
                "model_name": mname,
                "prob_up": prob_up,
                "pred_class": pred_cls,
                "signal": 1 if pred_cls == 1 else -1,
                "signal_str": "LONG" if pred_cls == 1 else "SHORT",
            }

        signals = []
        if "reg" in h_result:
            signals.append(h_result["reg"]["signal"])
        if "clf" in h_result:
            signals.append(h_result["clf"]["signal"])

        if signals:
            ensemble = int(np.sign(sum(signals)))
            h_result["ensemble_signal"] = ensemble
            h_result["ensemble_str"] = "LONG" if ensemble > 0 else ("SHORT" if ensemble < 0 else "FLAT")
            h_result["confidence"] = "ALTA" if len(set(signals)) == 1 else "BAJA"

        results[h] = h_result

    return results
