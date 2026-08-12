"""Feature engineering for ml3 — byte-for-byte port of the delivered preprocess.py.

Replicates ``apply_feature_engineering`` from inbox/a03/codigo/src/data_processing/
preprocess.py so the plugin reproduces the delivered model exactly (train-serving
skew included: in inference the features are computed on the 168-row window AFTER
truncation/padding, see manifest known_issues).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GDD_BASE_TEMP = 10.0
WET_LLUVIA_THRESHOLD = 0.1
WET_HR_THRESHOLD = 90.0


def _calcular_gdd_serie(temp_series: np.ndarray, t_base: float = GDD_BASE_TEMP) -> np.ndarray:
    """Thermal integral (Growing Degree Days) over a series, identical to vid_simulator."""
    n_steps = len(temp_series)
    gdd = np.zeros(n_steps)
    acumulado = 0.0
    pasos_dia = 24
    dias = n_steps // pasos_dia

    for d in range(dias):
        i0 = d * pasos_dia
        i1 = i0 + pasos_dia
        t_media = np.mean(temp_series[i0:i1])
        acumulado += max(0, t_media - t_base)
        gdd[i0:i1] = acumulado

    resto = n_steps % pasos_dia
    if resto > 0:
        gdd[dias * pasos_dia:] = acumulado

    return gdd


def _calcular_horas_mojado_serie(lluvia_series: np.ndarray, hr_series: np.ndarray) -> np.ndarray:
    """Consecutive leaf-wetness hours; the leaf is wet when rain > threshold OR HR > threshold."""
    n_steps = len(lluvia_series)
    wetness = np.zeros(n_steps)
    count = 0

    for t in range(n_steps):
        if lluvia_series[t] > WET_LLUVIA_THRESHOLD or hr_series[t] > WET_HR_THRESHOLD:
            count += 1
        else:
            count = 0
        wetness[t] = count

    return wetness


def apply_feature_engineering(df: pd.DataFrame, date_column: str = "Fecha",
                              series_column: str = "ID_Serie") -> pd.DataFrame:
    """Apply temporal and physical feature engineering, generating derived columns.

    Same contract as the delivered ``apply_feature_engineering``: Hora_Sin/Hora_Cos from the
    date column, GDD_Acumulado (base 10 °C, per series) and Horas_Humedad_Foliar (per series).
    A copy is returned — the caller's DataFrame is never mutated.
    """
    out = df.copy()

    if date_column not in out.columns:
        raise ValueError(
            f"Columna '{date_column}' no encontrada en el input. Es imprescindible para "
            "generar Hora_Sin y Hora_Cos."
        )
    if not pd.api.types.is_datetime64_any_dtype(out[date_column]):
        out[date_column] = pd.to_datetime(out[date_column])

    out["Hora"] = out[date_column].dt.hour
    out["Hora_Sin"] = np.sin(2 * np.pi * out["Hora"] / 24).astype("float32")
    out["Hora_Cos"] = np.cos(2 * np.pi * out["Hora"] / 24).astype("float32")

    if "GDD_Acumulado" not in out.columns:
        if "Temp_Amb_C" not in out.columns:
            raise ValueError(
                "Se necesita 'Temp_Amb_C' para calcular 'GDD_Acumulado', pero no está en el input."
            )
        if series_column in out.columns:
            gdd_results = []
            for _, group in out.groupby(series_column, sort=False):
                gdd_results.append(_calcular_gdd_serie(group["Temp_Amb_C"].values))
            out["GDD_Acumulado"] = np.concatenate(gdd_results).astype("float32")
        else:
            out["GDD_Acumulado"] = _calcular_gdd_serie(out["Temp_Amb_C"].values).astype("float32")

    if "Horas_Humedad_Foliar" not in out.columns:
        if "Lluvia_mm" not in out.columns or "Hum_Rel_Pct" not in out.columns:
            raise ValueError(
                "Se necesitan 'Lluvia_mm' y 'Hum_Rel_Pct' para calcular "
                "'Horas_Humedad_Foliar', pero no están en el input."
            )
        if series_column in out.columns:
            mojado_results = []
            for _, group in out.groupby(series_column, sort=False):
                mojado_results.append(
                    _calcular_horas_mojado_serie(
                        group["Lluvia_mm"].values, group["Hum_Rel_Pct"].values
                    )
                )
            out["Horas_Humedad_Foliar"] = np.concatenate(mojado_results).astype("float32")
        else:
            out["Horas_Humedad_Foliar"] = _calcular_horas_mojado_serie(
                out["Lluvia_mm"].values, out["Hum_Rel_Pct"].values
            ).astype("float32")

    return out
