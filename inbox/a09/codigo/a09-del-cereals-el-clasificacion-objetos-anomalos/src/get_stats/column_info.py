from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DESCRIPTIONS = {
    "sample_id": "Identificador de muestra (grupo temporal).",
    "timestamp": "Marca temporal de la observacion.",
    "co2_ppm": "Concentracion de CO2 en ppm.",
    "temp_c": "Temperatura en grados Celsius.",
    "ambient_rh_pct": "Humedad relativa ambiental en porcentaje.",
    "humidity_grain_pct": "Humedad del grano en porcentaje.",
    "target": "Etiqueta temporal por registro (0 sano, 1 insectos, 2 moho critico).",
    "target_global": "Clase base de la serie completa antes del etiquetado temporal v2.",
    "phase_name": "Nombre legible del estado temporal del registro.",
    "healthy_until_step": "Ultimo step del tramo inicial sano asignado a la muestra.",
    "transition_to_insect_end_step": "Step de cierre de la transicion hacia insectos.",
    "transition_to_moho_end_step": "Step de cierre de la transicion hacia moho critico.",
}


def build_column_info(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for col in df.columns:
        s = df[col]
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "null_pct": float(s.isna().mean()),
            "n_unique": int(s.nunique(dropna=True)),
            "description": DEFAULT_DESCRIPTIONS.get(col, ""),
        }

        if pd.api.types.is_numeric_dtype(s):
            row["min"] = float(s.min())
            row["max"] = float(s.max())
            row["mean"] = float(s.mean())
            row["std"] = float(s.std(ddof=0))
        else:
            row["min"] = None
            row["max"] = None
            row["mean"] = None
            row["std"] = None

        records.append(row)

    out = pd.DataFrame(records).sort_values("column").reset_index(drop=True)

    stats_path = Path(cfg["paths"]["stats_dataset"])
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(stats_path, index=False)
    return out
