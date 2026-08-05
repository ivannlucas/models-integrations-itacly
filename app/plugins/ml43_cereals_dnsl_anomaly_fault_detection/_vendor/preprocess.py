"""Preprocesamiento de datos: secuencias temporales y estadísticas.

Copied from a43-44-neurofuzzy-anomalias-fallas/src/data_processing/preprocess.py
and adapted to remove the project-specific logger import.
"""

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import logging

logger = logging.getLogger(__name__)

SENSOR_COLUMNS = [
    "temp_zona1", "temp_zona2", "temp_zona3", "temp_salida_gases",
    "presion_camara", "presion_ventilacion", "potencia_kw", "flujo_gas",
    "humedad_relativa", "temp_ambiente", "setpoint_temp",
    "posicion_valvula", "velocidad_ventilador",
]

STATS_CREATION = ["mean", "std", "slope", "max", "min"]


def _anomaly_mask(
    values: Sequence[Any],
    normal_tokens: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Convierte etiquetas heterogéneas a máscara booleana de anomalía."""
    series = pd.Series(values)

    numeric_values = pd.to_numeric(series, errors="coerce")
    if numeric_values.notna().all():
        return numeric_values.astype(float).to_numpy() > 0

    text_values = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": "", "none": ""})
    )
    tokens = [] if normal_tokens is None else [str(t).strip().lower() for t in normal_tokens]
    normal_mask = text_values.isin(tokens) | (text_values == "")
    return (~normal_mask).to_numpy()


def create_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_length: int = 180,
    solapamiento_beta: float = 0.5,
    id_column: Optional[str] = "cycle_id",
    timestamp_column: str = "timestamp",
    target_column: Optional[str] = None,
    normal_tokens: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[list]]:
    """Crea secuencias temporales [N, T, F] y etiquetas binarias [N].

    Una ventana se etiqueta como anómala si al menos el 50% de sus filas
    presentan anomalía (estrategia de mayoría), consistente con
    a43-44-neurofuzzy-anomalias-fallas/src/data_processing/preprocess.py.

    Returns:
        Tupla con (X_seq [N,T,F], y_seq [N] or None, cycle_ids list or None).
    """
    # Sort by id + timestamp if available
    sort_cols = []
    if id_column and id_column in df.columns:
        sort_cols.append(id_column)
    if timestamp_column and timestamp_column in df.columns:
        sort_cols.append(timestamp_column)
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    solapamiento_beta = float(solapamiento_beta)
    step = max(1, int(seq_length * (1 - solapamiento_beta)))

    X_seq: list[np.ndarray] = []
    y_seq: list = []
    cycle_ids: list = []

    def _window_label(window_labels: np.ndarray) -> int:
        anomaly_mask = _anomaly_mask(window_labels, normal_tokens=normal_tokens)
        return int(np.mean(anomaly_mask) >= 0.5)

    entity_col = id_column if (id_column and id_column in df.columns) else None

    if entity_col is not None:
        grouped = df.groupby(entity_col, sort=False)
        for gid, g in grouped:
            if len(g) < seq_length:
                continue
            X_vals = g[feature_cols].to_numpy(dtype=np.float32)
            y_vals = g[target_column].to_numpy() if target_column else None
            for i in range(0, len(g) - seq_length + 1, step):
                X_seq.append(X_vals[i:i + seq_length])
                cycle_ids.append(str(gid))
                if y_vals is not None:
                    y_seq.append(_window_label(y_vals[i:i + seq_length]))
    else:
        if len(df) < seq_length:
            n_feat = len(feature_cols)
            return (
                np.empty((0, seq_length, n_feat), dtype=np.float32),
                None,
                [],
            )
        X_vals = df[feature_cols].to_numpy(dtype=np.float32)
        y_vals = df[target_column].to_numpy() if target_column else None
        for i in range(0, len(df) - seq_length + 1, step):
            X_seq.append(X_vals[i:i + seq_length])
            cycle_ids.append(None)
            if y_vals is not None:
                y_seq.append(_window_label(y_vals[i:i + seq_length]))

    if not X_seq:
        n_feat = len(feature_cols)
        return (
            np.empty((0, seq_length, n_feat), dtype=np.float32),
            None,
            [],
        )

    X_arr = np.asarray(X_seq, dtype=np.float32)
    y_arr = np.asarray(y_seq, dtype=np.int64) if y_seq else None
    return X_arr, y_arr, cycle_ids


def stats_windows(
    x_seq: np.ndarray,
    feature_names: Optional[list[str]] = None,
    stats_creation: Optional[list[str]] = None,
    ddof: int = 0,
) -> pd.DataFrame:
    """Calcula estadísticas por ventana para secuencias temporales.

    Args:
        x_seq: Array 3D con forma [N, T, F].
        feature_names: Nombres de variables (longitud F).
        stats_creation: Lista de estadísticas a calcular por ventana.
        ddof: Delta degrees of freedom para std.

    Returns:
        DataFrame [N, len(stats)*F] con columnas tipo {stat}_{feature}.
    """
    if isinstance(x_seq, torch.Tensor):
        x_np = x_seq.detach().cpu().numpy()
    else:
        x_np = np.asarray(x_seq)

    if x_np.ndim != 3:
        raise ValueError(f"x_seq debe ser 3D [N,T,F]. Recibido: {x_np.shape}")

    _, seq_len, n_features = x_np.shape

    if feature_names is None:
        feature_names = [f"var_{i}" for i in range(n_features)]

    if stats_creation is None:
        stats_creation = ["mean", "std"]

    stats_creation = [str(s).strip().lower() for s in stats_creation if str(s).strip()]

    stats_funcs = {
        "max": lambda x: np.max(x, axis=1),
        "min": lambda x: np.min(x, axis=1),
        "mean": lambda x: np.mean(x, axis=1),
        "std": lambda x: np.std(x, axis=1, ddof=ddof),
        "slope": lambda x: (x[:, -1, :] - x[:, 0, :]) / max(seq_len - 1, 1),
        "diff": lambda x: x[:, -1, :] - x[:, 0, :],
        "range": lambda x: np.max(x, axis=1) - np.min(x, axis=1),
        "median": lambda x: np.median(x, axis=1),
        "p25": lambda x: np.quantile(x, 0.25, axis=1),
        "p75": lambda x: np.quantile(x, 0.75, axis=1),
    }

    out: dict[str, np.ndarray] = {}
    for stat in stats_creation:
        if stat not in stats_funcs:
            raise ValueError(f"Estadística no soportada: {stat}. Soportadas: {sorted(stats_funcs)}")
        stat_vals = stats_funcs[stat](x_np)
        for j, feat in enumerate(feature_names):
            out[f"{stat}_{feat}"] = stat_vals[:, j]

    return pd.DataFrame(out)
