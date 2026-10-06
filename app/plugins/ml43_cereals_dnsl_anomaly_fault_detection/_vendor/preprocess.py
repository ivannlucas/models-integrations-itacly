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


def _failure_rate(
    values: Sequence[Any],
    normal_tokens: Optional[Sequence[str]] = None,
) -> float:
    """Calcula el porcentaje de anomalías en un vector de etiquetas."""
    anomaly_mask = _anomaly_mask(values, normal_tokens=normal_tokens)
    return float(anomaly_mask.mean() * 100.0)


def split_train_val_test_by_id(
    df: pd.DataFrame,
    id_col: str,
    target_column: str,
    val_size: float = 0.1,
    test_size: float = 0.2,
    normal_tokens: Optional[Sequence[str]] = None,
) -> tuple[dict[str, pd.DataFrame], Optional[np.ndarray]]:
    """Divide el dataset en train/val/test priorizando split por entidad (cycle_id).

    Copied verbatim from a43-44-neurofuzzy-anomalias-fallas/src/data_processing/
    preprocess.py — the "ruta 2" (external dataset) split used by
    scripts/split_external_data.py, which the product docs mark as the one the
    platform must implement for retraining. Splitting whole cycles (not raw rows or
    windows) into train/val/test keeps every window's SEQ_LENGTH-row context inside a
    single split — a window is never assembled from rows that straddle two splits.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"La columna objetivo '{target_column}' no existe en el DataFrame."
        )

    if not (0 < val_size < 1):
        raise ValueError("val_size debe estar entre 0 y 1 (exclusivo).")
    if not (0 < test_size < 1):
        raise ValueError("test_size debe estar entre 0 y 1 (exclusivo).")
    if val_size + test_size >= 1:
        raise ValueError("La suma de val_size y test_size debe ser menor que 1.")

    unique_ids = None
    df_train = df_val = df_test = None

    if id_col in df.columns:
        unique_ids = df[id_col].unique()
        n_machines = len(unique_ids)

        if n_machines >= 3:
            train_end = max(1, int(n_machines * (1 - test_size - val_size)))
            val_end = max(train_end + 1, int(n_machines * (1 - test_size)))
            val_end = min(val_end, n_machines - 1)

            train_machines = unique_ids[:train_end]
            val_machines = unique_ids[train_end:val_end]
            test_machines = unique_ids[val_end:]

            df_train = df[df[id_col].isin(train_machines)]
            df_val = df[df[id_col].isin(val_machines)]
            df_test = df[df[id_col].isin(test_machines)]

            if len(df_train) > 0 and len(df_val) > 0 and len(df_test) > 0:
                logger.info("[Split por IDs]")
                logger.info(
                    "  Train IDs: %s - %s | samples=%d",
                    train_machines[0], train_machines[-1], len(df_train),
                )
                logger.info(
                    "  Val IDs:   %s - %s | samples=%d",
                    val_machines[0], val_machines[-1], len(df_val),
                )
                logger.info(
                    "  Test IDs:  %s - %s | samples=%d",
                    test_machines[0], test_machines[-1], len(df_test),
                )
            else:
                logger.warning(
                    "Split por IDs produjo subconjuntos vacíos. "
                    "Se recomienda usar split temporal."
                )
                df_train = df_val = df_test = None
        else:
            logger.warning(
                "Solo se detectaron %d IDs. Se recomienda usar split temporal.",
                n_machines,
            )

    if id_col not in df.columns or df_train is None or df_val is None or df_test is None:
        n_total = len(df)
        if n_total < 3:
            raise ValueError("No hay suficientes muestras para crear train/val/test.")

        train_end = max(1, int(n_total * (1 - test_size - val_size)))
        val_end = max(train_end + 1, int(n_total * (1 - test_size)))
        val_end = min(val_end, n_total - 1)

        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
        df_test = df.iloc[val_end:]

        if len(df_train) == 0 or len(df_val) == 0 or len(df_test) == 0:
            raise ValueError("Split temporal vacío detectado. Verifica tamaño del dataset.")

        logger.info(
            "[Split temporal %.1f/%.1f/%.1f]",
            1 - test_size - val_size, val_size, test_size,
        )
        logger.info(
            "  Train samples: %d | fallas=%.2f%%",
            len(df_train), _failure_rate(df_train[target_column], normal_tokens=normal_tokens),
        )
        logger.info(
            "  Val samples:   %d | fallas=%.2f%%",
            len(df_val), _failure_rate(df_val[target_column], normal_tokens=normal_tokens),
        )
        logger.info(
            "  Test samples:  %d | fallas=%.2f%%",
            len(df_test), _failure_rate(df_test[target_column], normal_tokens=normal_tokens),
        )
    else:
        logger.info(
            "  Fallas train/val/test: %.2f%% / %.2f%% / %.2f%%",
            _failure_rate(df_train[target_column], normal_tokens=normal_tokens),
            _failure_rate(df_val[target_column], normal_tokens=normal_tokens),
            _failure_rate(df_test[target_column], normal_tokens=normal_tokens),
        )

    split_data = {
        "train": df_train,
        "val": df_val,
        "test": df_test,
    }
    return split_data, unique_ids


def create_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_length: int = 180,
    solapamiento_beta: float = 0.5,
    id_column: Optional[str] = "cycle_id",
    timestamp_column: str = "timestamp",
    target_column: Optional[str] = None,
    normal_tokens: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[list], Optional[list]]:
    """Crea secuencias temporales [N, T, F] y etiquetas binarias [N].

    Una ventana se etiqueta como anómala si al menos el 50% de sus filas
    presentan anomalía (estrategia de mayoría), consistente con
    a43-44-neurofuzzy-anomalias-fallas/src/data_processing/preprocess.py.

    Returns:
        Tupla con (X_seq [N,T,F], y_seq [N] or None, cycle_ids list or None,
        window_timestamps list of (timestamp_init, timestamp_end) per window,
        or None if no timestamp_column is available).
    """
    # Sort by id + timestamp if available
    sort_cols = []
    if id_column and id_column in df.columns:
        sort_cols.append(id_column)
    has_ts = bool(timestamp_column and timestamp_column in df.columns)
    if has_ts:
        sort_cols.append(timestamp_column)
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    solapamiento_beta = float(solapamiento_beta)
    step = max(1, int(seq_length * (1 - solapamiento_beta)))

    X_seq: list[np.ndarray] = []
    y_seq: list = []
    cycle_ids: list = []
    window_timestamps: list = []

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
            ts_vals = g[timestamp_column].to_numpy() if has_ts else None
            for i in range(0, len(g) - seq_length + 1, step):
                X_seq.append(X_vals[i:i + seq_length])
                cycle_ids.append(str(gid))
                if ts_vals is not None:
                    window_timestamps.append((ts_vals[i], ts_vals[i + seq_length - 1]))
                if y_vals is not None:
                    y_seq.append(_window_label(y_vals[i:i + seq_length]))
    else:
        if len(df) < seq_length:
            n_feat = len(feature_cols)
            return (
                np.empty((0, seq_length, n_feat), dtype=np.float32),
                None,
                [],
                [] if has_ts else None,
            )
        X_vals = df[feature_cols].to_numpy(dtype=np.float32)
        y_vals = df[target_column].to_numpy() if target_column else None
        ts_vals = df[timestamp_column].to_numpy() if has_ts else None
        for i in range(0, len(df) - seq_length + 1, step):
            X_seq.append(X_vals[i:i + seq_length])
            cycle_ids.append(None)
            if ts_vals is not None:
                window_timestamps.append((ts_vals[i], ts_vals[i + seq_length - 1]))
            if y_vals is not None:
                y_seq.append(_window_label(y_vals[i:i + seq_length]))

    if not X_seq:
        n_feat = len(feature_cols)
        return (
            np.empty((0, seq_length, n_feat), dtype=np.float32),
            None,
            [],
            [] if has_ts else None,
        )

    X_arr = np.asarray(X_seq, dtype=np.float32)
    y_arr = np.asarray(y_seq, dtype=np.int64) if y_seq else None
    return X_arr, y_arr, cycle_ids, (window_timestamps if has_ts else None)


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
