"""Vendored verbatim (logic unchanged) from inbox/a45/codigo/src/data_processing/preprocess.py.

Only two things were adapted: the logger wiring (get_logger() -> logging.getLogger(__name__))
and the cross-module import, which now points at the vendored input_validation.py in this same
package instead of src.data_processing.input_validation.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from app.plugins.ml45_cereals_dnsl_critical_point_detection._vendor.input_validation import (
    prepare_model_input_dataframe,
    temporal_impute_partial_nulls,
    validate_model_input_data,
)

logger = logging.getLogger(__name__)


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


def _resolve_sequence_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    timestamp_column: str = "timestamp",
    id_column: Optional[str] = None,
) -> list[str]:
    """Devuelve columnas válidas de entrada, excluyendo target y metadatos de leakage."""
    excluded_cols = {
        target_column,
        timestamp_column,
        "fault_name",
        "fault_label",
        "grain_type",
        "phase",
    }
    if id_column is not None and id_column in df.columns:
        excluded_cols.add(id_column)

    return [c for c in df.columns if c not in excluded_cols]


def split_train_val_test_by_id(
    df: pd.DataFrame,
    id_col: str,
    target_column: str,
    val_size: float = 0.1,
    test_size: float = 0.2,
    normal_tokens: Optional[Sequence[str]] = None,
) -> tuple[dict[str, pd.DataFrame], Optional[np.ndarray]]:
    """Divide el dataset en train/val/test priorizando split por entidad."""
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
    target_column: str,
    seq_length: int = 50,
    solapamiento_beta: float = 0.5,
    id_column: Optional[str] = None,
    timestamp_column: str = "timestamp",
    normal_tokens: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[pd.Timestamp, pd.Timestamp]], list]:
    """Crea secuencias temporales [N, T, F] y etiquetas binarias [N].

    Args:
        df: DataFrame con los datos temporales.
        target_column: Nombre de la columna objetivo.
        seq_length: Longitud de cada secuencia temporal.
        solapamiento_beta: Control de solapamiento (0=sin solapamiento, 1=máximo).
        id_column: Columna de agrupación por entidad.
        normal_tokens: Tokens considerados como clase normal en etiquetas de texto.

    Returns:
         Returns:
        Tupla con (X_seq [N,T,F], y_seq [N], timestamp_windows_labels, entity_ids).
        - timestamp_windows_labels: lista de (timestamp_init, timestamp_end) por ventana.
        - entity_ids: lista de IDs de entidad por ventana.

    Raises:
        ValueError: Si la columna objetivo no existe o parámetros inválidos.
    """
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no existe en el DataFrame.")

    if seq_length < 1:
        raise ValueError("seq_length debe ser >= 1")

    entity_col = id_column if (id_column is not None and id_column in df.columns) else None

    if timestamp_column in df.columns:
        sort_cols = [entity_col, timestamp_column] if entity_col is not None else [timestamp_column]
        df_ord = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        df_ord = df.reset_index(drop=True)

    feature_cols = _resolve_sequence_feature_columns(
        df_ord,
        target_column=target_column,
        timestamp_column=timestamp_column,
        id_column=entity_col,
    )
    if len(feature_cols) == 0:
        raise ValueError("No hay columnas de entrada para construir secuencias.")

    X_seq: list[np.ndarray] = []
    y_seq: list = []
    timestamp_windows_labels: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    entity_ids: list = []
    # Asegurar que solapamiento_beta es float (puede venir como string desde YAML)
    solapamiento_beta = float(solapamiento_beta)
    step = max(1, int(seq_length * (1 - solapamiento_beta)))

    def _window_target_label(window_labels: np.ndarray) -> int:
        anomaly_mask = _anomaly_mask(window_labels, normal_tokens=normal_tokens)
        return int(np.any(anomaly_mask))

    if entity_col is not None:
        grouped = df_ord.groupby(entity_col, sort=False)
        for entity_id, g in grouped:
            if len(g) < seq_length:
                continue

            X_vals = g[feature_cols].to_numpy()
            y_vals = g[target_column].to_numpy()

            for i in range(0, len(g) - seq_length + 1, step):
                X_seq.append(X_vals[i:i + seq_length])
                window_labels = y_vals[i:i + seq_length]
                y_seq.append(_window_target_label(window_labels))
                if timestamp_column in g.columns:
                    timestamp_window = g[timestamp_column].iloc[i:i + seq_length]
                    timestamp_init = timestamp_window.iloc[0]
                    timestamp_end = timestamp_window.iloc[-1]
                    timestamp_windows_labels.append((timestamp_init, timestamp_end))

                entity_ids.append(entity_id)
    else:
        if len(df_ord) < seq_length:
            return (
                np.empty((0, seq_length, len(feature_cols))),
                np.empty((0,), dtype=df_ord[target_column].dtype),
                [],
                [],
            )

        X_vals = df_ord[feature_cols].to_numpy()
        y_vals = df_ord[target_column].to_numpy()

        for i in range(0, len(df_ord) - seq_length + 1, step):
            X_seq.append(X_vals[i:i + seq_length])
            window_labels = y_vals[i:i + seq_length]
            y_seq.append(_window_target_label(window_labels))
            if timestamp_column in df_ord.columns:
                timestamp_window = df_ord[timestamp_column].iloc[i:i + seq_length]
                timestamp_init = timestamp_window.iloc[0]
                timestamp_end = timestamp_window.iloc[-1]
                timestamp_windows_labels.append((timestamp_init, timestamp_end))

    if len(X_seq) == 0:
        return (
            np.empty((0, seq_length, len(feature_cols))),
            np.empty((0,), dtype=df_ord[target_column].dtype),
            [],
            [],
        )

    return np.asarray(X_seq), np.asarray(y_seq, dtype=np.int64), timestamp_windows_labels, entity_ids


def stats_windows(
    x_seq: np.ndarray,
    feature_names: Optional[list[str]] = None,
    stats_creation: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Calcula estadísticas por ventana para secuencias temporales.

    Args:
        x_seq: Tensor/array 3D con forma [N, T, F].
        feature_names: Nombres de variables (longitud F).
        stats_creation: Lista de estadísticas a calcular por ventana.

    Returns:
        DataFrame [N, len(stats)*F] con columnas tipo {stat}_{feature}.

    Raises:
        ValueError: Si x_seq no es 3D o estadísticas no soportadas.
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
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names debe tener longitud {n_features}. "
            f"Recibido: {len(feature_names)}"
        )

    if stats_creation is None:
        stats_creation = ['mean', 'std']

    if isinstance(stats_creation, str):
        stats_creation = [stats_creation]

    stats_creation = [str(s).strip().lower() for s in stats_creation if str(s).strip()]
    if len(stats_creation) == 0:
        raise ValueError("stats_creation no puede estar vacío.")

    seen: set[str] = set()
    stats_ordered: list[str] = []
    for stat in stats_creation:
        if stat not in seen:
            stats_ordered.append(stat)
            seen.add(stat)

    stats_funcs = {
        'max': lambda x: np.max(x, axis=1),
        'min': lambda x: np.min(x, axis=1),
        'mean': lambda x: np.mean(x, axis=1),
        'std': lambda x: np.std(x, axis=1, ddof=0),
        'slope': lambda x: (x[:, -1, :] - x[:, 0, :]) / max(seq_len - 1, 1),
        'diff_mean': lambda x: np.diff(x, axis=1).mean(axis=1),
        'diff_std': lambda x: np.diff(x, axis=1).std(axis=1, ddof=0),
        'range': lambda x: np.max(x, axis=1) - np.min(x, axis=1),
    }

    unsupported = [s for s in stats_ordered if s not in stats_funcs]
    if unsupported:
        raise ValueError(
            f"Estadísticas no soportadas: {unsupported}. "
            f"Soportadas: {sorted(stats_funcs.keys())}"
        )

    out: dict[str, np.ndarray] = {}
    for stat in stats_ordered:
        stat_vals = stats_funcs[stat](x_np)
        for j, feat in enumerate(feature_names):
            out[f"{stat}_{feat}"] = stat_vals[:, j]

    return pd.DataFrame(out)


def run_data_processing_pipeline(
    data_cfg: dict,
    df: Optional[pd.DataFrame] = None,
    pre_split: Optional[dict[str, pd.DataFrame]] = None,
    config: Optional[dict] = None,
) -> tuple[dict, dict[str, pd.DataFrame], StandardScaler, StandardScaler]:
    """Pipeline completo de preparación de datos para modelos Deep Neuro-Fuzzy.

    Args:
        df: DataFrame con los datos crudos (opcional).
        data_cfg: Diccionario de configuración de datos.
        pre_split: Dict con DataFrames pre-particionados {"train": df, "val": df, "test": df}.
            Si se provee, se usa directamente en lugar de re-partir df.
        config: Configuración completa del proyecto (opcional), usada para
            generación automática de splits cuando no hay datos de entrada.

    Returns:
        Tupla con (split_data, split_data_df_stats, scaler_x, scaler_num).
    """
    target_column = data_cfg['target_column'].lower()
    id_column = data_cfg.get('id_column')
    id_column = id_column.lower() if id_column else None
    timestamp_column = data_cfg.get('timestamp_column', 'timestamp')
    normal_tokens = data_cfg.get('normal_tokens')
    if normal_tokens is None:
        raise ValueError(
            "Falta 'data_processing.normal_tokens' en la configuración. "
            "Define la lista de tokens de normalidad en config.yaml."
        )
    solapamiento_beta = float(data_cfg.get('solapamiento_beta', 0.5))
    seq_length = int(data_cfg['sequence_length'])
    fuzzy_processing_cfg = data_cfg['fuzzy_processing']

    split_df: dict[str, pd.DataFrame]

    if pre_split is not None:
        split_df = {}
        for split_name, split_frame in pre_split.items():
            split_frame.columns = split_frame.columns.str.lower()
            if id_column and id_column in split_frame.columns and timestamp_column in split_frame.columns:
                split_frame = split_frame.sort_values(by=[id_column, timestamp_column]).reset_index(drop=True)
            elif timestamp_column in split_frame.columns:
                split_frame = split_frame.sort_values(by=timestamp_column).reset_index(drop=True)
            split_df[split_name] = split_frame
        logger.info("Usando splits pre-particionados: %s", list(split_df.keys()))

    elif df is not None:
        df.columns = df.columns.str.lower()

        if id_column and id_column in df.columns and timestamp_column in df.columns:
            df = df.sort_values(by=[id_column, timestamp_column]).reset_index(drop=True)
        elif timestamp_column in df.columns:
            df = df.sort_values(by=timestamp_column).reset_index(drop=True)

        split_df, _ = split_train_val_test_by_id(
            df,
            id_col=id_column if id_column else 'cycle_id',
            target_column=target_column,
            val_size=data_cfg['val_size'],
            test_size=data_cfg['test_size'],
            normal_tokens=normal_tokens,
        )

    else:
        # Prioriza lógica de entrada estilo notebook:
        # 1) Directorio con train/val/test
        # 2) Archivo único para split temporal
        # 3) Generación automática de splits
        split_filenames = data_cfg.get(
            'split_filenames',
            {'train': 'train.csv', 'val': 'val.csv', 'test': 'test.csv'},
        )

        if config is not None:
            default_data_path = config.get('paths', {}).get('splits', 'data/splits')
        else:
            default_data_path = 'data/splits'

        data_path = data_cfg.get('data_path', default_data_path)

        if os.path.isdir(data_path) and all(
            os.path.isfile(os.path.join(data_path, split_filenames[k]))
            for k in ('train', 'val', 'test')
        ):
            logger.info("Cargando datos desde archivos CSV en %s", data_path)
            split_df = {}
            for split_name, filename in split_filenames.items():
                split_path = os.path.join(data_path, filename)
                split_frame = pd.read_csv(split_path)
                split_frame.columns = split_frame.columns.str.lower()
                if id_column and id_column in split_frame.columns and timestamp_column in split_frame.columns:
                    split_frame = split_frame.sort_values(by=[id_column, timestamp_column]).reset_index(drop=True)
                elif timestamp_column in split_frame.columns:
                    split_frame = split_frame.sort_values(by=timestamp_column).reset_index(drop=True)
                split_df[split_name] = split_frame
                logger.info("  %s: %d muestras", split_name.capitalize(), len(split_frame))

        elif os.path.isfile(data_path):
            logger.info("Datos cargados desde %s", data_path)
            df = pd.read_csv(data_path)
            df.columns = df.columns.str.lower()

            if id_column and id_column in df.columns and timestamp_column in df.columns:
                df = df.sort_values(by=[id_column, timestamp_column]).reset_index(drop=True)
            elif timestamp_column in df.columns:
                df = df.sort_values(by=timestamp_column).reset_index(drop=True)

            split_df, _ = split_train_val_test_by_id(
                df,
                id_col=id_column if id_column else 'cycle_id',
                target_column=target_column,
                val_size=data_cfg['val_size'],
                test_size=data_cfg['test_size'],
                normal_tokens=normal_tokens,
            )

        else:
            raise FileNotFoundError(
                "No se encontraron datos de entrada para generar splits "
                f"(data_path={data_path!r}). El plugin no regenera el dataset "
                "sintético — provee un CSV o directorio con train/val/test."
            )

    split_names = list(split_df.keys())

    for split_name in split_names:
        validation_report = validate_model_input_data(
            df=split_df[split_name],
            config=config,
            context=f"entrenamiento/{split_name}",
            require_target=True,
        )
        split_df[split_name] = prepare_model_input_dataframe(
            df=split_df[split_name],
            config=config,
            context=f"entrenamiento/{split_name}",
        )
        split_df[split_name] = temporal_impute_partial_nulls(
            df=split_df[split_name],
            partial_null_stats=validation_report.get("partial_null_stats", {}),
            id_column=id_column,
            timestamp_column=timestamp_column,
        )

    # Creación de secuencias temporales
    X_seqs: dict[str, np.ndarray] = {}
    y_seqs: dict[str, np.ndarray] = {}
    for split in split_names:
        X_seqs[split], y_seqs[split], _, _ = create_sequences(
            split_df[split],
            target_column=target_column,
            seq_length=seq_length,
            solapamiento_beta=solapamiento_beta,
            timestamp_column=timestamp_column,
            id_column=id_column,
            normal_tokens=normal_tokens,
        )

        logger.info(
            "Split '%s' procesado: %s ventanas temporales creadas, con %d registros cada una y un solapamiento de %s (%d registros).",
            split,
            X_seqs[split].shape[0],
            X_seqs[split].shape[1],
            solapamiento_beta,
            int(X_seqs[split].shape[1] * (1 - solapamiento_beta)),
        )

    # Cálculo de estadísticas por ventana
    df_stats: dict[str, pd.DataFrame] = {}
    for split in split_names:
        feature_cols = _resolve_sequence_feature_columns(
            split_df[split],
            target_column=target_column,
            timestamp_column=timestamp_column,
            id_column=id_column,
        )
        df_stats[split] = stats_windows(
            X_seqs[split],
            feature_names=feature_cols,
            stats_creation=fuzzy_processing_cfg.get('stats_creation'),
        )

    # Escalado de estadísticas
    scaler_num = StandardScaler()
    df_stats_scaled: dict[str, pd.DataFrame] = {}
    feature_names = list(df_stats['train'].columns)

    df_stats_scaled['train'] = pd.DataFrame(
        scaler_num.fit_transform(df_stats['train']),
        columns=feature_names,
    )
    for split in ['val', 'test']:
        df_stats_scaled[split] = pd.DataFrame(
            scaler_num.transform(df_stats[split]),
            columns=feature_names,
        )

    # Escalado de secuencias X
    scaler_x = StandardScaler()
    X_scaled: dict[str, np.ndarray] = {}

    n_features = X_seqs['train'].shape[-1]
    X_scaled['train'] = scaler_x.fit_transform(
        X_seqs['train'].reshape(-1, n_features)
    ).reshape(X_seqs['train'].shape)

    for split in ['val', 'test']:
        X_scaled[split] = scaler_x.transform(
            X_seqs[split].reshape(-1, n_features)
        ).reshape(X_seqs[split].shape)

    split_data = {
        'X': {split: X_scaled[split] for split in split_names},
        'y': {split: y_seqs[split] for split in split_names},
    }

    return split_data, df_stats_scaled, scaler_x, scaler_num
