"""VENDORED from the a09 delivery — src/data_processing/preprocess.py.

Copied verbatim from inbox/a09/codigo/a09-del-cereals-el-clasificacion-objetos-anomalos/ so that
feature engineering and window construction are bit-for-bit identical to the AI team's pipeline.
Do NOT refactor: any change here breaks numerical reproducibility against manifest golden_cases.

Deliberate deviations from the original file (only these):
  1. `_ensure_dirs()` and `prepare_processed_datasets()` removed — they mkdir/write CSVs from
     cfg["paths"], which the plugin never uses (inference is in-memory).
  2. The mojibake in the split_target_column error message ("no vÃ¡lidos") fixed to proper UTF-8.
  3. Whitespace normalized for flake8 (blank lines between defs, no space before ':' in slices).
     Cosmetic only — verified afterwards against the manifest golden_cases.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_MIN_COLUMNS = {
    "sample_id",
    "timestamp",
    "co2_ppm",
    "temp_c",
    "ambient_rh_pct",
    "humidity_grain_pct",
    "target",
}


def validate_and_normalize_input_frame(
    df_input: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    require_target: bool,
) -> pd.DataFrame:
    group_col = cfg["data"]["group_column"]
    ts_col = cfg["data"]["timestamp_column"]
    sensor_columns = list(cfg["data"]["sensor_columns"])
    target_col = cfg["data"]["target_column"]
    split_target_col = str(cfg["data"].get("split_target_column", target_col))

    required_columns = [group_col, ts_col, *sensor_columns]
    if require_target:
        required_columns.append(target_col)

    missing = sorted(set(required_columns) - set(df_input.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas en entrada: {missing}")

    df = df_input.copy()

    null_columns = [col for col in required_columns if df[col].isna().any()]
    if null_columns:
        raise ValueError(
            "La entrada contiene valores nulos en columnas obligatorias: "
            f"{null_columns}"
        )

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if df[ts_col].isna().any():
        raise ValueError("Hay timestamps inválidos en la entrada.")

    invalid_numeric_columns: list[str] = []
    for col in sensor_columns:
        numeric_values = pd.to_numeric(df[col], errors="coerce")
        if numeric_values.isna().any():
            invalid_numeric_columns.append(col)
            continue
        df[col] = numeric_values.astype(float)

    if invalid_numeric_columns:
        raise ValueError(
            "Las columnas de sensorización deben ser numéricas y no contener valores inválidos. "
            f"Columnas afectadas: {invalid_numeric_columns}"
        )

    if require_target:
        target_values = pd.to_numeric(df[target_col], errors="coerce")
        if target_values.isna().any():
            raise ValueError("La columna target contiene valores no válidos.")
        df[target_col] = target_values.astype(int)

    if require_target and split_target_col in df.columns:
        split_target_values = pd.to_numeric(df[split_target_col], errors="coerce")
        if split_target_values.isna().any():
            raise ValueError(f"La columna {split_target_col} contiene valores no válidos.")
        df[split_target_col] = split_target_values.astype(int)

    return df


def _hours_to_periods(hours: int, step_minutes: int) -> tuple[int, float]:
    periods = max(1, int(round(hours * 60 / step_minutes)))
    actual_hours = periods * step_minutes / 60.0
    return periods, actual_hours


def build_features(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    group_col = cfg["data"]["group_column"]
    ts_col = cfg["data"]["timestamp_column"]
    sensors = list(cfg["data"]["sensor_columns"])
    step_minutes = int(cfg.get("synthetic_dataset", {}).get("step_minutes", 60))
    periods_24, hours_24 = _hours_to_periods(24, step_minutes)
    periods_48, hours_48 = _hours_to_periods(48, step_minutes)

    g = df.groupby(group_col, sort=False)
    feat = pd.DataFrame(index=df.index)

    for c in sensors:
        feat[c] = df[c].astype(float)
        feat[f"{c}_diff_1"] = g[c].diff(1).fillna(0.0)
        feat[f"{c}_diff_3"] = g[c].diff(3).fillna(0.0)
        feat[f"{c}_diff_6"] = g[c].diff(6).fillna(0.0)
        feat[f"{c}_diff_24"] = g[c].diff(periods_24).fillna(0.0)
        feat[f"{c}_slope_24h"] = g[c].diff(periods_24).fillna(0.0) / hours_24
        feat[f"{c}_slope_48h"] = g[c].diff(periods_48).fillna(0.0) / hours_48
        for w in [3, 6, 12, 24]:
            feat[f"{c}_roll_mean_{w}"] = (
                g[c].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
            )
        feat[f"{c}_roll_std_6"] = (
            g[c].rolling(window=6, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0)
        )
        feat[f"{c}_roll_std_24"] = (
            g[c].rolling(window=24, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0)
        )

    feat["co2_acceleration_24h"] = feat["co2_ppm_slope_24h"] - feat["co2_ppm_slope_48h"]
    feat["temp_acceleration_24h"] = feat["temp_c_slope_24h"] - feat["temp_c_slope_48h"]
    feat["co2_minus_roll6"] = feat["co2_ppm"] - feat["co2_ppm_roll_mean_6"]
    feat["co2_per_kelvin"] = feat["co2_ppm"] / (feat["temp_c"] + 273.15)
    feat["temp_x_rh"] = feat["temp_c"] * (feat["ambient_rh_pct"] / 100.0)
    feat["co2_x_temp"] = feat["co2_ppm"] * feat["temp_c"]
    feat["co2_x_grain_hum"] = feat["co2_ppm"] * feat["humidity_grain_pct"]
    feat["co2_rise_ratio"] = feat["co2_ppm_diff_6"] / (np.abs(feat["co2_ppm_roll_mean_6"]) + 1.0)
    feat["temp_rise_ratio"] = feat["temp_c_diff_6"] / (np.abs(feat["temp_c_roll_mean_6"]) + 0.5)

    hour = df[ts_col].dt.hour.astype(float)
    dow = df[ts_col].dt.dayofweek.astype(float)
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return feat, feat.columns.tolist()


def build_sequence_feature_frame(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    require_target: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Construye la tabla temporal con features listas para ventana secuencial."""
    group_col = cfg['data']['group_column']
    ts_col = cfg['data']['timestamp_column']
    target_col = cfg['data']['target_column']

    df_norm = validate_and_normalize_input_frame(df, cfg, require_target=require_target)
    df_norm = df_norm.sort_values([group_col, ts_col]).reset_index(drop=True)
    features, feature_columns = build_features(df_norm, cfg)

    feature_cfg = cfg.get('sequence', {}).get('feature_columns', 'auto')
    if isinstance(feature_cfg, list) and feature_cfg:
        missing = [col for col in feature_cfg if col not in features.columns]
        if missing:
            raise ValueError(f'Las features pedidas en sequence.feature_columns no existen: {missing}')
        features = features.loc[:, feature_cfg].copy()
        feature_columns = list(feature_cfg)

    cols = [group_col, ts_col] + ([target_col] if require_target and target_col in df_norm.columns else [])
    seq_df = pd.concat(
        [
            df_norm[cols].reset_index(drop=True),
            features.reset_index(drop=True),
        ],
        axis=1,
    )
    return seq_df, feature_columns


def split_sample_ids_stratified(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    validation_frac: float | None = None,
    holdout_frac: float | None = None,
    random_state: int | None = None,
) -> dict[str, np.ndarray]:
    """Divide sample_id completos en train/validation/test sin mezclar ventanas."""
    group_col = cfg['data']['group_column']
    target_col = cfg['data']['target_column']
    split_target_col = str(cfg['data'].get('split_target_column', target_col))
    if split_target_col not in df.columns:
        split_target_col = target_col
    train_cfg = cfg.get('training', {})
    seq_cfg = cfg.get('sequence', {})

    holdout_frac = float(holdout_frac if holdout_frac is not None else train_cfg.get('holdout_frac', 0.2))
    validation_frac = float(
        validation_frac if validation_frac is not None else seq_cfg.get('validation_frac', 0.15)
    )
    random_state = int(random_state if random_state is not None else train_cfg.get('random_state', 42))

    sample_meta = (
        df.groupby(group_col, sort=False)[split_target_col]
        .first()
        .reset_index()
        .rename(columns={split_target_col: 'sample_target'})
    )
    rng = np.random.default_rng(random_state)

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for _, cls_df in sample_meta.groupby('sample_target', sort=True):
        ids = cls_df[group_col].to_numpy().copy()
        if len(ids) < 3:
            raise ValueError('Cada clase necesita al menos 3 sample_id para crear train/validation/test sin fuga.')
        rng.shuffle(ids)

        n_test = max(1, int(round(len(ids) * holdout_frac)))
        n_val = max(1, int(round(len(ids) * validation_frac)))
        if n_test + n_val >= len(ids):
            n_val = max(1, len(ids) - n_test - 1)
        if n_test + n_val >= len(ids):
            raise ValueError('No se puede dividir la clase con los porcentajes actuales.')

        test_ids.extend(ids[:n_test].tolist())
        val_ids.extend(ids[n_test:n_test + n_val].tolist())
        train_ids.extend(ids[n_test + n_val:].tolist())

    return {
        'train_sample_ids': np.asarray(train_ids),
        'validation_sample_ids': np.asarray(val_ids),
        'test_sample_ids': np.asarray(test_ids),
    }


def build_sliding_windows(
    df_seq: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    feature_columns: list[str] | None = None,
    sample_ids: np.ndarray | list[str] | None = None,
    require_target: bool = True,
) -> dict[str, Any]:
    """Convierte una tabla temporal ordenada en ventanas para LSTM/GRU."""
    group_col = cfg['data']['group_column']
    ts_col = cfg['data']['timestamp_column']
    target_col = cfg['data']['target_column']
    seq_cfg = cfg.get('sequence', {})

    if feature_columns is None:
        ignore_cols = {group_col, ts_col}
        if target_col in df_seq.columns:
            ignore_cols.add(target_col)
        feature_columns = [col for col in df_seq.columns if col not in ignore_cols]
    else:
        feature_columns = list(feature_columns)

    window_size = int(seq_cfg.get('window_size', 48))
    stride = max(1, int(seq_cfg.get('stride', 12)))
    label_mode = str(seq_cfg.get('label_mode', 'last')).lower().strip()
    pad_short_sequences = bool(seq_cfg.get('pad_short_sequences', False))

    df_work = df_seq.copy()
    if sample_ids is not None:
        sample_ids_arr = np.asarray(sample_ids)
        df_work = df_work[df_work[group_col].isin(sample_ids_arr)].copy()

    df_work = df_work.sort_values([group_col, ts_col]).reset_index(drop=True)

    x_windows: list[np.ndarray] = []
    y_windows: list[int] = []
    group_windows: list[Any] = []
    timestamp_windows: list[pd.Timestamp] = []
    metadata_rows: list[dict[str, Any]] = []

    for sample_id, group_df in df_work.groupby(group_col, sort=False):
        group_df = group_df.sort_values(ts_col).reset_index(drop=True)
        features = group_df.loc[:, feature_columns].to_numpy(dtype=np.float32)
        labels = group_df[target_col].to_numpy(dtype=int) if require_target and target_col in group_df.columns else None
        timestamps = pd.to_datetime(group_df[ts_col], errors='coerce').reset_index(drop=True)

        if len(group_df) < window_size:
            if not pad_short_sequences:
                continue
            pad_len = window_size - len(group_df)
            window = np.zeros((window_size, len(feature_columns)), dtype=np.float32)
            window[pad_len:] = features
            label_slice = labels
            ts_slice = timestamps
            start_positions = [0]
        else:
            start_positions = list(range(0, len(group_df) - window_size + 1, stride))
            if not start_positions:
                start_positions = [0]

        for window_idx, start in enumerate(start_positions):
            end = min(start + window_size, len(group_df))
            if len(group_df) < window_size:
                current_window = window
                label_slice = labels
                ts_slice = timestamps
            else:
                if end - start < window_size:
                    if not pad_short_sequences:
                        continue
                    current_window = np.zeros((window_size, len(feature_columns)), dtype=np.float32)
                    current_window[-(end - start):] = features[start:end]
                else:
                    current_window = features[start:end]
                label_slice = labels[start:end] if labels is not None else None
                ts_slice = timestamps.iloc[start:end]

            if require_target:
                if label_mode == 'last':
                    label = int(label_slice[-1])
                elif label_mode == 'first':
                    label = int(label_slice[0])
                elif label_mode == 'majority':
                    values, counts = np.unique(label_slice, return_counts=True)
                    label = int(values[np.argmax(counts)])
                else:
                    raise ValueError('sequence.label_mode debe ser one of: last, first, majority')
            else:
                label = -1

            x_windows.append(np.asarray(current_window, dtype=np.float32))
            if require_target:
                y_windows.append(label)
            group_windows.append(sample_id)
            timestamp_windows.append(pd.Timestamp(ts_slice.iloc[-1]))
            metadata_rows.append(
                {
                    group_col: sample_id,
                    'window_index': int(window_idx),
                    'start_pos': int(start),
                    'end_pos': int(min(start + window_size, len(group_df)) - 1),
                    'window_size': int(window_size),
                    'stride': int(stride),
                    'label_mode': label_mode,
                    'window_label': int(label) if require_target else None,
                    'timestamp_start': pd.Timestamp(ts_slice.iloc[0]),
                    'timestamp_end': pd.Timestamp(ts_slice.iloc[-1]),
                }
            )

    if x_windows:
        X_seq = np.stack(x_windows, axis=0).astype(np.float32)
        y_seq = np.asarray(y_windows, dtype=np.int64) if require_target else np.asarray([], dtype=np.int64)
        group_seq = np.asarray(group_windows)
        timestamps_seq = np.asarray(timestamp_windows, dtype='datetime64[ns]')
    else:
        X_seq = np.zeros((0, window_size, len(feature_columns)), dtype=np.float32)
        y_seq = np.zeros((0,), dtype=np.int64)
        group_seq = np.asarray([], dtype=object)
        timestamps_seq = np.asarray([], dtype='datetime64[ns]')

    window_meta = pd.DataFrame(metadata_rows)
    return {
        'X_seq': X_seq,
        'y_seq': y_seq,
        'group_seq': group_seq,
        'timestamps_seq': timestamps_seq,
        'window_meta': window_meta,
        'feature_columns': feature_columns,
    }
