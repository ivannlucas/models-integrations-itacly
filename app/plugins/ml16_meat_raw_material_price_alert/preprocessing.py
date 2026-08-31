"""Feature engineering and windowing.

Faithful port of src/data_processing/preprocess.py::create_endogenous_features and
src/predict/predictor.py::prepare_input from inbox/a16/codigo/ (see inbox/a16/manifest.yaml).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.domain.services.exceptions import InsufficientRowsError
from app.plugins.ml16_meat_raw_material_price_alert.constants import (
    DEFAULT_HORIZON,
    DEFAULT_LOOKBACK,
    FEATURE_WARMUP_ROWS,
    RAW_REQUIRED_COLUMNS,
)


def build_raw_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build a raw monthly DataFrame from a list of row dicts (inline predict)."""
    return pd.DataFrame(rows)


def validate_required_columns(df: pd.DataFrame) -> None:
    """Reject a CSV/row-set missing the required base columns.

    'month' is functionally required by create_endogenous_features() but is derived
    automatically from 'fecha' if absent (see ensure_month_column) — an addition on top of
    the original code, which never derives it. Every other column is mandatory as-is.
    """
    missing = [c for c in RAW_REQUIRED_COLUMNS if c != "month" and c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")


def ensure_month_column(df: pd.DataFrame) -> pd.DataFrame:
    """Derive 'month' from 'fecha' if not supplied.

    Safe because 'fecha' is always day 1 of the month, so the numeric result is identical to
    the original schema's 'month' column. This is a plugin-level convenience on top of the
    delivered code, which never derives 'month' automatically — see manifest known_issues.
    """
    if "month" in df.columns:
        return df
    df = df.copy()
    df["month"] = pd.to_datetime(df["fecha"]).dt.month
    return df


def create_endogenous_features(df: pd.DataFrame) -> pd.DataFrame:
    """Faithful port of src/data_processing/preprocess.py::create_endogenous_features.

    Produces 11 endogenous price features (momentum, MA3, volatility, z-score, spread,
    cyclical month encoding) plus 8 exogenous derived features (epidemic/precipitation lags).
    """
    df = df.copy()

    for col in ("indice_animales", "indice_insumos"):
        df[f"mom_{col}"] = df[col].pct_change() * 100
        df[f"ma3_{col}"] = df[col].rolling(3).mean()
        df[f"vol3_{col}"] = df[col].rolling(3).std()
        df[f"dev_{col}"] = (df[col] - df[f"ma3_{col}"]) / df[f"vol3_{col}"].replace(0, np.nan)

    df["spread"] = df["indice_animales"] / df["indice_insumos"]
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    for lag in range(1, 4):
        df[f"animales_afectados_lag{lag}"] = df["animales_afectados"].shift(lag)
    df["mom_animales_afectados"] = (
        df["animales_afectados"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    )

    for lag in range(3, 7):
        df[f"precip_total_lag{lag}"] = df["precip_total"].shift(lag)

    return df


def create_sequences(data_x: np.ndarray, lookback: int) -> np.ndarray:
    """Port of src/training/trainer.py::create_sequences_clf (inference side — no y needed).

    Returns an array (n_seq, lookback, n_features): sequence i covers rows
    [i - lookback + 1 .. i] of data_x, for i in range(lookback, len(data_x)).
    """
    sequences = [data_x[i - lookback + 1:i + 1] for i in range(lookback, len(data_x))]
    return np.array(sequences)


def prepare_inference_input(df: pd.DataFrame, train_config: dict, scalers: dict) -> dict:
    """Port of src/predict/predictor.py::prepare_input.

    Parameters
    ----------
    df : raw monthly DataFrame — same schema as dataset_clasificacion_base.csv
    train_config : train_config.json contents (input_cols_per_target, lookback, horizon)
    scalers : {target: fitted MinMaxScaler}

    Returns
    -------
    dict with 'x_flat_per_target' (target -> 2D array), 'fechas' (target month, t + horizon),
    'df_processed' (post-feature-engineering frame).

    Raises
    ------
    InsufficientRowsError: not enough historical rows to build even one window, either before
    or after the lag-feature dropna (see FEATURE_WARMUP_ROWS + lookback + 1 in the manifest's
    inputs.constraints).
    """
    validate_required_columns(df)
    df = ensure_month_column(df)

    input_cols_per_target = train_config["input_cols_per_target"]
    lookback = int(train_config.get("lookback", DEFAULT_LOOKBACK))
    horizon = int(train_config.get("horizon", DEFAULT_HORIZON))

    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    min_rows_required = FEATURE_WARMUP_ROWS + lookback + 1
    n_rows = len(df)
    if n_rows < min_rows_required:
        date_min = df["fecha"].min().date() if n_rows else "N/A"
        date_max = df["fecha"].max().date() if n_rows else "N/A"
        raise InsufficientRowsError(
            f"Histórico insuficiente: se necesitan al menos {min_rows_required} meses "
            f"(warmup={FEATURE_WARMUP_ROWS} + lookback={lookback} + 1), pero se recibieron "
            f"{n_rows} meses ({date_min} -> {date_max})."
        )

    df = create_endogenous_features(df)
    all_feat_cols = {c for cols in input_cols_per_target.values() for c in cols}
    df = df.dropna(subset=list(all_feat_cols)).reset_index(drop=True)
    if len(df) <= lookback:
        raise InsufficientRowsError(
            "Tras la ingeniería de variables y el descarte de nulos (lags/rolling) no quedan "
            f"filas suficientes para construir ni una ventana de lookback={lookback}: "
            f"{len(df)} filas útiles de {n_rows} recibidas."
        )

    x_flat_per_target: dict[str, np.ndarray] = {}
    for target, cols in input_cols_per_target.items():
        scaled = scalers[target].transform(df[cols])
        x_seq = create_sequences(scaled, lookback)
        n_flat = lookback * len(cols)
        x_flat_per_target[target] = x_seq.reshape(x_seq.shape[0], n_flat)

    valid_dates = pd.DatetimeIndex(df["fecha"].values[lookback:]) + pd.DateOffset(months=horizon)

    return {
        "x_flat_per_target": x_flat_per_target,
        "fechas": valid_dates,
        "df_processed": df,
    }
