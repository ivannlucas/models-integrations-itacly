"""Feature engineering and sequence building for ml18 — faithful port of
src/data_processing/preprocess.py + the relevant parts of src/main.py::predict() (forecast
mode only — see inbox/a18/manifest.yaml known_issues) from the delivered code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.domain.services.exceptions import InsufficientRowsError
from app.plugins.ml18_meat_spatial_price_forecast.constants import (
    CCAA_COL,
    DATE_COL,
    FEATURE_COLUMNS,
    LOOKBACK,
    OWN_PRICE_FEATURE,
    PRODUCTO_COL,
    RAW_REQUIRED_COLS,
    TARGET_COL,
    VECINOS_CCAA,
)

_CARRY_FORWARD_COLS = ("CONSUMO X CAPITA", "PENETRACION (%)", "Poblacion", "RentaHogar")


def build_raw_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build and minimally validate a DataFrame from a list of panel row dicts."""
    if not rows:
        raise InsufficientRowsError(
            f"Historial vacío: se necesita al menos {LOOKBACK} meses de histórico real "
            "consecutivo por combinación CCAA-Producto."
        )
    df = pd.DataFrame(rows)
    missing = [c for c in RAW_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise InsufficientRowsError(f"Faltan columnas mínimas en el histórico: {missing}.")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    invalid = int(df[DATE_COL].isna().sum())
    if invalid:
        raise InsufficientRowsError(
            f"La columna '{DATE_COL}' contiene {invalid} valores no parseables (formato esperado YYYY-MM-DD)."
        )
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = np.nan
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out[DATE_COL].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out


def add_own_price_feature(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[OWN_PRICE_FEATURE] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    return out


def add_spatial_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of neighboring CCAA's same-month values (see manifest known_issues on naming)."""
    out = df.copy()
    base_cols = [DATE_COL, PRODUCTO_COL, CCAA_COL, "CONSUMO X CAPITA", TARGET_COL]
    base = out[base_cols].copy()

    edges = [
        {"CCAA": ccaa, "VECINA": vecina}
        for ccaa, vecinos in VECINOS_CCAA.items()
        for vecina in vecinos
    ]
    edges_df = pd.DataFrame(edges)

    if not edges_df.empty:
        lag_input = base.merge(edges_df, on="CCAA", how="left")
        vecinos_vals = base.rename(columns={
            "CCAA": "VECINA", "CONSUMO X CAPITA": "CONSUMO_VECINA", TARGET_COL: "PRECIO_VECINA",
        })
        lag_input = lag_input.merge(
            vecinos_vals[[DATE_COL, PRODUCTO_COL, "VECINA", "CONSUMO_VECINA", "PRECIO_VECINA"]],
            on=[DATE_COL, PRODUCTO_COL, "VECINA"], how="left",
        )
        agg = (
            lag_input.groupby([DATE_COL, PRODUCTO_COL, CCAA_COL], as_index=False)
            .agg(LAG_CONSUMO_VECINOS=("CONSUMO_VECINA", "mean"), LAG_PRECIO_VECINOS=("PRECIO_VECINA", "mean"))
        )
        out = out.merge(agg, on=[DATE_COL, PRODUCTO_COL, CCAA_COL], how="left")
    else:
        out["LAG_CONSUMO_VECINOS"] = np.nan
        out["LAG_PRECIO_VECINOS"] = np.nan

    out["LAG_CONSUMO_VECINOS"] = out["LAG_CONSUMO_VECINOS"].fillna(out["CONSUMO X CAPITA"]).fillna(0.0)
    out["LAG_PRECIO_VECINOS"] = out["LAG_PRECIO_VECINOS"].fillna(out[TARGET_COL]).fillna(0.0)
    return out


def fill_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out.groupby(CCAA_COL)[col].transform(lambda s: s.ffill())
        median = out[col].median()
        out[col] = out[col].fillna(median if pd.notna(median) else 0.0)
    return out


def build_forecast_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Append one future row per (CCAA, Producto) group, carrying forward exogenous columns."""
    out = df.copy()
    out["ES_FORECAST"] = False

    new_rows = []
    for (ccaa, producto), g in out.groupby([CCAA_COL, PRODUCTO_COL]):
        g = g.sort_values(DATE_COL)
        last_row = g.iloc[-1]
        new_row = {col: last_row[col] for col in _CARRY_FORWARD_COLS if col in g.columns}
        new_row[DATE_COL] = last_row[DATE_COL] + pd.DateOffset(months=1)
        new_row[CCAA_COL] = ccaa
        new_row[PRODUCTO_COL] = producto
        new_row[TARGET_COL] = np.nan
        new_row["ES_FORECAST"] = True
        new_rows.append(new_row)

    forecast_df = pd.DataFrame(new_rows)
    out = pd.concat([out, forecast_df], ignore_index=True, sort=False)
    return out.sort_values([CCAA_COL, PRODUCTO_COL, DATE_COL]).reset_index(drop=True)


def build_modeling_panel(rows: list[dict]) -> pd.DataFrame:
    """Full pipeline: raw rows -> validated, feature-engineered panel with a forecast row
    appended per (CCAA, Producto) group. Does not scale or build sequences — see run_inference.
    """
    df = build_raw_dataframe(rows)
    df = build_forecast_frame(df)
    df = add_time_features(df)
    df = add_spatial_lags(df)
    df = add_own_price_feature(df)
    df = fill_numeric(df, list(FEATURE_COLUMNS))
    return df


def build_sequences(df: pd.DataFrame) -> list[dict]:
    """Build one (CCAA, Producto, window) entry per group with >= LOOKBACK historical rows.

    Returns a list of {"ccaa", "producto", "window": np.ndarray[LOOKBACK, n_features],
    "target_date": pd.Timestamp} — one per valid group, in groupby order.
    """
    entries = []
    work = df.sort_values([CCAA_COL, PRODUCTO_COL, DATE_COL])
    for (ccaa, producto), g in work.groupby([CCAA_COL, PRODUCTO_COL]):
        g = g.reset_index(drop=True)
        if len(g) <= LOOKBACK:
            continue
        # Solo nos interesa la última ventana (el mes de forecast añadido por build_forecast_frame).
        i = len(g) - 1
        window = g.loc[i - LOOKBACK:i - 1, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        entries.append({
            "ccaa": ccaa,
            "producto": producto,
            "window": window,
            "target_date": g.loc[i, DATE_COL],
        })

    if not entries:
        raise InsufficientRowsError(
            "No se generaron secuencias: cada combinación CCAA-Producto necesita al menos "
            f"{LOOKBACK} meses de histórico real consecutivos (el servicio añade automáticamente "
            "la fila del mes siguiente a predecir, no hace falta aportarla)."
        )
    return entries
