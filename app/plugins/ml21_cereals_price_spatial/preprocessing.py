"""Preprocessing for ml21 — ESP-CEREAL spatial cereal price prediction.

Replicates the BLACKLIST filtering and one-hot encoding from the original
prepare_data.py to ensure feature alignment with model_metadata.json.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from app.domain.services.exceptions import DataContractError
from app.plugins.ml21_cereals_price_spatial.constants import (
    BLACKLIST,
    PROB_BEAR,
    PROB_BULL,
    RET_BEAR,
    RET_BULL,
    VALID_HORIZONS,
)

BASE_PRICE_COL = "precio_provincial_lag_1"
CUT_DATE = pd.Timestamp("2021-01-01")


def build_features_from_row(
    row: pd.DataFrame,
    expected_cols: list[str],
) -> pd.DataFrame:
    """Build model-ready feature matrix from raw panel rows, aligned to training schema.

    Mirrors prepare_data._build_features but does NOT require target columns.
    Columns not present in the inference batch are filled with 0 to match the training schema.
    """
    drop = set(BLACKLIST) | {"date"}
    for h in VALID_HORIZONS:
        drop.add(f"precio_provincial_TARGET_H{h}")
        drop.add(f"target_return_h{h}")
        drop.add(f"target_class_h{h}")

    keep = [c for c in row.columns if c not in drop]
    feat = row[keep].copy()

    obj_cols = feat.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in obj_cols:
        feat[col] = feat[col].astype(str)
    feat = pd.get_dummies(feat, columns=obj_cols, drop_first=False)
    feat = feat.reindex(sorted(feat.columns), axis=1)

    return feat.reindex(columns=expected_cols, fill_value=0)


def lookup_panel_row(
    panel_df: pd.DataFrame,
    province: str | None,
    cereal: str | None,
    month: str | None,
) -> pd.Series:
    """Find the base-panel row matching provincia/cereal_predominante/mes.

    Mirrors the "production inference" branch of run_single() in the original
    src/predict/predict_v1.py: filters the full panel to date >= CUT_DATE (the model's
    serving period — pre-CUT_DATE rows are training-only) and to the requested
    province/cereal/month, then takes the most recent matching row. This is what
    supplies the ~90 engineered columns (lat_centroide, month_sin/cos, fase_*, price
    lags, climate, MAPA indices...) that predict_inline's caller is never expected to
    provide directly (manifest inputs.fixed is only provincia/cereal_predominante/date).
    """
    candidate = panel_df[panel_df["date"] >= CUT_DATE].copy()
    if province:
        candidate = candidate[candidate["provincia"].astype(str).str.lower() == province.lower()]
    if cereal:
        candidate = candidate[
            candidate["cereal_predominante"].astype(str).str.lower() == cereal.lower()
        ]
    if month:
        m_ts = pd.to_datetime(f"{month}-01", errors="coerce")
        if pd.notna(m_ts):
            candidate = candidate[candidate["date"].dt.to_period("M") == m_ts.to_period("M")]

    if candidate.empty:
        raise DataContractError(
            f"Sin datos de panel para provincia={province!r} cereal={cereal!r} mes={month!r}. "
            "Verifica que la combinacion exista en el dataset base o actualiza el panel."
        )
    return candidate.sort_values("date").iloc[-1]


def validate_expected_columns(
    actual_cols: list[str],
    expected_cols: list[str],
    horizon: int,
    task: str,
) -> None:
    """Raise ValueError if feature columns don't match expected schema."""
    missing = sorted(set(expected_cols) - set(actual_cols))
    extra = sorted(set(actual_cols) - set(expected_cols))
    if missing or extra:
        raise ValueError(
            f"Column mismatch en H{horizon} {task}. Missing={missing[:10]} Extra={extra[:10]}"
        )


def get_selected_feature_columns(
    metadata: dict,
    horizon: int,
    task: str,
) -> list[str]:
    """Extract expected feature columns from metadata for a given horizon/task."""
    key = f"H{horizon}"
    block = metadata.get("selected_models", {}).get(key, {})
    if not block:
        raise ValueError(f"No se encontro bloque {key} en model_metadata.json")

    if task == "regresion":
        cols = block["regression"].get("expected_columns", [])
    else:
        cols = block["classification"].get("expected_columns", [])

    if not cols:
        raise ValueError(f"No se encontraron expected_columns para {key} {task}")
    return cols


def signal_from_prob_return(prob_up: float, expected_return: float) -> str:
    """Derive directional signal from probability and return."""
    if prob_up > PROB_BULL and expected_return > RET_BULL:
        return "ALCISTA"
    if prob_up < PROB_BEAR and expected_return < RET_BEAR:
        return "BAJISTA"
    return "NEUTRAL/ESPERA"


# ── Training helpers (replicate prepare_data.py logic) ──────────────────────


def _target_col(horizon: int) -> str:
    return f"precio_provincial_TARGET_H{horizon}"


def add_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create return and direction targets for a given horizon."""
    tgt_col = _target_col(horizon)
    if tgt_col not in df.columns:
        raise ValueError(f"Columna target no encontrada para H{horizon}: {tgt_col}")
    if BASE_PRICE_COL not in df.columns:
        raise ValueError(f"Columna base no encontrada para retornos: {BASE_PRICE_COL}")

    data = df.copy()
    base = pd.to_numeric(data[BASE_PRICE_COL], errors="coerce")
    future = pd.to_numeric(data[tgt_col], errors="coerce")

    valid = base.notna() & future.notna() & base.ne(0)
    data = data.loc[valid].copy()

    ret_col = f"target_return_h{horizon}"
    clf_col = f"target_class_h{horizon}"

    data[ret_col] = (data[tgt_col] - data[BASE_PRICE_COL]) / data[BASE_PRICE_COL]
    data[clf_col] = (data[ret_col] > 0).astype(int)
    return data


def build_features_batch(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build model-ready feature matrix from a batch of raw panel rows.

    Mirrors prepare_data._build_features: drops blacklist, targets, date;
    one-hot encodes object columns; sorts columns alphabetically.
    """
    drop_cols = set(BLACKLIST)
    drop_cols.add("date")

    for h in VALID_HORIZONS:
        drop_cols.add(_target_col(h))
        drop_cols.add(f"target_return_h{h}")
        drop_cols.add(f"target_class_h{h}")

    keep_cols = [c for c in df.columns if c not in drop_cols]
    feat = df[keep_cols].copy()

    obj_cols = feat.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in obj_cols:
        feat[col] = feat[col].astype(str)
    feat = pd.get_dummies(feat, columns=obj_cols, drop_first=False)
    feat = feat.reindex(sorted(feat.columns), axis=1)
    return feat


def split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split: train < CUT_DATE, test >= CUT_DATE."""
    train_mask = df["date"] < CUT_DATE
    test_mask = df["date"] >= CUT_DATE
    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            "Split temporal vacio. Verifica fechas y punto de corte 2021-01-01."
        )
    return train_df, test_df


def prepare_train_test(
    df: pd.DataFrame,
    horizon: int,
    task: Literal["regresion", "clasificacion"],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return model-ready train/test sets for one horizon and task.

    Mirrors prepare_data.get_prepared_data but operates on a user-provided
    DataFrame instead of loading from disk.
    """
    full = add_targets(df, horizon)
    train_df, test_df = split_temporal(full)

    y_col = f"target_return_h{horizon}" if task == "regresion" else f"target_class_h{horizon}"

    X_train = build_features_batch(train_df, horizon)
    X_test_raw = build_features_batch(test_df, horizon)
    X_test = X_test_raw.reindex(columns=X_train.columns, fill_value=0)

    y_train = train_df[y_col].astype(float if task == "regresion" else int)
    y_test = test_df[y_col].astype(float if task == "regresion" else int)

    return X_train, X_test, y_train, y_test
