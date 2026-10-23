"""Vendored from the AI team's src/training/spatial_price_model.py (a24, MODEL-24 / CU12).

Every function is copied verbatim, in the same order, EXCEPT these deliberate I/O adaptations
(the plugin keeps the delivered data in memory instead of re-reading CSV paths per request):

* ``build_product_panel`` receives the long-format series as a DataFrame (``series``) instead
  of ``input_csv``; the filtering/pivot logic is unchanged. Error messages say "serie de
  entrada" instead of the CSV path.
* ``load_spatial_weights`` receives the border-percentage matrix as a DataFrame already read
  with ``index_col=0`` instead of a path; the renaming/reindexing logic is unchanged.
* ``build_feature_base`` / ``build_supervised_training_frame`` / ``build_inference_feature_frame``
  keep their parameter name ``spatial_weights_csv`` but receive that DataFrame.
* Path/config helpers of the original (resolve_path, load_release_config, save_json, the weekly
  exogenous builders and ensure_* loaders, validate_inference_input) are not vendored: the
  plugin loads artifacts through ArtifactStore and validates input in preprocessing.py.

Do not edit the numerical logic here: inference and /train must stay bit-for-bit identical to
the delivered code (verified against inference_smoke.csv, see outputs/a24/verification_report.md).
"""
# pylint: skip-file

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

LAGS = (1, 3, 6, 12)
ROLL_WINDOWS = (3, 6, 12)
DEFAULT_INPUT_EXOG_PATTERNS = (
    "exog__milk_price",
    "exog__fut_corn_",
    "exog__fut_soymeal_",
    "exog__fut_brent_",
    "exog__gasoleo_precio",
    "exog__mibgas_d1",
    "exog__pvpc_medio",
    "exog__pienso_",
    "exog__lonja_",
    "exog__cebada_price",
    "exog__fert_idx1_alimentos_de_ganado",
    "exog__fert_idx1_fertiliz",
    "exog__fert_idx2_energia_y_lubricantes",
    "exog__fert_prepag1_",
    "exog__fert_prepag2_",
)


def smape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    denom = np.abs(y) + np.abs(p)
    valid = denom != 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(2.0 * np.abs(p[valid] - y[valid]) / denom[valid]) * 100.0)


def slugify_metric(metric_name: str) -> str:
    mapping = {
        "CONSUMO X CAPITA": "consumo_x_capita",
        "GASTO X CAPITA": "gasto_x_capita",
        "PENETRACION (%)": "penetracion_pct",
        "PRECIO MEDIO kg o litros": "precio_medio_kg_o_litros",
        "VALOR (Miles Euros)": "valor_miles_euros",
        "VOLUMEN (Miles kg o litros)": "volumen_miles_kg_o_litros",
    }
    if metric_name not in mapping:
        raise ValueError(f"Metrica no soportada: {metric_name}")
    return mapping[metric_name]


def slugify_product(product_name: str) -> str:
    cleaned = (
        unicodedata.normalize("NFKD", product_name).encode("ascii", "ignore").decode("ascii").lower()
        .replace(".", "")
        .replace(",", "")
        .replace("%", "pct")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace("/", "_")
    )
    slug = "_".join(cleaned.split())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


# Contrato minimo de entrada para inferencia con un CSV personalizado.
# El CSV debe seguir el mismo esquema que el ETL oficial (data_processing.py).
INFERENCE_REQUIRED_COLUMNS = ("anio", "mes", "ccaa", "producto", "metrica", "valor")
# Metrica objetivo del modelo (precio medio del producto ancla).
INFERENCE_TARGET_METRIC = "PRECIO MEDIO kg o litros"


def build_product_panel(
    series: pd.DataFrame,
    anchor_product: str,
    context_products: list[str],
    include_spain: bool,
    min_year: int | None,
    max_year: int | None,
    add_context_derivatives: bool,
) -> pd.DataFrame:
    # [plugin] original: source_path = resolve_path(input_csv); df = pd.read_csv(source_path, low_memory=False)
    df = series
    required = {"anio", "mes", "ccaa", "producto", "metrica", "valor", "es_reconstruido"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in processed series CSV: {sorted(missing)}")

    selected_products = [anchor_product, *context_products]
    work = df[df["producto"].isin(selected_products)].copy()
    if not include_spain:
        work = work[work["ccaa"] != "T.ESPANA"].copy()
    if min_year is not None:
        work = work[work["anio"] >= int(min_year)].copy()
    if max_year is not None:
        work = work[work["anio"] <= int(max_year)].copy()
    if work.empty:
        available_products = sorted(str(p) for p in df["producto"].dropna().unique())
        found = [p for p in selected_products if p in set(available_products)]
        raise ValueError(
            "No rows left after filtering product panel. "
            "La serie de entrada no contiene datos para los productos "
            f"requeridos por el modelo tras aplicar los filtros de CCAA/anio.\n"
            f"  Productos requeridos: {selected_products}\n"
            f"  De esos, presentes en el CSV: {found if found else 'ninguno'}\n"
            f"  Rango de anios filtrado: [{min_year}, {max_year}]\n"
            f"  Productos disponibles en el CSV: {available_products}"
        )

    work["period"] = work["anio"].astype(int).astype(str) + "-" + work["mes"].astype(int).astype(str).str.zfill(2)
    work["period_dt"] = pd.to_datetime(dict(year=work["anio"], month=work["mes"], day=1))
    work["feature_name"] = work["producto"].map(slugify_product) + "__" + work["metrica"].map(slugify_metric)

    panel = (
        work.pivot_table(
            index=["anio", "mes", "period", "period_dt", "ccaa"],
            columns="feature_name",
            values="valor",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["ccaa", "period_dt"])
    )

    reconstructed = (
        work.groupby(["anio", "mes", "period", "period_dt", "ccaa"], as_index=False)["es_reconstruido"]
        .max()
        .rename(columns={"es_reconstruido": "row_has_reconstructed_values"})
    )
    panel = panel.merge(reconstructed, on=["anio", "mes", "period", "period_dt", "ccaa"], how="left")
    panel["row_has_reconstructed_values"] = panel["row_has_reconstructed_values"].fillna(False).astype(bool)
    panel["anchor_product"] = anchor_product
    panel["context_products"] = "|".join(context_products)

    anchor_slug = slugify_product(anchor_product)
    anchor_target_col = f"{anchor_slug}__precio_medio_kg_o_litros"
    if anchor_target_col not in panel.columns:
        raise ValueError(f"Anchor target column not found in panel: {anchor_target_col}")

    if add_context_derivatives and context_products:
        anchor_volume_col = f"{anchor_slug}__volumen_miles_kg_o_litros"
        for product in context_products:
            product_slug = slugify_product(product)
            price_col = f"{product_slug}__precio_medio_kg_o_litros"
            volume_col = f"{product_slug}__volumen_miles_kg_o_litros"
            if price_col in panel.columns:
                panel[f"derived__price_diff_vs_{product_slug}"] = panel[anchor_target_col] - panel[price_col]
            if volume_col in panel.columns and anchor_volume_col in panel.columns:
                denom = panel[anchor_volume_col].replace(0, np.nan)
                panel[f"derived__volume_share_{product_slug}"] = panel[volume_col] / denom

    return panel


def parse_float_grid(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("Grid cannot be empty.")
    return values


def load_spatial_weights(path: pd.DataFrame, entities: list[str]) -> pd.DataFrame:
    # [plugin] original: weights = pd.read_csv(resolve_path(path), index_col=0)
    weights = path.copy()
    weights.index = weights.index.astype(str)
    weights.columns = weights.columns.astype(str)
    entity_slugs = [slugify_product(value) for value in entities]
    weights = weights.rename(index=lambda value: slugify_product(value), columns=lambda value: slugify_product(value))
    missing = set(entity_slugs) - set(weights.index)
    if missing:
        raise ValueError(f"Spatial weights missing rows for entities: {sorted(missing)}")
    return weights.reindex(index=entity_slugs, columns=entity_slugs).fillna(0.0)


def add_spatial_weighted_series(
    work: pd.DataFrame,
    entity_col: str,
    metric_cols: list[str],
    weights: pd.DataFrame,
) -> pd.DataFrame:
    out = work.copy()
    entity_lookup = (
        out[[entity_col]]
        .drop_duplicates()
        .assign(entity_slug=lambda frame: frame[entity_col].map(slugify_product))
    )
    for signal in metric_cols:
        pivot = out.pivot(index="period_dt", columns=entity_col, values=signal)
        pivot = pivot.rename(columns=lambda value: slugify_product(value)).reindex(columns=weights.index)
        spatial = pivot.dot(weights.T).stack().reset_index(name=f"{signal}_spatial_wavg")
        spatial = spatial.rename(columns={spatial.columns[1]: "entity_slug"})
        spatial = spatial.merge(entity_lookup, on="entity_slug", how="left").drop(columns=["entity_slug"])
        out = out.merge(spatial, on=["period_dt", entity_col], how="left")
    return out


def build_feature_base(
    panel: pd.DataFrame,
    entity_col: str,
    target_col: str,
    spatial_weights_csv: str | Path,
    publication_lag_months: int = 0,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    work = panel.copy()
    work["period_dt"] = pd.to_datetime(work["period_dt"])
    work = work.sort_values([entity_col, "period_dt"]).copy()

    generated_suffixes = (
        "_lag_1",
        "_lag_3",
        "_lag_6",
        "_lag_12",
        "_roll_mean_3",
        "_roll_std_3",
        "_roll_mean_6",
        "_roll_std_6",
        "_roll_mean_12",
        "_roll_std_12",
        "_spatial_wavg",
        "_spain_mean",
        "_local_minus_spatial_lag_1",
        "_local_minus_spain_lag_1",
    )
    metric_cols = [
        column
        for column in work.columns
        if "__" in column
        and pd.api.types.is_numeric_dtype(work[column])
        and work[column].notna().any()
        and not column.endswith(generated_suffixes)
    ]
    if target_col not in metric_cols:
        raise ValueError(f"Target column not found among metric columns: {target_col}")

    entities = sorted(work[entity_col].dropna().astype(str).unique().tolist())
    work[entity_col] = work[entity_col].astype(str)
    weights = load_spatial_weights(spatial_weights_csv, entities)
    work = add_spatial_weighted_series(work, entity_col=entity_col, metric_cols=metric_cols, weights=weights)
    work = work.sort_values([entity_col, "period_dt"]).copy()

    feature_cols: list[str] = [entity_col, "mes", "month_sin", "month_cos", "quarter"]
    categorical_cols = [entity_col]
    group = work.groupby(entity_col, group_keys=False)

    work["month_sin"] = np.sin(2 * np.pi * work["mes"].astype(int) / 12.0)
    work["month_cos"] = np.cos(2 * np.pi * work["mes"].astype(int) / 12.0)
    work["quarter"] = ((work["mes"].astype(int) - 1) // 3 + 1).astype(int)

    # publication_lag_months simula el retraso real de publicacion de la fuente MAPA:
    # todas las features derivadas de senales del panel (target y contexto, todas de
    # origen MAPA) se retrasan un numero adicional de meses para reflejar que, en
    # produccion, esos datos no estan disponibles de inmediato (ver Nota 1 del revisor).
    extra_lag = int(publication_lag_months)

    for signal in metric_cols:
        spain_mean_col = f"{signal}_spain_mean"
        work[spain_mean_col] = work.groupby("period_dt")[signal].transform("mean")

        for lag in LAGS:
            lag_col = f"{signal}_lag_{lag}"
            work[lag_col] = group[signal].shift(lag + extra_lag)
            feature_cols.append(lag_col)

        for window in ROLL_WINDOWS:
            mean_col = f"{signal}_roll_mean_{window}"
            std_col = f"{signal}_roll_std_{window}"
            work[mean_col] = group[signal].shift(1 + extra_lag).rolling(window).mean()
            work[std_col] = group[signal].shift(1 + extra_lag).rolling(window).std()
            feature_cols.extend([mean_col, std_col])

        spatial_base = f"{signal}_spatial_wavg"
        for lag in LAGS:
            spatial_lag_col = f"{spatial_base}_lag_{lag}"
            work[spatial_lag_col] = group[spatial_base].shift(lag + extra_lag)
            feature_cols.append(spatial_lag_col)

        local_minus_spatial_col = f"{signal}_local_minus_spatial_lag_1"
        work[local_minus_spatial_col] = group[signal].shift(1 + extra_lag) - group[spatial_base].shift(1 + extra_lag)
        feature_cols.append(local_minus_spatial_col)

        local_minus_spain_col = f"{signal}_local_minus_spain_lag_1"
        work[local_minus_spain_col] = group[signal].shift(1 + extra_lag) - group[spain_mean_col].shift(1 + extra_lag)
        feature_cols.append(local_minus_spain_col)

    target_ratio = work[target_col] / work.groupby("period_dt")[target_col].transform("mean")
    work["target_share_vs_spain_lag_12"] = target_ratio.groupby(work[entity_col]).shift(12 + extra_lag)
    work["target_abs_delta_1"] = np.abs(
        group[target_col].shift(1 + extra_lag) - group[target_col].shift(2 + extra_lag)
    )
    feature_cols.extend(["target_share_vs_spain_lag_12", "target_abs_delta_1"])
    return work, feature_cols, categorical_cols


def build_supervised_training_frame(
    panel: pd.DataFrame,
    entity_col: str,
    target_col: str,
    horizon_months: int,
    spatial_weights_csv: str | Path,
    publication_lag_months: int = 0,
) -> tuple[pd.DataFrame, list[str], list[str], pd.DataFrame]:
    work, full_feature_cols, categorical_cols = build_feature_base(
        panel=panel,
        entity_col=entity_col,
        target_col=target_col,
        spatial_weights_csv=spatial_weights_csv,
        publication_lag_months=publication_lag_months,
    )

    extra_lag = int(publication_lag_months)
    group = work.groupby(entity_col, group_keys=False)
    seasonal_shift = 12 - horizon_months
    if seasonal_shift < 0:
        raise ValueError("Only horizons up to 12 months are supported.")

    work["y_t_plus_h"] = group[target_col].shift(-horizon_months)
    work["y_persist_1"] = group[target_col].shift(0 + extra_lag)
    work["y_persist_12"] = group[target_col].shift(seasonal_shift + extra_lag)
    work["y_residual"] = work["y_t_plus_h"] - work["y_persist_1"]
    work["target_period_dt"] = group["period_dt"].shift(-horizon_months)
    work["target_period"] = group["period"].shift(-horizon_months)

    required = full_feature_cols + [
        "y_t_plus_h",
        "y_persist_1",
        "y_persist_12",
        "y_residual",
        "target_period_dt",
        "target_period",
    ]
    frame = work.dropna(subset=required).copy()
    return frame, full_feature_cols, categorical_cols, work


def build_inference_feature_frame(
    panel: pd.DataFrame,
    entity_col: str,
    target_col: str,
    horizon_months: int,
    spatial_weights_csv: str | Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    work, full_feature_cols, categorical_cols = build_feature_base(
        panel=panel,
        entity_col=entity_col,
        target_col=target_col,
        spatial_weights_csv=spatial_weights_csv,
    )

    group = work.groupby(entity_col, group_keys=False)
    seasonal_shift = 12 - horizon_months
    work["y_persist_1"] = work[target_col]
    work["y_persist_12"] = group[target_col].shift(seasonal_shift)
    work["target_period_dt"] = work["period_dt"] + pd.DateOffset(months=horizon_months)
    work["target_period"] = work["target_period_dt"].dt.to_period("M").astype(str)
    return work.copy(), full_feature_cols, categorical_cols


def add_compact_features(
    frame: pd.DataFrame,
    target_col: str,
    include_target_history: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Construye el bloque compacto de features siempre presente en el modelo.

    `include_target_history=False` excluye del feature set todo lo derivado del
    propio historico de precio MAPA (persistencia, hueco estacional, medias
    espaciales, cuota vs Espana): usado por el experimento "totalmente exogeno"
    de la Nota 1, que evalua si el modelo puede prescindir de la autorregresion
    sobre la fuente con retraso. Las columnas se siguen calculando (las necesita
    el baseline y otras interacciones), solo se excluyen del feature set del ML.
    """
    work = frame.copy()
    work["target_month"] = pd.to_datetime(work["target_period_dt"]).dt.month.astype(int)
    work["target_month_sin"] = np.sin(2.0 * np.pi * work["target_month"] / 12.0)
    work["target_month_cos"] = np.cos(2.0 * np.pi * work["target_month"] / 12.0)
    work["seasonal_gap"] = work["y_persist_1"] - work["y_persist_12"]
    work["abs_seasonal_gap"] = work["seasonal_gap"].abs()
    work["rel_seasonal_gap"] = work["seasonal_gap"] / work["y_persist_1"].replace(0, np.nan)

    feature_cols = ["ccaa", "target_month", "target_month_sin", "target_month_cos", "quarter"]
    if include_target_history:
        feature_cols.extend(
            [
                "y_persist_1",
                "y_persist_12",
                "seasonal_gap",
                "abs_seasonal_gap",
                "rel_seasonal_gap",
                "target_share_vs_spain_lag_12",
                "target_abs_delta_1",
            ]
        )
        spatial_candidates = [
            f"{target_col}_spatial_wavg_lag_1",
            f"{target_col}_spatial_wavg_lag_3",
            f"{target_col}_spatial_wavg_lag_6",
            f"{target_col}_spatial_wavg_lag_12",
            f"{target_col}_local_minus_spatial_lag_1",
            f"{target_col}_local_minus_spain_lag_1",
        ]
        feature_cols.extend([column for column in spatial_candidates if column in work.columns])
    categorical_cols = ["ccaa"]
    return work, feature_cols, categorical_cols


def add_monthly_exog_lags(
    frame: pd.DataFrame,
    monthly_exog: pd.DataFrame,
    entity_col: str,
    lags: tuple[int, ...] = (1, 2, 3, 6),
) -> tuple[pd.DataFrame, list[str]]:
    work = frame.merge(monthly_exog, on=["period", "period_dt"], how="left")
    exog_cols = [column for column in work.columns if column.startswith("exog__")]
    group = work.groupby(entity_col, group_keys=False)
    lag_cols: list[str] = []
    for column in exog_cols:
        for lag in lags:
            lag_col = f"{column}_lag_{lag}"
            work[lag_col] = group[column].shift(lag)
            lag_cols.append(lag_col)
    return work, lag_cols


def filter_input_exog_lag_cols(
    exog_lag_cols: list[str],
    patterns: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    active_patterns = tuple(patterns or DEFAULT_INPUT_EXOG_PATTERNS)
    return [
        column
        for column in exog_lag_cols
        if any(column.startswith(pattern) for pattern in active_patterns)
    ]


def get_anchor_slug(target_col: str) -> str:
    return target_col.split("__", 1)[0]


def filter_anchor_extra_feature_cols(full_feature_cols: list[str], target_col: str) -> list[str]:
    anchor_slug = get_anchor_slug(target_col)
    allowed_metric_prefixes = (
        f"{anchor_slug}__consumo_x_capita",
        f"{anchor_slug}__gasto_x_capita",
        f"{anchor_slug}__penetracion_pct",
        f"{anchor_slug}__valor_miles_euros",
        f"{anchor_slug}__volumen_miles_kg_o_litros",
    )
    return [
        column
        for column in full_feature_cols
        if any(column.startswith(prefix) for prefix in allowed_metric_prefixes)
    ]


def filter_context_feature_cols(
    full_feature_cols: list[str],
    panel_columns: list[str],
    target_col: str,
) -> list[str]:
    anchor_slug = get_anchor_slug(target_col)
    signal_prefixes = sorted({column.split("__", 1)[0] for column in panel_columns if "__" in column})
    context_prefixes = [prefix for prefix in signal_prefixes if prefix != anchor_slug]
    allowed_prefixes = tuple(f"{prefix}__" for prefix in context_prefixes)
    return [
        column
        for column in full_feature_cols
        if column.startswith("derived__") or any(column.startswith(prefix) for prefix in allowed_prefixes)
    ]


def _numeric_feature_cols(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
) -> list[str]:
    return [
        column
        for column in feature_cols
        if column in df.columns
        and column not in categorical_cols
        and pd.api.types.is_numeric_dtype(df[column])
    ]


def missing_feature_summary(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
) -> dict[str, int]:
    numeric_feature_cols = _numeric_feature_cols(df, feature_cols, categorical_cols)
    if not numeric_feature_cols:
        return {
            "n_numeric_feature_cols": 0,
            "n_feature_cols_with_missing": 0,
            "n_missing_feature_cells": 0,
        }
    missing = df[numeric_feature_cols].isna()
    return {
        "n_numeric_feature_cols": int(len(numeric_feature_cols)),
        "n_feature_cols_with_missing": int(missing.any().sum()),
        "n_missing_feature_cells": int(missing.sum().sum()),
    }


def apply_missing_value_strategy(
    dfs: list[pd.DataFrame],
    feature_cols: list[str],
    categorical_cols: list[str],
    training_cfg: dict[str, Any],
    entity_col: str,
    time_col: str = "period_dt",
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    strategy = str(training_cfg.get("missing_value_strategy", "native")).strip().lower()
    outputs = [df.copy() for df in dfs]
    if not outputs:
        return outputs, {"strategy": strategy}

    numeric_feature_cols = _numeric_feature_cols(outputs[0], feature_cols, categorical_cols)
    report: dict[str, Any] = {
        "strategy": strategy,
        "feature_cols": len(feature_cols),
        "numeric_feature_cols": len(numeric_feature_cols),
        "before": [missing_feature_summary(df, feature_cols, categorical_cols) for df in outputs],
    }
    if not numeric_feature_cols or strategy in {"native", "catboost_native", "none"}:
        report["after"] = report["before"]
        return outputs, report

    fill_values = pd.Series(dtype=float)
    if strategy == "median":
        fill_values = outputs[0][numeric_feature_cols].median(numeric_only=True)
        for df in outputs:
            df.loc[:, numeric_feature_cols] = df[numeric_feature_cols].fillna(fill_values)
    elif strategy == "zero":
        fill_values = pd.Series(0.0, index=numeric_feature_cols)
        for df in outputs:
            df.loc[:, numeric_feature_cols] = df[numeric_feature_cols].fillna(fill_values)
    elif strategy == "ffill_median":
        prepared_frames: list[pd.DataFrame] = []
        for source_idx, df in enumerate(outputs):
            temp = df.copy()
            temp["__source_idx"] = source_idx
            temp["__row_id"] = np.arange(len(temp))
            prepared_frames.append(temp)
        combined = pd.concat(prepared_frames, ignore_index=True)
        if entity_col in combined.columns and time_col in combined.columns:
            combined = combined.sort_values([entity_col, time_col, "__source_idx", "__row_id"]).copy()
            combined.loc[:, numeric_feature_cols] = combined.groupby(entity_col, group_keys=False)[numeric_feature_cols].ffill()
        fill_values = (
            combined[combined["__source_idx"] == 0][numeric_feature_cols]
            .median(numeric_only=True)
            .fillna(0.0)
        )
        combined.loc[:, numeric_feature_cols] = combined[numeric_feature_cols].fillna(fill_values)
        rebuilt_outputs: list[pd.DataFrame] = []
        for source_idx in range(len(outputs)):
            restored = (
                combined[combined["__source_idx"] == source_idx]
                .sort_values("__row_id")
                .drop(columns=["__source_idx", "__row_id"])
                .copy()
            )
            rebuilt_outputs.append(restored)
        outputs = rebuilt_outputs
    else:
        raise ValueError(f"Unsupported missing value strategy: {strategy}")

    report["fill_values_preview"] = {
        key: float(value) for key, value in fill_values.head(10).items()
    }
    report["after"] = [missing_feature_summary(df, feature_cols, categorical_cols) for df in outputs]
    return outputs, report


def select_stable_exogenous_features(
    train_df: pd.DataFrame,
    exog_lag_cols: list[str],
    target_col: str,
    top_k: int,
    stability_windows: int,
    min_support: int,
    training_cfg: dict[str, Any] | None = None,
    entity_col: str = "ccaa",
) -> list[str]:
    if not exog_lag_cols or top_k <= 0:
        return []

    work_df = train_df.copy()
    if training_cfg is not None:
        (work_df,), _ = apply_missing_value_strategy(
            dfs=[work_df],
            feature_cols=exog_lag_cols,
            categorical_cols=[],
            training_cfg=training_cfg,
            entity_col=entity_col,
        )

    def score_columns(df: pd.DataFrame) -> dict[str, float]:
        residual = df["y_t_plus_h"] - df[target_col]
        scores: dict[str, float] = {}
        for column in exog_lag_cols:
            series = df[column]
            if series.notna().sum() < 24:
                continue
            if series.nunique(dropna=True) <= 1:
                continue
            corr = series.corr(residual)
            if pd.notna(corr):
                scores[column] = abs(float(corr))
        return scores

    global_scores = score_columns(work_df)
    if not global_scores:
        return []

    ranked_global = sorted(global_scores.items(), key=lambda item: item[1], reverse=True)
    if stability_windows <= 1:
        return [column for column, _ in ranked_global[:top_k]]

    months = sorted(work_df["target_period_dt"].dropna().unique())
    if len(months) < stability_windows:
        return [column for column, _ in ranked_global[:top_k]]

    window_months = np.array_split(np.array(months), stability_windows)
    support: dict[str, int] = {}
    for chunk in window_months:
        if len(chunk) == 0:
            continue
        chunk_df = work_df[work_df["target_period_dt"].isin(chunk)].copy()
        chunk_scores = score_columns(chunk_df)
        if not chunk_scores:
            continue
        ranked_chunk = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)
        selected_chunk = [column for column, _ in ranked_chunk[:top_k]]
        for column in selected_chunk:
            support[column] = support.get(column, 0) + 1

    stable_features = [
        (column, support_count, global_scores[column])
        for column, support_count in support.items()
        if support_count >= min_support and column in global_scores
    ]
    stable_features.sort(key=lambda item: (item[1], item[2]), reverse=True)

    selected = [column for column, _, _ in stable_features[:top_k]]
    if len(selected) < top_k:
        for column, _ in ranked_global:
            if column not in selected:
                selected.append(column)
            if len(selected) >= top_k:
                break
    return selected


def build_model(training_cfg: dict[str, Any]) -> CatBoostRegressor:
    params: dict[str, Any] = {
        "loss_function": training_cfg["loss_function"],
        "iterations": int(training_cfg["catboost_iterations"]),
        "depth": int(training_cfg["catboost_depth"]),
        "learning_rate": float(training_cfg["catboost_learning_rate"]),
        "l2_leaf_reg": float(training_cfg["catboost_l2_leaf_reg"]),
        "random_seed": int(training_cfg["random_state"]),
        "task_type": training_cfg["task_type"],
        "devices": training_cfg["devices"] if training_cfg["task_type"] == "GPU" else None,
        "verbose": False,
        "allow_writing_files": False,
    }
    if training_cfg.get("bootstrap_type"):
        params["bootstrap_type"] = training_cfg["bootstrap_type"]
    if training_cfg.get("random_strength") is not None:
        params["random_strength"] = float(training_cfg["random_strength"])
    return CatBoostRegressor(**params)


def choose_baseline_weights(
    train_df: pd.DataFrame,
    calibration_months: int,
    grid_step: float,
    force_w1: float | None = None,
) -> tuple[float, float, float, str]:
    """`force_w1` fija w1 (persistencia a t-1) sin buscarlo en rejilla.

    Se usa en el experimento "totalmente exogeno" de la Nota 1 (`force_w1=0.0`)
    para que el baseline no dependa en absoluto del ultimo precio conocido de la
    fuente con retraso, quedando como referencia estacional pura (y_persist_12).
    """
    months = sorted(train_df["target_period_dt"].dropna().unique())
    if len(months) <= calibration_months:
        calibration_start = pd.Timestamp(months[0])
    else:
        calibration_start = pd.Timestamp(months[-calibration_months])
    calib_df = train_df[train_df["target_period_dt"] >= calibration_start].copy()

    if force_w1 is not None:
        best_w1 = float(force_w1)
        best_w12 = float(1.0 - best_w1)
        pred = best_w1 * calib_df["y_persist_1"] + best_w12 * calib_df["y_persist_12"]
        best_score = float(mean_absolute_error(calib_df["y_t_plus_h"], pred))
        return best_w1, best_w12, best_score, calibration_start.strftime("%Y-%m-%d")

    best_w1 = 1.0
    best_w12 = 0.0
    best_score = float(mean_absolute_error(calib_df["y_t_plus_h"], calib_df["y_persist_1"]))
    for w1 in np.arange(0.0, 1.0 + 1e-9, grid_step):
        w1 = float(round(w1, 10))
        w12 = float(round(1.0 - w1, 10))
        pred = w1 * calib_df["y_persist_1"] + w12 * calib_df["y_persist_12"]
        score = float(mean_absolute_error(calib_df["y_t_plus_h"], pred))
        if score < best_score - 1e-12:
            best_score = score
            best_w1 = w1
            best_w12 = w12
    return best_w1, best_w12, best_score, calibration_start.strftime("%Y-%m-%d")


def baseline_predict(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    training_cfg: dict[str, Any],
) -> tuple[np.ndarray, float, float, float, str]:
    force_w1 = training_cfg.get("baseline_force_w1")
    w1, w12, calib_mae, calibration_start = choose_baseline_weights(
        train_df=train_df,
        calibration_months=int(training_cfg["baseline_calibration_months"]),
        grid_step=float(training_cfg["weight_grid_step"]),
        force_w1=None if force_w1 is None else float(force_w1),
    )
    pred = w1 * target_df["y_persist_1"].to_numpy() + w12 * target_df["y_persist_12"].to_numpy()
    return pred, w1, w12, calib_mae, calibration_start


def choose_ml_blend_alpha(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    training_cfg: dict[str, Any],
) -> tuple[float, float, str]:
    months = sorted(train_df["target_period_dt"].dropna().unique())
    calibration_months = int(training_cfg["ml_blend_calibration_months"])
    if len(months) <= calibration_months + 12:
        return 1.0, np.nan, ""

    calibration_start = pd.Timestamp(months[-calibration_months])
    subtrain_df = train_df[train_df["target_period_dt"] < calibration_start].copy()
    calib_df = train_df[train_df["target_period_dt"] >= calibration_start].copy()
    if subtrain_df.empty or calib_df.empty:
        return 1.0, np.nan, ""

    calib_baseline, _, _, _, _ = baseline_predict(subtrain_df, calib_df, training_cfg)
    model = build_model(training_cfg)
    model.fit(subtrain_df[feature_cols], subtrain_df["y_t_plus_h"], cat_features=categorical_cols)
    calib_ml = model.predict(calib_df[feature_cols])

    best_alpha = 0.0
    best_score = float(mean_absolute_error(calib_df["y_t_plus_h"], calib_baseline))
    for alpha in parse_float_grid(training_cfg["ml_alpha_grid"]):
        pred = calib_baseline + alpha * (calib_ml - calib_baseline)
        score = float(mean_absolute_error(calib_df["y_t_plus_h"], pred))
        if score < best_score - 1e-12:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha, best_score, calibration_start.strftime("%Y-%m-%d")


def materialize_selected_feature_views(
    dfs: list[pd.DataFrame],
    base_feature_cols: list[str],
    selected_anchor_features: list[str],
    selected_context_features: list[str],
    selected_exog: list[str],
    add_selected_exog_missing_indicators: bool = False,
) -> tuple[list[pd.DataFrame], list[str], list[str]]:
    interaction_cols = [f"{column}__x_share" for column in selected_exog]
    missing_indicator_cols = [f"{column}__is_missing" for column in selected_exog]
    outputs: list[pd.DataFrame] = []
    for df in dfs:
        work = df.copy()
        if add_selected_exog_missing_indicators:
            for column in selected_exog:
                work[f"{column}__is_missing"] = work[column].isna().astype(int)
        for column in selected_exog:
            work[f"{column}__x_share"] = work[column] * work["target_share_vs_spain_lag_12"]
        outputs.append(work)

    feature_cols = (
        list(base_feature_cols)
        + list(selected_anchor_features)
        + list(selected_context_features)
        + list(selected_exog)
        + (missing_indicator_cols if add_selected_exog_missing_indicators else [])
        + interaction_cols
    )
    return outputs, feature_cols, interaction_cols + (
        missing_indicator_cols if add_selected_exog_missing_indicators else []
    )


def evaluate_predictions(
    eval_df: pd.DataFrame,
    entity_col: str,
    y_pred: np.ndarray,
    y_baseline: np.ndarray,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = eval_df[
        ["anchor_product", "context_products", "period", "period_dt", "target_period", "target_period_dt", entity_col]
    ].copy()
    preds = preds.rename(
        columns={
            "period": "feature_period",
            "period_dt": "feature_period_dt",
            "target_period": "period",
            "target_period_dt": "period_dt",
        }
    )
    preds["split_name"] = split_name
    preds["y_true"] = eval_df["y_t_plus_h"].to_numpy()
    preds["y_pred"] = y_pred
    preds["y_baseline_calibrated"] = y_baseline
    preds["y_persist_1"] = eval_df["y_persist_1"].to_numpy()
    preds["y_persist_12"] = eval_df["y_persist_12"].to_numpy()
    preds["abs_error"] = np.abs(preds["y_true"] - preds["y_pred"])
    preds["abs_error_baseline_calibrated"] = np.abs(preds["y_true"] - preds["y_baseline_calibrated"])

    rows: list[dict[str, Any]] = [
        {
            "split_name": split_name,
            "scope": "global",
            "entity": "ALL",
            "n_obs": int(len(preds)),
            "mae": float(mean_absolute_error(preds["y_true"], preds["y_pred"])),
            "smape": float(smape(preds["y_true"], preds["y_pred"])),
            "mae_baseline_calibrated": float(mean_absolute_error(preds["y_true"], preds["y_baseline_calibrated"])),
            "smape_baseline_calibrated": float(smape(preds["y_true"], preds["y_baseline_calibrated"])),
            "mae_vs_persist_1": float(mean_absolute_error(preds["y_true"], preds["y_persist_1"])),
            "smape_vs_persist_1": float(smape(preds["y_true"], preds["y_persist_1"])),
            "mae_vs_persist_12": float(mean_absolute_error(preds["y_true"], preds["y_persist_12"])),
            "smape_vs_persist_12": float(smape(preds["y_true"], preds["y_persist_12"])),
        }
    ]

    for entity, group_df in preds.groupby(entity_col):
        rows.append(
            {
                "split_name": split_name,
                "scope": "entity",
                "entity": entity,
                "n_obs": int(len(group_df)),
                "mae": float(mean_absolute_error(group_df["y_true"], group_df["y_pred"])),
                "smape": float(smape(group_df["y_true"], group_df["y_pred"])),
                "mae_baseline_calibrated": float(
                    mean_absolute_error(group_df["y_true"], group_df["y_baseline_calibrated"])
                ),
                "smape_baseline_calibrated": float(
                    smape(group_df["y_true"], group_df["y_baseline_calibrated"])
                ),
                "mae_vs_persist_1": float(mean_absolute_error(group_df["y_true"], group_df["y_persist_1"])),
                "smape_vs_persist_1": float(smape(group_df["y_true"], group_df["y_persist_1"])),
                "mae_vs_persist_12": float(mean_absolute_error(group_df["y_true"], group_df["y_persist_12"])),
                "smape_vs_persist_12": float(smape(group_df["y_true"], group_df["y_persist_12"])),
            }
        )
    return pd.DataFrame(rows), preds


def rolling_splits(frame: pd.DataFrame, cutoff_year: int | None = None) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    valid_periods = frame["target_period_dt"].dropna().sort_values().drop_duplicates()
    if valid_periods.empty:
        return []
    years = sorted(valid_periods.dt.year.unique().tolist())
    splits: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for year in years:
        if cutoff_year is not None and year < int(cutoff_year):
            continue
        valid_start = pd.Timestamp(f"{year}-01-01")
        valid_end = pd.Timestamp(f"{year}-12-01")
        train_end = pd.Timestamp(f"{year - 1}-12-01")
        if valid_start < valid_periods.min() or valid_end > valid_periods.max():
            continue
        splits.append((train_end, valid_start, valid_end))
    return splits
