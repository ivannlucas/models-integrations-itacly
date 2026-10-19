"""Input handling for ml24: series contract, product filter, as-of cut-off, CCAA and history checks.

The model consumes the long-format MAPA series (series_consumo_lacteos_ccaa.csv schema). The
original validates it with src/training/spatial_price_model.py::validate_inference_input and
re-reads the whole CSV per run; the plugin validates once and keeps only the five products the
panels use. That filter is exact: build_product_panel keeps those products and nothing else.
"""
# pylint: disable=line-too-long  # repo limit is 120 (.flake8), enforced by flake8
from __future__ import annotations

import re

import pandas as pd

from app.plugins.ml24_dairy_spatial_price_forecast.constants import (
    ANCHOR_PRODUCT,
    CCAA_CODE_TO_NAME,
    CONTEXT_PRODUCTS,
    MIN_HISTORY_MONTHS,
    MODEL_PRODUCTS,
    SERIES_REQUIRED_COLUMNS,
    SPAIN_NAME,
    TARGET_METRIC,
)
from app.plugins.ml24_dairy_spatial_price_forecast.exceptions import (
    SeriesContractError,
    UnknownCcaaError,
)

_AS_OF_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_READ_CHUNK_ROWS = 500_000


def _contract_message(problem: str) -> str:
    return (
        f"La serie de entrada no sirve para inferir con el modelo oficial: {problem}. Contrato "
        f"mínimo (memoria, sección 8.4): columnas {list(SERIES_REQUIRED_COLUMNS)}; producto ancla "
        f"'{ANCHOR_PRODUCT}' con la métrica '{TARGET_METRIC}'; productos de contexto "
        f"{list(CONTEXT_PRODUCTS)} (siempre obligatorios); al menos {MIN_HISTORY_MONTHS} meses de "
        "histórico mensual por CCAA. Formato largo de scripts/data_processing.py "
        "(series_consumo_lacteos_ccaa.csv)."
    )


def validate_series(series: pd.DataFrame) -> pd.DataFrame:
    """Check the inference contract and keep only the anchor + context product rows."""
    missing_cols = [c for c in SERIES_REQUIRED_COLUMNS if c not in series.columns]
    if missing_cols:
        raise SeriesContractError(_contract_message(f"faltan columnas obligatorias {missing_cols}"))
    available = set(str(p) for p in series["producto"].dropna().unique())
    problems = []
    missing_products = [p for p in MODEL_PRODUCTS if p not in available]
    if missing_products:
        problems.append(f"faltan los productos requeridos {missing_products}")
    anchor_rows = series[(series["producto"] == ANCHOR_PRODUCT) & (series["metrica"] == TARGET_METRIC)]
    if ANCHOR_PRODUCT in available and anchor_rows.empty:
        problems.append(f"el producto ancla '{ANCHOR_PRODUCT}' no tiene la métrica '{TARGET_METRIC}'")
    if problems:
        raise SeriesContractError(_contract_message("; ".join(problems)))
    return series[series["producto"].isin(MODEL_PRODUCTS)].reset_index(drop=True)


def read_series_csv(path: str) -> pd.DataFrame:
    """Read a long-format series CSV keeping only the model's products (chunked: the official
    file is ~340 MB, of which the five products are a small fraction)."""
    try:
        header = pd.read_csv(path, nrows=0)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise SeriesContractError(f"No se puede leer la serie '{path}': {type(exc).__name__}: {exc}") from exc
    missing_cols = [c for c in SERIES_REQUIRED_COLUMNS if c not in header.columns]
    if missing_cols:
        raise SeriesContractError(_contract_message(f"faltan columnas obligatorias {missing_cols}"))
    products_seen: set[str] = set()
    anchor_has_target = False
    parts = []
    for chunk in pd.read_csv(path, chunksize=_READ_CHUNK_ROWS, low_memory=False):
        products_seen.update(str(p) for p in chunk["producto"].dropna().unique())
        anchor_has_target = anchor_has_target or bool(
            ((chunk["producto"] == ANCHOR_PRODUCT) & (chunk["metrica"] == TARGET_METRIC)).any()
        )
        parts.append(chunk[chunk["producto"].isin(MODEL_PRODUCTS)])
    series = pd.concat(parts, ignore_index=True) if parts else header
    problems = []
    missing_products = [p for p in MODEL_PRODUCTS if p not in products_seen]
    if missing_products:
        problems.append(f"faltan los productos requeridos {missing_products}")
    if ANCHOR_PRODUCT in products_seen and not anchor_has_target:
        problems.append(f"el producto ancla '{ANCHOR_PRODUCT}' no tiene la métrica '{TARGET_METRIC}'")
    if problems:
        raise SeriesContractError(_contract_message("; ".join(problems)))
    return series


def truncate_series(series: pd.DataFrame, as_of_period: str | None) -> pd.DataFrame:
    """Keep the rows up to as_of_period ('YYYY-MM') inclusive — the same as running the original
    predictor with an --input CSV that ends at that month."""
    if not as_of_period:
        return series
    if not _AS_OF_RE.match(as_of_period):
        raise SeriesContractError(f"as_of_period debe tener formato 'YYYY-MM', no '{as_of_period}'.")
    year, month = (int(part) for part in as_of_period.split("-"))
    anio = series["anio"].astype(int)
    mes = series["mes"].astype(int)
    truncated = series[(anio < year) | ((anio == year) & (mes <= month))]
    if truncated.empty:
        raise SeriesContractError(f"La serie no tiene datos anteriores o iguales a {as_of_period}.")
    return truncated.reset_index(drop=True)


def resolve_ccaa(requested: list[str] | None) -> list[str] | None:
    """Translate 3-letter codes (case-insensitive, src/utils/ccaa_codes.py) or official names into
    the names of the ``ccaa`` column. None/empty means every CCAA."""
    if not requested:
        return None
    names = set(CCAA_CODE_TO_NAME.values())
    resolved, unknown = [], []
    for raw in requested:
        value = str(raw).strip().upper()
        name = CCAA_CODE_TO_NAME.get(value, value if value in names else None)
        if name is None or name == SPAIN_NAME:
            unknown.append(raw)
        elif name not in resolved:
            resolved.append(name)
    if unknown:
        valid = ", ".join(code for code, name in CCAA_CODE_TO_NAME.items() if name != SPAIN_NAME)
        raise UnknownCcaaError(
            f"CCAA no reconocida(s) o no servida(s) por el modelo: {unknown}. Códigos válidos: {valid} "
            "(o el nombre oficial de la columna ccaa). T.ESPANA/ESP no se predice: los paneles "
            "oficiales usan include_spain=false."
        )
    return resolved


def ccaa_with_insufficient_history(anchor_panel: pd.DataFrame, target_col: str) -> dict[str, str]:
    """CCAA whose last panel row lacks the 13 consecutive monthly anchor prices lag_12 needs.

    The original predictor takes each CCAA's last panel row and median-imputes whatever is
    missing (y_persist_1 included), so a CCAA with a short or broken history still gets a number.
    The memoria (8.4) promises that such a CCAA is omitted instead; the plugin keeps that promise.
    """
    reasons: dict[str, str] = {}
    for ccaa, group in anchor_panel.sort_values("period_dt").groupby("ccaa"):
        last = pd.Timestamp(group["period_dt"].iloc[-1])
        window = pd.date_range(end=last, periods=MIN_HISTORY_MONTHS, freq="MS")
        prices = group.set_index(pd.to_datetime(group["period_dt"]))[target_col]
        available = prices.reindex(window)
        if available.isna().any():
            reasons[str(ccaa)] = (
                f"faltan {int(available.isna().sum())} de los {MIN_HISTORY_MONTHS} precios mensuales "
                f"del producto ancla hasta {last.strftime('%Y-%m')}"
            )
    return reasons


def read_exog_csv(path: str, current_columns: list[str]) -> pd.DataFrame:
    """Read a new monthly exogenous block (input_costs_monthly.csv schema) for /train.

    Same reader as model_loader.load_reference_data (period_dt parsed). Checked against the
    columns of the exogenous block in use, so a broken ETL output is a 400 before training
    instead of features silently missing (they would be median-imputed) in the new base.
    """
    try:
        exog = pd.read_csv(path, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"exog_path: no se puede leer '{path}': {type(exc).__name__}: {exc}") from exc
    missing = [c for c in current_columns if c not in exog.columns]
    if missing:
        shown = missing[:10] + (["..."] if len(missing) > 10 else [])
        raise ValueError(
            f"exog_path: faltan {len(missing)} columnas de las exógenas mensuales vigentes "
            f"(input_costs_monthly.csv): {shown}"
        )
    exog["period_dt"] = pd.to_datetime(exog["period_dt"], errors="coerce")
    if exog["period_dt"].isna().any():
        raise ValueError("exog_path: la columna period_dt tiene valores vacíos o que no son fechas.")
    return exog
