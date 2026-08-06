from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SHELF_LIFE_DAYS = {"short": 7.0, "medium": 14.0, "long": 28.0}
UPSTREAM_TARGET = "synthetic_procurement_need"
TRIGGER_TARGET = "purchase_trigger_label"
QUANTITY_TARGET = "quantity_optimizer_target_tons"
PROHIBITED_UPSTREAM_INPUTS = [
    "order_quantity_tons",
    "quantity_optimizer_recommendation_tons",
    "quantity_optimizer_target_tons",
    "excess_tons",
    "stockout_tons",
    "purchase_trigger_flag",
    "purchase_trigger_proba",
]
PROHIBITED_QUANTITY_INPUTS = [
    *PROHIBITED_UPSTREAM_INPUTS,
    "purchase_trigger_label",
]
NEUTRAL_FEATURE_FILL_VALUES = {
    "supply_index": 100.0,
    "demand_supply_gap": 0.0,
    "demand_supply_ratio": 1.0,
}


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _series_or_default(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        return _safe_numeric(df[column]).fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _default_fill_value(column: str) -> float:
    if column.startswith("supply_index_"):
        return NEUTRAL_FEATURE_FILL_VALUES["supply_index"]
    return NEUTRAL_FEATURE_FILL_VALUES.get(column, 0.0)


def fit_feature_fill_values(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    fill_values: dict[str, float] = {}
    for column in feature_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            fill_values[column] = _default_fill_value(column)
        else:
            fill_values[column] = float(valid.median())
    return fill_values


def apply_feature_fill_values(
    frame: pd.DataFrame,
    feature_columns: list[str],
    fill_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction input is missing trained features: {missing}")

    resolved_fill = dict(fill_values or {})
    x = frame[feature_columns].copy()
    for column in feature_columns:
        fallback = _default_fill_value(column)
        x[column] = pd.to_numeric(x[column], errors="coerce").fillna(float(resolved_fill.get(column, fallback)))
    return x


def prohibited_features_for_stage(stage: str) -> list[str]:
    if stage == "quantity_optimizer":
        return list(dict.fromkeys(PROHIBITED_QUANTITY_INPUTS))
    return list(dict.fromkeys(PROHIBITED_UPSTREAM_INPUTS))


def validate_feature_columns_for_stage(feature_columns: list[str], *, stage: str) -> None:
    prohibited = set(prohibited_features_for_stage(stage))
    violations = sorted(prohibited.intersection(feature_columns))
    if violations:
        raise ValueError(f"Prohibited features found for stage={stage}: {violations}")


def derive_official_columns(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    enriched = df.copy()
    safety_stock_days = float(
        config.get("synthetic_data", {})
        .get("simulation_parameters", {})
        .get("safety_stock_days", 6.0)
    )

    if "destination_profile" not in enriched.columns:
        if "manufacturing_context_profile" in enriched.columns:
            enriched["destination_profile"] = enriched["manufacturing_context_profile"].astype(str)
        else:
            enriched["destination_profile"] = "mixed_context"
    if "product_family" in enriched.columns and "product_family_alias" not in enriched.columns:
        enriched["product_family_alias"] = enriched["product_family"].astype(str)

    enriched["raw_material_id"] = (
        enriched.get("product_family", enriched["destination_profile"])
        .astype(str)
        .str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
        .str.strip("_")
        .str.lower()
        .radd("rm_")
    )
    enriched["current_inventory_tons"] = _series_or_default(enriched, "synthetic_inventory_level", 0.0).clip(lower=0.0)
    enriched["expected_requirement_tons"] = _series_or_default(enriched, "synthetic_raw_material_requirement", 0.0).clip(lower=0.0)
    enriched["lead_time_days"] = _series_or_default(enriched, "synthetic_lead_time_days", 0.0).clip(lower=0.0)
    enriched["safety_coverage_days"] = _series_or_default(enriched, "process_lead_time_days", safety_stock_days) * 0.0 + safety_stock_days
    enriched["expected_yield_rate"] = _series_or_default(
        enriched,
        "expected_yield" if "expected_yield" in enriched.columns else "synthetic_yield_rate",
        0.85,
    ).clip(lower=0.5, upper=1.0)
    enriched["expected_waste_rate"] = _series_or_default(
        enriched,
        "expected_waste" if "expected_waste" in enriched.columns else "synthetic_waste_rate",
        0.03,
    ).clip(lower=0.0, upper=0.4)
    enriched["unit_purchase_cost"] = (_series_or_default(enriched, "purchase_price_index", 100.0) / 100.0 * 1000.0).clip(lower=0.0)
    enriched["shelf_life_days"] = (
        enriched.get("shelf_life_class", "medium")
        .astype(str)
        .map(SHELF_LIFE_DAYS)
        .fillna(SHELF_LIFE_DAYS["medium"])
        .astype(float)
    )

    projected_stock = enriched["current_inventory_tons"] - (
        enriched["expected_requirement_tons"] * enriched["lead_time_days"] / 7.0
    )
    safety_stock_tons = enriched["expected_requirement_tons"] * enriched["safety_coverage_days"] / 7.0
    gap_tons = safety_stock_tons - projected_stock
    relative_gap = gap_tons / np.maximum(safety_stock_tons.abs(), 1.0)

    enriched["projected_stock_after_lead_time_tons"] = projected_stock
    enriched["safety_stock_tons"] = safety_stock_tons.clip(lower=0.0)
    enriched["purchase_trigger_gap_tons"] = gap_tons
    enriched["purchase_trigger_label"] = (gap_tons > 0).astype(int)
    enriched["purchase_trigger_proba_heuristic"] = (1.0 / (1.0 + np.exp(-relative_gap))).clip(0.0, 1.0)

    replenishment_gap = np.maximum(
        0.0,
        enriched["safety_stock_tons"] + enriched["expected_requirement_tons"] - enriched["current_inventory_tons"],
    )
    effective_yield = np.maximum(
        enriched["expected_yield_rate"] * (1.0 - enriched["expected_waste_rate"]),
        0.45,
    )
    adjustment_factor = (1.0 / effective_yield).clip(lower=1.0, upper=1.35)
    quantity_target = replenishment_gap * adjustment_factor
    baseline_order = replenishment_gap * 1.24

    enriched["replenishment_gap_tons"] = replenishment_gap
    enriched["quantity_optimizer_target_tons"] = quantity_target.clip(lower=0.0)
    enriched["baseline_order_quantity_tons"] = baseline_order.clip(lower=0.0)
    return enriched


def trigger_feature_columns(df: pd.DataFrame) -> list[str]:
    base = [
        "current_inventory_tons",
        "expected_requirement_tons",
        "lead_time_days",
        "safety_coverage_days",
        "expected_yield_rate",
        "expected_waste_rate",
        "demand_index",
        "supply_index",
        "purchase_price_index",
        "demand_supply_gap",
    ]
    context_dummies = sorted(
        [
            column
            for column in df.columns
            if column.startswith(("manufacturing_context_profile__", "product_family__", "recipe_profile__", "shelf_life_class__"))
        ]
    )
    return [column for column in [*base, *context_dummies] if column in df.columns]


def quantity_feature_columns(df: pd.DataFrame) -> list[str]:
    base = [
        "purchase_trigger_proba_heuristic",
        "current_inventory_tons",
        "expected_requirement_tons",
        "lead_time_days",
        "safety_coverage_days",
        "expected_yield_rate",
        "expected_waste_rate",
        "replenishment_gap_tons",
        "demand_index",
        "supply_index",
        "purchase_price_index",
        "demand_supply_gap",
    ]
    context_dummies = sorted(
        [
            column
            for column in df.columns
            if column.startswith(("manufacturing_context_profile__", "product_family__", "recipe_profile__", "shelf_life_class__"))
        ]
    )
    return [column for column in [*base, *context_dummies] if column in df.columns]


def leakage_audit(df: pd.DataFrame, feature_columns: list[str], *, stage: str) -> pd.DataFrame:
    prohibited = prohibited_features_for_stage(stage)
    rows = []
    active_features = set(feature_columns)
    for column in prohibited:
        rows.append(
            {
                "stage": stage,
                "feature_name": column,
                "present_in_dataset": column in df.columns,
                "present_in_feature_set": column in active_features,
                "status": "fail" if column in active_features else "pass",
                "reason": "prohibited_downstream_or_decision_signal",
            }
        )
    return pd.DataFrame(rows)


def modeling_summary(df: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.Series(dtype="datetime64[ns]")
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "date_min": str(dates.min().date()) if not dates.empty and dates.notna().any() else None,
        "date_max": str(dates.max().date()) if not dates.empty and dates.notna().any() else None,
    }
