"""Vendored verbatim from inbox/a28/codigo/src/feature_engineering/build_features.py."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    "current_inventory_tons",
    "expected_requirement_tons",
    "lead_time_days",
    "safety_coverage_days",
    "expected_yield_rate",
    "expected_waste_rate",
    "unit_purchase_cost",
    "shelf_life_days",
]


def build_platform_features(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return a typed feature frame for the official CU28 platform."""

    feature_frame = df.copy()
    feature_frame["date"] = pd.to_datetime(feature_frame["date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        if column in feature_frame.columns:
            feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")

    feature_frame = feature_frame.sort_values(["raw_material_id", "date"]).reset_index(drop=True)
    feature_frame["effective_supply_rate"] = (
        feature_frame["expected_yield_rate"] * (1.0 - feature_frame["expected_waste_rate"])
    ).clip(lower=0.05, upper=1.0)
    feature_frame["expected_requirement_with_waste_tons"] = (
        feature_frame["expected_requirement_tons"] * (1.0 + feature_frame["expected_waste_rate"])
    )
    feature_frame["projected_stock_after_lead_time_tons"] = (
        feature_frame["current_inventory_tons"]
        - feature_frame["expected_requirement_tons"] * feature_frame["lead_time_days"] / 7.0
    )
    feature_frame["safety_stock_tons"] = (
        feature_frame["expected_requirement_tons"] * feature_frame["safety_coverage_days"] / 7.0
    )
    feature_frame["inventory_gap_to_safety_tons"] = (
        feature_frame["safety_stock_tons"] - feature_frame["projected_stock_after_lead_time_tons"]
    )
    feature_frame["replenishment_gap_tons"] = np.maximum(
        0.0,
        feature_frame["safety_stock_tons"] + feature_frame["expected_requirement_tons"] - feature_frame["current_inventory_tons"],
    )
    return feature_frame
