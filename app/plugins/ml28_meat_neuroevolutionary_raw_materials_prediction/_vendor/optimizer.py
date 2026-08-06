"""Vendored verbatim from inbox/a28/codigo/src/optimizer/quantity_optimizer.py."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run_quantity_optimizer(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Apply the quantity recommendation rule and trigger gating."""

    platform_config = (config or {}).get("platform", {})
    rounding_decimals = int(platform_config.get("rounding_decimals", 3))

    optimized = df.copy()
    recommendation = optimized["replenishment_gap_tons"] / optimized["effective_supply_rate"]
    recommendation = np.maximum(0.0, recommendation)
    optimized["quantity_optimizer_recommendation_tons"] = recommendation.round(rounding_decimals)
    optimized["order_quantity_tons"] = np.where(
        optimized["purchase_trigger_flag"].eq(1),
        optimized["quantity_optimizer_recommendation_tons"],
        0.0,
    ).round(rounding_decimals)
    return optimized
