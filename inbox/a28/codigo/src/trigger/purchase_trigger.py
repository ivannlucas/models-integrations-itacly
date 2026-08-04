"""Fallback purchase trigger for the batch/offline platform."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run_purchase_trigger(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Compute purchase trigger probability and decision flag."""

    platform_config = (config or {}).get("platform", {})
    heuristic_config = platform_config.get("heuristics", {})
    score_scale = float(heuristic_config.get("purchase_trigger_gap_sigmoid_scale", 5.0))

    triggered = df.copy()
    relative_gap = triggered["inventory_gap_to_safety_tons"] / np.maximum(
        triggered["safety_stock_tons"] + triggered["expected_requirement_tons"],
        1.0,
    )
    probability = 1.0 / (1.0 + np.exp(-score_scale * relative_gap))

    triggered["purchase_trigger_proba"] = np.clip(probability, 0.01, 0.99).round(4)
    triggered["purchase_trigger_flag"] = (
        triggered["projected_stock_after_lead_time_tons"] < triggered["safety_stock_tons"]
    ).astype(int)
    return triggered
