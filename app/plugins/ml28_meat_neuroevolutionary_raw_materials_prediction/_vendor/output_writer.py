"""Vendored (subset) from inbox/a28/codigo/src/reporting/output_writer.py.

Only _ensure_explainability_columns and RECOMMENDATION_COLUMNS/SIMULATION_COLUMNS are vendored —
write_platform_outputs (file I/O to a local outputs/ dir) is not used, the plugin returns the
frame directly instead of writing CSV/JSON files to disk.
"""
from __future__ import annotations

import pandas as pd

RECOMMENDATION_COLUMNS = [
    "date",
    "raw_material_id",
    "destination_profile",
    "current_inventory_tons",
    "expected_requirement_tons",
    "lead_time_days",
    "safety_coverage_days",
    "expected_yield_rate",
    "expected_waste_rate",
    "unit_purchase_cost",
    "shelf_life_days",
    "purchase_trigger_proba",
    "purchase_trigger_flag",
    "recommended_action",
    "quantity_optimizer_recommendation_tons",
    "order_quantity_tons",
    "decision_reason",
    "projected_stock_after_lead_time_tons",
    "safety_stock_tons",
    "coverage_gap_tons",
    "risk_level",
    "baseline_order_quantity_tons",
    "delta_order_vs_baseline_tons",
    "excess_tons",
    "stockout_tons",
]

SIMULATION_COLUMNS = RECOMMENDATION_COLUMNS + [
    "baseline_excess_tons",
    "baseline_stockout_tons",
]


def _ensure_explainability_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add business-readable decision fields to the platform output frame.

    Risk rule:
    - HIGH when simulated stockout remains positive after the recommendation,
      or when a positive coverage gap has trigger probability >= 0.75.
    - MEDIUM when a purchase is triggered, a coverage gap exists, or trigger
      probability is >= 0.50.
    - LOW otherwise.
    """

    enriched = frame.copy()
    if "coverage_gap_tons" not in enriched.columns:
        projected_stock = pd.to_numeric(enriched.get("projected_stock_after_lead_time_tons"), errors="coerce")
        safety_stock = pd.to_numeric(enriched.get("safety_stock_tons"), errors="coerce")
        enriched["coverage_gap_tons"] = (safety_stock - projected_stock).clip(lower=0.0)

    flag = pd.to_numeric(enriched["purchase_trigger_flag"], errors="coerce").fillna(0).astype(int)
    enriched["recommended_action"] = flag.map({1: "BUY", 0: "DO_NOT_BUY"}).fillna("DO_NOT_BUY")

    enriched["decision_reason"] = "Coverage remains above safety threshold; purchase blocked."
    positive_gap = pd.to_numeric(enriched["coverage_gap_tons"], errors="coerce").fillna(0.0) > 0.0
    buy_mask = flag.eq(1)
    enriched.loc[buy_mask, "decision_reason"] = "Purchase triggered and quantity optimized under current policy."
    enriched.loc[buy_mask & positive_gap, "decision_reason"] = (
        "Projected stock after lead time is below safety stock. "
        "Purchase triggered and quantity optimized under current policy."
    )

    if {"order_quantity_tons", "baseline_order_quantity_tons"}.issubset(enriched.columns):
        order = pd.to_numeric(enriched["order_quantity_tons"], errors="coerce")
        baseline = pd.to_numeric(enriched["baseline_order_quantity_tons"], errors="coerce")
        enriched["delta_order_vs_baseline_tons"] = order - baseline
    elif "delta_order_vs_baseline_tons" not in enriched.columns:
        enriched["delta_order_vs_baseline_tons"] = pd.NA

    probability = pd.to_numeric(enriched["purchase_trigger_proba"], errors="coerce").fillna(0.0)
    stockout = pd.to_numeric(enriched.get("stockout_tons"), errors="coerce").fillna(0.0)
    gap = pd.to_numeric(enriched["coverage_gap_tons"], errors="coerce").fillna(0.0)
    enriched["risk_level"] = "LOW"
    enriched.loc[buy_mask | gap.gt(0.0) | probability.ge(0.50), "risk_level"] = "MEDIUM"
    enriched.loc[stockout.gt(0.0) | (gap.gt(0.0) & probability.ge(0.75)), "risk_level"] = "HIGH"

    numeric_columns = [
        "quantity_optimizer_recommendation_tons",
        "order_quantity_tons",
        "projected_stock_after_lead_time_tons",
        "safety_stock_tons",
        "coverage_gap_tons",
        "baseline_order_quantity_tons",
        "delta_order_vs_baseline_tons",
        "excess_tons",
        "stockout_tons",
    ]
    for column in numeric_columns:
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce").round(3)
    return enriched
