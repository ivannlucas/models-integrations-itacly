"""Persist platform outputs to the configured output directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.validation import write_validation_report

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


def write_platform_outputs(
    recommendations_frame: pd.DataFrame,
    simulation_frame: pd.DataFrame,
    summary_metrics: dict[str, Any],
    validation_report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write the official CU28 platform output bundle."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    recommendations_path = output_path / "recommendations.csv"
    simulation_path = output_path / "policy_simulation_results.csv"
    summary_path = output_path / "summary_metrics.json"
    validation_path = output_path / "validation_report.json"

    recommendations_frame = _ensure_explainability_columns(simulation_frame.copy())
    recommendations_frame["date"] = recommendations_frame["date"].dt.strftime("%Y-%m-%d")
    recommendations_frame[RECOMMENDATION_COLUMNS].to_csv(recommendations_path, index=False)

    simulation_frame = _ensure_explainability_columns(simulation_frame.copy())
    simulation_frame["date"] = simulation_frame["date"].dt.strftime("%Y-%m-%d")
    simulation_frame[SIMULATION_COLUMNS].to_csv(simulation_path, index=False)

    summary_path.write_text(json.dumps(summary_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_validation_report(validation_report, validation_path)

    return {
        "recommendations": str(recommendations_path),
        "policy_simulation_results": str(simulation_path),
        "summary_metrics": str(summary_path),
        "validation_report": str(validation_path),
    }
