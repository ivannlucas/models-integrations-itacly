"""Reproducible policy simulation for the official CU28 platform."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_policy_metrics


def run_policy_simulation(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate the recommended policy against a simple baseline."""

    platform_config = (config or {}).get("platform", {})
    simulation_config = platform_config.get("simulation", {})
    guardrails = platform_config.get("guardrails", {})
    rounding_decimals = int(platform_config.get("rounding_decimals", 3))
    baseline_safety_factor = float(simulation_config.get("baseline_safety_stock_factor", 0.85))
    max_stockout_increase_pct = float(guardrails.get("max_stockout_increase_pct", 5.0))

    simulation_frame = df.copy()
    baseline_gap = np.maximum(
        0.0,
        simulation_frame["expected_requirement_tons"]
        + simulation_frame["safety_stock_tons"] * baseline_safety_factor
        - simulation_frame["current_inventory_tons"],
    )
    simulation_frame["baseline_order_quantity_tons"] = (
        baseline_gap / simulation_frame["effective_supply_rate"]
    ).round(rounding_decimals)

    realized_requirement = simulation_frame["expected_requirement_tons"] * (1.0 + simulation_frame["expected_waste_rate"])
    simulation_frame["excess_tons"] = np.maximum(
        0.0,
        simulation_frame["current_inventory_tons"] + simulation_frame["order_quantity_tons"] - realized_requirement,
    ).round(rounding_decimals)
    simulation_frame["stockout_tons"] = np.maximum(
        0.0,
        realized_requirement - (simulation_frame["current_inventory_tons"] + simulation_frame["order_quantity_tons"]),
    ).round(rounding_decimals)
    simulation_frame["baseline_excess_tons"] = np.maximum(
        0.0,
        simulation_frame["current_inventory_tons"] + simulation_frame["baseline_order_quantity_tons"] - realized_requirement,
    ).round(rounding_decimals)
    simulation_frame["baseline_stockout_tons"] = np.maximum(
        0.0,
        realized_requirement - (simulation_frame["current_inventory_tons"] + simulation_frame["baseline_order_quantity_tons"]),
    ).round(rounding_decimals)

    policy_metrics = compute_policy_metrics(
        simulation_frame,
        guardrail_name="platform_max_stockout_increase",
        allowed_stockout_increase_pct=max_stockout_increase_pct,
    )

    summary_metrics = {
        "row_count": int(len(simulation_frame)),
        "triggered_orders": int(simulation_frame["purchase_trigger_flag"].sum()),
        "total_order_quantity_tons": round(float(simulation_frame["order_quantity_tons"].sum()), rounding_decimals),
        "total_excess_tons": round(policy_metrics["policy_excess_tons"], rounding_decimals),
        "total_stockout_tons": round(policy_metrics["policy_stockout_tons"], rounding_decimals),
        "baseline_total_excess_tons": round(policy_metrics["baseline_excess_tons"], rounding_decimals),
        "baseline_total_stockout_tons": round(policy_metrics["baseline_stockout_tons"], rounding_decimals),
        "absolute_excess_reduction_tons": round(
            policy_metrics["absolute_excess_reduction_tons"],
            rounding_decimals,
        ),
        "aggregate_excess_reduction_pct": round(policy_metrics["aggregate_excess_reduction_pct"], 3),
        "aggregate_stockout_change_pct": round(policy_metrics["aggregate_stockout_change_pct"], 3),
        "guardrail": policy_metrics["guardrail"],
        "stockout_guardrail_pass": policy_metrics["stockout_guardrail_pass"],
        "n_periods": policy_metrics["n_periods"],
    }
    return simulation_frame, summary_metrics
