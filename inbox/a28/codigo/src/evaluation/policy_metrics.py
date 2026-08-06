from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics import compute_policy_metrics
from src.reproducibility.runtime import official_paths
from src.utils import ensure_directory, write_json


def simulate_policy_frame(
    predictions_df: pd.DataFrame,
    *,
    allowed_stockout_increase_pct: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = predictions_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    required = [
        "current_inventory_tons",
        "expected_requirement_tons",
        "safety_coverage_days",
        "order_quantity_tons",
        "baseline_order_quantity_tons",
        "purchase_trigger_flag",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Prediction frame is missing policy simulation columns: {missing}")

    df["safety_stock_tons"] = pd.to_numeric(df["expected_requirement_tons"], errors="coerce") * pd.to_numeric(
        df["safety_coverage_days"], errors="coerce"
    ) / 7.0
    df["baseline_inventory_after_order_tons"] = pd.to_numeric(df["current_inventory_tons"], errors="coerce") + pd.to_numeric(
        df["baseline_order_quantity_tons"], errors="coerce"
    )
    df["policy_inventory_after_order_tons"] = pd.to_numeric(df["current_inventory_tons"], errors="coerce") + pd.to_numeric(
        df["order_quantity_tons"], errors="coerce"
    )
    df["baseline_end_inventory_tons"] = df["baseline_inventory_after_order_tons"] - pd.to_numeric(
        df["expected_requirement_tons"], errors="coerce"
    )
    df["policy_end_inventory_tons"] = df["policy_inventory_after_order_tons"] - pd.to_numeric(
        df["expected_requirement_tons"], errors="coerce"
    )
    df["baseline_excess_tons"] = (df["baseline_end_inventory_tons"] - df["safety_stock_tons"]).clip(lower=0.0)
    df["excess_tons"] = (df["policy_end_inventory_tons"] - df["safety_stock_tons"]).clip(lower=0.0)
    df["baseline_stockout_tons"] = (-df["baseline_end_inventory_tons"]).clip(lower=0.0)
    df["stockout_tons"] = (-df["policy_end_inventory_tons"]).clip(lower=0.0)
    df["trigger_rule_respected"] = (
        (pd.to_numeric(df["purchase_trigger_flag"], errors="coerce").fillna(0).astype(int) == 1)
        | (pd.to_numeric(df["order_quantity_tons"], errors="coerce").fillna(0.0) == 0.0)
    )

    scenario_key = "destination_profile" if "destination_profile" in df.columns else "raw_material_id"
    if scenario_key not in df.columns:
        scenario_key = None

    scenario_rows = []
    if scenario_key:
        for scenario_name, frame in df.groupby(scenario_key):
            scenario_rows.append(
                {
                    "scenario_name": str(scenario_name),
                    "rows": int(len(frame)),
                    "baseline_excess_tons": float(frame["baseline_excess_tons"].sum()),
                    "policy_excess_tons": float(frame["excess_tons"].sum()),
                    "baseline_stockout_tons": float(frame["baseline_stockout_tons"].sum()),
                    "policy_stockout_tons": float(frame["stockout_tons"].sum()),
                }
            )
    scenario_df = pd.DataFrame(scenario_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **compute_policy_metrics(
            df,
            guardrail_name="bounded_stockout_increase",
            allowed_stockout_increase_pct=allowed_stockout_increase_pct,
        ),
        "trigger_rule_respected": bool(df["trigger_rule_respected"].all()),
        "baseline_policy_name": "operational_simple",
        "proposed_policy_name": "two_stage_mixed_context",
    }
    return df, scenario_df, summary


def write_policy_outputs(config: dict[str, Any], period_df: pd.DataFrame, scenario_df: pd.DataFrame, summary: dict[str, Any]) -> dict[str, str]:
    paths = official_paths(config)
    ensure_directory(paths["metrics_dir"])
    ensure_directory(paths["summary_dir"])
    ensure_directory(paths["official_metrics_dir"])
    summary_csv = paths["policy_simulation_summary_csv"]
    summary_json = paths["policy_simulation_summary_json"]
    official_period_csv = paths["policy_simulation_period_csv"]
    official_scenario_csv = paths["policy_simulation_scenario_csv"]

    period_df.to_csv(official_period_csv, index=False)
    scenario_df.to_csv(official_scenario_csv, index=False)
    repo_root = Path(config["project"]["repo_root"])
    summary["period_metrics_csv_path"] = official_period_csv.relative_to(repo_root).as_posix()
    summary["scenario_metrics_csv_path"] = official_scenario_csv.relative_to(repo_root).as_posix()
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    write_json(summary_json, summary)

    return {
        "summary_csv_path": str(summary_csv),
        "summary_json_path": str(summary_json),
        "period_metrics_csv_path": str(official_period_csv),
        "scenario_metrics_csv_path": str(official_scenario_csv),
    }
