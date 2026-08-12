from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.cli import run_platform_pipeline


def test_output_contract_contains_official_columns_and_metrics(tmp_path: Path) -> None:
    result = run_platform_pipeline(
        input_path="data/demo/customer_upload_example.csv",
        output_dir=tmp_path / "contract_run",
    )

    assert result["status"] == "success"

    recommendations = pd.read_csv(Path(result["output_paths"]["recommendations"]))
    summary_metrics = json.loads(Path(result["output_paths"]["summary_metrics"]).read_text(encoding="utf-8"))

    expected_columns = {
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
    }
    assert expected_columns.issubset(recommendations.columns)

    expected_metric_keys = {
        "aggregate_excess_reduction_pct",
        "aggregate_stockout_change_pct",
        "stockout_guardrail_pass",
    }
    assert expected_metric_keys.issubset(summary_metrics.keys())
