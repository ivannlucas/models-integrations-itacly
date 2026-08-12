from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.reproducibility.runtime import official_paths
from src.utils import read_json, write_json


def _reference_regression_run(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": record.get("run_id"),
        "model_family": record.get("model_family"),
        "feature_set": record.get("feature_set"),
        "target": record.get("target_column"),
        "validation": {
            "split": "validation",
            "model": record.get("model_name") or record.get("model_family"),
            "target": record.get("target_column"),
            "n_samples": record.get("validation_n_samples") or record.get("validation_rows"),
            "mae": record.get("validation_mae"),
            "rmse": record.get("validation_rmse"),
            "r2": record.get("validation_r2"),
        },
        "test": {
            "split": "test",
            "model": record.get("model_name") or record.get("model_family"),
            "target": record.get("target_column"),
            "n_samples": record.get("test_n_samples") or record.get("test_rows"),
            "mae": record.get("test_mae"),
            "rmse": record.get("test_rmse"),
            "r2": record.get("test_r2"),
        },
        # Backward-compatible aliases used by generated notebooks.
        "validation_rmse": record.get("validation_rmse"),
        "test_rmse": record.get("test_rmse"),
    }


def build_metrics_summary(config: dict[str, Any]) -> dict[str, Any]:
    paths = official_paths(config)

    baseline = read_json(paths["baseline_summary_json"]) if paths["baseline_summary_json"].exists() else {}
    neuro = read_json(paths["neuro_summary_json"]) if paths["neuro_summary_json"].exists() else {}
    trigger = read_json(paths["trigger_metrics_json"]) if paths["trigger_metrics_json"].exists() else {}
    quantity = read_json(paths["quantity_optimizer_metrics_json"]) if paths["quantity_optimizer_metrics_json"].exists() else {}
    quantity_baseline_comparison = (
        read_json(paths["quantity_optimizer_baseline_comparison_json"])
        if paths["quantity_optimizer_baseline_comparison_json"].exists()
        else {}
    )
    policy = read_json(paths["policy_simulation_summary_json"]) if paths["policy_simulation_summary_json"].exists() else {}

    best_baseline = dict(baseline.get("best_baseline_run") or {})
    best_neuro = dict(neuro.get("best_neuroevolution_run") or {})
    summary = {
        "scope": "mixed_context",
        "selection_criterion": "validation_rmse",
        "upstream": {
            "baseline_reference_run": _reference_regression_run(best_baseline),
            "neuroevolution_reference_run": _reference_regression_run(best_neuro),
            "recommendation": neuro.get("recommendation"),
        },
        "trigger": trigger,
        "quantity_optimizer": quantity,
        "quantity_optimizer_baseline_comparison": quantity_baseline_comparison,
        "policy_simulation": policy,
        "methodology_warnings": [
            "External sources are contextual proxies, not internal plant history.",
            "Synthetic plant variables remain synthetic unless replaced by customer-provided inputs.",
            "Selection of the upstream reference model is based on validation metrics, not test.",
            "order_quantity_tons is a calculated recommendation output, not an observed purchase record.",
        ],
    }
    write_json(paths["metrics_summary_json"], summary)

    flat_rows = [
        {"section": "upstream_baseline", "metric": "validation_rmse", "value": best_baseline.get("validation_rmse")},
        {"section": "upstream_baseline", "metric": "test_rmse", "value": best_baseline.get("test_rmse")},
        {"section": "upstream_neuroevolution", "metric": "validation_rmse", "value": best_neuro.get("validation_rmse")},
        {"section": "upstream_neuroevolution", "metric": "test_rmse", "value": best_neuro.get("test_rmse")},
        {"section": "trigger", "metric": "accuracy", "value": trigger.get("test", {}).get("accuracy")},
        {"section": "trigger", "metric": "false_negative_rate", "value": trigger.get("test", {}).get("false_negative_rate")},
        {"section": "quantity_optimizer", "metric": "test_rmse", "value": quantity.get("test", {}).get("rmse")},
        {"section": "quantity_optimizer", "metric": "test_mae", "value": quantity.get("test", {}).get("mae")},
        {"section": "policy_simulation", "metric": "aggregate_excess_reduction_pct", "value": policy.get("aggregate_excess_reduction_pct")},
        {"section": "policy_simulation", "metric": "aggregate_stockout_change_pct", "value": policy.get("aggregate_stockout_change_pct")},
    ]
    for row in quantity_baseline_comparison.get("metrics", []):
        model = str(row.get("model", "")).lower()
        split = str(row.get("split", "")).lower()
        for metric_name in ["rmse", "mae", "r2", "n_rows"]:
            flat_rows.append(
                {
                    "section": "quantity_optimizer_baseline_comparison",
                    "metric": f"{model}_{split}_{metric_name}",
                    "value": row.get(metric_name),
                }
            )
    pd.DataFrame(flat_rows).to_csv(paths["metrics_summary_csv"], index=False)
    return summary
