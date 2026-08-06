from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics_summary import build_metrics_summary
from src.evaluation.policy_metrics import simulate_policy_frame, write_policy_outputs
from src.reproducibility.manifest import build_reproducibility_manifest
from src.reproducibility.runtime import official_paths


def run_reproducibility_policy_simulation(config: dict[str, Any], logger) -> dict[str, Any]:
    paths = official_paths(config)
    predictions_df = pd.read_csv(paths["predictions_latest"])
    period_df, scenario_df, summary = simulate_policy_frame(predictions_df)
    output_paths = write_policy_outputs(config, period_df, scenario_df, summary)
    logger.info("Saved reproducible policy simulation summary to %s", output_paths["summary_json_path"])
    return {
        **output_paths,
        **summary,
    }


def run_reproducibility_get_stats(config: dict[str, Any], logger) -> dict[str, Any]:
    summary = build_metrics_summary(config)
    paths = official_paths(config)
    partial_manifest_path = paths["repro_manifest"].with_name("reproducibility_manifest_partial__mixed_context.json")
    manifest = build_reproducibility_manifest(
        config,
        commands_executed=["python -m src.main get_stats --mixed-context"],
        warnings=[],
        limitations=[
            "External sources are contextual proxies rather than internal plant histories.",
            "The synthetic plant layer remains synthetic unless replaced by customer-provided plant data.",
        ],
        output_path=partial_manifest_path,
        manifest_scope="partial_get_stats",
    )
    logger.info("Saved consolidated metrics summary to %s", paths["metrics_summary_json"])
    return {
        "summary_json_path": str(paths["metrics_summary_json"]),
        "summary_csv_path": str(paths["metrics_summary_csv"]),
        "summary": summary,
        "reproducibility_manifest_path": str(partial_manifest_path),
        "reproducibility_manifest_scope": manifest["scope"],
        "manifest_scope": manifest["manifest_scope"],
    }
