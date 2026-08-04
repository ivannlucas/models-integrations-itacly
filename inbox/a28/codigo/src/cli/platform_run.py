"""Batch/offline platform orchestration for CU28."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.feature_engineering import build_platform_features
from src.optimizer import run_quantity_optimizer
from src.reporting import write_platform_outputs
from src.simulation import run_policy_simulation
from src.trigger import run_purchase_trigger
from src.validation import validate_input_file, write_validation_report

DEFAULT_PLATFORM_CONFIG: dict[str, Any] = {
    "platform": {
        "name": "cu28_batch_offline_platform",
        "mode": "batch_offline",
        "rounding_decimals": 3,
        "heuristics": {"purchase_trigger_gap_sigmoid_scale": 5.0},
        "simulation": {"baseline_safety_stock_factor": 1.25},
        "guardrails": {"max_stockout_increase_pct": 5.0},
    },
    "destination_profiles": {},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_platform_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_PLATFORM_CONFIG)
    path = Path(config_path) if config_path else Path("config/platform_config.yaml")
    if not path.exists():
        return config

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _deep_merge(config, loaded)


def run_platform_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    platform_config = load_platform_config(config_path)
    input_frame, validation_report = validate_input_file(input_path)
    validation_report_path = output_dir / "validation_report.json"

    if not validation_report["valid"]:
        write_validation_report(validation_report, validation_report_path)
        return {
            "status": "invalid_input",
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "validation_report_path": str(validation_report_path),
            "validation_report": validation_report,
        }

    feature_frame = build_platform_features(input_frame, config=platform_config)
    trigger_frame = run_purchase_trigger(feature_frame, config=platform_config)
    optimized_frame = run_quantity_optimizer(trigger_frame, config=platform_config)
    simulation_frame, summary_metrics = run_policy_simulation(optimized_frame, config=platform_config)

    output_paths = write_platform_outputs(
        recommendations_frame=optimized_frame,
        simulation_frame=simulation_frame,
        summary_metrics=summary_metrics,
        validation_report=validation_report,
        output_dir=output_dir,
    )
    return {
        "status": "success",
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "validation_report_path": output_paths["validation_report"],
        "validation_report": validation_report,
        "summary": summary_metrics,
        "output_paths": output_paths,
    }
