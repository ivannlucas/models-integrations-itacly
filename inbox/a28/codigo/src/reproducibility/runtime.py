from __future__ import annotations

import importlib
import platform
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.utils import build_recipe_runtime_configs, find_repo_root, load_config, resolve_repo_path

DEFAULT_CONFIG_PATH = "config/config.yaml"
SCOPE = "mixed_context"
SMOKE_MODE = "smoke"
FULL_MODE = "full"
CACHED_MODE = "cached"
SUPPORTED_MODES = {SMOKE_MODE, FULL_MODE, CACHED_MODE}


def repo_root() -> Path:
    return find_repo_root(Path.cwd())


def require_official_end_to_end_run(config: dict[str, Any], *, stage: str) -> dict[str, Any]:
    """Reject partial commands that would overwrite official `latest` artifacts."""

    official_run = config.get("runtime", {}).get("official_run", {})
    if not official_run.get("publish_latest") or official_run.get("kind") != "end_to_end":
        raise RuntimeError(
            f"Stage '{stage}' cannot publish official mixed_context latest artifacts from a partial run. "
            "Use scripts/reproduce_mixed_context.py for the end-to-end official route."
        )
    return dict(official_run)


def ensure_optional_dependency(module_name: str, *, repo_root_path: Path | None = None):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        active_repo_root = repo_root_path or repo_root()
        candidate_paths = [
            active_repo_root / "venv" / "Lib" / "site-packages",
            active_repo_root / ".venv" / "Lib" / "site-packages",
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                sys.path.insert(0, str(candidate))
        return importlib.import_module(module_name)


def _base_reproducibility_overrides(config: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(config)

    updated.setdefault("paths", {})
    updated["paths"]["input_data_path"] = "data/processed/external/context/context_weekly_for_simulation.csv"
    updated["paths"]["processed_dataset_path"] = "data/processed/baseline/modeling_weekly.csv"
    updated["paths"]["processed_metadata_path"] = "data/processed/baseline/modeling_metadata.json"
    updated["paths"]["splits_dir"] = "data/splits/baseline/default"
    updated["paths"]["predictions_dir"] = "data/predictions"
    updated["paths"]["model_artifacts_dir"] = "models/artifacts"
    updated["paths"]["model_metrics_dir"] = "models/metrics"
    updated["paths"]["stats_dir"] = "models/metrics/summary"

    updated.setdefault("synthetic_data", {})
    updated["synthetic_data"]["input_dataset_path"] = "data/processed/external/context/context_weekly_for_simulation.csv"
    updated["synthetic_data"]["output_dataset_path"] = "data/processed/synthetic/plant/synthetic_plant_layer.csv"
    updated["synthetic_data"]["column_lineage_path"] = "data/processed/synthetic/plant/synthetic_plant_column_lineage.csv"
    updated["synthetic_data"]["simulation_parameters_path"] = "data/processed/synthetic/plant/synthetic_plant_metadata.json"
    updated["synthetic_data"]["environment_summary_path"] = "data/processed/synthetic/plant/synthetic_plant_environment_summary.json"

    updated.setdefault("feature_selection", {})
    updated["feature_selection"]["prefer_synthetic_input"] = True
    updated["feature_selection"]["input_dataset_path"] = "data/processed/synthetic/plant/synthetic_plant_layer.csv"
    updated["feature_selection"]["fallback_input_dataset_path"] = "data/processed/external/context/context_weekly_for_simulation.csv"
    updated["feature_selection"]["feature_catalog_path"] = "data/processed/baseline/feature_catalog.csv"
    updated["feature_selection"]["feature_contract_path"] = "data/processed/baseline/feature_contract.csv"
    updated["feature_selection"]["feature_roles_metadata_path"] = "data/processed/baseline/feature_roles_metadata.json"
    updated["feature_selection"]["feature_config_export_path"] = "data/processed/baseline/feature_selection.yaml"
    updated["feature_selection"]["prepared_dataset_path"] = "data/processed/baseline/feature_engineering_modeling.csv"

    updated.setdefault("prediction", {})
    updated["prediction"]["input_path"] = "data/processed/baseline/feature_engineering_modeling.csv"
    updated["prediction"]["output_filename"] = "predictions_latest.csv"

    updated.setdefault("training", {})
    updated["training"]["comparison_summary_csv_name"] = "baseline_comparison_latest.csv"
    updated["training"]["comparison_summary_json_name"] = "baseline_comparison_latest.json"
    updated["training"]["comparison_name"] = "baseline_comparison"
    updated["training"]["primary_metric"] = "validation_rmse"

    updated.setdefault("neuroevolution", {})
    updated["neuroevolution"]["comparison_summary_csv_name"] = "neuroevolution_comparison_latest.csv"
    updated["neuroevolution"]["comparison_summary_json_name"] = "neuroevolution_comparison_latest.json"

    updated.setdefault("policy_simulation", {})
    updated["policy_simulation"]["input_dataset_path"] = "data/processed/baseline/feature_engineering_modeling.csv"
    updated["policy_simulation"]["summary_csv_name"] = "policy_simulation_latest.csv"
    updated["policy_simulation"]["summary_json_name"] = "policy_simulation_latest.json"
    updated["policy_simulation"].setdefault("reference_runs", {})
    updated["policy_simulation"]["reference_runs"]["baseline_summary_json"] = "models/metrics/summary/baseline_comparison_latest.json"
    updated["policy_simulation"]["reference_runs"]["neuro_summary_json"] = "models/metrics/summary/neuroevolution_comparison_latest.json"

    updated.setdefault("get_stats", {})
    updated["get_stats"]["primary_metric"] = "validation_rmse"
    updated["get_stats"]["summary_csv_name"] = "metrics_summary.csv"
    updated["get_stats"]["summary_json_name"] = "metrics_summary.json"
    return updated


def _apply_mode_overrides(config: dict[str, Any], *, mode: str) -> dict[str, Any]:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported reproducibility mode: {mode}")

    updated = deepcopy(config)
    updated.setdefault("runtime", {})
    updated["runtime"]["reproducibility_mode"] = mode

    if mode == FULL_MODE:
        return updated

    updated["training"]["model_families"] = ["dummy", "linear_regression"]
    updated["training"]["feature_sets"] = ["ablation_reduced_context"]
    updated["training"]["primary_metric"] = "validation_rmse"
    updated["neuroevolution"]["enabled"] = True
    updated["neuroevolution"]["population_size"] = 8
    updated["neuroevolution"]["generations"] = 4
    updated["neuroevolution"]["elite_size"] = 2
    updated["neuroevolution"]["tournament_size"] = 2
    updated["neuroevolution"]["hidden_layer_options"] = [4, 8]
    updated["neuroevolution"]["stagnation_patience"] = 2
    updated["neuroevolution"]["log_every_generations"] = 1
    updated["policy_simulation"]["scenario_generation"]["seeds"] = [7]
    updated["policy_simulation"]["scenario_generation"]["scenario_templates"] = [
        dict(updated["policy_simulation"]["scenario_generation"]["scenario_templates"][0])
    ]
    updated["policy_simulation"]["scenario_generation"]["scenario_templates"][0]["horizon_weeks"] = 12

    if mode == CACHED_MODE:
        updated["runtime"]["reproducibility_cached_training_allowed"] = True
    return updated


def build_reproducibility_config(
    config_path: str | Path | None = None,
    *,
    mode: str = SMOKE_MODE,
    allow_synthetic_plant_layer: bool = True,
) -> dict[str, Any]:
    base_config = load_config(config_path or DEFAULT_CONFIG_PATH)
    base_config = _base_reproducibility_overrides(base_config)
    scoped_config = build_recipe_runtime_configs(base_config, mixed_context=True)[0]
    scoped_config = _apply_mode_overrides(scoped_config, mode=mode)
    scoped_config.setdefault("runtime", {})
    scoped_config["runtime"]["reproducibility_scope"] = SCOPE
    scoped_config["runtime"]["allow_synthetic_plant_layer"] = bool(allow_synthetic_plant_layer)

    if not allow_synthetic_plant_layer and scoped_config.get("synthetic_data", {}).get("enabled", False):
        raise ValueError("The official mixed_context route requires the declared synthetic plant layer.")
    return scoped_config


def official_paths(config: dict[str, Any]) -> dict[str, Any]:
    repo_root_path = Path(config["project"]["repo_root"])
    feature_cfg = config.get("feature_selection", {})
    paths_cfg = config.get("paths", {})
    synthetic_cfg = config.get("synthetic_data", {})
    return {
        "external_long": resolve_repo_path("data/processed/external/context/external_long.csv", repo_root_path),
        "context_weekly": resolve_repo_path("data/processed/external/context/context_weekly_for_simulation.csv", repo_root_path),
        "context_limitations": resolve_repo_path("data/processed/external/context/context_proxy_limitations.json", repo_root_path),
        "synthetic_layer": resolve_repo_path(synthetic_cfg["output_dataset_path"], repo_root_path),
        "synthetic_metadata": resolve_repo_path(synthetic_cfg["simulation_parameters_path"], repo_root_path),
        "modeling_weekly": resolve_repo_path(paths_cfg["processed_dataset_path"], repo_root_path),
        "modeling_metadata": resolve_repo_path(paths_cfg["processed_metadata_path"], repo_root_path),
        "feature_engineering_modeling": resolve_repo_path(feature_cfg["prepared_dataset_path"], repo_root_path),
        "feature_selection_export": resolve_repo_path(feature_cfg["feature_config_export_path"], repo_root_path),
        "feature_contract": resolve_repo_path(feature_cfg["feature_contract_path"], repo_root_path),
        "feature_roles_metadata": resolve_repo_path(feature_cfg["feature_roles_metadata_path"], repo_root_path),
        "splits_dir": resolve_repo_path(paths_cfg["splits_dir"], repo_root_path),
        "predictions_dir": resolve_repo_path(paths_cfg["predictions_dir"], repo_root_path),
        "predictions_latest": resolve_repo_path("data/predictions/predictions_latest__mixed_context.csv", repo_root_path),
        "artifacts_dir": resolve_repo_path(paths_cfg["model_artifacts_dir"], repo_root_path),
        "upstream_predictor_artifact": resolve_repo_path("models/artifacts/upstream_predictor_latest__mixed_context.pkl", repo_root_path),
        "purchase_trigger_artifact": resolve_repo_path("models/artifacts/purchase_trigger_latest__mixed_context.pkl", repo_root_path),
        "quantity_optimizer_artifact": resolve_repo_path("models/artifacts/quantity_optimizer_latest__mixed_context.pkl", repo_root_path),
        "model_manifest": resolve_repo_path("models/artifacts/model_manifest__mixed_context.json", repo_root_path),
        "metrics_dir": resolve_repo_path(paths_cfg["model_metrics_dir"], repo_root_path),
        "official_metrics_dir": resolve_repo_path("models/metrics/official", repo_root_path),
        "summary_dir": resolve_repo_path(paths_cfg["stats_dir"], repo_root_path),
        "baseline_summary_csv": resolve_repo_path("models/metrics/summary/baseline_comparison_latest__mixed_context.csv", repo_root_path),
        "baseline_summary_json": resolve_repo_path("models/metrics/summary/baseline_comparison_latest__mixed_context.json", repo_root_path),
        "neuro_summary_csv": resolve_repo_path("models/metrics/summary/neuroevolution_comparison_latest__mixed_context.csv", repo_root_path),
        "neuro_summary_json": resolve_repo_path("models/metrics/summary/neuroevolution_comparison_latest__mixed_context.json", repo_root_path),
        "trigger_metrics_json": resolve_repo_path("models/metrics/summary/trigger_metrics_latest__mixed_context.json", repo_root_path),
        "quantity_optimizer_metrics_json": resolve_repo_path("models/metrics/summary/quantity_optimizer_latest__mixed_context.json", repo_root_path),
        "quantity_optimizer_baseline_comparison_json": resolve_repo_path(
            "models/metrics/summary/quantity_optimizer_baseline_comparison_latest__mixed_context.json",
            repo_root_path,
        ),
        "quantity_optimizer_baseline_comparison_csv": resolve_repo_path(
            "models/metrics/summary/quantity_optimizer_baseline_comparison_latest__mixed_context.csv",
            repo_root_path,
        ),
        "trigger_predictions": {
            split: resolve_repo_path(
                f"models/metrics/official/purchase_trigger_predictions_{split}__mixed_context.csv",
                repo_root_path,
            )
            for split in ("train", "validation", "test")
        },
        "quantity_optimizer_predictions": {
            split: resolve_repo_path(
                f"models/metrics/official/quantity_optimizer_predictions_{split}__mixed_context.csv",
                repo_root_path,
            )
            for split in ("train", "validation", "test")
        },
        "policy_simulation_summary_csv": resolve_repo_path("models/metrics/summary/policy_simulation_latest__mixed_context.csv", repo_root_path),
        "policy_simulation_summary_json": resolve_repo_path("models/metrics/summary/policy_simulation_latest__mixed_context.json", repo_root_path),
        "policy_simulation_period_csv": resolve_repo_path(
            "models/metrics/official/policy_simulation_period_latest__mixed_context.csv",
            repo_root_path,
        ),
        "policy_simulation_scenario_csv": resolve_repo_path(
            "models/metrics/official/policy_simulation_scenario_latest__mixed_context.csv",
            repo_root_path,
        ),
        "metrics_summary_csv": resolve_repo_path("models/metrics/summary/metrics_summary__mixed_context.csv", repo_root_path),
        "metrics_summary_json": resolve_repo_path("models/metrics/summary/metrics_summary__mixed_context.json", repo_root_path),
        "official_report_dir": resolve_repo_path("reports/official", repo_root_path),
        "official_metrics_report": resolve_repo_path(
            "reports/official/cu28_metrics_official__mixed_context.md",
            repo_root_path,
        ),
        "official_formula_report": resolve_repo_path(
            "reports/official/synthetic_procurement_need_formula__mixed_context.md",
            repo_root_path,
        ),
        "official_formula_json": resolve_repo_path(
            "reports/official/synthetic_procurement_need_formula__mixed_context.json",
            repo_root_path,
        ),
        "audit_report": resolve_repo_path(
            "reports/audit/cu28_metrics_consistency_report.md",
            repo_root_path,
        ),
        "raw_manifest": resolve_repo_path("data/raw/external/raw_manifest__mixed_context.json", repo_root_path),
        "repro_manifest": resolve_repo_path("reproducibility_manifest__mixed_context.json", repo_root_path),
    }


def runtime_environment() -> dict[str, str]:
    return {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
