from __future__ import annotations

from pathlib import Path
from typing import Any

from src.reproducibility.hashes import describe_existing_files, sha256_file
from src.reproducibility.runtime import official_paths, require_official_end_to_end_run
from src.training.train_purchase_trigger import run_purchase_trigger_training
from src.training.train_quantity_optimizer import run_quantity_optimizer_training
from src.training.train_upstream_predictor import run_upstream_training
from src.utils import write_json


def run_reproducibility_training(config: dict[str, Any], logger) -> dict[str, Any]:
    official_run = require_official_end_to_end_run(config, stage="train")
    paths = official_paths(config)
    repo_root = Path(config["project"]["repo_root"])

    upstream = run_upstream_training(config, logger)
    trigger = run_purchase_trigger_training(config, logger)
    quantity = run_quantity_optimizer_training(config, logger)

    manifest = {
        "scope": "mixed_context",
        "official_run": official_run,
        "selection_criterion": "validation_rmse",
        "artifacts": describe_existing_files(
            [
                paths["upstream_predictor_artifact"],
                paths["purchase_trigger_artifact"],
                paths["quantity_optimizer_artifact"],
            ],
            repo_root=repo_root,
        ),
        "summary_metrics": describe_existing_files(
            [
                paths["baseline_summary_json"],
                paths["neuro_summary_json"],
                paths["trigger_metrics_json"],
                paths["quantity_optimizer_metrics_json"],
                paths["quantity_optimizer_baseline_comparison_json"],
                paths["quantity_optimizer_baseline_comparison_csv"],
                *paths["trigger_predictions"].values(),
                *paths["quantity_optimizer_predictions"].values(),
            ],
            repo_root=repo_root,
        ),
        "seeds": {
            "project_seed": config.get("project", {}).get("seed"),
            "training_seed": config.get("training", {}).get("random_seed"),
            "neuroevolution_seed": config.get("neuroevolution", {}).get("random_seed"),
        },
        "stages": {
            "upstream_predictor": upstream,
            "purchase_trigger": trigger,
            "quantity_optimizer": quantity,
        },
    }
    write_json(paths["model_manifest"], manifest)
    return {
        "upstream": upstream,
        "trigger": trigger,
        "quantity_optimizer": quantity,
        "model_manifest_path": str(paths["model_manifest"]),
    }
