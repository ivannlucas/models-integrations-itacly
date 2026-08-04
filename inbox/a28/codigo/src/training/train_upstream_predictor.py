from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from src.reproducibility.runtime import ensure_optional_dependency, official_paths
from src.utils import resolve_repo_path


def run_upstream_training(config: dict[str, Any], logger) -> dict[str, Any]:
    ensure_optional_dependency("sklearn", repo_root_path=Path(config["project"]["repo_root"]))
    from src.training.pipeline import run_training

    result = run_training(config, logger)
    paths = official_paths(config)
    repo_root = Path(config["project"]["repo_root"])

    best_baseline_run = dict(result.get("best_baseline_run") or {})
    artifact_value = best_baseline_run.get("artifact_path")
    if not artifact_value:
        raise FileNotFoundError("The upstream training stage completed but did not expose a best_baseline_run artifact.")

    artifact_path = Path(artifact_value)
    if not artifact_path.is_absolute():
        artifact_path = resolve_repo_path(artifact_path, repo_root)
    paths["upstream_predictor_artifact"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact_path, paths["upstream_predictor_artifact"])

    logger.info(
        "Copied best upstream baseline artifact to %s from run_id=%s",
        paths["upstream_predictor_artifact"],
        best_baseline_run.get("run_id"),
    )
    return {
        "baseline_summary_json_path": str(paths["baseline_summary_json"]),
        "baseline_summary_csv_path": str(paths["baseline_summary_csv"]),
        "neuro_summary_json_path": str(paths["neuro_summary_json"]),
        "neuro_summary_csv_path": str(paths["neuro_summary_csv"]),
        "upstream_predictor_artifact_path": str(paths["upstream_predictor_artifact"]),
        "best_baseline_run": best_baseline_run,
        "best_neuroevolution_run": result.get("best_neuroevolution_run"),
        "comparison_run_id": result.get("comparison_run_id"),
    }
