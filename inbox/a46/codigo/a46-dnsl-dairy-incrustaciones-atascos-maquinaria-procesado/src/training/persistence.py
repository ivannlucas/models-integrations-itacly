from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.utils.common import save_json
from src.utils.logging import get_logger
from src.utils.paths import to_repo_relative_path

LOGGER = get_logger(__name__)


def export_split_assets(train_assets: list[str], val_assets: list[str], test_assets: list[str], splits_dir: Path) -> dict[str, str]:
    splits_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, assets in {"train": train_assets, "val": val_assets, "test": test_assets}.items():
        path = splits_dir / f"{name}_assets.csv"
        pd.DataFrame({"asset_id": assets}).to_csv(path, index=False)
        outputs[name] = to_repo_relative_path(path)
    return outputs


def export_split_rows(telemetry_df: pd.DataFrame, train_assets: list[str], val_assets: list[str], test_assets: list[str], splits_dir: Path) -> dict[str, str]:
    outputs = {}
    for name, assets in {"train": train_assets, "val": val_assets, "test": test_assets}.items():
        path = splits_dir / f"{name}_rows.csv"
        telemetry_df.loc[telemetry_df["asset_id"].isin(assets)].to_csv(path, index=False)
        outputs[name] = to_repo_relative_path(path)
    return outputs


def save_feature_artifacts(artifacts: Any, path: Path) -> None:
    save_json(path, asdict(artifacts))


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def export_scenario_outputs(
    scenario_dir: Path,
    scenario_name: str,
    metrics_dir: Path,
    predictions_dir: Path,
) -> dict[str, list[str]]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metric_files = [
        "val_window_metrics_best.json",
        "test_window_metrics.json",
        "val_event_metrics_best.json",
        "test_event_metrics.json",
        "policy_thresholds.json",
        "feature_report.json",
        "summary.json",
        "training_history.csv",
    ]
    prediction_files = [
        "val_window_predictions.csv",
        "test_window_predictions.csv",
        "val_alerts.csv",
        "test_alerts.csv",
        "val_events_matched.csv",
        "test_events_matched.csv",
    ]
    exported = {"metrics": [], "predictions": []}
    for filename in metric_files:
        src = scenario_dir / filename
        if src.exists():
            dst = metrics_dir / f"{scenario_name}_{filename}"
            copy_if_exists(src, dst)
            exported["metrics"].append(to_repo_relative_path(dst))
    for filename in prediction_files:
        src = scenario_dir / filename
        if src.exists():
            dst = predictions_dir / f"{scenario_name}_{filename}"
            copy_if_exists(src, dst)
            exported["predictions"].append(to_repo_relative_path(dst))
    return exported


def build_model_manifest(
    selected_scenario: str,
    summary: Mapping[str, Any],
    checkpoint_path: Path,
    feature_artifacts_path: Path,
    config_snapshot_path: Path,
    selected_model_path: Path,
) -> dict[str, Any]:
    return {
        "selected_scenario": selected_scenario,
        "selected_checkpoint": to_repo_relative_path(checkpoint_path),
        "selected_model": to_repo_relative_path(selected_model_path),
        "feature_artifacts": to_repo_relative_path(feature_artifacts_path),
        "training_config": to_repo_relative_path(config_snapshot_path),
        "summary": dict(summary),
    }
