from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.data_processing.pipeline import data_processing_pipeline
from src.get_stats.pipeline import get_stats_pipeline
from src.predict.predictor import InferenceConfig, predict_pipeline
from src.training.common import TrainConfig
from src.training.pipeline import train_pipeline
from src.utils.config import load_yaml
from src.utils.paths import resolve_repo_path


def _resolve_device(device: str | None) -> str:
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def _paths_section(project_cfg: dict[str, Any]) -> dict[str, str]:
    return project_cfg.get("paths", {})


def data_processing(config_path: str = "config/config.yaml", **overrides: Any) -> dict[str, Any]:
    project_cfg = load_yaml(config_path)
    paths = _paths_section(project_cfg)
    section = dict(project_cfg.get("data_processing", {}))
    for key, value in overrides.items():
        if value is not None:
            section[key] = value
    section["out_telemetry"] = str(resolve_repo_path(section.get("out_telemetry", paths.get("processed_telemetry", "data/processed/telemetry_processed.csv"))))
    section["out_maintenance"] = str(resolve_repo_path(section.get("out_maintenance", paths.get("processed_maintenance", "data/processed/maintenance_processed.csv"))))
    section["out_meta"] = str(resolve_repo_path(section.get("out_meta", paths.get("processed_metadata", "data/processed/generation_metadata.json"))))
    return data_processing_pipeline(section)


def train(config_path: str = "config/config.yaml", **overrides: Any) -> dict[str, Any]:
    project_cfg = load_yaml(config_path)
    paths = _paths_section(project_cfg)
    section = dict(project_cfg.get("training", {}))
    for key, value in overrides.items():
        if value is not None:
            section[key] = value
    cfg = TrainConfig(
        telemetry=str(resolve_repo_path(section.pop("telemetry", paths.get("processed_telemetry", "data/processed/telemetry_processed.csv")))),
        maintenance=str(resolve_repo_path(section.pop("maintenance", paths.get("processed_maintenance", "data/processed/maintenance_processed.csv")))),
        artifacts_dir=str(resolve_repo_path(paths.get("artifacts_dir", "models/artifacts"))),
        metrics_dir=str(resolve_repo_path(paths.get("metrics_dir", "models/metrics"))),
        predictions_dir=str(resolve_repo_path(paths.get("predictions_dir", "data/predictions"))),
        splits_dir=str(resolve_repo_path(paths.get("splits_dir", "data/splits"))),
        device=_resolve_device(section.pop("device", None)),
        **section,
    )
    return train_pipeline(cfg)


def predict(config_path: str = "config/config.yaml", **overrides: Any) -> dict[str, Any]:
    project_cfg = load_yaml(config_path)
    paths = _paths_section(project_cfg)
    section = dict(project_cfg.get("inference", {}))
    for key, value in overrides.items():
        if value is not None:
            section[key] = value
    cfg = InferenceConfig(
        telemetry_input=str(resolve_repo_path(section.pop("telemetry_input", paths.get("processed_telemetry", "data/processed/telemetry_processed.csv")))),
        artifacts_dir=str(resolve_repo_path(paths.get("artifacts_dir", "models/artifacts"))),
        metrics_dir=str(resolve_repo_path(paths.get("metrics_dir", "models/metrics"))),
        predictions_dir=str(resolve_repo_path(paths.get("predictions_dir", "data/predictions"))),
        device=_resolve_device(section.pop("device", None)),
        **section,
    )
    return predict_pipeline(cfg)


def get_stats(config_path: str = "config/config.yaml", **overrides: Any) -> dict[str, Any]:
    project_cfg = load_yaml(config_path)
    paths = _paths_section(project_cfg)
    section = dict(project_cfg.get("stats", {}))
    for key, value in overrides.items():
        if value is not None:
            section[key] = value
    section["telemetry"] = str(resolve_repo_path(section.get("telemetry", paths.get("processed_telemetry", "data/processed/telemetry_processed.csv"))))
    section["maintenance"] = str(resolve_repo_path(section.get("maintenance", paths.get("processed_maintenance", "data/processed/maintenance_processed.csv"))))
    section["outdir"] = str(resolve_repo_path(section.get("outdir", paths.get("stats_output_dir", "models/metrics/stats"))))
    return get_stats_pipeline(section, artifacts_dir=str(resolve_repo_path(paths.get("artifacts_dir", "models/artifacts"))), metrics_dir=str(resolve_repo_path(paths.get("metrics_dir", "models/metrics"))))
