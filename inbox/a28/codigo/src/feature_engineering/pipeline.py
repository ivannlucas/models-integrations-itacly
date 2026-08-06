from __future__ import annotations

from pathlib import Path
from typing import Any

from src.etl.build_modeling_dataset import build_modeling_dataset
from src.reproducibility.runtime import official_paths


def run_feature_engineering(config: dict[str, Any], logger) -> dict[str, Any]:
    result = build_modeling_dataset(config)
    logger.info(
        "Feature engineering refreshed prepared dataset at %s",
        result["feature_engineering_modeling_path"],
    )
    return result
