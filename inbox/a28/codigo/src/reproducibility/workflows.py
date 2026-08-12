from __future__ import annotations

from typing import Any

from src.data_acquisition import run_data_acquisition
from src.etl import build_context_weekly, build_external_long, build_modeling_dataset, build_synthetic_plant_layer, make_splits
from src.feature_engineering import run_feature_engineering


def run_reproducibility_etl(config: dict[str, Any], logger, *, force_download: bool = False) -> dict[str, Any]:
    external_long = build_external_long(config, force_download=force_download)
    context_weekly = build_context_weekly(config)
    synthetic_layer = build_synthetic_plant_layer(config)
    modeling = build_modeling_dataset(config)
    logger.info("ETL reproducible completado hasta %s", modeling["feature_engineering_modeling_path"])
    return {
        "external_long": external_long,
        "context_weekly": context_weekly,
        "synthetic_layer": synthetic_layer,
        "modeling_dataset": modeling,
    }


def run_reproducibility_make_splits(config: dict[str, Any], logger) -> dict[str, Any]:
    result = make_splits(config)
    logger.info("Splits cronologicos generados en %s", result["split_metadata_path"])
    return result
