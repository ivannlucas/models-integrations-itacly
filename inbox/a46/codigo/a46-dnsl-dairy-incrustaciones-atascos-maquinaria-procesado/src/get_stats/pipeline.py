from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.utils.logging import get_logger
from src.utils.paths import resolve_saved_path

from .column_info import write_column_catalog
from .eda import EDAConfig, run_eda

LOGGER = get_logger(__name__)


def build_eda_config(config: Mapping[str, Any]) -> EDAConfig:
    return EDAConfig(**config)


def get_stats_pipeline(config: Mapping[str, Any], artifacts_dir: str, metrics_dir: str) -> dict[str, Any]:
    cfg = build_eda_config(config)
    outputs = run_eda(cfg)
    telemetry_df = pd.read_csv(resolve_saved_path(cfg.telemetry))
    catalog_outputs = write_column_catalog(
        df=telemetry_df,
        out_csv=resolve_saved_path(cfg.outdir) / "column_catalog.csv",
        artifacts_dir=artifacts_dir,
        metrics_dir=metrics_dir,
    )
    outputs.update(catalog_outputs)
    return outputs
