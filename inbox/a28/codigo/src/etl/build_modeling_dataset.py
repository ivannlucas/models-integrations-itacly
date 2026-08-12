from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_processing.pipeline import _export_prepared_feature_artifacts, prepare_modeling_frame
from src.reproducibility.hashes import describe_existing_files, sha256_file
from src.reproducibility.mixed_context import derive_official_columns
from src.reproducibility.runtime import official_paths
from src.utils import ensure_directory, write_json


def build_modeling_dataset(config: dict[str, Any], *, source_label: str = "synthetic_plant_layer__mixed_context") -> dict[str, Any]:
    repo_root = Path(config["project"]["repo_root"])
    paths = official_paths(config)
    logger = logging.getLogger(__name__)
    synthetic_df = pd.read_csv(paths["synthetic_layer"])
    synthetic_df["date"] = pd.to_datetime(synthetic_df["date"], errors="coerce")
    synthetic_df = synthetic_df.sort_values("date").reset_index(drop=True)
    synthetic_df = derive_official_columns(synthetic_df, config)
    synthetic_df.to_csv(paths["modeling_weekly"], index=False)

    export_result = _export_prepared_feature_artifacts(
        config,
        repo_root=repo_root,
        source_df=synthetic_df,
        source_path_label=source_label,
        logger=logger,
    )
    modeling_df = pd.read_csv(paths["feature_engineering_modeling"])
    modeling_df["date"] = pd.to_datetime(modeling_df["date"], errors="coerce")
    modeling_df = derive_official_columns(modeling_df, config)
    modeling_df.to_csv(paths["feature_engineering_modeling"], index=False)

    metadata = {
        "scope": "mixed_context",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dataset_path": "data/processed/synthetic/plant/synthetic_plant_layer__mixed_context.csv",
        "input_dataset_hash": sha256_file(paths["synthetic_layer"]),
        "modeling_weekly_path": "data/processed/baseline/modeling_weekly__mixed_context.csv",
        "prepared_dataset_path": "data/processed/baseline/feature_engineering_modeling__mixed_context.csv",
        "prepared_dataset_hash": sha256_file(paths["feature_engineering_modeling"]),
        "row_count": int(len(modeling_df)),
        "columns": modeling_df.columns.tolist(),
        "date_min": str(modeling_df["date"].min().date()) if not modeling_df.empty else None,
        "date_max": str(modeling_df["date"].max().date()) if not modeling_df.empty else None,
        "feature_artifacts": export_result,
    }
    write_json(paths["modeling_metadata"], metadata)

    interim_dir = ensure_directory(repo_root / "data" / "interim" / "synthetic")
    phase_metadata_path = interim_dir / "build_modeling_dataset__mixed_context.json"
    phase_metadata = {
        "scope": "mixed_context",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": describe_existing_files([paths["synthetic_layer"]], repo_root=repo_root),
        "outputs": describe_existing_files(
            [
                paths["modeling_weekly"],
                paths["feature_engineering_modeling"],
                paths["modeling_metadata"],
                paths["feature_selection_export"],
                paths["feature_contract"],
                paths["feature_roles_metadata"],
            ],
            repo_root=repo_root,
        ),
        "row_count": int(len(modeling_df)),
    }
    write_json(phase_metadata_path, phase_metadata)
    return {
        "modeling_weekly_path": str(paths["modeling_weekly"]),
        "feature_engineering_modeling_path": str(paths["feature_engineering_modeling"]),
        "modeling_metadata_path": str(paths["modeling_metadata"]),
        "metadata_path": str(phase_metadata_path),
    }
