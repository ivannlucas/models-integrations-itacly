from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_processing.synthetic_plant import build_synthetic_plant_dataset, write_synthetic_plant_outputs
from src.reproducibility.hashes import describe_existing_files
from src.reproducibility.mixed_context import derive_official_columns
from src.reproducibility.runtime import official_paths
from src.utils import ensure_directory, write_json


def build_synthetic_plant_layer(config: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(config["project"]["repo_root"])
    paths = official_paths(config)
    context_df = pd.read_csv(paths["context_weekly"])

    synthetic_df, lineage_df, assumptions_payload = build_synthetic_plant_dataset(
        context_df,
        config["synthetic_data"],
        recipe_runtime_context=config.get("runtime", {}).get("recipe_context", {}),
        recipe_registry_path=config.get("manufacturing_profiles", {}).get("registry_path"),
    )
    synthetic_df = derive_official_columns(synthetic_df, config)
    output_paths = write_synthetic_plant_outputs(
        synthetic_df,
        lineage_df,
        assumptions_payload,
        config["synthetic_data"],
        repo_root,
    )

    interim_dir = ensure_directory(repo_root / "data" / "interim" / "synthetic")
    metadata_path = interim_dir / "build_synthetic_plant_layer__mixed_context.json"
    metadata = {
        "scope": "mixed_context",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": describe_existing_files([paths["context_weekly"]], repo_root=repo_root),
        "outputs": describe_existing_files(
            [paths["synthetic_layer"], paths["synthetic_metadata"], repo_root / output_paths["column_lineage_path"]],
            repo_root=repo_root,
        ),
        "row_count": int(len(synthetic_df)),
        "columns": synthetic_df.columns.tolist(),
        "date_min": str(pd.to_datetime(synthetic_df["date"]).min().date()) if not synthetic_df.empty else None,
        "date_max": str(pd.to_datetime(synthetic_df["date"]).max().date()) if not synthetic_df.empty else None,
    }
    write_json(metadata_path, metadata)
    return {
        "synthetic_layer_path": str(paths["synthetic_layer"]),
        "synthetic_metadata_path": str(paths["synthetic_metadata"]),
        "metadata_path": str(metadata_path),
    }
