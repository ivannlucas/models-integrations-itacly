from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.reproducibility.hashes import describe_existing_files, sha256_file
from src.reproducibility.runtime import official_paths
from src.utils import ensure_directory, write_json


def make_splits(config: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(config["project"]["repo_root"])
    paths = official_paths(config)
    split_cfg = config["data_processing"]["split"]

    modeling_df = pd.read_csv(paths["feature_engineering_modeling"])
    modeling_df["date"] = pd.to_datetime(modeling_df["date"], errors="coerce")
    modeling_df = modeling_df.sort_values("date").reset_index(drop=True)
    total_rows = len(modeling_df)

    train_end = max(1, int(total_rows * float(split_cfg.get("train_size", 0.70))))
    validation_end = train_end + int(total_rows * float(split_cfg.get("valid_size", 0.15)))

    train_df = modeling_df.iloc[:train_end].reset_index(drop=True)
    validation_df = modeling_df.iloc[train_end:validation_end].reset_index(drop=True)
    test_df = modeling_df.iloc[validation_end:].reset_index(drop=True)

    splits_dir = ensure_directory(paths["splits_dir"])
    train_path = splits_dir / "train.csv"
    validation_path = splits_dir / "validation.csv"
    valid_alias_path = splits_dir / "valid.csv"
    test_path = splits_dir / "test.csv"
    split_metadata_path = splits_dir / "split_metadata.json"

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    validation_df.to_csv(valid_alias_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata = {
        "scope": "mixed_context",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "criterion": "chronological",
        "input_dataset_path": "data/processed/baseline/feature_engineering_modeling__mixed_context.csv",
        "input_dataset_hash": sha256_file(paths["feature_engineering_modeling"]),
        "target_columns": [
            "synthetic_procurement_need",
            "purchase_trigger_label",
            "quantity_optimizer_target_tons",
        ],
        "excluded_columns": list(config.get("feature_selection", {}).get("excluded_features", [])),
        "splits": {
            "train": {
                "path": "data/splits/baseline/default__mixed_context/train.csv",
                "rows": int(len(train_df)),
                "date_start": str(train_df["date"].min().date()) if not train_df.empty else None,
                "date_end": str(train_df["date"].max().date()) if not train_df.empty else None,
                "pct": round(len(train_df) / total_rows, 4) if total_rows else 0.0,
            },
            "validation": {
                "path": "data/splits/baseline/default__mixed_context/validation.csv",
                "rows": int(len(validation_df)),
                "date_start": str(validation_df["date"].min().date()) if not validation_df.empty else None,
                "date_end": str(validation_df["date"].max().date()) if not validation_df.empty else None,
                "pct": round(len(validation_df) / total_rows, 4) if total_rows else 0.0,
            },
            "test": {
                "path": "data/splits/baseline/default__mixed_context/test.csv",
                "rows": int(len(test_df)),
                "date_start": str(test_df["date"].min().date()) if not test_df.empty else None,
                "date_end": str(test_df["date"].max().date()) if not test_df.empty else None,
                "pct": round(len(test_df) / total_rows, 4) if total_rows else 0.0,
            },
        },
    }
    write_json(split_metadata_path, metadata)
    return {
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "test_path": str(test_path),
        "split_metadata_path": str(split_metadata_path),
    }
