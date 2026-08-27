"""Utility helpers for logging and data loading."""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_scaler_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load scaler metadata JSON with scaling parameters and denormalization guidance.
    
    Args:
        metadata_path: Path to scaler_metadata.json file.
    
    Returns:
        dict: Parsed metadata including column ranges and denormalization formulas.
    
    Raises:
        FileNotFoundError: If metadata file does not exist.
        json.JSONDecodeError: If metadata file is not valid JSON.
    """
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def denormalize_columns(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    columns_to_denormalize: list[str],
) -> pd.DataFrame:
    """Denormalize specified columns back to physical units using fitted scaler.
    
    This function applies the inverse min-max transformation:
        value_physical = value_normalized / scale + data_min
    
    Args:
        df: DataFrame with normalized values in [0, 1].
        scaler: Fitted MinMaxScaler instance with scale_ and data_min_ attributes.
        columns_to_denormalize: Column names to inverse-transform.
    
    Returns:
        pd.DataFrame: Copy with denormalized columns in physical units.
    
    Raises:
        ValueError: If columns are not found in DataFrame or scaler.
    """
    missing = [col for col in columns_to_denormalize if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    
    result = df.copy()
    # Assume column order in scaler matches the order of columns_to_denormalize
    for idx, col in enumerate(columns_to_denormalize):
        result[col] = result[col] / scaler.scale_[idx] + scaler.data_min_[idx]
    
    return result


def get_project_root() -> Path:
    """Return the project root path.

    Returns:
        Path: Absolute path to the project root.
    """

    return Path(__file__).resolve().parents[2]


def configure_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure and return a root logger for the application.

    Args:
        log_level: Logging level as a string.

    Returns:
        logging.Logger: Configured logger.
    """

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("optimization")


def load_dataset(dataset_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load a CSV dataset with robust error handling.

    Args:
        dataset_path: Absolute path to the dataset.
        logger: Logger instance.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the dataset path does not exist.
        RuntimeError: If pandas fails to parse the CSV file.
    """

    try:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        logger.info("Loading dataset from %s", dataset_path)
        return pd.read_csv(dataset_path)
    except FileNotFoundError:
        logger.exception("Dataset file is missing.")
        raise
    except Exception as exc:  # pragma: no cover - defensive error path
        logger.exception("Failed to read dataset CSV.")
        raise RuntimeError("Unable to load dataset.") from exc
