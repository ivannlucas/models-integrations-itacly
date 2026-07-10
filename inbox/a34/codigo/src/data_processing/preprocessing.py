"""
Data loading, filtering, temporal splitting and normalization functions.

Encapsulates all preprocessing performed in the EDA, training and
hyperparameter tuning notebooks.
"""

from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.constants import FEATURES, TARGETS
from src.utils.paths import (
    RAW_DATASET_PATH, PROCESSED_DATASET_PATH, PROCESSED_DATA_DIR,
    SPLITS_DIR, TRAIN_SPLIT_PATH, VAL_SPLIT_PATH, TEST_SPLIT_PATH,
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_raw_data(path: str = None) -> pd.DataFrame:
    """
    Load the raw dataset from CSV.

    Parameters
    ----------
    path : str, optional
        Path to the CSV file. If None, uses the standard project path.

    Returns
    -------
    pd.DataFrame
    """
    p = path or str(RAW_DATASET_PATH)
    return pd.read_csv(p)


def load_processed_data(path: str = None) -> pd.DataFrame:
    """
    Load the processed dataset (cleaning records already removed).

    Parameters
    ----------
    path : str, optional
        Path to the CSV file. If None, uses the standard project path.

    Returns
    -------
    pd.DataFrame
    """
    p = path or str(PROCESSED_DATASET_PATH)
    return pd.read_csv(p)


# =============================================================================
# FILTERING
# =============================================================================

def filter_production(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter production records (excludes CIP cleaning periods).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (with Is_Cleaning column).

    Returns
    -------
    pd.DataFrame
        Only records where Is_Cleaning == 0.
    """
    if "Is_Cleaning" in df.columns:
        return df[df["Is_Cleaning"] == 0].copy()
    return df.copy()


def save_processed_data(df_prod: pd.DataFrame, path: str = None) -> None:
    """
    Save the filtered production dataset to data/processed/.

    Parameters
    ----------
    df_prod : pd.DataFrame
        Filtered DataFrame (no CIP).
    path : str, optional
        Destination path.
    """
    p = path or str(PROCESSED_DATASET_PATH)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_prod.to_csv(p, index=False)


# =============================================================================
# TEMPORAL SPLIT BY QUARTILES
# =============================================================================

def temporal_split_by_quartiles(
    df: pd.DataFrame,
    features: list = None,
    targets: list = None,
    n_blocks: int = 4,
    train_ratio: float = 0.70,
    val_ratio: float = 0.85,
) -> Dict[str, np.ndarray]:
    """
    Perform a temporal quartile-based split of the DataFrame.

    Splits each temporal block into train/val/test while respecting
    temporal order (no data leakage), then concatenates blocks.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataset (production only), sorted by Time_min.
    features : list
        List of feature column names (default: FEATURES).
    targets : list
        List of target column names (default: TARGETS).
    n_blocks : int
        Number of temporal blocks.
    train_ratio : float
        Fraction of each block allocated to training.
    val_ratio : float
        Cumulative fraction up to which validation is included
        (e.g., 0.85 = 15% validation if train_ratio=0.70).

    Returns
    -------
    dict
        Dictionary with keys: X_train, X_val, X_test, y_train, y_val, y_test.
    """
    features = features or FEATURES
    targets = targets or TARGETS

    df = df.sort_values(by="Time_min").reset_index(drop=True)
    chunk_size = len(df) // n_blocks

    X_train_list, X_val_list, X_test_list = [], [], []
    y_train_list, y_val_list, y_test_list = [], [], []

    for i in range(n_blocks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_blocks - 1 else len(df)
        chunk = df.iloc[start_idx:end_idx]

        train_idx = int(len(chunk) * train_ratio)
        val_idx = int(len(chunk) * val_ratio)

        X_chunk = chunk[features].values
        y_chunk = chunk[targets].values

        X_train_list.append(X_chunk[:train_idx])
        X_val_list.append(X_chunk[train_idx:val_idx])
        X_test_list.append(X_chunk[val_idx:])

        y_train_list.append(y_chunk[:train_idx])
        y_val_list.append(y_chunk[train_idx:val_idx])
        y_test_list.append(y_chunk[val_idx:])

    return {
        "X_train": np.vstack(X_train_list),
        "X_val":   np.vstack(X_val_list),
        "X_test":  np.vstack(X_test_list),
        "y_train": np.vstack(y_train_list),
        "y_val":   np.vstack(y_val_list),
        "y_test":  np.vstack(y_test_list),
    }


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_data(
    splits: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], MinMaxScaler, MinMaxScaler]:
    """
    Apply MinMaxScaler [0, 1] to features and targets.

    Fits scalers exclusively on the training set (no data leakage)
    and transforms train, val and test.

    Parameters
    ----------
    splits : dict
        Dictionary with X_train, X_val, X_test, y_train, y_val, y_test.

    Returns
    -------
    tuple
        (scaled_splits, scaler_X, scaler_y) where scaled_splits has the
        same keys but with normalized float32 np.ndarray values.
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_s = scaler_X.fit_transform(splits["X_train"]).astype(np.float32)
    X_val_s = scaler_X.transform(splits["X_val"]).astype(np.float32)
    X_test_s = scaler_X.transform(splits["X_test"]).astype(np.float32)

    y_train_s = scaler_y.fit_transform(splits["y_train"]).astype(np.float32)
    y_val_s = scaler_y.transform(splits["y_val"]).astype(np.float32)
    y_test_s = scaler_y.transform(splits["y_test"]).astype(np.float32)

    scaled = {
        "X_train": X_train_s, "X_val": X_val_s, "X_test": X_test_s,
        "y_train": y_train_s, "y_val": y_val_s, "y_test": y_test_s,
    }
    return scaled, scaler_X, scaler_y


# =============================================================================
# SAVE / LOAD SPLITS (CSV)
# =============================================================================

def save_splits(
    splits: Dict[str, np.ndarray],
    features: list = None,
    targets: list = None,
) -> None:
    """
    Save train/val/test splits as CSV for auditing purposes.

    Parameters
    ----------
    splits : dict
        Dictionary with X_train, X_val, X_test, y_train, y_val, y_test.
    features : list, optional
    targets : list, optional
    """
    features = features or FEATURES
    targets = targets or TARGETS
    columns = features + targets
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, x_key, y_key, path in [
        ("train", "X_train", "y_train", TRAIN_SPLIT_PATH),
        ("val",   "X_val",   "y_val",   VAL_SPLIT_PATH),
        ("test",  "X_test",  "y_test",  TEST_SPLIT_PATH),
    ]:
        data = np.hstack((splits[x_key], splits[y_key]))
        pd.DataFrame(data, columns=columns).to_csv(path, index=False)


def load_splits(
    features: list = None,
    targets: list = None,
) -> Dict[str, np.ndarray]:
    """
    Reload saved splits from CSV.

    Returns
    -------
    dict
        Dictionary with X_train, X_val, X_test, y_train, y_val, y_test.
    """
    features = features or FEATURES
    targets = targets or TARGETS

    result = {}
    for split_name, path in [
        ("train", TRAIN_SPLIT_PATH),
        ("val", VAL_SPLIT_PATH),
        ("test", TEST_SPLIT_PATH),
    ]:
        df_split = pd.read_csv(path)
        result[f"X_{split_name}"] = df_split[features].values
        result[f"y_{split_name}"] = df_split[targets].values

    return result
