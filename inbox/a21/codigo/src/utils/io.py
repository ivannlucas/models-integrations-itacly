"""
Helpers de I/O reutilizables: carga de datasets, guardado de artefactos.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


# --- Datasets ----------------------------------------------------------------

def load_dataset(path: str | Path, **kwargs) -> pd.DataFrame:
    """Lee un CSV o Parquet y parsea la columna 'date' si existe."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, **kwargs)
    else:
        parse_dates = kwargs.pop("parse_dates", ["date"])
        df = pd.read_csv(path, parse_dates=parse_dates, **kwargs)
    log.info(f"Dataset cargado: {path.name}  shape={df.shape}")
    return df


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    """Guarda un DataFrame a CSV o Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    log.info(f"Dataset guardado: {path.name}  shape={df.shape}")


# --- Modelos / Artefactos ----------------------------------------------------

def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    log.info(f"Artefacto guardado (pickle): {path.name}")


def load_pickle(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artefacto no encontrado: {path}")
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    log.info(f"Artefacto cargado (pickle): {path.name}")
    return obj


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
    log.info(f"JSON guardado: {path.name}")


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    return obj


# --- Splits ------------------------------------------------------------------

def save_splits(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    split_dir: str | Path,
) -> None:
    """Persiste los splits X/y de train/test junto con la lista de features."""
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "X_train.npy", X_train)
    np.save(split_dir / "y_train.npy", y_train)
    np.save(split_dir / "X_test.npy", X_test)
    np.save(split_dir / "y_test.npy", y_test)
    save_json(feature_names, split_dir / "feature_names.json")
    log.info(f"Splits guardados en: {split_dir}")


def load_splits(split_dir: str | Path):
    """Carga los splits guardados con save_splits."""
    split_dir = Path(split_dir)
    X_train = np.load(split_dir / "X_train.npy")
    y_train = np.load(split_dir / "y_train.npy")
    X_test  = np.load(split_dir / "X_test.npy")
    y_test  = np.load(split_dir / "y_test.npy")
    features = load_json(split_dir / "feature_names.json")
    log.info(f"Splits cargados: train={X_train.shape}  test={X_test.shape}")
    return X_train, y_train, X_test, y_test, features
