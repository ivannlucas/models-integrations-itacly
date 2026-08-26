"""Data preparation utilities for reproducible H1/H2/H3 experiments.

This module is the single source of truth for model-ready train/test splits
used by training scripts.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from config.config import DATA_PROCESSED_DIR, RANDOM_SEED

RANDOM_STATE: int = int(RANDOM_SEED)
CUT_DATE: pd.Timestamp = pd.Timestamp("2021-01-01")
BASE_PRICE_COL: str = "precio_provincial_lag_1"

BLACKLIST: list[str] = [
    "precio_provincial_lag_1",
    "precio_provincial_lag_2",
    "precio_provincial_lag_3",
    "precio_vecinos_media_lag1",
    "precio_nacional_base_ma3",
    "precio_nacional_base_ma6",
    "precio_nacional_base_vol3",
    "precio_nacional_base_vol6",
]

VALID_HORIZONS: tuple[int, ...] = (1, 2, 3)
VALID_TASKS: tuple[str, ...] = ("regresion", "clasificacion")


def _set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Apply deterministic seed settings across supported RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _load_base_dataset() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "dataset_entrenamiento_final.csv"
    if not path.exists():
        raise FileNotFoundError(f"No se encontro dataset base en: {path}")

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("El dataset debe contener la columna 'date'.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values(["date", "provincia", "cereal_predominante"], kind="mergesort")
    df = df.reset_index(drop=True)
    return df


def _target_col(horizonte: int) -> str:
    return f"precio_provincial_TARGET_H{horizonte}"


def _add_targets(df: pd.DataFrame, horizonte: int) -> pd.DataFrame:
    """Create return and direction targets for a given horizon."""
    tgt_col = _target_col(horizonte)
    if tgt_col not in df.columns:
        raise ValueError(f"No existe columna de target para H{horizonte}: {tgt_col}")
    if BASE_PRICE_COL not in df.columns:
        raise ValueError(f"No existe columna base para retornos: {BASE_PRICE_COL}")

    data = df.copy()
    base = pd.to_numeric(data[BASE_PRICE_COL], errors="coerce")
    future = pd.to_numeric(data[tgt_col], errors="coerce")

    valid = base.notna() & future.notna() & base.ne(0)
    data = data.loc[valid].copy()

    ret_col = f"target_return_h{horizonte}"
    clf_col = f"target_class_h{horizonte}"

    data[ret_col] = (data[tgt_col] - data[BASE_PRICE_COL]) / data[BASE_PRICE_COL]
    data[clf_col] = (data[ret_col] > 0).astype(int)
    return data


def _build_features(df: pd.DataFrame, horizonte: int) -> pd.DataFrame:
    """Build strict signal-pure features, preserving deterministic schema."""
    drop_cols = set(BLACKLIST)
    drop_cols.add("date")

    for h in VALID_HORIZONS:
        drop_cols.add(_target_col(h))
        drop_cols.add(f"target_return_h{h}")
        drop_cols.add(f"target_class_h{h}")

    keep_cols = [c for c in df.columns if c not in drop_cols]
    feat = df[keep_cols].copy()

    # Keep deterministic dtypes and encoding for object/categorical columns.
    obj_cols = feat.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in obj_cols:
        feat[col] = feat[col].astype(str)
    feat = pd.get_dummies(feat, columns=obj_cols, drop_first=False)

    # Stable column order guarantees reproducible model inputs.
    feat = feat.reindex(sorted(feat.columns), axis=1)

    if not feat.shape[1]:
        raise ValueError(
            f"No quedan features tras blacklist para H{horizonte}. "
            "Revisa reglas de filtrado."
        )
    return feat


def _split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = df["date"] < CUT_DATE
    test_mask = df["date"] >= CUT_DATE
    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            "Split temporal vacio. Verifica fechas y punto de corte 2021-01-01."
        )
    return train_df, test_df


def get_prepared_data(
    horizonte: int,
    tarea: Literal["regresion", "clasificacion"],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return model-ready train/test sets for one horizon and task.

    Args:
        horizonte: Prediction horizon in months (1, 2, 3).
        tarea: "regresion" for return target or "clasificacion" for up/down.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    _set_global_seed(RANDOM_STATE)

    if horizonte not in VALID_HORIZONS:
        raise ValueError(f"Horizonte invalido: {horizonte}. Usa {VALID_HORIZONS}.")
    if tarea not in VALID_TASKS:
        raise ValueError(f"Tarea invalida: {tarea}. Usa {VALID_TASKS}.")

    full = _load_base_dataset()
    full = _add_targets(full, horizonte)
    train_df, test_df = _split_temporal(full)

    y_col = f"target_return_h{horizonte}" if tarea == "regresion" else f"target_class_h{horizonte}"

    X_train = _build_features(train_df, horizonte)
    X_test_raw = _build_features(test_df, horizonte)

    # Align test schema to training schema to avoid category drift issues.
    X_test = X_test_raw.reindex(columns=X_train.columns, fill_value=0)

    y_train = train_df[y_col].astype(float if tarea == "regresion" else int)
    y_test = test_df[y_col].astype(float if tarea == "regresion" else int)

    return X_train, X_test, y_train, y_test


def save_prepared_splits(output_dir: Path | None = None) -> None:
    """Persist all prepared splits for downstream training scripts."""
    _set_global_seed(RANDOM_STATE)
    out_dir = output_dir or (DATA_PROCESSED_DIR / "splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    for horizonte in VALID_HORIZONS:
        for tarea in VALID_TASKS:
            X_train, X_test, y_train, y_test = get_prepared_data(horizonte, tarea)  # noqa: PLW2901

            train_export = X_train.copy()
            test_export = X_test.copy()
            target_name = y_train.name if y_train.name else "target"

            train_export[target_name] = y_train.to_numpy()
            test_export[target_name] = y_test.to_numpy()

            train_path = out_dir / f"train_h{horizonte}_{tarea}.csv"
            test_path = out_dir / f"test_h{horizonte}_{tarea}.csv"

            train_export.to_csv(train_path, index=False)
            test_export.to_csv(test_path, index=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepara splits reproducibles H1/H2/H3 para regresion y clasificacion."
    )
    parser.add_argument(
        "--save-splits",
        action="store_true",
        help="Guarda todos los splits en data/processed/splits/.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directorio de salida opcional para --save-splits.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.save_splits:
        out = Path(args.output_dir) if args.output_dir else None
        save_prepared_splits(out)
        print("Splits preparados y guardados correctamente.")
        return

    for horizonte in VALID_HORIZONS:
        for tarea in VALID_TASKS:
            X_train, X_test, y_train, y_test = get_prepared_data(horizonte, tarea)
            print(
                f"H{horizonte} {tarea}: "
                f"X_train={X_train.shape}, X_test={X_test.shape}, "
                f"y_train={y_train.shape}, y_test={y_test.shape}"
            )


if __name__ == "__main__":
    main()