"""Artifact loading for ml19 — faithful port of src/predict/predict.py::load_artifacts_src()
and src/utils/io.py::validate_artifacts_compatibility() from the delivered code.

Loads training_meta.json (features list + best_reg/best_clf per horizon + train_end), the 6
per-horizon .pkl bundles ({"model", "scaler", "model_name"}) it references by name, and the
frozen reference dataset (dataset_v7_fe.csv) bundled alongside the model artifacts — see
constants.py module docstring and inbox/a19/manifest.yaml for why this dataset is bundled
instead of built from client-supplied data.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml19_cereals_cost_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
    DATASET_FE_FILENAME,
    DATE_COL,
    TRAINING_META_FILENAME,
)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def _get_model_expected_n_features(model) -> int | None:
    """io.py::_get_model_expected_n_features — n_features_in_ of a sklearn/xgboost model."""
    val = getattr(model, "n_features_in_", None)
    if isinstance(val, (int, np.integer)):
        return int(val)
    return None


def _validate_artifacts_compatibility(features: list[str], reg: dict, clf: dict) -> None:
    """io.py::validate_artifacts_compatibility — training_meta vs. .pkl coherence."""
    n_features = len(features)
    if n_features == 0:
        raise ValueError("training_meta.json contiene una lista de features vacia.")

    for task_key, bundles in (("reg", reg), ("clf", clf)):
        for h, art in bundles.items():
            sc = art.get("scaler")
            sc_n = getattr(sc, "n_features_in_", None) if sc is not None else None
            if isinstance(sc_n, (int, np.integer)) and int(sc_n) != n_features:
                raise ValueError(
                    f"Incompatibilidad scaler en {task_key} h={h}: espera {int(sc_n)} "
                    f"features pero training_meta tiene {n_features}."
                )
            mdl_n = _get_model_expected_n_features(art.get("model"))
            if mdl_n is not None and mdl_n != n_features:
                raise ValueError(
                    f"Incompatibilidad modelo en {task_key} h={h}: espera {mdl_n} "
                    f"features pero training_meta tiene {n_features}."
                )


def load_artifact_bundle() -> dict:
    """Load training_meta.json + per-horizon reg/clf bundles + the frozen reference dataset.

    Returns dict: meta, features, reg (dict[int, dict]), clf (dict[int, dict]),
    best_reg, best_clf, dataset (pd.DataFrame, parsed 'date' column).
    """
    meta_path = _store.path(TRAINING_META_FILENAME)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    features: list[str] = meta["features"]
    best_reg = {int(h): m for h, m in meta["best_reg"].items()}
    best_clf = {int(h): m for h, m in meta["best_clf"].items()}

    reg_artifacts: dict[int, dict] = {}
    for h, mname in best_reg.items():
        path = _store.path(f"best_reg_h{h}m_{mname.lower()}.pkl")
        reg_artifacts[h] = joblib.load(path)

    clf_artifacts: dict[int, dict] = {}
    for h, mname in best_clf.items():
        path = _store.path(f"best_clf_h{h}m_{mname.lower()}.pkl")
        clf_artifacts[h] = joblib.load(path)

    _validate_artifacts_compatibility(features, reg_artifacts, clf_artifacts)

    dataset = pd.read_csv(_store.path(DATASET_FE_FILENAME), parse_dates=[DATE_COL])

    return {
        "meta": meta,
        "features": features,
        "reg": reg_artifacts,
        "clf": clf_artifacts,
        "best_reg": best_reg,
        "best_clf": best_clf,
        "dataset": dataset,
    }
