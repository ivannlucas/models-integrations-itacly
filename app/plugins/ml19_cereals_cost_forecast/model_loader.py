"""Artifact loading for ml19 — faithful port of src/predict/predict.py::load_artifacts_src()
and src/utils/io.py::validate_artifacts_compatibility() from the delivered code.

Loads training_meta.json (features list + best_reg/best_clf per horizon + train_end), the 6
per-horizon .pkl bundles ({"model", "scaler", "model_name"}) it references by name, and the
reference dataset (dataset_v7_fe.csv) bundled alongside the model artifacts — see constants.py
module docstring and inbox/a19/manifest.yaml for why this dataset is bundled instead of built
from client-supplied data. v0 serves the delivered dataset (last row 2026-01); every /train
publishes a new base version with the dataset it was trained on.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml19_cereals_cost_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
    DATASET_FE_FILENAME,
    DATE_COL,
    TRAINING_META_FILENAME,
    clf_artifact_filename,
    reg_artifact_filename,
)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def _patch_sklearn_version_gap(model) -> None:
    """Repo review finding (PR #20): the repo pins scikit-learn==1.5.2 (must match
    retech-lote2-xai-explicabilidad's pin, see requirements.txt), but these artifacts were
    serialized with scikit-learn==1.8.0, where LogisticRegression.multi_class is no longer an
    instance attribute. 1.5.2's predict()/predict_proba() still reads self.multi_class, so
    unpickling the artifact as-is raises AttributeError on every single prediction that touches
    the h1 classifier. For a strictly binary classifier (classes_ == [0, 1], verified for every
    best_clf_*.pkl shipped with this model) 'ovr' and 'multinomial' compute the identical
    sigmoid decision function, so restoring the attribute does not change any prediction --
    verified against the manifest's golden_cases (bit-identical pred/prob_up).
    """
    if type(model).__name__ == "LogisticRegression" and not hasattr(model, "multi_class"):
        model.multi_class = "auto"


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


def _load_bundle(resolve) -> dict:
    """Build the bundle with resolve(filename) -> path (ArtifactStore or a local directory).

    Returns dict: meta, features, reg (dict[int, dict]), clf (dict[int, dict]),
    best_reg, best_clf, dataset (pd.DataFrame, parsed 'date' column).
    """
    meta_path = resolve(TRAINING_META_FILENAME)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    features: list[str] = meta["features"]
    best_reg = {int(h): m for h, m in meta["best_reg"].items()}
    best_clf = {int(h): m for h, m in meta["best_clf"].items()}

    reg_artifacts: dict[int, dict] = {}
    for h, mname in best_reg.items():
        reg_artifacts[h] = joblib.load(resolve(reg_artifact_filename(h, mname)))
        _patch_sklearn_version_gap(reg_artifacts[h]["model"])

    clf_artifacts: dict[int, dict] = {}
    for h, mname in best_clf.items():
        clf_artifacts[h] = joblib.load(resolve(clf_artifact_filename(h, mname)))
        _patch_sklearn_version_gap(clf_artifacts[h]["model"])

    _validate_artifacts_compatibility(features, reg_artifacts, clf_artifacts)

    dataset = pd.read_csv(resolve(DATASET_FE_FILENAME), parse_dates=[DATE_COL])

    return {
        "meta": meta,
        "features": features,
        "reg": reg_artifacts,
        "clf": clf_artifacts,
        "best_reg": best_reg,
        "best_clf": best_clf,
        "dataset": dataset,
    }


def load_artifact_bundle() -> dict:
    """Load the base bundle of the active version (training_meta.json, the 6 per-horizon
    reg/clf .pkl it names and the reference dataset_v7_fe.csv) via ArtifactStore."""
    return _load_bundle(_store.path)


def load_bundle_from_dir(directory: str | Path) -> dict:
    """Load a bundle written by training.save_training_artifacts (a retrained model from MLflow)."""
    base = Path(directory)
    return _load_bundle(lambda name: base / name)
