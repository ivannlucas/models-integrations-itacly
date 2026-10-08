"""Loads the ml26 GRU-PSO bundle via ArtifactStore and (de)serializes user fine-tuned bundles."""
# pylint: disable=duplicate-code  # LoadedModel mirrors the vendored SequenceBundle fields on purpose

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml26_wine_sulfite_gru_pso_forecast._vendor.gru_model import (
    GRUMultiTaskRegressor,
    build_gru_from_config,
)
from app.plugins.ml26_wine_sulfite_gru_pso_forecast._vendor.sequence_bundle import (
    SequenceBundle,
    load_sequence_bundle,
)
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
    BEST_MODEL_FILENAME,
    MODEL_FILENAME,
    TARGET_NAMES,
    USER_META_FILENAME,
    USER_STATE_FILENAME,
    USER_TRAINED_SUBDIR,
    WINDOW,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


@dataclass
class LoadedModel:  # pylint: disable=too-many-instance-attributes
    """Ready-to-serve model: eval-mode GRU + the normalization/feature contract it was trained
    with.
    """

    network: GRUMultiTaskRegressor
    feature_names: list[str]
    target_names: list[str]
    mean: np.ndarray  # (1, n_features) z-score mean over train windows
    std: np.ndarray  # (1, n_features)
    y_mean: np.ndarray  # (n_targets,)
    y_std: np.ndarray  # (n_targets,)
    config: dict[str, Any]
    source: str  # "fixed" (AI team artifact) or "mlflow:<run_id>"


def _validate_contract(
    feature_names: list[str], target_names: list[str], registry: dict | None
) -> None:
    if list(target_names) != TARGET_NAMES:
        raise ValueError(
            f"Unexpected target_names in bundle: {target_names} (expected {TARGET_NAMES})"
        )
    if registry is not None:
        expected = list(registry.get("feature_columns", []))
        if expected and expected != list(feature_names):
            raise ValueError(
                "gru_pso.pkl feature_names do not match best_model.json feature_columns"
            )


def _build_network(
    config: dict, state: dict, n_features: int, n_targets: int
) -> GRUMultiTaskRegressor:
    network = build_gru_from_config(config, input_dim=n_features, output_dim=n_targets)
    network.load_state_dict(state)
    network.eval()
    return network


def loaded_from_sequence_bundle(
    bundle: SequenceBundle, source: str = "fixed", registry: dict | None = None
) -> LoadedModel:
    """Turn the AI team's SequenceBundle into a LoadedModel (validates the feature/target
    contract).
    """
    _validate_contract(bundle.feature_names, bundle.target_names, registry)
    network = _build_network(
        bundle.config, bundle.model_state, len(bundle.feature_names), len(bundle.target_names)
    )
    return LoadedModel(
        network=network,
        feature_names=list(bundle.feature_names),
        target_names=list(bundle.target_names),
        mean=np.asarray(bundle.mean, dtype=np.float32),
        std=np.asarray(bundle.std, dtype=np.float32),
        y_mean=np.asarray(bundle.y_mean, dtype=np.float32),
        y_std=np.asarray(bundle.y_std, dtype=np.float32),
        config=dict(bundle.config),
        source=source,
    )


def load_artifacts() -> tuple[LoadedModel, dict]:
    """Load the fixed gru_pso.pkl + best_model.json. Returns (model, registry)."""
    with open(_store.path(BEST_MODEL_FILENAME), encoding="utf-8") as fh:
        registry = json.load(fh)
    bundle = load_sequence_bundle(_store.path(MODEL_FILENAME))
    model = loaded_from_sequence_bundle(bundle, source="fixed", registry=registry)
    if len(model.feature_names) != 22 or int(model.config.get("input_dim", 22)) != 22:
        raise ValueError("gru_pso bundle does not expose the expected 22 input features")
    logger.info(
        "ml26 artifacts loaded — %s hidden_dim=%s layers=%s bidirectional=%s window=%d",
        registry.get("best_model_name"),
        model.config.get("hidden_dim"),
        model.config.get("num_layers"),
        model.config.get("bidirectional"),
        WINDOW,
    )
    return model, registry


def user_trained_dir(run_name: str) -> Path:
    """Local folder for a user fine-tuned model — a sub-folder, so the fixed artifact is never
    overwritten.
    """
    return _store.local_dir / USER_TRAINED_SUBDIR / run_name


def save_user_model(model: LoadedModel, directory: str | Path) -> Path:
    """Persist a fine-tuned model as state_dict (.pt) + JSON metadata — no pickle involved."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.network.state_dict(), out / USER_STATE_FILENAME)
    meta = {
        "model_name": "gru",
        "feature_names": model.feature_names,
        "target_names": model.target_names,
        "mean": np.asarray(model.mean).tolist(),
        "std": np.asarray(model.std).tolist(),
        "y_mean": np.asarray(model.y_mean).tolist(),
        "y_std": np.asarray(model.y_std).tolist(),
        "config": model.config,
    }
    with open(out / USER_META_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return out


def load_user_model(directory: str | Path, source: str) -> LoadedModel:
    """Load a model saved with save_user_model (tensors with weights_only=True)."""
    base = Path(directory)
    with open(base / USER_META_FILENAME, encoding="utf-8") as fh:
        meta = json.load(fh)
    _validate_contract(meta["feature_names"], meta["target_names"], None)
    state = torch.load(base / USER_STATE_FILENAME, map_location="cpu", weights_only=True)
    network = _build_network(
        meta["config"], state, len(meta["feature_names"]), len(meta["target_names"])
    )
    return LoadedModel(
        network=network,
        feature_names=list(meta["feature_names"]),
        target_names=list(meta["target_names"]),
        mean=np.asarray(meta["mean"], dtype=np.float32),
        std=np.asarray(meta["std"], dtype=np.float32),
        y_mean=np.asarray(meta["y_mean"], dtype=np.float32),
        y_std=np.asarray(meta["y_std"], dtype=np.float32),
        config=dict(meta["config"]),
        source=source,
    )
