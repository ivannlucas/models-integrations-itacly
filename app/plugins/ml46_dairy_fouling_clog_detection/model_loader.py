"""Loads ml46 (DNSL TCN, no_clock scenario) artifacts via ArtifactStore."""
from __future__ import annotations

import json
import logging
from dataclasses import fields

import torch

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.artifact_validation import (
    validate_feature_artifacts,
    validate_policy_artifact,
)
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.common import FeatureArtifacts, TrainConfig
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.model_arch import PredictiveTCN, validate_checkpoint_compatibility
from app.plugins.ml46_dairy_fouling_clog_detection.constants import (
    ARTIFACT_FOLDER_NAME,
    FEATURE_ARTIFACTS_FILENAME,
    MODEL_FILENAME,
    MODEL_MANIFEST_FILENAME,
    POLICY_THRESHOLDS_FILENAME,
    SCENARIO,
    TRAINING_CONFIG_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


def _load_json(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_train_config() -> TrainConfig:
    """Load TrainConfig from training_config.json, ignoring unknown/legacy keys."""
    data = _load_json(_store.path(TRAINING_CONFIG_FILENAME))
    data["device"] = "cpu"
    known = {f.name for f in fields(TrainConfig)}
    filtered = {k: v for k, v in data.items() if k in known}
    if "dilations" in filtered:
        filtered["dilations"] = tuple(filtered["dilations"])
    return TrainConfig(**filtered)


def load_feature_artifacts() -> FeatureArtifacts:
    """Load FeatureArtifacts from feature_artifacts.json."""
    data = _load_json(_store.path(FEATURE_ARTIFACTS_FILENAME))
    known = {f.name for f in fields(FeatureArtifacts)}
    filtered = {k: v for k, v in data.items() if k in known}
    return FeatureArtifacts(**filtered)


def load_policy_thresholds() -> dict:
    """Load the calibrated no_clock alert-policy thresholds."""
    return _load_json(_store.path(POLICY_THRESHOLDS_FILENAME))


def load_manifest() -> dict:
    """Load model_manifest.json — the artifact/architecture contract for the served checkpoint."""
    return _load_json(_store.path(MODEL_MANIFEST_FILENAME))


def get_expected_architecture(manifest: dict, scenario: str) -> dict | None:
    """Pull the scenario's architecture contract out of model_manifest.json, if present."""
    artifact_contract = manifest.get("artifact_contract")
    if not isinstance(artifact_contract, dict):
        return None
    scenario_contracts = artifact_contract.get("scenario_contracts", {})
    sc_contract = scenario_contracts.get(scenario) if isinstance(scenario_contracts, dict) else None
    if not isinstance(sc_contract, dict):
        return None
    architecture = sc_contract.get("architecture")
    return architecture if isinstance(architecture, dict) else None


def build_model(train_cfg: TrainConfig, feature_artifacts: FeatureArtifacts) -> PredictiveTCN:
    """Instantiate the TCN architecture for the no_clock feature set (76 features)."""
    return PredictiveTCN(
        n_features=len(feature_artifacts.no_clock_feature_names),
        channels=int(train_cfg.channels),
        dilations=tuple(train_cfg.dilations),
        dropout=float(train_cfg.dropout),
    )


def load_artifacts() -> tuple[PredictiveTCN, TrainConfig, FeatureArtifacts, dict, dict]:
    """Load train_cfg, feature_artifacts, policy, model_manifest and the no_clock checkpoint.

    Validates the feature/policy artifacts and the checkpoint's architecture against
    model_manifest.json before serving them — catches an incompatible or mismatched
    artifact bundle at startup instead of failing silently/obscurely at predict time.

    Returns (model, train_cfg, feature_artifacts, policy, manifest).
    """
    train_cfg = load_train_config()
    feature_artifacts = load_feature_artifacts()
    policy = load_policy_thresholds()
    manifest = load_manifest()

    validate_feature_artifacts(feature_artifacts, SCENARIO, feature_artifacts.no_clock_feature_names)
    validate_policy_artifact(policy, SCENARIO)

    model = build_model(train_cfg, feature_artifacts)
    state = torch.load(_store.path(MODEL_FILENAME), map_location="cpu", weights_only=True)
    validate_checkpoint_compatibility(
        model, state,
        architecture_contract=get_expected_architecture(manifest, SCENARIO),
        feature_names=feature_artifacts.no_clock_feature_names,
        scenario=SCENARIO,
    )
    model.load_state_dict(state)
    model.eval()

    logger.info(
        "ml46 artifacts loaded — scenario=no_clock n_features=%d channels=%d",
        len(feature_artifacts.no_clock_feature_names), train_cfg.channels,
    )
    return model, train_cfg, feature_artifacts, policy, manifest
