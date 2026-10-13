"""Artifact loader for the ml36 dairy DNL CO2-emissions optimizer plugin."""
import json
import logging

import joblib
import torch
from torch import nn

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.constants import (
    ARTIFACT_FOLDER_NAME,
    MODEL_CONFIG_FILENAME,
    MODEL_FILENAME,
    POLICY_FILENAME,
    SCALER_X_FILENAME,
    SCALER_Y_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


class DynamicMLP(nn.Module):
    """MLP surrogate: [Linear + BatchNorm + activation + Dropout] x num_layers -> Linear.

    Mirrors the delivered src/training/model.py so MLP_Final_Best_From_Search.pt
    (state_dict keys ``net.*``) loads without remapping.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int,
        neurons: int,
        activation: str = "ReLU",
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, neurons))
            layers.append(nn.BatchNorm1d(neurons))
            layers.append(nn.ReLU() if activation == "ReLU" else nn.Tanh())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = neurons
        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the sequential MLP."""
        return self.net(x)


def build_model_from_config(config: dict) -> DynamicMLP:
    """Instantiate a DynamicMLP from a final_model_config.json dict."""
    return DynamicMLP(
        input_size=int(config["input_size"]),
        output_size=int(config["output_size"]),
        num_layers=int(config["num_layers"]),
        neurons=int(config["neurons"]),
        activation=str(config["activation"]),
        dropout_rate=float(config.get("dropout_rate", 0.1)),
    )


def load_policy() -> list[float]:
    """Load the global GA policy [T_serv, Delta_P, Regeneration_perc] (static mode)."""
    with open(_store.path(POLICY_FILENAME), "r", encoding="utf-8") as f:
        payload = json.load(f)
    best = payload.get("best_individual")
    if not isinstance(best, list) or len(best) != 3:
        raise ValueError(f"{POLICY_FILENAME} no contiene un best_individual de 3 genes.")
    return [float(g) for g in best]


def load_artifacts():
    """Load MLP weights, architecture config, scalers and the global GA policy.

    Returns (model in eval mode, scaler_X, scaler_Y, config_dict, policy).
    """
    with open(_store.path(MODEL_CONFIG_FILENAME), "r", encoding="utf-8") as f:
        config = json.load(f)

    model = build_model_from_config(config)
    model.load_state_dict(
        torch.load(_store.path(MODEL_FILENAME), map_location="cpu", weights_only=True)
    )
    model.eval()

    scaler_X = joblib.load(_store.path(SCALER_X_FILENAME))  # pylint: disable=invalid-name
    scaler_Y = joblib.load(_store.path(SCALER_Y_FILENAME))  # pylint: disable=invalid-name
    policy = load_policy()

    logger.info("ml36 artifacts loaded — DynamicMLP(%s layers x %s) + scalers + policy",
                config["num_layers"], config["neurons"])
    return model, scaler_X, scaler_Y, config, policy
