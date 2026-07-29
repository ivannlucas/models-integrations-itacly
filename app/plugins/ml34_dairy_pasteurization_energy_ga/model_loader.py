"""Artifact loader for the ml34 dairy pasteurization energy GA plugin."""
import json
import logging

import joblib
import torch
from torch import nn

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml34_dairy_pasteurization_energy_ga.constants import (
    ARTIFACT_FOLDER_NAME,
    MODEL_CONFIG_FILENAME,
    MODEL_FILENAME,
    SCALER_X_FILENAME,
    SCALER_Y_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


class DynamicMLP(nn.Module):
    """MLP surrogate dynamically built from hyperparameters (5 → hidden → 2).

    Mirrors the original src/training/model.py so the delivered
    ``mlp_predictor.pt`` state_dict loads without remapping.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int,
        neurons: int,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()
        layers = []
        in_features = input_size
        act_fn = nn.ReLU() if activation == "ReLU" else nn.Tanh()
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, neurons))
            layers.append(act_fn)
            in_features = neurons
        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the sequential MLP."""
        return self.net(x)


def build_model_from_config(config: dict) -> DynamicMLP:
    """Instantiate a DynamicMLP from a model_config.json dict (eval mode off)."""
    return DynamicMLP(
        input_size=config["input_size"],
        output_size=config["output_size"],
        num_layers=config["num_layers"],
        neurons=config["neurons"],
        activation=config["activation"],
    )


def load_artifacts():
    """Load MLP weights, architecture config and MinMax scalers.

    Returns (model in eval mode, scaler_X, scaler_y, config_dict).
    """
    with open(_store.path(MODEL_CONFIG_FILENAME), "r", encoding="utf-8") as f:
        config = json.load(f)

    model = build_model_from_config(config)
    model.load_state_dict(
        torch.load(_store.path(MODEL_FILENAME), map_location="cpu", weights_only=True)
    )
    model.eval()

    scaler_X = joblib.load(_store.path(SCALER_X_FILENAME))  # pylint: disable=invalid-name
    scaler_y = joblib.load(_store.path(SCALER_Y_FILENAME))

    logger.info("ml34 artifacts loaded — DynamicMLP(%s layers x %s) + scalers",
                config["num_layers"], config["neurons"])
    return model, scaler_X, scaler_y, config
