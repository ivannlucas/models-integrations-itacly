"""Loads modelo30 (neuroevolution MLP) artifacts for traceability scoring via ArtifactStore."""
import json
import logging
import sys
import types

import joblib
import torch
from torch import nn

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml30_meat_traceability_detection import tabular_preprocessor
from app.plugins.ml30_meat_traceability_detection.constants import (
    ARTIFACT_FOLDER_NAME,
    FEATURE_COLUMNS,
    MODEL_FILENAME,
    MODEL_PAYLOAD_FILENAME,
    PREPROCESSOR_FILENAME,
)

logger = logging.getLogger(__name__)

# preprocessor.pkl was pickled as `src.training.preprocessing.TabularPreprocessor` in the
# training repo. Alias that module path to our vendored copy so joblib.load resolves it.
sys.modules.setdefault("src", types.ModuleType("src"))
sys.modules.setdefault("src.training", types.ModuleType("src.training"))
sys.modules["src.training.preprocessing"] = tabular_preprocessor

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


class TorchBinaryMLP(nn.Module):
    """MLP whose layers live under ``self.network`` (matches checkpoint keys ``network.N.*``)."""

    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Return raw logits for the input batch."""
        return self.network(x)


def build_torch_mlp(payload: dict) -> TorchBinaryMLP:
    """Reconstruct the MLP architecture described in model_payload.json."""
    in_dim = int(payload["input_dim"])
    activation = payload.get("activation", "relu")
    layers: list[nn.Module] = []
    for hidden in [int(h) for h in payload.get("hidden_layers", [])]:
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.Tanh() if activation == "tanh" else nn.ReLU())
        in_dim = hidden
    layers.append(nn.Linear(in_dim, 1))
    return TorchBinaryMLP(layers)


def load_payload() -> dict:
    """Read and return the model architecture payload."""
    with open(_store.path(MODEL_PAYLOAD_FILENAME), encoding="utf-8") as fh:
        return json.load(fh)


def load_artifacts():
    """Load preprocessor, MLP and feature columns. Returns (preprocessor, mlp, feature_columns)."""
    preprocessor = joblib.load(_store.path(PREPROCESSOR_FILENAME))
    payload = load_payload()
    mlp = build_torch_mlp(payload)
    state_dict = torch.load(_store.path(MODEL_FILENAME), map_location="cpu", weights_only=True)
    mlp.load_state_dict(state_dict)
    mlp.eval()
    logger.info(
        "Ml30 artifacts loaded — input_dim=%d hidden=%s",
        payload["input_dim"], payload.get("hidden_layers"),
    )
    return preprocessor, mlp, FEATURE_COLUMNS
