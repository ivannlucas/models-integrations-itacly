"""Artifact loader for ml35 dairy ANN cleaning-cost plugin."""
import logging

import joblib
import torch
import torch.nn as nn

from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml35_dairy_ann_cleaning_cost.constants import (
    ARTIFACT_FOLDER_NAME,
    FEATURES,
    MODEL_FILENAME,
    SCALER_X_FILENAME,
    SCALER_Y_FILENAME,
)

logger = logging.getLogger(__name__)

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)


class PasteurizationANN(nn.Module):
    """ANN for pasteurization water-consumption prediction (8→128→64→32→1)."""

    def __init__(self, input_size: int = 8) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def load_artifacts():
    """Load ANN weights and scalers from artifact storage.

    Returns (model, scaler_X, scaler_y).
    """
    model = PasteurizationANN(input_size=len(FEATURES))
    model.load_state_dict(
        torch.load(_store.path(MODEL_FILENAME), map_location="cpu", weights_only=True)
    )
    model.eval()

    scaler_X = joblib.load(_store.path(SCALER_X_FILENAME))
    scaler_y = joblib.load(_store.path(SCALER_Y_FILENAME))

    logger.info("ml35 artifacts loaded — ANN + scaler_X + scaler_y")
    return model, scaler_X, scaler_y
