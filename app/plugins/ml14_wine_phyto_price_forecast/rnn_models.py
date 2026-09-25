"""LSTM architecture vendored from a14-rnn-vitivinicola-precios-mercado-fitosanitarios
(src/training/compare_models.py::LSTMModel). Only LSTMModel is included — the served
artifact is the LSTM (see constants.py::MODEL_NAME and inbox/a14/manifest.yaml::model_status);
GRU/XGBoost were comparison-only architectures, not selected for production.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """Stacked LSTM network predicting a scaled price residual from the last timestep."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """Initialize LSTM layers, dropout and final linear projection."""
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass; returns scalar prediction per batch element."""
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)
