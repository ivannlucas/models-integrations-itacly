"""GRU architecture vendored from model-runtime-modelo23 (a23-rnn-dairy-prediccion).

Only GRUModel is included — the artifact uses GRU exclusively.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """Stacked GRU network for direct future-price prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """Initialize GRU layers, dropout and final linear projection."""
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass; returns scalar prediction per batch element."""
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)
