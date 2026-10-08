"""GRU multitask regressor with attention pooling (vendored from
src/training/models/gru_model.py).
"""

from __future__ import annotations

import torch
from torch import nn


class AttentionPooling(nn.Module):
    """Softmax attention over the time axis of the recurrent outputs."""

    def __init__(self, hidden_dim: int) -> None:
        """Build the scoring layer."""
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, sequence_outputs: torch.Tensor) -> torch.Tensor:
        """Pool (batch, time, hidden) into (batch, hidden)."""
        attention_logits = self.score(sequence_outputs).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
        return torch.sum(sequence_outputs * attention_weights, dim=1)


class GRUMultiTaskRegressor(nn.Module):
    """GRU -> AttentionPooling -> LayerNorm/Linear/ReLU/Dropout/Linear head with one output per
    target.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
        bidirectional: bool = False,
        head_dropout: float | None = None,
    ) -> None:
        """Build the network exactly as the AI team did (dropout between GRU layers only if >1
        layer).
        """
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        head_dim = hidden_dim * direction_factor
        self.pool = AttentionPooling(head_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout if head_dropout is None else head_dropout),
            nn.Linear(head_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map (batch, window, features) normalized inputs to (batch, targets) normalized
        outputs.
        """
        outputs, _ = self.gru(inputs)
        pooled = self.pool(outputs)
        return self.head(pooled)


def build_gru_from_config(config: dict, input_dim: int, output_dim: int) -> GRUMultiTaskRegressor:
    """Instantiate the GRU from a SequenceBundle.config dict (same keys as
    models/metrics/gru_pso.json).
    """
    return GRUMultiTaskRegressor(
        input_dim=input_dim,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        output_dim=output_dim,
        bidirectional=bool(config.get("bidirectional", False)),
        head_dropout=float(config.get("head_dropout", config["dropout"])),
    )
