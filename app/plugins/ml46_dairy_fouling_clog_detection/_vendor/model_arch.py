"""Vendored verbatim from inbox/a46/codigo/.../src/training/model.py.

Must match the delivered checkpoint bit-for-bit — do not refactor layer names/order.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """Trims the right-padding added by a causal dilated Conv1d."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Remove the last chomp_size timesteps so causality is strict."""
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """Two causal dilated Conv1d layers with residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block and add the (possibly downsampled) residual."""
        out = self.net(x)
        res = self.down(x)
        return self.relu(out + res)


class PredictiveTCN(nn.Module):
    """Causal multi-output TCN: severity + stage + 3 binary horizons + 3 time-to-event heads."""

    def __init__(self, n_features: int, channels: int, dilations: Sequence[int], dropout: float) -> None:
        super().__init__()
        self.dilations = tuple(int(d) for d in dilations)
        layers: List[nn.Module] = []
        in_ch = n_features
        for d in self.dilations:
            layers.append(TemporalBlock(in_ch, channels, kernel_size=3, dilation=d, dropout=dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*layers)
        self.shared = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.severity_head = nn.Linear(channels, 1)
        self.stage_head = nn.Linear(channels, 3)
        self.foul_h_head = nn.Linear(channels, 1)
        self.actionable_foul_h_head = nn.Linear(channels, 1)
        self.clog_h_head = nn.Linear(channels, 1)
        self.tte_foul_head = nn.Linear(channels, 1)
        self.tte_clog_head = nn.Linear(channels, 1)
        self.ttu_head = nn.Linear(channels, 1)

    def receptive_field(self) -> int:
        """Theoretical receptive field in timesteps given the block dilations."""
        rf = 1
        for d in self.dilations:
            rf += 2 * d * 2
        return rf

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the TCN over a (batch, seq_len, n_features) window and return all 8 heads."""
        z = x.transpose(1, 2)
        h = self.tcn(z)
        h_last = h[:, :, -1]
        h_last = self.shared(h_last)
        return {
            "severity_scaled": F.softplus(self.severity_head(h_last).squeeze(-1)),
            "stage_logits": self.stage_head(h_last),
            "foul_h_logit": self.foul_h_head(h_last).squeeze(-1),
            "actionable_foul_h_logit": self.actionable_foul_h_head(h_last).squeeze(-1),
            "clog_h_logit": self.clog_h_head(h_last).squeeze(-1),
            "tte_foul_log": self.tte_foul_head(h_last).squeeze(-1),
            "tte_clog_log": self.tte_clog_head(h_last).squeeze(-1),
            "ttu_log": self.ttu_head(h_last).squeeze(-1),
        }
