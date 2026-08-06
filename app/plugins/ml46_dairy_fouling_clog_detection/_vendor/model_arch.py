"""Vendored verbatim from inbox/a46/codigo/.../src/training/model.py.

Must match the delivered checkpoint bit-for-bit — do not refactor layer names/order.
architecture_signature()/state_shape_signature() add plain-Python instance attributes only
(no new nn.Parameter/buffer), so they don't affect state_dict() or checkpoint compatibility.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.plugins.ml46_dairy_fouling_clog_detection._vendor.artifact_validation import (
    feature_names_hash,
    payload_sha256,
)


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
        self.n_features = int(n_features)
        self.channels = int(channels)
        self.kernel_size = 3
        self.dropout = float(dropout)
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

    def architecture_signature(self) -> dict[str, Any]:
        """Structural fingerprint compared against model_manifest.json's scenario contract."""
        return {
            "model_class": self.__class__.__name__,
            "n_features": self.n_features,
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "dilations": list(self.dilations),
            "dropout": self.dropout,
            "receptive_field_steps": self.receptive_field(),
            "heads": {
                "severity_head": 1,
                "stage_head": 3,
                "foul_h_head": 1,
                "actionable_foul_h_head": 1,
                "clog_h_head": 1,
                "tte_foul_head": 1,
                "tte_clog_head": 1,
                "ttu_head": 1,
            },
        }

    def state_shape_signature(self) -> dict[str, list[int]]:
        """Tensor-name -> shape map of this (freshly built, unloaded) model instance."""
        return {name: list(param.shape) for name, param in self.state_dict().items()}

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


def validate_checkpoint_compatibility(
    model: PredictiveTCN,
    state_dict: Mapping[str, torch.Tensor],
    architecture_contract: Mapping[str, Any] | None = None,
    feature_names: Sequence[str] | None = None,
    scenario: str | None = None,
) -> dict[str, Any]:
    """Validate that checkpoint tensors, architecture and feature contract agree.

    This check is intentionally performed before load_state_dict so an incompatible
    checkpoint cannot be silently used with a different feature layout or TCN shape.
    architecture_contract (from model_manifest.json's scenario_contracts) is optional —
    when absent (e.g. an MLflow user fine-tuned bundle with no manifest of its own), only
    the tensor-shape check against the freshly-built model instance runs.
    """
    expected_shapes = model.state_shape_signature()
    observed_shapes = {name: list(tensor.shape) for name, tensor in state_dict.items() if hasattr(tensor, "shape")}
    missing = sorted(set(expected_shapes) - set(observed_shapes))
    unexpected = sorted(set(observed_shapes) - set(expected_shapes))
    shape_mismatches = {
        name: {"expected": expected_shapes[name], "observed": observed_shapes[name]}
        for name in sorted(set(expected_shapes) & set(observed_shapes))
        if expected_shapes[name] != observed_shapes[name]
    }
    errors: list[str] = []
    if missing:
        errors.append(f"Missing checkpoint tensors: {missing[:10]}")
    if unexpected:
        errors.append(f"Unexpected checkpoint tensors: {unexpected[:10]}")
    if shape_mismatches:
        first_items = list(shape_mismatches.items())[:5]
        errors.append(f"Tensor shape mismatches: {first_items}")

    model_signature = model.architecture_signature()
    feature_hash = feature_names_hash(feature_names or []) if feature_names is not None else ""
    if architecture_contract:
        contract_arch = dict(architecture_contract.get("architecture", {}))
        if contract_arch:
            comparable_keys = ["model_class", "n_features", "channels", "kernel_size", "dilations", "heads"]
            for key in comparable_keys:
                if contract_arch.get(key) != model_signature.get(key):
                    errors.append(
                        f"Architecture contract mismatch for '{key}': "
                        f"expected {contract_arch.get(key)}, observed {model_signature.get(key)}"
                    )
        contract_feature_hash = architecture_contract.get("feature_names_hash")
        if feature_names is not None and contract_feature_hash and contract_feature_hash != feature_hash:
            errors.append(f"Feature list hash mismatch: expected {contract_feature_hash}, observed {feature_hash}")
        contract_scenario = architecture_contract.get("scenario")
        if scenario is not None and contract_scenario and str(contract_scenario) != str(scenario):
            errors.append(f"Scenario contract mismatch: expected {contract_scenario}, observed {scenario}")
        contract_state_hash = architecture_contract.get("state_shape_hash")
        observed_state_hash = payload_sha256(expected_shapes)
        if contract_state_hash and contract_state_hash != observed_state_hash:
            errors.append(f"State-shape contract mismatch: expected {contract_state_hash}, observed {observed_state_hash}")

    report = {
        "ok": not errors,
        "scenario": scenario,
        "architecture": model_signature,
        "feature_names_hash": feature_hash,
        "state_shape_hash": payload_sha256(expected_shapes),
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
        "shape_mismatches": shape_mismatches,
        "errors": errors,
    }
    if errors:
        raise ValueError("Checkpoint/model compatibility check failed: " + "; ".join(errors))
    return report
