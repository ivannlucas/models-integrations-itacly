"""Vendored from inbox/a46/codigo/.../src/training/datasets.py (verbatim algorithmic core)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from app.plugins.ml46_dairy_fouling_clog_detection._vendor.common import TrainConfig
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.utils_common import sequence_group_columns


@dataclass
class AssetSequence:
    """One continuous asset+cycle sequence with raw feature matrix and per-step targets."""

    sequence_id: str
    asset_id: str
    timestamps: np.ndarray
    x: np.ndarray
    y_severity: np.ndarray
    y_stage: np.ndarray
    y_foul_h: np.ndarray
    y_actionable_foul_h: np.ndarray
    y_clog_h: np.ndarray
    y_ttf_foul: np.ndarray
    y_ttf_clog: np.ndarray
    y_ttu: np.ndarray
    foul_weight: np.ndarray
    actionable_foul_weight: np.ndarray
    clog_weight: np.ndarray
    meta: pd.DataFrame


def make_sequences(feat_df: pd.DataFrame, cfg: TrainConfig, feature_cols: Sequence[str]) -> Dict[str, AssetSequence]:
    """Group the engineered dataframe into one AssetSequence per asset_id/sequence_id."""
    sequences: Dict[str, AssetSequence] = {}
    foul_col = f"fouling_onset_within_{cfg.fouling_horizon_min}min"
    clog_col = f"clog_onset_within_{cfg.clog_horizon_min}min"
    actionable_col = f"unplanned_fouling_within_{cfg.unplanned_fouling_horizon_min}min"
    group_cols = sequence_group_columns(feat_df)

    for _, g in feat_df.groupby(group_cols, sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True).copy()
        asset_id = str(g["asset_id"].iloc[0])
        if "sequence_id" in g.columns:
            sequence_id = str(g["sequence_id"].iloc[0])
        elif "cycle_id" in g.columns:
            sequence_id = asset_id + "::" + str(g["cycle_id"].iloc[0])
        else:
            sequence_id = asset_id

        severity_series = pd.to_numeric(g[cfg.severity_col], errors="coerce") if cfg.severity_col in g.columns else pd.Series(np.zeros(len(g)))
        y_severity = severity_series.fillna(0.0).astype(np.float32).to_numpy()
        if "fouling_stage_physical" not in g.columns:
            g["fouling_stage_physical"] = 0
        y_stage = pd.to_numeric(g["fouling_stage_physical"], errors="coerce").fillna(0).astype(int).clip(0, 2).to_numpy(dtype=np.int64)
        y_foul_h = pd.to_numeric(g.get(foul_col, 0), errors="coerce").fillna(0).astype(np.float32).clip(0.0, 1.0).to_numpy()
        y_actionable_foul_h = pd.to_numeric(g.get(actionable_col, 0), errors="coerce").fillna(0).astype(np.float32).clip(0.0, 1.0).to_numpy()
        y_clog_h = pd.to_numeric(g.get(clog_col, 0), errors="coerce").fillna(0).astype(np.float32).clip(0.0, 1.0).to_numpy()
        y_ttf_foul = pd.to_numeric(g.get("time_to_fouling_onset_min", np.nan), errors="coerce").astype(float).to_numpy()
        y_ttf_clog = pd.to_numeric(g.get("time_to_clog_onset_min", np.nan), errors="coerce").astype(float).to_numpy()
        y_ttu = pd.to_numeric(g.get("ttm_to_unplanned_event_min", np.nan), errors="coerce").astype(float).to_numpy()

        y_ttf_foul = np.where(np.isfinite(y_ttf_foul), np.clip(y_ttf_foul, 0.0, float(cfg.tte_fouling_cap_min)), float(cfg.tte_fouling_cap_min)).astype(np.float32)
        y_ttf_clog = np.where(np.isfinite(y_ttf_clog), np.clip(y_ttf_clog, 0.0, float(cfg.tte_clog_cap_min)), float(cfg.tte_clog_cap_min)).astype(np.float32)
        y_ttu = np.where(np.isfinite(y_ttu), np.clip(y_ttu, 0.0, float(cfg.ttu_cap_min)), float(cfg.ttu_cap_min)).astype(np.float32)

        phase_array = g["phase"].astype(str).to_numpy()
        foul_weight = np.ones(len(g), dtype=np.float32)
        near_foul = (y_ttf_foul <= cfg.fouling_horizon_min) & (phase_array == "production")
        foul_weight[near_foul] = 1.0 + 1.5 * (1.0 - y_ttf_foul[near_foul] / max(cfg.fouling_horizon_min, 1))
        foul_weight[y_foul_h == 1.0] = np.maximum(foul_weight[y_foul_h == 1.0], 3.0)

        actionable_foul_weight = np.ones(len(g), dtype=np.float32)
        t_unplanned_foul = pd.to_numeric(g.get("time_to_unplanned_fouling_min", np.nan), errors="coerce").astype(float).to_numpy()
        t_unplanned_foul = np.where(np.isfinite(t_unplanned_foul), np.clip(t_unplanned_foul, 0.0, float(cfg.unplanned_fouling_horizon_min)), float(cfg.unplanned_fouling_horizon_min))
        near_actionable = (t_unplanned_foul <= cfg.unplanned_fouling_horizon_min) & (phase_array == "production")
        actionable_foul_weight[near_actionable] = 1.0 + 1.5 * (1.0 - t_unplanned_foul[near_actionable] / max(cfg.unplanned_fouling_horizon_min, 1))
        actionable_foul_weight[y_actionable_foul_h == 1.0] = np.maximum(actionable_foul_weight[y_actionable_foul_h == 1.0], 3.0)

        clog_weight = np.ones(len(g), dtype=np.float32)
        near_clog = (y_ttf_clog <= cfg.clog_horizon_min) & (phase_array == "production")
        clog_weight[near_clog] = 1.0 + 1.5 * (1.0 - y_ttf_clog[near_clog] / max(cfg.clog_horizon_min, 1))
        clog_weight[y_clog_h == 1.0] = np.maximum(clog_weight[y_clog_h == 1.0], 3.0)

        x = g[list(feature_cols)].to_numpy(dtype=np.float32)
        meta_cols = [
            "timestamp",
            "asset_id",
            "sequence_id",
            "cycle_id",
            "phase",
            "maintenance_active",
            "maintenance_type",
            "fouling_stage_physical",
            cfg.severity_col,
            "clog_event",
            "flow_kg_s",
            "dP_kPa",
            "vibration_mm_s",
            "thermal_eff_proxy",
            "heat_proxy",
            "resid_flow_kg_s",
            "resid_dP_kPa",
            "resid_vibration_mm_s",
            "resid_thermal_eff_proxy",
            "dP_kPa_slope15",
            "vibration_mm_s_slope15",
            "milk_type",
            "asset_family",
            "fault_type",
            "time_to_fouling_onset_min",
            "time_to_unplanned_fouling_min",
            "time_to_clog_onset_min",
            "ttm_to_unplanned_event_min",
            "ttm_to_planned_cip_min",
        ]
        for col in meta_cols:
            if col not in g.columns:
                g[col] = np.nan
        meta = g[meta_cols].copy()

        sequences[sequence_id] = AssetSequence(
            sequence_id=sequence_id,
            asset_id=asset_id,
            timestamps=g["timestamp"].to_numpy(),
            x=x,
            y_severity=y_severity,
            y_stage=y_stage,
            y_foul_h=y_foul_h,
            y_actionable_foul_h=y_actionable_foul_h,
            y_clog_h=y_clog_h,
            y_ttf_foul=y_ttf_foul,
            y_ttf_clog=y_ttf_clog,
            y_ttu=y_ttu,
            foul_weight=foul_weight,
            actionable_foul_weight=actionable_foul_weight,
            clog_weight=clog_weight,
            meta=meta,
        )
    return sequences


class WindowDataset(Dataset):
    """Indexes valid (sequence_id, end_idx) window endpoints for the given asset_ids."""

    def __init__(
        self,
        sequences: Dict[str, AssetSequence],
        asset_ids: Sequence[str],
        feature_indices: Sequence[int],
        cfg: TrainConfig,
        stride: int | None = None,
    ) -> None:
        self.sequences = sequences
        self.asset_ids = list(asset_ids)
        self.feature_indices = np.asarray(list(feature_indices), dtype=np.int64)
        self.cfg = cfg
        self.sequence_ids = [sid for sid, seq in sequences.items() if seq.asset_id in self.asset_ids]
        self.index: List[Tuple[str, int]] = []
        step = int(stride) if stride is not None else cfg.stride
        for sequence_id in self.sequence_ids:
            seq = sequences[sequence_id]
            meta = seq.meta
            valid_end = (meta["phase"] == "production") & (meta["maintenance_active"].fillna(0).astype(int) == 0)
            for end_idx in range(cfg.seq_len - 1, len(seq.timestamps), step):
                if not bool(valid_end.iloc[end_idx]):
                    continue
                self.index.append((sequence_id, end_idx))

    def __len__(self) -> int:
        """Number of valid windows."""
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return the window tensor plus its per-step targets/weights and identifiers."""
        sequence_id, end_idx = self.index[idx]
        seq = self.sequences[sequence_id]
        start_idx = end_idx - self.cfg.seq_len + 1
        x = seq.x[start_idx:end_idx + 1][:, self.feature_indices]
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y_severity_scaled": torch.tensor(seq.y_severity[end_idx] * self.cfg.severity_scale(), dtype=torch.float32),
            "y_stage": torch.tensor(seq.y_stage[end_idx], dtype=torch.long),
            "y_foul_h": torch.tensor(seq.y_foul_h[end_idx], dtype=torch.float32),
            "y_actionable_foul_h": torch.tensor(seq.y_actionable_foul_h[end_idx], dtype=torch.float32),
            "y_clog_h": torch.tensor(seq.y_clog_h[end_idx], dtype=torch.float32),
            "y_ttf_foul_log": torch.tensor(np.log1p(seq.y_ttf_foul[end_idx]), dtype=torch.float32),
            "y_ttf_clog_log": torch.tensor(np.log1p(seq.y_ttf_clog[end_idx]), dtype=torch.float32),
            "y_ttu_log": torch.tensor(np.log1p(seq.y_ttu[end_idx]), dtype=torch.float32),
            "foul_weight": torch.tensor(seq.foul_weight[end_idx], dtype=torch.float32),
            "actionable_foul_weight": torch.tensor(seq.actionable_foul_weight[end_idx], dtype=torch.float32),
            "clog_weight": torch.tensor(seq.clog_weight[end_idx], dtype=torch.float32),
            "asset_id": seq.asset_id,
            "sequence_id": sequence_id,
            "end_idx": int(end_idx),
        }


def collate_window_batch(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Stack a list of WindowDataset items into batched tensors."""
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "y_severity_scaled": torch.stack([b["y_severity_scaled"] for b in batch], dim=0),
        "y_stage": torch.stack([b["y_stage"] for b in batch], dim=0),
        "y_foul_h": torch.stack([b["y_foul_h"] for b in batch], dim=0),
        "y_actionable_foul_h": torch.stack([b["y_actionable_foul_h"] for b in batch], dim=0),
        "y_clog_h": torch.stack([b["y_clog_h"] for b in batch], dim=0),
        "y_ttf_foul_log": torch.stack([b["y_ttf_foul_log"] for b in batch], dim=0),
        "y_ttf_clog_log": torch.stack([b["y_ttf_clog_log"] for b in batch], dim=0),
        "y_ttu_log": torch.stack([b["y_ttu_log"] for b in batch], dim=0),
        "foul_weight": torch.stack([b["foul_weight"] for b in batch], dim=0),
        "actionable_foul_weight": torch.stack([b["actionable_foul_weight"] for b in batch], dim=0),
        "clog_weight": torch.stack([b["clog_weight"] for b in batch], dim=0),
        "asset_id": [str(b["asset_id"]) for b in batch],
        "sequence_id": [str(b["sequence_id"]) for b in batch],
        "end_idx": [int(b["end_idx"]) for b in batch],
    }
