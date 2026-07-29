"""Vendored from inbox/a46/codigo/.../src/training/common.py (verbatim algorithmic core)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PHASES: Tuple[str, ...] = ("production", "CIP_alkaline", "CIP_acid", "idle", "maintenance")
MILK_TYPES: Tuple[str, ...] = ("whole", "semi_skim", "skim", "high_solids", "none")
ASSET_FAMILIES: Tuple[str, ...] = ("standard_phe", "robust_phe", "compact_phe", "unknown")
LAST_MAINT_TYPES: Tuple[str, ...] = ("none", "scheduled_CIP", "CIP_extra", "unclog", "mechanical_clean", "other")
STAGE_NAMES: Tuple[str, ...] = ("stable", "incipient", "advanced")

PHASE_TO_ID = {v: i for i, v in enumerate(PHASES)}
MILK_TO_ID = {v: i for i, v in enumerate(MILK_TYPES)}
FAMILY_TO_ID = {v: i for i, v in enumerate(ASSET_FAMILIES)}
LAST_MAINT_TO_ID = {v: i for i, v in enumerate(LAST_MAINT_TYPES)}

CLOCK_SOURCE_COLS = {
    "batch_elapsed_min",
    "time_since_last_cip_min",
    "time_since_last_maintenance_min",
    "batch_clock_log",
    "cip_clock_log",
    "maint_clock_log",
}
BASELINE_COLS: Tuple[str, ...] = ("flow_kg_s", "dP_kPa", "vibration_mm_s", "thermal_eff_proxy", "heat_proxy")
CIP_EVENT_TYPES = {"scheduled_CIP", "CIP_extra"}

LATENT_OR_TARGET_COLS = {
    "Rf_m2K_W",
    "m_total_kg_m2",
    "m_org_kg_m2",
    "m_min_kg_m2",
    "UA_W_K",
    "Q_W",
    "fouling_stage",
    "fouling_stage_physical",
    "fouling_stage_name",
    "fouling_onset_event",
    "clog_event",
    "clog_onset_event",
    "foul_score",
    "ttm_to_planned_cip_min",
    "ttm_to_unplanned_event_min",
    "time_to_fouling_onset_min",
    "time_to_clog_onset_min",
    "fouling_onset_within_30min",
    "clog_onset_within_15min",
    "unplanned_event_within_60min",
    "rf_stage_thr_incipient",
    "rf_stage_thr_advanced",
}
DEFAULT_NUMERIC_COLS: Tuple[str, ...] = (
    "flow_kg_s",
    "pressure_in_kPa",
    "pressure_out_kPa",
    "dP_kPa",
    "Th_in_C",
    "Tc_in_C",
    "Th_out_C",
    "Tc_out_C",
    "Twall_C",
    "vibration_mm_s",
    "ambient_T_C",
    "ambient_RH_pct",
    "flow_sp_kg_s",
    "Th_sp_C",
    "Tc_sp_C",
    "protein_g_L_nominal",
    "fat_g_L_nominal",
    "solids_g_L_nominal",
    "Ca_mM_nominal",
    "PO4_mM_nominal",
    "pH_nominal",
    "batch_thermal_history_factor",
    "batch_elapsed_min",
    "time_since_last_cip_min",
    "time_since_last_maintenance_min",
)


@dataclass
class TrainConfig:
    """Runtime + training configuration, mirrors models/artifacts/training_config.json."""

    telemetry: str = ""
    maintenance: str = ""
    artifacts_dir: str = ""
    metrics_dir: str = ""
    predictions_dir: str = ""
    splits_dir: str = ""
    severity_col: str = "Rf_m2K_W"
    stage_thr_incipient: Optional[float] = None
    stage_thr_advanced: Optional[float] = None
    fouling_horizon_min: int = 30
    clog_horizon_min: int = 15
    unplanned_fouling_horizon_min: int = 120
    tte_fouling_cap_min: int = 240
    tte_clog_cap_min: int = 120
    ttu_cap_min: int = 360
    dt: int = 60
    seq_len: int = 120
    stride: int = 5
    batch_size: int = 64
    epochs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 7
    device: str = "cpu"
    channels: int = 64
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16)
    dropout: float = 0.15
    baseline_prefix_hours: float = 8.0
    match_window_min: int = 240
    split_trials: int = 3000
    policy_max_candidates: int = 400
    ablate_clocks: bool = True
    cooldown_min_default: int = 60

    def resolved_stage_thresholds(self) -> Tuple[float, float]:
        """Return (incipient, advanced) severity thresholds for the 3-class stage."""
        if self.stage_thr_incipient is not None and self.stage_thr_advanced is not None:
            if self.stage_thr_advanced <= self.stage_thr_incipient:
                raise ValueError("stage_thr_advanced must be greater than stage_thr_incipient.")
            return float(self.stage_thr_incipient), float(self.stage_thr_advanced)
        if self.severity_col == "Rf_m2K_W":
            incipient = 5.5e-4 if self.stage_thr_incipient is None else float(self.stage_thr_incipient)
            advanced = 1.10e-3 if self.stage_thr_advanced is None else float(self.stage_thr_advanced)
            if advanced <= incipient:
                raise ValueError("stage_thr_advanced must be greater than stage_thr_incipient.")
            return incipient, advanced
        raise ValueError(
            "For severity columns other than Rf_m2K_W, pass explicit stage_thr_incipient and stage_thr_advanced."
        )

    def severity_scale(self) -> float:
        """Multiplier applied to severity before feeding it to the softplus regression head."""
        return 1000.0 if self.severity_col == "Rf_m2K_W" else 1.0

    def clock_feature_names(self) -> set[str]:
        """Return the raw column names ablated in the no_clock scenario."""
        return set(CLOCK_SOURCE_COLS)


@dataclass
class FeatureArtifacts:
    """Fitted scaling/baseline/weighting artifacts, mirrors models/artifacts/feature_artifacts.json."""

    numeric_feature_names: List[str]
    medians: Dict[str, float]
    iqrs: Dict[str, float]
    train_asset_baselines: Dict[str, Dict[str, float]]
    global_baseline: Dict[str, float]
    predicate_thresholds: Dict[str, float]
    stage_class_weights: List[float]
    foul_pos_weight: float
    clog_pos_weight: float
    actionable_foul_pos_weight: float
    full_feature_names: List[str]
    no_clock_feature_names: List[str]


def robust_quantiles(values: np.ndarray, q_lo: float, q_hi: float, default: Tuple[float, float]) -> Tuple[float, float]:
    """Quantiles of the finite values in *values*, or *default* if none are finite."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.quantile(arr, q_lo)), float(np.quantile(arr, q_hi))


def stage_to_label(stage: int) -> str:
    """Map a 0/1/2 stage code to its name."""
    return STAGE_NAMES[int(max(0, min(2, int(stage))))]


def derive_stage_from_severity(values: pd.Series | np.ndarray, thr_incipient: float, thr_advanced: float) -> np.ndarray:
    """Bucket a severity series into 0 (stable) / 1 (incipient) / 2 (advanced)."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    stage = np.where(arr >= thr_advanced, 2, np.where(arr >= thr_incipient, 1, 0))
    return stage.astype(np.int64)


def total_asset_days(telemetry_df: pd.DataFrame, asset_ids: Sequence[str]) -> float:
    """Total observed operating days across *asset_ids* (used to normalize false-alarm rate)."""
    sub = telemetry_df.loc[telemetry_df["asset_id"].isin(asset_ids)].copy()
    if len(sub) == 0:
        return 1e-6
    total_days = 0.0
    for _, g in sub.groupby("asset_id", sort=False):
        g = g.sort_values("timestamp")
        diffs = g["timestamp"].diff().dt.total_seconds().dropna()
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            total_days += 1e-6
            continue
        nominal = float(np.nanmedian(diffs))
        if not math.isfinite(nominal) or nominal <= 0:
            nominal = 60.0
        observed = float(diffs[diffs <= max(nominal * 1.5, nominal + 1e-9)].sum() + nominal)
        total_days += max(observed / 86400.0, 1e-6)
    return max(total_days, 1e-6)


def safe_binary_auc_ap(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """(AUC, AP) for a binary target, or (nan, nan) if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))
