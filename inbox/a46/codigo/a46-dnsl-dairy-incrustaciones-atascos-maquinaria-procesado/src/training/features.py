
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils.common import safe_float, sequence_group_columns

from .common import (
    BASELINE_COLS,
    DEFAULT_NUMERIC_COLS,
    LAST_MAINT_TO_ID,
    MILK_TO_ID,
    FAMILY_TO_ID,
    PHASE_TO_ID,
    CLOCK_SOURCE_COLS,
    LATENT_OR_TARGET_COLS,
    FeatureArtifacts,
    TrainConfig,
    robust_quantiles,
)

def engineer_row_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    global_amb_t = float(pd.to_numeric(out.get("ambient_T_C", pd.Series([25.0])), errors="coerce").median()) if "ambient_T_C" in out.columns else 25.0
    global_amb_rh = float(pd.to_numeric(out.get("ambient_RH_pct", pd.Series([50.0])), errors="coerce").median()) if "ambient_RH_pct" in out.columns else 50.0

    for c in DEFAULT_NUMERIC_COLS:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    groups: List[pd.DataFrame] = []
    group_cols = sequence_group_columns(out)
    look15 = 15
    look60 = 60
    for _, g in out.groupby(group_cols, sort=False):
        g = g.sort_values("timestamp").copy()
        g["flow_kg_s"] = g["flow_kg_s"].fillna(0.0)
        g["dP_kPa"] = g["dP_kPa"].fillna(0.0)
        g["vibration_mm_s"] = g["vibration_mm_s"].fillna(0.0)
        g["ambient_T_C"] = g["ambient_T_C"].fillna(global_amb_t)
        g["ambient_RH_pct"] = g["ambient_RH_pct"].fillna(global_amb_rh)
        g["flow_sp_kg_s"] = g["flow_sp_kg_s"].replace(0.0, np.nan)
        g["flow_sp_kg_s"] = g["flow_sp_kg_s"].fillna(g["flow_kg_s"].rolling(30, min_periods=1).median())
        g["Th_sp_C"] = g["Th_sp_C"].fillna(pd.to_numeric(g["Th_in_C"], errors="coerce").rolling(30, min_periods=1).median())
        g["Tc_sp_C"] = g["Tc_sp_C"].fillna(pd.to_numeric(g["Tc_in_C"], errors="coerce").rolling(30, min_periods=1).median())
        g["batch_thermal_history_factor"] = g["batch_thermal_history_factor"].fillna(1.0)
        g["batch_elapsed_min"] = g["batch_elapsed_min"].fillna(0.0)
        g["time_since_last_cip_min"] = pd.to_numeric(g["time_since_last_cip_min"], errors="coerce")
        g["time_since_last_maintenance_min"] = pd.to_numeric(g["time_since_last_maintenance_min"], errors="coerce")

        g["flow_error"] = g["flow_kg_s"] - g["flow_sp_kg_s"]
        g["flow_ratio"] = g["flow_kg_s"] / np.maximum(g["flow_sp_kg_s"], 0.25)
        g["hot_drop"] = pd.to_numeric(g["Th_in_C"], errors="coerce").fillna(0.0) - pd.to_numeric(g["Th_out_C"], errors="coerce").fillna(0.0)
        g["cold_lift"] = pd.to_numeric(g["Tc_out_C"], errors="coerce").fillna(0.0) - pd.to_numeric(g["Tc_in_C"], errors="coerce").fillna(0.0)
        g["temp_gap"] = pd.to_numeric(g["Th_in_C"], errors="coerce").fillna(0.0) - pd.to_numeric(g["Tc_in_C"], errors="coerce").fillna(0.0)
        g["thermal_eff_proxy"] = g["hot_drop"] / np.maximum(g["temp_gap"], 1.0)
        g["approach_temp"] = pd.to_numeric(g["Th_out_C"], errors="coerce").fillna(0.0) - pd.to_numeric(g["Tc_in_C"], errors="coerce").fillna(0.0)
        g["dP_per_flow"] = g["dP_kPa"] / np.maximum(g["flow_kg_s"], 0.25)
        g["vib_x_dp"] = g["vibration_mm_s"] * g["dP_kPa"]
        g["heat_proxy"] = g["flow_kg_s"] * g["hot_drop"]
        g["solids_x_thermal"] = pd.to_numeric(g["solids_g_L_nominal"], errors="coerce").fillna(0.0) * g["batch_thermal_history_factor"]
        g["cip_clock_log"] = np.log1p(np.clip(g["time_since_last_cip_min"], 0.0, 1e6))
        g["maint_clock_log"] = np.log1p(np.clip(g["time_since_last_maintenance_min"], 0.0, 1e6))
        g["batch_clock_log"] = np.log1p(np.clip(g["batch_elapsed_min"], 0.0, 1e6))

        for c in ["dP_kPa", "flow_kg_s", "vibration_mm_s", "thermal_eff_proxy", "heat_proxy"]:
            g[f"{c}_mean15"] = g[c].rolling(look15, min_periods=1).mean()
            g[f"{c}_std15"] = g[c].rolling(look15, min_periods=2).std().fillna(0.0)
            g[f"{c}_slope15"] = g[c] - g[c].shift(look15).fillna(g[c].iloc[0])
        for c in ["dP_kPa", "thermal_eff_proxy", "heat_proxy"]:
            g[f"{c}_mean60"] = g[c].rolling(look60, min_periods=1).mean()
            g[f"{c}_slope60"] = g[c] - g[c].shift(look60).fillna(g[c].iloc[0])
        groups.append(g)
    out = pd.concat(groups, axis=0).sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    return out

def fit_feature_artifacts(train_df: pd.DataFrame, cfg: TrainConfig) -> FeatureArtifacts:
    stable_mask = (
        (train_df["phase"] == "production")
        & (train_df["maintenance_active"].fillna(0).astype(int) == 0)
        & (train_df["fouling_stage_physical"] == 0)
    )
    stable_df = train_df.loc[stable_mask].copy()
    if len(stable_df) == 0:
        stable_df = train_df.loc[(train_df["phase"] == "production") & (train_df["maintenance_active"].fillna(0).astype(int) == 0)].copy()

    candidate_numeric = []
    allowed_prefixes = ("flow_", "hot_", "cold_", "temp_", "thermal_", "approach_", "dP_", "vib_", "heat_", "solids_", "cip_", "maint_", "batch_")
    for c in train_df.columns:
        if c in LATENT_OR_TARGET_COLS:
            continue
        if c in DEFAULT_NUMERIC_COLS or c.startswith(allowed_prefixes):
            if pd.api.types.is_numeric_dtype(train_df[c]):
                candidate_numeric.append(c)
    numeric_names = sorted(set(candidate_numeric))

    medians: Dict[str, float] = {}
    iqrs: Dict[str, float] = {}
    for c in numeric_names:
        arr = pd.to_numeric(train_df[c], errors="coerce").astype(float).values
        q25, q75 = robust_quantiles(arr, 0.25, 0.75, (0.0, 1.0))
        med = float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0
        medians[c] = med
        iqrs[c] = max(q75 - q25, 1e-6)

    train_asset_baselines: Dict[str, Dict[str, float]] = {}
    for asset_id, g in stable_df.groupby("asset_id", sort=False):
        base: Dict[str, float] = {}
        for c in BASELINE_COLS:
            arr = pd.to_numeric(g[c], errors="coerce").astype(float).values if c in g.columns else np.array([], dtype=float)
            if arr.size:
                base[c] = float(np.nanmedian(arr))
        train_asset_baselines[asset_id] = base

    global_baseline: Dict[str, float] = {}
    for c in BASELINE_COLS:
        arr = pd.to_numeric(stable_df[c], errors="coerce").astype(float).values if c in stable_df.columns else np.array([], dtype=float)
        global_baseline[c] = float(np.nanmedian(arr)) if arr.size else 0.0

    pred_th: Dict[str, float] = {}
    if len(stable_df):
        stable_centered_dp = stable_df["dP_kPa"] - stable_df.groupby("asset_id")["dP_kPa"].transform("median")
        stable_centered_flow = stable_df["flow_kg_s"] - stable_df.groupby("asset_id")["flow_kg_s"].transform("median")
        stable_centered_vib = stable_df["vibration_mm_s"] - stable_df.groupby("asset_id")["vibration_mm_s"].transform("median")
        stable_centered_therm = stable_df["thermal_eff_proxy"] - stable_df.groupby("asset_id")["thermal_eff_proxy"].transform("median")
        pred_th["high_dp_resid"] = float(np.nanquantile(stable_centered_dp.values, 0.90))
        pred_th["low_flow_resid"] = float(np.nanquantile(stable_centered_flow.values, 0.10))
        pred_th["high_vib_resid"] = float(np.nanquantile(stable_centered_vib.values, 0.90))
        pred_th["low_therm_eff_resid"] = float(np.nanquantile(stable_centered_therm.values, 0.10))
    else:
        pred_th["high_dp_resid"] = 1.0
        pred_th["low_flow_resid"] = -1.0
        pred_th["high_vib_resid"] = 1.0
        pred_th["low_therm_eff_resid"] = -1.0
    pred_th["dp_slope15"] = float(np.nanquantile(pd.to_numeric(train_df.get("dP_kPa_slope15", 0.0), errors="coerce").astype(float).values, 0.85))
    pred_th["vib_slope15"] = float(np.nanquantile(pd.to_numeric(train_df.get("vibration_mm_s_slope15", 0.0), errors="coerce").astype(float).values, 0.85))

    prod_df = train_df.loc[(train_df["phase"] == "production") & (train_df["maintenance_active"].fillna(0).astype(int) == 0)].copy()
    counts = prod_df["fouling_stage_physical"].value_counts().reindex([0, 1, 2], fill_value=1).astype(float).values
    inv = counts.sum() / np.maximum(counts, 1.0)
    inv = inv / inv.mean()

    foul_col = f"fouling_onset_within_{cfg.fouling_horizon_min}min"
    clog_col = f"clog_onset_within_{cfg.clog_horizon_min}min"
    actionable_col = f"unplanned_fouling_within_{cfg.unplanned_fouling_horizon_min}min"
    foul_pos = float(pd.to_numeric(prod_df.get(foul_col, 0), errors="coerce").fillna(0).sum())
    foul_neg = float(len(prod_df) - foul_pos)
    clog_pos = float(pd.to_numeric(prod_df.get(clog_col, 0), errors="coerce").fillna(0).sum())
    clog_neg = float(len(prod_df) - clog_pos)
    actionable_pos = float(pd.to_numeric(prod_df.get(actionable_col, 0), errors="coerce").fillna(0).sum())
    actionable_neg = float(len(prod_df) - actionable_pos)
    foul_pos_weight = foul_neg / max(foul_pos, 1.0)
    clog_pos_weight = clog_neg / max(clog_pos, 1.0)
    actionable_foul_pos_weight = actionable_neg / max(actionable_pos, 1.0)

    return FeatureArtifacts(
        numeric_feature_names=numeric_names,
        medians=medians,
        iqrs=iqrs,
        train_asset_baselines=train_asset_baselines,
        global_baseline=global_baseline,
        predicate_thresholds=pred_th,
        stage_class_weights=inv.tolist(),
        foul_pos_weight=float(foul_pos_weight),
        clog_pos_weight=float(clog_pos_weight),
        actionable_foul_pos_weight=float(actionable_foul_pos_weight),
        full_feature_names=[],
        no_clock_feature_names=[],
    )

def estimate_prefix_baseline(asset_df: pd.DataFrame, global_baseline: Mapping[str, float], prefix_rows: int) -> Dict[str, float]:
    g = asset_df.sort_values("timestamp").iloc[: max(prefix_rows, 1)].copy()
    cand = g.loc[(g["phase"] == "production") & (g["maintenance_active"].fillna(0).astype(int) == 0)].copy()
    if len(cand) == 0:
        cand = g.copy()
    base: Dict[str, float] = {}
    for c in BASELINE_COLS:
        if c in cand.columns:
            arr = pd.to_numeric(cand[c], errors="coerce").astype(float).values
            if np.isfinite(arr).any():
                prefix_med = float(np.nanmedian(arr))
                base[c] = 0.6 * prefix_med + 0.4 * float(global_baseline.get(c, prefix_med))
                continue
        base[c] = float(global_baseline.get(c, 0.0))
    return base

def build_feature_matrix(df: pd.DataFrame, artifacts: FeatureArtifacts, train_assets: Sequence[str], cfg: TrainConfig) -> Tuple[pd.DataFrame, List[str], List[str]]:
    out = df.copy()
    full_feature_cols: List[str] = []

    for c in artifacts.numeric_feature_names:
        med = artifacts.medians[c]
        iqr = artifacts.iqrs[c]
        col = pd.to_numeric(out[c], errors="coerce").fillna(med)
        out[f"z_{c}"] = ((col - med) / iqr).astype(np.float32)
        full_feature_cols.append(f"z_{c}")

    prefix_rows = max(1, int(round(cfg.baseline_prefix_hours * 3600 / max(cfg.dt, 1))))
    resid_feature_names = []
    for c in BASELINE_COLS:
        resid_values = np.zeros(len(out), dtype=np.float32)
        for asset_id, idx in out.groupby("asset_id", sort=False).groups.items():
            g = out.loc[idx].sort_values("timestamp")
            if asset_id in artifacts.train_asset_baselines:
                base = artifacts.train_asset_baselines[asset_id]
            else:
                base = estimate_prefix_baseline(g, artifacts.global_baseline, prefix_rows)
            b = float(base.get(c, artifacts.global_baseline.get(c, 0.0)))
            vals = pd.to_numeric(g[c], errors="coerce").fillna(b).to_numpy(dtype=float)
            resid_values[g.index.to_numpy()] = ((vals - b) / artifacts.iqrs.get(c, 1.0)).astype(np.float32)
        out[f"resid_{c}"] = resid_values
        resid_feature_names.append(f"resid_{c}")
    full_feature_cols.extend(resid_feature_names)

    def add_one_hot(series: pd.Series, mapping: Mapping[str, int], prefix: str) -> None:
        nonlocal out, full_feature_cols
        arr = np.zeros((len(series), len(mapping)), dtype=np.float32)
        idxs = [mapping.get(str(v), 0) for v in series.astype(str).tolist()]
        arr[np.arange(len(series)), idxs] = 1.0
        names = [f"{prefix}_{k}" for k in mapping.keys()]
        for i, name in enumerate(names):
            out[name] = arr[:, i]
            full_feature_cols.append(name)

    add_one_hot(out["phase"].where(out["phase"].isin(PHASE_TO_ID), "production"), PHASE_TO_ID, "phase")
    add_one_hot(out["milk_type"].where(out["milk_type"].isin(MILK_TO_ID), "none"), MILK_TO_ID, "milk")
    add_one_hot(out["asset_family"].where(out["asset_family"].isin(FAMILY_TO_ID), "unknown"), FAMILY_TO_ID, "family")
    add_one_hot(out["last_maintenance_type"].where(out["last_maintenance_type"].isin(LAST_MAINT_TO_ID), "other"), LAST_MAINT_TO_ID, "lastmaint")

    def is_clock_feature(feature_name: str) -> bool:
        raw = feature_name[2:] if feature_name.startswith("z_") else feature_name
        return raw in cfg.clock_feature_names()

    no_clock_feature_cols = [f for f in full_feature_cols if not is_clock_feature(f)]

    artifacts.full_feature_names = list(full_feature_cols)
    artifacts.no_clock_feature_names = list(no_clock_feature_cols)
    return out, full_feature_cols, no_clock_feature_cols
