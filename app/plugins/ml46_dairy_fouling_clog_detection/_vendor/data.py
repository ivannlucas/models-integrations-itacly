"""Vendored from inbox/a46/codigo/.../src/training/data.py (verbatim algorithmic core).

split_assets()/_score_asset_split() from the original file are intentionally NOT vendored:
the plugin's train() fine-tunes the already-served no_clock checkpoint on the caller's own
labeled CSV (see plugin.py), it does not re-run the original from-scratch asset-split search.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from app.plugins.ml46_dairy_fouling_clog_detection._vendor.common import (
    CIP_EVENT_TYPES,
    DEFAULT_NUMERIC_COLS,
    LAST_MAINT_TO_ID,
    TrainConfig,
    derive_stage_from_severity,
)
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.utils_common import (
    derive_cycle_columns,
    ensure_columns,
    minutes_since_last_event,
    minutes_to_next_event,
    normalize_fault_type,
    previous_event_value,
    sequence_group_columns,
)


def load_telemetry(df: pd.DataFrame, cfg: TrainConfig, require_targets: bool = True) -> pd.DataFrame:
    """Validate + normalize a raw telemetry dataframe (mirrors the original CSV loader).

    Unlike the original (which reads a CSV path), this takes an already-parsed DataFrame —
    the plugin's preprocessing layer is responsible for pd.read_csv / building it from rows.
    """
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("Telemetry CSV must contain a timestamp column.")
    if "asset_id" not in df.columns:
        raise ValueError("Telemetry CSV must contain an asset_id column.")
    if require_targets and cfg.severity_col not in df.columns:
        raise ValueError(f"Telemetry CSV must contain severity column '{cfg.severity_col}'.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Some telemetry timestamps could not be parsed.")
    df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    defaults = {
        "asset_family": "unknown",
        "milk_type": "none",
        "phase": "production",
        "maintenance_active": 0,
        "maintenance_type": "none",
        "fault_type": "none",
        "planned_event_type": "none",
        "maintenance_planned": 0,
        "batch_id": "none",
        "shift_id": 1,
        "batch_elapsed_min": 0.0,
        "time_since_last_cip_min": np.nan,
        "time_since_last_maintenance_min": np.nan,
        "ttm_to_planned_cip_min": np.nan,
        "ttm_to_unplanned_event_min": np.nan,
        "time_to_fouling_onset_min": np.nan,
        "time_to_clog_onset_min": np.nan,
        "time_to_unplanned_fouling_min": np.nan,
        f"fouling_onset_within_{cfg.fouling_horizon_min}min": np.nan,
        f"clog_onset_within_{cfg.clog_horizon_min}min": np.nan,
        f"unplanned_fouling_within_{cfg.unplanned_fouling_horizon_min}min": np.nan,
        "fouling_onset_event": 0,
        "clog_event": 0,
        "clog_onset_event": 0,
        "last_maintenance_type": "none",
        cfg.severity_col: np.nan,
    }
    df = ensure_columns(df, defaults)

    for c in DEFAULT_NUMERIC_COLS + (cfg.severity_col,):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    thr_incipient, thr_advanced = cfg.resolved_stage_thresholds()
    if require_targets:
        severity_for_stage = df[cfg.severity_col]
    else:
        severity_for_stage = df[cfg.severity_col].fillna(0.0)
    df["fouling_stage_physical"] = derive_stage_from_severity(severity_for_stage, thr_incipient, thr_advanced)
    if "fouling_stage" not in df.columns or df["fouling_stage"].isna().all():
        df["fouling_stage"] = df["fouling_stage_physical"]
    df["fouling_stage"] = pd.to_numeric(df["fouling_stage"], errors="coerce").fillna(df["fouling_stage_physical"]).astype(int)
    df.loc[df["phase"] != "production", "fouling_stage"] = -1

    if "fouling_stage_name" not in df.columns:
        df["fouling_stage_name"] = np.where(
            df["phase"] == "production",
            pd.Series(df["fouling_stage_physical"]).map({0: "stable", 1: "incipient", 2: "advanced"}),
            "not_production",
        )

    df = derive_cycle_columns(df)
    return df


def derive_maintenance_from_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct a maintenance-events table from maintenance_active runs when none is supplied."""
    rows: List[Dict[str, Any]] = []
    counter = 0
    if "maintenance_active" not in df.columns:
        return pd.DataFrame(rows)
    for asset_id, g in df.groupby("asset_id", sort=False):
        g = g.sort_values("timestamp")
        active = g["maintenance_active"].fillna(0).astype(int).to_numpy()
        starts = np.where((active == 1) & (np.r_[0, active[:-1]] == 0))[0]
        ends = np.where((active == 0) & (np.r_[0, active[:-1]] == 1))[0] - 1
        if len(active) > 0 and active[-1] == 1:
            ends = np.r_[ends, len(g) - 1]
        for s, e in zip(starts, ends):
            counter += 1
            sl = g.iloc[s:e + 1]
            mtype = str(sl["maintenance_type"].mode().iloc[0]) if "maintenance_type" in sl else "other"
            ftype = normalize_fault_type(str(sl["fault_type"].mode().iloc[0]) if "fault_type" in sl else "other")
            planned = int(sl.get("maintenance_planned", pd.Series([0])).mode().iloc[0]) if "maintenance_planned" in sl else 0
            rows.append(
                {
                    "maintenance_id": f"D{counter:07d}",
                    "asset_id": asset_id,
                    "start_time": sl["timestamp"].iloc[0],
                    "end_time": sl["timestamp"].iloc[-1],
                    "duration_min": float((sl["timestamp"].iloc[-1] - sl["timestamp"].iloc[0]).total_seconds() / 60.0),
                    "planned": planned,
                    "fault_type": ftype,
                    "maintenance_type": mtype,
                    "corrective_action": "derived_from_telemetry",
                    "severity_rf_at_start": float(pd.to_numeric(sl.get("Rf_m2K_W", pd.Series([0.0])), errors="coerce").iloc[0]) if "Rf_m2K_W" in sl else 0.0,
                    "notes": "derived from telemetry",
                }
            )
    return pd.DataFrame(rows)


def load_maintenance(maintenance_df: pd.DataFrame | None, telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a maintenance-events dataframe, or derive one from telemetry if absent."""
    if maintenance_df is not None and len(maintenance_df) > 0:
        df = maintenance_df.copy()
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
        df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce")
        if "planned" not in df.columns:
            df["planned"] = 0
        if "fault_type" not in df.columns:
            df["fault_type"] = "other"
        if "maintenance_type" not in df.columns:
            df["maintenance_type"] = "other"
        if "duration_min" not in df.columns:
            df["duration_min"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60.0
        if "severity_rf_at_start" not in df.columns:
            df["severity_rf_at_start"] = 0.0
        return df.sort_values(["asset_id", "start_time"]).reset_index(drop=True)
    return derive_maintenance_from_telemetry(telemetry_df)


def align_future_labels(telemetry_df: pd.DataFrame, maintenance_df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    """Compute future-looking targets (onset flags, time-to-event) per sequence, train-only."""
    out = telemetry_df.copy()
    out = derive_cycle_columns(out)
    out = ensure_columns(
        out,
        {
            "fouling_onset_event": 0,
            "clog_onset_event": 0,
            "clog_event": 0,
            "ttm_to_planned_cip_min": np.nan,
            "ttm_to_unplanned_event_min": np.nan,
            "time_to_fouling_onset_min": np.nan,
            "time_to_clog_onset_min": np.nan,
            "time_to_unplanned_fouling_min": np.nan,
            f"fouling_onset_within_{cfg.fouling_horizon_min}min": np.nan,
            f"clog_onset_within_{cfg.clog_horizon_min}min": np.nan,
            f"unplanned_fouling_within_{cfg.unplanned_fouling_horizon_min}min": np.nan,
            "last_maintenance_type": "none",
        },
    )
    out["fouling_onset_event"] = pd.to_numeric(out["fouling_onset_event"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    out["clog_onset_event"] = pd.to_numeric(out["clog_onset_event"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    out["clog_event"] = pd.to_numeric(out["clog_event"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    group_cols = sequence_group_columns(out)

    if out["fouling_onset_event"].sum() == 0:
        for _, g in out.groupby(group_cols, sort=False):
            g = g.sort_values("timestamp")
            prev = pd.Series(g["fouling_stage_physical"]).shift(1).fillna(0).astype(int)
            onset = ((prev == 0) & (g["fouling_stage_physical"] >= 1) & (g["phase"] == "production")).astype(int)
            out.loc[g.index, "fouling_onset_event"] = onset.values

    if out["clog_onset_event"].sum() == 0 and "clog_event" in out.columns:
        for _, g in out.groupby(group_cols, sort=False):
            g = g.sort_values("timestamp")
            prev = pd.Series(g["clog_event"]).shift(1).fillna(0).astype(int)
            onset = ((prev == 0) & (g["clog_event"] >= 1) & (g["phase"] == "production")).astype(int)
            out.loc[g.index, "clog_onset_event"] = onset.values

    maint = maintenance_df.copy()
    if len(maint) > 0:
        maint["start_time"] = pd.to_datetime(maint["start_time"], utc=True)
        maint["end_time"] = pd.to_datetime(maint["end_time"], utc=True)
        maint["fault_type_norm"] = maint["fault_type"].map(normalize_fault_type)
    else:
        maint = pd.DataFrame(columns=["asset_id", "start_time", "end_time", "planned", "fault_type", "maintenance_type", "fault_type_norm"])

    foul_col = f"fouling_onset_within_{cfg.fouling_horizon_min}min"
    clog_col = f"clog_onset_within_{cfg.clog_horizon_min}min"
    actionable_col = f"unplanned_fouling_within_{cfg.unplanned_fouling_horizon_min}min"

    for _, g in out.groupby(group_cols, sort=False):
        g = g.sort_values("timestamp").copy()
        ts_ns = g["timestamp"].astype("int64").to_numpy()
        asset_id = str(g["asset_id"].iloc[0])
        seq_start = pd.Timestamp(g["timestamp"].min())
        seq_end = pd.Timestamp(g["timestamp"].max()) + pd.Timedelta(seconds=max(int(cfg.dt), 1))

        onset_times = g.loc[g["fouling_onset_event"] == 1, "timestamp"].astype("int64").to_numpy()
        out.loc[g.index, "time_to_fouling_onset_min"] = minutes_to_next_event(ts_ns, onset_times)

        clog_times = g.loc[g["clog_onset_event"] == 1, "timestamp"].astype("int64").to_numpy()
        out.loc[g.index, "time_to_clog_onset_min"] = minutes_to_next_event(ts_ns, clog_times)

        asset_maint = maint.loc[maint["asset_id"] == asset_id].sort_values("start_time").copy()
        if len(asset_maint) > 0:
            asset_maint = asset_maint.loc[
                (asset_maint["start_time"] >= seq_start - pd.Timedelta(seconds=max(int(cfg.dt), 1)))
                & (asset_maint["start_time"] <= seq_end)
            ].copy()
        planned_starts = asset_maint.loc[asset_maint["planned"].fillna(0).astype(int) == 1, "start_time"].astype("int64").to_numpy() if len(asset_maint) else np.array([], dtype=np.int64)
        unplanned_df = asset_maint.loc[
            (asset_maint["planned"].fillna(0).astype(int) == 0)
            & (asset_maint["fault_type_norm"].isin(["fouling", "clogging"]))
        ].copy() if len(asset_maint) else asset_maint.copy()
        unplanned_starts = unplanned_df["start_time"].astype("int64").to_numpy() if len(unplanned_df) else np.array([], dtype=np.int64)
        unplanned_foul_starts = unplanned_df.loc[unplanned_df["fault_type_norm"] == "fouling", "start_time"].astype("int64").to_numpy() if len(unplanned_df) else np.array([], dtype=np.int64)

        out.loc[g.index, "ttm_to_planned_cip_min"] = minutes_to_next_event(ts_ns, planned_starts)
        out.loc[g.index, "ttm_to_unplanned_event_min"] = minutes_to_next_event(ts_ns, unplanned_starts)
        out.loc[g.index, "time_to_unplanned_fouling_min"] = minutes_to_next_event(ts_ns, unplanned_foul_starts)

        ttf = pd.to_numeric(out.loc[g.index, "time_to_fouling_onset_min"], errors="coerce").to_numpy(dtype=float)
        out.loc[g.index, foul_col] = ((ttf <= cfg.fouling_horizon_min) & np.isfinite(ttf)).astype(int)

        ttc = pd.to_numeric(out.loc[g.index, "time_to_clog_onset_min"], errors="coerce").to_numpy(dtype=float)
        out.loc[g.index, clog_col] = ((ttc <= cfg.clog_horizon_min) & np.isfinite(ttc)).astype(int)

        tuf = pd.to_numeric(out.loc[g.index, "time_to_unplanned_fouling_min"], errors="coerce").to_numpy(dtype=float)
        out.loc[g.index, actionable_col] = ((tuf <= cfg.unplanned_fouling_horizon_min) & np.isfinite(tuf)).astype(int)

    for asset_id, idx in out.groupby("asset_id", sort=False).groups.items():
        g = out.loc[idx].sort_values("timestamp")
        ts_ns = g["timestamp"].astype("int64").to_numpy()
        asset_maint = maint.loc[maint["asset_id"] == asset_id].sort_values("start_time").copy()

        maint_end_times = asset_maint["end_time"].astype("int64").to_numpy() if len(asset_maint) else np.array([], dtype=np.int64)
        last_type = previous_event_value(ts_ns, maint_end_times, asset_maint["maintenance_type"].astype(str).tolist(), default="none") if len(asset_maint) else np.array(["none"] * len(g), dtype=object)
        out.loc[g.index, "last_maintenance_type"] = last_type

        cip_end = asset_maint.loc[asset_maint["maintenance_type"].astype(str).isin(CIP_EVENT_TYPES), "end_time"].astype("int64").to_numpy() if len(asset_maint) else np.array([], dtype=np.int64)
        out.loc[g.index, "time_since_last_cip_min"] = minutes_since_last_event(ts_ns, cip_end)
        out.loc[g.index, "time_since_last_maintenance_min"] = minutes_since_last_event(ts_ns, maint_end_times)

    out[foul_col] = pd.to_numeric(out[foul_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    out[clog_col] = pd.to_numeric(out[clog_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    out[actionable_col] = pd.to_numeric(out[actionable_col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    if "last_maintenance_type" not in out.columns:
        out["last_maintenance_type"] = "none"
    out["last_maintenance_type"] = out["last_maintenance_type"].astype(str).where(out["last_maintenance_type"].isin(LAST_MAINT_TO_ID), "other")
    return out.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
