
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.common import (
    derive_cycle_columns,
    minutes_since_last_event,
    minutes_to_next_event,
    next_event_value,
    previous_event_value,
)

from .synthetic_generator import CIP_EVENT_TYPES, GeneratorConfig


def annotate_future_targets(telemetry_df: pd.DataFrame, maintenance_df: pd.DataFrame, cfg: GeneratorConfig) -> pd.DataFrame:
    out = telemetry_df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    if "cycle_id" not in out.columns:
        out["cycle_id"] = out.get("batch_id", out.get("episode_id", pd.Series(["cycle_00000"] * len(out)))).astype(str)
    if "sequence_id" not in out.columns:
        out["sequence_id"] = out["cycle_id"].astype(str)

    if len(maintenance_df) > 0:
        maint = maintenance_df.copy()
        maint["start_time"] = pd.to_datetime(maint["start_time"], utc=True)
        maint["end_time"] = pd.to_datetime(maint["end_time"], utc=True)
        if "cycle_id" not in maint.columns:
            maint["cycle_id"] = np.nan
    else:
        maint = pd.DataFrame(
            columns=["asset_id", "cycle_id", "start_time", "end_time", "planned", "fault_type", "maintenance_type"]
        )

    out["ttm_to_planned_cip_min"] = np.nan
    out["ttm_to_unplanned_event_min"] = np.nan
    out["time_to_fouling_onset_min"] = np.nan
    out["time_to_clog_onset_min"] = np.nan
    out[f"fouling_onset_within_{cfg.fouling_horizon_min}min"] = 0
    out[f"clog_onset_within_{cfg.clog_horizon_min}min"] = 0
    out[f"unplanned_event_within_{cfg.unplanned_horizon_min}min"] = 0
    out["time_since_last_cip_min"] = np.nan
    out["time_since_last_maintenance_min"] = np.nan
    out["last_maintenance_type"] = "none"
    out["next_unplanned_fault_type"] = "none"

    for asset_id, asset_idx in out.groupby("asset_id", sort=False).groups.items():
        asset_rows = out.loc[asset_idx].sort_values("timestamp")
        ts_asset_ns = asset_rows["timestamp"].astype("int64").to_numpy()
        asset_maint = maint.loc[maint["asset_id"] == asset_id].sort_values("start_time").copy()
        cip_end_times = asset_maint.loc[asset_maint["maintenance_type"].astype(str).isin(CIP_EVENT_TYPES), "end_time"].astype("int64").to_numpy()
        maint_end_times = asset_maint["end_time"].astype("int64").to_numpy()
        last_maint_type = previous_event_value(ts_asset_ns, maint_end_times, asset_maint.get("maintenance_type", pd.Series(dtype=str)).astype(str).tolist(), default="none")
        out.loc[asset_rows.index, "time_since_last_cip_min"] = minutes_since_last_event(ts_asset_ns, cip_end_times)
        out.loc[asset_rows.index, "time_since_last_maintenance_min"] = minutes_since_last_event(ts_asset_ns, maint_end_times)
        out.loc[asset_rows.index, "last_maintenance_type"] = last_maint_type

    for (asset_id, cycle_id), idx in out.groupby(["asset_id", "cycle_id"], sort=False).groups.items():
        g = out.loc[idx].sort_values("timestamp").copy()
        ts_ns = g["timestamp"].astype("int64").to_numpy()

        onset_times = g.loc[g["fouling_onset_event"] == 1, "timestamp"].astype("int64").to_numpy()
        clog_times = g.loc[g["clog_onset_event"] == 1, "timestamp"].astype("int64").to_numpy()
        ttf = minutes_to_next_event(ts_ns, onset_times)
        ttc = minutes_to_next_event(ts_ns, clog_times)

        asset_cycle_maint = maint.loc[(maint["asset_id"] == asset_id) & (maint["cycle_id"].astype(str) == str(cycle_id))].sort_values("start_time").copy()
        planned_starts = asset_cycle_maint.loc[asset_cycle_maint["planned"].fillna(0).astype(int) == 1, "start_time"].astype("int64").to_numpy()
        unplanned_df = asset_cycle_maint.loc[
            (asset_cycle_maint["planned"].fillna(0).astype(int) == 0)
            & (asset_cycle_maint["fault_type"].astype(str).isin(["fouling", "clogging"]))
        ].copy()
        unplanned_starts = unplanned_df["start_time"].astype("int64").to_numpy()
        next_unplanned_fault = next_event_value(ts_ns, unplanned_starts, unplanned_df["fault_type"].astype(str).tolist(), default="none")

        out.loc[g.index, "time_to_fouling_onset_min"] = ttf
        out.loc[g.index, "time_to_clog_onset_min"] = ttc
        out.loc[g.index, "ttm_to_planned_cip_min"] = minutes_to_next_event(ts_ns, planned_starts)
        out.loc[g.index, "ttm_to_unplanned_event_min"] = minutes_to_next_event(ts_ns, unplanned_starts)
        out.loc[g.index, f"fouling_onset_within_{cfg.fouling_horizon_min}min"] = ((ttf <= cfg.fouling_horizon_min) & np.isfinite(ttf)).astype(int)
        out.loc[g.index, f"clog_onset_within_{cfg.clog_horizon_min}min"] = ((ttc <= cfg.clog_horizon_min) & np.isfinite(ttc)).astype(int)
        out.loc[g.index, f"unplanned_event_within_{cfg.unplanned_horizon_min}min"] = (
            (out.loc[g.index, "ttm_to_unplanned_event_min"].to_numpy(dtype=float) <= cfg.unplanned_horizon_min)
            & np.isfinite(out.loc[g.index, "ttm_to_unplanned_event_min"].to_numpy(dtype=float))
        ).astype(int)
        out.loc[g.index, "next_unplanned_fault_type"] = next_unplanned_fault

    return out.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
