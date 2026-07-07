
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import base64
import io
import json
import math
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.paths import relativize_payload, resolve_saved_path, to_repo_relative_path

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"

PHASE_ORDER = ["production", "CIP_alkaline", "CIP_acid", "maintenance", "idle"]
STAGE_ORDER = ["stable", "incipient", "advanced"]
CIP_TYPES = {"scheduled_CIP", "CIP_extra"}
DEFAULT_PLOT_NUMERIC = [
    "Rf_m2K_W",
    "m_total_kg_m2",
    "flow_kg_s",
    "dP_kPa",
    "vibration_mm_s",
    "hot_drop_C",
    "cold_lift_C",
    "thermal_eff_proxy",
    "heat_proxy",
    "ttm_to_planned_cip_min",
    "ttm_to_unplanned_event_min",
    "time_to_fouling_onset_min",
    "time_to_clog_onset_min",
]
DEFAULT_CORR_COLS = [
    "Rf_m2K_W",
    "m_total_kg_m2",
    "flow_kg_s",
    "dP_kPa",
    "vibration_mm_s",
    "Th_in_C",
    "Tc_in_C",
    "Th_out_C",
    "Tc_out_C",
    "hot_drop_C",
    "cold_lift_C",
    "thermal_eff_proxy",
    "heat_proxy",
    "batch_elapsed_min",
    "time_since_last_cip_min",
    "time_since_last_maintenance_min",
    "ttm_to_planned_cip_min",
    "ttm_to_unplanned_event_min",
    "time_to_fouling_onset_min",
    "time_to_clog_onset_min",
]
@dataclass
class EDAConfig:
    telemetry: str
    maintenance: Optional[str]
    outdir: str
    seq_len: int = 120
    stride: int = 5
    sample_cycles: int = 10
    sample_rows_scatter: int = 50000
    random_seed: int = 7
    severity_col: str = "Rf_m2K_W"
    fouling_horizon_min: int = 30
    clog_horizon_min: int = 15
    unplanned_horizon_min: int = 60
    top_cycles_table_rows: int = 50

def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def _conv(x: Any) -> Any:
        if isinstance(x, (np.integer, np.int64, np.int32)):
            return int(x)
        if isinstance(x, (np.floating, np.float32, np.float64)):
            if math.isnan(float(x)) or math.isinf(float(x)):
                return None
            return float(x)
        if isinstance(x, (pd.Timestamp,)):
            return x.isoformat()
        return x
    with path.open("w", encoding="utf-8") as f:
        json.dump(relativize_payload(payload), f, indent=2, default=_conv)

def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def ensure_columns(df: pd.DataFrame, defaults: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    return out

def derive_cycle_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "asset_id" not in out.columns:
        out["asset_id"] = "asset_000"
    out["asset_id"] = out["asset_id"].astype(str)

    if "cycle_id" not in out.columns or out["cycle_id"].isna().all():
        candidate = None
        if "batch_id" in out.columns and (~out["batch_id"].astype(str).isin(["", "none", "nan", "None"])).any():
            candidate = out["batch_id"].astype(str)
        elif "episode_id" in out.columns and (~out["episode_id"].astype(str).isin(["", "none", "nan", "None"])).any():
            candidate = out["episode_id"].astype(str)
        if candidate is not None:
            out["cycle_id"] = candidate
        else:
            out["cycle_id"] = ""
            for asset_id, idx in out.groupby("asset_id", sort=False).groups.items():
                g = out.loc[idx].sort_values("timestamp")
                ts = pd.to_datetime(g["timestamp"], utc=True, errors="coerce")
                gap = ts.diff().dt.total_seconds().fillna(0.0)
                cycle_ord = (gap > 6 * 3600).cumsum() + 1
                out.loc[g.index, "cycle_id"] = [f"{asset_id}_C{int(v):05d}" for v in cycle_ord]
    out["cycle_id"] = out["cycle_id"].astype(str).replace({"nan": "none", "None": "none"})

    if "sequence_id" not in out.columns or out["sequence_id"].isna().all():
        out["sequence_id"] = out["asset_id"].astype(str) + "::" + out["cycle_id"].astype(str)
    else:
        out["sequence_id"] = out["sequence_id"].astype(str)
        pair_count = out[["asset_id", "cycle_id"]].drop_duplicates().shape[0]
        if out["sequence_id"].nunique() < pair_count:
            out["sequence_id"] = out["asset_id"].astype(str) + "::" + out["cycle_id"].astype(str)

    if "cycle_index" not in out.columns or pd.to_numeric(out["cycle_index"], errors="coerce").isna().all():
        out["cycle_index"] = 0
        for asset_id, idx in out.groupby("asset_id", sort=False).groups.items():
            g = out.loc[idx].sort_values("timestamp")
            seen: Dict[str, int] = {}
            nxt = 1
            vals: List[int] = []
            for cycle_id in g["cycle_id"].astype(str).tolist():
                if cycle_id not in seen:
                    seen[cycle_id] = nxt
                    nxt += 1
                vals.append(seen[cycle_id])
            out.loc[g.index, "cycle_index"] = vals
    out["cycle_index"] = pd.to_numeric(out["cycle_index"], errors="coerce").fillna(0).astype(int)
    return out

def add_derived_features(df: pd.DataFrame, cfg: EDAConfig) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values(["asset_id", "sequence_id", "timestamp"]).reset_index(drop=True)
    out = ensure_columns(
        out,
        {
            "phase": "unknown",
            "fouling_stage_name": "stable",
            "fouling_stage": 0,
            "maintenance_active": 0,
            "maintenance_planned": 0,
            "maintenance_type": "none",
            "fault_type": "none",
            "planned_event_type": "none",
            "clog_event": 0,
            "clog_onset_event": 0,
            "fouling_onset_event": 0,
            "ttm_to_planned_cip_min": np.nan,
            "ttm_to_unplanned_event_min": np.nan,
            "time_to_fouling_onset_min": np.nan,
            "time_to_clog_onset_min": np.nan,
            "time_since_last_cip_min": np.nan,
            "time_since_last_maintenance_min": np.nan,
            "last_maintenance_type": "none",
        },
    )

    num_defaults = {
        "flow_kg_s": np.nan,
        "pressure_in_kPa": np.nan,
        "pressure_out_kPa": np.nan,
        "dP_kPa": np.nan,
        "Th_in_C": np.nan,
        "Tc_in_C": np.nan,
        "Th_out_C": np.nan,
        "Tc_out_C": np.nan,
        "vibration_mm_s": np.nan,
        "batch_elapsed_min": np.nan,
        "Rf_m2K_W": np.nan,
        "m_total_kg_m2": np.nan,
    }
    out = ensure_columns(out, num_defaults)
    for col in num_defaults:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["hot_drop_C"] = out["Th_in_C"] - out["Th_out_C"]
    out["cold_lift_C"] = out["Tc_out_C"] - out["Tc_in_C"]
    thermal_span = (out["Th_in_C"] - out["Tc_in_C"]).replace(0.0, np.nan)
    out["thermal_eff_proxy"] = out["cold_lift_C"] / thermal_span
    out["heat_proxy"] = out["flow_kg_s"] * out["hot_drop_C"]

    if cfg.severity_col not in out.columns:
        fallback = "Rf_m2K_W" if "Rf_m2K_W" in out.columns else ("m_total_kg_m2" if "m_total_kg_m2" in out.columns else None)
        if fallback is None:
            out[cfg.severity_col] = np.nan
        else:
            out[cfg.severity_col] = pd.to_numeric(out[fallback], errors="coerce")

    out["severity_primary"] = pd.to_numeric(out[cfg.severity_col], errors="coerce")
    if "Rf_m2K_W" in out.columns and "rf_stage_thr_incipient" in out.columns and "rf_stage_thr_advanced" in out.columns:
        rf = pd.to_numeric(out["Rf_m2K_W"], errors="coerce")
        thr_i = pd.to_numeric(out["rf_stage_thr_incipient"], errors="coerce")
        thr_a = pd.to_numeric(out["rf_stage_thr_advanced"], errors="coerce")
        stage_ph = np.where(rf >= thr_a, 2, np.where(rf >= thr_i, 1, 0))
        out["fouling_stage_physical"] = pd.Series(stage_ph, index=out.index).fillna(0).astype(int)
    else:
        out["fouling_stage_physical"] = pd.to_numeric(out["fouling_stage"], errors="coerce").fillna(0).astype(int)

    out["fouling_stage"] = pd.to_numeric(out["fouling_stage"], errors="coerce").fillna(out["fouling_stage_physical"]).astype(int)
    if "fouling_stage_name" not in out.columns or out["fouling_stage_name"].isna().all():
        mapping = {0: "stable", 1: "incipient", 2: "advanced"}
        out["fouling_stage_name"] = out["fouling_stage"].map(mapping).fillna("stable")
    else:
        out["fouling_stage_name"] = out["fouling_stage_name"].astype(str).replace({"nan": "stable", "None": "stable"})

    # Sequence-level timing
    out["dt_min"] = np.nan
    out["row_in_sequence"] = 0
    out["sequence_rows"] = 0
    out["sequence_progress"] = np.nan
    out["sequence_start"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["sequence_end"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    for seq_id, idx in out.groupby("sequence_id", sort=False).groups.items():
        g = out.loc[idx].sort_values("timestamp")
        diffs = g["timestamp"].diff().dt.total_seconds().div(60.0)
        out.loc[g.index, "dt_min"] = diffs.fillna(diffs.dropna().median() if diffs.notna().any() else np.nan).astype(float)
        rows = np.arange(len(g), dtype=int)
        out.loc[g.index, "row_in_sequence"] = rows
        out.loc[g.index, "sequence_rows"] = len(g)
        out.loc[g.index, "sequence_progress"] = rows / max(len(g) - 1, 1)
        out.loc[g.index, "sequence_start"] = g["timestamp"].iloc[0]
        out.loc[g.index, "sequence_end"] = g["timestamp"].iloc[-1]

    out["phase"] = out["phase"].astype(str)
    out["maintenance_type"] = out["maintenance_type"].astype(str)
    out["fault_type"] = out["fault_type"].astype(str)
    out["planned_event_type"] = out["planned_event_type"].astype(str)
    out["last_maintenance_type"] = out["last_maintenance_type"].astype(str)

    # Informative vs low-information rows
    out["is_production"] = (out["phase"] == "production").astype(int)
    out["is_non_production"] = (out["phase"] != "production").astype(int)
    out["is_event_horizon_row"] = (
        (pd.to_numeric(out.get("fouling_onset_within_30min", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out.get("clog_onset_within_15min", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out.get("unplanned_event_within_60min", 0), errors="coerce").fillna(0) > 0)
    ).astype(int)
    out["is_informative_row"] = (
        (out["fouling_stage"] > 0)
        | (pd.to_numeric(out.get("clog_event", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out.get("clog_onset_event", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out.get("fouling_onset_event", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out.get("maintenance_active", 0), errors="coerce").fillna(0) > 0)
        | (out["is_event_horizon_row"] > 0)
        | (out["phase"] != "production")
    ).astype(int)
    out["is_low_information_production"] = (
        (out["phase"] == "production")
        & (out["fouling_stage"] == 0)
        & (pd.to_numeric(out.get("clog_event", 0), errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(out.get("clog_onset_event", 0), errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(out.get("fouling_onset_event", 0), errors="coerce").fillna(0) == 0)
        & (out["is_event_horizon_row"] == 0)
    ).astype(int)
    return out

def schema_summary(df: pd.DataFrame, max_top_values: int = 5) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        rec: Dict[str, Any] = {
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "missing_pct": float(100.0 * s.isna().mean()),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            ss = pd.to_numeric(s, errors="coerce")
            rec.update(
                {
                    "min": float(ss.min()) if ss.notna().any() else np.nan,
                    "p01": float(ss.quantile(0.01)) if ss.notna().any() else np.nan,
                    "p50": float(ss.quantile(0.50)) if ss.notna().any() else np.nan,
                    "p99": float(ss.quantile(0.99)) if ss.notna().any() else np.nan,
                    "max": float(ss.max()) if ss.notna().any() else np.nan,
                    "top_values": "",
                }
            )
        else:
            vc = s.astype(str).fillna("nan").value_counts(dropna=False).head(max_top_values)
            top = ", ".join([f"{k}:{int(v)}" for k, v in vc.items()])
            rec.update({"min": "", "p01": "", "p50": "", "p99": "", "max": "", "top_values": top})
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["missing_pct", "column"], ascending=[False, True])

def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    n = max(len(df), 1)
    for col in df.columns:
        recs.append(
            {
                "column": col,
                "missing_count": int(df[col].isna().sum()),
                "missing_pct": float(100.0 * df[col].isna().mean()),
                "non_null_count": int(df[col].notna().sum()),
                "filled_pct": float(100.0 * df[col].notna().mean()),
            }
        )
    return pd.DataFrame(recs).sort_values("missing_pct", ascending=False)

def safe_mode(series: pd.Series) -> Any:
    if len(series) == 0:
        return None
    mode = series.mode(dropna=True)
    if len(mode):
        return mode.iloc[0]
    if series.notna().any():
        return series.dropna().iloc[0]
    return None

def build_asset_summary(df: pd.DataFrame, maintenance_df: pd.DataFrame) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []
    maint_by_asset = {}
    if len(maintenance_df):
        maint_by_asset = {
            aid: g.copy()
            for aid, g in maintenance_df.groupby("asset_id", sort=False)
        }

    for asset_id, g in df.groupby("asset_id", sort=False):
        g = g.sort_values("timestamp")
        dt_vals = g["dt_min"].replace(0, np.nan)
        asset_secs = 60.0 * float(np.nansum(pd.to_numeric(dt_vals, errors="coerce").fillna(0.0)))
        mg = maint_by_asset.get(asset_id, pd.DataFrame())
        recs.append(
            {
                "asset_id": asset_id,
                "rows": int(len(g)),
                "cycles": int(g["sequence_id"].nunique()),
                "start_time": g["timestamp"].min(),
                "end_time": g["timestamp"].max(),
                "observed_hours_from_rows": float(asset_secs / 3600.0),
                "median_dt_min": float(np.nanmedian(g["dt_min"])) if g["dt_min"].notna().any() else np.nan,
                "phase_mode": safe_mode(g["phase"]),
                                "prod_row_pct": float(100.0 * (g["phase"] == "production").mean()),
                "informative_row_pct": float(100.0 * g["is_informative_row"].mean()),
                "low_info_prod_pct": float(100.0 * g["is_low_information_production"].mean()),
                "informative_within_production_pct": float(
                    100.0 * g.loc[g["phase"] == "production", "is_informative_row"].mean()
                ) if (g["phase"] == "production").any() else np.nan,
                "low_info_within_production_pct": float(
                    100.0 * g.loc[g["phase"] == "production", "is_low_information_production"].mean()
                ) if (g["phase"] == "production").any() else np.nan,
                "max_Rf_m2K_W": float(pd.to_numeric(g.get("Rf_m2K_W"), errors="coerce").max()) if "Rf_m2K_W" in g.columns else np.nan,
                "max_m_total_kg_m2": float(pd.to_numeric(g.get("m_total_kg_m2"), errors="coerce").max()) if "m_total_kg_m2" in g.columns else np.nan,
                "fouling_onset_events": int(pd.to_numeric(g.get("fouling_onset_event", 0), errors="coerce").fillna(0).sum()),
                "clog_onset_events": int(pd.to_numeric(g.get("clog_onset_event", 0), errors="coerce").fillna(0).sum()),
                "planned_maint_events": int((pd.to_numeric(mg.get("planned", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()) if len(mg) else 0,
                "unplanned_maint_events": int((pd.to_numeric(mg.get("planned", pd.Series(dtype=float)), errors="coerce").fillna(0) == 0).sum()) if len(mg) else 0,
                "maintenance_total": int(len(mg)),
            }
        )
    out = pd.DataFrame(recs)
    return out.sort_values(["cycles", "rows", "asset_id"], ascending=[False, False, True]).reset_index(drop=True)

def build_cycle_summary(df: pd.DataFrame, maintenance_df: pd.DataFrame) -> pd.DataFrame:
    maint_cycle = pd.DataFrame()
    if len(maintenance_df):
        m = maintenance_df.copy()
        if "cycle_id" not in m.columns:
            m["cycle_id"] = None
        m["cycle_id"] = m["cycle_id"].astype(str)
        m["planned"] = pd.to_numeric(m.get("planned", 0), errors="coerce").fillna(0).astype(int)
        maint_cycle = (
            m.groupby(["asset_id", "cycle_id"], dropna=False)
            .agg(
                maintenance_events=("maintenance_id", "count"),
                planned_maint_events=("planned", "sum"),
                unplanned_maint_events=("planned", lambda x: int((1 - x).clip(lower=0).sum())),
                maintenance_duration_min=("duration_min", "sum"),
                maintenance_types=("maintenance_type", lambda x: ",".join(sorted(set(x.astype(str))))),
                fault_types=("fault_type", lambda x: ",".join(sorted(set(x.astype(str))))),
            )
            .reset_index()
        )

    recs: List[Dict[str, Any]] = []
    for seq_id, g in df.groupby("sequence_id", sort=False):
        g = g.sort_values("timestamp")
        asset_id = str(g["asset_id"].iloc[0])
        cycle_id = str(g["cycle_id"].iloc[0])
        dt_med = float(np.nanmedian(g["dt_min"])) if g["dt_min"].notna().any() else np.nan
        duration_h = float((g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]).total_seconds() / 3600.0) if len(g) > 1 else 0.0
        prod = g.loc[g["phase"] == "production"]
        maint = g.loc[g["phase"] == "maintenance"]
        cip = g.loc[g["phase"].isin(["CIP_alkaline", "CIP_acid"])]
        rec = {
            "asset_id": asset_id,
            "sequence_id": seq_id,
            "cycle_id": cycle_id,
            "cycle_index": int(pd.to_numeric(g["cycle_index"], errors="coerce").iloc[0]) if "cycle_index" in g.columns else 0,
            "rows": int(len(g)),
            "start_time": g["timestamp"].iloc[0],
            "end_time": g["timestamp"].iloc[-1],
            "duration_h": duration_h,
            "observed_h_from_rows": float(np.nansum(g["dt_min"].fillna(0.0)) / 60.0),
            "median_dt_min": dt_med,
            "production_rows": int(len(prod)),
            "production_row_pct": float(100.0 * len(prod) / max(len(g), 1)),
            "cip_rows": int(len(cip)),
            "maintenance_rows": int(len(maint)),
                        "informative_row_pct": float(100.0 * g["is_informative_row"].mean()),
            "low_info_prod_pct": float(100.0 * g["is_low_information_production"].mean()),
            "informative_within_production_pct": float(
                100.0 * g.loc[g["phase"] == "production", "is_informative_row"].mean()
            ) if (g["phase"] == "production").any() else np.nan,
            "low_info_within_production_pct": float(
                100.0 * g.loc[g["phase"] == "production", "is_low_information_production"].mean()
            ) if (g["phase"] == "production").any() else np.nan,
            "max_stage": int(pd.to_numeric(g["fouling_stage"], errors="coerce").fillna(0).max()),
            "stage_mode": safe_mode(g["fouling_stage_name"]),
            "start_severity": float(pd.to_numeric(g["severity_primary"], errors="coerce").iloc[0]) if len(g) else np.nan,
            "end_severity": float(pd.to_numeric(g["severity_primary"], errors="coerce").iloc[-1]) if len(g) else np.nan,
            "max_severity": float(pd.to_numeric(g["severity_primary"], errors="coerce").max()) if len(g) else np.nan,
            "severity_increase": float(pd.to_numeric(g["severity_primary"], errors="coerce").iloc[-1] - pd.to_numeric(g["severity_primary"], errors="coerce").iloc[0]) if len(g) > 1 else np.nan,
            "fouling_onset_events": int(pd.to_numeric(g.get("fouling_onset_event", 0), errors="coerce").fillna(0).sum()),
            "clog_onset_events": int(pd.to_numeric(g.get("clog_onset_event", 0), errors="coerce").fillna(0).sum()),
            "clog_rows": int(pd.to_numeric(g.get("clog_event", 0), errors="coerce").fillna(0).sum()),
            "has_unplanned_horizon": int((pd.to_numeric(g.get("unplanned_event_within_60min", 0), errors="coerce").fillna(0) > 0).any()),
            "phase_set": ",".join(sorted(set(g["phase"].astype(str)))),
            "last_maintenance_type_mode": safe_mode(g["last_maintenance_type"]),
        }
        recs.append(rec)
    out = pd.DataFrame(recs)
    if len(maint_cycle):
        out = out.merge(maint_cycle, on=["asset_id", "cycle_id"], how="left")
    else:
        out["maintenance_events"] = 0
        out["planned_maint_events"] = 0
        out["unplanned_maint_events"] = 0
        out["maintenance_duration_min"] = 0.0
        out["maintenance_types"] = ""
        out["fault_types"] = ""
    for c in ["maintenance_events", "planned_maint_events", "unplanned_maint_events"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    out["maintenance_duration_min"] = pd.to_numeric(out["maintenance_duration_min"], errors="coerce").fillna(0.0)
    return out.sort_values(["asset_id", "cycle_index", "start_time"]).reset_index(drop=True)

def build_maintenance_summary(maintenance_df: pd.DataFrame) -> pd.DataFrame:
    if len(maintenance_df) == 0:
        return pd.DataFrame(columns=["maintenance_type", "planned", "fault_type", "events", "duration_min_sum", "duration_min_median"])
    m = maintenance_df.copy()
    m["planned"] = pd.to_numeric(m.get("planned", 0), errors="coerce").fillna(0).astype(int)
    m["duration_min"] = pd.to_numeric(m.get("duration_min", np.nan), errors="coerce")
    out = (
        m.groupby(["maintenance_type", "planned", "fault_type"], dropna=False)
        .agg(
            events=("maintenance_id", "count"),
            duration_min_sum=("duration_min", "sum"),
            duration_min_median=("duration_min", "median"),
        )
        .reset_index()
        .sort_values(["events", "maintenance_type"], ascending=[False, True])
    )
    return out

def build_target_consistency(df: pd.DataFrame, cfg: EDAConfig) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []

    def add_binary_consistency(bin_col: str, time_col: str, horizon: float) -> None:
        if bin_col not in df.columns or time_col not in df.columns:
            return
        b = pd.to_numeric(df[bin_col], errors="coerce").fillna(0).astype(int)
        t = pd.to_numeric(df[time_col], errors="coerce")
        should = ((t >= 0) & (t <= horizon)).fillna(False).astype(int)
        valid = t.notna() | (b == 0)
        mism = int(((b != should) & valid).sum())
        recs.append(
            {
                "check_name": f"{bin_col}_vs_{time_col}",
                "rows_evaluated": int(valid.sum()),
                "mismatch_count": mism,
                "mismatch_pct": float(100.0 * mism / max(int(valid.sum()), 1)),
                "detail": f"binary target should match 0 <= {time_col} <= {horizon}",
            }
        )

    add_binary_consistency("fouling_onset_within_30min", "time_to_fouling_onset_min", cfg.fouling_horizon_min)
    add_binary_consistency("clog_onset_within_15min", "time_to_clog_onset_min", cfg.clog_horizon_min)
    add_binary_consistency("unplanned_event_within_60min", "ttm_to_unplanned_event_min", cfg.unplanned_horizon_min)

    if "fouling_stage" in df.columns and "fouling_stage_physical" in df.columns:
        eval_mask = (
            (pd.to_numeric(df["fouling_stage"], errors="coerce").fillna(-999) >= 0)
            & (~df["phase"].astype(str).isin(["CIP_alkaline", "CIP_acid", "maintenance", "idle"]))
            & (df["fouling_stage_name"].astype(str) != "not_production")
        )
        a = pd.to_numeric(df.loc[eval_mask, "fouling_stage"], errors="coerce").fillna(0).astype(int)
        b = pd.to_numeric(df.loc[eval_mask, "fouling_stage_physical"], errors="coerce").fillna(0).astype(int)
        mism = int((a != b).sum())
        recs.append(
            {
                "check_name": "fouling_stage_vs_physical_thresholds",
                "rows_evaluated": int(eval_mask.sum()),
                "mismatch_count": mism,
                "mismatch_pct": float(100.0 * mism / max(int(eval_mask.sum()), 1)),
                "detail": "reported stage should match stage derived from physical thresholds (production rows only)",
            }
        )

    def countdown_consistency(time_col: str) -> None:
        if time_col not in df.columns:
            return
        diffs_total = 0
        bad = 0
        mae = []
        for _, g in df.groupby("sequence_id", sort=False):
            g = g.sort_values("timestamp")
            t = pd.to_numeric(g[time_col], errors="coerce")
            dt = pd.to_numeric(g["dt_min"], errors="coerce")
            mask = t.notna() & t.shift(-1).notna() & (t > 0) & (t.shift(-1) >= 0) & dt.shift(-1).notna()
            if not mask.any():
                continue
            lhs = t.shift(-1)[mask].to_numpy(dtype=float)
            rhs = (t[mask] - dt.shift(-1)[mask]).to_numpy(dtype=float)
            err = np.abs(lhs - rhs)
            diffs_total += int(len(err))
            bad += int((err > np.maximum(1.0, 0.15 * dt.shift(-1)[mask].to_numpy(dtype=float))).sum())
            mae.append(err.mean() if len(err) else np.nan)
        recs.append(
            {
                "check_name": f"{time_col}_countdown_step",
                "rows_evaluated": int(diffs_total),
                "mismatch_count": int(bad),
                "mismatch_pct": float(100.0 * bad / max(diffs_total, 1)),
                "detail": "next time-to-event should roughly equal current minus dt",
                "mae_min": float(np.nanmean(mae)) if len(mae) else np.nan,
            }
        )

    for col in ["ttm_to_planned_cip_min", "ttm_to_unplanned_event_min", "time_to_fouling_onset_min", "time_to_clog_onset_min"]:
        countdown_consistency(col)

    return pd.DataFrame(recs).sort_values(["mismatch_pct", "check_name"], ascending=[False, True]).reset_index(drop=True)

def build_sequence_qc(df: pd.DataFrame) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []
    for seq_id, g in df.groupby("sequence_id", sort=False):
        g = g.sort_values("timestamp")
        asset_id = str(g["asset_id"].iloc[0])
        cycle_id = str(g["cycle_id"].iloc[0])
        ts = g["timestamp"]
        diffs = ts.diff().dt.total_seconds().div(60.0)
        duplicate_ts = int(ts.duplicated().sum())
        non_monotonic = int((diffs.dropna() <= 0).sum())
        dt_pos = diffs[diffs > 0]
        dt_med = float(dt_pos.median()) if len(dt_pos) else np.nan
        dt_rel_dev = ((dt_pos - dt_med).abs() / max(dt_med, 1e-9)) if len(dt_pos) and math.isfinite(dt_med) else pd.Series(dtype=float)
        irregular_dt_rows = int((dt_rel_dev > 0.05).sum()) if len(dt_rel_dev) else 0

        prod = g.loc[g["phase"] == "production"].copy()
        neg_rf_steps = np.nan
        neg_mass_steps = np.nan
        if len(prod) > 1 and "Rf_m2K_W" in prod.columns:
            rf_diff = pd.to_numeric(prod["Rf_m2K_W"], errors="coerce").diff()
            neg_rf_steps = int((rf_diff.dropna() < -1e-12).sum())
        if len(prod) > 1 and "m_total_kg_m2" in prod.columns:
            mt_diff = pd.to_numeric(prod["m_total_kg_m2"], errors="coerce").diff()
            neg_mass_steps = int((mt_diff.dropna() < -1e-12).sum())

        recs.append(
            {
                "asset_id": asset_id,
                "sequence_id": seq_id,
                "cycle_id": cycle_id,
                "rows": int(len(g)),
                "duplicate_timestamps": duplicate_ts,
                "non_monotonic_diffs": non_monotonic,
                "median_dt_min": dt_med,
                "irregular_dt_rows": irregular_dt_rows,
                "negative_rf_steps_in_production": neg_rf_steps,
                "negative_mass_steps_in_production": neg_mass_steps,
                "fouling_onset_events": int(pd.to_numeric(g.get("fouling_onset_event", 0), errors="coerce").fillna(0).sum()),
                "clog_onset_events": int(pd.to_numeric(g.get("clog_onset_event", 0), errors="coerce").fillna(0).sum()),
                "unique_asset_in_seq": int(g["asset_id"].nunique()),
                "unique_cycle_in_seq": int(g["cycle_id"].nunique()),
            }
        )
    return pd.DataFrame(recs).sort_values(
        ["duplicate_timestamps", "non_monotonic_diffs", "irregular_dt_rows", "rows"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

def build_inter_cycle_gaps(df: pd.DataFrame) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []
    for asset_id, g in df.groupby("asset_id", sort=False):
        seq = (
            g.groupby("sequence_id", sort=False)
            .agg(
                cycle_id=("cycle_id", "first"),
                cycle_index=("cycle_index", "first"),
                start_time=("timestamp", "min"),
                end_time=("timestamp", "max"),
                rows=("sequence_id", "size"),
            )
            .reset_index()
            .sort_values("start_time")
        )
        seq["next_start_time"] = seq["start_time"].shift(-1)
        seq["next_cycle_id"] = seq["cycle_id"].shift(-1)
        seq["inter_cycle_gap_h"] = (seq["next_start_time"] - seq["end_time"]).dt.total_seconds().div(3600.0)
        seq = seq.iloc[:-1].copy()
        for _, r in seq.iterrows():
            recs.append(
                {
                    "asset_id": asset_id,
                    "cycle_id": r["cycle_id"],
                    "next_cycle_id": r["next_cycle_id"],
                    "cycle_index": int(r["cycle_index"]),
                    "end_time": r["end_time"],
                    "next_start_time": r["next_start_time"],
                    "inter_cycle_gap_h": float(r["inter_cycle_gap_h"]),
                    "rows_in_cycle": int(r["rows"]),
                }
            )
    return pd.DataFrame(recs).sort_values(["asset_id", "cycle_index"]).reset_index(drop=True)

def count_naive_cross_cycle_windows(df: pd.DataFrame, seq_len: int, stride: int) -> Dict[str, Any]:
    total_windows = 0
    crossing = 0
    by_asset = []
    for asset_id, g in df.groupby("asset_id", sort=False):
        g = g.sort_values("timestamp")
        seq = g["sequence_id"].astype(str).to_numpy()
        n = len(seq)
        if n < seq_len:
            by_asset.append({"asset_id": asset_id, "naive_windows": 0, "crossing_windows": 0})
            continue
        asset_total = 0
        asset_cross = 0
        for start in range(0, n - seq_len + 1, stride):
            asset_total += 1
            win = seq[start:start + seq_len]
            if win[0] != win[-1]:
                asset_cross += 1
        total_windows += asset_total
        crossing += asset_cross
        by_asset.append({"asset_id": asset_id, "naive_windows": asset_total, "crossing_windows": asset_cross})
    return {
        "seq_len": int(seq_len),
        "stride": int(stride),
        "naive_asset_windows_total": int(total_windows),
        "naive_cross_cycle_windows_total": int(crossing),
        "naive_cross_cycle_window_pct": float(100.0 * crossing / max(total_windows, 1)),
        "by_asset": by_asset,
    }

def expected_sequence_windows(df: pd.DataFrame, seq_len: int, stride: int) -> Dict[str, Any]:
    total = 0
    short = 0
    per_seq = []
    for seq_id, g in df.groupby("sequence_id", sort=False):
        n = len(g)
        possible = 0 if n < seq_len else 1 + (n - seq_len) // stride
        if n < seq_len:
            short += 1
        total += possible
        per_seq.append(
            {
                "sequence_id": seq_id,
                "asset_id": str(g["asset_id"].iloc[0]),
                "cycle_id": str(g["cycle_id"].iloc[0]),
                "rows": int(n),
                "windows": int(possible),
            }
        )
    return {
        "seq_len": int(seq_len),
        "stride": int(stride),
        "num_sequences": int(df["sequence_id"].nunique()),
        "sequences_shorter_than_seq_len": int(short),
        "sequence_aware_windows_total": int(total),
        "bad_windows_crossing_cycle_boundary": 0,
        "per_sequence": per_seq,
    }

def plot_bar_from_series(series: pd.Series, title: str, xlabel: str, ylabel: str, outpath: Path, rotation: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = series.copy()
    ax.bar(np.arange(len(vals)), vals.values)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels([str(x) for x in vals.index.tolist()], rotation=rotation, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_hist(series: pd.Series, title: str, xlabel: str, outpath: Path, bins: int = 50, logx: bool = False) -> None:
    arr = pd.to_numeric(series, errors="coerce")
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = arr.to_numpy(dtype=float)
    if logx and np.all(vals > 0):
        vals = np.log10(vals)
        ax.hist(vals, bins=bins)
        ax.set_xlabel(f"log10({xlabel})")
    else:
        ax.hist(vals, bins=bins)
        ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_box_by_category(df: pd.DataFrame, value_col: str, cat_col: str, title: str, outpath: Path, order: Optional[List[str]] = None) -> None:
    if value_col not in df.columns or cat_col not in df.columns:
        return
    sub = df[[value_col, cat_col]].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna()
    if len(sub) == 0:
        return
    cats = order if order is not None else sorted(sub[cat_col].astype(str).unique().tolist())
    data = [sub.loc[sub[cat_col].astype(str) == c, value_col].to_numpy(dtype=float) for c in cats]
    data = [d for d in data if len(d)]
    cats = [c for c in cats if len(sub.loc[sub[cat_col].astype(str) == c])]
    if len(data) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, tick_labels=cats, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel(cat_col)
    ax.set_ylabel(value_col)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str], title: str, outpath: Path, sample_n: int, seed: int) -> None:
    cols = [x_col, y_col] + ([color_col] if color_col else [])
    if any(c not in df.columns for c in cols if c):
        return
    sub = df[cols].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna(subset=[x_col, y_col])
    if len(sub) == 0:
        return
    if len(sub) > sample_n:
        sub = sub.sample(sample_n, random_state=seed)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    if color_col and color_col in sub.columns:
        cats = sub[color_col].astype(str)
        unique = sorted(cats.unique().tolist())
        for cat in unique:
            mask = cats == cat
            ax.scatter(sub.loc[mask, x_col], sub.loc[mask, y_col], s=7, alpha=0.35, label=cat)
        ax.legend(loc="best", fontsize=7)
    else:
        ax.scatter(sub[x_col], sub[y_col], s=7, alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_corr_heatmap(df: pd.DataFrame, cols: Sequence[str], outpath: Path, title: str) -> Optional[pd.DataFrame]:
    keep = [c for c in cols if c in df.columns]
    if len(keep) < 2:
        return None
    sub = df[keep].copy()
    for c in keep:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(how="all")
    corr = sub.corr(numeric_only=True)
    if corr.shape[0] < 2:
        return None
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, aspect="auto", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return corr

def plot_cycle_profile_by_outcome(cycle_summary: pd.DataFrame, outpath: Path, severity_label: str) -> None:
    if len(cycle_summary) == 0 or "max_severity" not in cycle_summary.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 100, 21)
    for label, mask in [
        ("planned_only", cycle_summary["unplanned_maint_events"].fillna(0) == 0),
        ("has_unplanned", cycle_summary["unplanned_maint_events"].fillna(0) > 0),
        ("has_clog", cycle_summary["clog_onset_events"].fillna(0) > 0),
    ]:
        vals = pd.to_numeric(cycle_summary.loc[mask, "max_severity"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        hist, edges = np.histogram(vals, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, hist, marker="o", label=label)
    ax.set_title(f"Distribución de severidad máxima por tipo de ciclo ({severity_label})")
    ax.set_xlabel("percentil interno de severidad máxima")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def plot_normalized_cycle_severity(df: pd.DataFrame, cycle_summary: pd.DataFrame, outpath: Path, severity_col: str) -> None:
    if severity_col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, seq_ids in [
        ("planned_only", cycle_summary.loc[cycle_summary["unplanned_maint_events"].fillna(0) == 0, "sequence_id"].tolist()),
        ("has_unplanned", cycle_summary.loc[cycle_summary["unplanned_maint_events"].fillna(0) > 0, "sequence_id"].tolist()),
        ("has_clog", cycle_summary.loc[cycle_summary["clog_onset_events"].fillna(0) > 0, "sequence_id"].tolist()),
    ]:
        if len(seq_ids) == 0:
            continue
        curves = []
        for seq_id in seq_ids[:200]:
            g = df.loc[df["sequence_id"] == seq_id, ["sequence_progress", severity_col]].copy()
            g[severity_col] = pd.to_numeric(g[severity_col], errors="coerce")
            g = g.dropna()
            if len(g) < 5:
                continue
            x = g["sequence_progress"].to_numpy(dtype=float)
            y = g[severity_col].to_numpy(dtype=float)
            grid = np.linspace(0, 1, 101)
            interp = np.interp(grid, x, y)
            curves.append(interp)
        if len(curves) == 0:
            continue
        mean_curve = np.nanmean(np.vstack(curves), axis=0)
        ax.plot(np.linspace(0, 100, 101), mean_curve, label=label)
    ax.set_title(f"Severidad media a lo largo del ciclo ({severity_col})")
    ax.set_xlabel("% del ciclo")
    ax.set_ylabel(severity_col)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def choose_sample_cycles(cycle_summary: pd.DataFrame, sample_cycles: int, seed: int) -> pd.DataFrame:
    if len(cycle_summary) == 0:
        return cycle_summary.copy()
    chosen: List[str] = []
    reasons: Dict[str, str] = {}

    def add_rows(rows: pd.DataFrame, reason: str) -> None:
        nonlocal chosen, reasons
        for _, r in rows.iterrows():
            seq = str(r["sequence_id"])
            if seq not in chosen:
                chosen.append(seq)
                reasons[seq] = reason
            if len(chosen) >= sample_cycles:
                return

    add_rows(cycle_summary.loc[cycle_summary["clog_onset_events"].fillna(0) > 0].sort_values("max_severity", ascending=False).head(sample_cycles), "clog_cycle")
    if len(chosen) < sample_cycles:
        add_rows(cycle_summary.loc[cycle_summary["unplanned_maint_events"].fillna(0) > 0].sort_values("max_severity", ascending=False).head(sample_cycles), "unplanned_cycle")
    if len(chosen) < sample_cycles:
        add_rows(cycle_summary.sort_values("max_severity", ascending=False).head(sample_cycles), "high_severity")
    if len(chosen) < sample_cycles:
        add_rows(cycle_summary.sort_values("duration_h", ascending=False).head(sample_cycles), "long_cycle")
    if len(chosen) < sample_cycles:
        rnd = cycle_summary.sample(min(sample_cycles, len(cycle_summary)), random_state=seed)
        add_rows(rnd, "random")

    out = cycle_summary.loc[cycle_summary["sequence_id"].isin(chosen)].copy()
    out["selection_reason"] = out["sequence_id"].map(reasons).fillna("selected")
    order = {seq: i for i, seq in enumerate(chosen)}
    out["selection_order"] = out["sequence_id"].map(order)
    return out.sort_values("selection_order").reset_index(drop=True)

def plot_sample_cycle(df: pd.DataFrame, maintenance_df: pd.DataFrame, sequence_id: str, outpath: Path, csv_outpath: Optional[Path], severity_col: str) -> None:
    g = df.loc[df["sequence_id"] == sequence_id].copy()
    if len(g) == 0:
        return
    g = g.sort_values("timestamp")
    if csv_outpath is not None:
        save_df(g, csv_outpath)

    asset_id = str(g["asset_id"].iloc[0])
    cycle_id = str(g["cycle_id"].iloc[0])
    mt = maintenance_df.loc[(maintenance_df["asset_id"].astype(str) == asset_id) & (maintenance_df.get("cycle_id", "").astype(str) == cycle_id)].copy() if len(maintenance_df) else pd.DataFrame()

    t = g["timestamp"]
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    # Panel 1: flow + dP
    axes[0].plot(t, pd.to_numeric(g["flow_kg_s"], errors="coerce"), label="flow_kg_s")
    if "dP_kPa" in g.columns:
        axes[0].plot(t, pd.to_numeric(g["dP_kPa"], errors="coerce"), label="dP_kPa")
    axes[0].set_ylabel("hidráulica")
    axes[0].legend(loc="best", fontsize=8)

    # Panel 2: thermal
    for col in ["Th_in_C", "Th_out_C", "Tc_in_C", "Tc_out_C"]:
        if col in g.columns:
            axes[1].plot(t, pd.to_numeric(g[col], errors="coerce"), label=col)
    axes[1].set_ylabel("temperaturas")
    axes[1].legend(loc="best", fontsize=8, ncol=2)

    # Panel 3: vibration / efficiency
    if "vibration_mm_s" in g.columns:
        axes[2].plot(t, pd.to_numeric(g["vibration_mm_s"], errors="coerce"), label="vibration_mm_s")
    if "thermal_eff_proxy" in g.columns:
        axes[2].plot(t, pd.to_numeric(g["thermal_eff_proxy"], errors="coerce"), label="thermal_eff_proxy")
    axes[2].set_ylabel("condición")
    axes[2].legend(loc="best", fontsize=8)

    # Panel 4: severity
    if severity_col in g.columns:
        axes[3].plot(t, pd.to_numeric(g[severity_col], errors="coerce"), label=severity_col)
    if "Rf_m2K_W" in g.columns and "rf_stage_thr_incipient" in g.columns and "rf_stage_thr_advanced" in g.columns and severity_col == "Rf_m2K_W":
        axes[3].plot(t, pd.to_numeric(g["rf_stage_thr_incipient"], errors="coerce"), linestyle="--", label="thr_incipient")
        axes[3].plot(t, pd.to_numeric(g["rf_stage_thr_advanced"], errors="coerce"), linestyle="--", label="thr_advanced")
    axes[3].set_ylabel("severidad")
    axes[3].legend(loc="best", fontsize=8)

    # Panel 5: countdowns + phase
    for col in ["time_to_fouling_onset_min", "time_to_clog_onset_min", "ttm_to_unplanned_event_min", "ttm_to_planned_cip_min"]:
        if col in g.columns:
            axes[4].plot(t, pd.to_numeric(g[col], errors="coerce"), label=col)
    axes[4].set_ylabel("minutos")
    axes[4].legend(loc="best", fontsize=7, ncol=2)

    # Event lines
    def add_event_lines(ax) -> None:
        if "fouling_onset_event" in g.columns:
            for ts in g.loc[pd.to_numeric(g["fouling_onset_event"], errors="coerce").fillna(0) > 0, "timestamp"]:
                ax.axvline(ts, linestyle="--", linewidth=0.8)
        if "clog_onset_event" in g.columns:
            for ts in g.loc[pd.to_numeric(g["clog_onset_event"], errors="coerce").fillna(0) > 0, "timestamp"]:
                ax.axvline(ts, linestyle=":", linewidth=1.0)
        if len(mt):
            for ts in pd.to_datetime(mt["start_time"], utc=True, errors="coerce").dropna():
                ax.axvline(ts, linestyle="-.", linewidth=0.8)

    for ax in axes:
        add_event_lines(ax)

    stage_text = ",".join(sorted(set(g["fouling_stage_name"].astype(str))))
    fig.suptitle(f"asset={asset_id} | cycle={cycle_id} | seq={sequence_id} | stages={stage_text}", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def build_overview_dict(df: pd.DataFrame, maintenance_df: pd.DataFrame, cfg: EDAConfig) -> Dict[str, Any]:
    time_min = df["timestamp"].min()
    time_max = df["timestamp"].max()
    phase_counts = df["phase"].astype(str).value_counts(dropna=False).to_dict()
    stage_counts = df["fouling_stage_name"].astype(str).value_counts(dropna=False).to_dict()
    overview: Dict[str, Any] = {
        "telemetry_rows": int(len(df)),
        "telemetry_columns": int(df.shape[1]),
        "assets": int(df["asset_id"].nunique()),
        "cycles": int(df["sequence_id"].nunique()),
        "time_min": time_min,
        "time_max": time_max,
        "time_span_h": float((time_max - time_min).total_seconds() / 3600.0) if pd.notna(time_min) and pd.notna(time_max) else np.nan,
        "median_dt_min": float(np.nanmedian(pd.to_numeric(df["dt_min"], errors="coerce"))),
        "phase_counts": phase_counts,
        "stage_counts": stage_counts,
                "informative_row_pct": float(100.0 * df["is_informative_row"].mean()),
        "low_info_production_row_pct": float(100.0 * df["is_low_information_production"].mean()),
        "production_row_pct": float(100.0 * (df["phase"] == "production").mean()),
        "informative_within_production_pct": float(
            100.0 * df.loc[df["phase"] == "production", "is_informative_row"].mean()
        ) if (df["phase"] == "production").any() else np.nan,
        "low_info_within_production_pct": float(
            100.0 * df.loc[df["phase"] == "production", "is_low_information_production"].mean()
        ) if (df["phase"] == "production").any() else np.nan,
        "severity_col": cfg.severity_col,
        "severity_min": float(pd.to_numeric(df["severity_primary"], errors="coerce").min()) if df["severity_primary"].notna().any() else np.nan,
        "severity_p50": float(pd.to_numeric(df["severity_primary"], errors="coerce").quantile(0.50)) if df["severity_primary"].notna().any() else np.nan,
        "severity_p99": float(pd.to_numeric(df["severity_primary"], errors="coerce").quantile(0.99)) if df["severity_primary"].notna().any() else np.nan,
        "severity_max": float(pd.to_numeric(df["severity_primary"], errors="coerce").max()) if df["severity_primary"].notna().any() else np.nan,
    }
    if len(maintenance_df):
        m = maintenance_df.copy()
        m["planned"] = pd.to_numeric(m.get("planned", 0), errors="coerce").fillna(0).astype(int)
        m["duration_min"] = pd.to_numeric(m.get("duration_min", np.nan), errors="coerce")
        overview.update(
            {
                "maintenance_rows": int(len(m)),
                "planned_maintenance_events": int((m["planned"] > 0).sum()),
                "unplanned_maintenance_events": int((m["planned"] == 0).sum()),
                "maintenance_duration_min_total": float(m["duration_min"].sum()) if m["duration_min"].notna().any() else np.nan,
                "maintenance_duration_min_median": float(m["duration_min"].median()) if m["duration_min"].notna().any() else np.nan,
            }
        )
    else:
        overview.update(
            {
                "maintenance_rows": 0,
                "planned_maintenance_events": 0,
                "unplanned_maintenance_events": 0,
                "maintenance_duration_min_total": 0.0,
                "maintenance_duration_min_median": np.nan,
            }
        )
    return overview

def render_table_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or len(df) == 0:
        return "<p><em>sin datos</em></p>"
    show = df.head(max_rows).copy()
    for c in show.columns:
        if pd.api.types.is_datetime64_any_dtype(show[c]):
            show[c] = show[c].astype(str)
    return show.to_html(index=False, border=0, classes="table table-sm")

def png_to_base64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def build_markdown_report(
    outdir: Path,
    overview: Mapping[str, Any],
    qc_summary: Mapping[str, Any],
    top_cycle_df: pd.DataFrame,
    target_consistency_df: pd.DataFrame,
    key_plot_paths: Sequence[Path],
    sample_plot_paths: Sequence[Path],
) -> None:
    lines: List[str] = []
    lines.append("# CU07 synthetic dataset EDA")
    lines.append("")
    lines.append("## Resumen ejecutivo")
    lines.append("")
    for k, v in overview.items():
        if isinstance(v, dict):
            continue
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Checks de calidad")
    lines.append("")
    for k, v in qc_summary.items():
        if isinstance(v, (dict, list)):
            continue
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Top ciclos por severidad")
    lines.append("")
    if len(top_cycle_df):
        lines.append(top_cycle_df.to_markdown(index=False))
    else:
        lines.append("_sin datos_")
    lines.append("")
    lines.append("## Consistencia de targets")
    lines.append("")
    if len(target_consistency_df):
        lines.append(target_consistency_df.to_markdown(index=False))
    else:
        lines.append("_sin datos_")
    lines.append("")
    lines.append("## Gráficos clave")
    lines.append("")
    for p in key_plot_paths:
        rel = p.relative_to(outdir).as_posix()
        lines.append(f"### {p.stem}")
        lines.append(f"![]({rel})")
        lines.append("")
    lines.append("## Muestras de ciclos")
    lines.append("")
    for p in sample_plot_paths:
        rel = p.relative_to(outdir).as_posix()
        lines.append(f"### {p.stem}")
        lines.append(f"![]({rel})")
        lines.append("")
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")

def build_html_report(
    outdir: Path,
    overview: Mapping[str, Any],
    qc_summary: Mapping[str, Any],
    asset_summary: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    target_consistency_df: pd.DataFrame,
    key_plot_paths: Sequence[Path],
    sample_plot_paths: Sequence[Path],
) -> None:
    overview_rows = "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>"
        for k, v in overview.items() if not isinstance(v, (dict, list))
    )
    qc_rows = "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>"
        for k, v in qc_summary.items() if not isinstance(v, (dict, list))
    )
    key_imgs = []
    for p in key_plot_paths:
        if p.exists():
            key_imgs.append(
                f"<h3>{p.stem}</h3><img src='data:image/png;base64,{png_to_base64(p)}' style='max-width:100%; border:1px solid #ddd;'/>"
            )
    sample_imgs = []
    for p in sample_plot_paths[:8]:
        if p.exists():
            sample_imgs.append(
                f"<h3>{p.stem}</h3><img src='data:image/png;base64,{png_to_base64(p)}' style='max-width:100%; border:1px solid #ddd;'/>"
            )
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>CU07 synthetic EDA</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.4; }}
h1, h2, h3 {{ color: #222; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; font-size: 13px; vertical-align: top; }}
th {{ background: #f4f4f4; }}
.small {{ font-size: 12px; color: #555; }}
</style>
</head>
<body>
<h1>CU07 synthetic dataset EDA</h1>
<p class="small">Este reporte es autogenerado por <code>cu07_full_eda_predictive.py</code>.</p>

<h2>Resumen ejecutivo</h2>
<table>{overview_rows}</table>

<h2>Checks de calidad</h2>
<table>{qc_rows}</table>

<h2>Top assets</h2>
{render_table_html(asset_summary.head(20), max_rows=20)}

<h2>Top ciclos</h2>
{render_table_html(cycle_summary.sort_values("max_severity", ascending=False).head(20), max_rows=20)}

<h2>Consistencia de targets</h2>
{render_table_html(target_consistency_df, max_rows=20)}

<h2>Gráficos clave</h2>
{''.join(key_imgs)}

<h2>Muestras de ciclos</h2>
{''.join(sample_imgs)}

<p class="small">Artefactos adicionales: ver <code>manifest.json</code>, carpetas <code>tables/</code>, <code>plots/</code>, <code>samples/</code> y <code>qc/</code>.</p>
</body>
</html>"""
    (outdir / "report.html").write_text(html, encoding="utf-8")

def build_manifest(outdir: Path) -> Dict[str, Any]:
    files = []
    for p in sorted(outdir.rglob("*")):
        if p.is_file():
            files.append(
                {
                    "relative_path": p.relative_to(outdir).as_posix(),
                    "size_bytes": p.stat().st_size,
                }
            )
    return {"outdir": to_repo_relative_path(outdir), "files": files, "file_count": len(files)}

def run_eda(cfg: EDAConfig) -> dict[str, Any]:
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    outdir = resolve_saved_path(cfg.outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    (outdir / "samples").mkdir(parents=True, exist_ok=True)
    (outdir / "qc").mkdir(parents=True, exist_ok=True)

    telemetry_df = pd.read_csv(resolve_saved_path(cfg.telemetry))
    telemetry_df = derive_cycle_columns(telemetry_df)
    telemetry_df = add_derived_features(telemetry_df, cfg)

    maintenance_df = pd.DataFrame()
    if cfg.maintenance:
        mpath = resolve_saved_path(cfg.maintenance)
        if mpath.exists():
            maintenance_df = pd.read_csv(mpath)
            if len(maintenance_df):
                maintenance_df = ensure_columns(
                    maintenance_df,
                    {
                        "maintenance_id": "",
                        "asset_id": "",
                        "cycle_id": "",
                        "cycle_index": np.nan,
                        "planned": 0,
                        "maintenance_type": "none",
                        "fault_type": "none",
                        "duration_min": np.nan,
                        "start_time": pd.NaT,
                        "end_time": pd.NaT,
                    },
                )
                maintenance_df["asset_id"] = maintenance_df["asset_id"].astype(str)
                maintenance_df["cycle_id"] = maintenance_df["cycle_id"].astype(str)
                maintenance_df["start_time"] = pd.to_datetime(maintenance_df["start_time"], utc=True, errors="coerce")
                maintenance_df["end_time"] = pd.to_datetime(maintenance_df["end_time"], utc=True, errors="coerce")
                maintenance_df["planned"] = pd.to_numeric(maintenance_df["planned"], errors="coerce").fillna(0).astype(int)
                maintenance_df["duration_min"] = pd.to_numeric(maintenance_df["duration_min"], errors="coerce")

    save_json(outdir / "eda_config.json", asdict(cfg))
    save_df(telemetry_df.head(1000), outdir / "tables" / "telemetry_head_1000.csv")
    if len(maintenance_df):
        save_df(maintenance_df.head(1000), outdir / "tables" / "maintenance_head_1000.csv")

    tele_schema = schema_summary(telemetry_df)
    tele_missing = missingness_summary(telemetry_df)
    save_df(tele_schema, outdir / "tables" / "telemetry_schema.csv")
    save_df(tele_missing, outdir / "tables" / "telemetry_missingness.csv")
    if len(maintenance_df):
        save_df(schema_summary(maintenance_df), outdir / "tables" / "maintenance_schema.csv")
        save_df(missingness_summary(maintenance_df), outdir / "tables" / "maintenance_missingness.csv")

    asset_summary = build_asset_summary(telemetry_df, maintenance_df)
    cycle_summary = build_cycle_summary(telemetry_df, maintenance_df)
    maintenance_summary = build_maintenance_summary(maintenance_df)
    target_consistency = build_target_consistency(telemetry_df, cfg)
    sequence_qc = build_sequence_qc(telemetry_df)
    inter_cycle_gaps = build_inter_cycle_gaps(telemetry_df)
    window_qc = expected_sequence_windows(telemetry_df, cfg.seq_len, cfg.stride)
    naive_window_qc = count_naive_cross_cycle_windows(telemetry_df, cfg.seq_len, cfg.stride)

    save_df(asset_summary, outdir / "tables" / "asset_summary.csv")
    save_df(cycle_summary, outdir / "tables" / "cycle_summary.csv")
    save_df(cycle_summary.sort_values("max_severity", ascending=False).head(cfg.top_cycles_table_rows), outdir / "tables" / "top_cycles_by_severity.csv")
    save_df(maintenance_summary, outdir / "tables" / "maintenance_summary.csv")
    save_df(target_consistency, outdir / "qc" / "target_consistency.csv")
    save_df(sequence_qc, outdir / "qc" / "sequence_qc.csv")
    save_df(inter_cycle_gaps, outdir / "qc" / "inter_cycle_gaps.csv")
    save_df(pd.DataFrame(window_qc["per_sequence"]), outdir / "qc" / "window_capacity_per_sequence.csv")
    save_df(pd.DataFrame(naive_window_qc["by_asset"]), outdir / "qc" / "naive_window_crossing_by_asset.csv")
    save_json(outdir / "qc" / "window_boundary_check.json", window_qc)
    save_json(outdir / "qc" / "naive_window_crossing_check.json", naive_window_qc)

    overview = build_overview_dict(telemetry_df, maintenance_df, cfg)
    phase_counts = telemetry_df["phase"].astype(str).value_counts().reindex(PHASE_ORDER).dropna()
    stage_counts = telemetry_df["fouling_stage_name"].astype(str).value_counts().reindex(STAGE_ORDER).dropna()
    maintenance_type_counts = maintenance_df["maintenance_type"].astype(str).value_counts() if len(maintenance_df) else pd.Series(dtype=int)
    fault_type_counts = maintenance_df["fault_type"].astype(str).value_counts() if len(maintenance_df) else pd.Series(dtype=int)

    save_df(phase_counts.rename_axis("phase").reset_index(name="rows"), outdir / "tables" / "phase_counts.csv")
    save_df(stage_counts.rename_axis("fouling_stage_name").reset_index(name="rows"), outdir / "tables" / "stage_counts.csv")
    if len(maintenance_type_counts):
        save_df(maintenance_type_counts.rename_axis("maintenance_type").reset_index(name="events"), outdir / "tables" / "maintenance_type_counts.csv")
    if len(fault_type_counts):
        save_df(fault_type_counts.rename_axis("fault_type").reset_index(name="events"), outdir / "tables" / "fault_type_counts.csv")

    key_plots: List[Path] = []
    p = outdir / "plots" / "phase_counts.png"
    plot_bar_from_series(phase_counts, "Rows por fase", "phase", "rows", p)
    key_plots.append(p)

    p = outdir / "plots" / "stage_counts.png"
    plot_bar_from_series(stage_counts, "Rows por stage", "fouling stage", "rows", p)
    key_plots.append(p)

    if len(maintenance_type_counts):
        p = outdir / "plots" / "maintenance_type_counts.png"
        plot_bar_from_series(maintenance_type_counts, "Eventos de mantenimiento por tipo", "maintenance_type", "events", p)
        key_plots.append(p)

    if len(fault_type_counts):
        p = outdir / "plots" / "maintenance_fault_type_counts.png"
        plot_bar_from_series(fault_type_counts, "Eventos de mantenimiento por fault_type", "fault_type", "events", p)
        key_plots.append(p)

    p = outdir / "plots" / "cycle_duration_hist.png"
    plot_hist(cycle_summary["duration_h"], "Duración de ciclo", "duration_h", p, bins=40)
    key_plots.append(p)

    p = outdir / "plots" / "rows_per_cycle_hist.png"
    plot_hist(cycle_summary["rows"], "Rows por ciclo", "rows", p, bins=40)
    key_plots.append(p)

    p = outdir / "plots" / "informative_row_pct_by_cycle_hist.png"
    plot_hist(cycle_summary["informative_row_pct"], "Fracción de rows informativos por ciclo", "informative_row_pct", p, bins=40)
    key_plots.append(p)

    if len(inter_cycle_gaps):
        p = outdir / "plots" / "inter_cycle_gap_hist.png"
        plot_hist(inter_cycle_gaps["inter_cycle_gap_h"], "Huecos entre ciclos", "inter_cycle_gap_h", p, bins=40)
        key_plots.append(p)

    for col in DEFAULT_PLOT_NUMERIC:
        if col in telemetry_df.columns:
            p = outdir / "plots" / f"{col}_hist.png"
            plot_hist(telemetry_df[col], f"Distribución de {col}", col, p, bins=50, logx=(col in {"Rf_m2K_W", "m_total_kg_m2"}))
            key_plots.append(p)

    if "severity_primary" in telemetry_df.columns:
        p = outdir / "plots" / "severity_by_stage_box.png"
        plot_box_by_category(telemetry_df, "severity_primary", "fouling_stage_name", "Severidad por stage", p, order=STAGE_ORDER)
        key_plots.append(p)

    if "dP_kPa" in telemetry_df.columns and "flow_kg_s" in telemetry_df.columns:
        p = outdir / "plots" / "flow_vs_dp_by_stage.png"
        plot_scatter(telemetry_df, "flow_kg_s", "dP_kPa", "fouling_stage_name", "Flow vs dP por stage", p, cfg.sample_rows_scatter, cfg.random_seed)
        key_plots.append(p)

    if "severity_primary" in telemetry_df.columns and "thermal_eff_proxy" in telemetry_df.columns:
        p = outdir / "plots" / "severity_vs_thermal_eff.png"
        plot_scatter(telemetry_df, "severity_primary", "thermal_eff_proxy", "fouling_stage_name", f"{cfg.severity_col} vs thermal_eff_proxy", p, cfg.sample_rows_scatter, cfg.random_seed)
        key_plots.append(p)

    if "severity_primary" in telemetry_df.columns and "dP_kPa" in telemetry_df.columns:
        p = outdir / "plots" / "severity_vs_dp.png"
        plot_scatter(telemetry_df, "severity_primary", "dP_kPa", "fouling_stage_name", f"{cfg.severity_col} vs dP_kPa", p, cfg.sample_rows_scatter, cfg.random_seed)
        key_plots.append(p)

    corr_path = outdir / "plots" / "correlation_heatmap.png"
    corr_df = plot_corr_heatmap(telemetry_df, DEFAULT_CORR_COLS, corr_path, "Correlaciones entre variables clave")
    if corr_df is not None:
        save_df(corr_df.reset_index().rename(columns={"index": "column"}), outdir / "tables" / "correlation_matrix.csv")
        key_plots.append(corr_path)

    p = outdir / "plots" / "normalized_cycle_severity.png"
    plot_normalized_cycle_severity(telemetry_df, cycle_summary, p, "severity_primary")
    key_plots.append(p)

    sample_cycle_table = choose_sample_cycles(cycle_summary, cfg.sample_cycles, cfg.random_seed)
    save_df(sample_cycle_table, outdir / "tables" / "sample_cycles_catalog.csv")
    sample_plots: List[Path] = []
    for _, row in sample_cycle_table.iterrows():
        seq_id = str(row["sequence_id"])
        stem = seq_id.replace("::", "__").replace("/", "_")
        plot_path = outdir / "samples" / f"{stem}.png"
        csv_path = outdir / "samples" / f"{stem}.csv"
        plot_sample_cycle(telemetry_df, maintenance_df, seq_id, plot_path, csv_path, "severity_primary")
        sample_plots.append(plot_path)

    severe_sequence_issues = sequence_qc.loc[
        (sequence_qc["duplicate_timestamps"] > 0)
        | (sequence_qc["non_monotonic_diffs"] > 0)
        | (sequence_qc["irregular_dt_rows"] > 0)
        | (sequence_qc["unique_asset_in_seq"] > 1)
        | (sequence_qc["unique_cycle_in_seq"] > 1)
    ]
    qc_summary: Dict[str, Any] = {
        "sequences_total": int(sequence_qc.shape[0]),
        "sequence_issue_rows": int(len(severe_sequence_issues)),
        "stage_threshold_mismatch_pct": float(target_consistency.loc[target_consistency["check_name"] == "fouling_stage_vs_physical_thresholds", "mismatch_pct"].iloc[0]) if (len(target_consistency) and (target_consistency["check_name"] == "fouling_stage_vs_physical_thresholds").any()) else np.nan,
        "max_target_mismatch_pct": float(target_consistency["mismatch_pct"].max()) if len(target_consistency) else np.nan,
        "naive_cross_cycle_window_pct_if_group_by_asset": float(naive_window_qc["naive_cross_cycle_window_pct"]),
        "sequence_aware_windows_total": int(window_qc["sequence_aware_windows_total"]),
        "bad_windows_crossing_cycle_boundary": int(window_qc["bad_windows_crossing_cycle_boundary"]),
        "sequences_shorter_than_seq_len": int(window_qc["sequences_shorter_than_seq_len"]),
        "median_inter_cycle_gap_h": float(inter_cycle_gaps["inter_cycle_gap_h"].median()) if len(inter_cycle_gaps) else np.nan,
        "median_negative_rf_steps_in_production_per_cycle": float(sequence_qc["negative_rf_steps_in_production"].dropna().median()) if sequence_qc["negative_rf_steps_in_production"].notna().any() else np.nan,
        "median_negative_mass_steps_in_production_per_cycle": float(sequence_qc["negative_mass_steps_in_production"].dropna().median()) if sequence_qc["negative_mass_steps_in_production"].notna().any() else np.nan,
    }
    save_json(outdir / "summary.json", {"overview": overview, "qc_summary": qc_summary})
    save_json(outdir / "qc" / "qc_summary.json", qc_summary)

    build_markdown_report(
        outdir=outdir,
        overview=overview,
        qc_summary=qc_summary,
        top_cycle_df=cycle_summary.sort_values("max_severity", ascending=False).head(20),
        target_consistency_df=target_consistency,
        key_plot_paths=key_plots[:20],
        sample_plot_paths=sample_plots,
    )
    build_html_report(
        outdir=outdir,
        overview=overview,
        qc_summary=qc_summary,
        asset_summary=asset_summary,
        cycle_summary=cycle_summary,
        target_consistency_df=target_consistency,
        key_plot_paths=key_plots[:12],
        sample_plot_paths=sample_plots,
    )

    manifest = build_manifest(outdir)
    save_json(outdir / "manifest.json", manifest)
    return {
        "outdir": to_repo_relative_path(outdir),
        "manifest": to_repo_relative_path(outdir / "manifest.json"),
        "summary": to_repo_relative_path(outdir / "summary.json"),
        "report_html": to_repo_relative_path(outdir / "report.html"),
        "report_md": to_repo_relative_path(outdir / "report.md"),
    }
