
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils.common import (
    derive_cycle_columns,
    ensure_columns,
    minutes_since_last_event,
    minutes_to_next_event,
    normalize_fault_type,
    previous_event_value,
    sequence_group_columns,
)

from .common import (
    CIP_EVENT_TYPES,
    DEFAULT_NUMERIC_COLS,
    LAST_MAINT_TO_ID,
    TrainConfig,
    derive_stage_from_severity,
)


def load_telemetry(path: str, cfg: TrainConfig, require_targets: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
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

def load_maintenance(path: str, telemetry_df: pd.DataFrame) -> pd.DataFrame:
    p = Path(path) if path else None
    if p and p.exists():
        df = pd.read_csv(path)
        if len(df) == 0:
            return derive_maintenance_from_telemetry(telemetry_df)
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

    # Sequence-local future targets: they must not look beyond the current cycle/sequence.
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

    # Asset-level clocks and last maintenance state can legitimately cross cycles.
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

def build_asset_profiles(telemetry_df: pd.DataFrame, maintenance_df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    foul_col = f"fouling_onset_within_{cfg.fouling_horizon_min}min"
    clog_col = f"clog_onset_within_{cfg.clog_horizon_min}min"
    actionable_col = f"unplanned_fouling_within_{cfg.unplanned_fouling_horizon_min}min"
    maint = maintenance_df.copy()
    if len(maint):
        maint["fault_type_norm"] = maint["fault_type"].map(normalize_fault_type)
    rows: List[Dict[str, Any]] = []
    for asset_id, g in telemetry_df.groupby("asset_id", sort=False):
        prod = g.loc[(g["phase"] == "production") & (g["maintenance_active"].fillna(0).astype(int) == 0)].copy()
        m = maint.loc[(maint["asset_id"] == asset_id) & (maint["planned"].fillna(0).astype(int) == 0)].copy() if len(maint) else pd.DataFrame()
        rows.append(
            {
                "asset_id": asset_id,
                "n_rows": int(len(g)),
                "n_prod_rows": int(len(prod)),
                "n_sequences": int(g["sequence_id"].nunique()) if "sequence_id" in g.columns else 1,
                "watch_foul_pos_rows": int(pd.to_numeric(prod.get(foul_col, 0), errors="coerce").fillna(0).sum()),
                "actionable_foul_pos_rows": int(pd.to_numeric(prod.get(actionable_col, 0), errors="coerce").fillna(0).sum()),
                "clog_pos_rows": int(pd.to_numeric(prod.get(clog_col, 0), errors="coerce").fillna(0).sum()),
                "unplanned_fouling_events": int((m.get("fault_type_norm", pd.Series(dtype=object)) == "fouling").sum()) if len(m) else 0,
                "unplanned_clog_events": int((m.get("fault_type_norm", pd.Series(dtype=object)) == "clogging").sum()) if len(m) else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("asset_id").reset_index(drop=True)

def _score_asset_split(train: List[str], val: List[str], test: List[str], profiles: pd.DataFrame) -> float:
    if not train or not val or not test:
        return 1e18
    metrics = [
        "n_prod_rows",
        "watch_foul_pos_rows",
        "actionable_foul_pos_rows",
        "clog_pos_rows",
        "unplanned_fouling_events",
        "unplanned_clog_events",
    ]
    target = {"train": 0.60, "val": 0.20, "test": 0.20}
    splits = {"train": train, "val": val, "test": test}
    total = {m: float(profiles[m].sum()) for m in metrics}
    score = 0.0
    for split_name, assets in splits.items():
        sub = profiles.loc[profiles["asset_id"].isin(assets)]
        for m in metrics:
            if total[m] <= 0:
                continue
            ratio = float(sub[m].sum()) / total[m]
            weight = 1.0
            if "events" in m:
                weight = 4.0
            elif "actionable" in m:
                weight = 2.5
            elif "watch" in m or "clog" in m:
                weight = 1.5
            score += weight * abs(ratio - target[split_name])
    # Strongly penalize validation/test with no event coverage when such events exist.
    for metric in ["unplanned_fouling_events", "unplanned_clog_events"]:
        total_metric = int(profiles[metric].sum())
        if total_metric >= 2:
            if int(profiles.loc[profiles["asset_id"].isin(val), metric].sum()) == 0:
                score += 50.0
            if int(profiles.loc[profiles["asset_id"].isin(test), metric].sum()) == 0:
                score += 50.0
        elif total_metric >= 1:
            if int(profiles.loc[profiles["asset_id"].isin(test), metric].sum()) == 0:
                score += 20.0
    return float(score)

def split_assets(asset_ids: Sequence[str], telemetry_df: pd.DataFrame, maintenance_df: pd.DataFrame, cfg: TrainConfig, seed: int) -> Tuple[List[str], List[str], List[str], Dict[str, Any], pd.DataFrame]:
    asset_ids = sorted(list(asset_ids))
    n = len(asset_ids)
    if n == 0:
        raise ValueError("No assets found.")
    if n == 1:
        report = {"strategy": "degenerate", "train": asset_ids, "val": asset_ids, "test": asset_ids}
        profiles = build_asset_profiles(telemetry_df, maintenance_df, cfg)
        return asset_ids[:], asset_ids[:], asset_ids[:], report, profiles
    if n == 2:
        profiles = build_asset_profiles(telemetry_df, maintenance_df, cfg)
        train = [asset_ids[0]]
        val = [asset_ids[0]]
        test = [asset_ids[1]]
        report = {
            "strategy": "degenerate_two_assets",
            "objective": 0.0,
            "train_assets": train,
            "val_assets": val,
            "test_assets": test,
            "overall": profiles.drop(columns=["asset_id"]).sum(numeric_only=True).to_dict(),
            "train": profiles.loc[profiles["asset_id"].isin(train)].drop(columns=["asset_id"]).sum(numeric_only=True).to_dict(),
            "val": profiles.loc[profiles["asset_id"].isin(val)].drop(columns=["asset_id"]).sum(numeric_only=True).to_dict(),
            "test": profiles.loc[profiles["asset_id"].isin(test)].drop(columns=["asset_id"]).sum(numeric_only=True).to_dict(),
        }
        return train, val, test, report, profiles
    profiles = build_asset_profiles(telemetry_df, maintenance_df, cfg)
    rng = np.random.default_rng(seed)
    n_train = max(1, int(round(n * 0.6)))
    n_val = max(1, int(round(n * 0.2)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = max(1, n - n_train - n_val)
    if n_train + n_val + n_test > n:
        n_train = max(1, n - n_val - n_test)

    best_score = 1e18
    best_split = None
    trials = max(int(cfg.split_trials), 50)
    for _ in range(trials):
        perm = asset_ids[:]
        rng.shuffle(perm)
        train = perm[:n_train]
        val = perm[n_train:n_train + n_val]
        test = perm[n_train + n_val:]
        score = _score_asset_split(train, val, test, profiles)
        if score < best_score:
            best_score = score
            best_split = (train[:], val[:], test[:])
    assert best_split is not None
    train, val, test = best_split
    report: Dict[str, Any] = {
        "strategy": "asset_random_search_stratified",
        "objective": float(best_score),
        "trials": trials,
        "train_assets": train,
        "val_assets": val,
        "test_assets": test,
        "overall": profiles.drop(columns=["asset_id"]).sum(numeric_only=True).to_dict(),
        "train": profiles.loc[profiles["asset_id"].isin(train)].drop(columns=["asset_id"]).sum(numeric_only=True).to_dict(),
        "val": profiles.loc[profiles["asset_id"].isin(val)].drop(columns=["asset_id"]).sum(numeric_only=True).to_dict(),
        "test": profiles.loc[profiles["asset_id"].isin(test)].drop(columns=["asset_id"]).sum(numeric_only=True).to_dict(),
    }
    return train, val, test, report, profiles
