from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.utils.common import save_json
from src.utils.logging import get_logger
from src.utils.paths import relativize_payload, resolve_saved_path, to_repo_relative_path

from .synthetic_generator import CleaningParams, EquipConsts, GeneratorConfig, generate_asset
from .targets import annotate_future_targets

LOGGER = get_logger(__name__)


def build_generator_config(config: Mapping[str, Any]) -> GeneratorConfig:
    return GeneratorConfig(**config)


def generate_synthetic_dataset(cfg: GeneratorConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    equip = EquipConsts()
    cleaning = CleaningParams()
    start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    tel_parts: list[pd.DataFrame] = []
    maint_parts: list[pd.DataFrame] = []

    for asset_idx in range(cfg.assets):
        tel_df, maint_df = generate_asset(
            asset_idx=asset_idx,
            cfg=cfg,
            equip=equip,
            cleaning=cleaning,
            start=start,
        )
        tel_parts.append(tel_df)
        if len(maint_df):
            maint_parts.append(maint_df)

    telemetry_df = pd.concat(tel_parts, axis=0, ignore_index=True)
    maintenance_df = pd.concat(maint_parts, axis=0, ignore_index=True) if maint_parts else pd.DataFrame(
        columns=[
            "maintenance_id",
            "asset_id",
            "cycle_id",
            "cycle_index",
            "start_time",
            "end_time",
            "duration_min",
            "planned",
            "fault_type",
            "maintenance_type",
            "corrective_action",
            "severity_rf_at_start",
            "notes",
        ]
    )

    telemetry_df = annotate_future_targets(telemetry_df, maintenance_df, cfg)
    telemetry_df["timestamp"] = pd.to_datetime(telemetry_df["timestamp"], utc=True)
    if len(maintenance_df):
        maintenance_df["start_time"] = pd.to_datetime(maintenance_df["start_time"], utc=True, errors="coerce")
        maintenance_df["end_time"] = pd.to_datetime(maintenance_df["end_time"], utc=True, errors="coerce")

    telemetry_df = telemetry_df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    maintenance_df = maintenance_df.sort_values(["asset_id", "start_time"]).reset_index(drop=True) if len(maintenance_df) else maintenance_df

    meta = {
        "generator": "cu07_predictive_synthetic_v2_cycle_centric",
        "seed": cfg.seed,
        "assets": cfg.assets,
        "cycles_per_asset": cfg.cycles_per_asset,
        "dt_s": cfg.dt,
        "fouling_anchor_rf": cfg.fouling_anchor_rf,
        "rf_stage_thr_incipient": cfg.resolved_stage_thresholds()[0],
        "rf_stage_thr_advanced": cfg.resolved_stage_thresholds()[1],
        "fouling_horizon_min": cfg.fouling_horizon_min,
        "clog_horizon_min": cfg.clog_horizon_min,
        "unplanned_horizon_min": cfg.unplanned_horizon_min,
        "stable_hours_min": cfg.stable_hours_min,
        "stable_hours_max": cfg.stable_hours_max,
        "incipient_hours_min": cfg.incipient_hours_min,
        "incipient_hours_max": cfg.incipient_hours_max,
        "advanced_hours_min": cfg.advanced_hours_min,
        "advanced_hours_max": cfg.advanced_hours_max,
        "idle_hours_min": cfg.idle_hours_min,
        "idle_hours_max": cfg.idle_hours_max,
        "inter_cycle_gap_min": cfg.inter_cycle_gap_min,
        "inter_cycle_gap_max": cfg.inter_cycle_gap_max,
        "emit_idle_phase": cfg.emit_idle_phase,
        "telemetry_rows": int(len(telemetry_df)),
        "maintenance_rows": int(len(maintenance_df)),
        "columns_telemetry": telemetry_df.columns.tolist(),
        "columns_maintenance": maintenance_df.columns.tolist(),
    }
    return telemetry_df, maintenance_df, meta


def persist_generated_data(
    telemetry_df: pd.DataFrame,
    maintenance_df: pd.DataFrame,
    meta: Mapping[str, Any],
    cfg: GeneratorConfig,
) -> dict[str, str]:
    out_tel = resolve_saved_path(cfg.out_telemetry)
    out_maint = resolve_saved_path(cfg.out_maintenance)
    out_meta = resolve_saved_path(cfg.out_meta)
    out_tel.parent.mkdir(parents=True, exist_ok=True)
    out_maint.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    telemetry_df.assign(timestamp=telemetry_df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")).to_csv(out_tel, index=False)
    if len(maintenance_df):
        maintenance_df.assign(
            start_time=maintenance_df["start_time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            end_time=maintenance_df["end_time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
        ).to_csv(out_maint, index=False)
    else:
        maintenance_df.to_csv(out_maint, index=False)

    save_json(out_meta, meta)
    LOGGER.info("Generated telemetry rows=%s maintenance rows=%s", len(telemetry_df), len(maintenance_df))
    return {
        "telemetry": to_repo_relative_path(out_tel),
        "maintenance": to_repo_relative_path(out_maint),
        "metadata": to_repo_relative_path(out_meta),
    }


def data_processing_pipeline(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = build_generator_config(config)
    telemetry_df, maintenance_df, meta = generate_synthetic_dataset(cfg)
    outputs = persist_generated_data(telemetry_df, maintenance_df, meta, cfg)
    return {
        "config": relativize_payload(asdict(cfg)),
        "outputs": outputs,
        "telemetry_rows": int(len(telemetry_df)),
        "maintenance_rows": int(len(maintenance_df)),
    }
