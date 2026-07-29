from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.training.common import FeatureArtifacts, TrainConfig
from src.training.data import load_telemetry
from src.training.datasets import WindowDataset, collate_window_batch, make_sequences
from src.training.evaluation import add_explanations, consolidate_alerts, default_policy
from src.training.features import build_feature_matrix, engineer_row_features
from src.training.model import PredictiveTCN
from src.utils.common import save_json
from src.utils.logging import get_logger
from src.utils.paths import resolve_saved_path, to_repo_relative_path

LOGGER = get_logger(__name__)


@dataclass
class InferenceConfig:
    telemetry_input: str
    artifacts_dir: str
    metrics_dir: str
    predictions_dir: str
    scenario: str = "auto"
    batch_size: int = 256
    device: str = "cpu"
    output_name: str | None = None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_train_config(path: Path, device: str) -> TrainConfig:
    data = _load_json(path)
    data["device"] = device
    return TrainConfig(**data)


def _load_feature_artifacts(path: Path) -> FeatureArtifacts:
    data = _load_json(path)
    return FeatureArtifacts(**data)


def load_model(cfg: InferenceConfig) -> tuple[PredictiveTCN, TrainConfig, FeatureArtifacts, dict[str, Any], list[str], dict[str, Any]]:
    artifacts_dir = resolve_saved_path(cfg.artifacts_dir)
    metrics_dir = resolve_saved_path(cfg.metrics_dir)
    manifest = _load_json(artifacts_dir / "model_manifest.json")
    scenario = manifest["selected_scenario"] if cfg.scenario in {"auto", "", None} else cfg.scenario
    train_cfg = _load_train_config(resolve_saved_path(manifest["training_config"]), cfg.device)
    feature_artifacts = _load_feature_artifacts(resolve_saved_path(manifest["feature_artifacts"]))
    feature_names = feature_artifacts.full_feature_names if scenario == "full" else feature_artifacts.no_clock_feature_names

    model = PredictiveTCN(
        n_features=len(feature_names),
        channels=int(train_cfg.channels),
        dilations=tuple(train_cfg.dilations),
        dropout=float(train_cfg.dropout),
    ).to(cfg.device)
    checkpoint = artifacts_dir / "selected_model.pt"
    if scenario != manifest["selected_scenario"]:
        candidate = artifacts_dir / "scenarios" / scenario / "best_model.pt"
        checkpoint = candidate if candidate.exists() else checkpoint
    state = torch.load(checkpoint, map_location=cfg.device)
    model.load_state_dict(state)
    model.eval()

    policy_path = metrics_dir / f"{scenario}_policy_thresholds.json"
    policy = _load_json(policy_path) if policy_path.exists() else default_policy(train_cfg)
    manifest["resolved_scenario"] = scenario
    manifest["checkpoint_used"] = to_repo_relative_path(checkpoint)
    return model, train_cfg, feature_artifacts, manifest, list(feature_names), policy


def _prepare_sequences(
    telemetry_input: str,
    train_cfg: TrainConfig,
    feature_artifacts: FeatureArtifacts,
    feature_names: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], list[int], list[str]]:
    telemetry_df = load_telemetry(str(resolve_saved_path(telemetry_input)), train_cfg, require_targets=False)
    telemetry_df = engineer_row_features(telemetry_df)
    telemetry_df, full_feature_names, _ = build_feature_matrix(telemetry_df, feature_artifacts, [], train_cfg)
    sequences = make_sequences(telemetry_df, train_cfg, full_feature_names)
    feature_to_idx = {name: i for i, name in enumerate(full_feature_names)}
    feature_indices = [feature_to_idx[name] for name in feature_names]
    asset_ids = sorted(telemetry_df["asset_id"].astype(str).unique().tolist())
    return telemetry_df, sequences, feature_indices, asset_ids


def run_inference(
    model: PredictiveTCN,
    sequences: dict[str, Any],
    feature_indices: list[int],
    asset_ids: list[str],
    train_cfg: TrainConfig,
    batch_size: int,
) -> pd.DataFrame:
    dataset = WindowDataset(sequences, asset_ids, feature_indices, train_cfg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_window_batch)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(train_cfg.device)
            pred = model(x)
            p_stage = torch.softmax(pred["stage_logits"], dim=-1).cpu().numpy()
            p_foul_h = torch.sigmoid(pred["foul_h_logit"]).cpu().numpy()
            p_actionable_foul_h = torch.sigmoid(pred["actionable_foul_h_logit"]).cpu().numpy()
            p_clog_h = torch.sigmoid(pred["clog_h_logit"]).cpu().numpy()
            pred_sev = (pred["severity_scaled"].cpu().numpy() / train_cfg.severity_scale()).clip(min=0.0)
            pred_ttf_foul = np.expm1(np.clip(pred["tte_foul_log"].cpu().numpy(), 0.0, np.log1p(train_cfg.tte_fouling_cap_min)))
            pred_ttf_clog = np.expm1(np.clip(pred["tte_clog_log"].cpu().numpy(), 0.0, np.log1p(train_cfg.tte_clog_cap_min)))
            pred_ttu = np.expm1(np.clip(pred["ttu_log"].cpu().numpy(), 0.0, np.log1p(train_cfg.ttu_cap_min)))

            for i, sequence_id in enumerate(batch["sequence_id"]):
                end_idx = int(batch["end_idx"][i])
                seq = sequences[sequence_id]
                meta = seq.meta.iloc[end_idx]
                pred_stage = int(np.argmax(p_stage[i]))
                rows.append(
                    {
                        "asset_id": seq.asset_id,
                        "sequence_id": sequence_id,
                        "timestamp": meta["timestamp"],
                        "phase": meta["phase"],
                        "maintenance_active": int(meta["maintenance_active"]) if pd.notna(meta["maintenance_active"]) else 0,
                        "pred_severity": float(pred_sev[i]),
                        "pred_stage": pred_stage,
                        "pred_stage_name": ["stable", "incipient", "advanced"][pred_stage],
                        "p_stage0": float(p_stage[i, 0]),
                        "p_stage1": float(p_stage[i, 1]),
                        "p_stage2": float(p_stage[i, 2]),
                        "p_foul_h": float(p_foul_h[i]),
                        "p_watch_fouling": float(p_foul_h[i]),
                        "p_actionable_foul_h": float(p_actionable_foul_h[i]),
                        "p_actionable_fouling": float(p_actionable_foul_h[i]),
                        "p_clog_h": float(p_clog_h[i]),
                        "pred_tte_foul_min": float(np.clip(pred_ttf_foul[i], 0.0, train_cfg.tte_fouling_cap_min)),
                        "pred_tte_clog_min": float(np.clip(pred_ttf_clog[i], 0.0, train_cfg.tte_clog_cap_min)),
                        "pred_ttu_min": float(np.clip(pred_ttu[i], 0.0, train_cfg.ttu_cap_min)),
                        "flow_kg_s": float(meta["flow_kg_s"]) if pd.notna(meta["flow_kg_s"]) else np.nan,
                        "dP_kPa": float(meta["dP_kPa"]) if pd.notna(meta["dP_kPa"]) else np.nan,
                        "vibration_mm_s": float(meta["vibration_mm_s"]) if pd.notna(meta["vibration_mm_s"]) else np.nan,
                        "thermal_eff_proxy": float(meta["thermal_eff_proxy"]) if pd.notna(meta["thermal_eff_proxy"]) else np.nan,
                        "heat_proxy": float(meta["heat_proxy"]) if pd.notna(meta["heat_proxy"]) else np.nan,
                        "resid_flow_kg_s": float(meta["resid_flow_kg_s"]) if pd.notna(meta["resid_flow_kg_s"]) else np.nan,
                        "resid_dP_kPa": float(meta["resid_dP_kPa"]) if pd.notna(meta["resid_dP_kPa"]) else np.nan,
                        "resid_vibration_mm_s": float(meta["resid_vibration_mm_s"]) if pd.notna(meta["resid_vibration_mm_s"]) else np.nan,
                        "resid_thermal_eff_proxy": float(meta["resid_thermal_eff_proxy"]) if pd.notna(meta["resid_thermal_eff_proxy"]) else np.nan,
                        "dP_kPa_slope15": float(meta["dP_kPa_slope15"]) if pd.notna(meta["dP_kPa_slope15"]) else np.nan,
                        "vibration_mm_s_slope15": float(meta["vibration_mm_s_slope15"]) if pd.notna(meta["vibration_mm_s_slope15"]) else np.nan,
                        "milk_type": str(meta["milk_type"]),
                        "asset_family": str(meta["asset_family"]),
                        "fault_type": str(meta["fault_type"]),
                        "ttm_to_planned_cip_min": float(meta["ttm_to_planned_cip_min"]) if pd.notna(meta["ttm_to_planned_cip_min"]) else np.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(["asset_id", "timestamp", "sequence_id"]).reset_index(drop=True)


def save_predictions(df_pred: pd.DataFrame, out_path: str | Path) -> None:
    path = resolve_saved_path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(path, index=False)


def predict_pipeline(cfg: InferenceConfig) -> dict[str, Any]:
    predictions_dir = resolve_saved_path(cfg.predictions_dir)
    model, train_cfg, feature_artifacts, manifest, feature_names, policy = load_model(cfg)
    train_cfg.device = cfg.device
    _, sequences, feature_indices, asset_ids = _prepare_sequences(
        telemetry_input=cfg.telemetry_input,
        train_cfg=train_cfg,
        feature_artifacts=feature_artifacts,
        feature_names=feature_names,
    )
    predictions = run_inference(
        model=model,
        sequences=sequences,
        feature_indices=feature_indices,
        asset_ids=asset_ids,
        train_cfg=train_cfg,
        batch_size=cfg.batch_size,
    )
    explained = add_explanations(predictions, policy, feature_artifacts.predicate_thresholds, train_cfg)
    alerts = consolidate_alerts(explained, int(policy.get("cooldown_min", train_cfg.cooldown_min_default)))

    stem = Path(cfg.telemetry_input).stem if cfg.output_name is None else Path(cfg.output_name).stem
    predictions_path = predictions_dir / f"{stem}_predictions.csv"
    alerts_path = predictions_dir / f"{stem}_alerts.csv"
    manifest_path = predictions_dir / f"{stem}_inference_manifest.json"
    save_predictions(explained, predictions_path)
    save_predictions(alerts, alerts_path)
    save_json(
        manifest_path,
        {
            "input": cfg.telemetry_input,
            "selected_scenario": manifest["resolved_scenario"],
            "checkpoint_used": manifest["checkpoint_used"],
            "predictions": str(predictions_path),
            "alerts": str(alerts_path),
            "rows_scored": int(len(explained)),
            "alert_rows": int(len(alerts)),
        },
    )
    return {
        "predictions": to_repo_relative_path(predictions_path),
        "alerts": to_repo_relative_path(alerts_path),
        "manifest": to_repo_relative_path(manifest_path),
        "rows_scored": int(len(explained)),
        "alert_rows": int(len(alerts)),
        "selected_scenario": manifest["resolved_scenario"],
    }
