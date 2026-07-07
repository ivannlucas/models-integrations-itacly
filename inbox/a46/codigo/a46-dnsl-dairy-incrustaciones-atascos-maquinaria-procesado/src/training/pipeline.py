from __future__ import annotations

import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.common import nan_to_zero, safe_float, save_json, set_seed
from src.utils.logging import get_logger
from src.utils.paths import resolve_saved_path, to_repo_relative_path

from .common import FeatureArtifacts, TrainConfig
from .data import align_future_labels, load_maintenance, load_telemetry, split_assets
from .datasets import collate_window_batch, make_sequences, WindowDataset
from .evaluation import (
    build_event_table,
    calibrate_policy,
    default_policy,
    evaluate_policy,
    predict_loader,
    save_confusion_matrices,
    stage_score_from_severity,
    window_metrics_from_preds,
)
from .features import build_feature_matrix, engineer_row_features, fit_feature_artifacts
from .model import PredictiveTCN
from .persistence import (
    build_model_manifest,
    copy_if_exists,
    export_scenario_outputs,
    export_split_assets,
    export_split_rows,
    save_feature_artifacts,
)

LOGGER = get_logger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    stage_weights: torch.Tensor,
    foul_pos_weight: torch.Tensor,
    actionable_foul_pos_weight: torch.Tensor,
    clog_pos_weight: torch.Tensor,
) -> Dict[str, float]:
    model.train()
    losses: List[float] = []
    for batch in loader:
        x = batch["x"].to(cfg.device)
        y_severity = batch["y_severity_scaled"].to(cfg.device)
        y_stage = batch["y_stage"].to(cfg.device)
        y_foul_h = batch["y_foul_h"].to(cfg.device)
        y_actionable_foul_h = batch["y_actionable_foul_h"].to(cfg.device)
        y_clog_h = batch["y_clog_h"].to(cfg.device)
        y_ttf_foul = batch["y_ttf_foul_log"].to(cfg.device)
        y_ttf_clog = batch["y_ttf_clog_log"].to(cfg.device)
        y_ttu = batch["y_ttu_log"].to(cfg.device)
        foul_weight = batch["foul_weight"].to(cfg.device)
        actionable_foul_weight = batch["actionable_foul_weight"].to(cfg.device)
        clog_weight = batch["clog_weight"].to(cfg.device)

        pred = model(x)
        loss_sev = F.smooth_l1_loss(pred["severity_scaled"], y_severity)
        loss_stage_raw = F.cross_entropy(pred["stage_logits"], y_stage, weight=stage_weights, reduction="none")
        loss_stage = (loss_stage_raw * (1.0 + 0.15 * (foul_weight - 1.0))).mean()

        loss_foul_raw = F.binary_cross_entropy_with_logits(pred["foul_h_logit"], y_foul_h, pos_weight=foul_pos_weight, reduction="none")
        loss_foul = (loss_foul_raw * foul_weight).mean()

        loss_actionable_foul_raw = F.binary_cross_entropy_with_logits(
            pred["actionable_foul_h_logit"], y_actionable_foul_h, pos_weight=actionable_foul_pos_weight, reduction="none"
        )
        loss_actionable_foul = (loss_actionable_foul_raw * actionable_foul_weight).mean()

        loss_clog_raw = F.binary_cross_entropy_with_logits(pred["clog_h_logit"], y_clog_h, pos_weight=clog_pos_weight, reduction="none")
        loss_clog = (loss_clog_raw * clog_weight).mean()

        loss_ttf_foul = F.smooth_l1_loss(pred["tte_foul_log"], y_ttf_foul)
        loss_ttf_clog = F.smooth_l1_loss(pred["tte_clog_log"], y_ttf_clog)
        loss_ttu = F.smooth_l1_loss(pred["ttu_log"], y_ttu)

        stage_probs = torch.softmax(pred["stage_logits"], dim=-1)
        stage_score = 0.5 * stage_probs[:, 1] + 1.0 * stage_probs[:, 2]
        sev_score = stage_score_from_severity(pred["severity_scaled"], cfg)
        loss_cons = F.mse_loss(sev_score, stage_score.detach())

        loss = (
            1.05 * loss_sev
            + 0.95 * loss_stage
            + 0.60 * loss_foul
            + 0.45 * loss_actionable_foul
            + 0.80 * loss_clog
            + 0.40 * loss_ttf_foul
            + 0.45 * loss_ttf_clog
            + 0.45 * loss_ttu
            + 0.10 * loss_cons
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return {"train_loss": float(np.mean(losses)) if losses else float("nan")}

def window_objective(metrics: Mapping[str, float], cfg: TrainConfig) -> float:
    _, thr_advanced = cfg.resolved_stage_thresholds()
    sev_norm = safe_float(metrics.get("severity_rmse"), default=thr_advanced) / max(thr_advanced, 1e-9)
    ttu_norm = safe_float(metrics.get("ttu_mae_min"), default=cfg.ttu_cap_min) / max(cfg.ttu_cap_min, 1.0)
    return (
        1.35 * nan_to_zero(metrics.get("stage_macro_f1", 0.0))
        + 0.75 * nan_to_zero(metrics.get("watch_foul_ap", 0.0))
        + 0.45 * nan_to_zero(metrics.get("actionable_foul_ap", 0.0))
        + 0.55 * nan_to_zero(metrics.get("clog_h_ap", 0.0))
        - 0.22 * sev_norm
        - 0.08 * ttu_norm
    )

def run_scenario(
    scenario_name: str,
    feature_names_used: Sequence[str],
    sequences: Dict[str, AssetSequence],
    all_feature_names: Sequence[str],
    train_assets: Sequence[str],
    val_assets: Sequence[str],
    test_assets: Sequence[str],
    telemetry_df: pd.DataFrame,
    maintenance_df: pd.DataFrame,
    artifacts: FeatureArtifacts,
    cfg: TrainConfig,
    outdir: Path,
) -> Dict[str, Any]:
    scenario_dir = outdir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] scenario={scenario_name} | features={len(feature_names_used)}")

    feature_to_idx = {name: i for i, name in enumerate(all_feature_names)}
    feature_indices = [feature_to_idx[name] for name in feature_names_used]
    train_ds = WindowDataset(sequences, train_assets, feature_indices, cfg)
    val_ds = WindowDataset(sequences, val_assets, feature_indices, cfg)
    test_ds = WindowDataset(sequences, test_assets, feature_indices, cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_window_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_window_batch)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_window_batch)

    model = PredictiveTCN(n_features=len(feature_indices), channels=cfg.channels, dilations=cfg.dilations, dropout=cfg.dropout).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    stage_weights = torch.tensor(artifacts.stage_class_weights, dtype=torch.float32, device=cfg.device)
    foul_pos_weight = torch.tensor(artifacts.foul_pos_weight, dtype=torch.float32, device=cfg.device)
    actionable_foul_pos_weight = torch.tensor(artifacts.actionable_foul_pos_weight, dtype=torch.float32, device=cfg.device)
    clog_pos_weight = torch.tensor(artifacts.clog_pos_weight, dtype=torch.float32, device=cfg.device)

    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, Any]] = None
    best_policy: Dict[str, float] = default_policy(cfg)
    best_val_pred: Optional[pd.DataFrame] = None
    best_val_win_metrics: Optional[Dict[str, Any]] = None
    best_score = -1e18

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            cfg,
            stage_weights,
            foul_pos_weight,
            actionable_foul_pos_weight,
            clog_pos_weight,
        )
        val_pred_df = predict_loader(model, val_loader, sequences, cfg)
        val_win_metrics = window_metrics_from_preds(val_pred_df, cfg) if len(val_pred_df) else {}
        score = window_objective(val_win_metrics, cfg) if val_win_metrics else -1e9

        rec: Dict[str, Any] = {
            "epoch": epoch,
            **train_stats,
            **{f"val_{k}": v for k, v in val_win_metrics.items()},
            "val_objective": score,
        }
        history.append(rec)
        print(
            f"[epoch {epoch:02d}] scenario={scenario_name} train_loss={train_stats.get('train_loss', float('nan')):.4f} "
            f"val_obj={score:.4f} stage_f1={val_win_metrics.get('stage_macro_f1', float('nan')):.4f} "
            f"watch_ap={val_win_metrics.get('watch_foul_ap', float('nan')):.4f} "
            f"actionable_ap={val_win_metrics.get('actionable_foul_ap', float('nan')):.4f} "
            f"clog_ap={val_win_metrics.get('clog_h_ap', float('nan')):.4f}"
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_val_pred = val_pred_df.copy()
            best_val_win_metrics = dict(val_win_metrics)

    if best_state is None:
        raise RuntimeError(f"Training failed for scenario '{scenario_name}'.")

    model.load_state_dict(best_state)
    torch.save(best_state, scenario_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(scenario_dir / "training_history.csv", index=False)

    if best_val_pred is None:
        best_val_pred = predict_loader(model, val_loader, sequences, cfg)
    if len(best_val_pred):
        best_val_pred.to_csv(scenario_dir / "val_window_predictions.csv", index=False)

    val_events = build_event_table(maintenance_df, val_assets)
    if len(best_val_pred) > 0:
        best_policy = calibrate_policy(best_val_pred, val_events, telemetry_df, val_assets, artifacts.predicate_thresholds, cfg)
        if len(val_events) > 0:
            best_val_event_metrics, val_alerts, matched_val_alerts, matched_val_events = evaluate_policy(
                best_val_pred, val_events, telemetry_df, val_assets, best_policy, artifacts.predicate_thresholds, cfg
            )
            val_alerts = val_alerts.merge(
                matched_val_alerts[["asset_id", "timestamp", "matched_event_id", "lead_min"]],
                on=["asset_id", "timestamp"],
                how="left",
                suffixes=("", "_matched"),
            )
            val_alerts.to_csv(scenario_dir / "val_alerts.csv", index=False)
            matched_val_events.to_csv(scenario_dir / "val_events_matched.csv", index=False)
        else:
            best_val_event_metrics = {}
            val_alerts = pd.DataFrame()
        save_confusion_matrices(best_val_pred, "val", scenario_dir, best_policy, cfg)
    else:
        best_val_event_metrics = {}
        val_alerts = pd.DataFrame()

    test_pred_df = predict_loader(model, test_loader, sequences, cfg)
    test_pred_df.to_csv(scenario_dir / "test_window_predictions.csv", index=False)
    test_win_metrics = window_metrics_from_preds(test_pred_df, cfg) if len(test_pred_df) else {}

    test_events = build_event_table(maintenance_df, test_assets)
    if len(test_events) > 0 and len(test_pred_df) > 0:
        test_event_metrics, test_alerts, matched_alerts, matched_events = evaluate_policy(
            test_pred_df, test_events, telemetry_df, test_assets, best_policy, artifacts.predicate_thresholds, cfg
        )
        test_alerts = test_alerts.merge(
            matched_alerts[["asset_id", "timestamp", "matched_event_id", "lead_min"]],
            on=["asset_id", "timestamp"],
            how="left",
            suffixes=("", "_matched"),
        )
        test_alerts.to_csv(scenario_dir / "test_alerts.csv", index=False)
        matched_events.to_csv(scenario_dir / "test_events_matched.csv", index=False)
    else:
        test_event_metrics = {}
        test_alerts = pd.DataFrame()
        matched_events = pd.DataFrame()
    if len(test_pred_df) > 0:
        save_confusion_matrices(test_pred_df, "test", scenario_dir, best_policy, cfg)

    save_json(scenario_dir / "test_window_metrics.json", test_win_metrics)
    if best_val_win_metrics is not None:
        save_json(scenario_dir / "val_window_metrics_best.json", best_val_win_metrics)
    if best_val_event_metrics is not None:
        save_json(scenario_dir / "val_event_metrics_best.json", best_val_event_metrics)
    save_json(scenario_dir / "test_event_metrics.json", test_event_metrics)
    save_json(scenario_dir / "policy_thresholds.json", best_policy)
    save_json(
        scenario_dir / "feature_report.json",
        {
            "scenario": scenario_name,
            "n_features": len(feature_names_used),
            "feature_names": list(feature_names_used),
            "receptive_field_steps": model.receptive_field(),
            "receptive_field_minutes": model.receptive_field() * cfg.dt / 60.0,
            "n_train_windows": len(train_ds),
            "n_val_windows": len(val_ds),
            "n_test_windows": len(test_ds),
            "severity_col": cfg.severity_col,
            "stage_thresholds": {
                "incipient": cfg.resolved_stage_thresholds()[0],
                "advanced": cfg.resolved_stage_thresholds()[1],
            },
            "baseline_strategy": {
                "seen_train_assets": "asset-specific healthy baseline from train only",
                "unseen_assets": "early prefix baseline shrunk toward train-global healthy baseline",
                "prefix_hours": cfg.baseline_prefix_hours,
            },
            "clock_ablation": {
                "ablation_enabled": cfg.ablate_clocks,
                "clock_source_cols": sorted(cfg.clock_feature_names()),
                "clock_features_removed_in_this_scenario": [f for f in all_feature_names if f not in feature_names_used and (f[2:] if f.startswith("z_") else f) in cfg.clock_feature_names()],
            },
            "target_contract": {
                "severity_now": cfg.severity_col,
                "stage_now": "derived from physical severity thresholds",
                f"fouling_onset_within_{cfg.fouling_horizon_min}min": "watch-level physical onset horizon",
                f"unplanned_fouling_within_{cfg.unplanned_fouling_horizon_min}min": "actionable risk horizon for unplanned fouling only",
                f"clog_onset_within_{cfg.clog_horizon_min}min": "binary future label",
                "time_to_fouling_onset_min": "future time-to-onset regression",
                "time_to_clog_onset_min": "future time-to-onset regression",
                "ttm_to_unplanned_event_min": "future time until next unplanned intervention",
            },
            "goal_note": "Model selection is state-first: severity/stage/watch/clog performance define the checkpoint. Actionable fouling is an auxiliary head used mainly for alerting, not as the primary definition of machine state.",
        },
    )

    summary = {
        "scenario": scenario_name,
        "best_val_objective": float(best_score),
        "val_window_metrics_best": best_val_win_metrics or {},
        "test_window_metrics": test_win_metrics,
        "val_event_metrics_best": best_val_event_metrics,
        "test_event_metrics": test_event_metrics,
        "policy": best_policy,
        "n_features": len(feature_names_used),
        "state_primary": True,
        "outputs": {
            "watch_fouling": f"p_foul_h / fouling_onset_within_{cfg.fouling_horizon_min}min",
            "actionable_fouling": f"p_actionable_foul_h / unplanned_fouling_within_{cfg.unplanned_fouling_horizon_min}min",
        },
    }
    save_json(scenario_dir / "summary.json", summary)
    print(f"[ok] scenario={scenario_name} best_val_objective={best_score:.4f}")
    return summary

def _scenario_feature_sets(cfg: TrainConfig, artifacts: FeatureArtifacts) -> list[tuple[str, list[str]]]:
    scenarios = [("full", list(artifacts.full_feature_names))]
    if cfg.ablate_clocks:
        scenarios.append(("no_clock", list(artifacts.no_clock_feature_names)))
    return scenarios


def train_pipeline(cfg: TrainConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    if str(cfg.device).lower() == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    artifacts_dir = resolve_saved_path(cfg.artifacts_dir)
    metrics_dir = resolve_saved_path(cfg.metrics_dir)
    predictions_dir = resolve_saved_path(cfg.predictions_dir)
    splits_dir = resolve_saved_path(cfg.splits_dir)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    telemetry_df = load_telemetry(str(resolve_saved_path(cfg.telemetry)), cfg, require_targets=True)
    maintenance_df = load_maintenance(str(resolve_saved_path(cfg.maintenance)), telemetry_df)
    telemetry_df = align_future_labels(telemetry_df, maintenance_df, cfg)
    telemetry_df = engineer_row_features(telemetry_df)

    asset_ids = sorted(telemetry_df["asset_id"].unique().tolist())
    train_assets, val_assets, test_assets, split_report, asset_profiles = split_assets(asset_ids, telemetry_df, maintenance_df, cfg, cfg.seed)
    save_json(splits_dir / "asset_split_report.json", split_report)
    asset_profiles.to_csv(splits_dir / "asset_split_profiles.csv", index=False)
    export_split_assets(train_assets, val_assets, test_assets, splits_dir)
    export_split_rows(telemetry_df, train_assets, val_assets, test_assets, splits_dir)

    train_rows = telemetry_df.loc[telemetry_df["asset_id"].isin(train_assets)].copy()
    feature_artifacts = fit_feature_artifacts(train_rows, cfg)
    telemetry_df, full_feature_names, no_clock_feature_names = build_feature_matrix(telemetry_df, feature_artifacts, train_assets, cfg)
    feature_artifacts.full_feature_names = list(full_feature_names)
    feature_artifacts.no_clock_feature_names = list(no_clock_feature_names)
    sequences = make_sequences(telemetry_df, cfg, full_feature_names)

    scenario_root = artifacts_dir / "scenarios"
    summaries: List[Dict[str, Any]] = []
    export_manifest: Dict[str, Any] = {}
    for scenario_name, feature_names in _scenario_feature_sets(cfg, feature_artifacts):
        summary = run_scenario(
            scenario_name=scenario_name,
            feature_names_used=feature_names,
            sequences=sequences,
            all_feature_names=full_feature_names,
            train_assets=train_assets,
            val_assets=val_assets,
            test_assets=test_assets,
            telemetry_df=telemetry_df,
            maintenance_df=maintenance_df,
            artifacts=feature_artifacts,
            cfg=cfg,
            outdir=scenario_root,
        )
        summaries.append(summary)
        scenario_dir = scenario_root / scenario_name
        export_manifest[scenario_name] = export_scenario_outputs(
            scenario_dir=scenario_dir,
            scenario_name=scenario_name,
            metrics_dir=metrics_dir,
            predictions_dir=predictions_dir,
        )

    ablation_summary: Dict[str, Any] = {
        "scenarios": summaries,
        "train_assets": train_assets,
        "val_assets": val_assets,
        "test_assets": test_assets,
        "severity_col": cfg.severity_col,
        "stage_thresholds": {
            "incipient": cfg.resolved_stage_thresholds()[0],
            "advanced": cfg.resolved_stage_thresholds()[1],
        },
        "split_report_file": to_repo_relative_path(splits_dir / "asset_split_report.json"),
        "note": "The no_clock scenario removes direct elapsed-time clock features to test schedule shortcut learning.",
    }
    if len(summaries) == 2:
        full_metrics = summaries[0]["test_window_metrics"]
        no_clock_metrics = summaries[1]["test_window_metrics"]
        ablation_summary["delta_full_minus_no_clock"] = {
            "stage_macro_f1": nan_to_zero(full_metrics.get("stage_macro_f1", 0.0)) - nan_to_zero(no_clock_metrics.get("stage_macro_f1", 0.0)),
            "watch_foul_ap": nan_to_zero(full_metrics.get("watch_foul_ap", 0.0)) - nan_to_zero(no_clock_metrics.get("watch_foul_ap", 0.0)),
            "actionable_foul_ap": nan_to_zero(full_metrics.get("actionable_foul_ap", 0.0)) - nan_to_zero(no_clock_metrics.get("actionable_foul_ap", 0.0)),
            "clog_h_ap": nan_to_zero(full_metrics.get("clog_h_ap", 0.0)) - nan_to_zero(no_clock_metrics.get("clog_h_ap", 0.0)),
            "severity_rmse": nan_to_zero(full_metrics.get("severity_rmse", 0.0)) - nan_to_zero(no_clock_metrics.get("severity_rmse", 0.0)),
            "ttu_mae_min": nan_to_zero(full_metrics.get("ttu_mae_min", 0.0)) - nan_to_zero(no_clock_metrics.get("ttu_mae_min", 0.0)),
        }
    save_json(metrics_dir / "ablation_summary.json", ablation_summary)

    selected_summary = max(summaries, key=lambda item: float(item.get("best_val_objective", float("-inf"))))
    selected_scenario = str(selected_summary["scenario"])
    selected_checkpoint = scenario_root / selected_scenario / "best_model.pt"
    selected_model = artifacts_dir / "selected_model.pt"
    if selected_checkpoint.exists():
        shutil.copy2(selected_checkpoint, selected_model)

    feature_artifacts_path = artifacts_dir / "feature_artifacts.json"
    config_snapshot_path = artifacts_dir / "training_config.json"
    save_feature_artifacts(feature_artifacts, feature_artifacts_path)
    save_json(config_snapshot_path, asdict(cfg))

    manifest = build_model_manifest(
        selected_scenario=selected_scenario,
        summary=selected_summary,
        checkpoint_path=selected_checkpoint,
        feature_artifacts_path=feature_artifacts_path,
        config_snapshot_path=config_snapshot_path,
        selected_model_path=selected_model,
    )
    manifest.update(
        {
            "full_feature_names": list(full_feature_names),
            "no_clock_feature_names": list(no_clock_feature_names),
            "export_manifest": export_manifest,
        }
    )
    save_json(artifacts_dir / "model_manifest.json", manifest)

    return {
        "selected_scenario": selected_scenario,
        "selected_model": to_repo_relative_path(selected_model),
        "feature_artifacts": to_repo_relative_path(feature_artifacts_path),
        "metrics_dir": to_repo_relative_path(metrics_dir),
        "predictions_dir": to_repo_relative_path(predictions_dir),
        "splits_dir": to_repo_relative_path(splits_dir),
        "summaries": summaries,
    }
