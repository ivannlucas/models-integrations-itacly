"""Vendored (trimmed) from inbox/a46/codigo/.../src/training/evaluation.py.

Kept: window-level metrics, alert-policy application (explanations + consolidation),
default_policy. Dropped vs. the original: calibrate_policy/evaluate_policy/
match_alerts_to_events/event_metrics_from_matches/build_event_table/save_confusion_matrices
(event-level policy recalibration against a fresh val/test asset split) — the plugin serves
and fine-tunes the already-calibrated no_clock policy (models/metrics/no_clock_policy_thresholds.json)
instead of recalibrating it per manifest known_issues.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader

from app.plugins.ml46_dairy_fouling_clog_detection._vendor.common import TrainConfig, safe_binary_auc_ap, stage_to_label
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.datasets import AssetSequence


def window_metrics_from_preds(df: pd.DataFrame, cfg: TrainConfig) -> Dict[str, float]:
    """Compute window-level regression/classification metrics against true_* columns."""
    out: Dict[str, float] = {}
    out["severity_rmse"] = float(np.sqrt(np.mean((df["true_severity"] - df["pred_severity"]) ** 2)))
    out["severity_mae"] = float(mean_absolute_error(df["true_severity"], df["pred_severity"]))
    out["stage_accuracy"] = float((df["pred_stage"] == df["true_stage"]).mean())
    out["stage_macro_f1"] = float(f1_score(df["true_stage"], df["pred_stage"], average="macro"))
    out["stage_balanced_accuracy"] = float(balanced_accuracy_score(df["true_stage"], df["pred_stage"]))

    auc, ap = safe_binary_auc_ap(df["true_foul_h"].to_numpy(), df["p_foul_h"].to_numpy())
    out["watch_foul_auc"] = auc
    out["watch_foul_ap"] = ap
    out["watch_foul_f1"] = float(f1_score(df["true_foul_h"], (df["p_foul_h"] >= 0.5).astype(int), zero_division=0))
    out["foul_h_auc"] = out["watch_foul_auc"]
    out["foul_h_ap"] = out["watch_foul_ap"]
    out["foul_h_f1"] = out["watch_foul_f1"]

    if "true_actionable_foul_h" in df.columns and "p_actionable_foul_h" in df.columns:
        auc, ap = safe_binary_auc_ap(df["true_actionable_foul_h"].to_numpy(), df["p_actionable_foul_h"].to_numpy())
        out["actionable_foul_auc"] = auc
        out["actionable_foul_ap"] = ap
        out["actionable_foul_f1"] = float(f1_score(df["true_actionable_foul_h"], (df["p_actionable_foul_h"] >= 0.5).astype(int), zero_division=0))

    auc, ap = safe_binary_auc_ap(df["true_clog_h"].to_numpy(), df["p_clog_h"].to_numpy())
    out["clog_h_auc"] = auc
    out["clog_h_ap"] = ap
    out["clog_h_f1"] = float(f1_score(df["true_clog_h"], (df["p_clog_h"] >= 0.5).astype(int), zero_division=0))

    out["tte_foul_mae_min"] = float(mean_absolute_error(df["true_tte_foul_min"], df["pred_tte_foul_min"]))
    out["tte_clog_mae_min"] = float(mean_absolute_error(df["true_tte_clog_min"], df["pred_tte_clog_min"]))
    out["ttu_mae_min"] = float(mean_absolute_error(df["true_ttu_min"], df["pred_ttu_min"]))
    return out


def stage_score_from_severity(severity_scaled: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    """Map predicted severity to a continuous 0-1 stage score (used by the coherence loss)."""
    severity = severity_scaled / cfg.severity_scale()
    thr_incipient, thr_advanced = cfg.resolved_stage_thresholds()
    denom = max(thr_advanced - thr_incipient, 1e-9)
    score = torch.clamp((severity - thr_incipient) / denom, min=0.0, max=2.0) / 2.0
    return score


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    sequences: Dict[str, AssetSequence],
    cfg: TrainConfig,
) -> pd.DataFrame:
    """Run the model over every window in *loader* and return one labeled row per window."""
    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(cfg.device)
            pred = model(x)
            p_stage = torch.softmax(pred["stage_logits"], dim=-1).detach().cpu().numpy()
            p_foul_h = torch.sigmoid(pred["foul_h_logit"]).detach().cpu().numpy()
            p_actionable_foul_h = torch.sigmoid(pred["actionable_foul_h_logit"]).detach().cpu().numpy()
            p_clog_h = torch.sigmoid(pred["clog_h_logit"]).detach().cpu().numpy()
            pred_sev = (pred["severity_scaled"].detach().cpu().numpy() / cfg.severity_scale()).clip(min=0.0)
            pred_ttf_foul = np.expm1(np.clip(pred["tte_foul_log"].detach().cpu().numpy(), 0.0, np.log1p(cfg.tte_fouling_cap_min)))
            pred_ttf_clog = np.expm1(np.clip(pred["tte_clog_log"].detach().cpu().numpy(), 0.0, np.log1p(cfg.tte_clog_cap_min)))
            pred_ttu = np.expm1(np.clip(pred["ttu_log"].detach().cpu().numpy(), 0.0, np.log1p(cfg.ttu_cap_min)))

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
                        "maintenance_active": int(meta["maintenance_active"]),
                        "true_severity": float(seq.y_severity[end_idx]),
                        "pred_severity": float(pred_sev[i]),
                        "true_stage": int(seq.y_stage[end_idx]),
                        "true_stage_name": stage_to_label(int(seq.y_stage[end_idx])),
                        "pred_stage": pred_stage,
                        "pred_stage_name": stage_to_label(pred_stage),
                        "p_stage0": float(p_stage[i, 0]),
                        "p_stage1": float(p_stage[i, 1]),
                        "p_stage2": float(p_stage[i, 2]),
                        "true_foul_h": int(seq.y_foul_h[end_idx]),
                        "true_actionable_foul_h": int(seq.y_actionable_foul_h[end_idx]),
                        "true_clog_h": int(seq.y_clog_h[end_idx]),
                        "p_foul_h": float(p_foul_h[i]),
                        "p_watch_fouling": float(p_foul_h[i]),
                        "p_actionable_foul_h": float(p_actionable_foul_h[i]),
                        "p_actionable_fouling": float(p_actionable_foul_h[i]),
                        "p_clog_h": float(p_clog_h[i]),
                        "true_tte_foul_min": float(seq.y_ttf_foul[end_idx]),
                        "true_tte_clog_min": float(seq.y_ttf_clog[end_idx]),
                        "true_ttu_min": float(seq.y_ttu[end_idx]),
                        "pred_tte_foul_min": float(np.clip(pred_ttf_foul[i], 0.0, cfg.tte_fouling_cap_min)),
                        "pred_tte_clog_min": float(np.clip(pred_ttf_clog[i], 0.0, cfg.tte_clog_cap_min)),
                        "pred_ttu_min": float(np.clip(pred_ttu[i], 0.0, cfg.ttu_cap_min)),
                        "flow_kg_s": float(meta["flow_kg_s"]),
                        "dP_kPa": float(meta["dP_kPa"]),
                        "vibration_mm_s": float(meta["vibration_mm_s"]),
                        "thermal_eff_proxy": float(meta["thermal_eff_proxy"]),
                        "heat_proxy": float(meta["heat_proxy"]),
                        "resid_flow_kg_s": float(meta["resid_flow_kg_s"]),
                        "resid_dP_kPa": float(meta["resid_dP_kPa"]),
                        "resid_vibration_mm_s": float(meta["resid_vibration_mm_s"]),
                        "resid_thermal_eff_proxy": float(meta["resid_thermal_eff_proxy"]),
                        "dP_kPa_slope15": float(meta["dP_kPa_slope15"]),
                        "vibration_mm_s_slope15": float(meta["vibration_mm_s_slope15"]),
                        "milk_type": str(meta["milk_type"]),
                        "asset_family": str(meta["asset_family"]),
                        "fault_type": str(meta["fault_type"]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["asset_id", "timestamp", "sequence_id"]).reset_index(drop=True)


def add_explanations(pred_df: pd.DataFrame, policy: Mapping[str, float], predicate_thresholds: Mapping[str, float], cfg: TrainConfig) -> pd.DataFrame:
    """Combine probabilities/severity/predicates into an operator-facing status per window."""
    df = pred_df.copy()

    statuses: List[str] = []
    priorities: List[str] = []
    actions: List[str] = []
    causes: List[str] = []
    predicates_str: List[str] = []
    alert_types: List[str] = []
    actionable: List[int] = []
    watch_flags: List[int] = []
    actionable_foul_flags: List[int] = []

    for _, r in df.iterrows():
        predicates: List[str] = []
        high_dp = r["resid_dP_kPa"] >= predicate_thresholds["high_dp_resid"]
        low_flow = r["resid_flow_kg_s"] <= predicate_thresholds["low_flow_resid"]
        high_vib = r["resid_vibration_mm_s"] >= predicate_thresholds["high_vib_resid"]
        therm_drop = r["resid_thermal_eff_proxy"] <= predicate_thresholds["low_therm_eff_resid"]
        dp_rising = r["dP_kPa_slope15"] >= predicate_thresholds["dp_slope15"]
        vib_spike = r["vibration_mm_s_slope15"] >= predicate_thresholds["vib_slope15"]
        if high_dp:
            predicates.append("HighDP")
        if low_flow:
            predicates.append("LowFlow")
        if high_vib:
            predicates.append("HighVib")
        if therm_drop:
            predicates.append("ThermalDrop")
        if dp_rising:
            predicates.append("DPRising")
        if vib_spike:
            predicates.append("VibSpike")
        if r["p_foul_h"] >= policy["watch_foul_prob_thr"]:
            predicates.append("WatchFoulProb")
        if r.get("p_actionable_foul_h", 0.0) >= policy["actionable_foul_prob_thr"]:
            predicates.append("ActionableFoulProb")
        if r["p_clog_h"] >= policy["clog_prob_thr"]:
            predicates.append("ClogProb")

        watch_fouling = int(
            (r["p_foul_h"] >= policy["watch_foul_prob_thr"])
            or (r["pred_tte_foul_min"] <= policy["tau_foul_watch"])
            or (r["pred_severity"] >= policy["severity_incipient_thr"])
            or (r["pred_stage"] >= 1)
        )

        actionable_fouling = int(
            (
                (r.get("p_actionable_foul_h", 0.0) >= policy["actionable_foul_prob_thr"])
                or (
                    (r["p_foul_h"] >= policy["watch_foul_prob_thr"])
                    and ((r["pred_severity"] >= policy["severity_advanced_thr"]) or (r["pred_stage"] >= 2))
                )
            )
            and (r["pred_ttu_min"] <= policy["foul_urgency_ttu"])
        )

        clogging = int(
            ((r["p_clog_h"] >= policy["clog_prob_thr"]) or (r["pred_tte_clog_min"] <= policy["tau_clog"]))
            and (r["pred_ttu_min"] <= policy["tau_unplanned"])
        )

        if clogging:
            status = "Obstrucción probable"
            priority = "high"
            action = "inspección hidráulica / desatasco"
            cause = "firma hidráulica compatible con obstrucción próxima"
            alert_type = "clogging"
            action_flag = 1
        elif actionable_fouling:
            status = "Fouling accionable"
            priority = "high" if r["pred_ttu_min"] <= 0.7 * policy["foul_urgency_ttu"] else "medium"
            action = "programar intervención antes de pérdida operativa"
            cause = "estado degradado con riesgo de evento no planificado"
            alert_type = "fouling"
            action_flag = 1
        elif watch_fouling:
            status = "Watch fouling"
            priority = "medium" if r["pred_stage"] >= 1 else "low"
            action = "seguir tendencia y preparar ventana óptima"
            cause = "señal de estado compatible con formación de depósito"
            alert_type = "none"
            action_flag = 0
        else:
            status = "Normal"
            priority = "low"
            action = "operación normal"
            cause = "sin evidencia suficiente"
            alert_type = "none"
            action_flag = 0

        statuses.append(status)
        priorities.append(priority)
        actions.append(action)
        causes.append(cause)
        predicates_str.append(" + ".join(predicates) if predicates else "none")
        alert_types.append(alert_type)
        actionable.append(action_flag)
        watch_flags.append(watch_fouling)
        actionable_foul_flags.append(actionable_fouling)

    df["operator_status"] = statuses
    df["priority"] = priorities
    df["recommended_action"] = actions
    df["cause"] = causes
    df["activated_predicates"] = predicates_str
    df["alert_type"] = alert_types
    df["actionable_alert"] = actionable
    df["watch_fouling"] = watch_flags
    df["actionable_fouling"] = actionable_foul_flags
    return df


def consolidate_alerts(pred_df: pd.DataFrame, cooldown_min: int) -> pd.DataFrame:
    """Collapse repeated actionable alerts of the same (asset, type) inside the cooldown window."""
    cand = pred_df.loc[pred_df["actionable_alert"] == 1].sort_values(["asset_id", "alert_type", "timestamp"]).copy()
    kept_rows: List[int] = []
    last_time: Dict[Tuple[str, str], pd.Timestamp] = {}
    for idx, row in cand.iterrows():
        key = (str(row["asset_id"]), str(row["alert_type"]))
        ts = pd.Timestamp(row["timestamp"])
        lt = last_time.get(key)
        if lt is None or (ts - lt).total_seconds() / 60.0 >= cooldown_min:
            kept_rows.append(idx)
            last_time[key] = ts
    return cand.loc[kept_rows].reset_index(drop=True)


def default_policy(cfg: TrainConfig) -> Dict[str, float]:
    """Fallback alert-policy thresholds used if no calibrated policy artifact is available."""
    thr_incipient, thr_advanced = cfg.resolved_stage_thresholds()
    return {
        "clog_prob_thr": 0.55,
        "watch_foul_prob_thr": 0.35,
        "actionable_foul_prob_thr": 0.45,
        "tau_clog": float(min(max(cfg.clog_horizon_min, 15), 60)),
        "tau_foul_watch": float(cfg.fouling_horizon_min),
        "tau_unplanned": float(min(cfg.ttu_cap_min, 90)),
        "foul_urgency_ttu": float(min(cfg.ttu_cap_min, cfg.unplanned_fouling_horizon_min)),
        "severity_incipient_thr": float(thr_incipient),
        "severity_advanced_thr": float(thr_advanced),
        "cooldown_min": float(cfg.cooldown_min_default),
        "objective": float("nan"),
    }
