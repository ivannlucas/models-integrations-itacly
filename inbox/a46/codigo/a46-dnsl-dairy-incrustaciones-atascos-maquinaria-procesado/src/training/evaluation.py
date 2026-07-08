
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)

from src.utils.common import nan_to_zero, normalize_fault_type, safe_float, save_json

from .common import TrainConfig, safe_binary_auc_ap, stage_to_label, total_asset_days


def window_metrics_from_preds(df: pd.DataFrame, cfg: TrainConfig) -> Dict[str, float]:
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
    model.eval()
    rows: List[Dict[str, Any]] = []
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
                    "time_to_unplanned_fouling_min": float(meta["time_to_unplanned_fouling_min"]) if pd.notna(meta["time_to_unplanned_fouling_min"]) else float("nan"),
                    "ttm_to_unplanned_event_min": float(meta["ttm_to_unplanned_event_min"]) if pd.notna(meta["ttm_to_unplanned_event_min"]) else float("nan"),
                    "ttm_to_planned_cip_min": float(meta["ttm_to_planned_cip_min"]) if pd.notna(meta["ttm_to_planned_cip_min"]) else float("nan"),
                }
            )
    return pd.DataFrame(rows).sort_values(["asset_id", "timestamp", "sequence_id"]).reset_index(drop=True)

def add_explanations(pred_df: pd.DataFrame, policy: Mapping[str, float], predicate_thresholds: Mapping[str, float], cfg: TrainConfig) -> pd.DataFrame:
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

def build_event_table(maintenance_df: pd.DataFrame, asset_ids: Sequence[str]) -> pd.DataFrame:
    df = maintenance_df.copy()
    if len(df) == 0:
        return df
    df["fault_type_norm"] = df["fault_type"].map(normalize_fault_type)
    df = df.loc[
        (df["asset_id"].isin(asset_ids))
        & (df["planned"].fillna(0).astype(int) == 0)
        & (df["fault_type_norm"].isin(["fouling", "clogging"]))
    ].copy()
    if "duration_min" not in df.columns:
        df["duration_min"] = (pd.to_datetime(df["end_time"]) - pd.to_datetime(df["start_time"])).dt.total_seconds() / 60.0
    return df.sort_values(["asset_id", "start_time"]).reset_index(drop=True)

def match_alerts_to_events(alerts_df: pd.DataFrame, events_df: pd.DataFrame, match_window_min: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    alerts = alerts_df.copy()
    alerts["matched_event_id"] = None
    alerts["lead_min"] = np.nan
    events = events_df.copy()
    events["matched"] = 0
    events["matched_alert_idx"] = None
    events["lead_min"] = np.nan

    if len(events) == 0 or len(alerts) == 0:
        return alerts, events

    for (asset_id, fault_type), ev_g in events.groupby(["asset_id", "fault_type_norm"], sort=False):
        al_g = alerts.loc[(alerts["asset_id"] == asset_id) & (alerts["alert_type"] == fault_type)].copy()
        if len(al_g) == 0:
            continue
        al_g = al_g.sort_values("timestamp")
        used_alerts: set[int] = set()
        for ev_idx, ev in ev_g.sort_values("start_time").iterrows():
            ev_start = pd.Timestamp(ev["start_time"])
            cand = al_g.loc[
                (~al_g.index.isin(used_alerts))
                & (al_g["timestamp"] <= ev_start)
                & (al_g["timestamp"] >= ev_start - pd.Timedelta(minutes=match_window_min))
            ]
            if len(cand) == 0:
                continue
            chosen_idx = cand.sort_values("timestamp").index[0]
            used_alerts.add(int(chosen_idx))
            lead = (ev_start - pd.Timestamp(alerts.loc[chosen_idx, "timestamp"])).total_seconds() / 60.0
            events.loc[ev_idx, "matched"] = 1
            events.loc[ev_idx, "matched_alert_idx"] = int(chosen_idx)
            events.loc[ev_idx, "lead_min"] = float(lead)
            alerts.loc[chosen_idx, "matched_event_id"] = str(ev.get("maintenance_id", ev_idx))
            alerts.loc[chosen_idx, "lead_min"] = float(lead)
    return alerts, events

def event_metrics_from_matches(matched_alerts: pd.DataFrame, matched_events: pd.DataFrame, telemetry_df: pd.DataFrame, asset_ids: Sequence[str]) -> Dict[str, Any]:
    asset_days = total_asset_days(telemetry_df, asset_ids)
    false_alarms = int(matched_alerts["matched_event_id"].isna().sum()) if len(matched_alerts) else 0
    tp_total = int(matched_events["matched"].sum()) if len(matched_events) else 0
    total_events = int(len(matched_events))
    fn_total = int(total_events - tp_total)
    precision = tp_total / max(tp_total + false_alarms, 1)
    recall = tp_total / max(total_events, 1)
    macro_parts = []
    per_type: Dict[str, Any] = {}
    for fault_type in ["clogging", "fouling"]:
        ev_t = matched_events.loc[matched_events["fault_type_norm"] == fault_type]
        al_t = matched_alerts.loc[matched_alerts["alert_type"] == fault_type]
        tp = int(ev_t["matched"].sum()) if len(ev_t) else 0
        fn = int(len(ev_t) - tp)
        fa = int(al_t["matched_event_id"].isna().sum()) if len(al_t) else 0
        prec = tp / max(tp + fa, 1)
        rec = tp / max(len(ev_t), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        macro_parts.append(f1)
        per_type[fault_type] = {
            "tp_events": tp,
            "fn_events": fn,
            "false_alarms": fa,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "avg_lead_min": float(ev_t["lead_min"].dropna().mean()) if tp > 0 else float("nan"),
        }
    macro_f1 = float(np.mean(macro_parts)) if macro_parts else 0.0
    return {
        "total_events": total_events,
        "tp_events_total": tp_total,
        "fn_events_total": fn_total,
        "false_alarms_total": false_alarms,
        "precision_total": precision,
        "recall_total": recall,
        "macro_event_f1": macro_f1,
        "false_alarms_per_day_total": false_alarms / asset_days,
        "per_type": per_type,
    }

def evaluate_policy(
    pred_df: pd.DataFrame,
    events_df: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    asset_ids: Sequence[str],
    policy: Mapping[str, float],
    predicate_thresholds: Mapping[str, float],
    cfg: TrainConfig,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    explained = add_explanations(pred_df, policy, predicate_thresholds, cfg)
    alerts = consolidate_alerts(explained, int(policy["cooldown_min"]))
    matched_alerts, matched_events = match_alerts_to_events(alerts, events_df, match_window_min=cfg.match_window_min)
    metrics = event_metrics_from_matches(matched_alerts, matched_events, telemetry_df, asset_ids)
    return metrics, alerts, matched_alerts, matched_events

def default_policy(cfg: TrainConfig) -> Dict[str, float]:
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

def calibrate_policy(
    val_pred_df: pd.DataFrame,
    val_events: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    val_assets: Sequence[str],
    predicate_thresholds: Mapping[str, float],
    cfg: TrainConfig,
) -> Dict[str, float]:
    if len(val_pred_df) == 0:
        return default_policy(cfg)
    thr_incipient, thr_advanced = cfg.resolved_stage_thresholds()
    grids = {
        "clog_prob_thr": [0.40, 0.55, 0.70, 0.85],
        "watch_foul_prob_thr": [0.25, 0.35, 0.45, 0.55],
        "actionable_foul_prob_thr": [0.25, 0.40, 0.55, 0.70],
        "tau_clog": [15.0, 30.0, 45.0, 60.0],
        "tau_foul_watch": [float(cfg.fouling_horizon_min), float(max(cfg.fouling_horizon_min * 2, 60))],
        "tau_unplanned": [30.0, 60.0, 90.0, 120.0],
        "foul_urgency_ttu": [30.0, 60.0, 90.0, 120.0],
        "severity_incipient_thr": [0.90 * thr_incipient, 1.00 * thr_incipient, 1.10 * thr_incipient],
        "severity_advanced_thr": [0.95 * thr_advanced, 1.00 * thr_advanced, 1.05 * thr_advanced],
        "cooldown_min": [60.0, 90.0, 120.0],
    }
    keys = list(grids.keys())
    mesh = list(product(*[grids[k] for k in keys]))
    rng = np.random.default_rng(cfg.seed)
    if len(mesh) > int(cfg.policy_max_candidates):
        choose_idx = rng.choice(len(mesh), size=int(cfg.policy_max_candidates), replace=False)
        candidate_values = [mesh[int(i)] for i in choose_idx]
    else:
        candidate_values = mesh
    best: Optional[Dict[str, float]] = None
    best_obj = -1e18
    for values in candidate_values:
        policy = {k: float(v) for k, v in zip(keys, values)}
        if policy["severity_advanced_thr"] <= policy["severity_incipient_thr"]:
            continue
        if len(val_events) > 0:
            metrics, _, _, _ = evaluate_policy(val_pred_df, val_events, telemetry_df, val_assets, policy, predicate_thresholds, cfg)
            obj = metrics["macro_event_f1"] - 0.03 * metrics["false_alarms_per_day_total"]
        else:
            # Fall back to a window-level proxy when no events are available in validation.
            watch_pred = (val_pred_df["p_foul_h"] >= policy["watch_foul_prob_thr"]).astype(int)
            actionable_pred = (val_pred_df["p_actionable_foul_h"] >= policy["actionable_foul_prob_thr"]).astype(int)
            clog_pred = (val_pred_df["p_clog_h"] >= policy["clog_prob_thr"]).astype(int)
            obj = (
                0.45 * f1_score(val_pred_df["true_foul_h"], watch_pred, zero_division=0)
                + 0.25 * f1_score(val_pred_df["true_actionable_foul_h"], actionable_pred, zero_division=0)
                + 0.30 * f1_score(val_pred_df["true_clog_h"], clog_pred, zero_division=0)
            )
        if obj > best_obj:
            best_obj = obj
            best = policy
    if best is None:
        best = default_policy(cfg)
        best_obj = float("nan")
    best["objective"] = float(best_obj)
    return best

def save_confusion_matrices(
    pred_df: pd.DataFrame,
    split_name: str,
    scenario_dir: Path,
    policy: Mapping[str, float],
    cfg: TrainConfig,
) -> None:
    if len(pred_df) == 0:
        return
    cm_dir = scenario_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []

    def _save(y_true: np.ndarray, y_pred: np.ndarray, label_ids: List[int], label_names: List[str], name: str, title: str) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=label_ids)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(float), np.maximum(row_sums, 1), where=np.maximum(row_sums, 1) > 0)
        counts_df = pd.DataFrame(cm, index=label_names, columns=label_names)
        norm_df = pd.DataFrame(cm_norm, index=label_names, columns=label_names)
        counts_path = cm_dir / f"{split_name}_{name}_counts.csv"
        norm_path = cm_dir / f"{split_name}_{name}_row_normalized.csv"
        png_path = cm_dir / f"{split_name}_{name}.png"
        counts_df.to_csv(counts_path)
        norm_df.to_csv(norm_path)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
        im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=25)
        ax.set_yticks(range(len(label_names)))
        ax.set_yticklabels(label_names)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(title)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                txt = f"{int(cm[i, j])}\n{100.0 * cm_norm[i, j]:.1f}%"
                color = "white" if cm_norm[i, j] > 0.45 else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Normalizado por fila")
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)
        manifest_rows.append(
            {
                "split": split_name,
                "target": name,
                "png": str(png_path.name),
                "counts_csv": str(counts_path.name),
                "row_normalized_csv": str(norm_path.name),
            }
        )

    _save(
        pred_df["true_stage"].to_numpy(dtype=int),
        pred_df["pred_stage"].to_numpy(dtype=int),
        [0, 1, 2],
        ["stable", "incipient", "advanced"],
        "stage_now",
        f"{split_name} · stage_now",
    )
    watch_thr = float(policy.get("watch_foul_prob_thr", 0.5))
    action_thr = float(policy.get("actionable_foul_prob_thr", 0.5))
    clog_thr = float(policy.get("clog_prob_thr", 0.5))
    _save(
        pred_df["true_foul_h"].to_numpy(dtype=int),
        (pred_df["p_foul_h"].to_numpy(dtype=float) >= watch_thr).astype(int),
        [0, 1],
        ["neg", "pos"],
        f"watch_fouling_onset_within_{cfg.fouling_horizon_min}min",
        f"{split_name} · fouling_onset_within_{cfg.fouling_horizon_min}min · thr={watch_thr:.2f}",
    )
    if "true_actionable_foul_h" in pred_df.columns and "p_actionable_foul_h" in pred_df.columns:
        _save(
            pred_df["true_actionable_foul_h"].to_numpy(dtype=int),
            (pred_df["p_actionable_foul_h"].to_numpy(dtype=float) >= action_thr).astype(int),
            [0, 1],
            ["neg", "pos"],
            f"unplanned_fouling_within_{cfg.unplanned_fouling_horizon_min}min",
            f"{split_name} · unplanned_fouling_within_{cfg.unplanned_fouling_horizon_min}min · thr={action_thr:.2f}",
        )
    _save(
        pred_df["true_clog_h"].to_numpy(dtype=int),
        (pred_df["p_clog_h"].to_numpy(dtype=float) >= clog_thr).astype(int),
        [0, 1],
        ["neg", "pos"],
        f"clog_onset_within_{cfg.clog_horizon_min}min",
        f"{split_name} · clog_onset_within_{cfg.clog_horizon_min}min · thr={clog_thr:.2f}",
    )
    pd.DataFrame(manifest_rows).to_csv(cm_dir / f"{split_name}_confusion_manifest.csv", index=False)
