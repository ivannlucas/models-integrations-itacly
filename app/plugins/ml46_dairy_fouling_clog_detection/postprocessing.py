"""Runs the TCN over prepared windows and attaches operator-facing explanations/alerts."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from app.plugins.ml46_dairy_fouling_clog_detection._vendor.common import TrainConfig
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.datasets import AssetSequence, WindowDataset, collate_window_batch
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.evaluation import add_explanations, consolidate_alerts
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.model_arch import PredictiveTCN


def run_inference(
    model: PredictiveTCN,
    sequences: dict[str, AssetSequence],
    feature_indices: list[int],
    asset_ids: list[str],
    train_cfg: TrainConfig,
    batch_size: int = 256,
    stride: int | None = None,
) -> pd.DataFrame:
    """Score every valid window for *asset_ids* and return one row per window (no true_* labels)."""
    dataset = WindowDataset(sequences, asset_ids, feature_indices, train_cfg, stride=stride)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_window_batch)
    return _run_loader(model, loader, sequences, train_cfg)


def run_inference_single_window(
    model: PredictiveTCN,
    sequences: dict[str, AssetSequence],
    feature_indices: list[int],
    train_cfg: TrainConfig,
    sequence_id: str,
    end_idx: int,
) -> pd.DataFrame:
    """Score exactly one (sequence_id, end_idx) window — used by predict_inline."""
    seq = sequences[sequence_id]
    start_idx = end_idx - train_cfg.seq_len + 1
    x = seq.x[start_idx:end_idx + 1][:, feature_indices]
    batch = {
        "x": torch.tensor(x, dtype=torch.float32).unsqueeze(0),
        "sequence_id": [sequence_id],
        "end_idx": [int(end_idx)],
    }
    return _run_loader(model, [batch], sequences, train_cfg)


def _run_loader(model: PredictiveTCN, loader: Any, sequences: dict[str, AssetSequence], train_cfg: TrainConfig) -> pd.DataFrame:
    model.eval()
    rows: list[dict] = []
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
                        "flow_kg_s": float(meta["flow_kg_s"]) if pd.notna(meta["flow_kg_s"]) else float("nan"),
                        "dP_kPa": float(meta["dP_kPa"]) if pd.notna(meta["dP_kPa"]) else float("nan"),
                        "vibration_mm_s": float(meta["vibration_mm_s"]) if pd.notna(meta["vibration_mm_s"]) else float("nan"),
                        "thermal_eff_proxy": float(meta["thermal_eff_proxy"]) if pd.notna(meta["thermal_eff_proxy"]) else float("nan"),
                        "heat_proxy": float(meta["heat_proxy"]) if pd.notna(meta["heat_proxy"]) else float("nan"),
                        "resid_flow_kg_s": float(meta["resid_flow_kg_s"]) if pd.notna(meta["resid_flow_kg_s"]) else float("nan"),
                        "resid_dP_kPa": float(meta["resid_dP_kPa"]) if pd.notna(meta["resid_dP_kPa"]) else float("nan"),
                        "resid_vibration_mm_s": float(meta["resid_vibration_mm_s"]) if pd.notna(meta["resid_vibration_mm_s"]) else float("nan"),
                        "resid_thermal_eff_proxy": float(meta["resid_thermal_eff_proxy"]) if pd.notna(meta["resid_thermal_eff_proxy"]) else float("nan"),
                        "dP_kPa_slope15": float(meta["dP_kPa_slope15"]) if pd.notna(meta["dP_kPa_slope15"]) else float("nan"),
                        "vibration_mm_s_slope15": float(meta["vibration_mm_s_slope15"]) if pd.notna(meta["vibration_mm_s_slope15"]) else float("nan"),
                        "milk_type": str(meta["milk_type"]),
                        "asset_family": str(meta["asset_family"]),
                        "fault_type": str(meta["fault_type"]),
                    }
                )
    return pd.DataFrame(rows).sort_values(["asset_id", "timestamp", "sequence_id"]).reset_index(drop=True)


def explain_and_alert(pred_df: pd.DataFrame, policy: dict, predicate_thresholds: dict, train_cfg: TrainConfig, cooldown_min: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach operator explanations to every window and consolidate them into alert episodes."""
    explained = add_explanations(pred_df, policy, predicate_thresholds, train_cfg)
    cooldown = cooldown_min if cooldown_min is not None else int(policy.get("cooldown_min", train_cfg.cooldown_min_default))
    alerts = consolidate_alerts(explained, cooldown)
    return explained, alerts
