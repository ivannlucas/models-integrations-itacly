"""Runs the TCN over prepared windows and attaches operator-facing explanations/alerts."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from app.plugins.ml46_dairy_fouling_clog_detection._vendor.common import TrainConfig
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.datasets import (
    AssetSequence,
    WindowDataset,
    collate_window_batch,
)
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.evaluation import (
    add_explanations,
    consolidate_alerts,
)
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.model_arch import PredictiveTCN


def run_inference(
    model: PredictiveTCN,
    prepared: tuple,
    train_cfg: TrainConfig,
    *,
    batch_size: int = 256,
    stride: int | None = None,
) -> pd.DataFrame:
    """Score every valid window and return one row per window (no true_* labels).

    ``prepared`` = (sequences, feature_indices, asset_ids) as returned by
    ``preprocessing.prepare_sequences``.
    """
    sequences, feature_indices, asset_ids = prepared
    dataset = WindowDataset(sequences, asset_ids, feature_indices, train_cfg, stride=stride)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_window_batch,
    )
    return _run_loader(model, loader, sequences, train_cfg)


def run_inference_single_window(
    model: PredictiveTCN,
    sequences: dict[str, AssetSequence],
    feature_indices: list[int],
    train_cfg: TrainConfig,
    window: tuple[str, int],
) -> pd.DataFrame:
    """Score exactly one ``window`` = (sequence_id, end_idx) — used by predict_inline."""
    sequence_id, end_idx = window
    seq = sequences[sequence_id]
    start_idx = end_idx - train_cfg.seq_len + 1
    x = seq.x[start_idx:end_idx + 1][:, feature_indices]
    batch = {
        "x": torch.tensor(x, dtype=torch.float32).unsqueeze(0),
        "sequence_id": [sequence_id],
        "end_idx": [int(end_idx)],
    }
    return _run_loader(model, [batch], sequences, train_cfg)


def _batch_outputs(pred: dict, train_cfg: TrainConfig) -> dict[str, np.ndarray]:
    """Probabilities and de-scaled regression outputs of one batch of TCN predictions."""
    return {
        "p_stage": torch.softmax(pred["stage_logits"], dim=-1).cpu().numpy(),
        "p_foul_h": torch.sigmoid(pred["foul_h_logit"]).cpu().numpy(),
        "p_actionable_foul_h": torch.sigmoid(pred["actionable_foul_h_logit"]).cpu().numpy(),
        "p_clog_h": torch.sigmoid(pred["clog_h_logit"]).cpu().numpy(),
        "pred_sev": (pred["severity_scaled"].cpu().numpy() / train_cfg.severity_scale()).clip(
            min=0.0
        ),
        "pred_ttf_foul": np.expm1(
            np.clip(
                pred["tte_foul_log"].cpu().numpy(), 0.0, np.log1p(train_cfg.tte_fouling_cap_min)
            )
        ),
        "pred_ttf_clog": np.expm1(
            np.clip(
                pred["tte_clog_log"].cpu().numpy(), 0.0, np.log1p(train_cfg.tte_clog_cap_min)
            )
        ),
        "pred_ttu": np.expm1(
            np.clip(pred["ttu_log"].cpu().numpy(), 0.0, np.log1p(train_cfg.ttu_cap_min))
        ),
    }


def _meta_float(meta: pd.Series, key: str) -> float:
    """``meta[key]`` as float, NaN when missing."""
    return float(meta[key]) if pd.notna(meta[key]) else float("nan")


def _window_row(
    seq: AssetSequence, sequence_id: str, meta: pd.Series, out: dict, train_cfg: TrainConfig
) -> dict:
    """Response row of one window; ``out`` holds that window's entries of _batch_outputs()."""
    pred_stage = int(np.argmax(out["p_stage"]))
    return {
        "asset_id": seq.asset_id,
        "sequence_id": sequence_id,
        "timestamp": meta["timestamp"],
        "phase": meta["phase"],
        "maintenance_active": (
            int(meta["maintenance_active"])
            if pd.notna(meta["maintenance_active"])
            else 0
        ),
        "pred_severity": float(out["pred_sev"]),
        "pred_stage": pred_stage,
        "pred_stage_name": ["stable", "incipient", "advanced"][pred_stage],
        "p_stage0": float(out["p_stage"][0]),
        "p_stage1": float(out["p_stage"][1]),
        "p_stage2": float(out["p_stage"][2]),
        "p_foul_h": float(out["p_foul_h"]),
        "p_watch_fouling": float(out["p_foul_h"]),
        "p_actionable_foul_h": float(out["p_actionable_foul_h"]),
        "p_actionable_fouling": float(out["p_actionable_foul_h"]),
        "p_clog_h": float(out["p_clog_h"]),
        "pred_tte_foul_min": float(
            np.clip(out["pred_ttf_foul"], 0.0, train_cfg.tte_fouling_cap_min)
        ),
        "pred_tte_clog_min": float(
            np.clip(out["pred_ttf_clog"], 0.0, train_cfg.tte_clog_cap_min)
        ),
        "pred_ttu_min": float(np.clip(out["pred_ttu"], 0.0, train_cfg.ttu_cap_min)),
        "flow_kg_s": _meta_float(meta, "flow_kg_s"),
        "dP_kPa": _meta_float(meta, "dP_kPa"),
        "vibration_mm_s": _meta_float(meta, "vibration_mm_s"),
        "thermal_eff_proxy": _meta_float(meta, "thermal_eff_proxy"),
        "heat_proxy": _meta_float(meta, "heat_proxy"),
        "resid_flow_kg_s": _meta_float(meta, "resid_flow_kg_s"),
        "resid_dP_kPa": _meta_float(meta, "resid_dP_kPa"),
        "resid_vibration_mm_s": _meta_float(meta, "resid_vibration_mm_s"),
        "resid_thermal_eff_proxy": _meta_float(meta, "resid_thermal_eff_proxy"),
        "dP_kPa_slope15": _meta_float(meta, "dP_kPa_slope15"),
        "vibration_mm_s_slope15": _meta_float(meta, "vibration_mm_s_slope15"),
        "milk_type": str(meta["milk_type"]),
        "asset_family": str(meta["asset_family"]),
        "fault_type": str(meta["fault_type"]),
    }


def _run_loader(
    model: PredictiveTCN, loader: Any, sequences: dict[str, AssetSequence], train_cfg: TrainConfig
) -> pd.DataFrame:
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(train_cfg.device)
            out = _batch_outputs(model(x), train_cfg)

            for i, sequence_id in enumerate(batch["sequence_id"]):
                end_idx = int(batch["end_idx"][i])
                seq = sequences[sequence_id]
                rows.append(
                    _window_row(
                        seq, sequence_id, seq.meta.iloc[end_idx],
                        {k: v[i] for k, v in out.items()}, train_cfg,
                    )
                )
    if not rows:  # no valid window: the caller reports why (sorting an empty frame raised KeyError)
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["asset_id", "timestamp", "sequence_id"])
        .reset_index(drop=True)
    )


def explain_and_alert(
    pred_df: pd.DataFrame,
    policy: dict,
    predicate_thresholds: dict,
    train_cfg: TrainConfig,
    cooldown_min: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach operator explanations to every window and consolidate them into alert episodes."""
    explained = add_explanations(pred_df, policy, predicate_thresholds, train_cfg)
    if cooldown_min is not None:
        cooldown = cooldown_min
    else:
        # Widened by alert_episode_gap_min so a manually-lowered policy cooldown can't split
        # one physical episode into several alerts (mirrors src/predict/predictor.py).
        policy_cooldown = float(policy.get("cooldown_min", train_cfg.cooldown_min_default))
        episode_gap = float(
            getattr(train_cfg, "alert_episode_gap_min", train_cfg.cooldown_min_default)
        )
        cooldown = int(max(policy_cooldown, episode_gap))
    alerts = consolidate_alerts(explained, cooldown)
    return explained, alerts
