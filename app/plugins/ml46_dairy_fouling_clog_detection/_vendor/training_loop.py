"""Vendored (adapted) from inbox/a46/codigo/.../src/training/pipeline.py.

Only train_one_epoch (same optimizer/loss/weights as the AI team's original code) is kept.
window_objective/run_scenario/train_pipeline are NOT vendored: those drive the original
from-scratch, dual-scenario, fresh-asset-split training run. The plugin's train() (see
plugin.py) fine-tunes the single already-served no_clock checkpoint on the caller's CSV
instead — see manifest known_issues and the plugin-integration skill's fine-tuning pattern.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.plugins.ml46_dairy_fouling_clog_detection._vendor.common import TrainConfig
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.evaluation import stage_score_from_severity


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
    """One training epoch over *loader*, identical loss mix to the AI team's original code."""
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
