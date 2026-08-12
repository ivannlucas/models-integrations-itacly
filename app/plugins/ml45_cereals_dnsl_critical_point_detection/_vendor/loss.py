"""Vendored verbatim from inbox/a45/codigo/src/training/loss.py — used only by trainer.py
(fine-tuning), same composite loss the original DNF model was trained with.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DNFLoss(nn.Module):
    """Loss compuesta para el modelo Deep Neuro-Fuzzy.

    Combina losses supervisadas de anomalía con losses auxiliares por rama
    y regularización fuzzy.

    Args:
        anomaly_pos_weight: Peso positivo para BCEWithLogitsLoss.
        anomaly_weight: Peso de la loss de anomalía final.
        aux_anom_dl_weight: Peso auxiliar anomalía rama DL.
        aux_anom_fuzzy_weight: Peso auxiliar anomalía rama fuzzy.
        diversity_weight: Peso de regularización por diversidad.
        alpha_entropy_weight: Peso de regularización por entropía alpha.
        rule_structure_div_weight: Peso de diversidad estructural.
        rule_usage_balance_weight: Peso de balance de uso de reglas.
        reduction: Tipo de reducción.
        eps: Epsilon numérico.
    """

    def __init__(
        self,
        anomaly_pos_weight: Optional[torch.Tensor] = None,
        anomaly_weight: float = 1.0,
        aux_anom_dl_weight: float = 0.10,
        aux_anom_fuzzy_weight: float = 0.30,
        diversity_weight: float = 0.01,
        alpha_entropy_weight: float = 0.005,
        rule_structure_div_weight: float = 0.01,
        rule_usage_balance_weight: float = 0.0,
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()

        self.anomaly_weight = anomaly_weight

        self.aux_anom_dl_weight = aux_anom_dl_weight
        self.aux_anom_fuzzy_weight = aux_anom_fuzzy_weight

        self.diversity_weight = diversity_weight
        self.alpha_entropy_weight = alpha_entropy_weight
        self.rule_structure_div_weight = rule_structure_div_weight
        self.rule_usage_balance_weight = rule_usage_balance_weight

        self.reduction = reduction
        self.eps = eps

        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=anomaly_pos_weight, reduction=reduction
        )

    def _offdiag_corr_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Penaliza correlación fuera de diagonal entre columnas de x.

        Args:
            x: Tensor [B, R] de activaciones de reglas.

        Returns:
            Penalización escalar.
        """
        if x.dim() != 2:
            raise ValueError(f"x debe ser [B,R], recibido {tuple(x.shape)}")

        B, R = x.shape
        if B < 2 or R < 2:
            return x.new_tensor(0.0)

        x_centered = x - x.mean(dim=0, keepdim=True)
        std = x_centered.std(dim=0, keepdim=True, unbiased=False).clamp_min(self.eps)
        x_std = x_centered / std

        corr = (x_std.T @ x_std) / B
        mask = ~torch.eye(R, device=x.device, dtype=torch.bool)
        offdiag_vals = corr[mask]
        return (offdiag_vals ** 2).mean()

    def _alpha_entropy_penalty(self, alpha_probs: torch.Tensor) -> torch.Tensor:
        """Penaliza entropía alta en alpha_probs para favorecer reglas nítidas.

        Args:
            alpha_probs: Tensor [R, F, M].

        Returns:
            Penalización escalar.
        """
        if alpha_probs.dim() != 3:
            raise ValueError(
                f"alpha_probs debe ser [R,F,M], recibido {tuple(alpha_probs.shape)}"
            )

        entropy = -(
            alpha_probs.clamp_min(self.eps)
            * torch.log(alpha_probs.clamp_min(self.eps))
        ).sum(dim=-1)
        return entropy.mean()

    def _rule_structure_diversity_penalty(
        self, alpha_probs: torch.Tensor
    ) -> torch.Tensor:
        """Penaliza reglas con estructuras muy similares.

        Args:
            alpha_probs: Tensor [R, F, M].

        Returns:
            Penalización escalar.
        """
        if alpha_probs.dim() != 3:
            raise ValueError(
                f"alpha_probs debe ser [R,F,M], recibido {tuple(alpha_probs.shape)}"
            )

        R = alpha_probs.size(0)
        if R < 2:
            return alpha_probs.new_tensor(0.0)

        flat = alpha_probs.reshape(R, -1)
        flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(self.eps)

        sim = flat @ flat.T
        mask = ~torch.eye(R, device=alpha_probs.device, dtype=torch.bool)
        offdiag_vals = sim[mask]
        return (offdiag_vals ** 2).mean()

    def _rule_usage_balance_penalty(
        self, rule_activations: torch.Tensor
    ) -> torch.Tensor:
        """Evita que pocas reglas monopolicen la activación.

        Args:
            rule_activations: Tensor [B, R].

        Returns:
            Penalización escalar.
        """
        if rule_activations.dim() != 2:
            raise ValueError(
                f"rule_activations debe ser [B,R], "
                f"recibido {tuple(rule_activations.shape)}"
            )

        mean_usage = rule_activations.mean(dim=0)
        target = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
        return F.mse_loss(mean_usage, target, reduction=self.reduction)

    def forward(
        self,
        anomaly_score: torch.Tensor,
        logit_anomaly_dl: Optional[torch.Tensor],
        logit_anomaly_fuzzy: Optional[torch.Tensor],
        rule_activations: torch.Tensor,
        alpha_probs: Optional[torch.Tensor],
        y_anomaly: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calcula la loss compuesta total.

        Args:
            anomaly_score: Score de anomalía final [B, 1].
            logit_anomaly_dl: Score anomalía rama DL.
            logit_anomaly_fuzzy: Score anomalía rama fuzzy.
            rule_activations: Activaciones de reglas [B, R].
            alpha_probs: Probabilidades alpha [R, F, M].
            y_anomaly: Etiquetas binarias [B].

        Returns:
            Tupla con (loss total, dict de losses individuales).
        """
        y_anomaly = y_anomaly.float().view(-1)

        ref = anomaly_score

        # Loss principal
        loss_anom_final = self.bce_loss(anomaly_score.view(-1), y_anomaly)

        # Losses auxiliares por rama
        loss_anom_dl = ref.new_tensor(0.0)
        loss_anom_fuzzy = ref.new_tensor(0.0)

        if logit_anomaly_dl is not None:
            loss_anom_dl = self.bce_loss(logit_anomaly_dl.view(-1), y_anomaly)
        if logit_anomaly_fuzzy is not None:
            loss_anom_fuzzy = self.bce_loss(logit_anomaly_fuzzy.view(-1), y_anomaly)

        # Regularización fuzzy
        loss_diversity = self._offdiag_corr_penalty(rule_activations)

        loss_alpha_entropy = ref.new_tensor(0.0)
        loss_rule_structure = ref.new_tensor(0.0)

        if alpha_probs is not None:
            loss_alpha_entropy = self._alpha_entropy_penalty(alpha_probs)
            loss_rule_structure = self._rule_structure_diversity_penalty(alpha_probs)

        loss_rule_usage = ref.new_tensor(0.0)
        if self.rule_usage_balance_weight > 0:
            loss_rule_usage = self._rule_usage_balance_penalty(rule_activations)

        # Total
        total_loss = (
            self.anomaly_weight * loss_anom_final
            + self.aux_anom_dl_weight * loss_anom_dl
            + self.aux_anom_fuzzy_weight * loss_anom_fuzzy
            + self.diversity_weight * loss_diversity
            + self.alpha_entropy_weight * loss_alpha_entropy
            + self.rule_structure_div_weight * loss_rule_structure
            + self.rule_usage_balance_weight * loss_rule_usage
        )

        loss_dict = {
            "loss_total": float(total_loss.detach().cpu()),
            "loss_anom_final": float(loss_anom_final.detach().cpu()),
            "loss_anom_dl": float(loss_anom_dl.detach().cpu()),
            "loss_anom_fuzzy": float(loss_anom_fuzzy.detach().cpu()),
            "loss_diversity": float(loss_diversity.detach().cpu()),
            "loss_alpha_entropy": float(loss_alpha_entropy.detach().cpu()),
            "loss_rule_structure": float(loss_rule_structure.detach().cpu()),
            "loss_rule_usage": float(loss_rule_usage.detach().cpu()),
        }

        return total_loss, loss_dict
