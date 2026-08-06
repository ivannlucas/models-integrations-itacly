"""Vendored verbatim (structure/math unchanged) from inbox/a45/codigo/src/training/model.py.

Must match the delivered checkpoint (best_dnf_model.pt) bit-for-bit — do not refactor layer
names/order, only the logger wiring was adapted (get_logger() -> logging.getLogger(__name__)).
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FuzzyMembershipLayer(nn.Module):
    """Capa de membresía fuzzy con funciones Gaussianas aprendibles.

    Args:
        n_features: Número de features de entrada.
        n_mf: Número de funciones de membresía por feature.
        entrenable: Si los parámetros son entrenables.
    """

    def __init__(self, n_features: int, n_mf: int, entrenable: bool = True):
        super().__init__()
        self.n_features = n_features
        self.n_mf = n_mf
        self.entrenable = entrenable

        self.centers = nn.Parameter(
            torch.randn(n_features, n_mf), requires_grad=entrenable
        )
        self.sigmas = nn.Parameter(
            torch.ones(n_features, n_mf) * 0.5, requires_grad=entrenable
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcula grados de membresía Gaussianos.

        Args:
            x: Tensor de entrada [B, F].

        Returns:
            Tensor de membresía [B, F, M].
        """
        x_exp = x.unsqueeze(-1)
        sigmas = torch.clamp(torch.abs(self.sigmas), min=1e-6)
        return torch.exp(-((x_exp - self.centers) ** 2) / (2 * sigmas ** 2))


class FuzzyRuleLayer(nn.Module):
    """Capa de reglas fuzzy estilo ANFIS (diferenciable).

    Args:
        n_features: Número de features.
        n_mf: Número de funciones de membresía por feature.
        entrenable: Si los parámetros son entrenables.
        n_rules: Número de reglas fuzzy.
        t_norm: T-norma a usar ('product', 'min').
        normalize: Si normalizar activaciones de reglas.
        eps: Epsilon numérico.
        temperature: Temperatura para softmax.
        init: Inicialización de alpha ('random' o 'sparse').
        use_log_bias: Si usar bias logarítmico en reglas.
        seed: Semilla para reproducibilidad.
    """

    def __init__(
        self,
        n_features: int,
        n_mf: int,
        entrenable: bool = True,
        n_rules: int = 48,
        t_norm: str = "product",
        normalize: bool = True,
        eps: float = 1e-8,
        temperature: float = 1.0,
        init: str = "random",
        use_log_bias: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.n_rules = n_rules
        self.n_features = n_features
        self.n_mf = n_mf
        self.t_norm = t_norm
        self.normalize = normalize
        self.eps = eps
        self.temperature = temperature
        self.use_log_bias = use_log_bias

        g = torch.Generator()
        g.manual_seed(seed)

        if init == "sparse":
            alpha = torch.zeros(n_rules, n_features, n_mf)
            idx = torch.randint(
                low=0, high=n_mf, size=(n_rules, n_features), generator=g
            )
            alpha.scatter_(2, idx.unsqueeze(-1), 5.0)
            alpha += 0.1 * torch.randn(n_rules, n_features, n_mf, generator=g)
        else:
            alpha = 0.2 * torch.randn(n_rules, n_features, n_mf, generator=g)

        self.alpha = nn.Parameter(alpha, requires_grad=entrenable)

        if self.use_log_bias:
            self.rule_log_bias = nn.Parameter(
                torch.zeros(n_rules), requires_grad=entrenable
            )

        self.consequent_anom = nn.Linear(n_features, n_rules)

    def forward(
        self, x: torch.Tensor, membership: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass de la capa de reglas.

        Args:
            x: Tensor de entrada [B, F].
            membership: Tensor de membresía [B, F, M].

        Returns:
            Tupla con (output_anomaly, rule_activations, alpha_probs, consequents_anom).
        """
        if membership.dim() != 3:
            raise RuntimeError(
                f"membership debe ser 3D [B,F,M], recibido: {membership.shape}"
            )

        _, f_in, m_in = membership.shape
        if f_in != self.n_features or m_in != self.n_mf:
            raise RuntimeError(
                f"Dimensión incorrecta:\n"
                f" - esperado [B,{self.n_features},{self.n_mf}]\n"
                f" - recibido {membership.shape}"
            )

        # Selección de MF por regla y feature (soft con temperatura)
        alpha_probs = torch.softmax(self.alpha / self.temperature, dim=-1)

        selected = torch.sum(
            membership.unsqueeze(1) * alpha_probs.unsqueeze(0), dim=-1
        )  # [B,R,F]
        selected = selected.clamp_min(self.eps)

        # T-norma y normalización
        if self.t_norm == "product":
            log_w = torch.sum(torch.log(selected), dim=-1)
            if self.use_log_bias:
                log_w = log_w + self.rule_log_bias.unsqueeze(0)
            if self.normalize:
                rule_activations = torch.softmax(log_w, dim=1)
            else:
                rule_activations = torch.exp(log_w)

        elif self.t_norm == "min":
            w = torch.min(selected, dim=-1).values
            if self.use_log_bias:
                w = w * torch.exp(self.rule_log_bias).unsqueeze(0)
            if self.normalize:
                rule_activations = w / torch.sum(
                    w, dim=1, keepdim=True
                ).clamp_min(self.eps)
            else:
                rule_activations = w

        else:
            raise ValueError(f"T-norma no reconocida: {self.t_norm}")

        # Consecuentes TSK lineales
        consequents_anom = self.consequent_anom(x)  # [B, R]
        output_anomaly = torch.sum(
            rule_activations * consequents_anom, dim=1, keepdim=True
        )  # [B,1]

        return output_anomaly, rule_activations, alpha_probs, consequents_anom


class ParallelDeepNeuroFuzzyModel(nn.Module):
    """Modelo híbrido paralelo Deep Learning + Neuro-Fuzzy.

    Combina una rama Deep Learning (LSTM + atención temporal) con una rama neuro-difusa
    (membresía + reglas) mediante fusión tardía ponderada.

    Args:
        model_cfg: Diccionario de configuración del modelo.
        seed: Semilla para reproducibilidad.
    """

    def __init__(self, model_cfg: dict, seed: int = 42):
        super().__init__()

        fuzzy_cfg = model_cfg["fuzzy"]
        lstm_cfg = model_cfg["lstm"]

        self.n_features = model_cfg["input_features"]
        self.n_stats_features = (
            self.n_features
            if model_cfg["n_stats_features"] is None
            else model_cfg["n_stats_features"]
        )
        self.lambda_anomaly = float(fuzzy_cfg.get("lambda_anomaly", 0.5))

        # Rama Deep Learning (LSTM + atención)
        bidirectional = bool(lstm_cfg["bidirectional"])
        lstm_hidden = lstm_cfg["hidden_size"]
        lstm_layers = lstm_cfg["num_layers"]
        lstm_dropout = 0.0 if lstm_layers == 1 else lstm_cfg["dropout"]

        self.lstm_branch = nn.LSTM(
            input_size=self.n_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.attention_layer = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_dim // 2, 1),
        )

        self.dl_projection = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_cfg["embedding_dim"]),
            nn.ReLU(),
            nn.Dropout(lstm_cfg["dropout"]),
        )

        self.anomaly_dl = nn.Linear(lstm_cfg["embedding_dim"], 1)

        # Rama neuro-difusa: Membresía + Reglas
        self.fuzzy_layer = FuzzyMembershipLayer(
            n_features=self.n_stats_features,
            n_mf=fuzzy_cfg["n_mf"],
            entrenable=fuzzy_cfg["train_membership_params"],
        )

        self.rule_layer = FuzzyRuleLayer(
            n_features=self.n_stats_features,
            n_mf=fuzzy_cfg["n_mf"],
            entrenable=fuzzy_cfg["train_rule_params"],
            n_rules=fuzzy_cfg["n_rules"],
            t_norm=fuzzy_cfg["t_norm"],
            normalize=fuzzy_cfg["normalize_rules"],
            init=fuzzy_cfg["init_alpha"],
            seed=seed,
            use_log_bias=fuzzy_cfg["use_log_bias"],
            temperature=fuzzy_cfg["temperature"],
        )

        logger.info("=" * 30)
        logger.info("Arquitectura configurada:")
        logger.info("=" * 30)
        logger.info(
            f"Rama DL: Input {(model_cfg['sequence_length'], self.n_features)} "
            f"-> LSTM(hidden={lstm_hidden}, layers={lstm_layers}, bidir={bidirectional}) "
            f"-> Atención -> Embedding {lstm_cfg['embedding_dim']} "
            f"-> Head anom 1"
        )
        logger.info(
            f"Rama Fuzzy: Input {self.n_stats_features} "
            f"-> Membership ({fuzzy_cfg['n_mf']} por feature) "
            f"-> Reglas ({fuzzy_cfg['n_rules']}) "
            f"-> Output anom 1"
        )
        logger.info(
            f"Mezcla tardía: lambda_anom={self.lambda_anomaly:.3f}"
        )

    def forward(
        self,
        x_window: torch.Tensor,
        s_stats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass del modelo completo.

        Args:
            x_window: Tensor de secuencias temporales [B, T, F].
            s_stats: Tensor de estadísticas para rama fuzzy [B, F_stats].

        Returns:
            Diccionario con todas las salidas intermedias y finales.
        """
        if x_window.dim() != 3:
            raise RuntimeError(
                f"x_window debe ser [B,T,F], recibido: {tuple(x_window.shape)}"
            )

        B, _, F_in = x_window.shape
        if F_in != self.n_features:
            raise RuntimeError(
                f"Última dimensión de x_window inválida: "
                f"esperado {self.n_features}, recibido {F_in}"
            )

        # Rama Deep Learning (LSTM + atención)
        lstm_out, _ = self.lstm_branch(x_window)
        attn_logits = self.attention_layer(lstm_out)
        attn_weights = F.softmax(attn_logits, dim=1)
        temporal_context = torch.sum(lstm_out * attn_weights, dim=1)
        dl_embed = self.dl_projection(temporal_context)

        logit_anomaly_dl = self.anomaly_dl(dl_embed)

        # Rama neuro-difusa
        membership = self.fuzzy_layer(s_stats)
        output_anom, rule_activations, alpha_probs, consequents_anom = (
            self.rule_layer(s_stats, membership)
        )
        logit_anomaly_fuzzy = output_anom

        # Mezcla tardía
        lambda_anom = min(max(self.lambda_anomaly, 0.0), 1.0)

        anomaly_score = (
            (1.0 - lambda_anom) * logit_anomaly_dl
            + lambda_anom * logit_anomaly_fuzzy
        )

        return {
            "anomaly_score": anomaly_score,
            "logit_anomaly_dl": logit_anomaly_dl,
            "logit_anomaly_fuzzy": logit_anomaly_fuzzy,
            "output_anomaly": output_anom,
            "rule_activations": rule_activations,
            "alpha_probs": alpha_probs,
            "membership": membership,
            "consequents": consequents_anom,
            "dl_embed": dl_embed,
            "attention_weights": attn_weights,
            "temporal_context": temporal_context,
            "s_stats": s_stats,
        }
