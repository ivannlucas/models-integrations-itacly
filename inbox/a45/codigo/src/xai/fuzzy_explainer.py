"""
Módulo de explicabilidad fuzzy: análisis de reglas, antecedentes y firmas fuzzy.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

from .utils import (
    safe_float,
    sigmoid,
    split_feature_base_and_stat,
    get_semantic_mf_name_for_feature,
)


def _compute_rule_effects(
    rule_activations: np.ndarray,
    consequents: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Computa efectos de reglas."""
    if consequents is None:
        raise ValueError("En modo tsk se requieren consequents en la salida del modelo.")
    contributions = rule_activations * consequents
    return {
        "mode": "tsk",
        "rule_effects": consequents,
        "rule_contributions": contributions,
        "head_bias": 0.0,
    }


def _build_rule_antecedents(
    feature_names: List[str],
    stats: Optional[List[str]],
    alpha_probs: np.ndarray,
    membership: np.ndarray,
    rule_idx: int,
    max_antecedents_per_rule: int = 6,
    min_specialization: float = 0.0,
    mf_semantics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Construye antecedentes de una regla."""
    n_features = alpha_probs.shape[1]
    ants = []

    for f_idx in range(n_features):
        alpha_rf = alpha_probs[rule_idx, f_idx, :]
        mf_idx = int(np.argmax(alpha_rf))
        specialization = float(np.max(alpha_rf))
        selected_membership = float(membership[f_idx, mf_idx])

        if specialization < float(min_specialization):
            continue

        feat_name = str(feature_names[f_idx]) if f_idx < len(feature_names) else f"feature_{f_idx}"
        feat_base, feat_stat = split_feature_base_and_stat(feat_name, stats=stats)

        if mf_semantics is not None:
            semantic_mf_name = get_semantic_mf_name_for_feature(
                mf_semantics=mf_semantics,
                feature_name=feat_name,
                mf_idx=mf_idx,
            )
        else:
            semantic_mf_name = f"mf_{mf_idx}"

        ants.append({
            "feature_idx": int(f_idx),
            "feature": feat_name,
            "feature_base": feat_base,
            "feature_stat": feat_stat,
            "mf_idx": int(mf_idx),
            "mf_name": semantic_mf_name,
            "specialization": specialization,
            "membership_value": selected_membership,
            "antecedent_score": specialization * selected_membership,
        })

    ants = sorted(
        ants,
        key=lambda x: (x["antecedent_score"], x["specialization"], x["membership_value"]),
        reverse=True,
    )

    return ants[:max_antecedents_per_rule]


class FuzzyExplainer:
    """Generador de explicaciones fuzzy."""

    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        mf_semantics: Optional[Dict[str, Any]] = None,
        stats_creation: Optional[List[str]] = None,
    ):
        self.model = model
        self.feature_names = feature_names
        self.mf_semantics = mf_semantics
        self.stats_creation = list(stats_creation) if stats_creation else None

    def explain_sample(
        self,
        x_window: np.ndarray,
        s_stats: np.ndarray,
        top_rules: int = 5,
        max_antecedents_per_rule: int = 6,
        min_rule_activation: float = 0.0,
        min_specialization: float = 0.0,
        anomaly_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Genera explicación fuzzy para una muestra."""
        self.model.eval()
        device = next(self.model.parameters()).device

        # Preparación de tensores
        if not torch.is_tensor(x_window):
            x_window_tensor = torch.tensor(x_window, dtype=torch.float32).unsqueeze(0)
        else:
            x_window_tensor = x_window.detach().float()
            if x_window_tensor.dim() == 2:
                x_window_tensor = x_window_tensor.unsqueeze(0)

        if s_stats is None:
            raise ValueError("s_stats no puede ser None en la explicación fuzzy.")

        s_stats = np.asarray(s_stats, dtype=np.float32)

        if not torch.is_tensor(s_stats):
            s_stats_tensor = torch.tensor(s_stats, dtype=torch.float32).unsqueeze(0)
        else:
            s_stats_tensor = s_stats.detach().float()
            if s_stats_tensor.dim() == 1:
                s_stats_tensor = s_stats_tensor.unsqueeze(0)

        x_window_tensor = x_window_tensor.to(device)
        s_stats_tensor = s_stats_tensor.to(device)

        with torch.no_grad():
            out = self.model(x_window_tensor, s_stats=s_stats_tensor)

        anomaly_score = safe_float(out["anomaly_score"].detach().cpu().numpy().reshape(-1)[0])
        global_anomaly_probability = float(sigmoid(anomaly_score))

        fuzzy_logit = safe_float(out["logit_anomaly_fuzzy"].detach().cpu().numpy().reshape(-1)[0])
        fuzzy_probability = float(sigmoid(fuzzy_logit))
        fuzzy_output_tsk = safe_float(out["output_anomaly"].detach().cpu().numpy().reshape(-1)[0])
        predicted_class = int(fuzzy_probability >= anomaly_threshold)

        rule_activations = out["rule_activations"].detach().cpu().numpy()[0]
        consequents = out["consequents"].detach().cpu().numpy().reshape(-1)
        alpha_probs = out["alpha_probs"].detach().cpu().numpy()
        membership = out["membership"].detach().cpu().numpy()[0]

        if (
        self.feature_names is not None
        and len(self.feature_names) != alpha_probs.shape[1]
        ):
            raise ValueError(
                "Número de nombres en rama neuro-difusa incompatible: "
                f"esperados {alpha_probs.shape[1]}, "
                f"recibidos {len(self.feature_names)}."
        )

        eff = _compute_rule_effects(
            rule_activations=rule_activations,
            consequents=consequents,
        )
        fuzzy_mode = eff["mode"]
        rule_effects = eff["rule_effects"]
        rule_contributions = eff["rule_contributions"]
        head_bias = float(eff.get("head_bias", 0.0))

        # Selección de reglas
        if predicted_class == 1:
            support_mask = rule_contributions > 0
        else:
            support_mask = rule_contributions < 0

        candidate_idx = []
        for r_idx, act in enumerate(rule_activations):
            if float(act) < float(min_rule_activation):
                continue
            candidate_idx.append(r_idx)

        support_idx = [r for r in candidate_idx if support_mask[r]]
        oppose_idx = [r for r in candidate_idx if not support_mask[r]]

        top_rules = int(top_rules)
        max_antecedents_per_rule = int(max_antecedents_per_rule)

        if top_rules < 0:
            raise ValueError("top_rules no puede ser negativo.")
        if max_antecedents_per_rule <= 0:
            raise ValueError(
                "max_antecedents_per_rule debe ser mayor que cero."
            )

        def _sort_rules(rule_indices):
            return sorted(
                rule_indices,
                key=lambda r: abs(float(rule_contributions[r])),
                reverse=True,
            )

        sorted_support_idx = _sort_rules(support_idx)
        sorted_oppose_idx = _sort_rules(oppose_idx)

        def _build_rule_record(
            rule_idx: int,
            antecedent_limit: int,
        ) -> Dict[str, Any]:
            ants = _build_rule_antecedents(
                feature_names=self.feature_names or [
                    f"feature_{i}"
                    for i in range(alpha_probs.shape[1])
                ],
                stats=self.stats_creation,
                alpha_probs=alpha_probs,
                membership=membership,
                rule_idx=rule_idx,
                mf_semantics=self.mf_semantics,
                max_antecedents_per_rule=antecedent_limit,
                min_specialization=min_specialization,
            )

            return {
                "rule_idx": int(rule_idx),
                "activation": float(rule_activations[rule_idx]),
                "effect": float(rule_effects[rule_idx]),
                "contribution": float(rule_contributions[rule_idx]),
                "supports_prediction": bool(support_mask[rule_idx]),
                "antecedents": ants,
            }

        # Evidencia completa para cálculos PCC.
        evidence_support_rules = [
            _build_rule_record(
                rule_idx=r,
                antecedent_limit=alpha_probs.shape[1],
            )
            for r in sorted_support_idx
        ]

        # Salida compacta destinada a presentación.
        top_rules_support = [
            _build_rule_record(r, max_antecedents_per_rule)
            for r in sorted_support_idx[:top_rules]
        ]
        top_rules_oppose = [
            _build_rule_record(r, max_antecedents_per_rule)
            for r in sorted_oppose_idx[:top_rules]
        ]

        var_counter = Counter()
        stat_counter = Counter()
        mf_counter = Counter()
        weighted_var_counter = defaultdict(float)
        weighted_stat_counter = defaultdict(float)

        for rule in evidence_support_rules:
            w_rule = abs(float(rule["contribution"]))

            valid_antecedents = []
            for ant in rule["antecedents"]:
                antecedent_score = float(ant["antecedent_score"])

                if not np.isfinite(antecedent_score) or antecedent_score <= 1e-12:
                    continue

                valid_antecedents.append((ant, antecedent_score))

            antecedent_mass = float(
                sum(score for _, score in valid_antecedents)
            )

            if antecedent_mass <= 1e-12:
                continue

            for ant, antecedent_score in valid_antecedents:
                feat_base = str(ant["feature_base"])
                feat_stat = str(ant["feature_stat"])
                mf_name = str(ant["mf_name"])

                # Cada regla reparte exactamente su masa entre sus antecedentes.
                antecedent_share = antecedent_score / antecedent_mass
                w_ant = w_rule * antecedent_share

                var_counter[feat_base] += 1
                stat_counter[feat_stat] += 1
                mf_counter[mf_name] += 1
                weighted_var_counter[feat_base] += w_ant
                weighted_stat_counter[feat_stat] += w_ant

        support_mass = float(
            sum(abs(float(rule_contributions[r])) for r in support_idx)
        )
        oppose_mass = float(
            sum(abs(float(rule_contributions[r])) for r in oppose_idx)
        )

        total_mass = support_mass + oppose_mass
        support_ratio = (
            support_mass / total_mass if total_mass > 0 else 0.0
        )

        case_signature = {
            "variable_weights": {
                str(k): float(v)
                for k, v in weighted_var_counter.items()
            },
            "stat_weights": {
                str(k): float(v)
                for k, v in weighted_stat_counter.items()
            },
            "dominant_variables": sorted(
                weighted_var_counter.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:8],
            "dominant_stats": sorted(
                weighted_stat_counter.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:8],
            "most_frequent_variables": var_counter.most_common(8),
            "most_frequent_stats": stat_counter.most_common(8),
            "most_frequent_mfs": mf_counter.most_common(8),
            "support_mass": float(support_mass),
            "oppose_mass": float(oppose_mass),
            "support_ratio": float(support_ratio),
        }

        return {
            "predicted_class": int(predicted_class),
            "decision_threshold": float(anomaly_threshold),
            "anomaly_score": float(anomaly_score),
            "anomaly_probability": float(fuzzy_probability),
            "global_anomaly_probability": float(global_anomaly_probability),
            "fuzzy_mode": fuzzy_mode,
            "fuzzy_logit": float(fuzzy_logit),
            "fuzzy_probability": float(fuzzy_probability),
            "fuzzy_output_tsk": float(fuzzy_output_tsk),
            "head_bias": float(head_bias),
            "top_rules_support": top_rules_support,
            "top_rules_oppose": top_rules_oppose,
            "fuzzy_case_signature": case_signature,
            "raw_fuzzy": {
                "rule_activations": rule_activations.tolist(),
                "rule_effects": rule_effects.tolist(),
                "rule_contributions": rule_contributions.tolist(),
            },
        }
