"""
Módulo de explicabilidad fuzzy: análisis de reglas, antecedentes y firmas fuzzy.
"""

import numpy as np
import torch
from typing import Any, Dict, List, Optional
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
    """Computa efectos y contribuciones de reglas TSK."""

    if consequents is None:
        raise ValueError("En modo tsk se requieren consequents en la salida del modelo.")
    contributions = rule_activations * consequents
    return {
        "rule_effects": consequents,
        "rule_contributions": contributions,
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
        include_raw: bool = False,
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
        predicted_class = int(fuzzy_probability >= anomaly_threshold)

        rule_activations = out["rule_activations"].detach().cpu().numpy()[0]
        consequents = out["consequents"].detach().cpu().numpy().reshape(-1)
        alpha_probs = out["alpha_probs"].detach().cpu().numpy()
        membership = out["membership"].detach().cpu().numpy()[0]

        eff = _compute_rule_effects(
            rule_activations=rule_activations,
            consequents=consequents,
        )
        rule_effects = eff["rule_effects"]
        rule_contributions = eff["rule_contributions"]

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

        def _sort_rules(rule_indices):
            return sorted(
                rule_indices,
                key=lambda r: abs(float(rule_contributions[r])),
                reverse=True,
            )

        top_support_idx = _sort_rules(support_idx)[:top_rules]
        top_oppose_idx = _sort_rules(oppose_idx)[:top_rules]

        # Construcción de reglas explicativas
        def _build_rule_record(rule_idx: int) -> Dict[str, Any]:
            ants = _build_rule_antecedents(
                feature_names=self.feature_names or [f"feature_{i}" for i in range(alpha_probs.shape[1])],
                stats=self.stats_creation,
                alpha_probs=alpha_probs,
                membership=membership,
                rule_idx=rule_idx,
                mf_semantics=self.mf_semantics,
                max_antecedents_per_rule=max_antecedents_per_rule,
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

        top_rules_support = [_build_rule_record(r) for r in top_support_idx]
        top_rules_oppose = [_build_rule_record(r) for r in top_oppose_idx]

        # Firma global fuzzy
        var_counter = Counter()
        stat_counter = Counter()
        mf_counter = Counter()
        weighted_var_counter = defaultdict(float)
        weighted_stat_counter = defaultdict(float)

        for rule in top_rules_support:
            w_rule = abs(float(rule["contribution"]))
            for ant in rule["antecedents"]:
                feat_base = str(ant["feature_base"])
                feat_stat = str(ant["feature_stat"])
                mf_name = str(ant["mf_name"])
                w_ant = float(ant["antecedent_score"]) * w_rule

                var_counter[feat_base] += 1
                stat_counter[feat_stat] += 1
                mf_counter[mf_name] += 1
                weighted_var_counter[feat_base] += w_ant
                weighted_stat_counter[feat_stat] += w_ant

        support_mass = float(np.sum(np.abs([r["contribution"] for r in top_rules_support]))) if top_rules_support else 0.0
        oppose_mass = float(np.sum(np.abs([r["contribution"] for r in top_rules_oppose]))) if top_rules_oppose else 0.0
        total_mass = support_mass + oppose_mass
        support_ratio = float(support_mass / total_mass) if total_mass > 0 else 0.0

        case_signature = {
            "dominant_variables": sorted(weighted_var_counter.items(), key=lambda x: x[1], reverse=True)[:8],
            "dominant_stats": sorted(weighted_stat_counter.items(), key=lambda x: x[1], reverse=True)[:8],
            "most_frequent_variables": var_counter.most_common(8),
            "most_frequent_stats": stat_counter.most_common(8),
            "most_frequent_mfs": mf_counter.most_common(8),
            "support_mass": float(support_mass),
            "oppose_mass": float(oppose_mass),
            "support_ratio": float(support_ratio),
        }

        report = {
            "predicted_class": int(predicted_class),
            "decision_threshold": float(anomaly_threshold),
            "anomaly_score": float(anomaly_score),
            "global_anomaly_probability": float(global_anomaly_probability),
            "logit": float(fuzzy_logit),
            "probability": float(fuzzy_probability),
            "top_rules_support": top_rules_support,
            "top_rules_oppose": top_rules_oppose,
            "case_signature": case_signature,
        }

        if include_raw:
            report["raw_fuzzy"] = {
                "rule_activations": rule_activations.tolist(),
                "rule_effects": rule_effects.tolist(),
                "rule_contributions": rule_contributions.tolist(),
            }

        return report
