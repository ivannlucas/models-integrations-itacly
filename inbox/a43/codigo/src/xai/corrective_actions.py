"""
Módulo de acciones correctivas: generación inteligente según estado y evidencia.
"""

import numpy as np
from typing import Any, Dict, Optional


# Configuración por defecto de bloques de acción
DEFAULT_ACTION_CONFIG = {
    "combustion_control_flujo": {
        "variables": ["flujo_gas", "posicion_valvula", "potencia_kw"],
        "actions": [
            "Verificar alimentación de gas y respuesta de la válvula de control.",
            "Comprobar restricciones en línea o desviaciones del lazo de control de caudal.",
        ],
    },
    "termico": {
        "variables": ["temp_zona1", "temp_zona2", "temp_zona3", "temp_salida_gases"],
        "actions": [
            "Revisar estabilidad térmica del horno y la respuesta del sistema de aporte energético.",
            "Comprobar coherencia entre potencia aplicada y evolución de temperaturas.",
        ],
    },
    "ventilacion_presion": {
        "variables": ["velocidad_ventilador", "presion_ventilacion", "presion_camara"],
        "actions": [
            "Inspeccionar ventilación y estado del actuador o ventilador.",
            "Comprobar si existen oscilaciones de control o respuesta lenta del sistema de aire.",
        ],
    },
}


def _get_fuzzy_variable_weights_from_fusion(fusion_report: Dict[str, Any]) -> Dict[str, float]:
    """Extrae pesos fuzzy desde el bloque fusionado."""
    pairs = fusion_report.get("fuzzy", {}).get("dominant_variables", [])
    return {str(k): float(v) for k, v in pairs}


def _get_temporal_variable_info_from_fusion(fusion_report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extrae información temporal desde el bloque fusionado."""
    out = {}
    for v in fusion_report.get("temporal", {}).get("top_variables", []):
        name = str(v.get("feature"))
        if not name:
            continue
        out[name] = {
            "importance": float(v.get("importance", 0.0)),
            "span": str(v.get("dominant_span", "unknown")),
        }
    return out


def _interpretive_state_label(
    fusion_report: Dict[str, Any],
    strong_margin: float = 0.20,
    weak_margin: float = 0.10,
    branch_gap_threshold: float = 0.30,
    support_ratio_min: float = 0.70,
) -> tuple[str, str]:
    """Calcula estado interpretativo y rama dominante."""
    pred = fusion_report.get("prediction", {})

    threshold = float(pred.get("decision_threshold", 0.5))
    fuzzy_prob = float(pred.get("fuzzy", {}).get("probability", 0.0))
    temporal_prob = float(pred.get("temporal", {}).get("probability", 0.0))
    ensemble_prob = float(pred.get("ensemble", {}).get("probability", 0.5 * (fuzzy_prob + temporal_prob)))

    support_ratio = float(fusion_report.get("fuzzy", {}).get("support_ratio", 1.0))

    fuzzy_class = int(fuzzy_prob >= threshold)
    temporal_class = int(temporal_prob >= threshold)
    ensemble_class = int(ensemble_prob >= threshold)

    signed_margin = ensemble_prob - threshold
    margin = abs(signed_margin)
    branch_gap = abs(fuzzy_prob - temporal_prob)

    if fuzzy_class == 1 and temporal_class == 1:
        dominant_branch = "both"
    elif fuzzy_class == 0 and temporal_class == 1:
        dominant_branch = "temporal"
    elif fuzzy_class == 1 and temporal_class == 0:
        dominant_branch = "fuzzy"
    else:
        dominant_branch = "none"

    if ensemble_class == 1:
        if fuzzy_class == 1 and temporal_class == 1 and margin >= weak_margin and support_ratio >= support_ratio_min:
            state_label = "confirmed_anomaly"
        else:
            state_label = "unconfirmed_alert"
    else:
        if margin >= strong_margin and branch_gap < branch_gap_threshold:
            state_label = "normal"
        else:
            state_label = "normal_with_signals"

    return state_label, dominant_branch


def _score_action_block(
    block_name: str,
    block_cfg: Dict[str, Any],
    fuzzy_vars: Dict[str, float],
    temporal_vars: Dict[str, Dict[str, Any]],
    fusion_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Puntúa relevancia de un bloque de acción."""
    cfg_vars = set(block_cfg.get("variables", []))
    fuzzy_total = float(sum(abs(v) for v in fuzzy_vars.values()))
    temporal_total = float(sum(abs(v.get("importance", 0.0)) for v in temporal_vars.values()))

    fuzzy_match_mass = 0.0
    fuzzy_matches = []
    for var, weight in fuzzy_vars.items():
        if var in cfg_vars:
            fuzzy_matches.append(var)
            fuzzy_match_mass += abs(float(weight))

    temporal_match_mass = 0.0
    temporal_matches = []
    support_spans = []
    for var, info in temporal_vars.items():
        if var in cfg_vars:
            temporal_matches.append(var)
            imp = abs(float(info.get("importance", 0.0)))
            temporal_match_mass += imp
            support_spans.append(str(info.get("span", "unknown")))

    fuzzy_score_norm = (fuzzy_match_mass / fuzzy_total) if fuzzy_total > 0 else 0.0
    temporal_score_norm = (temporal_match_mass / temporal_total) if temporal_total > 0 else 0.0
    coverage = (len(set(fuzzy_matches) | set(temporal_matches)) / len(cfg_vars)) if cfg_vars else 0.0

    lambda_anom = float(fusion_report.get("prediction", {}).get("lambda_anomaly", 0.5))
    lambda_anom = float(np.clip(lambda_anom, 0.05, 0.95))
    wf = lambda_anom
    wl = 1.0 - lambda_anom

    total_score = (wf * fuzzy_score_norm + wl * temporal_score_norm) * 0.95 + 0.05 * coverage
    total_score = float(min(max(total_score, 0.0), 1.0))

    support_vars = sorted(set(fuzzy_matches) | set(temporal_matches))

    return {
        "block": block_name,
        "score": float(total_score),
        "fuzzy_score": float(fuzzy_score_norm),
        "temporal_score": float(temporal_score_norm),
        "coverage": float(coverage),
        "support_variables": support_vars,
        "support_patterns": sorted(set(support_spans)),
        "actions": list(block_cfg.get("actions", [])),
    }


def generate_corrective_actions(
    fusion_report: Dict[str, Any],
    action_config: Optional[Dict[str, Any]] = None,
    corrective_actions_cfg: Optional[Dict[str, Any]] = None,
    top_blocks: int = 3,
    min_block_score: float = 0.10,
) -> Dict[str, Any]:
    """Genera acciones correctivas basadas en el reporte fusionado."""

    if action_config is None:
        action_config = DEFAULT_ACTION_CONFIG
    corrective_actions_cfg = corrective_actions_cfg or {}

    fuzzy_vars = _get_fuzzy_variable_weights_from_fusion(fusion_report)
    temporal_vars = _get_temporal_variable_info_from_fusion(fusion_report)
    state_label, dominant_branch = _interpretive_state_label(
        fusion_report,
        strong_margin=float(corrective_actions_cfg.get("strong_margin", 0.15)),
        weak_margin=float(corrective_actions_cfg.get("weak_margin", 0.10)),
        branch_gap_threshold=float(corrective_actions_cfg.get("branch_gap_threshold", 0.30)),
        support_ratio_min=float(corrective_actions_cfg.get("support_ratio_min", 0.70)),
    )

    scored_blocks = []
    for block_name, block_cfg in action_config.items():
        row = _score_action_block(
            block_name=block_name,
            block_cfg=block_cfg,
            fuzzy_vars=fuzzy_vars,
            temporal_vars=temporal_vars,
            fusion_report=fusion_report,
        )
        if row["score"] >= float(min_block_score):
            scored_blocks.append(row)

    scored_blocks = sorted(scored_blocks, key=lambda x: x["score"], reverse=True)[:top_blocks]

    suggested_actions = []
    for block in scored_blocks:
        for action_text in block["actions"]:
            suggested_actions.append({
                "action": action_text,
                "source_block": block["block"],
                "score": float(block["score"]),
                "support_variables": block["support_variables"],
                "support_patterns": block["support_patterns"],
            })

    # Quitar duplicados
    dedup = []
    seen = set()
    for item in suggested_actions:
        key = item["action"]
        if key not in seen:
            seen.add(key)
            dedup.append(item)

    if state_label == "normal":
        dedup = []
    elif state_label == "normal_with_signals":
        dedup = dedup[:1]
    elif state_label == "unconfirmed_alert":
        dedup = dedup[:2]
    else:
        dedup = dedup[:4]

    return {
        "state_label": state_label,
        "dominant_branch": dominant_branch,
        "ranked_blocks": scored_blocks,
        "suggested_actions": dedup,
        "prediction": dict(fusion_report.get("prediction", {})),
    }
