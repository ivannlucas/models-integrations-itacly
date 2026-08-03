"""Vendored verbatim from inbox/a45/codigo/src/xai/pcc.py — detección de Puntos Críticos de
Control (PCC) a partir de reportes XAI. Only the relative import (.utils -> vendored utils.py
in this same package) is unchanged since both live under _vendor/xai/.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from app.plugins.ml45_cereals_dnsl_critical_point_detection._vendor.xai.utils import safe_float

SUBSYSTEMS_CONFIG = [
    {
        "name": "humedad",
        "features": ["exhaust_air_humidity", "grain_moisture_in"],
    },
    {
        "name": "termico_transferencia",
        "features": ["plenum_temp", "exhaust_air_temp", "burner_power"],
    },
    {
        "name": "ventilacion_presion",
        "features": ["static_pressure", "fan_speed"],
    },
    {
        "name": "descarga_control",
        "features": ["discharge_frequency", "setpoint_temp"],
    },
    {
        "name": "contexto_operativo",
        "features": ["ambient_temp", "ambient_humidity"],
    }
]


def _canon_subsystem_name(name: str, subsystems_config: Optional[List[Dict[str, Any]]] = None) -> str:
    valid_names = {str(item["name"]).strip() for item in (subsystems_config or []) if item.get("name")}
    name = str(name or "").strip()
    if name not in valid_names:
        raise ValueError(
            f"Subsistema no definido en SUBSYSTEM_CONFIG: {name!r}. "
            f"Subsistemas válidos: {sorted(valid_names)}"
        )

    return name


def _catalog_key(sub1: str, sub2: str, span: str, subsystems_config):
    sub1 = _canon_subsystem_name(sub1, subsystems_config)
    sub2 = _canon_subsystem_name(sub2, subsystems_config)

    pair = tuple(sorted([sub1, sub2]))
    return (pair[0], pair[1], str(span or "unknown").strip().lower())


def normalize_pcc_catalog(pcc_catalog: Dict, subsystems_config: Optional[List[Dict[str, Any]]] = None) -> Dict:
    """Normaliza claves del catálogo para que sub1-sub2 == sub2-sub1."""
    normalized = {}

    for key, value in (pcc_catalog or {}).items():
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError(
                "Las claves de pcc_catalog deben ser tuplas: (sub1, sub2, dominant_shap_span)."
            )

        sub1, sub2, span = key
        normalized[_catalog_key(sub1, sub2, span, subsystems_config)] = value

    return normalized


def build_pcc_catalog_from_records(records, subsystems_config):
    catalog = {}

    for record in records:
        key = _catalog_key(
            record["sub1"],
            record["sub2"],
            record["span"],
            subsystems_config,
        )

        catalog[key] = {
            "name": record.get("name", "PCC sin nombre"),
            "message": record.get("message", ""),
            "recommendation": record.get("recommendation", ""),
        }

    return catalog


DEFAULT_MONITOR_POLICY = {
    "normal_margin": 0.30,
    "critical_margin": 0.15,
    "min_support_catalog": 0.75,
    "min_subsystem_score": 0.05,
    "min_subsystem_variables": 1,
}

PCC_CATALOG = {
    ("humedad", "termico_transferencia", "final"): {
        "name": "PCC térmico-humedad tardío",
        "message": (
            "Perfil recurrente asociado al acoplamiento entre transferencia térmica "
            "y eliminación de humedad."
        ),
        "recommendation": (
            "Revisar evolución de humedad del grano, temperatura de plenum, temperatura de salida "
            "y potencia del quemador. Validar que la transferencia térmica permite una evacuación "
            "adecuada de humedad."
        ),
    },

    ("descarga_control", "termico_transferencia", "medio"): {
        "name": "PCC térmico-descarga intermedio",
        "message": (
            "Perfil asociado a la interacción entre dinámica de descarga y transferencia térmica."
        ),
        "recommendation": (
            "Comprobar que la regulación de descarga no compromete la transferencia térmica."
        ),
    },

    ("termico_transferencia", "ventilacion_presion", "medio"): {
        "name": "PCC térmico-ventilación intermedio",
        "message": (
            "Perfil emergente asociado a transferencia térmica condicionada por ventilación y presión."
        ),
        "recommendation": (
            "Comprobar estabilidad del flujo de aire, presión estática, velocidad del ventilador "
            "y su impacto sobre la transferencia térmica."
        ),
    },

    ("descarga_control", "termico_transferencia", "final"): {
        "name": "Perfil de vigilancia térmico-descarga",
        "message": (
            "Perfil frecuente asociado a interacción entre descarga y transferencia térmica, "
            "con separación limitada entre normalidad y anomalía."
        ),
        "recommendation": (
            "Mantener seguimiento reforzado de frecuencia de descarga, potencia y temperaturas. "
        ),
    },

    ("termico_transferencia", "ventilacion_presion", "final"): {
        "name": "Perfil de vigilancia térmico-ventilación",
        "message": (
            "Perfil recurrente asociado a transferencia térmica y ventilación/presión, "
            "con comportamiento más ambiguo."
        ),
        "recommendation": (
            "Monitorizar presión estática, ventilación y evolución térmica."
        ),
    },
}


def _normalize_evidence(
    raw_evidence: Dict[str, float],
) -> Dict[str, float]:
    """Limpia y normaliza evidencia para que su masa total sea uno."""
    cleaned = {}

    for key, value in (raw_evidence or {}).items():
        name = str(key).strip().lower()

        try:
            weight = abs(float(value))
        except (TypeError, ValueError):
            continue

        if not name or not np.isfinite(weight) or weight <= 1e-12:
            continue

        cleaned[name] = cleaned.get(name, 0.0) + weight

    total = float(sum(cleaned.values()))
    if total <= 1e-12:
        return {}

    return {
        name: weight / total
        for name, weight in cleaned.items()
    }


def _score_subsystem(
    subsystem_name: str,
    subsystem_features: List[str],
    fuzzy_vars: Dict[str, float],
    lstm_vars: Dict[str, float],
) -> Dict[str, Any]:
    """
    Calcula densidad media de evidencia por variable.

    Dividir por el número de variables evita favorecer automáticamente
    a subsistemas con más sensores.
    """
    cfg_vars = {
        str(value).strip().lower()
        for value in subsystem_features
        if str(value).strip()
    }

    fuzzy_norm = _normalize_evidence(fuzzy_vars)
    lstm_norm = _normalize_evidence(lstm_vars)

    shared_feature_weights = {
        feature: min(
            fuzzy_norm.get(feature, 0.0),
            lstm_norm.get(feature, 0.0),
        )
        for feature in set(fuzzy_norm) & set(lstm_norm)
    }

    global_overlap = float(sum(shared_feature_weights.values()))

    present_vars = {
        feature
        for feature in cfg_vars
        if feature in fuzzy_norm or feature in lstm_norm
    }

    n_features = len(cfg_vars)

    if n_features == 0:
        fuzzy_density = 0.0
        lstm_density = 0.0
        shared_density = 0.0
        coverage = 0.0
    else:
        fuzzy_density = (
            sum(fuzzy_norm.get(feature, 0.0) for feature in cfg_vars)
            / n_features
        )
        lstm_density = (
            sum(lstm_norm.get(feature, 0.0) for feature in cfg_vars)
            / n_features
        )
        shared_density = (
            sum(
                shared_feature_weights.get(feature, 0.0)
                for feature in cfg_vars
            )
            / n_features
        )
        coverage = len(present_vars) / n_features

    return {
        "subsystem": str(subsystem_name),
        "fuzzy_density": float(fuzzy_density),
        "lstm_density": float(lstm_density),
        "shared_density": float(shared_density),
        "global_overlap": float(global_overlap),
        "coverage": float(coverage),
        "support_variables": sorted(present_vars),
        "n_features": n_features,
    }


def _rank_subsystems(
    rep_fuzzy: Dict[str, Any],
    rep_lstm: Dict[str, Any],
    rep_fusion: Dict[str, Any],
    subsystems: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if subsystems is None:
        subsystems = SUBSYSTEMS_CONFIG

    fuzzy_signature = rep_fuzzy.get(
        "fuzzy_case_signature",
        {},
    )

    fuzzy_full = fuzzy_signature.get("variable_weights")
    if isinstance(fuzzy_full, dict):
        fuzzy_vars = {
            str(name): float(weight)
            for name, weight in fuzzy_full.items()
        }
    else:
        fuzzy_vars = {
            str(name): float(weight)
            for name, weight in fuzzy_signature.get(
                "dominant_variables",
                [],
            )
        }

    lstm_full = rep_lstm.get(
        "variable_importance_by_feature"
    )
    if isinstance(lstm_full, dict):
        lstm_vars = {
            str(name): float(weight)
            for name, weight in lstm_full.items()
        }
    else:
        lstm_vars = {
            str(row["feature"]): float(
                row.get("importance", 0.0)
            )
            for row in rep_lstm.get("top_variables", [])
            if row.get("feature") is not None
        }

    lambda_anomaly = float(
        rep_fusion.get("prediction", {}).get(
            "lambda_anomaly",
            rep_fusion.get(
                "fusion_summary",
                {},
            ).get("lambda_anomaly", 0.5),
        )
    )
    lambda_anomaly = float(
        np.clip(lambda_anomaly, 0.0, 1.0)
    )

    raw_rows = [
        _score_subsystem(
            subsystem_name=str(
                subsystem.get("name", "unknown")
            ),
            subsystem_features=list(
                subsystem.get("features", [])
            ),
            fuzzy_vars=fuzzy_vars,
            lstm_vars=lstm_vars,
        )
        for subsystem in subsystems
    ]

    def _relative_component(
        component_name: str,
    ) -> np.ndarray:
        values = np.asarray(
            [
                max(0.0, float(row.get(component_name, 0.0)))
                for row in raw_rows
            ],
            dtype=float,
        )

        total = float(values.sum())
        if total <= 1e-12:
            return np.zeros_like(values)

        return values / total

    fuzzy_distribution = _relative_component(
        "fuzzy_density"
    )
    lstm_distribution = _relative_component(
        "lstm_density"
    )
    shared_distribution = _relative_component(
        "shared_density"
    )
    coverage_distribution = _relative_component(
        "coverage"
    )

    global_overlap = (
        float(raw_rows[0]["global_overlap"])
        if raw_rows
        else 0.0
    )

    ranked = []

    for index, row in enumerate(raw_rows):
        fuzzy_score = float(fuzzy_distribution[index])
        lstm_score = float(lstm_distribution[index])

        branch_score = (
            lambda_anomaly * fuzzy_score
            + (1.0 - lambda_anomaly) * lstm_score
        )

        # El término compartido conserva la intensidad real del acuerdo.
        shared_score = float(
            global_overlap * shared_distribution[index]
        )

        coverage_score = float(
            coverage_distribution[index]
        )

        subsystem_score = (
            0.70 * branch_score
            + 0.20 * shared_score
            + 0.10 * coverage_score
        )

        ranked.append({
            **row,
            "score": float(
                np.clip(subsystem_score, 0.0, 1.0)
            ),
            "fuzzy_score": fuzzy_score,
            "lstm_score": lstm_score,
            "shared_score": shared_score,
            "coverage_score": coverage_score,
            "branch_score": float(branch_score),
        })

    ranked.sort(
        key=lambda row: row["score"],
        reverse=True,
    )

    if ranked:
        dominant_subsystem = str(
            ranked[0]["subsystem"]
        )
        dominant_score = float(ranked[0]["score"])
    else:
        dominant_subsystem = "unknown"
        dominant_score = 0.0

    return {
        "dominant_subsystem": dominant_subsystem,
        "dominant_score": dominant_score,
        "lambda_anomaly": lambda_anomaly,
        "global_branch_overlap": global_overlap,
        "ranked_subsystems": ranked,
    }


def _compute_online_state(
    prob: float,
    threshold: float,
    pcc_match: Optional[Dict[str, Any]],
    support_ratio: Optional[float] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Estados:
      - normal: no supera umbral, no hay PCC catalogado y hay margen suficiente
      - vigilancia: supera umbral pero no hay PCC catalogado o soporte insuficiente / no supera umbral pero hay PCC catalogado (falso negativo potencial)
      - criticidad detectada: supera umbral con un margen significativo, hay PCC catalogado y soporte suficiente
    """

    policy = policy or DEFAULT_MONITOR_POLICY
    normal_margin = float(policy.get("normal_margin", 0.25))
    critical_margin = float(policy.get("critical_margin", 0.15))
    min_support = float(policy.get("min_support_catalog", 0.75))

    has_pcc = bool(pcc_match)
    is_anomaly = float(prob) >= float(threshold)
    signed_margin = float(prob) - float(threshold)
    abs_margin = abs(signed_margin)

    support_ok = True
    if support_ratio is not None:
        support_ok = float(support_ratio) >= min_support

    if not is_anomaly and not has_pcc and abs_margin >= normal_margin:
        return "Normal"

    if has_pcc and is_anomaly and signed_margin >= critical_margin and support_ok:
        return "Criticidad detectada"

    return "Vigilancia"


def build_pcc_report(
    fuzzy_report: Dict[str, Any],
    lstm_report: Dict[str, Any],
    fusion_report: Dict[str, Any],
    subsystems: Optional[List[Dict[str, Any]]] = None,
    pcc_catalog: Optional[Dict[Tuple[str, str, str], Dict[str, Any]]] = None,
    monitor_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    subsystems = subsystems or SUBSYSTEMS_CONFIG
    pcc_catalog = pcc_catalog or PCC_CATALOG
    monitor_policy = {
        **DEFAULT_MONITOR_POLICY,
        **(monitor_policy or {}),
    }

    pcc_catalog = normalize_pcc_catalog(pcc_catalog or {}, subsystems_config=subsystems)

    pred = fusion_report.get("prediction", {}) or {}

    threshold = safe_float(pred.get("decision_threshold"), default=np.nan)
    prob = safe_float(pred.get("ensemble_prob", np.nan), default=np.nan)
    margin = safe_float(abs(prob - threshold) if np.isfinite(prob) else np.nan, default=np.nan)

    fuzzy_top = [str(v) for v, _ in (fuzzy_report.get("fuzzy_case_signature", {}).get("dominant_variables", []) or [])]
    lstm_top = [
        str(row.get("feature"))
        for row in (lstm_report.get("top_variables", []) or [])
        if row.get("feature") is not None
    ]
    shared_vars = [str(v) for v in (fusion_report.get("agreement", {}).get("shared_variables", []) or [])]
    dominant_span = str(lstm_report.get("dominant_span_shap", "unknown")).lower()
    support_ratio = safe_float(
        fuzzy_report.get("fuzzy_case_signature", {}).get("support_ratio"),
        default=np.nan,
    )

    subsystem_info = _rank_subsystems(
        rep_fuzzy=fuzzy_report,
        rep_lstm=lstm_report,
        rep_fusion=fusion_report,
        subsystems=subsystems,
    )
    ranked = subsystem_info.get("ranked_subsystems", [])

    min_score = float(monitor_policy.get("min_subsystem_score", 0.05))
    min_variables = int(monitor_policy.get("min_subsystem_variables", 1))

    eligible = [
        row for row in ranked
        if float(row.get("score", 0.0)) >= min_score
        and len(row.get("support_variables", [])) >= min_variables
    ]

    if len(eligible) >= 2:
        top1 = _canon_subsystem_name(eligible[0]["subsystem"], subsystems)
        top2 = _canon_subsystem_name(eligible[1]["subsystem"], subsystems)
        match_key = _catalog_key(top1, top2, dominant_span, subsystems)
        subsystem_pair = match_key[:2]
        pcc_match = pcc_catalog.get(match_key)
    else:
        top1 = eligible[0]["subsystem"] if eligible else "unknown"
        top2 = "unknown"
        subsystem_pair = tuple(v for v in (top1, top2) if v != "unknown")
        pcc_match = None

    key_variables = []
    for v in shared_vars + fuzzy_top + lstm_top:
        if v not in key_variables:
            key_variables.append(v)
    key_variables = key_variables[:3]

    state = _compute_online_state(
        prob=prob,
        threshold=threshold,
        pcc_match=pcc_match,
        support_ratio=support_ratio,
        policy=monitor_policy,
    )

    pcc_name = pcc_match.get("name") if pcc_match else "No identificado"

    if pcc_match:
        recommendation = pcc_match.get("recommendation")
        message = pcc_match.get("message")
    elif state == "Vigilancia":
        recommendation = "Se recomienda vigilancia reforzada y seguimiento."
        message = "No se identifica un perfil catalogado de criticidad, pero hay indicios de anomalia."
    else:
        recommendation = "Mantener monitorizacion ordinaria."
        message = "No se identifica un perfil catalogado de criticidad y no hay indicios fuertes de anomalia."

    return {
        "state": state,
        "pcc_candidate": pcc_name,
        "message": message,
        "probability": prob,
        "margin": margin,
        "threshold": threshold,
        "support_ratio": support_ratio,
        "top1_subsystem": top1,
        "top2_subsystem": top2,
        "subsystem_pair": subsystem_pair,
        "dominant_shap_span": dominant_span,
        "key_variables": key_variables,
        "fuzzy_top_variables": fuzzy_top[:8],
        "lstm_top_variables": lstm_top[:8],
        "shared_variables": shared_vars[:8],
        "recommendation": recommendation,
        "pcc_match": pcc_match,
        "subsystem_dominance": subsystem_info,
        "prediction": dict(pred),
    }
