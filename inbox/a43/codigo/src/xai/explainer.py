"""
Módulo principal XAI: orquestador de explicaciones.
Integra fuzzy, temporal, fusión y acciones correctivas.
"""

import numpy as np
from typing import Any, Dict, List, Optional

from .fuzzy_explainer import FuzzyExplainer
from .temporal_explainer import TemporalExplainer
from .fusion import fuse_fuzzy_temporal_reports
from .corrective_actions import generate_corrective_actions, DEFAULT_ACTION_CONFIG
from .utils import infer_semantic_mf_names_from_centers


class DNFLExplainer:
    """
    Orquestador principal para explicabilidad XAI.

    Combina:
      1. Explicaciones fuzzy (reglas, antecedentes)
      2. Explicaciones temporales (SHAP, patrones)
      3. Fusión de ambas ramas
      4. Generación de acciones correctivas
      5. Reporte final compacto
    """

    def __init__(
        self,
        model,
        feature_names_stats: List[str],
        feature_names_original: Optional[List[str]] = None,
        action_config: Optional[Dict[str, Any]] = None,
        corrective_actions_cfg: Optional[Dict[str, Any]] = None,
        stats_creation: Optional[List[str]] = None,
        model_cfg: Optional[Dict[str, Any]] = None,
    ):
        """
        Inicializa the explainer.

        Args:
            model: Modelo DNFL entrenado
            feature_names_stats: Nombres de features estadísticas (para fuzzy)
            feature_names_original: Nombres de features originales (para temporal)
            action_config: Configuración personalizada de acciones correctivas
        """
        self.model = model
        self.feature_names_stats = list(feature_names_stats)
        self.feature_names_original = feature_names_original or self.feature_names_stats
        self.action_config = action_config or DEFAULT_ACTION_CONFIG
        self.corrective_actions_cfg = corrective_actions_cfg or {}
        self.stats_creation = list(stats_creation) if stats_creation else None
        self.model_cfg = model_cfg or {}

        # Infer semantic names para fuzzy
        self.mf_semantics = None
        try:
            self.mf_semantics = infer_semantic_mf_names_from_centers(
                model,
                feature_names=self.feature_names_stats,
            )
        except Exception:
            pass

        # Inicializar sub-explainers
        self.fuzzy_explainer = FuzzyExplainer(
            model=model,
            feature_names=self.feature_names_stats,
            mf_semantics=self.mf_semantics,
            stats_creation=self.stats_creation,
        )

        self.temporal_explainer = TemporalExplainer(
            model=model,
            feature_names=self.feature_names_original,
        )

    def explain(
        self,
        x_window: np.ndarray,
        s_stats: np.ndarray,
        background_windows: np.ndarray,
        anomaly_threshold: float = 0.5,
        n_background: int = 64,
        random_state: int = 42,
        top_rules: int = 5,
        top_variables: int = 8,
        max_antecedents_per_rule: int = 6,
        min_rule_activation: float = 0.0,
        top_blocks: int = 3,
        min_block_score: float = 0.10,
        include_raw: bool = False,
    ) -> Dict[str, Any]:
        """
        Explicación completa de una muestra.

        Args:
            x_window: Ventana temporal [T, F]
            s_stats: Estadísticas de ventana [F_stats]
            background_windows: Ventanas de background para SHAP [N, T, F]
            anomaly_threshold: Umbral de decisión
            n_background: Número de samples de background para SHAP
            random_state: Seed para reproducibilidad
            top_rules: K reglas fuzzy principales
            top_variables: K variables temporales principales
            max_antecedents_per_rule: Antecedentes máximos por regla
            min_rule_activation: Activación mínima de regla
            top_blocks: Bloques de acción principales
            min_block_score: Score mínimo de bloque
            include_raw: Si True, incluye arrays y trazas internas pesadas

        Returns:
            Reporte final con predicción, explicaciones y acciones
        """

        # 1. Explicación fuzzy
        fuzzy_report = self.fuzzy_explainer.explain_sample(
            x_window=x_window,
            s_stats=s_stats,
            top_rules=top_rules,
            max_antecedents_per_rule=max_antecedents_per_rule,
            min_rule_activation=min_rule_activation,
            min_specialization=0.0,
            anomaly_threshold=anomaly_threshold,
            include_raw=include_raw,
        )

        # 2. Explicación temporal
        temporal_report = self.temporal_explainer.explain_sample(
            x_window=x_window,
            background_windows=background_windows,
            decision_threshold=anomaly_threshold,
            n_background=n_background,
            random_state=random_state,
            top_variables=top_variables,
            include_raw=include_raw,
        )

        # 3. Fusión
        fusion_report = fuse_fuzzy_temporal_reports(
            fuzzy_report,
            temporal_report,
            model_cfg=self.model_cfg,
        )

        # 4. Acciones correctivas
        action_report = generate_corrective_actions(
            fusion_report=fusion_report,
            action_config=self.action_config,
            corrective_actions_cfg=self.corrective_actions_cfg,
            top_blocks=top_blocks,
            min_block_score=min_block_score,
        )

        # 5. Reporte final
        final_report = self._build_final_report(
            action_report=action_report,
        )

        prediction = dict(action_report.get("prediction", {}))

        return {
            "prediction": {
                "predicted_class": str(action_report.get("state_label", "unconfirmed_alert")),
                "anomaly_probability": float(prediction.get("ensemble", {}).get("probability", 0.0)),
                "fuzzy_probability": float(prediction.get("fuzzy", {}).get("probability", 0.0)),
                "temporal_probability": float(prediction.get("temporal", {}).get("probability", 0.0)),
            },
            "explanation": {
                "fuzzy": fuzzy_report,
                "temporal": temporal_report,
                "fusion": fusion_report,
            },
            "corrective_actions": action_report,
            "final_report": final_report,
        }

    def _build_final_report(
        self,
        action_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Construye reporte final interpretativo alineado con la salida acordada."""
        state_map = {
            "normal": "Normal",
            "normal_with_signals": "Normal con señales",
            "unconfirmed_alert": "Alerta no confirmada",
            "confirmed_anomaly": "Anomalía confirmada",
        }

        dominant_branch_map = {
            "both": "Ambas ramas",
            "temporal": "Rama temporal",
            "fuzzy": "Rama neuro-difusa",
            "none": "Ninguna rama dominante",
        }

        prediction = dict(action_report.get("prediction", {}))
        anomaly_prob = float(prediction.get("ensemble", {}).get("probability"))
        threshold = float(prediction.get("decision_threshold"))
        margin = float(anomaly_prob - threshold)

        state_label = str(action_report.get("state_label"))
        dominant_branch = str(action_report.get("dominant_branch"))

        ranked_blocks = action_report.get("ranked_blocks", [])
        top_blocks = [
            str(row.get("block", ""))
            for row in ranked_blocks
            if str(row.get("block", "")).strip()
        ]

        key_vars = []
        seen_vars = set()
        for row in ranked_blocks:
            for var in row.get("support_variables", []):
                value = str(var)
                if value and value not in seen_vars:
                    seen_vars.add(value)
                    key_vars.append(value)
                if len(key_vars) >= 3:
                    break
            if len(key_vars) >= 3:
                break

        suggested_actions = [
            str(item.get("action", ""))
            for item in action_report.get("suggested_actions", [])[:6]
            if str(item.get("action", "")).strip()
        ]

        estado_del_sistema = {
            "Estado_interpretativo": state_map.get(state_label, state_label),
            "Probabilidad_anomalia": round(float(anomaly_prob), 3),
            "Umbral_decision": round(float(threshold), 2),
        }

        if state_label != "normal":
            estado_del_sistema["Margen_umbral"] = round(float(margin), 3)

        if state_label == "unconfirmed_alert":
            estado_del_sistema["Evidencia_dominante"] = dominant_branch_map.get(dominant_branch, dominant_branch)

        final_report = {
            "Estado_del_sistema": estado_del_sistema,
        }

        if state_label == "normal":
            final_report["Mensaje_operativo"] = [
                "No se requieren acciones correctivas.",
                "Mantener monitorización ordinaria.",
            ]
            return final_report

        final_report["Bloques_principales"] = top_blocks
        final_report["Variables_clave"] = key_vars
        final_report["Acciones_sugeridas"] = suggested_actions
        return final_report
