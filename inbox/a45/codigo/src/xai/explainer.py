"""
Módulo principal XAI: orquestador de explicaciones.
Integra fuzzy, temporal, fusión y detección PCC.
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional

from .fuzzy_explainer import FuzzyExplainer
from .temporal_explainer import TemporalExplainer
from .fusion import fuse_fuzzy_lstm_reports 
from .pcc import (
    DEFAULT_MONITOR_POLICY,
    SUBSYSTEMS_CONFIG,
    PCC_CATALOG,
    build_pcc_report,
    build_pcc_catalog_from_records,
    normalize_pcc_catalog,
)
from .utils import infer_semantic_mf_names_from_centers


class DNFLExplainer:
    """
    Orquestador principal para explicabilidad XAI.

    Combina:
      1. Explicaciones fuzzy (reglas, antecedentes)
      2. Explicaciones temporales (SHAP, patrones)
      3. Fusión de ambas ramas
      4. Detección PCC y estado de monitorización
      5. Reporte final compacto
    """

    def __init__(
        self,
        model,
        feature_names_stats: List[str],
        feature_names_original: Optional[List[str]] = None,
        pcc_cfg: Optional[Dict[str, Any]] = None,
        stats_creation: Optional[List[str]] = None,
        model_cfg: Optional[Dict[str, Any]] = None,
    ):
        """
        Inicializa the explainer.

        Args:
            model: Modelo DNFL entrenado
            feature_names_stats: Nombres de features estadísticas (para fuzzy)
            feature_names_original: Nombres de features originales (para temporal)
            pcc_cfg: Configuración PCC (subsystems, catalog, monitor_policy)
        """
        self.model = model
        self.feature_names_stats = list(feature_names_stats)
        self.feature_names_original = feature_names_original or self.feature_names_stats
        self.pcc_cfg = pcc_cfg or {}
        self.subsystems_config = self.pcc_cfg.get("subsystems") or SUBSYSTEMS_CONFIG

        # Resolve catalog: support YAML list-of-records, dict-with-tuple-keys, or fallback
        _raw_catalog = self.pcc_cfg.get("catalog")
        if isinstance(_raw_catalog, list) and _raw_catalog:
            self.pcc_catalog = build_pcc_catalog_from_records(
                _raw_catalog, self.subsystems_config
            )
        elif isinstance(_raw_catalog, dict) and _raw_catalog:
            self.pcc_catalog = normalize_pcc_catalog(
                _raw_catalog, subsystems_config=self.subsystems_config
            )
        else:
            self.pcc_catalog = PCC_CATALOG

        self.monitor_policy = self.pcc_cfg.get("monitor_policy") or DEFAULT_MONITOR_POLICY
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

        Returns:
            Reporte final con predicción, explicaciones y PCC
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
        )

        # 2. Explicación temporal
        lstm_report = self.temporal_explainer.explain_sample(
            x_window=x_window,
            background_windows=background_windows,
            decision_threshold=anomaly_threshold,
            n_background=n_background,
            random_state=random_state,
            top_variables=top_variables,
            top_timesteps=8,
        )

        # 3. Fusión
        fusion_report = fuse_fuzzy_lstm_reports(
            fuzzy_report=fuzzy_report,
            lstm_report=lstm_report,
            model_cfg=self.model_cfg,
        )

        # 4. Detección PCC
        pcc_report = build_pcc_report(
            fuzzy_report=fuzzy_report,
            lstm_report=lstm_report,
            fusion_report=fusion_report,
            subsystems=self.subsystems_config,
            pcc_catalog=self.pcc_catalog,
            monitor_policy=self.monitor_policy,
        )

        # 5. Reporte final
        final_report = self._build_final_report(
            pcc_report=pcc_report,
        )

        prediction = dict(pcc_report.get("prediction", fusion_report.get("prediction", {})))
        predicted_state = str(pcc_report.get("state", "vigilancia"))

        return {
            "prediction": {
                "predicted_class": predicted_state,
                "anomaly_probability": float(prediction.get("ensemble_prob", 0.0)),
                "fuzzy_probability": float(prediction.get("fuzzy_prob", 0.0)),
                "temporal_probability": lstm_report["probability_dl"],
            },
            "explanation": {
                "fuzzy": fuzzy_report,
                "temporal": lstm_report,
                "fusion": fusion_report,
            },
            "pcc": pcc_report,
            "final_report": final_report,
        }

    def _build_final_report(
        self,
        pcc_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Construye reporte final orientado a detección PCC."""
      
        return {
                "Estado interpretativo": str(pcc_report.get("state")),
                "Evidencia": str(pcc_report.get("message")),
                "Probabilidad de anomalia": round(float(pcc_report.get("probability")), 2),
                "Umbral de detección de anomalias": round(float(pcc_report.get("threshold")), 2),
                "Margen respecto al umbral": round(float(pcc_report.get("margin")), 2),
                # "support_ratio": float(pcc_report.get("support_ratio", 0.0)),
                # "top1_subsystem": str(pcc_report.get("top1_subsystem", "unknown")),
                # "top2_subsystem": str(pcc_report.get("top2_subsystem", "none")),
                # "dominant_shap_span": str(pcc_report.get("dominant_shap_span", "unknown")),
                # "key_variables": list(pcc_report.get("key_variables", [])),
                "Recomendacion": str(pcc_report.get("recommendation", "Mantener monitorizacion ordinaria")),
            }