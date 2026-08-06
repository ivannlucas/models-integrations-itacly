"""
XAI (Explainable AI) - Capa de interpretabilidad para el sistema de detección de anomalías.

Módulos:
  - fuzzy_explainer: Explicaciones basadas en lógica fuzzy
  - temporal_explainer: Explicaciones basadas en análisis temporal LSTM+SHAP
  - fusion: Fusión de explicaciones fuzzy y temporal
  - pcc: Detección de puntos críticos de control (PCC)
  - explainer: Orquestador principal de XAI
"""

from .explainer import DNFLExplainer

__all__ = [
    "DNFLExplainer"
]
