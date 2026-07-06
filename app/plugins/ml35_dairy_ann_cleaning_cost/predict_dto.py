"""Pydantic DTOs for the ml35 dairy ANN /predict endpoint.

Three modes via discriminated union:
  - "inline":   predict consumo_agua_l given all 8 ANN features
  - "optimize": run GA to find optimal setpoints given 4 context inputs
  - "batch":    run inline prediction on each row of a CSV
"""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictBatchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Path to CSV with 8 ANN input features")
    mlflow_run_id: str = ""


class PredictBatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Single-sample prediction: provide all 8 ANN inputs."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None
    mlflow_run_id: str = ""

    temp_entrada_leche: float = Field(..., description="Temperatura de entrada de la leche (°C)")
    temp_ambiente: float = Field(..., description="Temperatura ambiente (°C)")
    temp_setpoint_leche: float = Field(..., description="Setpoint de temperatura de leche (°C)")
    temp_proceso_leche: float | None = Field(
        default=None, description="Temperatura de proceso real (°C); si no se aporta = temp_setpoint_leche"
    )
    temp_agua_servicio: float | None = Field(
        default=None, description="Temperatura del agua de servicio (°C); si no se aporta = temp_proceso_leche + 10"
    )
    flujo_leche_lh: float = Field(..., description="Caudal de leche (L/h)")
    horas_desde_limpieza: float = Field(..., description="Horas desde la última limpieza (h)")
    presion_diferencial_bar: float = Field(..., description="Presión diferencial en el intercambiador (bar)")


class PredictInlineResponse(BaseModel):
    """Inline prediction response: ANN water consumption + computed PU."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    consumo_agua_l: float = Field(..., description="Consumo de agua de refrigeración predicho (L)")
    pu_logrado: float = Field(..., description="Unidades de pasteurización (fórmula determinista, debe ser ≥ 13)")


class PredictOptimizeRequest(BaseModel):
    """Optimization mode: provide the 4 fixed-context inputs; the GA finds optimal setpoints."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["optimize"] = "optimize"
    # model_key="optimize" tells predict_inline to dispatch to the GA branch
    model_key: str = "optimize"
    threshold: float | None = None
    mlflow_run_id: str = ""

    temp_entrada_leche: float = Field(..., description="Temperatura de entrada de la leche (°C)")
    temp_ambiente: float = Field(..., description="Temperatura ambiente (°C)")
    horas_desde_limpieza: float = Field(..., description="Horas desde la última limpieza (h)")
    presion_diferencial_bar: float = Field(..., description="Presión diferencial en el intercambiador (bar)")


class PredictOptimizeResponse(BaseModel):
    """Optimization response: recommended setpoints + water savings vs. baseline."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    opt_temp_leche: float = Field(..., description="Setpoint óptimo de temperatura leche (°C)")
    opt_temp_agua: float = Field(..., description="Temperatura óptima del agua de servicio (°C)")
    opt_flujo: float = Field(..., description="Caudal óptimo de leche (L/h)")
    consumo_estandar: float = Field(..., description="Consumo con setpoints estándar (L)")
    consumo_optimizado: float = Field(..., description="Consumo con setpoints optimizados (L)")
    ahorro_l: float = Field(..., description="Ahorro absoluto de agua (L)")
    ahorro_pct: float = Field(..., description="Porcentaje de reducción vs. baseline estándar (%)")
    pu_logrado: float = Field(..., description="PU de la solución óptima (debe ser ≥ 13)")


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest, PredictOptimizeRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse, PredictOptimizeResponse]
