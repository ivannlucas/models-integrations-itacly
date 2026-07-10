"""Pydantic DTOs for the ml34 dairy pasteurization energy GA /predict endpoint.

Three modes via discriminated union:
  - "inline":   MLP surrogate prediction (E_consumo, T_out) from the 5 features
  - "optimize": single-objective GA recommends (F_flow, T_servicio) setpoints
                for a scenario given the 3 non-controllable inputs
  - "batch":    run inline prediction on each row of a CSV
"""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictBatchRequest(BaseModel):
    """Batch prediction request: CSV path with the 5 MLP features per row."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Path to CSV with the 5 MLP input features")
    mlflow_run_id: str = ""


class PredictBatchResponse(BaseModel):
    """Batch prediction response: one prediction dict per input row."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Single-sample MLP prediction: provide the 5 process features."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None
    mlflow_run_id: str = ""

    T_in_leche: float = Field(..., description="Temperatura de entrada de la leche cruda (°C)")
    F_flow: float = Field(..., description="Caudal volumétrico de leche (L/h)")
    T_servicio: float = Field(..., description="Temperatura del fluido de servicio (°C)")
    t_ciclo: float = Field(..., description="Tiempo desde la última limpieza CIP (min)")
    Delta_P: float = Field(..., description="Caída de presión en el intercambiador (bar)")


class PredictInlineResponse(BaseModel):
    """Inline prediction response: MLP surrogate outputs in real units."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    E_consumo_pred: float = Field(..., description="Consumo energético total predicho (kW)")
    T_out_pred: float = Field(..., description="Temperatura de salida de la leche predicha (°C)")


class PredictOptimizeRequest(BaseModel):
    """Optimization mode: provide the 3 non-controllable scenario inputs.

    The GA searches (F_flow, T_servicio) minimizing E_consumo/F_flow subject
    to T_out >= 72.3 °C. ``seed`` makes each scenario deterministic — the AI
    team's backtesting used seed = 1 + row_index over the test CSV.
    """

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["optimize"] = "optimize"
    # model_key="optimize" tells predict_inline to dispatch to the GA branch
    model_key: str = "optimize"
    threshold: float | None = None
    mlflow_run_id: str = ""

    T_in_leche: float = Field(..., description="Temperatura de entrada de la leche cruda (°C)")
    Delta_P: float = Field(..., description="Caída de presión en el intercambiador (bar)")
    t_ciclo: float = Field(..., description="Tiempo desde la última limpieza CIP (min)")
    seed: int = Field(default=1, description="Semilla del GA para reproducibilidad por escenario")


class PredictOptimizeResponse(BaseModel):
    """Optimization response: recommended setpoints + predicted outcome."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    IA_F_flow: float = Field(..., description="Setpoint óptimo de caudal (L/h)")
    IA_T_servicio: float = Field(..., description="Setpoint óptimo de T de servicio (°C)")
    IA_E_consumo: float = Field(..., description="Consumo predicho con setpoints IA (kW)")
    IA_T_out: float = Field(..., description="T de salida predicha con setpoints IA (°C)")
    IA_consumo_especifico: float = Field(..., description="IA_E_consumo/IA_F_flow (kW/(L/h))")
    IA_factible: bool = Field(..., description="True si IA_T_out >= 72.3 °C")
    fitness_final: float = Field(..., description="Fitness del mejor individuo al final del GA")
    seed: int = Field(..., description="Semilla del GA utilizada")


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest, PredictOptimizeRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse, PredictOptimizeResponse]
