"""Pydantic DTOs for the ml34 dairy pasteurization energy GA /predict endpoint.

The transport contract exposes only two modes (discriminated union on ``mode``),
matching the platform/orchestrator flow which only ever sends ``inline`` or
``batch``:
  - "inline":  single-sample request. The plugin's ``predict_inline`` then
               dispatches by ``model_key``:
                 * model_key != "optimize" (default) → MLP surrogate prediction
                   (E_consumo, T_out) from the 5 process features.
                 * model_key == "optimize" → single-objective GA recommends
                   (F_flow, T_servicio) setpoints for the scenario given the 3
                   non-controllable inputs (+ optional seed).
  - "batch":   run the MLP inline prediction on each row of a CSV.

Keeping the "optimize" operation inside the "inline" mode (differentiated by
model_key) means the whole upstream flow stays on inline/batch and the
GA-vs-MLP decision lives here, in the model container.
"""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    """Single-sample request. ``predict_inline`` dispatches by ``model_key``:

      - model_key != "optimize" (default): MLP surrogate → needs all 5 features.
      - model_key == "optimize": single-objective GA recommends (F_flow,
        T_servicio); only the 3 non-controllable inputs + optional ``seed`` are
        needed, so ``F_flow``/``T_servicio`` are optional at the schema level and
        enforced for the MLP path by the validator below.
    """

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    # model_key="optimize" tells predict_inline to dispatch to the GA branch;
    # any other value (or None) runs the MLP surrogate prediction.
    model_key: str | None = None
    threshold: float | None = None
    mlflow_run_id: str = ""

    # Non-controllable scenario inputs — required by both operations.
    T_in_leche: float = Field(..., description="Temperatura de entrada de la leche cruda (°C)")
    t_ciclo: float = Field(..., description="Tiempo desde la última limpieza CIP (min)")
    Delta_P: float = Field(..., description="Caída de presión en el intercambiador (bar)")

    # Controllable setpoints — required for the MLP inline prediction; in optimize
    # mode the GA chooses them, so they are optional here (see validator).
    F_flow: float | None = Field(
        default=None,
        description="Caudal volumétrico de leche (L/h) — requerido en inline MLP; lo decide el GA en optimize",
    )
    T_servicio: float | None = Field(
        default=None,
        description="Temperatura del fluido de servicio (°C) — requerido en inline MLP; lo decide el GA en optimize",
    )

    # GA reproducibility — used only when model_key == "optimize".
    seed: int = Field(default=1, description="Semilla del GA para reproducibilidad por escenario (solo optimize)")

    @model_validator(mode="after")
    def _require_setpoints_for_mlp(self) -> "PredictInlineRequest":
        """MLP inline needs the two setpoints; the GA optimize path does not."""
        if self.model_key != "optimize":
            missing = [n for n, v in (("F_flow", self.F_flow), ("T_servicio", self.T_servicio)) if v is None]
            if missing:
                raise ValueError(
                    f"En modo inline (MLP) los setpoints {missing} son obligatorios; "
                    f"para recomendar setpoints use model_key='optimize'."
                )
        return self


class PredictInlineResponse(BaseModel):
    """Inline prediction response: MLP surrogate outputs in real units."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    E_consumo_pred: float = Field(..., description="Consumo energético total predicho (kW)")
    T_out_pred: float = Field(..., description="Temperatura de salida de la leche predicha (°C)")


class PredictOptimizeResponse(BaseModel):
    """Optimization response: recommended setpoints + predicted outcome.

    Returned by ``predict_inline`` when ``model_key == "optimize"``. There is no
    separate optimize *request* type — the operation travels as an inline request
    with model_key="optimize".
    """

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
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse, PredictOptimizeResponse]
