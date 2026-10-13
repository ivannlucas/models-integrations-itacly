"""Pydantic DTOs for the ml36 dairy DNL CO2-emissions optimizer /predict endpoint.

Same transport contract as ml34: two modes (discriminated union on ``mode``),
``inline`` and ``batch``. The GA operation travels inside them, selected by
``model_key``:
  - model_key != "optimize" (default) → MLP digital twin: predicts (T_out,
    CO2_emissions) from the 8 base process variables (the 3 derived features
    are computed by the plugin).
  - model_key == "optimize" → recommends setpoints (T_serv, Delta_P,
    Regeneration_perc) minimising CO2 subject to T_out >= 72.5 °C. Only the 5
    non-controllable context variables are needed; the current setpoints are
    optional and, if given, add the savings versus current operation.
"""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTROL_FIELDS = ("T_serv", "Delta_P", "Regeneration_perc")


class PredictBatchRequest(BaseModel):
    """Batch request: CSV path, dispatched by ``model_key`` like inline."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(
        ...,
        description=(
            "Path to CSV — 8 base MLP features, or the 5 context columns when model_key='optimize' "
            "(optional per-row columns in optimize: seed, decision_mode, and the 3 current setpoints)"
        ),
    )
    model_key: str | None = Field(
        default=None,
        description="'optimize' runs the GA per row; omitted/other runs the MLP digital twin",
    )
    mlflow_run_id: str = ""


class PredictBatchResponse(BaseModel):
    """Batch response: one prediction dict per input row."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Single-sample request; ``predict_inline`` dispatches by ``model_key``."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None
    mlflow_run_id: str = ""

    # Non-controllable context — required by both operations.
    F_milk: float = Field(..., description="Caudal volumétrico de leche (L/h)")
    T_in: float = Field(..., description="Temperatura de entrada de la leche (°C)")
    Fat_perc: float = Field(..., description="% de grasa del lote de leche")
    Viscosity: float = Field(..., description="Viscosidad dinámica de la leche")
    t_ciclo: float = Field(..., description="Tiempo transcurrido desde el inicio del ciclo (min)")

    # Controllable variables — all three required for the MLP prediction; optional in
    # optimize (the GA decides them; if given, they are the current operation).
    T_serv: float | None = Field(default=None, description="Temperatura del líquido de calentamiento (°C)")
    Delta_P: float | None = Field(default=None, description="Presión de impulsión de las bombas")
    Regeneration_perc: float | None = Field(default=None, description="Eficiencia de regeneración térmica (%)")

    # Optimize-only knobs
    decision_mode: Literal["static", "adaptive", "hybrid"] = Field(
        default="hybrid", description="Solo optimize: static (política global), adaptive o hybrid",
    )
    seed: int = Field(
        default=43,
        description="Semilla del GA adaptativo (solo optimize). El original usa 42 + índice de fila (desde 1)",
    )

    @model_validator(mode="after")
    def _validate_controls(self) -> "PredictInlineRequest":
        """MLP path needs all 3 controls; optimize accepts all-or-none (current operation)."""
        given = [n for n in CONTROL_FIELDS if getattr(self, n) is not None]
        if self.model_key != "optimize":
            missing = [n for n in CONTROL_FIELDS if getattr(self, n) is None]
            if missing:
                raise ValueError(
                    f"En modo inline (MLP) las variables {missing} son obligatorias; "
                    f"para recomendar setpoints use model_key='optimize'."
                )
        elif given and len(given) != len(CONTROL_FIELDS):
            raise ValueError(
                f"En optimize, si se informa la operación actual deben venir las 3 variables "
                f"{list(CONTROL_FIELDS)}; recibidas: {given}."
            )
        return self


class PredictInlineResponse(BaseModel):
    """Inline MLP response in real units."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    T_out_pred: float = Field(..., description="Temperatura de salida de la leche predicha (°C)")
    CO2_emissions_pred: float = Field(..., description="Emisiones de CO2 predichas (kg)")


class PredictOptimizeResponse(BaseModel):
    """Optimization response: recommended setpoints + predicted outcome + savings."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    decision_mode: str = Field(..., description="static | adaptive | hybrid")
    used_adaptive_ga: bool = Field(..., description="True si se ejecutó el GA adaptativo")
    seed: int = Field(..., description="Semilla del GA adaptativo utilizada")
    recommended_T_serv: float = Field(..., description="Setpoint óptimo de T_serv (°C)")
    recommended_Delta_P: float = Field(..., description="Setpoint óptimo de Delta_P")
    recommended_Regeneration_perc: float = Field(..., description="Setpoint óptimo de regeneración (%)")
    recommended_CO2_pred: float = Field(..., description="CO2 predicho con los setpoints recomendados (kg)")
    recommended_T_out_pred: float = Field(..., description="T_out predicha con los setpoints recomendados (°C)")
    recommended_factible: bool = Field(..., description="True si recommended_T_out_pred >= 72.5 °C")
    policy_T_serv: float = Field(..., description="T_serv de la política global estática")
    policy_Delta_P: float = Field(..., description="Delta_P de la política global estática")
    policy_Regeneration_perc: float = Field(..., description="Regeneración de la política global estática")
    policy_CO2_pred: float = Field(..., description="CO2 predicho con la política global (kg)")
    policy_T_out_pred: float = Field(..., description="T_out predicha con la política global (°C)")
    co2_saving_vs_policy: float = Field(..., description="policy_CO2_pred - recommended_CO2_pred (kg)")
    current_CO2_pred: float | None = Field(default=None, description="CO2 predicho con la operación actual (kg)")
    current_T_out_pred: float | None = Field(default=None, description="T_out predicha con la operación actual (°C)")
    co2_saving_vs_current: float | None = Field(
        default=None, description="current_CO2_pred - recommended_CO2_pred (kg); solo si se aporta la operación actual",
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse, PredictOptimizeResponse]
