"""Pydantic request/response DTOs for the meat-traceability /predict endpoint."""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictBatchRequest(BaseModel):
    """Batch request: a CSV of traceability event records."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Path to CSV file with traceability event records")


class PredictBatchResponse(BaseModel):
    """Batch response: one scoring dict per row."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


class PredictInlineRequest(BaseModel):
    """Inline request: the 33 feature fields of a single traceability event."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None
    # numeric
    has_prev_event: float | None = None
    cold_start_lot: float | None = None
    stage_order_obs: float | None = None
    time_since_prev_lot_hours: float | None = None
    lot_stage_order_delta: float | None = None
    events_seen_before: float | None = None
    max_stage_order_seen_before: float | None = None
    stages_seen_count_before: float | None = None
    prior_sequence_anomaly_count: float | None = None
    sensor_temp_c: float | None = None
    sensor_ph: float | None = None
    sensor_weight_kg: float | None = None
    ts_hour: float | None = None
    ts_dayofweek: float | None = None
    sensor_temp_c_delta_from_prev: float | None = None
    sensor_ph_delta_from_prev: float | None = None
    sensor_weight_kg_delta_from_prev: float | None = None
    yield_pct_from_parent: float | None = None
    yield_delta_from_expected: float | None = None
    # categorical
    stage: str | None = None
    prev_stage: str | None = None
    plant_line: str | None = None
    operator_shift: str | None = None
    process_route: str | None = None
    packaging_type: str | None = None
    trace_unit_type: str | None = None
    prev_trace_unit_type: str | None = None
    temp_sensor_location: str | None = None
    prev_temp_sensor_location: str | None = None
    ph_measurement_source: str | None = None
    prev_ph_measurement_source: str | None = None
    cold_room_id: str | None = None
    scale_id: str | None = None


class PredictInlineResponse(BaseModel):
    """Inline response: predicted incident class, score and confidence."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    pred_traceability_incident: int = Field(..., description="0 (sin incidencia) o 1 (incidencia)")
    pred_score: float = Field(..., description="Probabilidad de incidencia (0.0–1.0)")
    confidence: float = Field(..., description="Probabilidad de la clase predicha")
    model_name: str
    xai_feature_values: dict[str, Any] | None = Field(
        default=None, description="Valores de features usados — consumido por el servicio XAI",
    )


PredictRequest = Annotated[
    Union[PredictBatchRequest, PredictInlineRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictBatchResponse, PredictInlineResponse]
