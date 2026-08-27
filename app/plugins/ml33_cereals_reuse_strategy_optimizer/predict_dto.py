"""Pydantic DTOs for the ml33 cereal reuse-strategy MILP optimizer /predict endpoint.

Two modes via discriminated union on ``mode``, matching the platform/orchestrator flow
which only ever sends ``inline`` or ``batch``:
  - "inline": a list of lots (>=1) processed in blocks of ``lots_per_day`` -> a
              per-lot assignment plus an aggregate summary.
  - "batch":  a CSV with the required lot columns -> the same, over the whole file.

Capacity/lots_per_day/fallback_strategy fields are execution parameters (not lot
attributes) and default to the values validated in the original delivery's
config/pipeline_config.json ("base validated inference regime" per its README.md).
"""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from app.plugins.ml33_cereals_reuse_strategy_optimizer.constants import (
    DEFAULT_ANIMAL_FEED_CAPACITY_T,
    DEFAULT_BIOCHAR_CAPACITY_T,
    DEFAULT_BIOMASS_COMBUSTION_CAPACITY_T,
    DEFAULT_COMPOSTING_CAPACITY_T,
    DEFAULT_FALLBACK_STRATEGY,
    DEFAULT_LOTS_PER_DAY,
)


# ── shared lot / execution-parameter fields ────────────────────────────────────

class LotInput(BaseModel):
    """One subproduct lot to assign to a reuse strategy."""

    model_config = ConfigDict(str_strip_whitespace=True)

    generated_volume_tons: float = Field(gt=0, description="Volumen del lote (t)")
    moisture_pct: float = Field(ge=0, description="Humedad del lote (%)")
    subproduct_type: str = Field(..., description="Categoría del subproducto (p.ej. Husk, Straw, Silo dust, Bran)")
    season: str = Field(..., description="Estación (p.ej. Dry, Rainy) — validada pero no usada por la fórmula de emisiones, ver docs")
    process_temperature_c: float | None = Field(
        default=None,
        description=(
            "Aceptado por compatibilidad de esquema con el delivery original; NUNCA usado "
            "como entrada de decisión — la temperatura se deriva de la estrategia elegida."
        ),
    )


class _ExecutionParams(BaseModel):
    """Capacity/runtime parameters shared by inline and batch requests."""

    model_config = ConfigDict(protected_namespaces=())

    lots_per_day: int = Field(default=DEFAULT_LOTS_PER_DAY, gt=0, description="Lotes antes de resetear capacidades (bloque operativo)")
    animal_feed_capacity: float = Field(default=DEFAULT_ANIMAL_FEED_CAPACITY_T, gt=0, description="Capacidad diaria 'Animal feed' (t)")
    composting_capacity: float = Field(default=DEFAULT_COMPOSTING_CAPACITY_T, gt=0, description="Capacidad diaria 'Composting' (t)")
    biochar_capacity: float = Field(default=DEFAULT_BIOCHAR_CAPACITY_T, gt=0, description="Capacidad diaria 'Biochar' (t)")
    biomass_combustion_capacity: float = Field(
        default=DEFAULT_BIOMASS_COMBUSTION_CAPACITY_T, gt=0, description="Capacidad diaria 'Biomass combustion' (t)"
    )
    fallback_strategy: str = Field(default=DEFAULT_FALLBACK_STRATEGY, description="Estrategia de guarda si el MILP no puede colocar un lote")


class LotResult(BaseModel):
    """Assignment result for a single lot."""

    model_config = ConfigDict(protected_namespaces=())

    row: int
    ai_assigned_strategy: str
    ai_assignment_source: str = Field(..., description="exact_min_emissions | capacity_fallback")
    ai_is_fallback: bool
    estimated_emissions_kg: float


# ── inline ──────────────────────────────────────────────────────────────────

class PredictInlineRequest(_ExecutionParams):
    """Inline request: a list of lots (>=1), processed in blocks of ``lots_per_day``."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["inline"] = "inline"
    model_key: str | None = None
    threshold: float | None = None
    mlflow_run_id: str = ""

    lots: list[LotInput] = Field(..., min_length=1, description="Lotes a asignar, en orden de llegada")


class PredictInlineResponse(BaseModel):
    """Per-lot assignments plus an aggregate summary."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    results: list[LotResult]
    distribution: dict[str, Any] = Field(..., description="{counts, percentages} por estrategia asignada")
    capacity_fallback_count: int
    total_estimated_emissions_kg: float


# ── batch ───────────────────────────────────────────────────────────────────

class PredictBatchRequest(BaseModel):
    """Batch request: a CSV with the required lot columns, one lot per row.

    Deliberately just {mode, data_path, mlflow_run_id} — matches every other plugin's
    batch contract in this repo. PredictModelUseCase.execute() only ever forwards
    data_path/mlflow_run_id (+model_key for the one plugin that declares it) to
    predict_batch, so execution-parameter overrides (capacities/lots_per_day/
    fallback_strategy) would never reach the plugin if declared here — batch mode
    always runs with the manifest's default capacity regime. Use inline mode to
    override them.
    """

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Path a un CSV con generated_volume_tons, moisture_pct, subproduct_type, season")
    mlflow_run_id: str = ""


class PredictBatchResponse(BaseModel):
    """Per-row assignments plus an aggregate summary, mirroring the inline response."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    n_rows: int
    predictions: list[dict[str, Any]]
    distribution: dict[str, Any]
    capacity_fallback_count: int
    total_estimated_emissions_kg: float
    output_path: str | None = None


PredictRequest = Annotated[
    Union[PredictInlineRequest, PredictBatchRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictInlineResponse, PredictBatchResponse]
