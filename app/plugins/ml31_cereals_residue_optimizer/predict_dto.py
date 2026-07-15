"""Pydantic DTOs for the ml31 cereal residue-optimizer /predict endpoint (v2.0 LP).

Three modes via discriminated union on ``mode``:
  - "optimize": solve a single LP scenario (minimize_residue / maximize_benefit)
                and return the optimal crop allocation + impact vs baseline.
  - "pareto":   trace the residue-vs-benefit Pareto frontier (epsilon-constraint).
  - "batch":    optimize each scenario row of a CSV.

The optimization objective (minimize_residue|maximize_benefit) is carried in the
``optimization_mode`` field to avoid clashing with the ``mode`` discriminator.
``model_key`` routes predict_inline to the optimize/pareto branch (batch is
dispatched to predict_batch by the generic use case).
"""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


# ── optimize ────────────────────────────────────────────────────────────────
class PredictOptimizeRequest(BaseModel):
    """Single-scenario LP optimization request. All fields have manifest defaults."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["optimize"] = "optimize"
    model_key: str = "optimize"  # routes predict_inline to the optimize branch
    threshold: float | None = None
    mlflow_run_id: str = ""

    reference_year: int = Field(default=2023, description="Año de referencia (superficies históricas)")
    optimization_mode: Literal["minimize_residue", "maximize_benefit"] = Field(
        default="minimize_residue",
        description="Objetivo del LP: minimize_residue (Modo A) o maximize_benefit (Modo B)",
    )
    climate_factor: float = Field(default=1.0, description="Multiplicador sobre los rendimientos")
    expected_spring_rain_mm: float = Field(
        default=130.0, description="Lluvia de primavera esperada (mm); <100 activa estrés hídrico"
    )
    min_benefit_eur: float | None = Field(
        default=None, description="Beneficio mínimo exigido (R6, Modo A). Null => se deriva del baseline"
    )
    min_benefit_pct_of_baseline: float = Field(
        default=1.0, description="% del beneficio baseline a preservar cuando min_benefit_eur es null"
    )
    max_residue_t: float | None = Field(
        default=None, description="Residuo máximo admisible (R7, Modo B)"
    )
    surface_tolerance_pct: float = Field(
        default=25.0, description="Banda ±% sobre la superficie histórica de cada cultivo (R4c)"
    )
    min_secano_use_pct: float = Field(default=95.0, description="Uso mínimo de secano (R1b)")
    min_regadio_use_pct: float = Field(default=95.0, description="Uso mínimo de regadío (R2b)")
    crop_constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Restricciones por cultivo (min_production_t/min_surface_pct/max_surface_pct); clave '_default' como fallback",
    )
    price_overrides: dict[str, float] = Field(
        default_factory=dict, description="Sobrescribe precio de grano por cultivo (EUR/kg)"
    )
    cost_overrides: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Sobrescribe costes secano/regadío por cultivo (EUR/ha)"
    )
    total_secano_ha: float | None = Field(
        default=None, description="Superficie total de secano disponible (override; null => año de referencia)"
    )
    total_regadio_ha: float | None = Field(
        default=None, description="Superficie total de regadío disponible (override; null => año de referencia)"
    )


class PredictOptimizeResponse(BaseModel):
    """Optimal crop-allocation plan plus impact metrics vs the historical baseline."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    reference_year: int
    optimization_mode: str
    crop_allocation: dict[str, dict[str, float]] = Field(
        ..., description="Por cultivo: {secano_ha, regadio_ha, production_t, residue_t, benefit_eur}"
    )
    total_production_t: float
    total_residue_t: float
    total_benefit_eur: float
    baseline_total_production_t: float
    baseline_total_residue_t: float
    baseline_total_benefit_eur: float
    residue_reduction_pct: float = Field(
        ..., description="(baseline-óptimo)/baseline*100: POSITIVO indica reducción de residuo"
    )
    benefit_change_eur: float
    benefit_change_pct: float
    production_change_pct: float
    solver_status: str = Field(..., description="Estado CBC: OPTIMAL | INFEASIBLE | UNBOUNDED | NOT_SOLVED")
    solve_time_seconds: float
    verdict: str = Field(..., description="PASADO/FALLADO según criterios de éxito objetivos")


# ── pareto ──────────────────────────────────────────────────────────────────
class PredictParetoRequest(BaseModel):
    """Pareto-frontier request (residue vs benefit via epsilon-constraint)."""

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["pareto"] = "pareto"
    model_key: str = "pareto"  # routes predict_inline to the pareto branch
    threshold: float | None = None
    mlflow_run_id: str = ""

    reference_year: int = Field(default=2023, description="Año de referencia")
    num_points: int = Field(default=20, description="Nº de puntos del barrido de beneficio")
    benefit_range_pct_of_max: list[float] = Field(
        default=[0.50, 1.00], description="Rango [low, high] como fracción del beneficio máximo"
    )
    climate_factor: float = 1.0
    expected_spring_rain_mm: float = 130.0
    surface_tolerance_pct: float = 25.0
    min_secano_use_pct: float = 95.0
    min_regadio_use_pct: float = 95.0
    price_overrides: dict[str, float] = Field(default_factory=dict)
    cost_overrides: dict[str, dict[str, float]] = Field(default_factory=dict)


class PredictParetoResponse(BaseModel):
    """Pareto frontier: bounds, non-dominated points (in millions) and knee point."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    reference_year: int
    bounds: dict[str, float] = Field(
        ..., description="{min_residue_t, benefit_at_min_res_eur, max_benefit_eur, residue_at_max_ben_t}"
    )
    pareto_points: list[dict[str, Any]] = Field(
        ..., description="Puntos no dominados: {benefit_eur_M, residue_t_M, production_t_M, is_knee}"
    )
    knee_point: dict[str, float] | None = Field(
        default=None, description="Mejor compromiso: {benefit_eur, residue_t, production_t} (absoluto)"
    )
    num_sweep_points: int
    num_pareto_points: int


# ── batch ───────────────────────────────────────────────────────────────────
class PredictBatchRequest(BaseModel):
    """Batch request: a CSV where each row is an optimization scenario.

    Recognised columns (all optional; defaults from the optimize contract):
    reference_year, optimization_mode, climate_factor, expected_spring_rain_mm,
    min_benefit_eur, min_benefit_pct_of_baseline, max_residue_t,
    surface_tolerance_pct, min_secano_use_pct, min_regadio_use_pct.
    """

    model_config = ConfigDict(protected_namespaces=())
    mode: Literal["batch"] = "batch"
    data_path: str = Field(..., description="Path to CSV with one optimization scenario per row")
    mlflow_run_id: str = ""


class PredictBatchResponse(BaseModel):
    """Batch response: one optimization summary dict per scenario row."""

    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    predictions: list[dict[str, Any]]
    output_path: str | None = None


PredictRequest = Annotated[
    Union[PredictOptimizeRequest, PredictParetoRequest, PredictBatchRequest],
    Field(discriminator="mode"),
]

PredictResponse = Union[PredictOptimizeResponse, PredictParetoResponse, PredictBatchResponse]
