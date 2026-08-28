"""Ml33CerealsReuseStrategyOptimizerPlugin — deterministic MILP reuse-strategy optimizer.

Deployed decision engine for cereal-byproduct reuse-strategy assignment. Solves each
capacity-reset operational block (``lots_per_day`` lots) to proven optimality as a small
mixed-integer linear program (``scipy.optimize.milp``, HiGHS backend), minimizing
simulated CO2 emissions. No trained weights, no artifact, no random seed — the same
input always produces the same output. See ``optimizer.py`` for the ported solver
(originally ``src/predict/exact_optimizer.py`` in the model delivery,
``inbox/a33``, originally delivered as
``a33-cnp-cereals-neuroevolutivo-reduccion-ambiental-residuos`` — see manifest.yaml).

Modes:
  - inline (predict_inline): a list of >=1 lots -> per-lot assignment + summary.
  - batch  (predict_batch):  a CSV with the 4 required columns -> the same, over the
                              whole file (always under the default capacity regime —
                              see predict_dto.PredictBatchRequest).

train() is not supported: this is an exact combinatorial solver, not a learned model.
NEAT (retained by the AI team only as a benchmark, never the deployed engine — see the
inbox manifest's known_issues) is intentionally not exposed by this plugin.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError, TrainingNotSupportedError
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml33_cereals_reuse_strategy_optimizer.constants import (
    DEFAULT_ANIMAL_FEED_CAPACITY_T,
    DEFAULT_BIOCHAR_CAPACITY_T,
    DEFAULT_BIOMASS_COMBUSTION_CAPACITY_T,
    DEFAULT_COMPOSTING_CAPACITY_T,
    DEFAULT_FALLBACK_STRATEGY,
    DEFAULT_LOTS_PER_DAY,
    FRAMEWORK,
    MODEL_ID,
    VERSION,
)
from app.plugins.ml33_cereals_reuse_strategy_optimizer.mlflow_utils import (
    download_user_model_from_mlflow,
)
from app.plugins.ml33_cereals_reuse_strategy_optimizer.model_loader import load_artifacts
from app.plugins.ml33_cereals_reuse_strategy_optimizer.optimizer import (
    ExactEmissionsOptimizer,
    summarize_strategy_distribution,
)
from app.plugins.ml33_cereals_reuse_strategy_optimizer.predict_dto import (
    LotResult,
    PredictBatchResponse,
    PredictInlineResponse,
)

logger = logging.getLogger(__name__)


class Ml33CerealsReuseStrategyOptimizerPlugin(ModelPluginPort):
    """Deterministic MILP optimizer for cereal reuse-strategy assignment."""

    def __init__(self) -> None:
        self._loaded = False
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        self._loaded = load_artifacts()
        logger.info("Ml33CerealsReuseStrategyOptimizerPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        return self._loaded

    def _require_loaded(self) -> None:
        if not self.is_loaded():
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    @staticmethod
    def _build_engine(features: dict) -> ExactEmissionsOptimizer:
        """Assemble the exact optimizer from request execution parameters."""
        capacities = {
            "Animal feed": float(features.get("animal_feed_capacity", DEFAULT_ANIMAL_FEED_CAPACITY_T)),
            "Composting": float(features.get("composting_capacity", DEFAULT_COMPOSTING_CAPACITY_T)),
            "Biochar": float(features.get("biochar_capacity", DEFAULT_BIOCHAR_CAPACITY_T)),
            "Biomass combustion": float(
                features.get("biomass_combustion_capacity", DEFAULT_BIOMASS_COMBUSTION_CAPACITY_T)
            ),
        }
        return ExactEmissionsOptimizer(
            capacities=capacities,
            lots_per_day=int(features.get("lots_per_day", DEFAULT_LOTS_PER_DAY)),
            fallback_strategy=features.get("fallback_strategy", DEFAULT_FALLBACK_STRATEGY),
            logger_=logger,
        )

    @staticmethod
    def _trace_to_results(df: pd.DataFrame, trace_df: pd.DataFrame) -> list[LotResult]:
        """Zip the traceability columns back into typed per-lot results."""
        _ = df
        return [
            LotResult(
                row=int(idx),
                ai_assigned_strategy=str(row["ai_assigned_strategy"]),
                ai_assignment_source=str(row["ai_assignment_source"]),
                ai_is_fallback=bool(row["ai_is_fallback"]),
                estimated_emissions_kg=float(row["estimated_emissions_kg"]),
                candidates=row["candidate_emissions"],
            )
            for idx, row in trace_df.iterrows()
        ]

    # ── predict_inline ──────────────────────────────────────────────────────
    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        _ = model_key, threshold
        self._require_loaded()
        if mlflow_run_id:
            download_user_model_from_mlflow(mlflow_run_id)  # always None; logs a warning

        lots = features.get("lots") or []
        df = pd.DataFrame(lots)
        engine = self._build_engine(features)
        trace_df = engine.infer_dataframe(df)

        results = self._trace_to_results(df, trace_df)
        distribution = summarize_strategy_distribution(trace_df["ai_assigned_strategy"])
        self._record()
        return PredictInlineResponse(
            model_id=MODEL_ID,
            results=results,
            distribution=distribution,
            capacity_fallback_count=engine.capacity_fallback_count,
            total_estimated_emissions_kg=round(float(trace_df["estimated_emissions_kg"].sum()), 4),
        )

    # ── predict_batch ───────────────────────────────────────────────────────
    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        self._require_loaded()
        if mlflow_run_id:
            download_user_model_from_mlflow(mlflow_run_id)  # always None; logs a warning

        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)

        # Batch mode always uses the default capacity regime — see PredictBatchRequest
        # docstring (the generic predict use case never forwards execution-parameter
        # overrides to predict_batch). Use inline mode to override capacities.
        engine = self._build_engine({})
        trace_df = engine.infer_dataframe(df)

        predictions = []
        for idx, row in trace_df.iterrows():
            row_out = {
                "row": int(idx),
                "ai_assigned_strategy": row["ai_assigned_strategy"],
                "ai_assignment_source": row["ai_assignment_source"],
                "ai_is_fallback": bool(row["ai_is_fallback"]),
                "estimated_emissions_kg": float(row["estimated_emissions_kg"]),
            }
            # "candidates" (why this strategy) is only surfaced for row 0 of a batch —
            # the platform's XAI panel only ever explains one row at a time, and
            # repeating it for every one of up to 10k rows would needlessly bloat an
            # already large batch response (mirrors ml31's own predict_batch, which
            # applies the same idx==0 restriction to its own explanation field).
            if idx == 0:
                row_out["candidates"] = row["candidate_emissions"]
            predictions.append(row_out)
        distribution = summarize_strategy_distribution(trace_df["ai_assigned_strategy"])
        self._record()
        logger.info("predict_batch done — %d lots", len(predictions))
        return PredictBatchResponse(
            model_id=MODEL_ID,
            n_rows=len(predictions),
            predictions=predictions,
            distribution=distribution,
            capacity_fallback_count=engine.capacity_fallback_count,
            total_estimated_emissions_kg=round(float(trace_df["estimated_emissions_kg"].sum()), 4),
            output_path=None,
        )

    # ── train (not supported) ──────────────────────────────────────────────
    def train(self, *, data_path: str, mlflow_run_id: str = "") -> Any:
        """Training is not supported: this is a deterministic MILP optimizer with no
        learned weights (HTTP 501)."""
        _ = data_path, mlflow_run_id
        raise TrainingNotSupportedError(
            "Este modelo es un optimizador MILP determinista (asignación exacta de "
            "lotes a estrategia de reuso); no tiene pesos entrenables ni soporta "
            "reentrenamiento por usuario."
        )

    # ── stats ───────────────────────────────────────────────────────────────
    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        if mlflow_run_id:
            logger.warning(
                "mlflow_run_id=%s provided but model '%s' is a deterministic MILP optimizer "
                "(no user training)", mlflow_run_id, MODEL_ID,
            )
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Optimizador MILP determinista (scipy.optimize.milp, HiGHS) para la asignación "
                "de lotes de subproducto cerealista a estrategia de reuso (Animal feed / "
                "Composting / Biochar / Biomass combustion), bajo capacidades de planta que se "
                "resetean cada lots_per_day lotes. Minimiza las emisiones de CO2 simuladas de "
                "cada bloque a optimalidad probada. Sin pesos entrenados ni semilla aleatoria. "
                "NEAT se retiene en el delivery original solo como benchmark de referencia "
                "(no desplegado en este plugin)."
            ),
            task_type="optimization",
            framework=FRAMEWORK,
            inputs=[
                InputField(name="generated_volume_tons", type="float", description="Volumen del lote (t) [inline/batch]"),
                InputField(name="moisture_pct", type="float", description="Humedad del lote (%) [inline/batch]"),
                InputField(name="subproduct_type", type="str", description="Categoría del subproducto [inline/batch]"),
                InputField(name="season", type="str", description="Estación; validada pero no usada por la fórmula de emisiones [inline/batch]"),
                InputField(name="lots_per_day", type="int", default=DEFAULT_LOTS_PER_DAY,
                           description="Lotes antes de resetear capacidades [inline]"),
                InputField(name="animal_feed_capacity", type="float", default=DEFAULT_ANIMAL_FEED_CAPACITY_T,
                           description="Capacidad diaria 'Animal feed' (t) [inline]"),
                InputField(name="composting_capacity", type="float", default=DEFAULT_COMPOSTING_CAPACITY_T,
                           description="Capacidad diaria 'Composting' (t) [inline]"),
                InputField(name="biochar_capacity", type="float", default=DEFAULT_BIOCHAR_CAPACITY_T,
                           description="Capacidad diaria 'Biochar' (t) [inline]"),
                InputField(name="biomass_combustion_capacity", type="float", default=DEFAULT_BIOMASS_COMBUSTION_CAPACITY_T,
                           description="Capacidad diaria 'Biomass combustion' (t) [inline]"),
                InputField(name="fallback_strategy", type="str", default=DEFAULT_FALLBACK_STRATEGY,
                           description="Estrategia de guarda si el MILP no puede colocar un lote [inline]"),
            ],
            outputs=[
                OutputField(name="ai_assigned_strategy", type="str", description="Estrategia de reutilización asignada al lote"),
                OutputField(name="ai_assignment_source", type="str", description="exact_min_emissions | capacity_fallback"),
                OutputField(name="ai_is_fallback", type="bool", description="True si se usó la guarda defensiva en vez del MILP"),
                OutputField(name="estimated_emissions_kg", type="float", description="Emisión de CO2 estimada del lote bajo la estrategia asignada"),
                OutputField(name="candidates", type="list",
                            description="Emisión estimada de las 4 estrategias candidatas (exacta, no SHAP) — explica por qué se eligió ai_assigned_strategy [inline; solo fila 0 en batch]"),
                OutputField(name="distribution", type="dict", description="Conteo/% por estrategia asignada (agregado)"),
                OutputField(name="capacity_fallback_count", type="int", description="Nº de lotes resueltos por guarda defensiva"),
            ],
            metrics={
                # models/metrics/inference_evaluation_report.json del delivery — split de
                # test completo (10000 filas), capacidades por defecto (ver manifest).
                "n_rows_test": 10000,
                "total_emissions_reduction_pct": 27.63,
                "ai_estimated_total_emissions_kg": 30893908.4,
                "baseline_total_emissions_kg": 42691705.58,
                "distribution_ai_pct_normalized": {
                    "composting": 37.18, "animal_feed": 27.27,
                    "biomass_combustion": 17.86, "biochar": 17.69,
                },
                "stochastic_total_reduction_pct_p05": 27.27,
                "stochastic_total_reduction_pct_p50": 27.66,
                "stochastic_total_reduction_pct_p95": 28.04,
                "nota": "Dataset sintético (CTGAN + reglas causales); reducción medida contra baseline sintético, no operación real.",
            },
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=None,
            ),
        )
