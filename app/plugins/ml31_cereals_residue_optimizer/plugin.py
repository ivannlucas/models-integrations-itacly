"""Ml31CerealsResidueOptimizerPlugin — v2.0 deterministic LP optimizer (PuLP/CBC).

Prescriptive optimizer for cereal crop allocation. Given the historical surface
allocation of a reference year, it reassigns rainfed/irrigated area per crop to
minimize available soil residue (Mode A) or maximize economic benefit (Mode B),
subject to hard agronomic/economic constraints (R1–R7).

Modes:
  - optimize (predict_inline, model_key="optimize"): one LP scenario -> optimal plan.
  - pareto   (predict_inline, model_key="pareto"):    residue-vs-benefit frontier.
  - batch    (predict_batch):                         optimize each CSV scenario row.

train() is not supported: the LP is deterministic, has no learned weights and no
random seed, so results are exactly reproducible from the reference data.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import (
    InfeasibleOptimizationError,
    ModelNotLoadedError,
    TrainingNotSupportedError,
)
from app.plugins.ml31_cereals_residue_optimizer.constants import (
    BENEFIT_PRESERVATION_TARGET_PCT,
    FRAMEWORK,
    MIN_BENEFIT_CHANGE_EUR,
    MODEL_ID,
    RESIDUE_REDUCTION_TARGET_PCT,
    VERSION,
)
from app.plugins.ml31_cereals_residue_optimizer.lp_solver import LPOptimizer
from app.plugins.ml31_cereals_residue_optimizer.mlflow_utils import (
    download_user_model_from_mlflow,
)
from app.plugins.ml31_cereals_residue_optimizer.model_loader import load_artifacts
from app.plugins.ml31_cereals_residue_optimizer.predict_dto import (
    PredictBatchResponse,
    PredictOptimizeResponse,
    PredictParetoResponse,
)
from app.plugins.ml31_cereals_residue_optimizer.preprocessing import (
    ParameterExtractor,
    PriceCostManager,
    filter_dominated,
    find_knee_point,
)

logger = logging.getLogger(__name__)

# Scenario fields that map to optimize-request attributes (for batch CSV rows).
_SCENARIO_DEFAULTS: dict[str, Any] = {
    "reference_year": 2023,
    "optimization_mode": "minimize_residue",
    "climate_factor": 1.0,
    "expected_spring_rain_mm": 130.0,
    "min_benefit_eur": None,
    "min_benefit_pct_of_baseline": 1.0,
    "max_residue_t": None,
    "surface_tolerance_pct": 25.0,
    "min_secano_use_pct": 95.0,
    "min_regadio_use_pct": 95.0,
}


class Ml31CerealsResidueOptimizerPlugin(ModelPluginPort):
    """Deterministic LP optimizer for cereal residue reduction / benefit."""

    def __init__(self) -> None:
        self._economics: dict | None = None
        self._hi: dict | None = None
        self._df: pd.DataFrame | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        self._economics, self._hi, self._df = load_artifacts()
        logger.info("Ml31CerealsResidueOptimizerPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        return self._df is not None and self._economics is not None

    def _require_loaded(self) -> None:
        if not self.is_loaded():
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    # ── shared LP scenario setup ──────────────────────────────────────────────
    def _resolve_year(self, reference_year: int) -> int:
        """Fall back to the latest available year if reference_year is absent."""
        available = sorted(int(y) for y in self._df["Año"].unique())
        if reference_year not in available:
            fallback = max(available)
            logger.warning(
                "reference_year %s not in data %s; using latest available %s",
                reference_year, available, fallback,
            )
            return fallback
        return reference_year

    def _build_optimizer(
        self,
        *,
        reference_year: int,
        climate_factor: float,
        spring_rain_mm: float,
        surface_tolerance_pct: float | None,
        min_secano_use_pct: float,
        min_regadio_use_pct: float,
        crop_constraints: dict,
        price_overrides: dict,
        cost_overrides: dict,
        total_secano_ha: float | None,
        total_regadio_ha: float | None,
    ) -> tuple[LPOptimizer, dict, dict]:
        """Assemble the LP optimizer + baseline for a scenario (mirrors run_inference)."""
        extractor = ParameterExtractor(self._df)
        params = extractor.extract_for_year(reference_year)

        pcm = PriceCostManager(self._economics, self._hi)
        pcm.override({"price_overrides": price_overrides, "cost_overrides": cost_overrides})

        harvest_fraction = extractor.compute_harvest_fraction(pcm)

        if total_secano_ha is not None:
            params["total_secano_ha"] = float(total_secano_ha)
        if total_regadio_ha is not None:
            params["total_regadio_ha"] = float(total_regadio_ha)

        baseline = extractor.get_baseline_metrics(
            reference_year, params, pcm,
            harvest_fraction=harvest_fraction, climate_factor=climate_factor,
        )

        min_secano_use_ha = min_secano_use_pct / 100.0 * params["total_secano_ha"]
        min_regadio_use_ha = min_regadio_use_pct / 100.0 * params["total_regadio_ha"]

        hist_area = {
            c: params["hist_secano"].get(c, 0.0) + params["hist_regadio"].get(c, 0.0)
            for c in params["crops"]
        }

        optimizer = LPOptimizer(
            crops=params["crops"],
            price_cost_mgr=pcm,
            total_secano_ha=params["total_secano_ha"],
            total_regadio_ha=params["total_regadio_ha"],
            yield_s=params["yield_s"],
            yield_r=params["yield_r"],
            climate_factor=climate_factor,
            spring_rain_mm=spring_rain_mm,
            crop_constraints=crop_constraints,
            harvest_fraction=harvest_fraction,
            min_secano_use_ha=min_secano_use_ha,
            min_regadio_use_ha=min_regadio_use_ha,
            hist_area=hist_area,
            surface_tolerance_pct=surface_tolerance_pct,
        )
        return optimizer, params, baseline

    # ── predict_inline (optimize / pareto) ────────────────────────────────────
    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictOptimizeResponse | PredictParetoResponse:
        """Dispatch to the LP optimize or the Pareto frontier branch."""
        _ = threshold
        user_temp_dir = None
        saved = (self._economics, self._hi, self._df)
        if mlflow_run_id:
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._economics, self._hi, self._df, user_temp_dir = loaded
        try:
            self._require_loaded()
            if model_key == "pareto":
                result = self._run_pareto(features)
            else:
                result = self._run_optimize(features)
            self._record()
            return result
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)
                self._economics, self._hi, self._df = saved

    def _run_optimize(self, features: dict) -> PredictOptimizeResponse:
        reference_year = self._resolve_year(int(features["reference_year"]))
        opt_mode = features["optimization_mode"]
        climate_factor = float(features["climate_factor"])
        spring_rain_mm = float(features["expected_spring_rain_mm"])
        surface_tolerance_pct = features.get("surface_tolerance_pct")

        optimizer, _params, baseline = self._build_optimizer(
            reference_year=reference_year,
            climate_factor=climate_factor,
            spring_rain_mm=spring_rain_mm,
            surface_tolerance_pct=surface_tolerance_pct,
            min_secano_use_pct=float(features["min_secano_use_pct"]),
            min_regadio_use_pct=float(features["min_regadio_use_pct"]),
            crop_constraints=features.get("crop_constraints") or {},
            price_overrides=features.get("price_overrides") or {},
            cost_overrides=features.get("cost_overrides") or {},
            total_secano_ha=features.get("total_secano_ha"),
            total_regadio_ha=features.get("total_regadio_ha"),
        )

        min_benefit = features.get("min_benefit_eur")
        max_residue = features.get("max_residue_t")
        if opt_mode == "minimize_residue" and min_benefit is None:
            threshold_pct = float(features.get("min_benefit_pct_of_baseline", 1.0))
            min_benefit = baseline["total_benefit_eur"] * threshold_pct

        solution = optimizer.solve(
            mode=opt_mode, min_benefit_eur=min_benefit, max_residue_t=max_residue
        )

        if solution.status != "OPTIMAL":
            raise InfeasibleOptimizationError(
                f"El problema LP no tiene solución óptima (estado={solution.status}) para el "
                f"escenario (año={reference_year}, modo={opt_mode}). Revise que las restricciones "
                f"(min_benefit, banda de superficie ±{surface_tolerance_pct}%, uso mínimo de "
                f"tierra) sean compatibles."
            )

        b_res = baseline["total_residue_t"]
        b_ben = baseline["total_benefit_eur"]
        b_prod = baseline["total_production_t"]

        res_red = b_res - solution.total_residue_t
        res_red_pct = (res_red / b_res * 100) if b_res else 0.0
        ben_chg = solution.total_benefit_eur - b_ben
        ben_chg_pct = (ben_chg / b_ben * 100) if b_ben else 0.0
        prod_chg = solution.total_production_t - b_prod
        prod_chg_pct = (prod_chg / b_prod * 100) if b_prod else 0.0

        # Objective success criteria (memoria Tabla 10 / config success_criteria).
        residue_pass = res_red_pct >= RESIDUE_REDUCTION_TARGET_PCT
        benefit_preservation_pct = (solution.total_benefit_eur / b_ben * 100) if b_ben else 0.0
        benefit_pass = benefit_preservation_pct >= BENEFIT_PRESERVATION_TARGET_PCT - 0.005
        min_benefit_pass = ben_chg >= MIN_BENEFIT_CHANGE_EUR - 2.5
        verdict = "PASADO" if (residue_pass and benefit_pass and min_benefit_pass) else "FALLADO"

        crop_allocation = {
            c: {
                "secano_ha": solution.secano[c],
                "regadio_ha": solution.regadio[c],
                "production_t": solution.production[c],
                "residue_t": solution.residue[c],
                "benefit_eur": solution.benefit[c],
            }
            for c in solution.crops
        }

        return PredictOptimizeResponse(
            model_id=MODEL_ID,
            reference_year=reference_year,
            optimization_mode=opt_mode,
            crop_allocation=crop_allocation,
            total_production_t=solution.total_production_t,
            total_residue_t=solution.total_residue_t,
            total_benefit_eur=solution.total_benefit_eur,
            baseline_total_production_t=b_prod,
            baseline_total_residue_t=b_res,
            baseline_total_benefit_eur=b_ben,
            residue_reduction_pct=round(res_red_pct, 2),
            benefit_change_eur=round(ben_chg, 2),
            benefit_change_pct=round(ben_chg_pct, 2),
            production_change_pct=round(prod_chg_pct, 2),
            solver_status=solution.status,
            solve_time_seconds=solution.solve_time_seconds,
            verdict=verdict,
        )

    def _run_pareto(self, features: dict) -> PredictParetoResponse:
        reference_year = self._resolve_year(int(features["reference_year"]))
        n_points = int(features["num_points"])
        pct_range = features["benefit_range_pct_of_max"]
        climate_factor = float(features["climate_factor"])
        spring_rain_mm = float(features["expected_spring_rain_mm"])
        surface_tolerance_pct = features.get("surface_tolerance_pct")

        optimizer, _params, _baseline = self._build_optimizer(
            reference_year=reference_year,
            climate_factor=climate_factor,
            spring_rain_mm=spring_rain_mm,
            surface_tolerance_pct=surface_tolerance_pct,
            min_secano_use_pct=float(features["min_secano_use_pct"]),
            min_regadio_use_pct=float(features["min_regadio_use_pct"]),
            crop_constraints={},
            price_overrides=features.get("price_overrides") or {},
            cost_overrides=features.get("cost_overrides") or {},
            total_secano_ha=None,
            total_regadio_ha=None,
        )

        # Step 1: bounds
        sol_min_res = optimizer.solve(mode="minimize_residue", min_benefit_eur=0)
        if sol_min_res.status != "OPTIMAL":
            raise InfeasibleOptimizationError(
                f"No se puede calcular el residuo mínimo (estado={sol_min_res.status})."
            )
        min_residue = sol_min_res.total_residue_t
        benefit_at_min_res = sol_min_res.total_benefit_eur

        sol_max_ben = optimizer.solve(mode="maximize_benefit", max_residue_t=None)
        if sol_max_ben.status != "OPTIMAL":
            raise InfeasibleOptimizationError(
                f"No se puede calcular el beneficio máximo (estado={sol_max_ben.status})."
            )
        max_benefit = sol_max_ben.total_benefit_eur
        residue_at_max_ben = sol_max_ben.total_residue_t

        # Step 2: sweep the benefit constraint
        b_low = max_benefit * pct_range[0]
        b_high = max_benefit * pct_range[1]
        benefit_values = np.linspace(b_low, b_high, n_points)

        pareto_points: list[dict] = []
        for i, b_k in enumerate(benefit_values):
            sol = optimizer.solve(mode="minimize_residue", min_benefit_eur=b_k)
            if sol.status == "OPTIMAL":
                pareto_points.append({
                    "index": i,
                    "benefit_eur": round(sol.total_benefit_eur, 2),
                    "residue_t": round(sol.total_residue_t, 2),
                    "production_t": round(sol.total_production_t, 2),
                })

        # Step 3: filter dominated + knee
        pareto_front = filter_dominated(pareto_points)
        knee = find_knee_point(pareto_front)

        points_out = [
            {
                "benefit_eur_M": p["benefit_eur"] / 1e6,
                "residue_t_M": p["residue_t"] / 1e6,
                "production_t_M": p["production_t"] / 1e6,
                "is_knee": (knee is not None and p is knee),
            }
            for p in pareto_front
        ]

        knee_out = (
            {
                "benefit_eur": knee["benefit_eur"],
                "residue_t": knee["residue_t"],
                "production_t": knee["production_t"],
            }
            if knee else None
        )

        return PredictParetoResponse(
            model_id=MODEL_ID,
            reference_year=reference_year,
            bounds={
                "min_residue_t": min_residue,
                "benefit_at_min_res_eur": benefit_at_min_res,
                "max_benefit_eur": max_benefit,
                "residue_at_max_ben_t": residue_at_max_ben,
            },
            pareto_points=points_out,
            knee_point=knee_out,
            num_sweep_points=len(pareto_points),
            num_pareto_points=len(pareto_front),
        )

    # ── predict_batch ──────────────────────────────────────────────────────────
    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Optimize each scenario row of a CSV (one LP solve per row)."""
        user_temp_dir = None
        saved = (self._economics, self._hi, self._df)
        if mlflow_run_id:
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._economics, self._hi, self._df, user_temp_dir = loaded
        try:
            self._require_loaded()
            df = pd.read_csv(data_path)
            predictions: list[dict] = []
            for idx, row in df.iterrows():
                try:
                    features = self._row_to_scenario(row)
                    result = self._run_optimize(features)
                    predictions.append({
                        "row": int(idx),
                        "reference_year": result.reference_year,
                        "optimization_mode": result.optimization_mode,
                        "total_production_t": result.total_production_t,
                        "total_residue_t": result.total_residue_t,
                        "total_benefit_eur": result.total_benefit_eur,
                        "residue_reduction_pct": result.residue_reduction_pct,
                        "benefit_change_pct": result.benefit_change_pct,
                        "production_change_pct": result.production_change_pct,
                        "solver_status": result.solver_status,
                        "verdict": result.verdict,
                    })
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.warning("Error en fila %s: %s", idx, exc)
                    predictions.append({"row": int(idx), "error": str(exc)})
            self._record()
            logger.info("predict_batch done — %d scenarios", len(predictions))
            return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)
                self._economics, self._hi, self._df = saved

    @staticmethod
    def _row_to_scenario(row: pd.Series) -> dict:
        """Build an optimize-features dict from a CSV row, filling manifest defaults."""
        features: dict[str, Any] = {}
        for key, default in _SCENARIO_DEFAULTS.items():
            if key in row and pd.notna(row[key]):
                features[key] = row[key]
            else:
                features[key] = default
        # Overrides / per-crop constraints are not expressed per CSV row.
        features["crop_constraints"] = {}
        features["price_overrides"] = {}
        features["cost_overrides"] = {}
        features["total_secano_ha"] = None
        features["total_regadio_ha"] = None
        return features

    # ── train (not supported) ──────────────────────────────────────────────────
    def train(self, *, data_path: str, mlflow_run_id: str = "") -> Any:
        """Training is not supported: the v2.0 model is a deterministic LP optimizer
        with no learned weights (HTTP 501)."""
        _ = data_path, mlflow_run_id
        raise TrainingNotSupportedError(
            "Este modelo es un optimizador de Programación Lineal determinista; no tiene "
            "pesos entrenables ni soporta reentrenamiento por usuario."
        )

    # ── stats ────────────────────────────────────────────────────────────────
    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        if mlflow_run_id:
            logger.warning(
                "mlflow_run_id=%s provided but model '%s' is a deterministic LP optimizer "
                "(no user training)", mlflow_run_id, MODEL_ID,
            )
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Optimizador prescriptivo determinista (Programación Lineal, PuLP/CBC) para la "
                "reducción de residuos vegetales en el sector cerealístico. Reasigna superficie "
                "de secano/regadío por cultivo (20 variables de decisión). Modo optimize: plan "
                "óptimo que minimiza residuo (Modo A) o maximiza beneficio (Modo B) bajo las "
                "restricciones R1–R7. Modo pareto: frontera residuo-vs-beneficio. Modo batch: "
                "optimiza cada escenario de un CSV. Sin pesos entrenados ni semilla aleatoria."
            ),
            task_type="optimization",
            framework=FRAMEWORK,
            inputs=[
                InputField(name="reference_year", type="int", default=2023,
                           description="Año de referencia (superficies históricas) [optimize/pareto]"),
                InputField(name="optimization_mode", type="str", default="minimize_residue",
                           description="minimize_residue (Modo A) | maximize_benefit (Modo B) [optimize]"),
                InputField(name="climate_factor", type="float", default=1.0,
                           description="Multiplicador sobre los rendimientos [optimize/pareto]"),
                InputField(name="expected_spring_rain_mm", type="float", default=130.0,
                           description="Lluvia de primavera esperada (mm); <100 activa estrés hídrico"),
                InputField(name="min_benefit_eur", type="float",
                           description="Beneficio mínimo exigido (R6, Modo A); null => baseline"),
                InputField(name="min_benefit_pct_of_baseline", type="float", default=1.0,
                           description="% del beneficio baseline a preservar [optimize Modo A]"),
                InputField(name="max_residue_t", type="float",
                           description="Residuo máximo admisible (R7, Modo B)"),
                InputField(name="surface_tolerance_pct", type="float", default=25.0,
                           description="Banda ±% sobre la superficie histórica por cultivo (R4c)"),
                InputField(name="min_secano_use_pct", type="float", default=95.0,
                           description="Uso mínimo de secano (R1b)"),
                InputField(name="min_regadio_use_pct", type="float", default=95.0,
                           description="Uso mínimo de regadío (R2b)"),
                InputField(name="num_points", type="int", default=20,
                           description="Nº de puntos del barrido de beneficio [pareto]"),
                InputField(name="benefit_range_pct_of_max", type="list",
                           description="Rango [low, high] como fracción del beneficio máximo [pareto]"),
            ],
            outputs=[
                OutputField(name="crop_allocation", type="dict",
                            description="Plan óptimo por cultivo: secano_ha/regadio_ha/production_t/residue_t/benefit_eur [optimize]"),
                OutputField(name="total_residue_t", type="float",
                            description="Residuo total de la solución óptima (objetivo Modo A) [optimize]"),
                OutputField(name="total_benefit_eur", type="float",
                            description="Beneficio total de la solución óptima (objetivo Modo B) [optimize]"),
                OutputField(name="residue_reduction_pct", type="float",
                            description="Reducción de residuo vs baseline (positivo=reducción) [optimize]"),
                OutputField(name="verdict", type="str",
                            description="PASADO/FALLADO según criterios de éxito [optimize]"),
                OutputField(name="solver_status", type="str",
                            description="Estado CBC: OPTIMAL | INFEASIBLE | UNBOUNDED | NOT_SOLVED"),
                OutputField(name="bounds", type="dict",
                            description="Extremos de la frontera de Pareto [pareto]"),
                OutputField(name="pareto_points", type="list",
                            description="Puntos no dominados de la frontera (en millones) [pareto]"),
                OutputField(name="knee_point", type="dict",
                            description="Mejor compromiso residuo/beneficio [pareto]"),
            ],
            metrics={
                # Escenario de referencia 2023 (memoria Tabla 12 / optimization_metrics.json).
                "residue_reduction_pct_2023": 31.18,
                "benefit_preservation_pct_2023": 100.0,
                "production_change_pct_2023": -5.05,
                "residuo_por_euro_baseline_t_eur": 0.0191,
                "residuo_por_euro_optimo_t_eur": 0.0132,
                "solver_status_2023": "OPTIMAL",
                "solve_time_seconds_2023": 0.0138,
                "nota_signo": ("residue_reduction_pct es POSITIVO cuando hay reducción; la memoria "
                               "reporta el mismo dato con signo negativo como 'variación'."),
            },
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=None,
            ),
        )
