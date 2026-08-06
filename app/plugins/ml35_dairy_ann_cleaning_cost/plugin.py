"""Ml35DairyAnnCleaningCostPlugin — ANN + GA for dairy pasteurization water optimization."""
from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError, PuConstraintViolationError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml35_dairy_ann_cleaning_cost.constants import (
    BASELINE_FLUJO,
    BASELINE_TEMP_AGUA,
    BASELINE_TEMP_LECHE,
    FEATURES,
    FRAMEWORK,
    GA_GENE_SPACE,
    GA_NUM_GENERATIONS,
    GA_NUM_PARENTS_MATING,
    GA_RANDOM_SEED,
    GA_SOL_PER_POP,
    MODEL_FILENAME,
    MODEL_ID,
    PU_MIN,
    SCALER_X_FILENAME,
    SCALER_Y_FILENAME,
    T_REF_CELSIUS,
    TARGET_COL,
    TEMP_MIN_CELSIUS,
    VERSION,
    VOLUMEN_RETENCION_L,
    Z_VALUE,
)
from app.plugins.ml35_dairy_ann_cleaning_cost.model_loader import PasteurizationANN, load_artifacts, _store
from app.plugins.ml35_dairy_ann_cleaning_cost.mlflow_utils import download_user_model_from_mlflow
from app.plugins.ml35_dairy_ann_cleaning_cost.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
    PredictOptimizeResponse,
)
from app.plugins.ml35_dairy_ann_cleaning_cost.train_dto import TrainResponse

logger = logging.getLogger(__name__)


def _compute_pu(temp_proceso: float, flujo: float) -> float:
    """Compute pasteurization units (deterministic HTST formula)."""
    t_res = (VOLUMEN_RETENCION_L / flujo) * 3600
    return t_res * (10 ** ((temp_proceso - T_REF_CELSIUS) / Z_VALUE))


def _infer(model: Any, scaler_X: Any, scaler_y: Any, df: pd.DataFrame) -> float:
    """Scale input → ANN forward → descale output → return float (L)."""
    x_scaled = scaler_X.transform(df)
    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(x_scaled)).numpy()
    return float(scaler_y.inverse_transform(pred_scaled)[0, 0])


def _resolve_features(features: dict) -> dict:
    """Fill temp_proceso_leche/temp_agua_servicio defaults when the caller omits them."""
    data = dict(features)
    if data.get("temp_proceso_leche") is None:
        data["temp_proceso_leche"] = data["temp_setpoint_leche"]
    if data.get("temp_agua_servicio") is None:
        data["temp_agua_servicio"] = data["temp_proceso_leche"] + 10.0
    return data


def _xai_values_from_features(data: dict) -> dict[str, float] | None:
    """Build xai_feature_values from a resolved (post-default) feature dict."""
    xai: dict[str, float] = {}
    for f in FEATURES:
        val = data.get(f)
        if val is not None and pd.notna(val):
            try:
                xai[f] = float(val)
            except (TypeError, ValueError):
                pass
    return xai or None


def _build_df(temp_entrada, temp_ambiente, temp_setpoint, temp_proceso, temp_agua, flujo,
              horas, presion) -> pd.DataFrame:
    return pd.DataFrame([[
        temp_entrada, temp_ambiente, temp_setpoint, temp_proceso,
        temp_agua, flujo, horas, presion,
    ]], columns=FEATURES)


class Ml35DairyAnnCleaningCostPlugin(ModelPluginPort):
    """ANN regressor + pygad GA optimizer for pasteurization water consumption."""

    def __init__(self) -> None:
        self._model: Any = None
        self._scaler_X: Any = None
        self._scaler_y: Any = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        self._model, self._scaler_X, self._scaler_y = load_artifacts()
        logger.info("Ml35DairyAnnCleaningCostPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        return self._model is not None

    def _require_loaded(self) -> None:
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse | PredictOptimizeResponse:
        """Dispatch to ANN predict or GA optimize based on model_key."""
        user_temp_dir = None
        saved = (self._model, self._scaler_X, self._scaler_y)
        if mlflow_run_id:
            logger.info("predict_inline — using user model from MLflow run_id=%s", mlflow_run_id)
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._model, self._scaler_X, self._scaler_y, user_temp_dir = loaded
        try:
            self._require_loaded()
            if model_key == "optimize":
                result = self._run_optimize(features)
            else:
                result = self._run_predict(features)
            self._record()
            return result
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)
                self._model, self._scaler_X, self._scaler_y = saved

    def _run_predict(self, features: dict) -> PredictInlineResponse:
        """ANN single-sample inference with PU validation."""
        data = _resolve_features(features)

        temp_setpoint = float(data["temp_setpoint_leche"])
        temp_proceso = float(data["temp_proceso_leche"])
        flujo = float(data["flujo_leche_lh"])
        pu = _compute_pu(temp_proceso, flujo)

        if temp_setpoint < TEMP_MIN_CELSIUS or pu < PU_MIN:
            raise PuConstraintViolationError(
                f"Restricción de seguridad alimentaria incumplida: "
                f"temp_setpoint_leche={temp_setpoint}°C (mín {TEMP_MIN_CELSIUS}°C), "
                f"PU={pu:.3f} (mín {PU_MIN}). Ajuste los setpoints."
            )

        df = _build_df(
            data["temp_entrada_leche"], data["temp_ambiente"],
            temp_setpoint, temp_proceso,
            data["temp_agua_servicio"], flujo,
            data["horas_desde_limpieza"], data["presion_diferencial_bar"],
        )
        consumo = _infer(self._model, self._scaler_X, self._scaler_y, df)
        return PredictInlineResponse(
            model_id=MODEL_ID,
            consumo_agua_l=round(consumo, 2),
            pu_logrado=round(pu, 4),
        )

    def _run_optimize(self, features: dict) -> PredictOptimizeResponse:
        """Run pygad GA to find minimal-water setpoints respecting PU ≥ 13."""
        import pygad  # pylint: disable=import-outside-toplevel

        ctx = {k: float(features[k]) for k in (
            "temp_entrada_leche", "temp_ambiente",
            "horas_desde_limpieza", "presion_diferencial_bar",
        )}

        model = self._model
        scaler_X = self._scaler_X
        scaler_y = self._scaler_y

        def _fitness(ga_instance, solution, solution_idx):
            t_leche, t_agua, flujo = float(solution[0]), float(solution[1]), float(solution[2])
            pu = _compute_pu(t_leche, flujo)
            if t_leche < TEMP_MIN_CELSIUS or pu < PU_MIN:
                return 1e-8
            df = _build_df(
                ctx["temp_entrada_leche"], ctx["temp_ambiente"],
                t_leche, t_leche, t_agua, flujo,
                ctx["horas_desde_limpieza"], ctx["presion_diferencial_bar"],
            )
            consumo = _infer(model, scaler_X, scaler_y, df)
            return 1.0 / (consumo + 1e-6)

        ga = pygad.GA(
            num_generations=GA_NUM_GENERATIONS,
            num_parents_mating=GA_NUM_PARENTS_MATING,
            fitness_func=_fitness,
            sol_per_pop=GA_SOL_PER_POP,
            num_genes=3,
            gene_space=GA_GENE_SPACE,
            suppress_warnings=True,
            random_seed=GA_RANDOM_SEED,
        )
        ga.run()
        best_sol, _, _ = ga.best_solution()
        opt_t_leche = float(best_sol[0])
        opt_t_agua = float(best_sol[1])
        opt_flujo = float(best_sol[2])
        pu_logrado = _compute_pu(opt_t_leche, opt_flujo)

        def _consumo(t_l, t_a, flujo):
            df = _build_df(
                ctx["temp_entrada_leche"], ctx["temp_ambiente"],
                t_l, t_l, t_a, flujo,
                ctx["horas_desde_limpieza"], ctx["presion_diferencial_bar"],
            )
            return _infer(model, scaler_X, scaler_y, df)

        consumo_estandar = _consumo(BASELINE_TEMP_LECHE, BASELINE_TEMP_AGUA, BASELINE_FLUJO)
        consumo_optimizado = _consumo(opt_t_leche, opt_t_agua, opt_flujo)
        ahorro_l = consumo_estandar - consumo_optimizado
        ahorro_pct = (ahorro_l / consumo_estandar * 100.0) if consumo_estandar else 0.0

        logger.info(
            "GA optimization done — opt=(%.2f°C, %.2f°C, %.0f L/h) ahorro=%.2f%% PU=%.2f",
            opt_t_leche, opt_t_agua, opt_flujo, ahorro_pct, pu_logrado,
        )
        return PredictOptimizeResponse(
            model_id=MODEL_ID,
            opt_temp_leche=round(opt_t_leche, 4),
            opt_temp_agua=round(opt_t_agua, 4),
            opt_flujo=round(opt_flujo, 2),
            consumo_estandar=round(consumo_estandar, 2),
            consumo_optimizado=round(consumo_optimizado, 2),
            ahorro_l=round(ahorro_l, 2),
            ahorro_pct=round(ahorro_pct, 4),
            pu_logrado=round(pu_logrado, 4),
        )

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Batch ANN inference over a CSV file."""
        user_temp_dir = None
        saved = (self._model, self._scaler_X, self._scaler_y)
        if mlflow_run_id:
            logger.info("predict_batch — using user model from MLflow run_id=%s", mlflow_run_id)
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._model, self._scaler_X, self._scaler_y, user_temp_dir = loaded
        try:
            self._require_loaded()
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)
            predictions: list[dict] = []
            for idx, row in df.iterrows():
                try:
                    row_dict = row.to_dict()
                    pred = self._run_predict(row_dict)
                    predictions.append({
                        "row": int(idx),
                        "consumo_agua_l": pred.consumo_agua_l,
                        "pu_logrado": pred.pu_logrado,
                        "xai_feature_values": _xai_values_from_features(_resolve_features(row_dict)),
                    })
                except Exception as exc:
                    logger.warning("Error en fila %s: %s", idx, exc)
                    predictions.append({"row": int(idx), "error": str(exc)})
            self._record()
            logger.info("predict_batch done — %d rows, mlflow=%s", len(predictions), bool(mlflow_run_id))
            return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)
                self._model, self._scaler_X, self._scaler_y = saved

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> TrainResponse:  # pylint: disable=too-many-locals
        """Fine-tune the ANN on user-labeled data (consumo_agua_l target required)."""
        import tempfile  # pylint: disable=import-outside-toplevel
        import joblib  # pylint: disable=import-outside-toplevel
        import torch.nn as nn  # pylint: disable=import-outside-toplevel
        import torch.optim as optim  # pylint: disable=import-outside-toplevel

        EPOCHS = 100
        LR = 0.001

        if mlflow_run_id:
            tracker: BaseMLflowTracker | None = BaseMLflowTracker(mlflow_run_id)
            tracker.log_params({"epochs": EPOCHS, "lr": LR, "optimizer": "Adam"})
        else:
            tracker = None

        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)
        if "temp_proceso_leche" not in df.columns and "temp_setpoint_leche" in df.columns:
            df["temp_proceso_leche"] = df["temp_setpoint_leche"]
        if "temp_agua_servicio" not in df.columns and "temp_proceso_leche" in df.columns:
            df["temp_agua_servicio"] = df["temp_proceso_leche"] + 10.0

        missing = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
        if missing:
            raise ValueError(f"CSV falta columnas requeridas: {missing}")

        self._require_loaded()
        x_scaled = self._scaler_X.transform(df[FEATURES])
        y_scaled = self._scaler_y.transform(df[[TARGET_COL]])

        # Clone weights into a new model instance to avoid mutating the live model
        fine_model = PasteurizationANN(input_size=len(FEATURES))
        fine_model.load_state_dict(self._model.state_dict())
        fine_model.train()
        optimizer = optim.Adam(fine_model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        x_tensor = torch.FloatTensor(x_scaled)
        y_tensor = torch.FloatTensor(y_scaled)

        for _ in range(EPOCHS):
            optimizer.zero_grad()
            loss = criterion(fine_model(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()

        fine_model.eval()
        with torch.no_grad():
            y_pred_scaled = fine_model(x_tensor).numpy()
        y_pred = self._scaler_y.inverse_transform(y_pred_scaled)
        y_real = df[[TARGET_COL]].values
        mae = float(np.mean(np.abs(y_real - y_pred)))
        ss_res = np.sum((y_real - y_pred) ** 2)
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot else 0.0

        _store.local_dir.mkdir(parents=True, exist_ok=True)
        torch.save(fine_model.state_dict(), _store.local_dir / MODEL_FILENAME)

        if tracker:
            tracker.log_metrics({"mae": mae, "r2": r2, "n_samples": len(df)})
            try:
                mlflow_tmp = tempfile.mkdtemp(prefix="ml35_mlflow_")
                torch.save(fine_model.state_dict(), os.path.join(mlflow_tmp, MODEL_FILENAME))
                joblib.dump(self._scaler_X, os.path.join(mlflow_tmp, SCALER_X_FILENAME))
                joblib.dump(self._scaler_y, os.path.join(mlflow_tmp, SCALER_Y_FILENAME))
                tracker.upload_artifacts(mlflow_tmp, artifact_path="model")
                shutil.rmtree(mlflow_tmp, ignore_errors=True)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("MLflow artifact upload failed: %s", exc)

        self.load()
        logger.info("train() done — mae=%.2f r2=%.4f n=%d mlflow=%s", mae, r2, len(df), bool(mlflow_run_id))
        return TrainResponse(
            detail="Fine-tuning completado",
            mae=round(mae, 2),
            r2=round(r2, 4),
            n_samples=int(len(df)),
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Red neuronal ANN (PyTorch) + algoritmo genético (pygad) para optimización "
                "del consumo de agua de refrigeración en pasteurización HTST láctea. "
                "Modo inline: predice consumo_agua_l. Modo optimize: sugiere setpoints óptimos."
            ),
            task_type="regression_prescriptive",
            framework=FRAMEWORK,
            inputs=[
                InputField(name="temp_entrada_leche", type="float",
                           description="Temperatura de entrada de la leche (°C)"),
                InputField(name="temp_ambiente", type="float",
                           description="Temperatura ambiente (°C)"),
                InputField(name="temp_setpoint_leche", type="float",
                           description="Setpoint de temperatura de leche (°C, ≥ 72)"),
                InputField(name="temp_proceso_leche", type="float",
                           description="Temperatura de proceso real (°C); si no se aporta = temp_setpoint_leche"),
                InputField(name="temp_agua_servicio", type="float",
                           description="Temperatura del agua de servicio (°C); si no se aporta = temp_proceso_leche + 10"),
                InputField(name="flujo_leche_lh", type="float",
                           description="Caudal de leche (L/h)"),
                InputField(name="horas_desde_limpieza", type="float",
                           description="Horas desde la última limpieza — proxy de fouling (h)"),
                InputField(name="presion_diferencial_bar", type="float",
                           description="Presión diferencial en el intercambiador de calor (bar)"),
            ],
            outputs=[
                OutputField(name="consumo_agua_l", type="float",
                            description="Consumo de agua de refrigeración predicho (L) [modo inline]"),
                OutputField(name="pu_logrado", type="float",
                            description="Unidades de pasteurización logradas (PU ≥ 13 requerido)"),
                OutputField(name="opt_temp_leche", type="float",
                            description="Setpoint óptimo de temperatura leche (°C) [modo optimize]"),
                OutputField(name="opt_temp_agua", type="float",
                            description="Temperatura óptima agua de servicio (°C) [modo optimize]"),
                OutputField(name="opt_flujo", type="float",
                            description="Caudal óptimo de leche (L/h) [modo optimize]"),
                OutputField(name="ahorro_pct", type="float",
                            description="Ahorro de agua vs. baseline estándar (%) [modo optimize]"),
            ],
            metrics={
                "r2_test": 0.9932,
                "mae_l": 341.58,
                "rmse_l": 417.75,
                "mae_relativo_pct": 1.39,
                "n_test": 750,
                "split": "70/15/15",
            },
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=None,
            ),
        )
        if mlflow_run_id:
            try:
                tracker = BaseMLflowTracker(mlflow_run_id)
                base.metrics["mlflow"] = {
                    "params": tracker.get_params(),
                    "metrics": tracker.get_metrics(),
                }
                logger.info("Stats enriched with MLflow data for run_id=%s", mlflow_run_id)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Could not fetch MLflow stats for run_id=%s: %s", mlflow_run_id, exc)
        return base
