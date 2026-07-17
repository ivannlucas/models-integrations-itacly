"""Ml34DairyPasteurizationEnergyGaPlugin — MLP surrogate + single-objective GA (DEAP)
for energy optimization of dairy pasteurization.

Inline mode predicts (E_consumo, T_out_leche) with the MLP digital twin.
Optimize mode runs the GA v4 per scenario: minimizes the specific consumption
E_consumo/F_flow subject to the food-safety constraint T_out >= 72.3 °C.

Attribute/variable names like _scaler_X intentionally mirror the original
codebase and its artifacts.
"""
# pylint: disable=invalid-name
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
from app.domain.services.exceptions import (
    ModelNotLoadedError,
    ThermalSafetyViolationError,
)
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml34_dairy_pasteurization_energy_ga.constants import (
    FEATURES,
    FRAMEWORK,
    GA_CXPB,
    GA_MUTPB,
    GA_N_GEN,
    GA_POP_SIZE,
    MODEL_CONFIG_FILENAME,
    MODEL_FILENAME,
    MODEL_ID,
    SCALER_X_FILENAME,
    SCALER_Y_FILENAME,
    T_OUT_MIN,
    TARGETS,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_PATIENCE,
    TRAIN_SEED,
    VERSION,
)
from app.plugins.ml34_dairy_pasteurization_energy_ga.ga_optimizer import (
    predict_scenario,
    run_ga_single,
    setup_ga_toolbox,
)
from app.plugins.ml34_dairy_pasteurization_energy_ga.mlflow_utils import (
    download_user_model_from_mlflow,
)
from app.plugins.ml34_dairy_pasteurization_energy_ga.model_loader import (
    build_model_from_config,
    load_artifacts,
)
from app.plugins.ml34_dairy_pasteurization_energy_ga.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
    PredictOptimizeResponse,
)
from app.plugins.ml34_dairy_pasteurization_energy_ga.train_dto import TrainResponse

logger = logging.getLogger(__name__)


class Ml34DairyPasteurizationEnergyGaPlugin(ModelPluginPort):
    """MLP digital twin + DEAP single-objective GA for pasteurization setpoints."""

    def __init__(self) -> None:
        self._model: Any = None
        self._scaler_X: Any = None
        self._scaler_y: Any = None
        self._config: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        self._model, self._scaler_X, self._scaler_y, self._config = load_artifacts()
        logger.info("Ml34DairyPasteurizationEnergyGaPlugin loaded: %s", MODEL_ID)

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
        """Dispatch to MLP predict or GA optimize based on model_key."""
        user_temp_dir = None
        saved = (self._model, self._scaler_X, self._scaler_y, self._config)
        if mlflow_run_id:
            logger.info("predict_inline — using user model from MLflow run_id=%s", mlflow_run_id)
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._model, self._scaler_X, self._scaler_y, self._config, user_temp_dir = loaded
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
                self._model, self._scaler_X, self._scaler_y, self._config = saved

    def _run_predict(self, features: dict) -> PredictInlineResponse:
        """MLP surrogate single-sample inference in real units."""
        e_consumo, t_out = predict_scenario(
            self._model, self._scaler_X, self._scaler_y,
            float(features["T_in_leche"]), float(features["F_flow"]),
            float(features["T_servicio"]), float(features["t_ciclo"]),
            float(features["Delta_P"]),
        )
        return PredictInlineResponse(
            model_id=MODEL_ID,
            E_consumo_pred=round(e_consumo, 4),
            T_out_pred=round(t_out, 4),
        )

    def _run_optimize(self, features: dict) -> PredictOptimizeResponse:
        """Run the single-objective GA for one scenario (deterministic per seed)."""
        t_in = float(features["T_in_leche"])
        delta_p = float(features["Delta_P"])
        t_ciclo = float(features["t_ciclo"])
        seed = int(features.get("seed", 1))

        toolbox = setup_ga_toolbox()
        hof = run_ga_single(
            toolbox, T_in_leche=t_in, t_ciclo=t_ciclo, Delta_P=delta_p,
            model=self._model, scaler_X=self._scaler_X, scaler_y=self._scaler_y,
            seed=seed,
        )

        best = hof[0]
        f_flow, t_servicio = float(best[0]), float(best[1])
        e_consumo, t_out = predict_scenario(
            self._model, self._scaler_X, self._scaler_y,
            t_in, f_flow, t_servicio, t_ciclo, delta_p,
        )
        specific = e_consumo / max(f_flow, 1.0)
        feasible = t_out >= T_OUT_MIN

        if not feasible:
            raise ThermalSafetyViolationError(
                f"El GA no encontró solución factible para el escenario "
                f"(T_in_leche={t_in}, Delta_P={delta_p}, t_ciclo={t_ciclo}): "
                f"mejor T_out predicha = {t_out:.2f} °C < {T_OUT_MIN} °C. "
                f"Revise que el escenario esté dentro del rango de entrenamiento."
            )

        logger.info(
            "GA done — seed=%d setpoints=(%.2f L/h, %.2f °C) E/F=%.6f T_out=%.2f",
            seed, f_flow, t_servicio, specific, t_out,
        )
        return PredictOptimizeResponse(
            model_id=MODEL_ID,
            IA_F_flow=round(f_flow, 2),
            IA_T_servicio=round(t_servicio, 2),
            IA_E_consumo=round(e_consumo, 4),
            IA_T_out=round(t_out, 2),
            IA_consumo_especifico=round(specific, 6),
            IA_factible=feasible,
            fitness_final=round(float(best.fitness.values[0]), 6),
            seed=seed,
        )

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Batch MLP inference over a CSV file (one prediction per row)."""
        user_temp_dir = None
        saved = (self._model, self._scaler_X, self._scaler_y, self._config)
        if mlflow_run_id:
            logger.info("predict_batch — using user model from MLflow run_id=%s", mlflow_run_id)
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._model, self._scaler_X, self._scaler_y, self._config, user_temp_dir = loaded
        try:
            self._require_loaded()
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                raise ValueError(f"CSV falta columnas requeridas: {missing}")
            predictions: list[dict] = []
            for idx, row in df.iterrows():
                try:
                    pred = self._run_predict(row.to_dict())
                    predictions.append({
                        "row": int(idx),
                        "E_consumo_pred": pred.E_consumo_pred,
                        "T_out_pred": pred.T_out_pred,
                    })
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.warning("Error en fila %s: %s", idx, exc)
                    predictions.append({"row": int(idx), "error": str(exc)})
            self._record()
            logger.info("predict_batch done — %d rows, mlflow=%s", len(predictions), bool(mlflow_run_id))
            return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)
                self._model, self._scaler_X, self._scaler_y, self._config = saved

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> TrainResponse:  # noqa: C901  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
        """Fine-tune the MLP on user data (5 features + 2 targets required).

        Follows the original training recipe (src/main.py::train): Adam
        lr=0.0005, MSELoss over both targets, batch_size=128, max 300 epochs
        with early-stopping patience 15 monitored on a temporal 15% hold-out.
        """
        import copy  # pylint: disable=import-outside-toplevel
        import json  # pylint: disable=import-outside-toplevel
        import tempfile  # pylint: disable=import-outside-toplevel
        import joblib  # pylint: disable=import-outside-toplevel
        from torch import nn, optim  # pylint: disable=import-outside-toplevel
        from torch.utils.data import (  # pylint: disable=import-outside-toplevel
            DataLoader,
            TensorDataset,
        )

        if mlflow_run_id:
            tracker: BaseMLflowTracker | None = BaseMLflowTracker(mlflow_run_id)
            tracker.log_params({
                "epochs_max": TRAIN_EPOCHS, "lr": TRAIN_LR, "optimizer": "Adam",
                "batch_size": TRAIN_BATCH_SIZE, "patience": TRAIN_PATIENCE,
            })
        else:
            tracker = None

        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)
        missing = [c for c in FEATURES + TARGETS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV falta columnas requeridas: {missing}")

        self._require_loaded()
        torch.manual_seed(TRAIN_SEED)
        np.random.seed(TRAIN_SEED)

        x_scaled = self._scaler_X.transform(df[FEATURES].values)
        y_scaled = self._scaler_y.transform(df[TARGETS].values)

        # Temporal hold-out for early stopping (last 15% of rows)
        n = len(df)
        n_val = max(1, int(n * 0.15)) if n >= 10 else 0
        if n_val:
            x_tr, y_tr = x_scaled[:-n_val], y_scaled[:-n_val]
            x_val, y_val = x_scaled[-n_val:], y_scaled[-n_val:]
        else:
            x_tr, y_tr = x_scaled, y_scaled
            x_val, y_val = x_scaled, y_scaled

        # Clone weights into a new model instance to avoid mutating the live model
        fine_model = build_model_from_config(self._config)
        fine_model.load_state_dict(self._model.state_dict())
        fine_model.train()

        optimizer = optim.Adam(fine_model.parameters(), lr=TRAIN_LR)
        criterion = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(x_tr), torch.FloatTensor(y_tr)),
            batch_size=TRAIN_BATCH_SIZE, shuffle=True,
            generator=torch.Generator().manual_seed(TRAIN_SEED),
        )
        x_val_t = torch.FloatTensor(x_val)
        y_val_t = torch.FloatTensor(y_val)

        best_val = float("inf")
        best_state = copy.deepcopy(fine_model.state_dict())
        no_improve = 0
        epochs_executed = 0
        for epoch in range(TRAIN_EPOCHS):
            fine_model.train()
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(fine_model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
            fine_model.eval()
            with torch.no_grad():
                val_loss = criterion(fine_model(x_val_t), y_val_t).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(fine_model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            epochs_executed = epoch + 1
            if no_improve >= TRAIN_PATIENCE:
                break

        fine_model.load_state_dict(best_state)
        fine_model.eval()

        with torch.no_grad():
            y_pred_scaled = fine_model(torch.FloatTensor(x_scaled)).numpy()
        y_pred = self._scaler_y.inverse_transform(y_pred_scaled)
        y_real = df[TARGETS].values

        metrics: dict[str, float] = {}
        for i, target in enumerate(TARGETS):
            err = y_real[:, i] - y_pred[:, i]
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((y_real[:, i] - np.mean(y_real[:, i])) ** 2))
            r2 = float(1 - ss_res / ss_tot) if ss_tot else 0.0
            metrics[f"rmse_{target}"] = rmse
            metrics[f"mae_{target}"] = mae
            metrics[f"r2_{target}"] = r2

        if tracker:
            tracker.log_metrics({**metrics, "n_samples": len(df)})
            try:
                mlflow_tmp = tempfile.mkdtemp(prefix="ml34_mlflow_")
                torch.save(fine_model.state_dict(), os.path.join(mlflow_tmp, MODEL_FILENAME))
                with open(os.path.join(mlflow_tmp, MODEL_CONFIG_FILENAME), "w", encoding="utf-8") as f:
                    json.dump(self._config, f, indent=4)
                joblib.dump(self._scaler_X, os.path.join(mlflow_tmp, SCALER_X_FILENAME))
                joblib.dump(self._scaler_y, os.path.join(mlflow_tmp, SCALER_Y_FILENAME))
                tracker.upload_artifacts(mlflow_tmp, artifact_path="model")
                shutil.rmtree(mlflow_tmp, ignore_errors=True)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("MLflow artifact upload failed: %s", exc)

        logger.info(
            "train() done — mae_E=%.2f r2_E=%.4f n=%d epochs=%d mlflow=%s",
            metrics["mae_E_consumo"], metrics["r2_E_consumo"], len(df),
            epochs_executed, bool(mlflow_run_id),
        )
        return TrainResponse(
            detail="Fine-tuning completado",
            rmse_E_consumo=round(metrics["rmse_E_consumo"], 4),
            mae_E_consumo=round(metrics["mae_E_consumo"], 4),
            r2_E_consumo=round(metrics["r2_E_consumo"], 4),
            rmse_T_out_leche=round(metrics["rmse_T_out_leche"], 4),
            mae_T_out_leche=round(metrics["mae_T_out_leche"], 4),
            r2_T_out_leche=round(metrics["r2_T_out_leche"], 4),
            n_samples=int(len(df)),
            epochs_executed=int(epochs_executed),
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Gemelo digital MLP (PyTorch) + algoritmo genético mono-objetivo (DEAP) "
                "para reducción del consumo energético específico (E_consumo/F_flow) en "
                "pasteurización HTST láctea. Modo inline: predice E_consumo y T_out_leche. "
                "Modo optimize: recomienda setpoints (F_flow, T_servicio) con restricción "
                "de seguridad alimentaria T_out >= 72.3 °C."
            ),
            task_type="regression_prescriptive",
            framework=FRAMEWORK,
            inputs=[
                InputField(name="T_in_leche", type="float",
                           description="Temperatura de entrada de la leche cruda (°C) "
                                       "[inline y optimize]"),
                InputField(name="F_flow", type="float",
                           description="Caudal volumétrico de leche (L/h) "
                                       "[inline; en optimize lo decide el GA]"),
                InputField(name="T_servicio", type="float",
                           description="Temperatura del fluido de servicio (°C) "
                                       "[inline; en optimize lo decide el GA]"),
                InputField(name="t_ciclo", type="float",
                           description="Tiempo desde la última limpieza CIP (min) "
                                       "[inline y optimize]"),
                InputField(name="Delta_P", type="float",
                           description="Caída de presión en el intercambiador, proxy de "
                                       "fouling (bar) [inline y optimize]"),
                InputField(name="seed", type="int",
                           description="Semilla del GA para reproducibilidad por escenario "
                                       "[optimize, opcional]"),
            ],
            outputs=[
                OutputField(name="E_consumo_pred", type="float",
                            description="Consumo energético total predicho (kW) [modo inline]"),
                OutputField(name="T_out_pred", type="float",
                            description="Temperatura de salida de la leche predicha (°C) "
                                        "[modo inline]"),
                OutputField(name="IA_F_flow", type="float",
                            description="Setpoint óptimo de caudal (L/h) [modo optimize]"),
                OutputField(name="IA_T_servicio", type="float",
                            description="Setpoint óptimo de temperatura de servicio (°C) "
                                        "[modo optimize]"),
                OutputField(name="IA_E_consumo", type="float",
                            description="Consumo predicho con los setpoints IA (kW) "
                                        "[modo optimize]"),
                OutputField(name="IA_T_out", type="float",
                            description="Temperatura de salida predicha con setpoints IA (°C) "
                                        "[modo optimize]"),
                OutputField(name="IA_consumo_especifico", type="float",
                            description="Consumo específico IA (kW/(L/h)) — KPI principal "
                                        "[modo optimize]"),
                OutputField(name="IA_factible", type="bool",
                            description="Cumplimiento de la restricción T_out >= 72.3 °C "
                                        "[modo optimize]"),
            ],
            metrics={
                # MLP surrogate — test hold-out (train_metrics.json entregado)
                "E_consumo_rmse_kw": 5.3838,
                "E_consumo_mae_kw": 4.2571,
                "E_consumo_r2": 0.9779,
                "T_out_rmse_c": 0.0643,
                "T_out_mae_c": 0.0473,
                "T_out_r2": 0.3759,
                "n_test": 7708,
                "split": "70/15/15 temporal por cuartiles",
                # GA backtesting — evaluation_rt_backtesting_report.json entregado
                "ga_mejora_consumo_especifico_pct": 11.73,
                "ga_ahorro_consumo_absoluto_pct": 1.78,
                "ga_delta_caudal_pct": 10.62,
                "ga_cumplimiento_t_out_pct": 100.0,
                "ga_config": f"pop={GA_POP_SIZE}, gen={GA_N_GEN}, cxpb={GA_CXPB}, "
                             f"mutpb={GA_MUTPB}",
                "aviso": "Métricas calculadas sobre datos sintéticos (simulador físico); "
                         "validación con datos reales de planta pendiente.",
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
