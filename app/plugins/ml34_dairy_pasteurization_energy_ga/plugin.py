"""Ml34DairyPasteurizationEnergyGaPlugin — MLP surrogate + single-objective GA (DEAP)
for energy optimization of dairy pasteurization.

Inline mode predicts (E_consumo, T_out_leche) with the MLP digital twin.
Optimize mode runs the GA v4 per scenario: minimizes the specific consumption
E_consumo/F_flow subject to the food-safety constraint T_out >= 72.3 °C.

Response field names like E_consumo_pred intentionally mirror the original
codebase and its artifacts.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from app.application.dto.stats_dto import InputField, OutputField, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ThermalSafetyViolationError
from app.domain.services.mlflow_artifacts import (
    add_mlflow_run_to_stats,
    removing_dir,
    require_train_tracker,
    upload_model_to_mlflow,
)
from app.domain.services.plugin_runtime import PluginRuntimeMixin
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
    SCENARIO_FEATURES,
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


def _xai_values_from_features(data: dict) -> dict[str, float] | None:
    """Build xai_feature_values from a predict_batch row dict.

    Without this, predict_batch rows only carry row/E_consumo_pred/T_out_pred —
    the platform's XAI-context builder filters those out as known output columns,
    is left with zero input features, and silently skips calling the explain
    service (no error, no request — just no explanation). See
    ml35_dairy_ann_cleaning_cost's _xai_values_from_features for the same fix.
    """
    xai: dict[str, float] = {}
    for f in FEATURES:
        val = data.get(f)
        if val is not None and pd.notna(val):
            try:
                xai[f] = float(val)
            except (TypeError, ValueError):
                pass
    return xai or None


def _ga_best_individual(ctx: tuple, *, t_in: float, t_ciclo: float, delta_p: float, seed: int):
    """Run the GA v4 for one scenario (``ctx = (model, scaler_X, scaler_y)``); best individual."""
    model, scaler_x, scaler_y = ctx
    toolbox = setup_ga_toolbox()
    hof = run_ga_single(
        toolbox, T_in_leche=t_in, t_ciclo=t_ciclo, Delta_P=delta_p,
        model=model, scaler_X=scaler_x, scaler_y=scaler_y,
        seed=seed,
    )
    return hof[0]


def _load_train_frame(data_path: str) -> pd.DataFrame:
    """Read the training CSV and check it has the 5 features and the 2 targets."""
    with local_file_path(data_path) as local_path:
        df = pd.read_csv(local_path)
    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV falta columnas requeridas: {missing}")
    return df


def _temporal_holdout(x_scaled: np.ndarray, y_scaled: np.ndarray) -> tuple:
    """Split off the last 15% of rows for early stopping: (x_tr, y_tr, x_val, y_val).

    With fewer than 10 rows there is no hold-out and the training set is also the
    validation set.
    """
    n = len(x_scaled)
    n_val = max(1, int(n * 0.15)) if n >= 10 else 0
    if n_val:
        return x_scaled[:-n_val], y_scaled[:-n_val], x_scaled[-n_val:], y_scaled[-n_val:]
    return x_scaled, y_scaled, x_scaled, y_scaled


def _target_metrics(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE and R² per target (keys ``rmse_<target>``, ``mae_<target>``, ``r2_<target>``)."""
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
    return metrics


def _train_one_epoch(model, loader, optimizer, criterion) -> None:
    """One pass over the training batches (train mode, Adam step per batch)."""
    model.train()
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()


def _train_with_early_stopping(model, loader, *, optimizer, criterion, val: tuple) -> int:
    """Train up to TRAIN_EPOCHS epochs, stopping after TRAIN_PATIENCE without improvement.

    ``val`` is ``(x_val_t, y_val_t)``. Leaves ``model`` with the weights of the best
    validation epoch and returns the number of epochs executed.
    """
    x_val_t, y_val_t = val
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    epochs_executed = 0
    for epoch in range(TRAIN_EPOCHS):
        _train_one_epoch(model, loader, optimizer, criterion)
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val_t), y_val_t).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        epochs_executed = epoch + 1
        if no_improve >= TRAIN_PATIENCE:
            break

    model.load_state_dict(best_state)
    return epochs_executed


class Ml34DairyPasteurizationEnergyGaPlugin(PluginRuntimeMixin, ModelPluginPort):
    """MLP digital twin + DEAP single-objective GA for pasteurization setpoints."""

    def __init__(self) -> None:
        self._model: Any = None
        self._scaler_x: Any = None
        self._scaler_y: Any = None
        self._config: dict | None = None

    def load(self) -> None:
        self._model, self._scaler_x, self._scaler_y, self._config = load_artifacts()
        logger.info("Ml34DairyPasteurizationEnergyGaPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        return self._model is not None

    # ── predict_inline ────────────────────────────────────────────────────────

    def _resolve_model(self, mlflow_run_id: str) -> tuple[tuple, str | None]:
        """Return ((model, scaler_X, scaler_y), temp_dir) to serve this request.

        The user's retrained model (if mlflow_run_id) is returned, never assigned to self:
        the plugin instance is shared by concurrent requests, so swapping it into self leaked
        a user's model to other requests — and, with two overlapping runs, left it in place
        as the "base" model. temp_dir is None for the base model.
        """
        if not mlflow_run_id:
            self._require_loaded()
            return (self._model, self._scaler_x, self._scaler_y), None
        model, scaler_x, scaler_y, _config, user_temp_dir = download_user_model_from_mlflow(
            mlflow_run_id
        )
        return (model, scaler_x, scaler_y), user_temp_dir

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse | PredictOptimizeResponse:
        """Dispatch to MLP predict or GA optimize based on model_key."""
        if mlflow_run_id:
            logger.info("predict_inline — using user model from MLflow run_id=%s", mlflow_run_id)
        ctx, user_temp_dir = self._resolve_model(mlflow_run_id)
        with removing_dir(user_temp_dir):
            if model_key == "optimize":
                result = self._run_optimize(features, ctx)
            else:
                result = self._run_predict(features, ctx)
            self._record()
            return result

    def _run_predict(self, features: dict, ctx: tuple) -> PredictInlineResponse:
        """MLP surrogate single-sample inference in real units."""
        model, scaler_x, scaler_y = ctx
        e_consumo, t_out = predict_scenario(
            model, scaler_x, scaler_y,
            float(features["T_in_leche"]), float(features["F_flow"]),
            float(features["T_servicio"]), float(features["t_ciclo"]),
            float(features["Delta_P"]),
        )
        return PredictInlineResponse(
            model_id=MODEL_ID,
            E_consumo_pred=round(e_consumo, 4),
            T_out_pred=round(t_out, 4),
        )

    def _run_optimize(self, features: dict, ctx: tuple) -> PredictOptimizeResponse:
        """Run the single-objective GA for one scenario (deterministic per seed)."""
        t_in = float(features["T_in_leche"])
        delta_p = float(features["Delta_P"])
        t_ciclo = float(features["t_ciclo"])
        seed = int(features.get("seed", 1))

        best = _ga_best_individual(ctx, t_in=t_in, t_ciclo=t_ciclo, delta_p=delta_p, seed=seed)
        f_flow, t_servicio = float(best[0]), float(best[1])
        e_consumo, t_out = predict_scenario(
            *ctx,
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

    def predict_batch(
        self, *, data_path: str, model_key: str | None = None, mlflow_run_id: str = "",
    ) -> PredictBatchResponse:
        """Batch inference over a CSV file — MLP estimate, or GA optimize per row.

        Same GA-vs-MLP dispatch as predict_inline, keyed off model_key: "optimize"
        runs one GA search per CSV row (reusing _run_optimize, the same code path
        the inline endpoint uses), anything else runs the MLP surrogate.
        """
        if mlflow_run_id:
            logger.info("predict_batch — using user model from MLflow run_id=%s", mlflow_run_id)
        ctx, user_temp_dir = self._resolve_model(mlflow_run_id)
        with removing_dir(user_temp_dir):
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)

            if model_key == "optimize":
                predictions = self._predict_batch_optimize(df, ctx)
            else:
                predictions = self._predict_batch_estimate(df, ctx)

            self._record()
            logger.info(
                "predict_batch done — %d rows, model_key=%s, mlflow=%s",
                len(predictions), model_key or "(estimate)", bool(mlflow_run_id),
            )
            return PredictBatchResponse(
                model_id=MODEL_ID, predictions=predictions, output_path=None
            )

    def _predict_batch_estimate(self, df: pd.DataFrame, ctx: tuple) -> list[dict]:
        """MLP surrogate inference, one row of the CSV = one scenario."""
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"CSV falta columnas requeridas: {missing}")
        predictions: list[dict] = []
        for idx, row in df.iterrows():
            try:
                row_dict = row.to_dict()
                pred = self._run_predict(row_dict, ctx)
                predictions.append({
                    "row": int(idx),
                    "E_consumo_pred": pred.E_consumo_pred,
                    "T_out_pred": pred.T_out_pred,
                    "xai_feature_values": _xai_values_from_features(row_dict),
                })
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Error en fila %s: %s", idx, exc)
                predictions.append({"row": int(idx), "error": str(exc)})
        return predictions

    def _predict_batch_optimize(self, df: pd.DataFrame, ctx: tuple) -> list[dict]:
        """GA per scenario row — same per-row loop as the delivered
        src/main.py::optimize(), reusing _run_optimize (the inline GA path) per row.

        Row seed: an explicit 'seed' column overrides it per row; otherwise
        ``1 + row_index``, matching optimize()'s ``seed=seed_base+idx`` default.

        The row echoes T_in_leche/Delta_P/t_ciclo (the scenario, not part of
        PredictOptimizeResponse) alongside IA_F_flow/IA_T_servicio/IA_E_consumo/
        IA_T_out. The GA search itself isn't SHAP-explainable (stochastic
        population search, not a deterministic function), but IA_F_flow/
        IA_T_servicio is just another point in the same 5-feature MLP space the
        "estimate" sub-mode already models — the platform explains *that* point
        post-hoc, the same way it does for inline optimize (see
        dairy_pasteurization_energy in predict-task-manager.ts), which is why
        the 3 scenario inputs need to survive into the row.
        """
        missing = [c for c in SCENARIO_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"CSV falta columnas requeridas: {missing}")
        has_seed_col = "seed" in df.columns
        predictions: list[dict] = []
        for idx, row in df.iterrows():
            try:
                t_in, delta_p, t_ciclo = (
                    float(row["T_in_leche"]), float(row["Delta_P"]), float(row["t_ciclo"])
                )
                seed = int(row["seed"]) if has_seed_col and pd.notna(row["seed"]) else 1 + int(idx)
                features = {
                    "T_in_leche": t_in, "Delta_P": delta_p, "t_ciclo": t_ciclo, "seed": seed
                }
                pred = self._run_optimize(features, ctx).model_dump()
                pred.update({
                    "row": int(idx), "T_in_leche": t_in, "Delta_P": delta_p, "t_ciclo": t_ciclo
                })
                predictions.append(pred)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Error en fila %s (optimize): %s", idx, exc)
                predictions.append({"row": int(idx), "error": str(exc)})
        return predictions

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Fine-tune the MLP on user data (5 features + 2 targets required).

        Follows the original training recipe (src/main.py::train): Adam
        lr=0.0005, MSELoss over both targets, batch_size=128, max 300 epochs
        with early-stopping patience 15 monitored on a temporal 15% hold-out.
        """
        # The retrained model lives only in its own MLflow run (predict with that
        # mlflow_run_id); the served base model and its local artifacts are never replaced.
        tracker = require_train_tracker(mlflow_run_id)
        tracker.log_params({
            "epochs_max": TRAIN_EPOCHS, "lr": TRAIN_LR, "optimizer": "Adam",
            "batch_size": TRAIN_BATCH_SIZE, "patience": TRAIN_PATIENCE,
        })

        df = _load_train_frame(data_path)

        self._require_loaded()
        torch.manual_seed(TRAIN_SEED)
        np.random.seed(TRAIN_SEED)

        x_scaled = self._scaler_x.transform(df[FEATURES].values)
        y_scaled = self._scaler_y.transform(df[TARGETS].values)
        fine_model, epochs_executed = self._fine_tune(*_temporal_holdout(x_scaled, y_scaled))

        with torch.no_grad():
            y_pred_scaled = fine_model(torch.FloatTensor(x_scaled)).numpy()
        metrics = _target_metrics(
            df[TARGETS].values, self._scaler_y.inverse_transform(y_pred_scaled)
        )

        tracker.log_metrics({**metrics, "n_samples": len(df)})

        def _write(model_dir: str) -> None:
            torch.save(fine_model.state_dict(), os.path.join(model_dir, MODEL_FILENAME))
            with open(
                os.path.join(model_dir, MODEL_CONFIG_FILENAME), "w", encoding="utf-8"
            ) as f:
                json.dump(self._config, f, indent=4)
            joblib.dump(self._scaler_x, os.path.join(model_dir, SCALER_X_FILENAME))
            joblib.dump(self._scaler_y, os.path.join(model_dir, SCALER_Y_FILENAME))

        upload_model_to_mlflow(tracker, _write, prefix="ml34_mlflow_")

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
            upload_warning=None,
        )

    def _fine_tune(self, x_tr, y_tr, x_val, y_val) -> tuple[Any, int]:
        """Fine-tune a copy of the served MLP (never the live model) with early stopping.

        Returns ``(model with the best validation weights, epochs executed)``.
        """
        from torch import nn, optim  # pylint: disable=import-outside-toplevel
        from torch.utils.data import (  # pylint: disable=import-outside-toplevel
            DataLoader,
            TensorDataset,
        )

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
        epochs_executed = _train_with_early_stopping(
            fine_model, loader, optimizer=optimizer, criterion=criterion,
            val=(torch.FloatTensor(x_val), torch.FloatTensor(y_val)),
        )
        fine_model.eval()
        return fine_model, epochs_executed

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
            runtime_stats=self._runtime_stats(),
        )
        return add_mlflow_run_to_stats(base, mlflow_run_id)
