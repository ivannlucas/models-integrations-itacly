"""Ml36DairyDnlCo2EmissionsOptimizerPlugin — MLP digital twin + per-instance GA (DEAP)
to reduce CO2 emissions in dairy pasteurization (T_out >= 72.5 °C).

Inline mode predicts (T_out, CO2_emissions) with the MLP digital twin.
Optimize mode (model_key="optimize") recommends (T_serv, Delta_P, Regeneration_perc)
by static policy, adaptive GA or hybrid (default), as in the delivered
realtime_decision_pipeline.

Attribute/variable names like _scaler_X intentionally mirror the original codebase.
"""
# pylint: disable=invalid-name
from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError, ModelPersistenceError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.constants import (
    BASE_FEATURES,
    CONTEXT_COLS,
    DECISION_MODES,
    DEFAULT_DECISION_MODE,
    DEFAULT_ROW_SEED,
    FRAMEWORK,
    GA_CXPB,
    GA_MUTPB,
    GA_N_GEN,
    GA_POP_SIZE,
    GENE_ORDER,
    MODEL_CONFIG_FILENAME,
    MODEL_FILENAME,
    MODEL_ID,
    SCALER_X_FILENAME,
    SCALER_Y_FILENAME,
    T_OUT_IDX,
    T_OUT_MIN,
    CO2_IDX,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_PATIENCE,
    VERSION,
)
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.ga_optimizer import (
    predict_scenario,
    run_adaptive_ga,
)
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.mlflow_utils import (
    download_user_model_from_mlflow,
)
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.model_loader import load_artifacts
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
    PredictOptimizeResponse,
)
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.preprocessing import build_feature_frame
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.train_dto import TrainResponse
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.training import train_model

logger = logging.getLogger(__name__)


def _xai_values_from_features(data: dict) -> dict[str, float] | None:
    """Build xai_feature_values from a predict_batch row dict.

    Without this, predict_batch rows only carry row/T_out_pred/CO2_emissions_pred —
    the platform's XAI-context builder would be left with zero input features and
    silently skip the explain service (same fix as ml34/ml35).
    """
    xai: dict[str, float] = {}
    for f in BASE_FEATURES:
        val = data.get(f)
        if val is not None and pd.notna(val):
            try:
                xai[f] = float(val)
            except (TypeError, ValueError):
                pass
    return xai or None


def _present(value: Any) -> bool:
    """True when a CSV/DTO value is neither None nor NaN."""
    return value is not None and not pd.isna(value)


class Ml36DairyDnlCo2EmissionsOptimizerPlugin(ModelPluginPort):
    """MLP digital twin + DEAP per-instance GA for pasteurization setpoints."""

    def __init__(self) -> None:
        self._model: Any = None
        self._scaler_X: Any = None
        self._scaler_Y: Any = None
        self._config: dict | None = None
        self._policy: list[float] | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        self._model, self._scaler_X, self._scaler_Y, self._config, self._policy = load_artifacts()
        logger.info("Ml36DairyDnlCo2EmissionsOptimizerPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        return self._model is not None

    def _require_loaded(self) -> None:
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _resolve_model(self, mlflow_run_id: str) -> tuple[tuple, str | None]:
        """(model, scaler_X, scaler_Y) for this request and the temp dir to remove afterwards.

        The user's model is resolved into locals, never into self (shared across concurrent
        requests), and a run without a loadable model raises — never falls back to the base.
        """
        self._require_loaded()  # the GA policy always comes from the base artifacts
        if not mlflow_run_id:
            return (self._model, self._scaler_X, self._scaler_Y), None
        logger.info("Using user model from MLflow run_id=%s", mlflow_run_id)
        model, scaler_x, scaler_y, _config, temp_dir = download_user_model_from_mlflow(mlflow_run_id)
        return (model, scaler_x, scaler_y), temp_dir

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
        bundle, user_temp_dir = self._resolve_model(mlflow_run_id)
        try:
            if model_key == "optimize":
                result = self._run_optimize(features, bundle)
            else:
                result = self._run_predict(features, bundle)
            self._record()
            return result
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)

    @staticmethod
    def _run_predict(features: dict, bundle: tuple) -> PredictInlineResponse:
        """MLP digital-twin single-sample inference in real units."""
        model, scaler_x, scaler_y = bundle
        row = pd.DataFrame([{f: float(features[f]) for f in BASE_FEATURES}])
        x_scaled = scaler_x.transform(build_feature_frame(row))
        with torch.no_grad():
            y_scaled = model(torch.tensor(x_scaled, dtype=torch.float32)).cpu().numpy()
        y_real = scaler_y.inverse_transform(y_scaled)[0]
        return PredictInlineResponse(
            model_id=MODEL_ID,
            T_out_pred=round(float(y_real[T_OUT_IDX]), 4),
            CO2_emissions_pred=round(float(y_real[CO2_IDX]), 4),
        )

    def _run_optimize(self, features: dict, bundle: tuple) -> PredictOptimizeResponse:
        """Static / adaptive / hybrid decision for one context row (deterministic per seed)."""
        decision_mode = str(features.get("decision_mode") or DEFAULT_DECISION_MODE).lower()
        if decision_mode not in DECISION_MODES:
            raise ValueError(f"decision_mode debe ser uno de {DECISION_MODES}, recibido: {decision_mode!r}")
        seed = int(features.get("seed", DEFAULT_ROW_SEED))
        context = {c: float(features[c]) for c in CONTEXT_COLS}

        current = None
        if all(_present(features.get(g)) for g in GENE_ORDER):
            current = [float(features[g]) for g in GENE_ORDER]

        policy = self._policy
        policy_co2, policy_t_out = predict_scenario(
            *bundle, context, policy)

        selected = list(policy)
        used_adaptive = decision_mode in ("adaptive", "hybrid")
        if used_adaptive:
            seeds = [list(policy)] + ([current] if current else [])
            selected = run_adaptive_ga(
                *bundle, context, seeds, seed)

        opt_co2, opt_t_out = predict_scenario(
            *bundle, context, selected)
        # As in the delivered pipeline, T_out >= T_OUT_MIN is only a soft penalty inside the
        # GA: the best individual is returned as-is (confirmed by the AI team). The platform
        # must check recommended_T_out_pred before applying the setpoints; the flag below
        # is informational only.
        feasible = opt_t_out >= T_OUT_MIN
        if not feasible:
            logger.warning(
                "optimize — recomendación con T_out=%.2f °C < %.1f °C (devuelta sin rechazo) para %s",
                opt_t_out, T_OUT_MIN, context,
            )

        current_co2 = current_t_out = saving_vs_current = None
        if current is not None:
            current_co2, current_t_out = predict_scenario(
                *bundle, context, current)
            saving_vs_current = round(current_co2 - opt_co2, 4)
            current_co2, current_t_out = round(current_co2, 4), round(current_t_out, 4)

        logger.info(
            "optimize done — mode=%s seed=%d setpoints=(%.2f, %.3f, %.2f) CO2=%.3f T_out=%.2f",
            decision_mode, seed, selected[0], selected[1], selected[2], opt_co2, opt_t_out,
        )
        return PredictOptimizeResponse(
            model_id=MODEL_ID,
            decision_mode=decision_mode,
            used_adaptive_ga=used_adaptive,
            seed=seed,
            recommended_T_serv=round(selected[0], 4),
            recommended_Delta_P=round(selected[1], 4),
            recommended_Regeneration_perc=round(selected[2], 4),
            recommended_CO2_pred=round(opt_co2, 4),
            recommended_T_out_pred=round(opt_t_out, 4),
            recommended_factible=feasible,
            policy_T_serv=round(policy[0], 4),
            policy_Delta_P=round(policy[1], 4),
            policy_Regeneration_perc=round(policy[2], 4),
            policy_CO2_pred=round(policy_co2, 4),
            policy_T_out_pred=round(policy_t_out, 4),
            co2_saving_vs_policy=round(policy_co2 - opt_co2, 4),
            current_CO2_pred=current_co2,
            current_T_out_pred=current_t_out,
            co2_saving_vs_current=saving_vs_current,
        )

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(
        self, *, data_path: str, model_key: str | None = None, mlflow_run_id: str = "",
    ) -> PredictBatchResponse:
        """Batch inference over a CSV file — MLP estimate, or GA optimize per row.

        Same dispatch as predict_inline, keyed off model_key: "optimize" runs one
        decision per CSV row (reusing _run_optimize), anything else runs the MLP.
        """
        bundle, user_temp_dir = self._resolve_model(mlflow_run_id)
        try:
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)

            if model_key == "optimize":
                predictions = self._predict_batch_optimize(df, bundle)
            else:
                predictions = self._predict_batch_estimate(df, bundle)

            self._record()
            logger.info(
                "predict_batch done — %d rows, model_key=%s, mlflow=%s",
                len(predictions), model_key or "(estimate)", bool(mlflow_run_id),
            )
            return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)

    def _predict_batch_estimate(self, df: pd.DataFrame, bundle: tuple) -> list[dict]:
        """MLP digital-twin inference, one CSV row = one sample."""
        missing = [c for c in BASE_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"CSV falta columnas requeridas: {missing}")
        predictions: list[dict] = []
        for idx, row in df.iterrows():
            try:
                row_dict = row.to_dict()
                pred = self._run_predict(row_dict, bundle)
                predictions.append({
                    "row": int(idx),
                    "T_out_pred": pred.T_out_pred,
                    "CO2_emissions_pred": pred.CO2_emissions_pred,
                    "xai_feature_values": _xai_values_from_features(row_dict),
                })
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Error en fila %s: %s", idx, exc)
                predictions.append({"row": int(idx), "error": str(exc)})
        return predictions

    def _predict_batch_optimize(self, df: pd.DataFrame, bundle: tuple) -> list[dict]:
        """One decision per context row — same per-row loop as the delivered
        realtime_decision_pipeline (row seed = 42 + i, i from 1 → 43 + row index).

        Optional columns: ``seed`` (overrides the row seed), ``decision_mode`` and the
        3 current setpoints (T_serv, Delta_P, Regeneration_perc; all-or-none per row).
        The context is echoed onto each row because it is not part of the GA response
        and the platform's batch XAI context builder needs it.
        """
        missing = [c for c in CONTEXT_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV falta columnas requeridas: {missing}")
        predictions: list[dict] = []
        for idx, row in df.iterrows():
            try:
                features = {c: float(row[c]) for c in CONTEXT_COLS}
                features["seed"] = (
                    int(row["seed"]) if "seed" in df.columns and _present(row["seed"])
                    else DEFAULT_ROW_SEED + int(idx)
                )
                if "decision_mode" in df.columns and _present(row["decision_mode"]):
                    features["decision_mode"] = str(row["decision_mode"])
                for g in GENE_ORDER:
                    if g in df.columns and _present(row[g]):
                        features[g] = float(row[g])
                pred = self._run_optimize(features, bundle).model_dump()
                pred.update({"row": int(idx), **{c: features[c] for c in CONTEXT_COLS}})
                predictions.append(pred)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Error en fila %s (optimize): %s", idx, exc)
                predictions.append({"row": int(idx), "error": str(exc)})
        return predictions

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Retrain the MLP with the AI team's procedure (see training.py).

        Split by cycle (balanced by season), scalers refitted on train, a new 1x64 MLP trained from
        scratch with early stopping, metrics on the test split in real units. The model and its
        scalers live only in their MLflow run (predict with that mlflow_run_id); the served model
        and its artifacts are never replaced.
        """
        import json  # pylint: disable=import-outside-toplevel
        import tempfile  # pylint: disable=import-outside-toplevel
        import joblib  # pylint: disable=import-outside-toplevel

        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)
        result = train_model(df)
        metrics = result["metrics"]
        flat = {
            f"{name}_{key}": metrics[target][name]
            for target, key in (("T_out", "t_out"), ("CO2_emissions", "co2"))
            for name in ("mae", "mse", "r2")
        }

        mlflow_tmp = tempfile.mkdtemp(prefix="ml36_mlflow_")
        try:
            tracker = BaseMLflowTracker(mlflow_run_id)
            tracker.log_params({
                "epochs_max": TRAIN_EPOCHS, "lr": TRAIN_LR, "optimizer": "Adam",
                "batch_size": TRAIN_BATCH_SIZE, "patience": TRAIN_PATIENCE,
            })
            tracker.log_metrics({**flat, "n_train": result["n_train"], "n_val": result["n_val"],
                                 "n_test": result["n_test"], "epochs_run": result["epochs_run"]})
            torch.save(result["model"].state_dict(), os.path.join(mlflow_tmp, MODEL_FILENAME))
            with open(os.path.join(mlflow_tmp, MODEL_CONFIG_FILENAME), "w", encoding="utf-8") as f:
                json.dump(result["config"], f, indent=4)
            joblib.dump(result["scaler_x"], os.path.join(mlflow_tmp, SCALER_X_FILENAME))
            joblib.dump(result["scaler_y"], os.path.join(mlflow_tmp, SCALER_Y_FILENAME))
            tracker.upload_artifacts(mlflow_tmp, artifact_path="model")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("MLflow artifact upload failed: %s", exc)
            raise ModelPersistenceError(f"El modelo reentrenado no se ha podido guardar en MLflow: {exc}") from exc
        finally:
            shutil.rmtree(mlflow_tmp, ignore_errors=True)

        return TrainResponse(
            detail="Reentrenamiento completado (procedimiento train del equipo de IA); métricas sobre el split de test",
            mae_t_out=round(flat["mae_t_out"], 4),
            mse_t_out=round(flat["mse_t_out"], 6),
            r2_t_out=round(flat["r2_t_out"], 4),
            mae_co2=round(flat["mae_co2"], 4),
            mse_co2=round(flat["mse_co2"], 6),
            r2_co2=round(flat["r2_co2"], 4),
            n_samples=int(len(df)),
            n_train=result["n_train"],
            n_val=result["n_val"],
            n_test=result["n_test"],
            epochs_executed=result["epochs_run"],
            best_val_loss=result["best_val_loss"],
            training_time_s=result["training_time_s"],
            upload_warning=None,
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Gemelo digital MLP (PyTorch) + algoritmo genético por instancia (DEAP) para "
                "reducir las emisiones de CO2 en la pasteurización láctea sin comprometer la "
                "seguridad térmica. Modo inline: predice T_out y CO2_emissions. Modo optimize: "
                "recomienda setpoints (T_serv, Delta_P, Regeneration_perc) penalizando T_out < "
                "72.5 °C dentro del GA; la plataforma debe validar recommended_T_out_pred >= "
                "72.5 °C antes de aplicar los setpoints."
            ),
            task_type="regression_prescriptive",
            framework=FRAMEWORK,
            inputs=[
                InputField(name="F_milk", type="float",
                           description="Caudal volumétrico de leche (L/h) [inline y optimize]"),
                InputField(name="T_in", type="float",
                           description="Temperatura de entrada de la leche (°C) [inline y optimize]"),
                InputField(name="Fat_perc", type="float",
                           description="% de grasa del lote [inline y optimize]"),
                InputField(name="Viscosity", type="float",
                           description="Viscosidad dinámica de la leche (cP) [inline y optimize]"),
                InputField(name="t_ciclo", type="float",
                           description="Tiempo desde el inicio del ciclo (min) [inline y optimize]"),
                InputField(name="T_serv", type="float",
                           description="Temperatura del líquido de calentamiento (°C) "
                                       "[inline; en optimize opcional = operación actual]"),
                InputField(name="Delta_P", type="float",
                           description="Presión de impulsión de las bombas (bar) "
                                       "[inline; en optimize opcional = operación actual]"),
                InputField(name="Regeneration_perc", type="float",
                           description="Eficiencia de regeneración térmica (%) "
                                       "[inline; en optimize opcional = operación actual]"),
                InputField(name="decision_mode", type="str",
                           description="static | adaptive | hybrid (por defecto) [optimize]"),
                InputField(name="seed", type="int",
                           description="Semilla del GA adaptativo [optimize, opcional]"),
            ],
            outputs=[
                OutputField(name="T_out_pred", type="float",
                            description="Temperatura de salida predicha (°C) [modo inline]"),
                OutputField(name="CO2_emissions_pred", type="float",
                            description="Emisiones de CO2 predichas (kg) [modo inline]"),
                OutputField(name="recommended_T_serv", type="float",
                            description="Setpoint óptimo de T_serv (°C) [modo optimize]"),
                OutputField(name="recommended_Delta_P", type="float",
                            description="Setpoint óptimo de Delta_P [modo optimize]"),
                OutputField(name="recommended_Regeneration_perc", type="float",
                            description="Setpoint óptimo de regeneración (%) [modo optimize]"),
                OutputField(name="recommended_CO2_pred", type="float",
                            description="CO2 predicho con la recomendación (kg) [modo optimize]"),
                OutputField(name="recommended_T_out_pred", type="float",
                            description="T_out predicha con la recomendación (°C) [modo optimize]"),
                OutputField(name="co2_saving_vs_policy", type="float",
                            description="Ahorro de CO2 frente a la política global (kg) [modo optimize]"),
                OutputField(name="co2_saving_vs_current", type="float",
                            description="Ahorro de CO2 frente a la operación actual (kg), si se aporta "
                                        "[modo optimize]"),
            ],
            metrics={
                # MLP surrogate — test hold-out (final_test_metrics.json entregado)
                "T_out_mae_c": 0.019092,
                "T_out_mse": 0.000592,
                "T_out_r2": 0.999492,
                "CO2_mae_kg": 0.307587,
                "CO2_mse": 0.147651,
                "CO2_r2": 0.895145,
                "n_test": 7704,
                # GA en validación completa — models/metrics/ga/kpi_summary.csv entregado
                "ga_n_instancias_validacion": 7755,
                "ga_co2_original_kg": 156067.35,
                "ga_co2_optimizado_kg": 122231.46,
                "ga_mejora_co2_pct": 21.68,
                "ga_config": f"pop={GA_POP_SIZE}, gen={GA_N_GEN}, cxpb={GA_CXPB}, mutpb={GA_MUTPB}",
                "aviso": "Métricas calculadas sobre datos sintéticos (simulador físico de "
                         "intercambiador de placas); validación con datos reales de planta pendiente.",
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
