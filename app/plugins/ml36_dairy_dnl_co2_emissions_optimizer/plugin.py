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
    TARGETS,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_PATIENCE,
    TRAIN_SEED,
    TRAIN_VAL_FRACTION,
    VERSION,
)
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.ga_optimizer import (
    predict_scenario,
    run_adaptive_ga,
)
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.mlflow_utils import (
    download_user_model_from_mlflow,
)
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.model_loader import (
    build_model_from_config,
    load_artifacts,
)
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
    PredictOptimizeResponse,
)
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.preprocessing import build_feature_frame
from app.plugins.ml36_dairy_dnl_co2_emissions_optimizer.train_dto import TrainResponse

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
        saved = (self._model, self._scaler_X, self._scaler_Y, self._config)
        if mlflow_run_id:
            logger.info("predict_inline — using user model from MLflow run_id=%s", mlflow_run_id)
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._model, self._scaler_X, self._scaler_Y, self._config, user_temp_dir = loaded
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
                self._model, self._scaler_X, self._scaler_Y, self._config = saved

    def _run_predict(self, features: dict) -> PredictInlineResponse:
        """MLP digital-twin single-sample inference in real units."""
        row = pd.DataFrame([{f: float(features[f]) for f in BASE_FEATURES}])
        x_scaled = self._scaler_X.transform(build_feature_frame(row))
        with torch.no_grad():
            y_scaled = self._model(torch.tensor(x_scaled, dtype=torch.float32)).cpu().numpy()
        y_real = self._scaler_Y.inverse_transform(y_scaled)[0]
        return PredictInlineResponse(
            model_id=MODEL_ID,
            T_out_pred=round(float(y_real[T_OUT_IDX]), 4),
            CO2_emissions_pred=round(float(y_real[CO2_IDX]), 4),
        )

    def _run_optimize(self, features: dict) -> PredictOptimizeResponse:
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
            self._model, self._scaler_X, self._scaler_Y, context, policy)

        selected = list(policy)
        used_adaptive = decision_mode in ("adaptive", "hybrid")
        if used_adaptive:
            seeds = [list(policy)] + ([current] if current else [])
            selected = run_adaptive_ga(
                self._model, self._scaler_X, self._scaler_Y, context, seeds, seed)

        opt_co2, opt_t_out = predict_scenario(
            self._model, self._scaler_X, self._scaler_Y, context, selected)
        feasible = opt_t_out >= T_OUT_MIN
        if not feasible:
            raise ThermalSafetyViolationError(
                f"La recomendación no cumple la restricción térmica para el escenario "
                f"{context}: T_out predicha = {opt_t_out:.2f} °C < {T_OUT_MIN} °C. "
                f"Revise que el escenario esté dentro del rango de entrenamiento."
            )

        current_co2 = current_t_out = saving_vs_current = None
        if current is not None:
            current_co2, current_t_out = predict_scenario(
                self._model, self._scaler_X, self._scaler_Y, context, current)
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
        user_temp_dir = None
        saved = (self._model, self._scaler_X, self._scaler_Y, self._config)
        if mlflow_run_id:
            logger.info("predict_batch — using user model from MLflow run_id=%s", mlflow_run_id)
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._model, self._scaler_X, self._scaler_Y, self._config, user_temp_dir = loaded
        try:
            self._require_loaded()
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)

            if model_key == "optimize":
                predictions = self._predict_batch_optimize(df)
            else:
                predictions = self._predict_batch_estimate(df)

            self._record()
            logger.info(
                "predict_batch done — %d rows, model_key=%s, mlflow=%s",
                len(predictions), model_key or "(estimate)", bool(mlflow_run_id),
            )
            return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)
                self._model, self._scaler_X, self._scaler_Y, self._config = saved

    def _predict_batch_estimate(self, df: pd.DataFrame) -> list[dict]:
        """MLP digital-twin inference, one CSV row = one sample."""
        missing = [c for c in BASE_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"CSV falta columnas requeridas: {missing}")
        predictions: list[dict] = []
        for idx, row in df.iterrows():
            try:
                row_dict = row.to_dict()
                pred = self._run_predict(row_dict)
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

    def _predict_batch_optimize(self, df: pd.DataFrame) -> list[dict]:
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
                pred = self._run_optimize(features).model_dump()
                pred.update({"row": int(idx), **{c: features[c] for c in CONTEXT_COLS}})
                predictions.append(pred)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Error en fila %s (optimize): %s", idx, exc)
                predictions.append({"row": int(idx), "error": str(exc)})
        return predictions

    # ── train ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _split_validation(df: pd.DataFrame) -> int:
        """Return the index where the validation tail starts (rows ``[idx:]`` are hold-out).

        With a ``cycle_id`` column whole cycles stay together (last 15% of cycles);
        otherwise the last 15% of rows. Tiny datasets validate on the training data.
        """
        n = len(df)
        if "cycle_id" in df.columns:
            cycles = df["cycle_id"].drop_duplicates().tolist()
            if len(cycles) >= 3:
                n_val_cycles = max(1, int(round(len(cycles) * TRAIN_VAL_FRACTION)))
                val_cycles = set(cycles[-n_val_cycles:])
                first_val = int(np.argmax(df["cycle_id"].isin(val_cycles).to_numpy()))
                if 0 < first_val < n:
                    return first_val
        if n >= 10:
            return n - max(1, int(n * TRAIN_VAL_FRACTION))
        return n

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> TrainResponse:  # noqa: C901  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
        """Fine-tune the MLP on user data (8 base features + 2 targets required).

        Follows the original recipe (training_model1.final_model): Adam lr=1e-4,
        MSELoss over both targets, batch_size=128, max 400 epochs, early-stopping
        patience 30 on a validation hold-out (last 15% of cycles/rows).
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

        tracker: BaseMLflowTracker | None = None
        if mlflow_run_id:
            tracker = BaseMLflowTracker(mlflow_run_id)
            tracker.log_params({
                "epochs_max": TRAIN_EPOCHS, "lr": TRAIN_LR, "optimizer": "Adam",
                "batch_size": TRAIN_BATCH_SIZE, "patience": TRAIN_PATIENCE,
            })

        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)
        missing = [c for c in BASE_FEATURES + TARGETS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV falta columnas requeridas: {missing}")
        if len(df) < 2:
            raise ValueError("Se necesitan al menos 2 filas para el fine-tuning.")

        self._require_loaded()
        torch.manual_seed(TRAIN_SEED)
        np.random.seed(TRAIN_SEED)

        x_scaled = self._scaler_X.transform(build_feature_frame(df))
        y_scaled = self._scaler_Y.transform(df[TARGETS].values)

        split = self._split_validation(df)
        if split < len(df):
            x_tr, y_tr = x_scaled[:split], y_scaled[:split]
            x_val, y_val = x_scaled[split:], y_scaled[split:]
        else:
            x_tr, y_tr = x_scaled, y_scaled
            x_val, y_val = x_scaled, y_scaled

        # Clone weights into a new model instance to avoid mutating the live model
        fine_model = build_model_from_config(self._config)
        fine_model.load_state_dict(self._model.state_dict())

        optimizer = optim.Adam(fine_model.parameters(), lr=TRAIN_LR)
        criterion = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(x_tr), torch.FloatTensor(y_tr)),
            batch_size=TRAIN_BATCH_SIZE, shuffle=True,
            # BatchNorm cannot train on a batch of one sample
            drop_last=len(x_tr) % TRAIN_BATCH_SIZE == 1 and len(x_tr) > 1,
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
        y_pred = self._scaler_Y.inverse_transform(y_pred_scaled)
        y_real = df[TARGETS].values

        metrics: dict[str, float] = {}
        for i, key in enumerate(("t_out", "co2")):
            err = y_real[:, i] - y_pred[:, i]
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((y_real[:, i] - np.mean(y_real[:, i])) ** 2))
            metrics[f"mae_{key}"] = float(np.mean(np.abs(err)))
            metrics[f"mse_{key}"] = float(np.mean(err ** 2))
            metrics[f"r2_{key}"] = float(1 - ss_res / ss_tot) if ss_tot else 0.0

        # The retrained model lives only in its own MLflow run (predict with that
        # mlflow_run_id); the served base model and its local artifacts are never replaced.
        upload_warning = None
        if not tracker:
            upload_warning = "Sin run de MLflow: el modelo reentrenado no se ha guardado."
        else:
            tracker.log_metrics({**metrics, "n_samples": len(df)})
            mlflow_tmp = None
            try:
                mlflow_tmp = tempfile.mkdtemp(prefix="ml36_mlflow_")
                torch.save(fine_model.state_dict(), os.path.join(mlflow_tmp, MODEL_FILENAME))
                with open(os.path.join(mlflow_tmp, MODEL_CONFIG_FILENAME), "w", encoding="utf-8") as f:
                    json.dump(self._config, f, indent=4)
                joblib.dump(self._scaler_X, os.path.join(mlflow_tmp, SCALER_X_FILENAME))
                joblib.dump(self._scaler_Y, os.path.join(mlflow_tmp, SCALER_Y_FILENAME))
                tracker.upload_artifacts(mlflow_tmp, artifact_path="model")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("MLflow artifact upload failed: %s", exc)
                upload_warning = f"El modelo reentrenado no se ha podido guardar en MLflow: {exc}"
            finally:
                if mlflow_tmp:
                    shutil.rmtree(mlflow_tmp, ignore_errors=True)

        logger.info(
            "train() done — mae_co2=%.4f r2_co2=%.4f n=%d epochs=%d mlflow=%s",
            metrics["mae_co2"], metrics["r2_co2"], len(df), epochs_executed, bool(mlflow_run_id),
        )
        return TrainResponse(
            detail="Fine-tuning completado",
            mae_t_out=round(metrics["mae_t_out"], 4),
            mse_t_out=round(metrics["mse_t_out"], 6),
            r2_t_out=round(metrics["r2_t_out"], 4),
            mae_co2=round(metrics["mae_co2"], 4),
            mse_co2=round(metrics["mse_co2"], 6),
            r2_co2=round(metrics["r2_co2"], 4),
            n_samples=int(len(df)),
            epochs_executed=int(epochs_executed),
            upload_warning=upload_warning,
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
                "recomienda setpoints (T_serv, Delta_P, Regeneration_perc) con restricción "
                "T_out >= 72.5 °C."
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
                           description="Viscosidad dinámica de la leche [inline y optimize]"),
                InputField(name="t_ciclo", type="float",
                           description="Tiempo desde el inicio del ciclo (min) [inline y optimize]"),
                InputField(name="T_serv", type="float",
                           description="Temperatura del líquido de calentamiento (°C) "
                                       "[inline; en optimize opcional = operación actual]"),
                InputField(name="Delta_P", type="float",
                           description="Presión de impulsión de las bombas "
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
