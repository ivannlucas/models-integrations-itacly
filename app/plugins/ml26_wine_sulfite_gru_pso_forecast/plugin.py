"""Ml26WineSulfiteGruPsoForecastPlugin — bidirectional GRU (hyperparameters found by PSO) that
forecasts, per wine tank, free SO2 and an underprotection risk score 72 h ahead from a 48 h window
of process readings, lab analytics and SO2 additions.

See inbox/a26/manifest.yaml for the full contract, golden cases and known_issues — in particular
KI-01: stage_progress is MANDATORY on every reading. The audited model was trained with it and
the AI team's fallback for a missing value triples the error; with it, the original preprocessing
reproduces the audited windows exactly. The preprocessing is the AI team's, unchanged.
"""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError, ModelPersistenceError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml26_wine_sulfite_gru_pso_forecast import (
    model_loader,
    postprocessing,
    preprocessing,
    training,
)
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.constants import (
    LOT_COLUMNS,
    MLFLOW_ARTIFACT_PATH,
    MODEL_ID,
    READING_OPTIONAL_COLUMNS,
    READING_REQUIRED_COLUMNS,
    REPORTED_METRICS,
    TARGET_RISK,
    TARGET_SO2,
    TASK_TYPE,
    TRAIN_HARD_REQUIRED_COLUMNS,
    VERSION,
    FRAMEWORK,
)
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.mlflow_utils import (
    download_user_model_from_mlflow,
)
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.predict_dto import (
    LotPrediction,
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml26_wine_sulfite_gru_pso_forecast.train_dto import TrainResponse

logger = logging.getLogger(__name__)


def _clean(value: Any) -> Any:
    """Convert numpy scalars to JSON-safe Python types."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        fv = float(value)
        return fv if math.isfinite(fv) else None
    return value


class Ml26WineSulfiteGruPsoForecastPlugin(ModelPluginPort):
    """GRU-PSO multitask regressor: free SO2 and underprotection risk at 72 h per wine tank."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with zeroed runtime counters."""
        self._model: model_loader.LoadedModel | None = None
        self._registry: dict = {}
        self._predict_count: int = 0
        self._total_latency_ms: float = 0.0
        self._last_predict_at: str | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the AI team's gru_pso.pkl bundle and best_model.json from the artifact store."""
        self._model, self._registry = model_loader.load_artifacts()
        logger.info("Ml26WineSulfiteGruPsoForecastPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        """Return True once the model bundle is loaded."""
        return self._model is not None

    def _resolve_model(self, mlflow_run_id: str) -> tuple[model_loader.LoadedModel, str | None]:
        """Return (model, temp_dir_to_cleanup). User models come from MLflow; else the fixed
        artifact.
        """
        if mlflow_run_id:
            # @require_user_model raises UserModelUnavailableError (→ 422) when the run has no
            # loadable model: never fall back to the base model.
            return download_user_model_from_mlflow(mlflow_run_id)
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")
        return self._model, None

    def _record(self, started: float) -> None:
        self._predict_count += 1
        self._total_latency_ms += (time.perf_counter() - started) * 1000.0
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    # ── shared inference ──────────────────────────────────────────────────────

    @staticmethod
    def _lot_predictions(
        model: model_loader.LoadedModel, lots: pd.DataFrame, readings: pd.DataFrame
    ) -> tuple[list[LotPrediction], pd.DataFrame]:
        raw = preprocessing.build_raw_frame(lots, readings)
        windows, meta, last_rows = preprocessing.prepare_windows(raw, model.feature_names)
        preds = postprocessing.predict_windows(model, windows)
        results = []
        for i, row in meta.iterrows():
            risk = float(preds[i, 1])
            results.append(
                LotPrediction(
                    lot_id=row["lot_id"],
                    timestamp=row["timestamp"],
                    timestamp_index=int(row["timestamp_index"]),
                    future_free_sulfite_72h=float(preds[i, 0]),
                    underprotection_risk_72h=risk,
                    risk_band=postprocessing.risk_band(risk),
                )
            )
        return results, last_rows

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(  # pylint: disable=too-many-locals
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Forecast one lot from its reading history (or from a pre-processed feature_window)."""
        _ = model_key, threshold  # regression model: no model variants, no decision threshold
        started = time.perf_counter()
        model, user_tmp = self._resolve_model(mlflow_run_id)
        try:
            if features.get("feature_window") is not None:
                window = preprocessing.validate_feature_window(
                    features["feature_window"], len(model.feature_names)
                )
                preds = postprocessing.predict_windows(model, window)
                risk = float(preds[0, 1])
                prediction = LotPrediction(
                    future_free_sulfite_72h=float(preds[0, 0]),
                    underprotection_risk_72h=risk,
                    risk_band=postprocessing.risk_band(risk),
                )
                xai = dict(zip(model.feature_names, (float(v) for v in window[0, -1])))
            else:
                lots, readings = preprocessing.frames_from_inline(features)
                predictions, last_rows = self._lot_predictions(model, lots, readings)
                prediction = predictions[0]
                xai = {k: _clean(v) for k, v in last_rows.iloc[0].items()}
            self._record(started)
            logger.info(
                "predict_inline done — lot=%s so2=%.3f risk=%.3f source=%s",
                prediction.lot_id,
                prediction.future_free_sulfite_72h,
                prediction.underprotection_risk_72h,
                model.source,
            )
            return PredictInlineResponse(
                **prediction.model_dump(),
                model_id=MODEL_ID,
                model_name=MODEL_ID,
                model_source=model.source,
                xai_feature_values=xai,
            )
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Forecast every lot in a readings CSV (lot columns repeated per row); one prediction per
        lot.
        """
        started = time.perf_counter()
        model, user_tmp = self._resolve_model(mlflow_run_id)
        try:
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)
            lots, readings = preprocessing.frames_from_batch_csv(df)
            predictions, _ = self._lot_predictions(model, lots, readings)
            self._record(started)
            logger.info("predict_batch done — %d lots, source=%s", len(predictions), model.source)
            return PredictBatchResponse(
                model_id=MODEL_ID,
                model_source=model.source,
                n_lots=len(predictions),
                predictions=predictions,
            )
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)

    # ── train (fine-tuning) ───────────────────────────────────────────────────

    def train(  # pylint: disable=too-many-locals
        self, *, data_path: str, mlflow_run_id: str
    ) -> TrainResponse:
        """Fine-tune a clone of the served GRU on a CSV in the AI team's sequential format.

        The retrained model lives only in its MLflow run (served later via
        predict(mlflow_run_id=...)); nothing is written to artifacts/ and the served model in
        memory is unchanged. If it cannot be uploaded, ModelPersistenceError (→ 502).
        """
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")
        with local_file_path(data_path) as local_path:
            raw_df = pd.read_csv(local_path)
        missing = [c for c in TRAIN_HARD_REQUIRED_COLUMNS if c not in raw_df.columns]
        if missing:
            raise ValueError(f"CSV falta columnas requeridas: {missing}")
        target_only = raw_df.get("target_only", pd.Series(False, index=raw_df.index))
        operational = ~target_only.fillna(False).astype(bool)
        if raw_df.loc[operational, "stage_progress"].isna().any():
            raise ValueError(
                "stage_progress es obligatorio en todas las filas operativas del CSV (KI-01)"
            )

        hyperparams = dict(self._model.config)
        tracker = BaseMLflowTracker(mlflow_run_id)
        tracker.log_params(
            {
                k: hyperparams[k]
                for k in (
                    "hidden_dim",
                    "num_layers",
                    "bidirectional",
                    "dropout",
                    "head_dropout",
                    "batch_size",
                    "learning_rate",
                    "weight_decay",
                    "epochs",
                    "patience",
                    "lr_patience",
                    "lr_decay_factor",
                    "gradient_clip",
                )
            }
            | {"optimizer": "AdamW", "loss": "SmoothL1Loss", "mode": "fine_tuning_from_gru_pso"}
        )

        def _on_epoch(epoch: int, train_loss: float, val_rmse: float) -> None:
            tracker.log_metrics({"train_loss": train_loss, "val_rmse": val_rmse}, step=epoch)

        data = training.prepare_training_data(raw_df, self._model)
        result = training.fine_tune(self._model, data, hyperparams, on_epoch=_on_epoch)
        val, test = result.val_report, result.test_report or {}

        metrics = {
            "val_rmse_future_free_sulfite_72h": val[TARGET_SO2]["rmse"],
            "val_mae_future_free_sulfite_72h": val[TARGET_SO2]["mae"],
            "val_rmse_underprotection_risk_72h": val[TARGET_RISK]["rmse"],
            "val_mae_underprotection_risk_72h": val[TARGET_RISK]["mae"],
            "val_overall_rmse": val["overall_rmse"],
            "val_overall_mae": val["overall_mae"],
        }
        test_metrics = {
            "test_rmse_future_free_sulfite_72h": test[TARGET_SO2]["rmse"] if test else None,
            "test_mae_future_free_sulfite_72h": test[TARGET_SO2]["mae"] if test else None,
            "test_rmse_underprotection_risk_72h": test[TARGET_RISK]["rmse"] if test else None,
            "test_mae_underprotection_risk_72h": test[TARGET_RISK]["mae"] if test else None,
        }
        mlflow_tmp = tempfile.mkdtemp(prefix="ml26_mlflow_")
        try:
            tracker.log_metrics(metrics | {k: v for k, v in test_metrics.items() if v is not None})
            model_loader.save_user_model(result.model, mlflow_tmp)
            tracker.upload_artifacts(mlflow_tmp, artifact_path=MLFLOW_ARTIFACT_PATH)
        except Exception as exc:
            logger.error("MLflow upload failed for ml26: %s", exc)
            raise ModelPersistenceError(
                f"El modelo reentrenado no se ha podido guardar en MLflow: {exc}"
            ) from exc
        finally:
            shutil.rmtree(mlflow_tmp, ignore_errors=True)

        logger.info(
            "ml26 train() done — epochs=%d best=%d val_overall_rmse=%.4f run=%s",
            result.epochs_run,
            result.best_epoch,
            val["overall_rmse"],
            mlflow_run_id,
        )
        return TrainResponse(
            detail=(
                "Fine-tuning completado a partir de gru_pso (sin nueva búsqueda PSO — ver "
                "manifest KI-04)."
            ),
            n_lots_train=result.n_lots["train"],
            n_lots_val=result.n_lots["val"],
            n_lots_test=result.n_lots["test"],
            n_windows_train=result.n_windows["train"],
            n_windows_val=result.n_windows["val"],
            epochs_run=result.epochs_run,
            best_epoch=result.best_epoch,
            **metrics,
            **test_metrics,
            mlflow_run_id=mlflow_run_id,
            upload_warning=None,
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the raw input contract and the reported (synthetic) test
        metrics.
        """
        lot_desc = {
            "lot_id": ("string", "Identificador del lote/depósito"),
            "wine_type": ("string", "red / rose / white"),
            "volume_l": ("float", "Volumen del depósito (L)"),
            "ambient_temp_c": ("float", "Temperatura ambiente de la bodega (°C)"),
        }
        inputs = [
            InputField(name=f"lot.{n}", type=t, description=d)
            for n, (t, d) in lot_desc.items()
            if n in LOT_COLUMNS
        ]
        inputs += [
            InputField(
                name=f"readings[].{n}",
                type="string" if n in ("lot_id", "stage") else "float",
                description="Obligatorio en cada lectura (paso de 2 h, mínimo 24 lecturas)",
            )
            for n in READING_REQUIRED_COLUMNS
            if n != "lot_id"
        ]
        inputs += [
            InputField(
                name=f"readings[].{n}",
                type="float" if n not in ("timestamp",) else "string",
                default=None,
                description="Opcional (analítica/dosis solo en las filas en que existan)",
            )
            for n in READING_OPTIONAL_COLUMNS
            if n != "actual_dose_mg_l"
        ]
        outputs = [
            OutputField(
                name=TARGET_SO2,
                type="float",
                description="SO2 libre esperado dentro de 72 h (mg/L)",
            ),
            OutputField(
                name=TARGET_RISK,
                type="float",
                description="Score de riesgo de infraprotección a 72 h [0, 1]",
            ),
            OutputField(
                name="risk_band",
                type="str",
                description="bajo / medio / alto (umbrales 0.33 / 0.66)",
            ),
        ]
        avg = self._total_latency_ms / self._predict_count if self._predict_count else None
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "GRU bidireccional multitarea (hiperparámetros optimizados con PSO) que estima, "
                "para cada depósito "
                "de vino, el SO2 libre y un score de riesgo de infraprotección a 72 h a partir "
                "de 48 h de lecturas "
                "de proceso, analítica de laboratorio y adiciones de SO2. Entrenado con datos "
                "sintéticos — "
                "validación con datos reales pendiente."
            ),
            task_type=TASK_TYPE,
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics=dict(REPORTED_METRICS)
            | {"model_config": dict(self._model.config) if self._model else None},
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=round(avg, 3) if avg is not None else None,
            ),
        )
        if mlflow_run_id:
            try:
                tracker = BaseMLflowTracker(mlflow_run_id)
                base.metrics["mlflow"] = {
                    "params": tracker.get_params(),
                    "metrics": tracker.get_metrics(),
                }
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Could not fetch MLflow stats for run_id=%s: %s", mlflow_run_id, exc)
        return base
