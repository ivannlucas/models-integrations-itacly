"""Ml18MeatSpatialPriceForecastPlugin — GRU meat spatial price forecast, t+1 month.

Predicts PRECIO MEDIO KG (meat sector, Spain) one month ahead per (CCAA, Producto)
combination, using spatial lag features from neighboring CCAA (PyTorch-free: Keras/TensorFlow
GRU). See inbox/a18/manifest.yaml for the full input/output contract, the golden-dataset
verification and known issues.

Accepted for production: GRU test MAPE 9.93% vs. the 20% KPI threshold defined in the memoria
(Ficha de Valoración del Estado Técnico, Opción A, 21/09/2026) — see manifest model_status.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError, ModelPersistenceError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml18_meat_spatial_price_forecast import inference, model_loader, training
from app.plugins.ml18_meat_spatial_price_forecast.constants import (
    FRAMEWORK,
    LOOKBACK,
    METRICS_REPORTED,
    MODEL_ID,
    RAW_REQUIRED_COLS,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_RATIO,
    TRAIN_SEED,
    VAL_RATIO,
    VERSION,
)
from app.plugins.ml18_meat_spatial_price_forecast.mlflow_utils import (
    download_user_model_from_mlflow,
    upload_artifacts_to_mlflow,
)
from app.plugins.ml18_meat_spatial_price_forecast.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml18_meat_spatial_price_forecast.train_dto import TrainResponse

logger = logging.getLogger(__name__)


class Ml18MeatSpatialPriceForecastPlugin(ModelPluginPort):
    """GRU plugin for the spatial meat price forecast (t+1 month)."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._bundle: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the fixed GRU artifact (model + frozen scalers) via ArtifactStore."""
        self._bundle = model_loader.load_artifact_bundle()
        logger.info("ml18 plugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        """Return True once the artifact bundle is loaded."""
        return self._bundle is not None

    def _require_loaded(self) -> None:
        if not self.is_loaded():
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _resolve_bundle(self, mlflow_run_id: str) -> tuple[dict, str | None]:
        """Return (bundle, temp_dir) as locals — never stored on self (shared across requests).

        A run without a loadable model raises UserModelUnavailableError (→ 422): never fall
        back silently to the base model. The caller must rmtree temp_dir in a finally block.
        """
        if mlflow_run_id:
            return download_user_model_from_mlflow(mlflow_run_id)
        self._require_loaded()
        return self._bundle, None

    def _record_prediction(self) -> None:
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
    ) -> PredictInlineResponse:
        """Predict next-month PRECIO MEDIO KG for every (CCAA, Producto) group in the panel."""
        _ = model_key, threshold
        bundle, user_tmp = self._resolve_bundle(mlflow_run_id)
        try:
            predictions = inference.run_inference(bundle, features["rows"])
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)
        self._record_prediction()
        logger.info(
            "predict_inline done — %d predicciones, count=%d", len(predictions), self._predict_count,
        )
        return PredictInlineResponse(
            model_id=MODEL_ID, predictions=predictions, n_predictions=len(predictions),
        )

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Predict next-month PRECIO MEDIO KG for every (CCAA, Producto) group in the CSV panel."""
        bundle, user_tmp = self._resolve_bundle(mlflow_run_id)
        try:
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path, sep=";")
            predictions = inference.run_inference(bundle, df.to_dict(orient="records"))
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)
        self._record_prediction()
        logger.info(
            "predict_batch done — %d predicciones, count=%d", len(predictions), self._predict_count,
        )
        return PredictBatchResponse(
            model_id=MODEL_ID, predictions=predictions, n_predictions=len(predictions), output_path=None,
        )

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Retrain the GRU from scratch with the AI team's procedure (see training.py).

        The data comes from data_path (the original reads a fixed config path; the procedure
        is the same). A new model is trained: the served one is never touched. The result
        lives only in its MLflow run; if it cannot be uploaded, ModelPersistenceError (→ 502).
        """
        with local_file_path(data_path) as local_path:
            history: list[tuple[int, dict]] = []
            result = training.train_gru(
                local_path, on_epoch=lambda epoch, logs: history.append((epoch, logs)),
            )

        artifact_tmp = tempfile.mkdtemp(prefix="ml18_train_")
        try:
            training.save_training_artifacts(Path(artifact_tmp), result)
            upload_artifacts_to_mlflow(
                artifact_tmp, mlflow_run_id, metrics=result.metrics,
                params={"model": "GRU", "lookback": LOOKBACK, "epochs_max": TRAIN_EPOCHS,
                        "batch_size": TRAIN_BATCH_SIZE, "train_ratio": TRAIN_RATIO,
                        "val_ratio": VAL_RATIO, "seed": TRAIN_SEED},
                history=history,
            )
        except Exception as exc:
            logger.error("ml18: MLflow upload failed: %s", exc)
            raise ModelPersistenceError(
                f"El modelo reentrenado no se ha podido guardar en MLflow: {exc}"
            ) from exc
        finally:
            shutil.rmtree(artifact_tmp, ignore_errors=True)

        return TrainResponse(
            detail="GRU reentrenada desde cero con el procedimiento del equipo de IA",
            **result.metrics,
            mlflow_run_id=mlflow_run_id,
            upload_warning=None,
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _metrics_with_run(mlflow_run_id: str) -> dict:
        metrics = dict(METRICS_REPORTED)
        if mlflow_run_id:
            tracker = BaseMLflowTracker(mlflow_run_id)
            metrics["mlflow"] = {"params": tracker.get_params(), "metrics": tracker.get_metrics()}
        return metrics

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and the real hold-out metrics.

        With mlflow_run_id, the params and metrics of that retraining run are added under
        metrics["mlflow"] (read-only; the base metrics stay as reported).
        """
        inputs = [
            InputField(
                name="rows", type="list[dict]",
                description=(
                    f"Panel mensual, mínimo {LOOKBACK} filas consecutivas por combinación "
                    f"CCAA-Producto. Cada fila: {', '.join(RAW_REQUIRED_COLS)} + 'PRECIO MEDIO "
                    "KG' (opcional en la última fila conocida)."
                ),
            ),
        ]
        outputs = [
            OutputField(
                name="predictions", type="list[dict]",
                description="Una entrada {CCAA, Producto, Fecha, predicted_price} por combinación con histórico suficiente.",
            ),
        ]
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Predicción mensual del precio medio por kg de productos cárnicos (España) a "
                "horizonte de 1 mes, por combinación CCAA-Producto, mediante GRU (Keras/"
                "TensorFlow) con lags espaciales de CCAA vecinas. Ver inbox/a18/manifest.yaml "
                "para el contrato completo y las métricas de test frente al baseline."
            ),
            task_type="regression_timeseries_spatial",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics=self._metrics_with_run(mlflow_run_id),
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=None,
            ),
        )
