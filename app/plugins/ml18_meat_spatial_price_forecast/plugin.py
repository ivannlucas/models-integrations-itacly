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
from datetime import datetime, timezone

import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError, TrainingNotSupportedError
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml18_meat_spatial_price_forecast import inference, model_loader
from app.plugins.ml18_meat_spatial_price_forecast.constants import (
    FRAMEWORK,
    LOOKBACK,
    METRICS_REPORTED,
    MODEL_ID,
    RAW_REQUIRED_COLS,
    VERSION,
)
from app.plugins.ml18_meat_spatial_price_forecast.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)

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
        _ = model_key, threshold, mlflow_run_id
        self._require_loaded()
        rows: list[dict] = features["rows"]
        predictions = inference.run_inference(self._bundle, rows)
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
        _ = mlflow_run_id
        self._require_loaded()
        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path, sep=";")
        rows: list[dict] = df.to_dict(orient="records")

        predictions = inference.run_inference(self._bundle, rows)
        self._record_prediction()
        logger.info(
            "predict_batch done — %d predicciones, count=%d", len(predictions), self._predict_count,
        )
        return PredictBatchResponse(
            model_id=MODEL_ID, predictions=predictions, n_predictions=len(predictions), output_path=None,
        )

    # ── train (no soportado — ver inbox/a18/manifest.yaml::training) ──────────

    def train(self, *, data_path: str = "", mlflow_run_id: str = "") -> None:
        """Raise TrainingNotSupportedError — the delivered training procedure has no
        client-data path (src.main::train() always reads the fixed dataset_path from
        config.yaml, never a --data/--input CSV — see manifest.training)."""
        _ = data_path, mlflow_run_id
        raise TrainingNotSupportedError(
            "ml18 no soporta reentrenamiento por usuario: el procedimiento de entrenamiento "
            "entregado (src.main::train()) siempre reentrena sobre el mismo dataset fijo "
            "bundled (config.yaml::data.dataset_path) -- no acepta ningún CSV de datos de "
            "cliente, solo hiperparámetros. Reentrenar requiere el repo original "
            "a18-rnn-carnico-espacial-prediccion-modas-gustos-areas y volver a subir el "
            "artefacto a S3 bajo artifacts/fixed/ml18_meat_spatial_price_forecast/."
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and the real hold-out metrics."""
        _ = mlflow_run_id
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
            metrics=dict(METRICS_REPORTED),
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=None,
            ),
        )
