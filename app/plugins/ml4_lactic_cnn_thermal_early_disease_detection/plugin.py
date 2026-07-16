"""Ml4LacticCnnThermal — thermal-udder subclinical-mastitis (SCM) image classifier.

Serves an externally-trained EfficientNet checkpoint, so train() raises 501.
"""
from __future__ import annotations

import base64
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone

import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import (
    InvalidImageError,
    ModelNotLoadedError,
    TrainingNotSupportedError,
)
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.constants import (
    BACKBONE,
    DROPOUT,
    FRAMEWORK,
    IMAGE_EXTENSIONS,
    MODEL_ID,
    VERSION,
)
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.model_loader import load_model
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.postprocessing import decode_logits
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.preprocessing import (
    preprocess_image,
)

logger = logging.getLogger(__name__)


class Ml4LacticCnnThermalEarlyDiseaseDetectionPlugin(ModelPluginPort):
    """EfficientNet-B0 binary classifier (Healthy / SCM) on thermal udder images."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._model = None
        self._device = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        """Load the EfficientNet checkpoint."""
        self._model, self._device = load_model()
        logger.info("Ml4LacticCnnThermalEarlyDiseaseDetectionPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        """Return True if the model is loaded."""
        return self._model is not None

    def _require_loaded(self) -> None:
        """Raise ModelNotLoadedError if the model is not loaded."""
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _infer_image(self, image_bytes: bytes) -> dict:
        """Preprocess image bytes, run the model and decode the prediction dict."""
        tensor = preprocess_image(image_bytes).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
        return decode_logits(logits)

    def _record(self) -> None:
        """Update runtime counters after a prediction."""
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Classify a single thermal image (base64 or path)."""
        _ = mlflow_run_id
        self._require_loaded()
        if features.get("image_path"):
            path = features["image_path"]
            if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS:
                raise InvalidImageError(f"Extensión no soportada. Usa {sorted(IMAGE_EXTENSIONS)}")
            with open(path, "rb") as fh:
                image_bytes = fh.read()
            used = ["image_path"]
        elif features.get("image_base64"):
            image_bytes = base64.b64decode(features["image_base64"])
            used = ["image_base64"]
        else:
            raise InvalidImageError("features debe contener 'image_path' o 'image_base64'")

        result = self._infer_image(image_bytes)
        self._record()
        return PredictInlineResponse(
            model_id=MODEL_ID, threshold=threshold, features_used=used, **result
        )

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Classify every thermal image in a directory or ZIP (local path or ``s3://`` URI)."""
        _ = mlflow_run_id
        self._require_loaded()
        temp_dir: str | None = None
        predictions: list[dict] = []
        with local_file_path(data_path) as local_data_path:
            image_dir = local_data_path
            if local_data_path.lower().endswith(".zip"):
                temp_dir = tempfile.mkdtemp(prefix="ml4_thermal_batch_")
                with zipfile.ZipFile(local_data_path, "r") as zf:
                    zf.extractall(temp_dir)
                entries = os.listdir(temp_dir)
                if len(entries) == 1 and os.path.isdir(os.path.join(temp_dir, entries[0])):
                    image_dir = os.path.join(temp_dir, entries[0])
                else:
                    image_dir = temp_dir

            try:
                image_files = sorted(
                    (root, fname)
                    for root, _, files in os.walk(image_dir)
                    for fname in files
                    if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
                )
                for idx, (root, fname) in enumerate(image_files):
                    try:
                        with open(os.path.join(root, fname), "rb") as fh:
                            image_bytes = fh.read()
                        result = self._infer_image(image_bytes)
                        row = {"filename": fname, **result}
                        if idx == 0:
                            # Echo the first image back so the platform can request a real
                            # GradCAM explanation for the batch (mirrors the inline flow,
                            # which has a single image_path to read from — batch has none
                            # once this temp dir is removed below).
                            row["image_base64"] = base64.b64encode(image_bytes).decode("ascii")
                        predictions.append(row)
                    except Exception as exc:
                        predictions.append({"filename": fname, "error": str(exc)})
            finally:
                if temp_dir:
                    shutil.rmtree(temp_dir, ignore_errors=True)

        self._record()
        return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> PredictInlineResponse:
        """Training is not supported: the model uses externally trained artifacts (HTTP 501)."""
        _ = data_path, mlflow_run_id
        raise TrainingNotSupportedError(
            "Este modelo usa artefactos externos; el reentrenamiento no está disponible."
        )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        _ = mlflow_run_id
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                f"Clasificación binaria de imágenes térmicas de ubre de oveja para detección de "
                f"mamitis subclínica (backbone={BACKBONE}, dropout={DROPOUT})."
            ),
            task_type="binary_classification",
            framework=FRAMEWORK,
            inputs=[
                InputField(
                    name="image",
                    type="file",
                    format=["jpg", "jpeg", "png", "bmp"],
                    description="Imagen térmica de ubre (base64/path inline; .zip o dir en batch)",
                ),
            ],
            outputs=[
                OutputField(name="prediction", type="str",
                            description="Clase predicha: 'Healthy' o 'SCM'"),
                OutputField(name="confidence", type="float",
                            description="Probabilidad softmax de la clase ganadora [0, 1]"),
                OutputField(name="probability_healthy", type="float",
                            description="P(clase=Healthy)"),
                OutputField(name="probability_scm", type="float", description="P(clase=SCM)"),
            ],
            metrics={},
            runtime_stats=RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=None),
        )
