"""Plugin for fungal leaf-disease detection on grapevine leaves using a custom CNN (LeafCNN).

Serves a single-task image classifier with five disease/health classes. The model
is trained externally, so ``train()`` is not implemented and ``/train`` returns 501.
"""
from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import (
    InvalidImageError,
    ModelNotLoadedError,
    TrainingNotSupportedError,
)
from app.plugins.ml2_fungal_cnn_disease_detection.constants import (
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    MODEL_ID,
)
from app.plugins.ml2_fungal_cnn_disease_detection.model_loader import load_model_bundle
from app.plugins.ml2_fungal_cnn_disease_detection.postprocessing import (
    build_batch_response,
    build_inline_response,
    compute_and_encode_cam,
)
from app.plugins.ml2_fungal_cnn_disease_detection.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml2_fungal_cnn_disease_detection.preprocessing import (
    image_base64_to_tensor,
    image_path_to_tensor_and_image,
)

logger = logging.getLogger(__name__)


class Ml2FungalCnnDiseaseDetectionPlugin(ModelPluginPort):
    """LeafCNN-based plugin classifying grapevine leaf images into fungal-disease classes."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._bundle: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        """Load the LeafCNN checkpoint into memory."""
        self._bundle = load_model_bundle()
        logger.info("Ml2FungalCnnDiseaseDetectionPlugin loaded: %s", self._bundle["model_id"])

    def is_loaded(self) -> bool:
        """Return True if the model bundle is loaded and ready for inference."""
        return self._bundle is not None

    def _require_bundle(self) -> dict:
        """Return the loaded bundle or raise :class:`ModelNotLoadedError`."""
        if self._bundle is None:
            raise ModelNotLoadedError("El modelo no está cargado.")
        return self._bundle

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Classify a single base64 image and return the typed inline response."""
        if mlflow_run_id:
            logger.warning("mlflow_run_id=%s provided but model '%s' does not support user training — using standard model",
                           mlflow_run_id, MODEL_ID)
        bundle = self._require_bundle()
        tensor = image_base64_to_tensor(
            features["image_base64"], image_size=bundle["image_size"]
        ).to(bundle["device"])

        bundle["model"].eval()
        with torch.no_grad():
            logits = bundle["model"](tensor)

        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info("predict_inline done — count=%d", self._predict_count)

        return PredictInlineResponse(**build_inline_response(
            logits,
            classes=bundle["classes"],
            model_id=bundle["model_id"],
        ))

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Classify every image inside a ZIP (local path or ``s3://`` URI)."""
        if mlflow_run_id:
            logger.warning("mlflow_run_id=%s provided but model '%s' does not support user training — using standard model",
                           mlflow_run_id, MODEL_ID)
        bundle = self._require_bundle()
        model = bundle["model"]
        device: torch.device = bundle["device"]
        image_size: int = bundle["image_size"]
        predictions: list[dict] = []

        tmp_zip: str | None = None
        local_data_path = data_path
        if data_path.startswith("s3://"):
            tmp_zip = self._download_zip_from_s3(data_path)
            local_data_path = tmp_zip

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(local_data_path, "r") as zf:
                    zf.extractall(tmp_dir)

                image_paths = sorted(
                    p for p in Path(tmp_dir).rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )

                model.eval()
                for image_path in image_paths:
                    try:
                        tensor, image = image_path_to_tensor_and_image(image_path, image_size=image_size)
                        tensor = tensor.to(device)
                        with torch.no_grad():
                            logits, feature_map = model(tensor, return_features=True)
                        result = build_inline_response(
                            logits,
                            classes=bundle["classes"],
                            model_id=bundle["model_id"],
                        )
                        result["filename"] = image_path.name
                        class_idx = bundle["classes"].index(result["prediction"])
                        result["heatmap_url"] = compute_and_encode_cam(
                            feature_map, model.classifier.weight, class_idx, image
                        )
                        predictions.append(result)
                    except InvalidImageError as exc:
                        predictions.append({"filename": image_path.name, "error": str(exc)})
                    except Exception as exc:
                        logger.warning("Unexpected error processing %s: %s", image_path.name, exc)
                        predictions.append({"filename": image_path.name, "error": str(exc)})
        finally:
            if tmp_zip and os.path.exists(tmp_zip):
                os.unlink(tmp_zip)

        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info(
            "predict_batch done — %d predictions count=%d", len(predictions), self._predict_count
        )

        return PredictBatchResponse(
            **build_batch_response(predictions, model_id=bundle["model_id"])
        )

    @staticmethod
    def _download_zip_from_s3(s3_uri: str) -> str:
        """Download a ZIP from an ``s3://bucket/key`` URI to a temp file and return its path."""
        import boto3
        from botocore.client import Config as BotoConfig

        without_prefix = s3_uri[5:]
        bucket, _, s3_key = without_prefix.partition("/")
        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ.get("CUSTOM_S3_ENDPOINT"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID_XAI"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY_XAI"),
            config=BotoConfig(signature_version="s3v4"),
            region_name=os.environ.get("CUSTOM_REGION", "us-east-1"),
        )
        fd, tmp_zip = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        logger.info("Downloading batch data from s3://%s/%s", bucket, s3_key)
        s3.download_file(bucket, s3_key, tmp_zip)
        return tmp_zip

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Training is not supported: this model uses externally trained artifacts (HTTP 501)."""
        _ = data_path, mlflow_run_id
        raise TrainingNotSupportedError(
            "Este modelo usa artefactos externos; el reentrenamiento no está disponible."
        )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        if mlflow_run_id:
            logger.warning("mlflow_run_id=%s provided but model '%s' does not support user training",
                           mlflow_run_id, MODEL_ID)
        model_id_v = self._bundle["model_id"] if self._bundle is not None else MODEL_ID

        return StatsResponse(
            model_name=model_id_v,
            version="1.0.0",
            description=(
                "Detección de enfermedades fúngicas en hojas de vid mediante una "
                "CNN personalizada (LeafCNN)."
            ),
            task_type="image-classification",
            framework="pytorch",
            inputs=[
                InputField(
                    name="image",
                    type="file",
                    format=["jpg", "jpeg", "png", "bmp"],
                    description="Imagen de hoja de vid (base64 para inline, ZIP para batch)",
                ),
            ],
            outputs=[
                OutputField(
                    name="prediction",
                    type="str",
                    description=f"Clase predicha (una de: {', '.join(CLASS_NAMES)})",
                ),
                OutputField(
                    name="confidence",
                    type="float",
                    description="Probabilidad softmax de la clase ganadora [0, 1]",
                ),
                OutputField(
                    name="probabilities",
                    type="dict",
                    description="Probabilidad softmax por clase",
                ),
                OutputField(
                    name="heatmap_url",
                    type="str",
                    description="Mapa de activación de clase (CAM) superpuesto en base64 JPEG data URI (solo en predict_batch)",
                ),
            ],
            metrics={},
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=None,
            ),
        )
