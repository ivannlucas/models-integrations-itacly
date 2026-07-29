"""Ml7CerealsGrainPestDetection — YOLOv8 pest detection in stored-cereal images.

Serves an externally-trained YOLO checkpoint, so train() raises 501.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import (
    InvalidImageError,
    ModelNotLoadedError,
    TrainingNotSupportedError,
)
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml7_cereals_grain_pest_detection.constants import (
    CONF_THRESHOLD,
    FRAMEWORK,
    IMAGE_EXTENSIONS,
    MODEL_ID,
    VERSION,
)
from app.plugins.ml7_cereals_grain_pest_detection.model_loader import load_yolo
from app.plugins.ml7_cereals_grain_pest_detection.postprocessing import (
    CLASS_NAMES,
    yolo_results_to_dict,
)
from app.plugins.ml7_cereals_grain_pest_detection.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml7_cereals_grain_pest_detection.preprocessing import (
    image_base64_to_numpy,
    image_path_to_numpy,
)

logger = logging.getLogger(__name__)

# ponytail: CPU only; the source forced cpu and YOLO infers fine on it. Add device
# selection if GPU inference is ever needed.
_DEVICE = "cpu"


def _to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    """ultralytics expects numpy frames in BGR; our loaders produce RGB."""
    return np.ascontiguousarray(img_rgb[:, :, ::-1])


class Ml7CerealsGrainPestDetectionPlugin(ModelPluginPort):
    """YOLOv8 detector for stored-grain pest species."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._model = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        """Load the YOLO checkpoint into memory."""
        self._model = load_yolo()
        logger.info("Ml7CerealsGrainPestDetectionPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        """Return True if the model is loaded."""
        return self._model is not None

    def _require_model(self):
        """Return the loaded model or raise ModelNotLoadedError."""
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")
        return self._model

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
        """Detect pests in a single image (path or base64)."""
        if mlflow_run_id:
            logger.warning("mlflow_run_id=%s provided but model '%s' does not support user training — using standard model",
                           mlflow_run_id, MODEL_ID)
        model = self._require_model()

        if features.get("image_path"):
            img_np = image_path_to_numpy(features["image_path"])
            features_used = ["image_path"]
        elif features.get("image_base64"):
            img_np = image_base64_to_numpy(features["image_base64"])
            features_used = ["image_base64"]
        else:
            raise InvalidImageError("features must contain 'image_path' or 'image_base64'")

        conf = threshold if threshold is not None else CONF_THRESHOLD
        results = model.predict(_to_bgr(img_np), verbose=False, conf=conf, device=_DEVICE)
        result = yolo_results_to_dict(results[0], img_np, conf_threshold=conf)
        self._record()

        return PredictInlineResponse(
            model_id=MODEL_ID,
            threshold=threshold,
            features_used=features_used,
            **result,
        )

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Detect pests in every image of a directory or ZIP."""
        if mlflow_run_id:
            logger.warning("mlflow_run_id=%s provided but model '%s' does not support user training — using standard model",
                           mlflow_run_id, MODEL_ID)
        model = self._require_model()
        tmp_dir: str | None = None

        with local_file_path(data_path) as local_data_path:
            data_p = Path(local_data_path)

            if data_p.suffix.lower() == ".zip":
                if not zipfile.is_zipfile(data_p):
                    raise ValueError(f"data_path is not a valid ZIP file: {data_path}")
                tmp_dir = tempfile.mkdtemp(prefix="grain_pest_batch_")
                with zipfile.ZipFile(data_p) as zf:
                    zf.extractall(tmp_dir)
                data_p = Path(tmp_dir)
            elif not data_p.is_dir():
                raise ValueError(f"data_path must be a directory or .zip file, got: {data_path}")

            try:
                image_files = sorted(
                    f for f in data_p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS
                )
                if not image_files:
                    raise ValueError(f"No images found in: {data_path}")

                predictions: list[dict] = []
                for img_file in image_files:
                    try:
                        img_np = image_path_to_numpy(str(img_file))
                        results = model.predict(
                            _to_bgr(img_np), verbose=False, conf=CONF_THRESHOLD, device=_DEVICE
                        )
                        result = yolo_results_to_dict(results[0], img_np)
                        predictions.append({"filename": img_file.name, **result})
                    except Exception as exc:
                        logger.warning("Error processing %s: %s", img_file.name, exc)
                        predictions.append({"filename": img_file.name, "error": str(exc)})

                self._record()
                return PredictBatchResponse(
                    model_id=MODEL_ID, predictions=predictions, output_path=None
                )
            finally:
                if tmp_dir is not None:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> PredictInlineResponse:
        """Training is not supported: the model uses externally trained artifacts (HTTP 501)."""
        _ = data_path, mlflow_run_id
        raise TrainingNotSupportedError(
            "Este modelo usa artefactos externos; el reentrenamiento no está disponible."
        )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        if mlflow_run_id:
            logger.warning("mlflow_run_id=%s provided but model '%s' does not support user training",
                           mlflow_run_id, MODEL_ID)
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description="Detección de insectos plaga en cereal almacenado mediante YOLOv8.",
            task_type="object_detection",
            framework=FRAMEWORK,
            inputs=[
                InputField(
                    name="image",
                    type="file",
                    format=["jpg", "jpeg", "png", "bmp", "tif"],
                    description="Imagen (image_path/base64 inline; dir o .zip en batch)",
                ),
            ],
            outputs=[
                OutputField(name="prediction", type="str",
                            description=f"Especie dominante: uno de {', '.join(CLASS_NAMES)}"),
                OutputField(
                    name="confidence", type="float",
                    description=(
                        "Confianza de la detección más alta entre las cajas de la especie "
                        "predicha (no el máximo/media/mínimo de todas las especies detectadas "
                        "en la imagen) [0, 1]"
                    ),
                ),
                OutputField(name="total_detections", type="int",
                            description="Nº de bounding boxes"),
                OutputField(name="species_counts", type="dict",
                            description="Nº de detecciones por especie"),
                OutputField(name="detections", type="list",
                            description="[{class, class_name, confidence, bbox}]"),
                OutputField(name="annotated_image", type="str",
                            description="Imagen con cajas, base64 JPEG"),
            ],
            metrics={},
            runtime_stats=RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=None),
        )
