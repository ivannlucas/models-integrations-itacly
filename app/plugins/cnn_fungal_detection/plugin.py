from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import InvalidImageError, ModelNotLoadedError
from app.plugins.cnn_fungal_detection.model_loader import ARTIFACT_PATH, CLASSES, LeafCNN, load_leafcnn
from app.plugins.cnn_fungal_detection.postprocessing import logits_to_prediction
from app.plugins.cnn_fungal_detection.preprocessing import image_base64_to_tensor, image_path_to_tensor

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class CnnFungalDetectionPlugin(ModelPluginPort):
    MODEL_ID = "cnn-fungal-detection"
    MODEL_TYPE = "LeafCNN"
    FRAMEWORK = "pytorch"
    VERSION = "1.0.0"

    def __init__(self) -> None:
        self._model: LeafCNN | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        self._model = load_leafcnn(self._device)
        logger.info("CnnFungalDetectionPlugin ready on device=%s", self._device)

    def is_loaded(self) -> bool:
        return self._model is not None

    def _assert_loaded(self) -> None:
        if not self.is_loaded():
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _update_stats(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(timezone.utc).isoformat()

    def predict_batch(self, *, data_path: str) -> dict:
        self._assert_loaded()
        temp_dir: str | None = None
        image_dir = Path(data_path)

        if data_path.lower().endswith(".zip"):
            temp_dir = tempfile.mkdtemp(prefix="cnn_fungal_batch_")
            with zipfile.ZipFile(data_path, "r") as zf:
                zf.extractall(temp_dir)
            entries = list(Path(temp_dir).iterdir())
            if len(entries) == 1 and entries[0].is_dir():
                image_dir = entries[0]
            else:
                image_dir = Path(temp_dir)

        if not image_dir.is_dir():
            raise ValueError(f"data_path debe ser un directorio o fichero .zip, recibido: {data_path}")

        image_files = sorted(
            f for f in image_dir.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not image_files:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError(f"No se encontraron imágenes en: {data_path}")

        predictions = []
        try:
            for img_file in image_files:
                tensor = image_path_to_tensor(str(img_file)).to(self._device)
                with torch.no_grad():
                    logits = self._model(tensor)
                result = logits_to_prediction(logits)
                predictions.append({"filename": img_file.name, **result})
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

        self._update_stats()
        return {"model_id": self.MODEL_ID, "predictions": predictions, "output_path": None}

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
    ) -> dict:
        self._assert_loaded()

        if "image_path" in features:
            image_path = features["image_path"]
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise InvalidImageError(f"Extensión no soportada: {ext}. Usa {SUPPORTED_EXTENSIONS}")
            tensor = image_path_to_tensor(image_path).to(self._device)
        elif "image_base64" in features:
            tensor = image_base64_to_tensor(features["image_base64"]).to(self._device)
        else:
            raise ValueError("features debe contener 'image_path' o 'image_base64'")

        with torch.no_grad():
            logits = self._model(tensor)
        result = logits_to_prediction(logits)
        self._update_stats()

        return {
            "model_id": self.MODEL_ID,
            "threshold": threshold,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "features_used": ["image_path" if "image_path" in features else "image_base64"],
            "probabilities": result["probabilities"],
        }

    def stats(self) -> StatsResponse:
        return StatsResponse(
            model_name=self.MODEL_ID,
            model_type=self.MODEL_TYPE,
            framework=self.FRAMEWORK,
            artifact_path=str(ARTIFACT_PATH),
            input_schema={
                "inline": {"image_path": "str — absolute path to JPEG/PNG/BMP image"},
                "batch": {"data_path": "str — directory or .zip with images"},
            },
            output_schema={
                "prediction": f"str (one of: {', '.join(CLASSES)})",
                "confidence": "float [0, 1]",
                "probabilities": "dict[str, float] — one entry per class",
            },
            predict_count=self._predict_count,
            last_predict_at=self._last_predict_at,
        )
