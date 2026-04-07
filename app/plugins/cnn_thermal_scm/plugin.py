import base64
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Any

from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import InvalidImageError, ModelNotLoadedError

logger = logging.getLogger(__name__)

MODEL_NAME = "cnn-thermal-scm"
MODEL_VERSION = "1.0.0"

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class CnnThermalScmPlugin(ModelPluginPort):
    def __init__(self) -> None:
        self._model: Any = None
        self._device: Any = None
        self._loaded: bool = False
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        from app.plugins.cnn_thermal_scm.model_loader import load_model
        self._model, self._device = load_model()
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def _infer_image(self, image_bytes: bytes) -> dict:
        import torch
        from app.plugins.cnn_thermal_scm.postprocessing import decode_logits
        from app.plugins.cnn_thermal_scm.preprocessing import preprocess_image

        image_tensor = preprocess_image(image_bytes).to(self._device)
        with torch.no_grad():
            logits = self._model(image_tensor)
        response = decode_logits(logits)
        return {
            "prediction": response.prediction,
            "confidence": response.confidence,
            "predicted_class_index": response.predicted_class_index,
            "probability_healthy": response.probability_healthy,
            "probability_scm": response.probability_scm,
        }

    def predict_batch(self, *, data_path: str) -> dict:
        temp_dir: str | None = None
        image_dir = data_path

        if data_path.lower().endswith(".zip"):
            temp_dir = tempfile.mkdtemp(prefix="cnn_thermal_batch_")
            with zipfile.ZipFile(data_path, "r") as zf:
                zf.extractall(temp_dir)
            entries = os.listdir(temp_dir)
            if len(entries) == 1 and os.path.isdir(os.path.join(temp_dir, entries[0])):
                image_dir = os.path.join(temp_dir, entries[0])
            else:
                image_dir = temp_dir

        predictions = []
        try:
            image_files = sorted(
                (root, fname)
                for root, _, files in os.walk(image_dir)
                for fname in files
                if os.path.splitext(fname)[1].lower() in _SUPPORTED_EXTENSIONS
            )
            for root, fname in image_files:
                try:
                    with open(os.path.join(root, fname), "rb") as f:
                        image_bytes = f.read()
                    result = self._infer_image(image_bytes)
                    predictions.append({"filename": fname, **result})
                except Exception as exc:
                    predictions.append({"filename": fname, "error": str(exc)})
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info("predict_batch done — %d images count=%d", len(predictions), self._predict_count)

        return {"model_id": MODEL_NAME, "predictions": predictions, "output_path": None}

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
    ) -> dict:
        if "image_path" in features:
            image_path = features["image_path"]
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in _SUPPORTED_EXTENSIONS:
                raise InvalidImageError(f"Extensión no soportada: {ext}. Usa {_SUPPORTED_EXTENSIONS}")
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        elif "image_base64" in features:
            image_bytes = base64.b64decode(features["image_base64"])
        else:
            raise ValueError("features debe contener 'image_path' o 'image_base64'")

        result = self._infer_image(image_bytes)
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info(
            "predict_inline done — result=%s confidence=%.4f count=%d",
            result["prediction"], result["confidence"], self._predict_count,
        )

        return {
            "model_id": MODEL_NAME,
            "threshold": threshold,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "features_used": ["image_base64"],
            "predicted_class_index": result["predicted_class_index"],
            "probability_healthy": result["probability_healthy"],
            "probability_scm": result["probability_scm"],
        }

    def stats(self) -> StatsResponse:
        from app.plugins.cnn_thermal_scm.model_loader import ARTIFACT_FILENAME, BACKBONE, DROPOUT
        return StatsResponse(
            model_name=MODEL_NAME,
            model_type=f"EfficientNet-B0 binary classifier (backbone={BACKBONE}, dropout={DROPOUT})",
            framework="torch + timm",
            artifact_path=f"model-runtime-cnn_thermal_scm/artifacts/{ARTIFACT_FILENAME}",
            input_schema={
                "mode=inline": {"image_path": "str — thermal image file (JPEG, PNG, or BMP)"},
                "mode=batch": {"data_path": "str — directory with thermal images"},
            },
            output_schema={
                "batch": {"predictions": "list[dict] — per-image classification results"},
                "inline": {
                    "prediction": "str — 'Healthy' or 'SCM'",
                    "confidence": "float — softmax probability of winning class",
                    "probability_healthy": "float", "probability_scm": "float",
                },
            },
            predict_count=self._predict_count,
            last_predict_at=self._last_predict_at,
        )
