from __future__ import annotations

import logging
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import InvalidImageError, ModelNotLoadedError
from app.plugins.ml8_cereals_img_anomaly_detector.constants import IMAGE_EXTENSIONS, MODEL_ID
from app.plugins.ml8_cereals_img_anomaly_detector.model_loader import load_model_bundle
from app.plugins.ml8_cereals_img_anomaly_detector.postprocessing import build_batch_response, build_inline_response
from app.plugins.ml8_cereals_img_anomaly_detector.preprocessing import image_base64_to_tensor, image_path_to_tensor

logger = logging.getLogger(__name__)


class Ml8CerealsImgAnomalyDetectorPlugin(ModelPluginPort):

    def __init__(self) -> None:
        self._bundle: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        self._bundle = load_model_bundle()
        logger.info("Ml8CerealsImgAnomalyDetectorPlugin loaded: %s", self._bundle["model_id"])

    def is_loaded(self) -> bool:
        return self._bundle is not None

    def _require_bundle(self) -> dict:
        if self._bundle is None:
            raise ModelNotLoadedError("El modelo no está cargado.")
        return self._bundle

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
    ) -> dict:
        bundle = self._require_bundle()
        model = bundle["model"]
        device: torch.device = bundle["device"]
        image_size: int = bundle["image_size"]

        tensor = image_base64_to_tensor(features["image_base64"], image_size=image_size)
        tensor = tensor.to(device)

        model.eval()
        with torch.no_grad():
            logits_cat, logits_cer = model(tensor)

        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info("predict_inline done — count=%d", self._predict_count)

        return build_inline_response(
            logits_cat,
            logits_cer,
            idx_to_class=bundle["idx_to_class"],
            idx_to_cereal=bundle["idx_to_cereal"],
            model_id=bundle["model_id"],
        )

    def predict_batch(self, *, data_path: str) -> dict:
        bundle = self._require_bundle()
        model = bundle["model"]
        device: torch.device = bundle["device"]
        image_size: int = bundle["image_size"]

        predictions: list[dict] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(data_path, "r") as zf:
                zf.extractall(tmp_dir)

            image_paths = sorted(
                p for p in Path(tmp_dir).rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )

            model.eval()
            for image_path in image_paths:
                try:
                    tensor = image_path_to_tensor(image_path, image_size=image_size).to(device)
                    with torch.no_grad():
                        logits_cat, logits_cer = model(tensor)
                    result = build_inline_response(
                        logits_cat,
                        logits_cer,
                        idx_to_class=bundle["idx_to_class"],
                        idx_to_cereal=bundle["idx_to_cereal"],
                        model_id=bundle["model_id"],
                    )
                    result["filename"] = image_path.name
                    predictions.append(result)
                except InvalidImageError as exc:
                    predictions.append({"filename": image_path.name, "error": str(exc)})
                except Exception as exc:
                    logger.warning("Unexpected error processing %s: %s", image_path.name, exc)
                    predictions.append({"filename": image_path.name, "error": str(exc)})

        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info("predict_batch done — %d predictions count=%d", len(predictions), self._predict_count)

        return build_batch_response(predictions, model_id=bundle["model_id"])

    def stats(self) -> StatsResponse:
        arch = self._bundle["arch"] if self._bundle is not None else "unknown"
        model_id = self._bundle["model_id"] if self._bundle is not None else MODEL_ID
        idx_to_class = self._bundle["idx_to_class"] if self._bundle is not None else {}
        idx_to_cereal = self._bundle["idx_to_cereal"] if self._bundle is not None else {}
        category_names = list(idx_to_class.values())
        cereal_names = list(idx_to_cereal.values())

        return StatsResponse(
            model_name=model_id,
            model_type=f"MultiTask CNN ({arch})",
            framework="pytorch",
            artifact_path=f"artifacts/{MODEL_ID}",
            input_schema={
                "mode=inline": {"image_base64": "str — Base64-encoded image (JPEG/PNG/BMP)"},
                "mode=batch": {"data_path": "str — path to ZIP file containing images"},
            },
            output_schema={
                "inline": {
                    "categoria": f"str — one of {category_names}",
                    "cereal": f"str — one of {cereal_names}",
                    "confianza_categoria": "float [0, 1]",
                    "confianza_cereal": "float [0, 1]",
                    "probabilidades_categoria": "dict[str, float]",
                    "probabilidades_cereal": "dict[str, float]",
                },
                "batch": {"predictions": "list[dict] — per-image predictions"},
            },
            predict_count=self._predict_count,
            last_predict_at=self._last_predict_at,
        )
