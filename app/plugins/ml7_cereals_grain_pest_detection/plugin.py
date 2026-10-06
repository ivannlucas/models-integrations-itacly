"""Ml7CerealsGrainPestDetection — YOLO pest detection in stored-cereal images.

Serves a fixed, AI-team-trained YOLO checkpoint by default. Real fine-tuning is
supported (train()) — it faithfully ports src/training/train.py::train_model() from
the delivered code (inbox/a07/codigo/), persisting the retrained checkpoint to MLflow
only, never overwriting the served artifact.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import InvalidImageError, ModelNotLoadedError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml7_cereals_grain_pest_detection.constants import (
    CONF_THRESHOLD,
    FRAMEWORK,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    MODEL_FILENAME,
    MODEL_ID,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_EVAL_CONF,
    TRAIN_HYPERPARAMS,
    TRAIN_PATIENCE,
    TRAIN_SEED,
    TRAIN_WORKERS,
    VERSION,
)
from app.plugins.ml7_cereals_grain_pest_detection.mlflow_utils import (
    download_user_model_from_mlflow,
    upload_artifacts_to_mlflow,
)
from app.plugins.ml7_cereals_grain_pest_detection.model_loader import load_yolo, safe_device
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
from app.plugins.ml7_cereals_grain_pest_detection.train_dto import TrainResponse

logger = logging.getLogger(__name__)

# Base/fixed-model reference metrics — memoria sección 7.2, Tabla 7 (detección, "all").
# Overlaid by stats(mlflow_run_id=...) with the real per-run metrics logged by train().
_BASE_METRICS = {"map50": 0.87, "map50_95": None, "precision": 0.84, "recall": 0.81}


def _to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    """ultralytics expects numpy frames in BGR; our loaders produce RGB."""
    return np.ascontiguousarray(img_rgb[:, :, ::-1])


def _resolve_dataset_yaml(data_path: str) -> Path:
    """Resolve a YOLO dataset.yaml from a data_path that may be the yaml itself or a
    dataset root directory containing one."""
    p = Path(data_path)
    if p.is_file():
        return p
    if p.is_dir():
        candidate = p / "dataset.yaml"
        if candidate.exists():
            return candidate
        candidate = p / "splits" / "dataset.yaml"
        if candidate.exists():
            return candidate
    raise ValueError(
        f"data_path debe ser un dataset.yaml de YOLO o un directorio que lo contenga: {data_path}"
    )


def _count_images(dataset_yaml: Path, split: str) -> int:
    """Count images for a given split declared in dataset.yaml (best-effort, for metrics)."""
    try:
        with open(dataset_yaml, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        split_dir = Path(cfg[split])
        if not split_dir.is_absolute():
            split_dir = dataset_yaml.parent / split_dir
        return sum(1 for f in split_dir.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("Could not count images for split=%s: %s", split, exc)
        return 0


class Ml7CerealsGrainPestDetectionPlugin(ModelPluginPort):
    """YOLO detector for stored-grain pest species."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._model = None
        self._device = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        """Load the YOLO checkpoint into memory and resolve the inference device."""
        self._model = load_yolo()
        self._device = safe_device()
        logger.info(
            "Ml7CerealsGrainPestDetectionPlugin loaded: %s (device=%s)", MODEL_ID, self._device
        )

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

    def _resolve_model_for_predict(self, mlflow_run_id: str):
        """Return (model, temp_dir_to_cleanup) — the user's MLflow checkpoint if
        mlflow_run_id is given and downloadable, otherwise the served base model.
        Never mutates self._model, so concurrent requests without mlflow_run_id are
        unaffected by one that uses a retrained checkpoint."""
        if not mlflow_run_id:
            return self._require_model(), None
        logger.info("Using user-trained model from MLflow run_id=%s", mlflow_run_id)
        loaded = download_user_model_from_mlflow(mlflow_run_id)
        if loaded is None:
            logger.warning(
                "MLflow download failed for run_id=%s, falling back to standard model",
                mlflow_run_id,
            )
            return self._require_model(), None
        model, temp_dir = loaded
        return model, temp_dir

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Detect pests in a single image (path or base64)."""
        _ = model_key
        model, temp_dir = self._resolve_model_for_predict(mlflow_run_id)
        try:
            if features.get("image_path"):
                img_np = image_path_to_numpy(features["image_path"])
                features_used = ["image_path"]
            elif features.get("image_base64"):
                img_np = image_base64_to_numpy(features["image_base64"])
                features_used = ["image_base64"]
            else:
                raise InvalidImageError("features must contain 'image_path' or 'image_base64'")

            conf = threshold if threshold is not None else CONF_THRESHOLD
            results = model.predict(
                _to_bgr(img_np), verbose=False, conf=conf, imgsz=IMG_SIZE, device=self._device,
            )
            result = yolo_results_to_dict(results[0], img_np, conf_threshold=conf)
            self._record()

            return PredictInlineResponse(
                model_id=MODEL_ID,
                threshold=threshold,
                features_used=features_used,
                **result,
            )
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Detect pests in every image of a directory or ZIP."""
        model, mlflow_temp_dir = self._resolve_model_for_predict(mlflow_run_id)
        tmp_dir: str | None = None

        try:
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
                    raise ValueError(
                        f"data_path must be a directory or .zip file, got: {data_path}"
                    )

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
                                _to_bgr(img_np), verbose=False, conf=CONF_THRESHOLD,
                                imgsz=IMG_SIZE, device=self._device,
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
        finally:
            if mlflow_temp_dir:
                shutil.rmtree(mlflow_temp_dir, ignore_errors=True)

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Fine-tune the YOLO detector, faithfully porting
        src/training/train.py::train_model() from inbox/a07/codigo/.

        Loads a FRESH YOLO instance from the served base weights (never mutates
        self._model in place, so concurrent predict() calls without mlflow_run_id keep
        using the untouched served model). The fine-tuned checkpoint is written only to
        a temp dir and uploaded to MLflow — the served artifacts/.../best.pt is never
        overwritten (see inbox/a07/manifest.yaml::training).
        """
        dataset_yaml = _resolve_dataset_yaml(data_path)
        device = safe_device()

        train_project_dir = Path(tempfile.mkdtemp(prefix="ml7_train_"))
        upload_dir = Path(tempfile.mkdtemp(prefix="ml7_train_upload_"))
        upload_warning = None
        try:
            model = load_yolo()  # fresh instance from the served base checkpoint
            model.train(
                data=str(dataset_yaml),
                device=device,
                workers=TRAIN_WORKERS,
                epochs=TRAIN_EPOCHS,
                imgsz=IMG_SIZE,
                batch=TRAIN_BATCH_SIZE,
                seed=TRAIN_SEED,
                patience=TRAIN_PATIENCE,
                pretrained=True,
                plots=False,
                project=str(train_project_dir),
                name="train",
                exist_ok=True,
                **TRAIN_HYPERPARAMS,
            )

            best_weights = train_project_dir / "train" / "weights" / "best.pt"
            if not best_weights.exists():
                raise RuntimeError(f"Entrenamiento finalizado pero no se generó {best_weights}")

            # Evaluate on the val split, same as src/training/validation.py::validate_model()
            val_model = _load_yolo_from_path(best_weights)
            val_metrics = val_model.val(
                data=str(dataset_yaml),
                split="val",
                imgsz=IMG_SIZE,
                batch=TRAIN_BATCH_SIZE,
                device=device,
                plots=False,
                conf=TRAIN_EVAL_CONF,
            )
            map50 = float(val_metrics.box.map50)
            map50_95 = float(val_metrics.box.map)
            precision = float(val_metrics.box.mp)
            recall = float(val_metrics.box.mr)
            n_images = _count_images(dataset_yaml, "train")

            shutil.copy2(best_weights, upload_dir / MODEL_FILENAME)

            if not mlflow_run_id:
                upload_warning = "Sin run de MLflow: el modelo reentrenado no se ha guardado."
            else:
                try:
                    upload_artifacts_to_mlflow(
                        str(upload_dir),
                        mlflow_run_id,
                        metrics={
                            "map50": map50, "map50_95": map50_95,
                            "precision": precision, "recall": recall, "n_images": n_images,
                        },
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.error("MLflow artifact upload failed: %s", exc)
                    upload_warning = (
                        f"El modelo reentrenado no se ha podido guardar en MLflow: {exc}"
                    )

            logger.info(
                "train() done — map50=%.4f map50_95=%.4f precision=%.4f recall=%.4f "
                "n_images=%d mlflow=%s",
                map50, map50_95, precision, recall, n_images, bool(mlflow_run_id),
            )
            return TrainResponse(
                detail="Fine-tuning completado",
                map50=round(map50, 4),
                map50_95=round(map50_95, 4),
                precision=round(precision, 4),
                recall=round(recall, 4),
                n_images=n_images,
                upload_warning=upload_warning,
            )
        finally:
            shutil.rmtree(train_project_dir, ignore_errors=True)
            shutil.rmtree(upload_dir, ignore_errors=True)

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        metrics = dict(_BASE_METRICS)
        if mlflow_run_id:
            try:
                mlflow_metrics = BaseMLflowTracker(mlflow_run_id).get_metrics()
                for key in ("map50", "map50_95", "precision", "recall", "n_images"):
                    if key in mlflow_metrics:
                        metrics[key] = mlflow_metrics[key]
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Could not fetch MLflow metrics for run_id=%s: %s", mlflow_run_id, exc
                )

        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description="Detección de insectos plaga en cereal almacenado mediante YOLO.",
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
                    name="confidence", type="List",
                    description=(
                        "Confianza de cada detección individual (caja delimitadora) en la "
                        "imagen, de todas las especies detectadas — no solo de la ganadora."
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
            metrics=metrics,
            runtime_stats=RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=None),
        )


def _load_yolo_from_path(path: Path):
    """Load a YOLO model directly from a checkpoint path (used for post-train val())."""
    from ultralytics import YOLO  # noqa: PLC0415 — heavy import kept lazy

    return YOLO(str(path))
