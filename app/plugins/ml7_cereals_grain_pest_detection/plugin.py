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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

from app.application.dto.stats_dto import InputField, OutputField, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.batch_rows import row_error_capture
from app.domain.services.exceptions import (
    InvalidImageError,
    ModelNotLoadedError,
    ModelPersistenceError,
)
from app.domain.services.mlflow_artifacts import removing_dir, require_train_tracker
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.infrastructure.archive import safe_extract_zip
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
    TRAINING_CLASS_ID_TO_CODE,
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
        f"data_path debe ser un ZIP, un dataset.yaml de YOLO o un directorio que lo contenga: "
        f"{data_path}"
    )


def _is_zip(path: Path) -> bool:
    """True for a ZIP file, by extension or by content (s3:// downloads keep the key's suffix)."""
    return path.is_file() and (path.suffix.lower() == ".zip" or zipfile.is_zipfile(path))


def _zip_dataset_yaml(extract_dir: Path) -> Path:
    """dataset.yaml of an extracted training ZIP, rewritten to point inside ``extract_dir``.

    The ZIP may hold the dataset at its root or inside one top-level folder. The split paths of
    its yaml are not trusted: the delivered create_dataset_yaml() writes absolute paths of the
    author's machine, and a user's yaml could point anywhere in the container. They are replaced
    by the layout TrainRequest documents (images/{train,val[,test]} next to the yaml).
    """
    entries = [e for e in extract_dir.iterdir() if e.name != "__MACOSX"]
    root = entries[0] if len(entries) == 1 and entries[0].is_dir() else extract_dir
    dataset_yaml = _resolve_dataset_yaml(str(root))
    base = dataset_yaml.parent
    with open(dataset_yaml, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    names = cfg.get("names")
    n_classes = len(names) if names is not None else cfg.get("nc")
    if n_classes != len(TRAINING_CLASS_ID_TO_CODE):
        raise ValueError(
            f"El dataset.yaml declara {n_classes} clases; el detector usa "
            f"{len(TRAINING_CLASS_ID_TO_CODE)} ({', '.join(TRAINING_CLASS_ID_TO_CODE.values())})."
        )
    for split in ("train", "val"):
        if not (base / "images" / split).is_dir():
            raise ValueError(f"El ZIP no contiene images/{split}/ junto a {dataset_yaml.name}")

    cfg.update(path=str(base), train="images/train", val="images/val")
    if (base / "images" / "test").is_dir():
        cfg["test"] = "images/test"
    else:
        cfg.pop("test", None)
    with open(dataset_yaml, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return dataset_yaml


@contextmanager
def _local_dataset_yaml(data_path: str) -> Iterator[Path]:
    """Yield the dataset.yaml to train on, from a ZIP (local or s3://), a dir or a yaml.

    A ZIP is extracted with safe_extract_zip to a temp dir, removed on exit. A local dir or
    yaml is used in place, as before.
    """
    with local_file_path(data_path) as local_path:
        local = Path(local_path)
        if _is_zip(local):
            extract_dir = Path(tempfile.mkdtemp(prefix="ml7_train_data_"))
            try:
                safe_extract_zip(str(local), str(extract_dir))
                yield _zip_dataset_yaml(extract_dir)
            finally:
                shutil.rmtree(extract_dir, ignore_errors=True)
        elif data_path.startswith("s3://"):
            raise ValueError(f"Un data_path s3:// debe ser un ZIP con el dataset YOLO: {data_path}")
        else:
            yield _resolve_dataset_yaml(local_path)


def _count_images(dataset_yaml: Path, split: str) -> int:
    """Count images for a given split declared in dataset.yaml (best-effort, for metrics)."""
    try:
        with open(dataset_yaml, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        split_dir = Path(cfg[split])
        if not split_dir.is_absolute():
            split_dir = dataset_yaml.parent / split_dir
        return sum(1 for f in split_dir.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
    except Exception as exc:  # pylint: disable=broad-exception-caught  # count is best-effort
        logger.warning("Could not count images for split=%s: %s", split, exc)
        return 0


def _validate_best(
    best_weights: Path, dataset_yaml: Path, device, project_dir: Path
) -> dict[str, float]:
    """Evaluate the fine-tuned checkpoint on the val split.

    Same as src/training/validation.py::validate_model(). ``project_dir`` is train()'s temp dir:
    without it Ultralytics writes runs/detect/val-N under its global runs_dir.
    """
    val_model = _load_yolo_from_path(best_weights)
    val_metrics = val_model.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=IMG_SIZE,
        batch=TRAIN_BATCH_SIZE,
        device=device,
        plots=False,
        conf=TRAIN_EVAL_CONF,
        project=str(project_dir),
        name="val",
        exist_ok=True,
    )
    return {
        "map50": float(val_metrics.box.map50),
        "map50_95": float(val_metrics.box.map),
        "precision": float(val_metrics.box.mp),
        "recall": float(val_metrics.box.mr),
    }


class Ml7CerealsGrainPestDetectionPlugin(PluginRuntimeMixin, ModelPluginPort):
    """YOLO detector for stored-grain pest species."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin (runtime counters start at the mixin defaults)."""
        self._model = None
        self._device = None

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

    def _resolve_model_for_predict(self, mlflow_run_id: str):
        """Return (model, temp_dir_to_cleanup) — the user's MLflow checkpoint if
        mlflow_run_id is given and downloadable, otherwise the served base model.
        Never mutates self._model, so concurrent requests without mlflow_run_id are
        unaffected by one that uses a retrained checkpoint."""
        if not mlflow_run_id:
            return self._require_model(), None
        logger.info("Using user-trained model from MLflow run_id=%s", mlflow_run_id)
        loaded = download_user_model_from_mlflow(mlflow_run_id)
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
        with removing_dir(temp_dir):
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

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Detect pests in every image of a directory or ZIP."""
        model, mlflow_temp_dir = self._resolve_model_for_predict(mlflow_run_id)
        tmp_dir: str | None = None

        with removing_dir(mlflow_temp_dir):
            with local_file_path(data_path) as local_data_path:
                data_p = Path(local_data_path)

                if data_p.suffix.lower() == ".zip":
                    if not zipfile.is_zipfile(data_p):
                        raise ValueError(f"data_path is not a valid ZIP file: {data_path}")
                    tmp_dir = tempfile.mkdtemp(prefix="grain_pest_batch_")
                    safe_extract_zip(data_p, tmp_dir)
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
                        with row_error_capture(
                            predictions, {"filename": img_file.name}, what=img_file.name
                        ):
                            img_np = image_path_to_numpy(str(img_file))
                            results = model.predict(
                                _to_bgr(img_np), verbose=False, conf=CONF_THRESHOLD,
                                imgsz=IMG_SIZE, device=self._device,
                            )
                            result = yolo_results_to_dict(results[0], img_np)
                            predictions.append({"filename": img_file.name, **result})

                    self._record()
                    return PredictBatchResponse(
                        model_id=MODEL_ID, predictions=predictions, output_path=None
                    )
                finally:
                    if tmp_dir is not None:
                        shutil.rmtree(tmp_dir, ignore_errors=True)

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Fine-tune the YOLO detector, faithfully porting
        src/training/train.py::train_model() from inbox/a07/codigo/.

        Loads a FRESH YOLO instance from the served base weights (never mutates
        self._model in place, so concurrent predict() calls without mlflow_run_id keep
        using the untouched served model). The fine-tuned checkpoint is written only to
        a temp dir and uploaded to MLflow — the served artifacts/.../best.pt is never
        overwritten (see inbox/a07/manifest.yaml::training).

        ``data_path``: a ZIP of the YOLO dataset (local or s3://, like the other image plugins),
        or a local dataset root / dataset.yaml.
        """
        # Without a run the retrained checkpoint would be lost: fail before training.
        require_train_tracker(mlflow_run_id)
        with _local_dataset_yaml(data_path) as dataset_yaml:
            return self._train_on_dataset(dataset_yaml, mlflow_run_id)

    @staticmethod
    def _train_on_dataset(dataset_yaml: Path, mlflow_run_id: str) -> TrainResponse:
        """Fine-tune on a resolved dataset.yaml, validate and upload best.pt to MLflow."""
        device = safe_device()
        train_project_dir = Path(tempfile.mkdtemp(prefix="ml7_train_"))
        upload_dir = Path(tempfile.mkdtemp(prefix="ml7_train_upload_"))
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

            val = _validate_best(best_weights, dataset_yaml, device, train_project_dir)
            n_images = _count_images(dataset_yaml, "train")

            shutil.copy2(best_weights, upload_dir / MODEL_FILENAME)

            try:
                upload_artifacts_to_mlflow(
                    str(upload_dir), mlflow_run_id, metrics={**val, "n_images": n_images},
                )
            except Exception as exc:
                logger.error("MLflow artifact upload failed: %s", exc)
                raise ModelPersistenceError(
                    f"El modelo reentrenado no se ha podido guardar en MLflow: {exc}"
                ) from exc

            logger.info(
                "train() done — map50=%.4f map50_95=%.4f precision=%.4f recall=%.4f "
                "n_images=%d mlflow=%s",
                val["map50"], val["map50_95"], val["precision"], val["recall"], n_images,
                bool(mlflow_run_id),
            )
            return TrainResponse(
                detail="Fine-tuning completado",
                map50=round(val["map50"], 4),
                map50_95=round(val["map50_95"], 4),
                precision=round(val["precision"], 4),
                recall=round(val["recall"], 4),
                n_images=n_images,
                upload_warning=None,
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
            except Exception as exc:  # pylint: disable=broad-exception-caught  # stats stay usable
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
            runtime_stats=self._runtime_stats(),
        )


def _load_yolo_from_path(path: Path):
    """Load a YOLO model directly from a checkpoint path (used for post-train val())."""
    from ultralytics import YOLO  # noqa: PLC0415 # pylint: disable=import-outside-toplevel

    return YOLO(str(path))
