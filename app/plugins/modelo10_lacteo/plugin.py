"""Plugin Modelo10Lacteo: Detección y clasificación de vectores en imágenes de ganado lechero."""
from __future__ import annotations

import csv
import gc
import json
import logging
import os
import random
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from datetime import datetime, timezone

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.application.dto.train_dto import TrainResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import InvalidImageError, ModelNotLoadedError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.modelo10_lacteo.model_loader import load_detector_and_classifier, safe_device
from app.plugins.modelo10_lacteo.postprocessing import build_inline_result, classify_crop
from app.plugins.modelo10_lacteo.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.modelo10_lacteo.preprocessing import (
    crop_to_tensor,
    image_base64_to_pil,
    image_path_to_pil,
)
from app.plugins.modelo10_lacteo.constants import (
    ARTIFACT_FOLDER_NAME,
    CLASSIFIER_FILENAME,
    CLASS_NAMES_FILENAME,
    DEFAULT_CLS_CONF,
    DEFAULT_DET_CONF,
    FRAMEWORK,
    MODEL_ID,
    VERSION,
)
from app.plugins.modelo10_lacteo.mlflow_utils import download_user_classifier_from_mlflow

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_CLASSES = ["fly", "mos", "tick"]

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)

# ── Training helpers ──────────────────────────────────────────────────────────


def _cls_transforms(imgsz: int = 224):
    """Build train/eval transforms for the MobileNetV3 classifier."""
    train_tfm = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tfm, eval_tfm


def _create_splits(data_root: Path, classes: list, tmp_base: str) -> Path:
    """Auto-split flat {class}/*.jpg → train/val/test folders (80/10/10)."""
    splits_root = Path(tmp_base) / "_splits"
    for phase in ("train", "val", "test"):
        for cls in classes:
            (splits_root / phase / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        cls_src = data_root / cls
        if not cls_src.is_dir():
            continue
        files = [f for f in cls_src.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        random.shuffle(files)
        total = len(files)
        idx_train = int(total * 0.8)
        idx_val = idx_train + int(total * 0.1)
        for phase, batch in zip(
            ("train", "val", "test"),
            (files[:idx_train], files[idx_train:idx_val], files[idx_val:]),
        ):
            for f in batch:
                shutil.copy2(str(f), str(splits_root / phase / cls / f.name))

    return splits_root


def _cls_train_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch and return average loss and accuracy."""
    model.train()
    total_loss = correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += inputs.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def _cls_validate(model, loader, criterion, device):
    """Run one validation pass and return average loss and accuracy."""
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += inputs.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


class Modelo10LacteoPlugin(ModelPluginPort):
    """Plugin para detección y clasificación de vectores en imágenes de ganado lechero.
    Usa un pipeline de dos etapas: detección con un modelo YOLOv8 y clasificación con MobileNetV3.
    Soporta predicción inline (una imagen) y batch (CSV/ZIP/directorio).
    El entrenamiento solo afecta al clasificador MobileNetV3, que se guarda como artifact."""

    MODEL_ID = MODEL_ID
    FRAMEWORK = FRAMEWORK
    VERSION = VERSION

    def __init__(self) -> None:
        """Initialize plugin with no loaded models and zero stats."""
        self._detector = None
        self._classifier = None
        self._class_names: list[str] = []
        self._device = safe_device()
        self._predict_count: int = 0
        self._total_latency_ms: float = 0.0
        self._last_predict_at: str | None = None
        self._model_metrics: dict = {}

    def load(self) -> None:
        """Carga los modelos desde artifacts/ y los prepara para inferencia."""
        self._detector, self._classifier, self._class_names = load_detector_and_classifier(self._device)
        logger.info("Plugin Modelo10Lacteo listo en device=%s, clases=%s", self._device, self._class_names)

    def is_loaded(self) -> bool:
        """Devuelve True si el detector y el clasificador están cargados."""
        return self._detector is not None and self._classifier is not None

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Ejecuta inferencia en una sola imagen (base64 o path) y devuelve la predicción."""
        self._assert_loaded()
        user_clf = None
        user_cls_names = None
        user_temp_dir = None

        if mlflow_run_id:
            logger.info(" [INLINE] Loading user-trained classifier from MLflow run_id=%s", mlflow_run_id)
            loaded = download_user_classifier_from_mlflow(mlflow_run_id)
            if loaded:
                logger.info(" [INLINE] MLflow classifier loaded successfully")
                user_clf, user_cls_names, user_temp_dir = loaded

        if features.get("image_path"):
            img_path = features["image_path"]
            ext = os.path.splitext(img_path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise InvalidImageError(f"Extensión no soportada: {ext}. Usa {SUPPORTED_EXTENSIONS}")
            image_pil = image_path_to_pil(img_path)
        elif features.get("image_base64"):
            image_pil = image_base64_to_pil(features["image_base64"])
        else:
            raise ValueError("features debe contener 'image_path' o 'image_base64'")

        det_conf = float(features.get("det_conf_thresh", DEFAULT_DET_CONF))
        cls_conf = float(features.get("cls_conf_thresh", DEFAULT_CLS_CONF))

        t0 = time.perf_counter()

        try:
            detections = self._run_pipeline(
                image_pil, det_conf, cls_conf,
                classifier=user_clf, class_names=user_cls_names,
            )
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)

        self._update_stats(latency_ms=(time.perf_counter() - t0) * 1000)

        return PredictInlineResponse(**build_inline_result(self.MODEL_ID, detections))

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Ejecuta inferencia en batch sobre un CSV/ZIP/directorio de imágenes y devuelve las predicciones."""
        self._assert_loaded()
        user_clf = None
        user_cls_names = None
        user_temp_dir = None

        if mlflow_run_id:
            logger.info(" [BATCH] Loading user-trained classifier from MLflow run_id=%s", mlflow_run_id)
            loaded = download_user_classifier_from_mlflow(mlflow_run_id)
            if loaded:
                logger.info(" [BATCH] MLflow classifier loaded successfully")
                user_clf, user_cls_names, user_temp_dir = loaded

        _tmp_zip: str | None = None
        local_data_path = data_path
        if data_path.startswith("s3://"):
            import boto3
            from botocore.client import Config as BotoConfig
            without_prefix = data_path[5:]
            bucket, _, s3_key = without_prefix.partition("/")
            s3 = boto3.client(
                "s3",
                endpoint_url=os.environ.get("CUSTOM_S3_ENDPOINT"),
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_ID"),
                config=BotoConfig(signature_version="s3v4"),
                region_name=os.environ.get("CUSTOM_REGION", "us-east-1"),
            )
            fd, _tmp_zip = tempfile.mkstemp(suffix=".zip")
            os.close(fd)
            logger.info("Downloading batch data from s3://%s/%s", bucket, s3_key)
            s3.download_file(bucket, s3_key, _tmp_zip)
            local_data_path = _tmp_zip

        temp_dir: str | None = None
        image_files: list[Path] = []
        predictions = []
        temps: list[str] = []
        t0 = time.perf_counter()

        try:
            if local_data_path.lower().endswith(".csv"):
                image_files = self._image_paths_from_csv(local_data_path)
            elif local_data_path.lower().endswith(".zip"):
                temp_dir = tempfile.mkdtemp(prefix="modelo10_lacteo_batch_")
                with zipfile.ZipFile(local_data_path, "r") as zf:
                    zf.extractall(temp_dir)
                entries = list(Path(temp_dir).iterdir())
                image_dir = entries[0] if len(entries) == 1 and entries[0].is_dir() else Path(temp_dir)
                image_files = sorted(f for f in image_dir.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS)
            else:
                image_dir = Path(local_data_path)
                if not image_dir.is_dir():
                    raise ValueError(f"data_path debe ser CSV, ZIP o directorio, recibido: {local_data_path}")
                image_files = sorted(f for f in image_dir.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS)

            temps = [d for d in (temp_dir, user_temp_dir) if d]

            if not image_files:
                for d in temps:
                    shutil.rmtree(d, ignore_errors=True)
                raise ValueError(f"No se encontraron imágenes en: {data_path}")

            for img_path in image_files:
                try:
                    image_pil = image_path_to_pil(str(img_path))
                    detections = self._run_pipeline(image_pil, DEFAULT_DET_CONF, DEFAULT_CLS_CONF, classifier=user_clf, class_names=user_cls_names)
                    row = build_inline_result(self.MODEL_ID, detections)
                    row.pop("model_id", None)
                    predictions.append({"filename": img_path.name, **row})
                except Exception as exc:
                    logger.warning("Error procesando %s: %s", img_path.name, exc)
                    predictions.append({"filename": img_path.name, "status": "error", "error_message": str(exc)})
        finally:
            for d in temps:
                shutil.rmtree(d, ignore_errors=True)
            if _tmp_zip:
                os.unlink(_tmp_zip)

        self._update_stats(latency_ms=(time.perf_counter() - t0) * 1000)
        return PredictBatchResponse(
            model_id=self.MODEL_ID, predictions=predictions, output_path=None
        )

    # ── get_stats ─────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Devuelve estadísticas de uso y metadata del modelo."""
        avg = self._total_latency_ms / self._predict_count if self._predict_count > 0 else None
        base = StatsResponse(
            model_name=self.MODEL_ID,
            version=self.VERSION,
            description="Detección y clasificación de vectores (mosca, mosquito, garrapata) en imágenes de ganado lechero.",
            task_type="object-detection+classification",
            framework=self.FRAMEWORK,
            inputs=[
                InputField(
                    name="image",
                    type="file",
                    format=["jpg", "jpeg", "png", "bmp", "tif"],
                    description="Imagen de ganado lechero (base64 para inline, CSV/ZIP/directorio para batch)",
                ),
            ],
            outputs=[
                OutputField(name="prediction", type="str", description="Especie dominante detectada (fly | mos | tick | no_vectors)"),
                OutputField(name="confidence", type="float", description="Confianza de clasificación de la detección dominante [0, 1]"),
                OutputField(name="vectors_count", type="int", description="Número total de vectores detectados y clasificados"),
                OutputField(name="detections", type="list", description="Lista de detecciones con species, det_conf, cls_conf, bbox"),
                OutputField(name="species_summary", type="dict", description="Resumen de cantidad por especie"),
            ],
            metrics={},
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=avg,
            ),
        )
        if mlflow_run_id:
            try:
                tracker = BaseMLflowTracker(mlflow_run_id)
                base.metrics["mlflow"] = {
                    "params": tracker.get_params(),
                    "metrics": tracker.get_metrics(),
                }
                logger.info("Stats enriched with MLflow data for run_id=%s", mlflow_run_id)
            except Exception as exc:
                logger.warning("Could not fetch MLflow stats for run_id=%s: %s", mlflow_run_id, exc)
        return base

    # ── private helpers ───────────────────────────────────────────────────────
    def _run_pipeline(
        self, image_pil, det_conf_thresh: float, cls_conf_thresh: float,
        classifier=None, class_names=None,
    ) -> list[dict]:
        """Ejecuta el pipeline de dos etapas.

        Si se proporcionan classifier/class_names se usan en lugar de los atributos
        de instancia (para clasificadores de usuario sin sobrescribir los generales).
        """
        clf = classifier if classifier is not None else self._classifier
        cnames = class_names if class_names is not None else self._class_names

        results = self._detector.predict(source=image_pil, conf=det_conf_thresh, save=False, verbose=False)
        if not results:
            return []

        detections = []
        res = results[0]
        for box in res.boxes:
            det_conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue

            tensor = crop_to_tensor(image_pil, x1, y1, x2, y2)
            species, cls_conf = classify_crop(clf, tensor, cnames, self._device)

            if cls_conf >= cls_conf_thresh:
                detections.append({
                    "species": species,
                    "det_conf": round(det_conf, 4),
                    "cls_conf": round(cls_conf, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })
        return detections

    def _image_paths_from_csv(self, csv_path: str) -> list[Path]:
        """Lee la columna image_path de un CSV y devuelve las rutas como Path."""
        paths = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ip = row.get("image_path", "").strip()
                if ip:
                    paths.append(Path(ip))
        return paths

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> TrainResponse:
        """Train the MobileNetV3 classifier from a ZIP.

        Accepted ZIP structures:
          - Flat:     {fly|mos|tick}/*.jpg  (auto-split 80/10/10)
          - Pre-split: {train|val}/{fly|mos|tick}/*.jpg
        If mlflow_run_id is provided, logs params/metrics and uploads artifacts to MLflow.
        """
        _tmp_zip: str | None = None
        local_data_path = data_path
        if data_path.startswith("s3://"):
            import boto3
            from botocore.client import Config as BotoConfig
            without_prefix = data_path[5:]
            bucket, _, s3_key = without_prefix.partition("/")
            s3 = boto3.client(
                "s3",
                endpoint_url=os.environ.get("CUSTOM_S3_ENDPOINT"),
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_ID"),
                config=BotoConfig(signature_version="s3v4"),
                region_name=os.environ.get("CUSTOM_REGION", "us-east-1"),
            )
            fd, _tmp_zip = tempfile.mkstemp(suffix=".zip")
            os.close(fd)
            logger.info("Downloading training data from s3://%s/%s", bucket, s3_key)
            s3.download_file(bucket, s3_key, _tmp_zip)
            local_data_path = _tmp_zip

        if not local_data_path.lower().endswith(".zip"):
            raise ValueError("data_path debe ser un fichero .zip")

        temp_dir = tempfile.mkdtemp(prefix="modelo10_train_")
        try:
            with zipfile.ZipFile(local_data_path, "r") as zf:
                zf.extractall(temp_dir)

            entries = list(Path(temp_dir).iterdir())
            data_root = entries[0] if len(entries) == 1 and entries[0].is_dir() else Path(temp_dir)

            has_train_dir = (data_root / "train").is_dir()
            has_flat_classes = any((data_root / cls).is_dir() for cls in _CLASSES)

            if has_train_dir:
                splits_root = data_root
            elif has_flat_classes:
                splits_root = _create_splits(data_root, _CLASSES, temp_dir)
                logger.info("Auto-split 80/10/10 creado en %s", splits_root)
            else:
                raise ValueError(
                    "ZIP sin estructura válida. "
                    "Esperado: {fly|mos|tick}/*.jpg o {train|val}/{fly|mos|tick}/*.jpg"
                )

            train_tfm, eval_tfm = _cls_transforms(224)
            train_ds = datasets.ImageFolder(str(splits_root / "train"), train_tfm)
            val_ds = datasets.ImageFolder(str(splits_root / "val"), eval_tfm)
            class_names = train_ds.classes
            logger.info("Dataset: %d train, %d val, clases=%s", len(train_ds), len(val_ds), class_names)

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

            # Build model: pretrained MobileNetV3, frozen backbone, trainable classifier head
            os.environ.setdefault("TORCH_HOME", "/tmp/.torch")
            model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
            for param in model.parameters():
                param.requires_grad = False
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, len(class_names))
            model.to(self._device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

            best_acc = 0.0
            best_state = None
            patience_cfg = 10
            patience_counter = 0
            val_accs: list = []

            # ── MLflow logging ──────────────────────────────────────────────
            if mlflow_run_id:
                tracker = BaseMLflowTracker(mlflow_run_id)
                tracker.log_params({
                    "lr": 0.01,
                    "momentum": 0.9,
                    "batch_size": 32,
                    "max_epochs": 50,
                    "patience": patience_cfg,
                    "optimizer": "SGD",
                    "scheduler": "StepLR(step_size=10)",
                    "model": "MobileNetV3_Large",
                })

            t0 = time.perf_counter()
            for epoch in range(50):
                tr_loss, tr_acc = _cls_train_epoch(model, train_loader, optimizer, criterion, self._device)
                val_loss, val_acc = _cls_validate(model, val_loader, criterion, self._device)
                scheduler.step()
                val_accs.append(val_acc)
                if mlflow_run_id:
                    tracker.log_metrics({
                        "train_loss": tr_loss,
                        "train_accuracy": tr_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                    }, step=epoch)
                logger.info("Epoch %d | tr_loss=%.4f tr_acc=%.4f val_loss=%.4f val_acc=%.4f",
                            epoch + 1, tr_loss, tr_acc, val_loss, val_acc)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience_cfg:
                        logger.info("Early stopping en epoch %d", epoch + 1)
                        break

            elapsed = time.perf_counter() - t0
            if best_state:
                model.load_state_dict(best_state)

            # Save artifacts locally
            torch.save(model.state_dict(), _store.local_dir / CLASSIFIER_FILENAME)
            with open(_store.path(CLASS_NAMES_FILENAME), "w") as fh:
                json.dump(class_names, fh)
            logger.info("Clasificador guardado. Clases: %s", class_names)

            # ── Upload to MLflow ────────────────────────────────────────────
            if mlflow_run_id:
                try:
                    # Save to a temporary dir for MLflow upload (matching artifact_path="classifier")
                    mlflow_tmp = tempfile.mkdtemp(prefix="modelo10_mlflow_")
                    torch.save(model.state_dict(), os.path.join(mlflow_tmp, CLASSIFIER_FILENAME))
                    with open(os.path.join(mlflow_tmp, CLASS_NAMES_FILENAME), "w") as fh:
                        json.dump(class_names, fh)
                    tracker.upload_artifacts(mlflow_tmp, artifact_path="classifier")
                    shutil.rmtree(mlflow_tmp, ignore_errors=True)
                except Exception as exc:
                    logger.error("MLflow artifact upload failed: %s", exc)

            self._reload_classifier()

            # ── Log final metrics to MLflow ─────────────────────────────────
            if mlflow_run_id:
                tracker.log_metrics({
                    "best_val_accuracy": round(best_acc * 100, 1),
                    "training_time_min": round(elapsed / 60, 1),
                })

            self._model_metrics = {
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "classes": class_names,
                "epochs_run": len(val_accs),
                "best_val_acc": round(best_acc * 100, 1),
                "time_min": round(elapsed / 60, 1),
            }

            return TrainResponse(
                detail="Entrenamiento del clasificador completado",
                metrics=self._model_metrics
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if _tmp_zip and os.path.exists(_tmp_zip):
                os.unlink(_tmp_zip)
            gc.collect()

    def _reload_classifier(self) -> None:
        """Reload only the MobileNetV3 classifier from artifacts (preserves detector)."""
        with open(_store.path(CLASS_NAMES_FILENAME)) as fh:
            self._class_names = json.load(fh)
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(self._class_names))
        state_dict = torch.load(
            _store.path(CLASSIFIER_FILENAME), map_location=self._device, weights_only=False
        )
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self._device)
        self._classifier = model
        logger.info("Clasificador recargado. Clases: %s", self._class_names)

    def _assert_loaded(self) -> None:
        """Lanza un error si el modelo no está cargado."""
        if not self.is_loaded():
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _update_stats(self, latency_ms: float = 0.0) -> None:
        """Actualiza las estadísticas de uso del modelo después de una predicción."""
        self._predict_count += 1
        self._total_latency_ms += latency_ms
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
