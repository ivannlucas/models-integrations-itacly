"""Plugin Ml10DairyDiseaseVectorDetection: Detección y clasificación de vectores en imágenes de ganado lechero."""
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
from app.domain.services.exceptions import InvalidImageError, ModelNotLoadedError, ModelPersistenceError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.archive import safe_extract_zip
from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml10_dairy_disease_vector_detection.model_loader import load_detector_and_classifier, safe_device
from app.plugins.ml10_dairy_disease_vector_detection.postprocessing import (
    build_heatmap_crops,
    build_inline_result,
    classify_crop,
    render_annotated_image,
)
from app.plugins.ml10_dairy_disease_vector_detection.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml10_dairy_disease_vector_detection.preprocessing import (
    crop_to_tensor,
    image_base64_to_pil,
    image_path_to_pil,
)
from app.plugins.ml10_dairy_disease_vector_detection.constants import (
    ARTIFACT_FOLDER_NAME,
    CLASSIFIER_FILENAME,
    CLASS_NAMES_FILENAME,
    DEFAULT_CLS_CONF,
    DEFAULT_DET_CONF,
    FRAMEWORK,
    MODEL_ID,
    VERSION,
)
from app.plugins.ml10_dairy_disease_vector_detection.mlflow_utils import download_user_classifier_from_mlflow

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_CLASSES = ["fly", "mos", "tick"]
# config/config.yaml of the delivered code: `seed` (src/utils/seed.py::set_seed) and
# `splits.max_ticks` (create_classification_splits balances ticks before splitting).
TRAIN_SEED = 42
MAX_TICKS = 1400

_store = ArtifactStore(ARTIFACT_FOLDER_NAME)

# ── Training helpers ──────────────────────────────────────────────────────────


def _cls_transforms(imgsz: int = 224):
    """Build train/eval transforms for the MobileNetV3 classifier, matching
    ``_get_classification_transforms`` in the original delivered ``src/training/trainer.py``
    exactly (same plain square ``Resize((imgsz, imgsz))`` for both train and eval, no
    CenterCrop — training images are already cropped-to-subject by
    ``data_processing/preprocess.py`` before this transform ever runs).

    eval_tfm here intentionally does NOT reuse ``preprocessing.build_eval_transform()``: that
    function now replicates the *production* full-pipeline transform
    (``Resize(256)+CenterCrop(224)``, see its docstring), which is a different, later step in
    the original pipeline applied to raw detector crops — not to the pre-cropped ImageFolder
    images this function's eval_tfm validates against during (re)training. Reusing it here
    would silently change what "best_val_acc" measures relative to the original delivered
    training code, making retrain metrics incomparable to the officially reported ones.
    """
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
    """Auto-split flat {class}/*.jpg → train/val/test folders.

    Ratios (70/15/15) and the tick cap (``splits.max_ticks``) match the original delivered
    ``config/config.yaml`` and ``create_classification_splits`` in
    ``src/training/utils_train.py`` — not chosen independently. The original shuffles
    ``os.listdir`` under the global ``set_seed(42)``; here files are sorted first and shuffled
    with a local ``Random(42)``, so the split is reproducible on any filesystem and does not
    touch the process-wide RNG of a shared server.
    """
    rng = random.Random(TRAIN_SEED)
    splits_root = Path(tmp_base) / "_splits"
    for phase in ("train", "val", "test"):
        for cls in classes:
            (splits_root / phase / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        cls_src = data_root / cls
        if not cls_src.is_dir():
            continue
        files = sorted(f for f in cls_src.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
        rng.shuffle(files)
        if cls == "tick" and len(files) > MAX_TICKS:
            files = files[:MAX_TICKS]
        total = len(files)
        idx_train = int(total * 0.7)
        idx_val = idx_train + int(total * 0.15)
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


def _cls_test_metrics(model, loader, class_names, device) -> dict:
    """Accuracy and per-class precision/recall/F1 on the test split, with the formulas of the
    delivered src/predict/predictor.py (classification_metrics.json)."""
    model.eval()
    labels_all, preds_all = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            preds_all += model(inputs.to(device)).argmax(1).cpu().tolist()
            labels_all += labels.tolist()
    pairs = list(zip(labels_all, preds_all))
    per_class = {}
    for idx, name in enumerate(class_names):
        tp = sum(1 for t, p in pairs if t == idx and p == idx)
        fp = sum(1 for t, p in pairs if t != idx and p == idx)
        fn = sum(1 for t, p in pairs if t == idx and p != idx)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": round(precision, 4), "recall": round(recall, 4),
                           "f1": round(f1, 4), "support": sum(1 for t, _ in pairs if t == idx)}
    accuracy = sum(1 for t, p in pairs if t == p) / len(pairs) if pairs else 0.0
    return {"test_accuracy": round(accuracy, 4), "test_per_class": per_class}


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


class Ml10DairyDiseaseVectorDetectionPlugin(ModelPluginPort):
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

    def load(self) -> None:
        """Carga los modelos desde artifacts/ y los prepara para inferencia."""
        self._detector, self._classifier, self._class_names = load_detector_and_classifier(self._device)
        logger.info("Plugin Ml10DairyDiseaseVectorDetection listo en device=%s, clases=%s", self._device, self._class_names)

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

        result = build_inline_result(self.MODEL_ID, detections)
        # Same annotated (bbox) image predict_batch already renders per row.
        result["annotated_image"] = render_annotated_image(image_pil, detections)
        # Per-detection crops (top 5 by cls_conf) — same crop the classifier itself was fed
        # for each one (see preprocessing.crop_to_tensor). The platform's explainability panel
        # runs Grad-CAM against each crop and composites the results onto annotated_image,
        # instead of running a single Grad-CAM against the full uncropped scene (that
        # classifier only ever saw tight per-vector crops in training, so Grad-CAM against the
        # full scene produced a heatmap with no real spatial correspondence to any detection)
        # or only explaining the single highest-confidence detection.
        result["heatmap_crops"] = build_heatmap_crops(image_pil, detections, max_crops=5)
        return PredictInlineResponse(**result)

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
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
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
                temp_dir = tempfile.mkdtemp(prefix="ml10_dairy_disease_vector_detection_batch_")
                safe_extract_zip(local_data_path, temp_dir)
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
                    row["annotated_image"] = render_annotated_image(image_pil, detections)
                    # Same per-detection crops predict_inline returns, but only on the first row:
                    # the platform's explainability panel only ever reads dataSample[0] (see
                    # runXaiExplain in ws-sources.js), so repeating up to 5 JPEG crops on every
                    # row would only bloat the batch response. Without it the panel falls back to
                    # a full-scene Grad-CAM request that carries no image_base64 at all.
                    if not predictions:
                        row["heatmap_crops"] = build_heatmap_crops(image_pil, detections, max_crops=5)
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

        # device=self._device is required here: without it, ultralytics picks its own device
        # via a bare torch.cuda.is_available() check, independent of safe_device() above —
        # on hardware where CUDA is *visible* but not actually usable by the installed torch
        # build (e.g. a GPU compute capability older than what this torch wheel ships kernels
        # for), that check returns True and the real conv kernel then crashes with
        # "CUDA error: no kernel image is available for execution on the device" on every
        # single /predict call, a 500 found only by hitting the real server (the detector is
        # mocked out in tests/unit/, so pytest never exercises this). The classifier already
        # avoided this by loading onto self._device explicitly in model_loader.py; the
        # detector must follow the same device for the two stages to agree.
        results = self._detector.predict(
            source=image_pil, conf=det_conf_thresh, device=self._device, save=False, verbose=False,
        )
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

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Train the MobileNetV3 classifier from a ZIP.

        Accepted ZIP structures:
          - Flat:     {fly|mos|tick}/*.jpg  (auto-split 70/15/15)
          - Pre-split: {train|val}/{fly|mos|tick}/*.jpg
        Logs params/metrics and uploads the classifier to the (mandatory) mlflow_run_id. Reports
        best validation accuracy and, when a test split exists, the test metrics of the original
        predictor (accuracy + per-class precision/recall/F1).
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
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID_XAI"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY_XAI"),
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

        temp_dir = tempfile.mkdtemp(prefix="ml10_dairy_train_")
        try:
            safe_extract_zip(local_data_path, temp_dir)

            entries = list(Path(temp_dir).iterdir())
            data_root = entries[0] if len(entries) == 1 and entries[0].is_dir() else Path(temp_dir)

            has_train_dir = (data_root / "train").is_dir()
            has_flat_classes = any((data_root / cls).is_dir() for cls in _CLASSES)

            if has_train_dir:
                splits_root = data_root
            elif has_flat_classes:
                splits_root = _create_splits(data_root, _CLASSES, temp_dir)
                logger.info("Auto-split 70/15/15 creado en %s", splits_root)
            else:
                raise ValueError(
                    "ZIP sin estructura válida. "
                    "Esperado: {fly|mos|tick}/*.jpg o {train|val}/{fly|mos|tick}/*.jpg"
                )

            train_tfm, eval_tfm = _cls_transforms(224)
            train_ds = datasets.ImageFolder(str(splits_root / "train"), train_tfm)
            val_ds = datasets.ImageFolder(str(splits_root / "val"), eval_tfm)
            class_names = train_ds.classes
            # Auto-split always has test/; a pre-split ZIP may bring it or not.
            test_dir = splits_root / "test"
            has_test = test_dir.is_dir() and any(p.is_file() for p in test_dir.rglob("*"))
            test_ds = datasets.ImageFolder(str(test_dir), eval_tfm) if has_test else None
            if test_ds is not None and test_ds.classes != class_names:
                raise ValueError(f"Las clases de test {test_ds.classes} no coinciden con las de train {class_names}.")
            logger.info("Dataset: %d train, %d val, %s test, clases=%s", len(train_ds), len(val_ds),
                        len(test_ds) if test_ds is not None else 0, class_names)
            torch.manual_seed(TRAIN_SEED)  # set_seed(42) of the delivered run_training

            # Hyperparams below match config/config.yaml's `classifier` section in the original
            # delivered code exactly (lr, weight_decay, optimizer=Adam, batch_size, epochs,
            # patience — see src/training/trainer.py::train_classifier /
            # _get_classification_transforms), not chosen independently.
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

            # Build model: pretrained MobileNetV3, frozen backbone, trainable classifier head
            os.environ.setdefault("TORCH_HOME", "/tmp/.torch")
            model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
            for param in model.parameters():
                param.requires_grad = False
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, len(class_names))
            model.to(self._device)

            lr = 0.007980063451685896
            weight_decay = 0.0003438715138565418
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

            best_acc = 0.0
            best_state = None
            patience_cfg = 15
            patience_counter = 0
            val_accs: list = []

            # ── MLflow logging ──────────────────────────────────────────────
            tracker = BaseMLflowTracker(mlflow_run_id)
            tracker.log_params({
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": 64,
                "max_epochs": 100,
                "patience": patience_cfg,
                "optimizer": "Adam",
                "scheduler": "StepLR(step_size=10)",
                "model": "MobileNetV3_Large",
            })

            t0 = time.perf_counter()
            for epoch in range(100):
                tr_loss, tr_acc = _cls_train_epoch(model, train_loader, optimizer, criterion, self._device)
                val_loss, val_acc = _cls_validate(model, val_loader, criterion, self._device)
                scheduler.step()
                val_accs.append(val_acc)
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

            test_metrics = {}
            if test_ds is not None:
                test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
                test_metrics = _cls_test_metrics(model, test_loader, class_names, self._device)
                logger.info("Test: accuracy=%.4f", test_metrics["test_accuracy"])

            # The retrained classifier lives only in its own MLflow run (predict with that
            # mlflow_run_id); the served base classifier and its local artifacts are never replaced.
            # mlflow_run_id is mandatory in TrainRequest, so there is always a run to save into.
            mlflow_tmp = tempfile.mkdtemp(prefix="ml10_dairy_mlflow_")
            try:
                # Save to a temporary dir for MLflow upload (matching artifact_path="classifier")
                torch.save(model.state_dict(), os.path.join(mlflow_tmp, CLASSIFIER_FILENAME))
                with open(os.path.join(mlflow_tmp, CLASS_NAMES_FILENAME), "w") as fh:
                    json.dump(class_names, fh)
                tracker.upload_artifacts(mlflow_tmp, artifact_path="classifier")
            except Exception as exc:
                logger.error("MLflow artifact upload failed: %s", exc)
                raise ModelPersistenceError(f"El modelo reentrenado no se ha podido guardar en MLflow: {exc}") from exc
            finally:
                shutil.rmtree(mlflow_tmp, ignore_errors=True)

            # ── Log final metrics to MLflow ─────────────────────────────────
            final_metrics = {
                "best_val_accuracy": round(best_acc * 100, 1),
                "training_time_min": round(elapsed / 60, 1),
            }
            if test_metrics:
                final_metrics["test_accuracy"] = test_metrics["test_accuracy"]
            tracker.log_metrics(final_metrics)

            train_metrics = {
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "classes": class_names,
                "epochs_run": len(val_accs),
                "best_val_acc": round(best_acc * 100, 1),
                "test_samples": len(test_ds) if test_ds is not None else 0,
                **test_metrics,
                "time_min": round(elapsed / 60, 1),
            }

            return TrainResponse(
                detail="Entrenamiento del clasificador completado",
                metrics=train_metrics,
                mlflow_run_id=mlflow_run_id,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if _tmp_zip and os.path.exists(_tmp_zip):
                os.unlink(_tmp_zip)
            gc.collect()

    def _assert_loaded(self) -> None:
        """Lanza un error si el modelo no está cargado."""
        if not self.is_loaded():
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _update_stats(self, latency_ms: float = 0.0) -> None:
        """Actualiza las estadísticas de uso del modelo después de una predicción."""
        self._predict_count += 1
        self._total_latency_ms += latency_ms
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
