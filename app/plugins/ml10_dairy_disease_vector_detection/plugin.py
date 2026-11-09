"""Plugin Ml10DairyDiseaseVectorDetection.

Detección y clasificación de vectores en imágenes de ganado lechero.
"""
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

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.application.dto.train_dto import TrainResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import InvalidImageError
from app.domain.services.mlflow_artifacts import (
    add_mlflow_run_to_stats,
    removing_dir,
    require_train_tracker,
    upload_model_to_mlflow,
)
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.infrastructure.archive import safe_extract_zip
from app.infrastructure.artifact_store import ArtifactStore, download_s3_file, local_file_path
from app.plugins.ml10_dairy_disease_vector_detection.model_loader import (
    load_detector_and_classifier,
    safe_device,
)
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
from app.plugins.ml10_dairy_disease_vector_detection.mlflow_utils import (
    download_user_classifier_from_mlflow,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_CLASSES = ["fly", "mos", "tick"]
# config/config.yaml of the delivered code: `seed` (src/utils/seed.py::set_seed) and
# `splits.max_ticks` (create_classification_splits balances ticks before splitting).
TRAIN_SEED = 42
MAX_TICKS = 1400
# config/config.yaml `classifier` section of the delivered code (src/training/trainer.py::
# train_classifier): Adam(lr, weight_decay), StepLR(10, 0.1), batch 64, 100 epochs, patience 15.
_CLS_LR = 0.007980063451685896
_CLS_WEIGHT_DECAY = 0.0003438715138565418
_CLS_PATIENCE = 15
_CLS_MAX_EPOCHS = 100
_CLS_BATCH_SIZE = 64

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


def _class_metrics(pairs: list[tuple[int, int]], idx: int) -> dict:
    """Precision/recall/F1/support of class ``idx`` from (true, predicted) pairs."""
    tp = sum(1 for t, p in pairs if t == idx and p == idx)
    fp = sum(1 for t, p in pairs if t != idx and p == idx)
    fn = sum(1 for t, p in pairs if t == idx and p != idx)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "support": sum(1 for t, _ in pairs if t == idx)}


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
    per_class = {name: _class_metrics(pairs, idx) for idx, name in enumerate(class_names)}
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


def _download_training_zip(data_path: str) -> tuple[str, str | None]:
    """Local path of the training ZIP: ``(path, temp file to delete or None)``."""
    if not data_path.startswith("s3://"):
        return data_path, None
    tmp_zip = download_s3_file(data_path)
    return tmp_zip, tmp_zip


def _resolve_splits_root(temp_dir: str) -> Path:
    """Folder with train/val[/test] of the extracted ZIP (auto-split 70/15/15 if it is flat)."""
    entries = list(Path(temp_dir).iterdir())
    data_root = entries[0] if len(entries) == 1 and entries[0].is_dir() else Path(temp_dir)

    has_train_dir = (data_root / "train").is_dir()
    has_flat_classes = any((data_root / cls).is_dir() for cls in _CLASSES)

    if has_train_dir:
        return data_root
    if has_flat_classes:
        splits_root = _create_splits(data_root, _CLASSES, temp_dir)
        logger.info("Auto-split 70/15/15 creado en %s", splits_root)
        return splits_root
    raise ValueError(
        "ZIP sin estructura válida. "
        "Esperado: {fly|mos|tick}/*.jpg o {train|val}/{fly|mos|tick}/*.jpg"
    )


def _load_datasets(splits_root: Path) -> tuple:
    """ImageFolder datasets ``(train, val, test or None)``; test must have train's classes."""
    train_tfm, eval_tfm = _cls_transforms(224)
    train_ds = datasets.ImageFolder(str(splits_root / "train"), train_tfm)
    val_ds = datasets.ImageFolder(str(splits_root / "val"), eval_tfm)
    class_names = train_ds.classes
    # Auto-split always has test/; a pre-split ZIP may bring it or not.
    test_dir = splits_root / "test"
    has_test = test_dir.is_dir() and any(p.is_file() for p in test_dir.rglob("*"))
    test_ds = datasets.ImageFolder(str(test_dir), eval_tfm) if has_test else None
    if test_ds is not None and test_ds.classes != class_names:
        raise ValueError(
            f"Las clases de test {test_ds.classes} no coinciden con las de train "
            f"{class_names}."
        )
    logger.info("Dataset: %d train, %d val, %s test, clases=%s", len(train_ds), len(val_ds),
                len(test_ds) if test_ds is not None else 0, class_names)
    return train_ds, val_ds, test_ds


def _build_classifier(n_classes: int, device) -> nn.Module:
    """Pretrained MobileNetV3, frozen backbone, new trainable classifier head."""
    os.environ.setdefault("TORCH_HOME", "/tmp/.torch")
    model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, n_classes)
    model.to(device)
    return model


def _fit_classifier(model, loaders: tuple, tracker: BaseMLflowTracker, device) -> dict:
    """Train the head with early stopping on val accuracy.

    Returns ``{"best_acc", "best_state", "epochs_run", "elapsed"}``; the caller restores
    ``best_state``.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.classifier.parameters(), lr=_CLS_LR, weight_decay=_CLS_WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0
    best_state = None
    patience_counter = 0
    epochs_run = 0

    tracker.log_params({
        "lr": _CLS_LR,
        "weight_decay": _CLS_WEIGHT_DECAY,
        "batch_size": _CLS_BATCH_SIZE,
        "max_epochs": _CLS_MAX_EPOCHS,
        "patience": _CLS_PATIENCE,
        "optimizer": "Adam",
        "scheduler": "StepLR(step_size=10)",
        "model": "MobileNetV3_Large",
    })

    t0 = time.perf_counter()
    for epoch in range(_CLS_MAX_EPOCHS):
        train_m = _cls_train_epoch(model, loaders[0], optimizer, criterion, device)
        val_m = _cls_validate(model, loaders[1], criterion, device)
        scheduler.step()
        epochs_run += 1
        tracker.log_metrics({
            "train_loss": train_m[0],
            "train_accuracy": train_m[1],
            "val_loss": val_m[0],
            "val_accuracy": val_m[1],
        }, step=epoch)
        logger.info("Epoch %d | tr_loss=%.4f tr_acc=%.4f val_loss=%.4f val_acc=%.4f",
                    epoch + 1, train_m[0], train_m[1], val_m[0], val_m[1])
        if val_m[1] > best_acc:
            best_acc = val_m[1]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= _CLS_PATIENCE:
                logger.info("Early stopping en epoch %d", epoch + 1)
                break

    return {"best_acc": best_acc, "best_state": best_state, "epochs_run": epochs_run,
            "elapsed": time.perf_counter() - t0}


def _is_zip_input(local_data_path: str) -> bool:
    """True for a ZIP batch input, by extension or by content.

    An s3:// input is downloaded to a temp file named after the object key's suffix, so a ZIP
    uploaded under a key without ".zip" is still recognised by its content.
    """
    path = local_data_path.lower()
    if path.endswith(".csv"):
        return False
    return path.endswith(".zip") or zipfile.is_zipfile(local_data_path)


def _batch_image_files(local_data_path: str, extract_dir: str | None) -> list[Path]:
    """Images of a batch input: a CSV with image_path, a ZIP (extracted to extract_dir) or a dir."""
    if local_data_path.lower().endswith(".csv"):
        return _image_paths_from_csv(local_data_path)
    if extract_dir is not None:
        safe_extract_zip(local_data_path, extract_dir)
        entries = list(Path(extract_dir).iterdir())
        image_dir = entries[0] if len(entries) == 1 and entries[0].is_dir() else Path(extract_dir)
    else:
        image_dir = Path(local_data_path)
        if not image_dir.is_dir():
            raise ValueError(
                f"data_path debe ser CSV, ZIP o directorio, recibido: {local_data_path}"
            )
    return sorted(f for f in image_dir.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS)


def _image_paths_from_csv(csv_path: str) -> list[Path]:
    """Lee la columna image_path de un CSV y devuelve las rutas como Path."""
    paths = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ip = row.get("image_path", "").strip()
            if ip:
                paths.append(Path(ip))
    return paths


def _load_inline_image(features: dict):
    """PIL image of an inline request (image_path or image_base64)."""
    if features.get("image_path"):
        img_path = features["image_path"]
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise InvalidImageError(
                f"Extensión no soportada: {ext}. Usa {SUPPORTED_EXTENSIONS}"
            )
        return image_path_to_pil(img_path)
    if features.get("image_base64"):
        return image_base64_to_pil(features["image_base64"])
    raise ValueError("features debe contener 'image_path' o 'image_base64'")


class Ml10DairyDiseaseVectorDetectionPlugin(PluginRuntimeMixin, ModelPluginPort):
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
        self._detector, self._classifier, self._class_names = load_detector_and_classifier(
            self._device
        )
        logger.info(
            "Plugin Ml10DairyDiseaseVectorDetection listo en device=%s, clases=%s",
            self._device, self._class_names,
        )

    def is_loaded(self) -> bool:
        """Devuelve True si el detector y el clasificador están cargados."""
        return self._detector is not None and self._classifier is not None

    # ── predict_inline ────────────────────────────────────────────────────────

    def _user_classifier(self, mlflow_run_id: str, tag: str) -> tuple:
        """``(classifier, class_names, temp_dir)`` of the user's run; all None for the base one."""
        if not mlflow_run_id:
            return None, None, None
        logger.info(
            " [%s] Loading user-trained classifier from MLflow run_id=%s", tag, mlflow_run_id
        )
        loaded = download_user_classifier_from_mlflow(mlflow_run_id)
        if not loaded:
            return None, None, None
        logger.info(" [%s] MLflow classifier loaded successfully", tag)
        return loaded

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Ejecuta inferencia en una sola imagen (base64 o path) y devuelve la predicción."""
        self._require_loaded()
        user_clf, user_cls_names, user_temp_dir = self._user_classifier(mlflow_run_id, "INLINE")

        # The user model's temp dir is removed also when the image or thresholds are invalid.
        with removing_dir(user_temp_dir):
            image_pil = _load_inline_image(features)

            det_conf = float(features.get("det_conf_thresh", DEFAULT_DET_CONF))
            cls_conf = float(features.get("cls_conf_thresh", DEFAULT_CLS_CONF))

            t0 = time.perf_counter()
            detections = self._run_pipeline(
                image_pil, det_conf, cls_conf,
                classifier=user_clf, class_names=user_cls_names,
            )

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

    def predict_batch(
        self, *, data_path: str, mlflow_run_id: str = ""
    ) -> PredictBatchResponse:
        """Ejecuta inferencia en batch sobre un CSV/ZIP/directorio de imágenes.

        Devuelve las predicciones.
        """
        self._require_loaded()
        user_clf, user_cls_names, user_temp_dir = self._user_classifier(mlflow_run_id, "BATCH")

        with removing_dir(user_temp_dir), local_file_path(data_path) as local_data_path:
            t0 = time.perf_counter()
            extract_dir = (
                tempfile.mkdtemp(prefix="ml10_dairy_disease_vector_detection_batch_")
                if _is_zip_input(local_data_path)
                else None
            )
            with removing_dir(extract_dir):
                image_files = _batch_image_files(local_data_path, extract_dir)
                if not image_files:
                    raise ValueError(f"No se encontraron imágenes en: {data_path}")
                predictions = [
                    self._predict_file(img_path, user_clf, user_cls_names, first=index == 0)
                    for index, img_path in enumerate(image_files)
                ]

        self._update_stats(latency_ms=(time.perf_counter() - t0) * 1000)
        return PredictBatchResponse(
            model_id=self.MODEL_ID, predictions=predictions, output_path=None
        )

    def _predict_file(self, img_path: Path, clf, class_names, *, first: bool) -> dict:
        """Batch row of one image (an error row when it cannot be processed)."""
        try:
            image_pil = image_path_to_pil(str(img_path))
            detections = self._run_pipeline(
                image_pil, DEFAULT_DET_CONF, DEFAULT_CLS_CONF,
                classifier=clf, class_names=class_names,
            )
            row = build_inline_result(self.MODEL_ID, detections)
            row.pop("model_id", None)
            row["annotated_image"] = render_annotated_image(image_pil, detections)
            # Same per-detection crops predict_inline returns, but only on the first row:
            # the platform's explainability panel only ever reads dataSample[0] (see
            # runXaiExplain in ws-sources.js), so repeating up to 5 JPEG crops on every
            # row would only bloat the batch response. Without it the panel falls back to
            # a full-scene Grad-CAM request that carries no image_base64 at all.
            if first:
                row["heatmap_crops"] = build_heatmap_crops(image_pil, detections, max_crops=5)
            return {"filename": img_path.name, **row}
        except Exception as exc:  # pylint: disable=broad-exception-caught  # per-image error row
            logger.warning("Error procesando %s: %s", img_path.name, exc)
            return {"filename": img_path.name, "status": "error", "error_message": str(exc)}

    # ── get_stats ─────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Devuelve estadísticas de uso y metadata del modelo."""
        avg = self._total_latency_ms / self._predict_count if self._predict_count > 0 else None
        base = StatsResponse(
            model_name=self.MODEL_ID,
            version=self.VERSION,
            description=(
                "Detección y clasificación de vectores (mosca, mosquito, garrapata) "
                "en imágenes de ganado lechero."
            ),
            task_type="object-detection+classification",
            framework=self.FRAMEWORK,
            inputs=[
                InputField(
                    name="image",
                    type="file",
                    format=["jpg", "jpeg", "png", "bmp", "tif"],
                    description=(
                        "Imagen de ganado lechero "
                        "(base64 para inline, CSV/ZIP/directorio para batch)"
                    ),
                ),
            ],
            outputs=[
                OutputField(
                    name="prediction", type="str",
                    description="Especie dominante detectada (fly | mos | tick | no_vectors)",
                ),
                OutputField(
                    name="confidence", type="float",
                    description="Confianza de clasificación de la detección dominante [0, 1]",
                ),
                OutputField(
                    name="vectors_count", type="int",
                    description="Número total de vectores detectados y clasificados",
                ),
                OutputField(
                    name="detections", type="list",
                    description="Lista de detecciones con species, det_conf, cls_conf, bbox",
                ),
                OutputField(
                    name="species_summary", type="dict",
                    description="Resumen de cantidad por especie",
                ),
            ],
            metrics={},
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=avg,
            ),
        )
        return add_mlflow_run_to_stats(base, mlflow_run_id)

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
        for box in results[0].boxes:
            detection = self._classify_box(image_pil, box, (clf, cnames), cls_conf_thresh)
            if detection is not None:
                detections.append(detection)
        return detections

    def _classify_box(self, image_pil, box, classifier: tuple, cls_conf_thresh: float):
        """Detection dict of one detector box, or None if empty or below cls_conf_thresh.

        ``classifier`` is ``(model, class_names)``.
        """
        det_conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            return None

        tensor = crop_to_tensor(image_pil, x1, y1, x2, y2)
        species, cls_conf = classify_crop(classifier[0], tensor, classifier[1], self._device)

        if cls_conf >= cls_conf_thresh:
            return {
                "species": species,
                "det_conf": round(det_conf, 4),
                "cls_conf": round(cls_conf, 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
        return None

    _image_paths_from_csv = staticmethod(_image_paths_from_csv)

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Train the MobileNetV3 classifier from a ZIP.

        Accepted ZIP structures:
          - Flat:     {fly|mos|tick}/*.jpg  (auto-split 70/15/15)
          - Pre-split: {train|val}/{fly|mos|tick}/*.jpg
        Logs params/metrics and uploads the classifier to the (mandatory) mlflow_run_id. Reports
        best validation accuracy and, when a test split exists, the test metrics of the original
        predictor (accuracy + per-class precision/recall/F1).
        """
        # The retrained classifier lives only in its own MLflow run (predict with that
        # mlflow_run_id); the served base classifier and its local artifacts are never replaced.
        tracker = require_train_tracker(mlflow_run_id)
        local_data_path, tmp_zip = _download_training_zip(data_path)

        if not (local_data_path.lower().endswith(".zip") or zipfile.is_zipfile(local_data_path)):
            raise ValueError("data_path debe ser un fichero .zip")

        temp_dir = tempfile.mkdtemp(prefix="ml10_dairy_train_")
        try:
            safe_extract_zip(local_data_path, temp_dir)
            return self._train_on_splits(_resolve_splits_root(temp_dir), tracker, mlflow_run_id)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if tmp_zip and os.path.exists(tmp_zip):
                os.unlink(tmp_zip)
            gc.collect()

    def _train_on_splits(
        self, splits_root: Path, tracker: BaseMLflowTracker, mlflow_run_id: str
    ) -> TrainResponse:
        """Train, evaluate and upload the classifier of a train/val[/test] folder."""
        train_ds, val_ds, test_ds = _load_datasets(splits_root)
        class_names = train_ds.classes
        torch.manual_seed(TRAIN_SEED)  # set_seed(42) of the delivered run_training

        loaders = (
            DataLoader(train_ds, batch_size=_CLS_BATCH_SIZE, shuffle=True, num_workers=0),
            DataLoader(val_ds, batch_size=_CLS_BATCH_SIZE, shuffle=False, num_workers=0),
        )
        model = _build_classifier(len(class_names), self._device)
        fit = _fit_classifier(model, loaders, tracker, self._device)
        if fit["best_state"]:
            model.load_state_dict(fit["best_state"])

        test_metrics = {}
        if test_ds is not None:
            test_loader = DataLoader(
                test_ds, batch_size=_CLS_BATCH_SIZE, shuffle=False, num_workers=0
            )
            test_metrics = _cls_test_metrics(model, test_loader, class_names, self._device)
            logger.info("Test: accuracy=%.4f", test_metrics["test_accuracy"])

        def _write(mlflow_tmp: str) -> None:
            # Matching artifact_path="classifier", the folder mlflow_utils downloads.
            torch.save(model.state_dict(), os.path.join(mlflow_tmp, CLASSIFIER_FILENAME))
            with open(os.path.join(mlflow_tmp, CLASS_NAMES_FILENAME), "w", encoding="utf-8") as fh:
                json.dump(class_names, fh)

        upload_model_to_mlflow(
            tracker, _write, prefix="ml10_dairy_mlflow_", artifact_path="classifier"
        )

        # ── Log final metrics to MLflow ─────────────────────────────────
        final_metrics = {
            "best_val_accuracy": round(fit["best_acc"] * 100, 1),
            "training_time_min": round(fit["elapsed"] / 60, 1),
        }
        if test_metrics:
            final_metrics["test_accuracy"] = test_metrics["test_accuracy"]
        tracker.log_metrics(final_metrics)

        return TrainResponse(
            detail="Entrenamiento del clasificador completado",
            metrics={
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "classes": class_names,
                "epochs_run": fit["epochs_run"],
                "best_val_acc": round(fit["best_acc"] * 100, 1),
                "test_samples": len(test_ds) if test_ds is not None else 0,
                **test_metrics,
                "time_min": round(fit["elapsed"] / 60, 1),
            },
            mlflow_run_id=mlflow_run_id,
        )

    def _update_stats(self, latency_ms: float = 0.0) -> None:
        """Actualiza las estadísticas de uso del modelo después de una predicción."""
        self._record()
        self._total_latency_ms += latency_ms
