"""Ml4LacticCnnThermal — thermal-udder subclinical-mastitis (SCM) image classifier.

Serves a fixed EfficientNet-B0 checkpoint trained by the AI team. train() ALSO supports a
real fine-tuning procedure (manifest.training.supported=true, inbox/a04/manifest.yaml) that
clones the served model, fine-tunes it on user-supplied labeled images and persists the
result to MLflow only — the fixed base artifact in S3/artifacts/ is never overwritten.
"""
from __future__ import annotations

import base64
import copy
import csv
import io
import logging
import os
import tempfile
import time
import zipfile
from typing import Any

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.application.dto.stats_dto import InputField, OutputField, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.batch_rows import row_error_capture
from app.domain.services.exceptions import InvalidImageError
from app.domain.services.image_cam import compute_and_encode_cam
from app.domain.services.mlflow_artifacts import (
    add_mlflow_run_to_stats,
    removing_dir,
    require_train_tracker,
    upload_model_to_mlflow,
)
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.infrastructure.archive import safe_extract_zip
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.constants import (
    BACKBONE,
    DROPOUT,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    FRAMEWORK,
    IMAGE_EXTENSIONS,
    MIN_TRAIN_SAMPLES,
    MODEL_FILENAME,
    MODEL_ID,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    SCHEDULER_PATIENCE,
    TRAIN_BATCH_SIZE,
    TRAIN_EARLY_STOPPING_PATIENCE,
    TRAIN_LEARNING_RATE,
    TRAIN_MAX_EPOCHS,
    TRAIN_SEED,
    TRAIN_VAL_SPLIT,
    TRAIN_WEIGHT_DECAY,
    VERSION,
)
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.mlflow_utils import (
    download_user_model_from_mlflow,
)
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.model_loader import load_model
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.postprocessing import decode_logits
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.preprocessing import (
    load_and_augment_for_training,
    load_and_transform_for_eval,
    preprocess_image,
)
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.train_dto import TrainResponse

logger = logging.getLogger(__name__)

_LABELS_FILENAME = "ID_Labels.csv"

# Hyperparameters logged to the retrain's MLflow run (manifest.training, see constants.py).
_TRAIN_PARAMS = {
    "optimizer": "adam",
    "learning_rate": TRAIN_LEARNING_RATE,
    "weight_decay": TRAIN_WEIGHT_DECAY,
    "batch_size": TRAIN_BATCH_SIZE,
    "loss": "focal_loss",
    "focal_alpha": FOCAL_ALPHA,
    "focal_gamma": FOCAL_GAMMA,
    "scheduler": "reduce_on_plateau",
    "scheduler_factor": SCHEDULER_FACTOR,
    "scheduler_patience": SCHEDULER_PATIENCE,
    "early_stopping_patience": TRAIN_EARLY_STOPPING_PATIENCE,
    "max_epochs": TRAIN_MAX_EPOCHS,
    "image_size": 224,
    "backbone": BACKBONE,
    "dropout": DROPOUT,
    "seed": TRAIN_SEED,
}


class _FocalLoss(nn.Module):
    """Focal Loss — transcribed verbatim from the delivered src/training/losses.py::FocalLoss.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA) -> None:
        """Store the focusing hyperparameters (same defaults as the original training run)."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the mean focal loss over the batch."""
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        return (self.alpha * (1 - p_t) ** self.gamma * ce_loss).mean()


class _ThermalImageDataset(Dataset):
    """Loads (image_path, label) pairs with either the train-augmentation or eval transform."""

    def __init__(self, paths: list[str], labels: list[int], is_train: bool) -> None:
        """Store the file list/labels and which transform pipeline to apply per item."""
        self.paths = paths
        self.labels = labels
        self.is_train = is_train

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.paths)

    def __getitem__(self, idx: int):
        """Load and transform image ``idx``; returns ``(tensor, label)``."""
        loader = load_and_augment_for_training if self.is_train else load_and_transform_for_eval
        return loader(self.paths[idx]), self.labels[idx]


def _read_inline_image(features: dict) -> tuple[bytes, list[str]]:
    """Return the image bytes of an inline request and which feature carried them."""
    if features.get("image_path"):
        path = features["image_path"]
        if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS:
            raise InvalidImageError(
                f"Extensión no soportada. Usa {sorted(IMAGE_EXTENSIONS)}"
            )
        with open(path, "rb") as fh:
            return fh.read(), ["image_path"]
    if features.get("image_base64"):
        return base64.b64decode(features["image_base64"]), ["image_base64"]
    raise InvalidImageError("features debe contener 'image_path' o 'image_base64'")


def _inline_heatmap(model, feature_map, result: dict, image_bytes: bytes) -> str | None:
    """CAM overlay of the inline image, or None if it cannot be built."""
    try:
        source_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return compute_and_encode_cam(
            feature_map, model.classifier[1].weight,
            result["predicted_class_index"], source_image,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught  # heatmap is auxiliary: never fail the prediction for it
        logger.warning("ml4 inline CAM failed: %s", exc)
        return None


def _make_loaders(paths: list[str], labels: list[int]) -> tuple[DataLoader, DataLoader, int, int]:
    """Stratified train/val split (seeded) and its loaders; returns (train, val, n_train, n_val)."""
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=TRAIN_VAL_SPLIT, stratify=labels, random_state=TRAIN_SEED,
    )
    train_loader = DataLoader(
        _ThermalImageDataset(train_paths, train_labels, is_train=True),
        batch_size=TRAIN_BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        _ThermalImageDataset(val_paths, val_labels, is_train=False),
        batch_size=TRAIN_BATCH_SIZE, shuffle=False,
    )
    return train_loader, val_loader, len(train_paths), len(val_paths)


def _train_epoch(model, train_loader, criterion, optimizer, device) -> None:
    """One optimisation pass over the training split."""
    model.train()
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        optimizer.step()


def _single_subdir_or_self(folder: str) -> str:
    """The only entry of ``folder`` when it is a directory (ZIP with a root folder), else folder."""
    entries = [e for e in os.listdir(folder) if e != "__MACOSX"]
    if len(entries) == 1 and os.path.isdir(os.path.join(folder, entries[0])):
        return os.path.join(folder, entries[0])
    return folder


def _is_zip(local_path: str) -> bool:
    """True for a ZIP file, by extension or by content (an s3:// key may have no ".zip" suffix)."""
    return os.path.isfile(local_path) and (
        local_path.lower().endswith(".zip") or zipfile.is_zipfile(local_path)
    )


def _labels_root(folder: str) -> str:
    """Directory under ``folder`` holding ID_Labels.csv (root first, then any depth).

    A user-built ZIP may wrap the dataset in one or more folders (or add ``__MACOSX/``); the
    TIDS layout is anchored on ID_Labels.csv, so that file locates the dataset root.
    """
    if os.path.exists(os.path.join(folder, _LABELS_FILENAME)):
        return folder
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d != "__MACOSX")
        if _LABELS_FILENAME in files:
            return root
    return folder


def _predict_batch_image(model, device, root: str, fname: str, idx: int) -> dict:
    """Classify one batch image (with its CAM); the caller turns errors into rows."""
    with open(os.path.join(root, fname), "rb") as fh:
        image_bytes = fh.read()
    tensor = preprocess_image(image_bytes).to(device)
    with torch.no_grad():
        logits, feature_map = model(tensor, return_features=True)
    result = decode_logits(logits)
    row = {"filename": fname, **result}
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    row["heatmap_url"] = compute_and_encode_cam(
        feature_map, model.classifier[1].weight,
        result["predicted_class_index"], image,
    )
    if idx == 0 or result["predicted_class_index"] != 0:
        # Echo the first image back so the platform can request a real
        # GradCAM explanation for the batch (mirrors the inline flow,
        # which has a single image_path to read from — batch has none
        # once this temp dir is removed below). Non-healthy images
        # (class index != 0) are echoed too so their instances carry
        # the source image; healthy ones stay out to keep the payload small.
        row["image_base64"] = base64.b64encode(image_bytes).decode("ascii")
    return row


def _read_labeled_images(data_root: str) -> tuple[list[str], list[int]]:
    """Read ID_Labels.csv (TIDS layout) and resolve every labeled image path."""
    csv_path = os.path.join(data_root, _LABELS_FILENAME)
    if not os.path.exists(csv_path):
        raise ValueError(
            "data_path debe ser un ZIP o una carpeta con 'ID_Labels.csv' (columnas 'ID' y "
            "'label', 0=Healthy, 1=SCM) y las imágenes en TIDS_cropped/{healthy,SCM}/<ID>.jpg, "
            "igual que el dataset TIDS original."
        )

    paths: list[str] = []
    labels: list[int] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"ID", "label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"ID_Labels.csv no contiene las columnas requeridas: {sorted(missing)}"
            )
        for row in reader:
            img_id = str(row["ID"]).strip()
            label = int(row["label"])
            folder = "SCM" if label == 1 else "healthy"
            img_path = os.path.join(data_root, "TIDS_cropped", folder, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                raise ValueError(f"Imagen no encontrada para ID={img_id}: {img_path}")
            paths.append(img_path)
            labels.append(label)
    return paths, labels


def _load_training_set(data_path: str, tmp_dir: str) -> tuple[list[str], list[int]]:
    """Resolve the training folder (extracting a ZIP into ``tmp_dir``) and read its labels."""
    with local_file_path(data_path) as local_path:
        if _is_zip(local_path):
            safe_extract_zip(local_path, tmp_dir)
            data_root = _labels_root(tmp_dir)
        elif os.path.isdir(local_path):
            data_root = _labels_root(local_path)
        else:
            raise ValueError(
                f"data_path no es un ZIP ni una carpeta: {data_path}. Sube un .zip con "
                f"'{_LABELS_FILENAME}' y TIDS_cropped/{{healthy,SCM}}/<ID>.jpg."
            )
        paths, labels = _read_labeled_images(data_root)

    if len(paths) < MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"Se requieren al menos {MIN_TRAIN_SAMPLES} imágenes etiquetadas para "
            f"reentrenar; se encontraron {len(paths)}."
        )
    return paths, labels


def _validation_epoch(model, val_loader, criterion, device) -> tuple[float, float]:
    """Mean validation loss and binary F1 of one epoch."""
    model.eval()
    val_true, val_pred, val_loss_total = [], [], 0.0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            val_loss_total += criterion(logits, lbls).item()
            val_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
            val_true.extend(lbls.cpu().tolist())
    val_loss = val_loss_total / max(len(val_loader), 1)
    return val_loss, f1_score(val_true, val_pred, average="binary", zero_division=0)


def _final_scores(model, val_loader, device) -> dict:
    """Accuracy/precision/recall/F1/AUC of the best model on the validation split."""
    val_true, val_pred, val_prob = [], [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1)
            val_pred.extend(torch.argmax(probs, dim=1).cpu().tolist())
            val_true.extend(lbls.tolist())
            val_prob.extend(probs[:, 1].cpu().tolist())

    return {
        "accuracy": accuracy_score(val_true, val_pred),
        "precision": precision_score(val_true, val_pred, average="binary", zero_division=0),
        "recall": recall_score(val_true, val_pred, average="binary", zero_division=0),
        "f1": f1_score(val_true, val_pred, average="binary", zero_division=0),
        "auc": roc_auc_score(val_true, val_prob) if len(set(val_true)) > 1 else None,
    }


class Ml4LacticCnnThermalEarlyDiseaseDetectionPlugin(PluginRuntimeMixin, ModelPluginPort):
    """EfficientNet-B0 binary classifier (Healthy / SCM) on thermal udder images."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin (runtime counters start at the mixin defaults)."""
        self._model = None
        self._device = None

    def load(self) -> None:
        """Load the EfficientNet checkpoint."""
        self._model, self._device = load_model()
        logger.info("Ml4LacticCnnThermalEarlyDiseaseDetectionPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        """Return True if the model is loaded."""
        return self._model is not None

    def _resolve_model(self, mlflow_run_id: str) -> tuple[Any, Any, str | None]:
        """Return (model, device, temp_dir) to serve this request.

        The user's retrained model (if mlflow_run_id) is returned, never assigned to self:
        the plugin instance is shared by concurrent requests, so swapping it into self leaked
        a user's model to other requests — and, with two overlapping runs, left it in place
        as the "base" model. temp_dir is None for the base model.
        """
        if not mlflow_run_id:
            self._require_loaded()
            return self._model, self._device, None
        return download_user_model_from_mlflow(mlflow_run_id)

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Classify a single thermal image (base64 or path)."""
        if mlflow_run_id:
            logger.info("predict_inline — using MLflow user model run_id=%s", mlflow_run_id)
        model, device, user_temp_dir = self._resolve_model(mlflow_run_id)
        with removing_dir(user_temp_dir):
            image_bytes, used = _read_inline_image(features)

            tensor = preprocess_image(image_bytes).to(device)
            with torch.no_grad():
                logits, feature_map = model(tensor, return_features=True)
            result = decode_logits(logits)
            heatmap_url = _inline_heatmap(model, feature_map, result, image_bytes)
            self._record()
            return PredictInlineResponse(
                model_id=MODEL_ID, threshold=threshold, features_used=used,
                heatmap_url=heatmap_url, **result,
            )

    def predict_batch(
        self, *, data_path: str, mlflow_run_id: str = "",
    ) -> PredictBatchResponse:
        """Classify every thermal image in a directory or ZIP (local path or ``s3://`` URI)."""
        if mlflow_run_id:
            logger.info("predict_batch — using MLflow user model run_id=%s", mlflow_run_id)
        model, device, user_temp_dir = self._resolve_model(mlflow_run_id)
        with removing_dir(user_temp_dir):
            temp_dir: str | None = None
            predictions: list[dict] = []
            with local_file_path(data_path) as local_data_path:
                image_dir = local_data_path
                if _is_zip(local_data_path):
                    temp_dir = tempfile.mkdtemp(prefix="ml4_thermal_batch_")
                with removing_dir(temp_dir):
                    if temp_dir:
                        safe_extract_zip(local_data_path, temp_dir)
                        image_dir = _single_subdir_or_self(temp_dir)
                    image_files = sorted(
                        (root, fname)
                        for root, _, files in os.walk(image_dir)
                        for fname in files
                        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
                    )
                    for idx, (root, fname) in enumerate(image_files):
                        # one bad image never fails the batch
                        with row_error_capture(predictions, {"filename": fname}, what=fname):
                            predictions.append(
                                _predict_batch_image(model, device, root, fname, idx)
                            )

            self._record()
            return PredictBatchResponse(
                model_id=MODEL_ID, predictions=predictions, output_path=None,
            )

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Fine-tune a fresh clone of the served EfficientNet-B0 on user-labeled thermal images.

        Follows the original recipe exactly (manifest.training in inbox/a04/manifest.yaml):
        Adam(lr=1e-4, weight_decay=1e-4) + Focal Loss(alpha=0.25, gamma=2.0) +
        ReduceLROnPlateau(factor=0.5, patience=5) + early stopping (patience=15, max 100 epochs).
        The retrained checkpoint is persisted to MLflow only — the fixed base artifact served by
        model_loader.py is never overwritten (predict with this run's mlflow_run_id to use it).
        """
        self._require_loaded()
        tracker = require_train_tracker(mlflow_run_id)
        tracker.log_params(_TRAIN_PARAMS)

        t0 = time.perf_counter()
        tmp_dir = tempfile.mkdtemp(prefix="ml4_train_")
        with removing_dir(tmp_dir):
            paths, labels = _load_training_set(data_path, tmp_dir)
            train_loader, val_loader, n_train, n_val = _make_loaders(paths, labels)

            model = self._fine_tune(train_loader, val_loader, tracker)
            scores = _final_scores(model, val_loader, self._device)

            tracker.log_metrics({
                "accuracy": scores["accuracy"], "precision": scores["precision"],
                "recall": scores["recall"], "f1": scores["f1"],
                "auc": scores["auc"] if scores["auc"] is not None else 0.0,
                "n_train": n_train, "n_val": n_val,
            })

            # Persisted ONLY to this MLflow run — the fixed base artifact filename used by
            # model_loader.py is never touched, so predict/stats without mlflow_run_id keeps
            # serving the original AI-team checkpoint.
            def _write(model_dir: str) -> None:
                torch.save(
                    {"model_state_dict": model.state_dict()},
                    os.path.join(model_dir, MODEL_FILENAME),
                )

            upload_model_to_mlflow(tracker, _write, prefix="ml4_mlflow_upload_")

            return TrainResponse(
                detail="Reentrenamiento completado",
                **scores,
                n_train=n_train,
                n_val=n_val,
                training_time_s=round(time.perf_counter() - t0, 2),
                mlflow_run_id=mlflow_run_id,
                upload_warning=None,
            )

    def _fine_tune(self, train_loader, val_loader, tracker):
        """Fine-tune a clone of the served model with early stopping on the validation F1.

        Returns the clone with the best epoch's weights loaded, in eval mode.
        """
        # Clone the served weights into a brand-new instance — self._model is never mutated
        # in-place until training finishes, so a concurrent predict never sees a half-trained
        # model (plugin-integration skill, "train() — patrón de reentrenamiento").
        torch.manual_seed(TRAIN_SEED)
        model = copy.deepcopy(self._model).to(self._device)
        criterion = _FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=TRAIN_LEARNING_RATE, weight_decay=TRAIN_WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE,
            min_lr=SCHEDULER_MIN_LR,
        )

        best_f1, best_state, patience_counter = -1.0, None, 0
        for epoch in range(TRAIN_MAX_EPOCHS):
            _train_epoch(model, train_loader, criterion, optimizer, self._device)

            val_loss, val_f1 = _validation_epoch(model, val_loader, criterion, self._device)
            scheduler.step(val_loss)
            tracker.log_metrics({"val_loss": val_loss, "val_f1": val_f1}, step=epoch)

            if val_f1 > best_f1:
                best_f1, patience_counter = val_f1, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= TRAIN_EARLY_STOPPING_PATIENCE:
                    logger.info("ml4 train(): early stopping at epoch %d", epoch + 1)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                f"Clasificación binaria de imágenes térmicas de ubre de oveja para detección de "
                f"mamitis subclínica (backbone={BACKBONE}, dropout={DROPOUT})."
            ),
            task_type="binary_classification",
            framework=FRAMEWORK,
            inputs=[
                InputField(
                    name="image",
                    type="file",
                    format=["jpg", "jpeg", "png", "bmp"],
                    description="Imagen térmica de ubre (base64/path inline; .zip o dir en batch)",
                ),
            ],
            outputs=[
                OutputField(name="prediction", type="str",
                            description="Clase predicha: 'Healthy' o 'SCM'"),
                OutputField(name="confidence", type="float",
                            description="Probabilidad softmax de la clase ganadora [0, 1]"),
                OutputField(name="probability_healthy", type="float",
                            description="P(clase=Healthy)"),
                OutputField(name="probability_scm", type="float", description="P(clase=SCM)"),
                OutputField(name="heatmap_url", type="str",
                            description=(
                                "Mapa de activación de clase (CAM) superpuesto en base64 "
                                "JPEG data URI"
                            )),
            ],
            metrics={},
            runtime_stats=self._runtime_stats(),
        )
        return add_mlflow_run_to_stats(base, mlflow_run_id)
