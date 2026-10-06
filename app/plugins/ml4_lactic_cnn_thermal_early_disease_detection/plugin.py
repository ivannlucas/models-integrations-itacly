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
import shutil
import tempfile
import time
import zipfile
from datetime import datetime, timezone

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import (
    InvalidImageError,
    ModelNotLoadedError,
)
from app.domain.services.mlflow_tracker import BaseMLflowTracker
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
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.postprocessing import (
    compute_and_encode_cam,
    decode_logits,
)
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


class Ml4LacticCnnThermalEarlyDiseaseDetectionPlugin(ModelPluginPort):
    """EfficientNet-B0 binary classifier (Healthy / SCM) on thermal udder images."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._model = None
        self._device = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        """Load the EfficientNet checkpoint."""
        self._model, self._device = load_model()
        logger.info("Ml4LacticCnnThermalEarlyDiseaseDetectionPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        """Return True if the model is loaded."""
        return self._model is not None

    def _require_loaded(self) -> None:
        """Raise ModelNotLoadedError if the model is not loaded."""
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _infer_image(self, image_bytes: bytes) -> dict:
        """Preprocess image bytes, run the model and decode the prediction dict."""
        tensor = preprocess_image(image_bytes).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
        return decode_logits(logits)

    def _record(self) -> None:
        """Update runtime counters after a prediction."""
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def predict_inline(  # pylint: disable=too-many-locals
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Classify a single thermal image (base64 or path)."""
        user_temp_dir = None
        saved_model, saved_device = self._model, self._device
        if mlflow_run_id:
            logger.info("predict_inline — using MLflow user model run_id=%s", mlflow_run_id)
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._model, self._device, user_temp_dir = loaded
        try:
            self._require_loaded()
            if features.get("image_path"):
                path = features["image_path"]
                if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS:
                    raise InvalidImageError(
                        f"Extensión no soportada. Usa {sorted(IMAGE_EXTENSIONS)}"
                    )
                with open(path, "rb") as fh:
                    image_bytes = fh.read()
                used = ["image_path"]
            elif features.get("image_base64"):
                image_bytes = base64.b64decode(features["image_base64"])
                used = ["image_base64"]
            else:
                raise InvalidImageError("features debe contener 'image_path' o 'image_base64'")

            tensor = preprocess_image(image_bytes).to(self._device)
            with torch.no_grad():
                logits, feature_map = self._model(tensor, return_features=True)
            result = decode_logits(logits)
            heatmap_url = None
            try:
                source_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                heatmap_url = compute_and_encode_cam(
                    feature_map, self._model.classifier[1].weight,
                    result["predicted_class_index"], source_image,
                )
            except Exception as exc:  # heatmap is auxiliary: never fail the prediction for it
                logger.warning("ml4 inline CAM failed: %s", exc)
            self._record()
            return PredictInlineResponse(
                model_id=MODEL_ID, threshold=threshold, features_used=used,
                heatmap_url=heatmap_url, **result,
            )
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)
                self._model, self._device = saved_model, saved_device

    def predict_batch(  # pylint: disable=too-many-locals
        self, *, data_path: str, mlflow_run_id: str = "",
    ) -> PredictBatchResponse:
        """Classify every thermal image in a directory or ZIP (local path or ``s3://`` URI)."""
        user_temp_dir = None
        saved_model, saved_device = self._model, self._device
        if mlflow_run_id:
            logger.info("predict_batch — using MLflow user model run_id=%s", mlflow_run_id)
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._model, self._device, user_temp_dir = loaded
        try:
            self._require_loaded()
            temp_dir: str | None = None
            predictions: list[dict] = []
            with local_file_path(data_path) as local_data_path:
                image_dir = local_data_path
                if local_data_path.lower().endswith(".zip"):
                    temp_dir = tempfile.mkdtemp(prefix="ml4_thermal_batch_")
                    with zipfile.ZipFile(local_data_path, "r") as zf:
                        zf.extractall(temp_dir)
                    entries = os.listdir(temp_dir)
                    if len(entries) == 1 and os.path.isdir(os.path.join(temp_dir, entries[0])):
                        image_dir = os.path.join(temp_dir, entries[0])
                    else:
                        image_dir = temp_dir

                try:
                    image_files = sorted(
                        (root, fname)
                        for root, _, files in os.walk(image_dir)
                        for fname in files
                        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
                    )
                    for idx, (root, fname) in enumerate(image_files):
                        try:
                            with open(os.path.join(root, fname), "rb") as fh:
                                image_bytes = fh.read()
                            tensor = preprocess_image(image_bytes).to(self._device)
                            with torch.no_grad():
                                logits, feature_map = self._model(tensor, return_features=True)
                            result = decode_logits(logits)
                            row = {"filename": fname, **result}
                            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                            row["heatmap_url"] = compute_and_encode_cam(
                                feature_map, self._model.classifier[1].weight,
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
                            predictions.append(row)
                        except Exception as exc:
                            predictions.append({"filename": fname, "error": str(exc)})
                finally:
                    if temp_dir:
                        shutil.rmtree(temp_dir, ignore_errors=True)

            self._record()
            return PredictBatchResponse(
                model_id=MODEL_ID, predictions=predictions, output_path=None,
            )
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)
                self._model, self._device = saved_model, saved_device

    def train(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self, *, data_path: str, mlflow_run_id: str,
    ) -> TrainResponse:
        """Fine-tune a fresh clone of the served EfficientNet-B0 on user-labeled thermal images.

        Follows the original recipe exactly (manifest.training in inbox/a04/manifest.yaml):
        Adam(lr=1e-4, weight_decay=1e-4) + Focal Loss(alpha=0.25, gamma=2.0) +
        ReduceLROnPlateau(factor=0.5, patience=5) + early stopping (patience=15, max 100 epochs).
        The retrained checkpoint is persisted to MLflow only — the fixed base artifact served by
        model_loader.py is never overwritten (predict with this run's mlflow_run_id to use it).
        """
        self._require_loaded()
        tracker = BaseMLflowTracker(mlflow_run_id)
        tracker.log_params({
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
        })

        t0 = time.perf_counter()
        tmp_dir = tempfile.mkdtemp(prefix="ml4_train_")
        try:
            with local_file_path(data_path) as local_path:
                if os.path.isfile(local_path) and local_path.lower().endswith(".zip"):
                    with zipfile.ZipFile(local_path, "r") as zf:
                        zf.extractall(tmp_dir)
                    entries = os.listdir(tmp_dir)
                    data_root = (
                        os.path.join(tmp_dir, entries[0])
                        if len(entries) == 1 and os.path.isdir(os.path.join(tmp_dir, entries[0]))
                        else tmp_dir
                    )
                else:
                    data_root = local_path

                csv_path = os.path.join(data_root, "ID_Labels.csv")
                if not os.path.exists(csv_path):
                    raise ValueError(
                        "data_path debe contener 'ID_Labels.csv' con columnas 'ID' y 'label' "
                        "(0=Healthy, 1=SCM), igual que el dataset TIDS original."
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

            if len(paths) < MIN_TRAIN_SAMPLES:
                raise ValueError(
                    f"Se requieren al menos {MIN_TRAIN_SAMPLES} imágenes etiquetadas para "
                    f"reentrenar; se encontraron {len(paths)}."
                )

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
                model.train()
                for imgs, lbls in train_loader:
                    imgs, lbls = imgs.to(self._device), lbls.to(self._device)
                    optimizer.zero_grad()
                    loss = criterion(model(imgs), lbls)
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_true, val_pred, val_loss_total = [], [], 0.0
                with torch.no_grad():
                    for imgs, lbls in val_loader:
                        imgs, lbls = imgs.to(self._device), lbls.to(self._device)
                        logits = model(imgs)
                        val_loss_total += criterion(logits, lbls).item()
                        val_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
                        val_true.extend(lbls.cpu().tolist())
                val_loss = val_loss_total / max(len(val_loader), 1)
                scheduler.step(val_loss)
                val_f1 = f1_score(val_true, val_pred, average="binary", zero_division=0)
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

            val_true, val_pred, val_prob = [], [], []
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs = imgs.to(self._device)
                    probs = torch.softmax(model(imgs), dim=1)
                    val_pred.extend(torch.argmax(probs, dim=1).cpu().tolist())
                    val_true.extend(lbls.tolist())
                    val_prob.extend(probs[:, 1].cpu().tolist())

            accuracy = accuracy_score(val_true, val_pred)
            precision = precision_score(val_true, val_pred, average="binary", zero_division=0)
            recall = recall_score(val_true, val_pred, average="binary", zero_division=0)
            f1 = f1_score(val_true, val_pred, average="binary", zero_division=0)
            auc = roc_auc_score(val_true, val_prob) if len(set(val_true)) > 1 else None

            tracker.log_metrics({
                "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                "auc": auc if auc is not None else 0.0,
                "n_train": len(train_paths), "n_val": len(val_paths),
            })

            # Persisted ONLY to this MLflow run — the fixed base artifact filename used by
            # model_loader.py is never touched, so predict/stats without mlflow_run_id keeps
            # serving the original AI-team checkpoint.
            upload_warning = None
            mlflow_tmp = tempfile.mkdtemp(prefix="ml4_mlflow_upload_")
            try:
                ckpt_path = os.path.join(mlflow_tmp, MODEL_FILENAME)
                torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
                tracker.upload_artifacts(mlflow_tmp, artifact_path="model")
            except Exception as exc:
                logger.error("ml4 train(): MLflow artifact upload failed: %s", exc)
                upload_warning = f"El modelo reentrenado no se ha podido guardar en MLflow: {exc}"
            finally:
                shutil.rmtree(mlflow_tmp, ignore_errors=True)

            return TrainResponse(
                detail="Reentrenamiento completado",
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                auc=auc,
                n_train=len(train_paths),
                n_val=len(val_paths),
                training_time_s=round(time.perf_counter() - t0, 2),
                mlflow_run_id=mlflow_run_id,
                upload_warning=upload_warning,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

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
            runtime_stats=RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=None),
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
