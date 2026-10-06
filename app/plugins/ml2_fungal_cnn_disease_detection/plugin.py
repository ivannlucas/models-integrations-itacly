"""Plugin for fungal leaf-disease detection on grapevine leaves using a custom CNN (LeafCNN).

Serves a single-task image classifier with five disease/health classes. ``train()`` fine-
tunes-from-scratch a fresh LeafCNN on a user-supplied ZIP (per-class subfolders), exactly
replicating the delivered training procedure (inbox/a02/codigo/.../src/training/train.py) —
see ``inbox/a02/manifest.yaml#training`` for the full traceability of every hyperparameter.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import InvalidImageError, ModelNotLoadedError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.plugins.ml2_fungal_cnn_disease_detection.constants import (
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    MODEL_FILENAME,
    MODEL_ID,
    TRAIN_BATCH_SIZE,
    TRAIN_LR,
    TRAIN_MAX_EPOCHS,
    TRAIN_PATIENCE,
    TRAIN_SCHEDULER_FACTOR,
    TRAIN_SCHEDULER_PATIENCE,
    TRAIN_SPLIT_SEED,
    TRAIN_VAL_SPLIT,
    TRAIN_WEIGHT_DECAY,
)
from app.plugins.ml2_fungal_cnn_disease_detection.mlflow_utils import (
    download_user_model_from_mlflow,
)
from app.plugins.ml2_fungal_cnn_disease_detection.model_loader import (
    create_model,
    load_model_bundle,
)
from app.plugins.ml2_fungal_cnn_disease_detection.postprocessing import (
    build_batch_response,
    build_inline_response,
    compute_and_encode_cam,
)
from app.plugins.ml2_fungal_cnn_disease_detection.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml2_fungal_cnn_disease_detection.preprocessing import (
    build_train_transform,
    image_base64_to_tensor,
    image_path_to_tensor_and_image,
)
from app.plugins.ml2_fungal_cnn_disease_detection.train_dto import TrainResponse

logger = logging.getLogger(__name__)


class Ml2FungalCnnDiseaseDetectionPlugin(ModelPluginPort):
    """LeafCNN-based plugin classifying grapevine leaf images into fungal-disease classes."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._bundle: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        """Load the LeafCNN checkpoint into memory."""
        self._bundle = load_model_bundle()
        logger.info("Ml2FungalCnnDiseaseDetectionPlugin loaded: %s", self._bundle["model_id"])

    def is_loaded(self) -> bool:
        """Return True if the model bundle is loaded and ready for inference."""
        return self._bundle is not None

    def _require_bundle(self) -> dict:
        """Return the loaded bundle or raise :class:`ModelNotLoadedError`."""
        if self._bundle is None:
            raise ModelNotLoadedError("El modelo no está cargado.")
        return self._bundle

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Classify a single base64 image and return the typed inline response."""
        user_temp_dir: str | None = None
        saved_bundle = self._bundle
        if mlflow_run_id:
            logger.info(
                "predict_inline — using user-trained model from MLflow run_id=%s", mlflow_run_id
            )
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._bundle, user_temp_dir = loaded
        try:
            bundle = self._require_bundle()
            tensor = image_base64_to_tensor(
                features["image_base64"], image_size=bundle["image_size"]
            ).to(bundle["device"])

            bundle["model"].eval()
            with torch.no_grad():
                logits = bundle["model"](tensor)

            self._predict_count += 1
            self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
            logger.info(
                "predict_inline done — count=%d mlflow=%s", self._predict_count, bool(mlflow_run_id)
            )

            return PredictInlineResponse(**build_inline_response(
                logits,
                classes=bundle["classes"],
                model_id=bundle["model_id"],
            ))
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)
                self._bundle = saved_bundle

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Classify every image inside a ZIP (local path or ``s3://`` URI)."""
        user_temp_dir: str | None = None
        saved_bundle = self._bundle
        if mlflow_run_id:
            logger.info(
                "predict_batch — using user-trained model from MLflow run_id=%s", mlflow_run_id
            )
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._bundle, user_temp_dir = loaded
        try:
            bundle = self._require_bundle()
            model = bundle["model"]
            device: torch.device = bundle["device"]
            image_size: int = bundle["image_size"]
            predictions: list[dict] = []

            tmp_zip: str | None = None
            local_data_path = data_path
            if data_path.startswith("s3://"):
                tmp_zip = self._download_zip_from_s3(data_path)
                local_data_path = tmp_zip

            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    with zipfile.ZipFile(local_data_path, "r") as zf:
                        zf.extractall(tmp_dir)

                    image_paths = sorted(
                        p for p in Path(tmp_dir).rglob("*")
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                    )

                    model.eval()
                    for image_path in image_paths:
                        try:
                            tensor, image = image_path_to_tensor_and_image(image_path, image_size=image_size)
                            tensor = tensor.to(device)
                            with torch.no_grad():
                                logits, feature_map = model(tensor, return_features=True)
                            result = build_inline_response(
                                logits,
                                classes=bundle["classes"],
                                model_id=bundle["model_id"],
                            )
                            result["filename"] = image_path.name
                            class_idx = bundle["classes"].index(result["prediction"])
                            result["heatmap_url"] = compute_and_encode_cam(
                                feature_map, model.classifier.weight, class_idx, image
                            )
                            predictions.append(result)
                        except InvalidImageError as exc:
                            predictions.append({"filename": image_path.name, "error": str(exc)})
                        except Exception as exc:
                            logger.warning("Unexpected error processing %s: %s", image_path.name, exc)
                            predictions.append({"filename": image_path.name, "error": str(exc)})
            finally:
                if tmp_zip and os.path.exists(tmp_zip):
                    os.unlink(tmp_zip)

            self._predict_count += 1
            self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
            logger.info(
                "predict_batch done — %d predictions count=%d mlflow=%s",
                len(predictions), self._predict_count, bool(mlflow_run_id),
            )

            return PredictBatchResponse(
                **build_batch_response(predictions, model_id=bundle["model_id"])
            )
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)
                self._bundle = saved_bundle

    @staticmethod
    def _download_zip_from_s3(s3_uri: str) -> str:
        """Download a ZIP from an ``s3://bucket/key`` URI to a temp file and return its path."""
        import boto3
        from botocore.client import Config as BotoConfig

        without_prefix = s3_uri[5:]
        bucket, _, s3_key = without_prefix.partition("/")
        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ.get("CUSTOM_S3_ENDPOINT"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            config=BotoConfig(signature_version="s3v4"),
            region_name=os.environ.get("CUSTOM_REGION", "us-east-1"),
        )
        fd, tmp_zip = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        logger.info("Downloading batch data from s3://%s/%s", bucket, s3_key)
        s3.download_file(bucket, s3_key, tmp_zip)
        return tmp_zip

    def train(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self, *, data_path: str, mlflow_run_id: str,
    ) -> TrainResponse:
        """Train a fresh LeafCNN from scratch on a user-supplied ZIP of per-class folders.

        Replicates ``src/training/train.py:run_training`` from the delivered repository:
        same architecture/init (``model_loader.create_model``), same optimizer/scheduler/loss
        and the same max-epochs/early-stopping-patience (see ``constants.py`` — all copied
        verbatim from ``config/config.py``). The retrained checkpoint is uploaded ONLY to the
        given MLflow run's ``model`` artifact path — it never touches the served S3/local base
        artifact (``leafcnn_best.pth``); predicting with this ``mlflow_run_id`` afterwards is
        what serves it (see ``mlflow_utils.download_user_model_from_mlflow``).
        """
        # pylint: disable=import-outside-toplevel
        import random

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
        # pylint: enable=import-outside-toplevel

        from app.plugins.ml2_fungal_cnn_disease_detection.model_loader import (  # pylint: disable=import-outside-toplevel
            _safe_device,
        )
        from app.plugins.ml2_fungal_cnn_disease_detection.preprocessing import (  # pylint: disable=import-outside-toplevel
            build_eval_transform,
        )

        tracker = BaseMLflowTracker(mlflow_run_id)
        tracker.log_params({
            "optimizer": "AdamW",
            "lr": TRAIN_LR,
            "weight_decay": TRAIN_WEIGHT_DECAY,
            "batch_size": TRAIN_BATCH_SIZE,
            "max_epochs": TRAIN_MAX_EPOCHS,
            "patience": TRAIN_PATIENCE,
            "scheduler": (
                f"ReduceLROnPlateau(factor={TRAIN_SCHEDULER_FACTOR}, "
                f"patience={TRAIN_SCHEDULER_PATIENCE})"
            ),
            "image_size": IMAGE_SIZE,
            "loss": "CrossEntropyLoss",
        })

        tmp_zip: str | None = None
        local_data_path = data_path
        if data_path.startswith("s3://"):
            tmp_zip = self._download_zip_from_s3(data_path)
            local_data_path = tmp_zip

        extract_dir = tempfile.mkdtemp(prefix="ml2_train_")
        mlflow_tmp: str | None = None
        upload_warning: str | None = None
        try:
            with zipfile.ZipFile(local_data_path, "r") as zf:
                zf.extractall(extract_dir)

            root = Path(extract_dir)
            entries = [p for p in root.iterdir() if p.is_dir()]
            data_root = entries[0] if len(entries) == 1 else root

            class_dirs = sorted(
                (p for p in data_root.iterdir() if p.is_dir()), key=lambda p: p.name
            )
            if len(class_dirs) < 2:
                raise ValueError(
                    "El ZIP debe contener al menos 2 subcarpetas de clase "
                    "(p.ej. black_rot/, downy_mildew/, healthy/, powdery_mildew/, "
                    "trunk_disease/) con imágenes JPG/PNG/BMP dentro."
                )
            classes = [p.name for p in class_dirs]
            class_to_idx = {cls: i for i, cls in enumerate(classes)}

            samples_by_class: dict[str, list[str]] = {}
            for class_dir in class_dirs:
                images = sorted(
                    str(p) for p in class_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )
                if not images:
                    raise ValueError(f"La clase '{class_dir.name}' no tiene ninguna imagen válida.")
                samples_by_class[class_dir.name] = images

            # Fixed-seed stratified 80/20 split per class (see constants.TRAIN_VAL_SPLIT) —
            # every class contributes to both train and val whenever it has >=2 images.
            rng = random.Random(TRAIN_SPLIT_SEED)
            train_samples: list[tuple[str, str]] = []
            val_samples: list[tuple[str, str]] = []
            for cls, paths in samples_by_class.items():
                shuffled = paths[:]
                rng.shuffle(shuffled)
                if len(shuffled) >= 2:
                    n_val = max(1, round(len(shuffled) * TRAIN_VAL_SPLIT))
                    n_val = min(n_val, len(shuffled) - 1)
                else:
                    n_val = 0
                val_samples += [(p, cls) for p in shuffled[:n_val]]
                train_samples += [(p, cls) for p in shuffled[n_val:]]

            if not val_samples:
                raise ValueError(
                    "No hay suficientes imágenes para separar un conjunto de validación "
                    "(se necesitan >=2 imágenes en alguna clase)."
                )

            train_tfm = build_train_transform(IMAGE_SIZE)
            eval_tfm = build_eval_transform(IMAGE_SIZE)

            class _LeafDataset(Dataset):  # pylint: disable=too-few-public-methods
                def __init__(self, items: list[tuple[str, str]], transform, training: bool):
                    self.items = items
                    self.transform = transform
                    self.training = training

                def __len__(self) -> int:
                    return len(self.items)

                def __getitem__(self, idx: int):
                    from PIL import Image  # pylint: disable=import-outside-toplevel
                    path, label = self.items[idx]
                    image = Image.open(path).convert("RGB")
                    return self.transform(image), class_to_idx[label]

            device = _safe_device()
            train_loader = DataLoader(
                _LeafDataset(train_samples, train_tfm, training=True),
                batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0,
            )
            val_loader = DataLoader(
                _LeafDataset(val_samples, eval_tfm, training=False),
                batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=0,
            )

            model = create_model(num_classes=len(classes), device=device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=TRAIN_SCHEDULER_FACTOR,
                patience=TRAIN_SCHEDULER_PATIENCE,
            )

            best_val_loss = float("inf")
            best_state = model.state_dict()
            epochs_no_improve = 0
            epochs_run = 0

            for epoch in range(TRAIN_MAX_EPOCHS):
                model.train()
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_running_loss = 0.0
                val_total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_running_loss += loss.item() * images.size(0)
                        val_total += images.size(0)
                val_loss = val_running_loss / max(val_total, 1)
                scheduler.step(val_loss)
                epochs_run = epoch + 1

                if tracker.is_connected():
                    tracker.log_metrics({"val_loss": val_loss}, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= TRAIN_PATIENCE:
                    logger.info("Early stopping en epoch %d (ml2 train)", epochs_run)
                    break

            model.load_state_dict(best_state)
            model.eval()

            all_preds: list[int] = []
            all_labels: list[int] = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1).cpu().tolist()
                    all_preds.extend(preds)
                    all_labels.extend(labels.tolist())

            accuracy = float(accuracy_score(all_labels, all_preds))
            precision = float(
                precision_score(all_labels, all_preds, average="weighted", zero_division=0)
            )
            recall = float(
                recall_score(all_labels, all_preds, average="weighted", zero_division=0)
            )
            f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))

            tracker.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_train": len(train_samples),
                "n_val": len(val_samples),
                "epochs_run": epochs_run,
            })

            try:
                mlflow_tmp = tempfile.mkdtemp(prefix="ml2_mlflow_")
                checkpoint = {
                    "model_state_dict": best_state,
                    "classes": classes,
                    "image_size": IMAGE_SIZE,
                    "model_id": MODEL_ID,
                }
                torch.save(checkpoint, os.path.join(mlflow_tmp, MODEL_FILENAME))
                tracker.upload_artifacts(mlflow_tmp, artifact_path="model")
            except Exception as exc:  # pragma: no cover - network/MLflow failure path
                logger.error("MLflow artifact upload failed: %s", exc)
                upload_warning = f"El modelo reentrenado no se ha podido guardar en MLflow: {exc}"

            logger.info(
                "train() done — accuracy=%.4f f1=%.4f epochs_run=%d classes=%s",
                accuracy, f1, epochs_run, classes,
            )

            return TrainResponse(
                detail="Entrenamiento completado",
                accuracy=round(accuracy, 4),
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                n_train=len(train_samples),
                n_val=len(val_samples),
                classes=classes,
                epochs_run=epochs_run,
                upload_warning=upload_warning,
                mlflow_run_id=mlflow_run_id,
            )
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)
            if mlflow_tmp:
                shutil.rmtree(mlflow_tmp, ignore_errors=True)
            if tmp_zip and os.path.exists(tmp_zip):
                os.unlink(tmp_zip)

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        user_temp_dir: str | None = None
        saved_bundle = self._bundle
        mlflow_metrics: dict | None = None
        if mlflow_run_id:
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._bundle, user_temp_dir = loaded
            tracker = BaseMLflowTracker(mlflow_run_id)
            mlflow_metrics = {"params": tracker.get_params(), "metrics": tracker.get_metrics()}

        try:
            model_id_v = self._bundle["model_id"] if self._bundle is not None else MODEL_ID
            metrics: dict = {}
            if mlflow_metrics is not None:
                metrics["mlflow"] = mlflow_metrics

            return StatsResponse(
                model_name=model_id_v,
                version="1.0.0",
                description=(
                    "Detección de enfermedades fúngicas en hojas de vid mediante una "
                    "CNN personalizada (LeafCNN)."
                ),
                task_type="image-classification",
                framework="pytorch",
                inputs=[
                    InputField(
                        name="image",
                        type="file",
                        format=["jpg", "jpeg", "png", "bmp"],
                        description="Imagen de hoja de vid (base64 para inline, ZIP para batch)",
                    ),
                ],
                outputs=[
                    OutputField(
                        name="prediction",
                        type="str",
                        description=f"Clase predicha (una de: {', '.join(CLASS_NAMES)})",
                    ),
                    OutputField(
                        name="confidence",
                        type="float",
                        description="Probabilidad softmax de la clase ganadora [0, 1]",
                    ),
                    OutputField(
                        name="probabilities",
                        type="dict",
                        description="Probabilidad softmax por clase",
                    ),
                    OutputField(
                        name="heatmap_url",
                        type="str",
                        description=(
                            "Mapa de activación de clase (CAM) superpuesto en base64 JPEG "
                            "data URI (solo en predict_batch)"
                        ),
                    ),
                ],
                metrics=metrics,
                runtime_stats=RuntimeStats(
                    total_predictions=self._predict_count,
                    avg_latency_ms=None,
                ),
            )
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)
                self._bundle = saved_bundle
