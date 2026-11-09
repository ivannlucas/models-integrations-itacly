"""Plugin for fungal leaf-disease detection on grapevine leaves using a custom CNN (LeafCNN).

Serves a single-task image classifier with five disease/health classes. ``train()`` fine-
tunes-from-scratch a fresh LeafCNN on a user-supplied ZIP (per-class subfolders), exactly
replicating the delivered training procedure (inbox/a02/codigo/.../src/training/train.py) —
see ``inbox/a02/manifest.yaml#training`` for the full traceability of every hyperparameter.
"""
from __future__ import annotations

import logging
import os
import random
import tempfile
from pathlib import Path
from typing import NamedTuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.application.dto.stats_dto import InputField, OutputField, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import InvalidImageError, ModelNotLoadedError
from app.domain.services.mlflow_artifacts import (
    mlflow_run_summary,
    removing_dir,
    require_train_tracker,
    upload_model_to_mlflow,
)
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.infrastructure.archive import safe_extract_zip
from app.infrastructure.artifact_store import local_file_path
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
    _safe_device,
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
    build_eval_transform,
    build_train_transform,
    image_base64_to_tensor,
    image_path_to_tensor_and_image,
)
from app.plugins.ml2_fungal_cnn_disease_detection.train_dto import TrainResponse

logger = logging.getLogger(__name__)

# Hyperparameters logged to the retrain's MLflow run (see constants.py for their origin).
_TRAIN_PARAMS = {
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
}


class _LeafDataset(Dataset):
    """(image path, class name) samples read as RGB PIL images and transformed."""

    def __init__(self, items: list[tuple[str, str]], transform, class_to_idx: dict[str, int]):
        self.items = items
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), self.class_to_idx[label]


def _collect_class_samples(extract_dir: str) -> dict[str, list[str]]:
    """Map each class subfolder of the extracted ZIP (sorted by name) to its sorted images."""
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

    samples_by_class: dict[str, list[str]] = {}
    for class_dir in class_dirs:
        images = sorted(
            str(p) for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"La clase '{class_dir.name}' no tiene ninguna imagen válida.")
        samples_by_class[class_dir.name] = images
    return samples_by_class


def _split_samples(
    samples_by_class: dict[str, list[str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Fixed-seed stratified 80/20 split per class (see constants.TRAIN_VAL_SPLIT).

    Every class contributes to both train and val whenever it has >=2 images.
    """
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
    return train_samples, val_samples


def _train_one_epoch(model, train_loader, criterion, optimizer, device) -> None:
    """One optimisation pass over the training split."""
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def _validation_loss(model, val_loader, criterion, device) -> float:
    """Mean validation loss of one epoch."""
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
    return val_running_loss / max(val_total, 1)


def _fit(model, loaders: tuple[DataLoader, DataLoader], device, tracker) -> tuple[dict, int]:
    """AdamW + ReduceLROnPlateau training with early stopping on the validation loss.

    Returns (best_state, epochs_run) — same loop as the delivered ``run_training``.
    """
    train_loader, val_loader = loaders
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
        _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _validation_loss(model, val_loader, criterion, device)
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
    return best_state, epochs_run


def _validation_scores(model, val_loader, device) -> dict[str, float]:
    """Weighted accuracy/precision/recall/F1 of ``model`` on the validation split."""
    # pylint: disable-next=import-outside-toplevel  # sklearn is loaded lazily in plugins
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(
            precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
    }


class _TrainOutcome(NamedTuple):
    """What train() logs, uploads and reports after fitting a LeafCNN."""

    classes: list[str]
    n_train: int
    n_val: int
    best_state: dict
    epochs_run: int
    scores: dict[str, float]


def _train_on_folders(extract_dir: str, tracker) -> _TrainOutcome:
    """Split the extracted per-class folders, fit a fresh LeafCNN and score it on val."""
    samples_by_class = _collect_class_samples(extract_dir)
    classes = list(samples_by_class)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    train_samples, val_samples = _split_samples(samples_by_class)

    train_tfm = build_train_transform(IMAGE_SIZE)
    eval_tfm = build_eval_transform(IMAGE_SIZE)

    device = _safe_device()
    train_loader = DataLoader(
        _LeafDataset(train_samples, train_tfm, class_to_idx),
        batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        _LeafDataset(val_samples, eval_tfm, class_to_idx),
        batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=0,
    )

    model = create_model(num_classes=len(classes), device=device)
    best_state, epochs_run = _fit(model, (train_loader, val_loader), device, tracker)

    model.load_state_dict(best_state)
    model.eval()
    return _TrainOutcome(
        classes=classes,
        n_train=len(train_samples),
        n_val=len(val_samples),
        best_state=best_state,
        epochs_run=epochs_run,
        scores=_validation_scores(model, val_loader, device),
    )


def _predict_image(bundle: dict, image_path: Path) -> dict:
    """Classify one batch image and attach its CAM heatmap; errors become the row's ``error``."""
    model = bundle["model"]
    try:
        tensor, image = image_path_to_tensor_and_image(
            image_path, image_size=bundle["image_size"]
        )
        tensor = tensor.to(bundle["device"])
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
        return result
    except InvalidImageError as exc:
        return {"filename": image_path.name, "error": str(exc)}
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # per-image error capture: one bad image never fails the batch
        logger.warning(
            "Unexpected error processing %s: %s", image_path.name, exc
        )
        return {"filename": image_path.name, "error": str(exc)}


class Ml2FungalCnnDiseaseDetectionPlugin(PluginRuntimeMixin, ModelPluginPort):
    """LeafCNN-based plugin classifying grapevine leaf images into fungal-disease classes."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin (runtime counters start at the mixin defaults)."""
        self._bundle: dict | None = None

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

    def _resolve_bundle(self, mlflow_run_id: str) -> tuple[dict, str | None]:
        """Return (bundle, temp_dir) to serve this request.

        The user's retrained bundle (if mlflow_run_id) is returned, never assigned to
        self._bundle: the plugin instance is shared by concurrent requests, so swapping it
        into self leaked a user's model to other requests — and, with two overlapping runs,
        left it in place as the "base" model. temp_dir is None for the base model.
        """
        if not mlflow_run_id:
            return self._require_bundle(), None
        return download_user_model_from_mlflow(mlflow_run_id)

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Classify a single base64 image and return the typed inline response."""
        if mlflow_run_id:
            logger.info(
                "predict_inline — using user-trained model from MLflow run_id=%s", mlflow_run_id
            )
        bundle, user_temp_dir = self._resolve_bundle(mlflow_run_id)
        with removing_dir(user_temp_dir):
            tensor = image_base64_to_tensor(
                features["image_base64"], image_size=bundle["image_size"]
            ).to(bundle["device"])

            bundle["model"].eval()
            with torch.no_grad():
                logits = bundle["model"](tensor)

            self._record()
            logger.info(
                "predict_inline done — count=%d mlflow=%s", self._predict_count, bool(mlflow_run_id)
            )

            return PredictInlineResponse(**build_inline_response(
                logits,
                classes=bundle["classes"],
                model_id=bundle["model_id"],
            ))

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Classify every image inside a ZIP (local path or ``s3://`` URI)."""
        if mlflow_run_id:
            logger.info(
                "predict_batch — using user-trained model from MLflow run_id=%s", mlflow_run_id
            )
        bundle, user_temp_dir = self._resolve_bundle(mlflow_run_id)
        with removing_dir(user_temp_dir):
            predictions: list[dict] = []
            with local_file_path(data_path) as local_data_path, \
                    tempfile.TemporaryDirectory() as tmp_dir:
                safe_extract_zip(local_data_path, tmp_dir)

                image_paths = sorted(
                    p for p in Path(tmp_dir).rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )

                bundle["model"].eval()
                for image_path in image_paths:
                    predictions.append(_predict_image(bundle, image_path))

            self._record()
            logger.info(
                "predict_batch done — %d predictions count=%d mlflow=%s",
                len(predictions), self._predict_count, bool(mlflow_run_id),
            )

            return PredictBatchResponse(
                **build_batch_response(predictions, model_id=bundle["model_id"])
            )

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Train a fresh LeafCNN from scratch on a user-supplied ZIP of per-class folders.

        Replicates ``src/training/train.py:run_training`` from the delivered repository:
        same architecture/init (``model_loader.create_model``), same optimizer/scheduler/loss
        and the same max-epochs/early-stopping-patience (see ``constants.py`` — all copied
        verbatim from ``config/config.py``). The retrained checkpoint is uploaded ONLY to the
        given MLflow run's ``model`` artifact path — it never touches the served S3/local base
        artifact (``leafcnn_best.pth``); predicting with this ``mlflow_run_id`` afterwards is
        what serves it (see ``mlflow_utils.download_user_model_from_mlflow``).
        """
        tracker = require_train_tracker(mlflow_run_id)
        tracker.log_params(_TRAIN_PARAMS)

        with local_file_path(data_path) as local_data_path:
            extract_dir = tempfile.mkdtemp(prefix="ml2_train_")
            with removing_dir(extract_dir):
                safe_extract_zip(local_data_path, extract_dir)
                outcome = _train_on_folders(extract_dir, tracker)
        classes, best_state, epochs_run, scores = (
            outcome.classes, outcome.best_state, outcome.epochs_run, outcome.scores
        )

        tracker.log_metrics({
            **scores,
            "n_train": outcome.n_train,
            "n_val": outcome.n_val,
            "epochs_run": epochs_run,
        })

        def _write(model_dir: str) -> None:
            checkpoint = {
                "model_state_dict": best_state,
                "classes": classes,
                "image_size": IMAGE_SIZE,
                "model_id": MODEL_ID,
            }
            torch.save(checkpoint, os.path.join(model_dir, MODEL_FILENAME))

        upload_model_to_mlflow(tracker, _write, prefix="ml2_mlflow_")

        logger.info(
            "train() done — accuracy=%.4f f1=%.4f epochs_run=%d classes=%s",
            scores["accuracy"], scores["f1"], epochs_run, classes,
        )

        return TrainResponse(
            detail="Entrenamiento completado",
            accuracy=round(scores["accuracy"], 4),
            precision=round(scores["precision"], 4),
            recall=round(scores["recall"], 4),
            f1=round(scores["f1"], 4),
            n_train=outcome.n_train,
            n_val=outcome.n_val,
            classes=classes,
            epochs_run=epochs_run,
            upload_warning=None,
            mlflow_run_id=mlflow_run_id,
        )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        user_temp_dir: str | None = None
        bundle = self._bundle
        mlflow_metrics: dict | None = None
        if mlflow_run_id:
            bundle, user_temp_dir = download_user_model_from_mlflow(mlflow_run_id)

        with removing_dir(user_temp_dir):
            if mlflow_run_id:
                mlflow_metrics = mlflow_run_summary(mlflow_run_id)
            model_id_v = bundle["model_id"] if bundle is not None else MODEL_ID
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
                runtime_stats=self._runtime_stats(),
            )
