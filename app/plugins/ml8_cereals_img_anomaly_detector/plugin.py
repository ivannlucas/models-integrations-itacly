"""Ml8CerealsImgAnomalyDetectorPlugin — multitask CNN (sanitary category + cereal) for grain."""
from __future__ import annotations

import gc
import logging
import os
import random
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models as _tv_models
from torchvision import transforms

from app.application.dto.stats_dto import InputField, OutputField, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.batch_rows import row_error_capture
from app.domain.services.exceptions import ModelNotLoadedError
from app.domain.services.mlflow_artifacts import (
    add_mlflow_run_to_stats,
    removing_dir,
    require_train_tracker,
    upload_model_to_mlflow,
)
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.infrastructure.archive import safe_extract_zip
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml8_cereals_img_anomaly_detector.constants import (
    CATEGORY_NAMES,
    CEREAL_NAMES,
    IMAGE_EXTENSIONS,
    MODEL_FILENAME,
    MODEL_ID,
)
from app.plugins.ml8_cereals_img_anomaly_detector.mlflow_utils import (
    download_user_model_from_mlflow,
)
from app.plugins.ml8_cereals_img_anomaly_detector.model_loader import (
    MultiTaskMobileNetV3Large,
    _safe_device,
    load_model_bundle,
)
from app.plugins.ml8_cereals_img_anomaly_detector.postprocessing import (
    build_batch_response,
    build_inline_response,
    encode_image_base64,
)
from app.plugins.ml8_cereals_img_anomaly_detector.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml8_cereals_img_anomaly_detector.preprocessing import (
    image_base64_to_tensor,
    image_path_to_tensor_and_image,
)
from app.plugins.ml8_cereals_img_anomaly_detector.train_dto import TrainResponse

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

_TRAIN_SEED = 42
_TRAIN_PARAMS = {
    "batch_size": 16,
    "fase1_lr": 0.001,
    "fase1_epochs": 10,
    "fase1_optimizer": "Adam",
    "fase2_lr": 0.00001,
    "fase2_epochs": 5,
    "fase2_optimizer": "Adam",
    "model": "MultiTaskMobileNetV3Large",
    "backbone_weights": "IMAGENET1K_V1",
    "accumulation_steps": 2,
    "seed": _TRAIN_SEED,
    "class_balancing": (
        "balance_hongos + class/cereal_weights (CrossEntropyLoss) + "
        "WeightedRandomSampler"
    ),
}
_IMAGENET_NORMALIZE = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# ── Training helpers ──────────────────────────────────────────────────────────

def _scan_split(data_root: Path, split_names: tuple) -> list:
    """Return (path, categoria, cereal) records for {cereal}/{split}/{categoria}/*.ext."""
    records = []
    for cereal in CEREAL_NAMES:
        for split in split_names:
            for categoria in CATEGORY_NAMES:
                cat_dir = data_root / cereal / split / categoria
                if not cat_dir.is_dir():
                    continue
                for p in cat_dir.iterdir():
                    if p.suffix.lower() in _SUPPORTED_EXTENSIONS:
                        records.append((str(p), categoria, cereal))
    return records


def _balance_hongos(records: list, max_samples_hongos: int | None, seed: int) -> list:
    """Replicates src/training/dataset.py:balance_hongos — undersamples the 'hongos'
    category in the TRAIN split only, down to max_samples_hongos (or, if None, to the
    average count of the other categories). Part of the real delivered training
    procedure (memoria sección 6.2: "esquema de balanceo en entrenamiento"), not optional."""
    counts = Counter(categoria for _, categoria, _ in records)
    current_hongos = counts.get("hongos", 0)
    if current_hongos == 0:
        return records

    if max_samples_hongos is not None:
        target = int(max_samples_hongos)
    else:
        others = [n for cat, n in counts.items() if cat != "hongos"]
        target = int(sum(others) / len(others)) if others else current_hongos

    if current_hongos <= target:
        return records

    rng = random.Random(seed)
    hongos_records = [r for r in records if r[1] == "hongos"]
    other_records = [r for r in records if r[1] != "hongos"]
    sampled_hongos = rng.sample(hongos_records, target)
    balanced = other_records + sampled_hongos
    rng.shuffle(balanced)
    logger.info("balance_hongos: %d -> %d muestras de 'hongos' en train", current_hongos, target)
    return balanced


def _load_mobilenetv3_backbone():
    """Replicates src/training/model.py:_load_backbone for mobilenet_v3_large — tries the
    real IMAGENET1K_V1 pretrained weights (as used in training/model.py and notebook
    04_MobileNetV3.ipynb), falling back to weights=None if the download is unreachable
    (e.g. sandboxed/offline environment)."""
    try:
        return _tv_models.mobilenet_v3_large(
            weights=_tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught  # offline -> weights=None
        logger.warning(
            "No se pudieron descargar los pesos preentrenados IMAGENET1K_V1 (%s). "
            "Continuando con inicialización aleatoria (weights=None).", exc,
        )
        return _tv_models.mobilenet_v3_large(weights=None)


class _TrainSetup(NamedTuple):
    """What every epoch of both training phases shares."""

    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    crit_cat: nn.Module
    crit_cer: nn.Module
    device: torch.device
    accumulation_steps: int = 1


def _train_epoch(setup: _TrainSetup, optimizer):
    """Replicates src/training/train.py:train_one_epoch — gradient accumulation included,
    since the delivered training procedure uses accumulation_steps=2 (config.yaml)."""
    model = setup.model
    model.train()
    total_loss = correct_cat = correct_cer = total = 0
    optimizer.zero_grad()
    batch_idx = -1
    for batch_idx, (imgs, lbls_cat, lbls_cer, _) in enumerate(setup.train_loader):
        imgs = imgs.to(setup.device)
        lbls_cat = lbls_cat.to(setup.device)
        lbls_cer = lbls_cer.to(setup.device)
        out_cat, out_cer = model(imgs)
        loss_full = setup.crit_cat(out_cat, lbls_cat) + setup.crit_cer(out_cer, lbls_cer)
        (loss_full / setup.accumulation_steps).backward()
        if (batch_idx + 1) % setup.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss_full.item()
        correct_cat += (out_cat.argmax(1) == lbls_cat).sum().item()
        correct_cer += (out_cer.argmax(1) == lbls_cer).sum().item()
        total += lbls_cat.size(0)
    if (batch_idx + 1) % setup.accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    n = max(len(setup.train_loader), 1)
    return (
        total_loss / n,
        100 * correct_cat / total if total else 0.0,
        100 * correct_cer / total if total else 0.0,
    )


def _validate_epoch(setup: _TrainSetup):
    """Mean val loss and category/cereal accuracy (%) of the current model."""
    model = setup.model
    model.eval()
    total_loss = correct_cat = correct_cer = total = 0
    with torch.no_grad():
        for imgs, lbls_cat, lbls_cer, _ in setup.val_loader:
            imgs = imgs.to(setup.device)
            lbls_cat = lbls_cat.to(setup.device)
            lbls_cer = lbls_cer.to(setup.device)
            out_cat, out_cer = model(imgs)
            total_loss += (
                setup.crit_cat(out_cat, lbls_cat) + setup.crit_cer(out_cer, lbls_cer)
            ).item()
            correct_cat += (out_cat.argmax(1) == lbls_cat).sum().item()
            correct_cer += (out_cer.argmax(1) == lbls_cer).sum().item()
            total += lbls_cat.size(0)
    n = max(len(setup.val_loader), 1)
    return (
        total_loss / n,
        100 * correct_cat / total if total else 0.0,
        100 * correct_cer / total if total else 0.0,
    )


def _run_phase(setup: _TrainSetup, optimizer, *, epochs: int, patience: int, phase_name: str):
    """One training phase with early stopping on val loss; restores the best weights."""
    model = setup.model
    best_loss, no_improve, best_state = float("inf"), 0, None
    history: dict = {"val_acc_cat": [], "val_acc_cer": []}
    logger.info("Iniciando %s", phase_name)
    for epoch in range(epochs):
        tr_loss, tr_acc_cat, tr_acc_cer = _train_epoch(setup, optimizer)
        val = _validate_epoch(setup)
        history["val_acc_cat"].append(val[1])
        history["val_acc_cer"].append(val[2])
        logger.info(
            "  Epoch %d/%d | tr=%.4f val=%.4f cat=%.1f%%/%.1f%% cer=%.1f%%/%.1f%%",
            epoch + 1, epochs, tr_loss, val[0],
            tr_acc_cat, val[1], tr_acc_cer, val[2],
        )
        if val[0] < best_loss:
            best_loss, no_improve = val[0], 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("  Early stopping en epoch %d", epoch + 1)
                break
    if best_state:
        model.load_state_dict(best_state)
    return history


class _CerealDataset(Dataset):
    """(image tensor, category idx, cereal idx, path) per record; unreadable image → zeros."""

    def __init__(self, records, tfm):
        self.records = records
        self.tfm = tfm

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, categoria, cereal = self.records[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.tfm(img)
        except Exception:  # pylint: disable=broad-exception-caught  # unreadable image -> zeros
            img = torch.zeros(3, 224, 224)
        return img, CATEGORY_NAMES.index(categoria), CEREAL_NAMES.index(cereal), img_path


def _load_records(tmp_dir: str) -> tuple[list, list]:
    """Train/val records of the extracted ZIP (auto-split 80/20 when it has no validation)."""
    entries = list(Path(tmp_dir).iterdir())
    data_root = entries[0] if len(entries) == 1 and entries[0].is_dir() else Path(tmp_dir)

    train_records = _scan_split(data_root, ("train",))
    val_records = _scan_split(data_root, ("validation", "val"))

    if not train_records:
        raise ValueError(
            "No se encontraron imágenes de entrenamiento. "
            "Estructura esperada: {cereal}/train/{categoria}/*.jpg"
        )
    if not val_records:
        random.shuffle(train_records)
        split_idx = int(len(train_records) * 0.8)
        val_records = train_records[split_idx:]
        train_records = train_records[:split_idx]
        logger.info(
            "Sin validación — auto-split 80/20: %d train, %d val",
            len(train_records), len(val_records),
        )
    return train_records, val_records


def _inverse_frequency_weights(counts: Counter, names, total_train: int) -> torch.Tensor:
    """class/cereal weights for CrossEntropyLoss — igual que src/main.py:run_training.

    max(..., 1): a diferencia del notebook original (que asume las 4 categorías/cereales
    siempre presentes en train), aquí el ZIP lo sube un usuario vía API y una categoría/cereal
    podría venir ausente — evitamos un ZeroDivisionError en lugar de que /train devuelva 500
    por un dataset de usuario incompleto.
    """
    weights = torch.tensor(
        [total_train / max(counts.get(c, 0), 1) for c in names],
        dtype=torch.float32,
    )
    return weights / weights.sum() * len(names)


def _build_loaders(train_records: list, val_records: list) -> tuple:
    """``(train_loader, val_loader, class_weights, cereal_weights)`` of the real balancing scheme.

    Esquema de balanceo real (memoria 6.2 / src/main.py:run_training), SOLO sobre train:
    balance_hongos ya aplicado, class/cereal weights para la loss y WeightedRandomSampler
    (balanceo por lote, basado en categoría) en lugar del shuffle=True plano.
    """
    cat_counts = Counter(categoria for _, categoria, _ in train_records)
    cer_counts = Counter(cereal for _, _, cereal in train_records)
    total_train = len(train_records)
    class_weights = _inverse_frequency_weights(cat_counts, CATEGORY_NAMES, total_train)
    cereal_weights = _inverse_frequency_weights(cer_counts, CEREAL_NAMES, total_train)

    sample_weights = [
        1.0 / max(cat_counts.get(categoria, 0), 1) for _, categoria, _ in train_records
    ]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(*_IMAGENET_NORMALIZE),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(*_IMAGENET_NORMALIZE),
    ])
    train_loader = DataLoader(
        _CerealDataset(train_records, train_tfm),
        batch_size=16, sampler=sampler, num_workers=0,
    )
    val_loader = DataLoader(
        _CerealDataset(val_records, val_tfm),
        batch_size=16, shuffle=False, num_workers=0,
    )
    return train_loader, val_loader, class_weights, cereal_weights


def _build_train_model(device: torch.device) -> nn.Module:
    """MobileNetV3-Large multitask with IMAGENET1K_V1 backbone, only the classifier trainable.

    Mismo backbone que src/training/model.py:_load_backbone y notebook 04_MobileNetV3.ipynb.
    """
    os.environ.setdefault("TORCH_HOME", "/tmp/.torch")
    base = _load_mobilenetv3_backbone()
    for param in base.parameters():
        param.requires_grad = False
    for param in base.classifier.parameters():
        param.requires_grad = True
    model = MultiTaskMobileNetV3Large(
        num_classes=len(CATEGORY_NAMES),
        num_cereals=len(CEREAL_NAMES),
        base_model=base,
    )
    model.to(device)
    return model


def _make_setup(train_records: list, val_records: list, device: torch.device) -> _TrainSetup:
    """Loaders, weighted losses and the model to train (gradient accumulation of 2 batches)."""
    train_loader, val_loader, class_weights, cereal_weights = _build_loaders(
        train_records, val_records
    )
    logger.info("Dataset: %d train, %d val", len(train_records), len(val_records))
    model = _build_train_model(device)
    return _TrainSetup(
        model, train_loader, val_loader,
        nn.CrossEntropyLoss(weight=class_weights.to(device)),
        nn.CrossEntropyLoss(weight=cereal_weights.to(device)),
        device, accumulation_steps=2,
    )


def _log_phase_history(tracker, prefix: str, history: dict) -> None:
    """Log the per-epoch val accuracies of one phase to MLflow."""
    for ep in range(len(history["val_acc_cat"])):
        tracker.log_metrics({
            f"{prefix}_val_acc_cat": history["val_acc_cat"][ep],
            f"{prefix}_val_acc_cer": history["val_acc_cer"][ep],
        }, step=ep)


def _best(values_h2: list, values_h1: list) -> float:
    """Best val accuracy of phase 2 (phase 1 if phase 2 ran no epoch), rounded to 0.1."""
    combined = values_h2 or values_h1
    return round(max(combined), 1) if combined else 0.0


def _fit(setup: _TrainSetup, tracker) -> tuple[dict, dict, float, float]:
    """Fase 1 (transfer learning, neck + heads) and Fase 2 (fine-tune completo).

    Returns ``(history 1, history 2, seconds 1, seconds 2)``.
    """
    model = setup.model
    t0 = time.perf_counter()
    opt1 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
    h1 = _run_phase(setup, opt1, epochs=10, patience=3, phase_name="Fase 1: Transfer Learning")
    t1 = time.perf_counter() - t0
    _log_phase_history(tracker, "fase1", h1)
    gc.collect()

    for param in model.parameters():
        param.requires_grad = True
    opt2 = torch.optim.Adam(model.parameters(), lr=0.00001)
    t0 = time.perf_counter()
    h2 = _run_phase(setup, opt2, epochs=5, patience=3, phase_name="Fase 2: Fine-tuning")
    t2 = time.perf_counter() - t0
    _log_phase_history(tracker, "fase2", h2)
    gc.collect()
    return h1, h2, t1, t2


def _checkpoint(model: nn.Module) -> dict:
    """The checkpoint layout of the delivered training (same keys load_model_bundle reads)."""
    class_to_idx = {c: i for i, c in enumerate(CATEGORY_NAMES)}
    cereal_to_idx = {c: i for i, c in enumerate(CEREAL_NAMES)}
    return {
        "model_name": "mobilenet_v3_large",
        "model_state_dict": model.state_dict(),
        "num_classes": len(CATEGORY_NAMES),
        "num_cereales": len(CEREAL_NAMES),
        "class_to_idx": class_to_idx,
        "cereal_to_idx": cereal_to_idx,
        "idx_to_class": {v: k for k, v in class_to_idx.items()},
        "idx_to_cereal": {v: k for k, v in cereal_to_idx.items()},
    }


def _predict_image(bundle: dict, image_path: Path) -> dict:
    """Batch row of one image (the caller turns errors into ``error`` rows)."""
    tensor, image = image_path_to_tensor_and_image(
        image_path, image_size=bundle["image_size"]
    )
    tensor = tensor.to(bundle["device"])
    with torch.no_grad():
        logits_cat, logits_cer = bundle["model"](tensor)
    result = build_inline_response(
        logits_cat,
        logits_cer,
        idx_to_class=bundle["idx_to_class"],
        idx_to_cereal=bundle["idx_to_cereal"],
        model_id=bundle["model_id"],
    )
    result["filename"] = image_path.name
    result["annotated_image"] = encode_image_base64(image)
    return result


# ── Plugin ────────────────────────────────────────────────────────────────────

class Ml8CerealsImgAnomalyDetectorPlugin(PluginRuntimeMixin, ModelPluginPort):
    """ModelPluginPort implementation for ml8-cereals-img-anomaly-detector."""

    def __init__(self) -> None:
        self._bundle: dict | None = None

    def load(self) -> None:
        self._bundle = load_model_bundle()
        logger.info("Ml8CerealsImgAnomalyDetectorPlugin loaded: %s", self._bundle["model_id"])

    def is_loaded(self) -> bool:
        return self._bundle is not None

    def _require_bundle(self) -> dict:
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
                logits_cat, logits_cer = bundle["model"](tensor)

            self._record()
            logger.info(
                "predict_inline done — count=%d mlflow=%s", self._predict_count, bool(mlflow_run_id)
            )

            return PredictInlineResponse(**build_inline_response(
                logits_cat,
                logits_cer,
                idx_to_class=bundle["idx_to_class"],
                idx_to_cereal=bundle["idx_to_cereal"],
                model_id=bundle["model_id"],
            ))

    def predict_batch(
        self, *, data_path: str, mlflow_run_id: str = ""
    ) -> PredictBatchResponse:
        if mlflow_run_id:
            logger.info(
                "predict_batch — using user-trained model from MLflow run_id=%s", mlflow_run_id
            )
        bundle, user_temp_dir = self._resolve_bundle(mlflow_run_id)
        with removing_dir(user_temp_dir):
            predictions: list[dict] = []
            with local_file_path(data_path) as local_data_path:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    safe_extract_zip(local_data_path, tmp_dir)

                    image_paths = sorted(
                        p for p in Path(tmp_dir).rglob("*")
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                    )

                    bundle["model"].eval()
                    for image_path in image_paths:
                        # one bad image (unreadable, model error…) never fails the batch
                        with row_error_capture(
                            predictions, {"filename": image_path.name}, what=image_path.name
                        ):
                            predictions.append(_predict_image(bundle, image_path))

            self._record()
            logger.info("predict_batch done — %d predictions count=%d mlflow=%s",
                        len(predictions), self._predict_count, bool(mlflow_run_id))

            return PredictBatchResponse(
                **build_batch_response(predictions, model_id=bundle["model_id"])
            )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        arch = self._bundle["arch"] if self._bundle is not None else "unknown"
        model_id_v = self._bundle["model_id"] if self._bundle is not None else MODEL_ID

        base = StatsResponse(
            model_name=model_id_v,
            version="1.0.0",
            description=(
                "Clasificación multitarea de imágenes de cereales: predice la categoría "
                "(anomalía) y el tipo de cereal a partir de una imagen."
            ),
            task_type="image-classification",
            framework="pytorch",
            inputs=[
                InputField(
                    name="image",
                    type="file",
                    format=["jpg", "jpeg", "png", "bmp", "tif"],
                    description="Imagen de cereal (base64 para inline, ZIP de imágenes para batch)",
                ),
            ],
            outputs=[
                OutputField(
                    name="categoria", type="str", description="Categoría predicha (anomalía/estado)"
                ),
                OutputField(name="cereal", type="str", description="Tipo de cereal predicho"),
                OutputField(
                    name="confianza_categoria",
                    type="float",
                    description="Confianza de la categoría [0, 1]",
                ),
                OutputField(
                    name="confianza_cereal", type="float", description="Confianza del cereal [0, 1]"
                ),
                OutputField(
                    name="probabilidades_categoria",
                    type="dict",
                    description="Probabilidad por categoría",
                ),
                OutputField(
                    name="probabilidades_cereal", type="dict", description="Probabilidad por cereal"
                ),
                OutputField(
                    name="annotated_image",
                    type="str",
                    description="Imagen re-escalada en base64 JPEG (solo en predict_batch)",
                ),
            ],
            metrics={"arch": arch},
            runtime_stats=self._runtime_stats(),
        )
        return add_mlflow_run_to_stats(base, mlflow_run_id)

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        # The retrained checkpoint lives only in its own MLflow run (predict with that
        # mlflow_run_id); the served base model and its local artifacts are never replaced.
        tracker = require_train_tracker(mlflow_run_id)
        logger.info("Training with MLflow tracking, run_id=%s", mlflow_run_id)
        tracker.log_params(_TRAIN_PARAMS)

        # An s3:// object is always downloaded to a local file and opened as a ZIP.
        if not (data_path.startswith("s3://") or data_path.lower().endswith(".zip")):
            raise ValueError("data_path debe ser un fichero .zip")

        try:
            with local_file_path(data_path) as local_data_path:
                return self._train_from_zip(local_data_path, tracker, mlflow_run_id)
        finally:
            gc.collect()

    def _train_from_zip(self, zip_path: str, tracker, mlflow_run_id: str) -> TrainResponse:
        """Two-phase training on the extracted ZIP; uploads the checkpoint to the MLflow run."""
        torch.manual_seed(_TRAIN_SEED)
        random.seed(_TRAIN_SEED)
        device = _safe_device()

        tmp_dir = tempfile.mkdtemp(prefix="ml8_train_")
        with removing_dir(tmp_dir):
            safe_extract_zip(zip_path, tmp_dir)
            train_records, val_records = _load_records(tmp_dir)
            # 1) balance_hongos: recorte de la categoría mayoritaria 'hongos' (solo train)
            train_records = _balance_hongos(
                train_records, max_samples_hongos=None, seed=_TRAIN_SEED
            )
            setup = _make_setup(train_records, val_records, device)
            h1, h2, t1, t2 = _fit(setup, tracker)

            checkpoint = _checkpoint(setup.model)
            tracker.log_metrics({
                "best_val_acc_cat": _best(h2["val_acc_cat"], h1["val_acc_cat"]),
                "best_val_acc_cer": _best(h2["val_acc_cer"], h1["val_acc_cer"]),
                "train_samples": len(train_records),
                "val_samples": len(val_records),
            })
            upload_model_to_mlflow(
                tracker,
                lambda model_dir: torch.save(checkpoint, os.path.join(model_dir, MODEL_FILENAME)),
                prefix="ml8_mlflow_",
            )

            return TrainResponse(
                detail="Entrenamiento completado",
                train_samples=len(train_records),
                val_samples=len(val_records),
                fase1_epochs=len(h1["val_acc_cat"]),
                fase2_epochs=len(h2["val_acc_cat"]),
                fase1_time_min=round(t1 / 60, 1),
                fase2_time_min=round(t2 / 60, 1),
                best_val_acc_cat=_best(h2["val_acc_cat"], h1["val_acc_cat"]),
                best_val_acc_cer=_best(h2["val_acc_cer"], h1["val_acc_cer"]),
                mlflow_run_id=mlflow_run_id,
                upload_warning=None,
            )
