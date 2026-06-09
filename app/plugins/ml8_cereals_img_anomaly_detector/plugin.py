from __future__ import annotations

import gc
import logging
import random
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import InvalidImageError, ModelNotLoadedError
from app.plugins.ml8_cereals_img_anomaly_detector.constants import (
    ARTIFACT_FOLDER_NAME,
    CATEGORY_NAMES,
    CEREAL_NAMES,
    IMAGE_EXTENSIONS,
    MODEL_FILENAME,
    MODEL_ID,
)
from app.plugins.ml8_cereals_img_anomaly_detector.model_loader import load_model_bundle
from app.plugins.ml8_cereals_img_anomaly_detector.postprocessing import build_batch_response, build_inline_response
from app.plugins.ml8_cereals_img_anomaly_detector.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml8_cereals_img_anomaly_detector.preprocessing import image_base64_to_tensor, image_path_to_tensor
from app.plugins.ml8_cereals_img_anomaly_detector.train_dto import TrainResponse

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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


def _train_epoch(model, loader, optimizer, crit_cat, crit_cer, device):
    from torch.nn import CrossEntropyLoss  # noqa: F401 — already imported by caller
    model.train()
    total_loss = correct_cat = correct_cer = total = 0
    for imgs, lbls_cat, lbls_cer, _ in loader:
        imgs = imgs.to(device)
        lbls_cat = lbls_cat.to(device)
        lbls_cer = lbls_cer.to(device)
        out_cat, out_cer = model(imgs)
        loss = crit_cat(out_cat, lbls_cat) + crit_cer(out_cer, lbls_cer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_cat += (out_cat.argmax(1) == lbls_cat).sum().item()
        correct_cer += (out_cer.argmax(1) == lbls_cer).sum().item()
        total += lbls_cat.size(0)
    n = max(len(loader), 1)
    return total_loss / n, 100 * correct_cat / total if total else 0.0, 100 * correct_cer / total if total else 0.0


def _validate_epoch(model, loader, crit_cat, crit_cer, device):
    model.eval()
    total_loss = correct_cat = correct_cer = total = 0
    with torch.no_grad():
        for imgs, lbls_cat, lbls_cer, _ in loader:
            imgs = imgs.to(device)
            lbls_cat = lbls_cat.to(device)
            lbls_cer = lbls_cer.to(device)
            out_cat, out_cer = model(imgs)
            total_loss += (crit_cat(out_cat, lbls_cat) + crit_cer(out_cer, lbls_cer)).item()
            correct_cat += (out_cat.argmax(1) == lbls_cat).sum().item()
            correct_cer += (out_cer.argmax(1) == lbls_cer).sum().item()
            total += lbls_cat.size(0)
    n = max(len(loader), 1)
    return total_loss / n, 100 * correct_cat / total if total else 0.0, 100 * correct_cer / total if total else 0.0


def _run_phase(model, train_loader, val_loader, optimizer, crit_cat, crit_cer,
               device, epochs, patience, phase_name):
    best_loss, no_improve, best_state = float("inf"), 0, None
    history: dict = {"val_acc_cat": [], "val_acc_cer": []}
    logger.info("Iniciando %s", phase_name)
    for epoch in range(epochs):
        tr_loss, tr_acc_cat, tr_acc_cer = _train_epoch(
            model, train_loader, optimizer, crit_cat, crit_cer, device
        )
        val_loss, val_acc_cat, val_acc_cer = _validate_epoch(
            model, val_loader, crit_cat, crit_cer, device
        )
        history["val_acc_cat"].append(val_acc_cat)
        history["val_acc_cer"].append(val_acc_cer)
        logger.info(
            "  Epoch %d/%d | tr=%.4f val=%.4f cat=%.1f%%/%.1f%% cer=%.1f%%/%.1f%%",
            epoch + 1, epochs, tr_loss, val_loss,
            tr_acc_cat, val_acc_cat, tr_acc_cer, val_acc_cer,
        )
        if val_loss < best_loss:
            best_loss, no_improve = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("  Early stopping en epoch %d", epoch + 1)
                break
    if best_state:
        model.load_state_dict(best_state)
    return history


# ── Plugin ────────────────────────────────────────────────────────────────────

class Ml8CerealsImgAnomalyDetectorPlugin(ModelPluginPort):

    def __init__(self) -> None:
        self._bundle: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        self._bundle = load_model_bundle()
        logger.info("Ml8CerealsImgAnomalyDetectorPlugin loaded: %s", self._bundle["model_id"])

    def is_loaded(self) -> bool:
        return self._bundle is not None

    def _require_bundle(self) -> dict:
        if self._bundle is None:
            raise ModelNotLoadedError("El modelo no está cargado.")
        return self._bundle

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
    ) -> PredictInlineResponse:
        bundle = self._require_bundle()
        tensor = image_base64_to_tensor(
            features["image_base64"], image_size=bundle["image_size"]
        ).to(bundle["device"])

        bundle["model"].eval()
        with torch.no_grad():
            logits_cat, logits_cer = bundle["model"](tensor)

        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info("predict_inline done — count=%d", self._predict_count)

        return PredictInlineResponse(**build_inline_response(
            logits_cat,
            logits_cer,
            idx_to_class=bundle["idx_to_class"],
            idx_to_cereal=bundle["idx_to_cereal"],
            model_id=bundle["model_id"],
        ))

    def predict_batch(self, *, data_path: str) -> PredictBatchResponse:
        bundle = self._require_bundle()
        model = bundle["model"]
        device: torch.device = bundle["device"]
        image_size: int = bundle["image_size"]
        predictions: list[dict] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(data_path, "r") as zf:
                zf.extractall(tmp_dir)

            image_paths = sorted(
                p for p in Path(tmp_dir).rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )

            model.eval()
            for image_path in image_paths:
                try:
                    tensor = image_path_to_tensor(image_path, image_size=image_size).to(device)
                    with torch.no_grad():
                        logits_cat, logits_cer = model(tensor)
                    result = build_inline_response(
                        logits_cat,
                        logits_cer,
                        idx_to_class=bundle["idx_to_class"],
                        idx_to_cereal=bundle["idx_to_cereal"],
                        model_id=bundle["model_id"],
                    )
                    result["filename"] = image_path.name
                    predictions.append(result)
                except InvalidImageError as exc:
                    predictions.append({"filename": image_path.name, "error": str(exc)})
                except Exception as exc:
                    logger.warning("Unexpected error processing %s: %s", image_path.name, exc)
                    predictions.append({"filename": image_path.name, "error": str(exc)})

        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info("predict_batch done — %d predictions count=%d", len(predictions), self._predict_count)

        return PredictBatchResponse(**build_batch_response(predictions, model_id=bundle["model_id"]))

    def stats(self) -> StatsResponse:
        arch = self._bundle["arch"] if self._bundle is not None else "unknown"
        model_id = self._bundle["model_id"] if self._bundle is not None else MODEL_ID

        return StatsResponse(
            model_name=model_id,
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
                OutputField(name="categoria", type="str", description="Categoría predicha (anomalía/estado)"),
                OutputField(name="cereal", type="str", description="Tipo de cereal predicho"),
                OutputField(name="confianza_categoria", type="float", description="Confianza de la categoría [0, 1]"),
                OutputField(name="confianza_cereal", type="float", description="Confianza del cereal [0, 1]"),
                OutputField(name="probabilidades_categoria", type="dict", description="Probabilidad por categoría"),
                OutputField(name="probabilidades_cereal", type="dict", description="Probabilidad por cereal"),
            ],
            metrics={"arch": arch},
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=None,
            ),
        )

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def train(self, *, data_path: str) -> TrainResponse:
        import torch.nn as nn
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms

        from app.infrastructure.artifact_store import ArtifactStore
        from app.plugins.ml8_cereals_img_anomaly_detector.model_loader import MultiTaskMobileNetV3Large

        if not data_path.lower().endswith(".zip"):
            raise ValueError("data_path debe ser un fichero .zip")

        from app.plugins.ml8_cereals_img_anomaly_detector.model_loader import _safe_device
        device = _safe_device()
        store = ArtifactStore(ARTIFACT_FOLDER_NAME)

        class_to_idx = {c: i for i, c in enumerate(CATEGORY_NAMES)}
        cereal_to_idx = {c: i for i, c in enumerate(CEREAL_NAMES)}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        idx_to_cereal = {v: k for k, v in cereal_to_idx.items()}

        _train_tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        _val_tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        class _CerealDataset(Dataset):
            def __init__(self, records, is_train):
                self.records = records
                self.tfm = _train_tfm if is_train else _val_tfm

            def __len__(self):
                return len(self.records)

            def __getitem__(self, idx):
                img_path, categoria, cereal = self.records[idx]
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = self.tfm(img)
                except Exception:
                    img = torch.zeros(3, 224, 224)
                return img, class_to_idx[categoria], cereal_to_idx[cereal], img_path

        tmp_dir = tempfile.mkdtemp(prefix="ml8_train_")
        try:
            with zipfile.ZipFile(data_path, "r") as zf:
                zf.extractall(tmp_dir)

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

            train_loader = DataLoader(
                _CerealDataset(train_records, is_train=True),
                batch_size=16, shuffle=True, num_workers=0,
            )
            val_loader = DataLoader(
                _CerealDataset(val_records, is_train=False),
                batch_size=16, shuffle=False, num_workers=0,
            )
            logger.info("Dataset: %d train, %d val", len(train_records), len(val_records))

            # Backbone preentrenado, clasificador descongelado para fase 1
            base = models.mobilenet_v3_large(weights="IMAGENET1K_V2")
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

            crit_cat = nn.CrossEntropyLoss()
            crit_cer = nn.CrossEntropyLoss()

            # Fase 1: transfer learning (neck + heads)
            t0 = time.perf_counter()
            opt1 = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=0.001
            )
            h1 = _run_phase(
                model, train_loader, val_loader, opt1, crit_cat, crit_cer,
                device, epochs=10, patience=3, phase_name="Fase 1: Transfer Learning",
            )
            t1 = time.perf_counter() - t0
            gc.collect()

            # Fase 2: fine-tune completo
            for param in model.parameters():
                param.requires_grad = True
            opt2 = torch.optim.Adam(model.parameters(), lr=0.00001)
            t0 = time.perf_counter()
            h2 = _run_phase(
                model, train_loader, val_loader, opt2, crit_cat, crit_cer,
                device, epochs=5, patience=3, phase_name="Fase 2: Fine-tuning",
            )
            t2 = time.perf_counter() - t0
            gc.collect()

            # Guardar checkpoint
            artifact_path = store.local_dir / MODEL_FILENAME
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_name": "mobilenet_v3_large",
                "model_state_dict": model.state_dict(),
                "num_classes": len(CATEGORY_NAMES),
                "num_cereales": len(CEREAL_NAMES),
                "class_to_idx": class_to_idx,
                "cereal_to_idx": cereal_to_idx,
                "idx_to_class": idx_to_class,
                "idx_to_cereal": idx_to_cereal,
            }, artifact_path)
            logger.info("Checkpoint guardado en %s", artifact_path)

            upload_warning: str | None = None
            try:
                store.upload(MODEL_FILENAME)
            except Exception as exc:
                upload_warning = f"Artefactos guardados en local; fallo en S3: {exc}"
                logger.warning(upload_warning)

            self.load()

            combined_cat = h2["val_acc_cat"] or h1["val_acc_cat"]
            combined_cer = h2["val_acc_cer"] or h1["val_acc_cer"]
            return TrainResponse(
                detail="Entrenamiento completado",
                train_samples=len(train_records),
                val_samples=len(val_records),
                fase1_epochs=len(h1["val_acc_cat"]),
                fase2_epochs=len(h2["val_acc_cat"]),
                fase1_time_min=round(t1 / 60, 1),
                fase2_time_min=round(t2 / 60, 1),
                best_val_acc_cat=round(max(combined_cat), 1) if combined_cat else 0.0,
                best_val_acc_cer=round(max(combined_cer), 1) if combined_cer else 0.0,
                upload_warning=upload_warning,
            )
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            gc.collect()
