"""Ml6MeatPoultryDiseaseClassification — deep-ensemble poultry disease classifier (use case a06).

Serves the AI team's two majority-voting ensembles (src/main.py::predict):

- ``model_key="thermal"``: DenseNet-121 + EfficientNetV2-S + ResNet-50 → healthy / sick.
- ``model_key="rgb"``: VGG19-BN + EfficientNet-B7 + MobileNetV3-L → healthy / cocci / ncd / salmo.

train() retrains the three base models of one ensemble with the original procedure
(training.py) and keeps the result ONLY in the MLflow run (manifest.training,
inbox/a06/manifest.yaml); the base artifacts are never written.
"""
from __future__ import annotations

import base64
import binascii
import copy
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import (
    InvalidImageError,
    ModelNotLoadedError,
    ModelPersistenceError,
    UserModelUnavailableError,
)
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.archive import safe_extract_zip
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml6_meat_poultry_disease_classification.constants import (
    CLASS_NAMES,
    FRAMEWORK,
    IMAGE_EXTENSIONS,
    MLFLOW_METADATA_FILENAME,
    MODEL_FILENAMES,
    MODEL_ID,
    REPORTED_TEST_METRICS,
    TASK_MODELS,
    TASKS,
    TRAIN_EPOCHS,
    TRAIN_GAMMA,
    TRAIN_HYPERPARAMS,
    TRAIN_MIN_DELTA,
    TRAIN_PATIENCE,
    TRAIN_SEED,
    TRAIN_STEP_SIZE,
    VERSION,
)
from app.plugins.ml6_meat_poultry_disease_classification.exceptions import UnknownImageModalityError
from app.plugins.ml6_meat_poultry_disease_classification.mlflow_utils import (
    download_user_model_from_mlflow,
    upload_artifacts_to_mlflow,
)
from app.plugins.ml6_meat_poultry_disease_classification.model_loader import load_ensembles
from app.plugins.ml6_meat_poultry_disease_classification.postprocessing import (
    classification_metrics,
    run_ensemble,
)
from app.plugins.ml6_meat_poultry_disease_classification.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml6_meat_poultry_disease_classification.preprocessing import (
    decode_image,
    input_warnings,
    load_image,
    to_tensor,
)
from app.plugins.ml6_meat_poultry_disease_classification.train_dto import TrainResponse
from app.plugins.ml6_meat_poultry_disease_classification.training import (
    class_index_of_folder,
    ensemble_predictions,
    make_loaders,
    predict_loader,
    split_dataset,
    train_model,
)

logger = logging.getLogger(__name__)


def _require_task(model_key: str | None) -> str:
    """Validate the ensemble selector (model_key / task)."""
    if model_key not in TASKS:
        raise UnknownImageModalityError(
            f"model_key debe ser 'thermal' (imagen térmica del ave) o 'rgb' (imagen RGB de heces); "
            f"recibido: {model_key!r}."
        )
    return model_key


def _unwrap_single_dir(path: str) -> str:
    """A ZIP whose only entry is a folder → that folder."""
    entries = os.listdir(path)
    if len(entries) == 1 and os.path.isdir(os.path.join(path, entries[0])):
        return os.path.join(path, entries[0])
    return path


class Ml6MeatPoultryDiseaseClassificationPlugin(ModelPluginPort):
    """Two independent 3-model majority-voting image ensembles (thermal / rgb)."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._ensembles: dict[str, dict[str, Any]] | None = None
        self._device: torch.device | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the six base checkpoints (both ensembles)."""
        self._ensembles, self._device = load_ensembles()
        logger.info("Ml6MeatPoultryDiseaseClassificationPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        """Return True if both ensembles are loaded."""
        return self._ensembles is not None

    def _require_loaded(self) -> None:
        if self._ensembles is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _resolve_models(
        self, task: str, mlflow_run_id: str,
    ) -> tuple[dict, torch.device, str | None]:
        """Return ``(models, device, temp_dir)`` for this request — never assigned to self."""
        if not mlflow_run_id:
            self._require_loaded()
            return self._ensembles[task], self._device, None
        run_task, models, device, tmp = download_user_model_from_mlflow(mlflow_run_id)
        if run_task != task:
            shutil.rmtree(tmp, ignore_errors=True)
            raise UserModelUnavailableError(
                f"El run de MLflow '{mlflow_run_id}' contiene un ensemble '{run_task}' "
                f"reentrenado, no '{task}'. No se usa el modelo base en su lugar."
            )
        return models, device, tmp

    # ── predict ────────────────────────────────────────────────────────────

    def predict_inline(  # pylint: disable=too-many-locals
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Classify one image with the ensemble chosen by ``model_key``."""
        task = _require_task(model_key)
        if features.get("image_path"):
            path = features["image_path"]
            if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS:
                raise InvalidImageError(f"Extensión no soportada. Usa {sorted(IMAGE_EXTENSIONS)}")
            if not os.path.isfile(path):
                raise InvalidImageError(f"image_path no existe: {path}")
            with open(path, "rb") as fh:
                image_bytes = fh.read()
            used = ["image_path"]
        elif features.get("image_base64"):
            try:
                image_bytes = base64.b64decode(features["image_base64"], validate=False)
            except (binascii.Error, ValueError) as exc:
                raise InvalidImageError(f"image_base64 no es base64 válido: {exc}") from exc
            used = ["image_base64"]
        else:
            raise InvalidImageError("features debe contener 'image_path' o 'image_base64'")
        image = decode_image(image_bytes)

        models, device, user_tmp = self._resolve_models(task, mlflow_run_id)
        try:
            result = run_ensemble(models, to_tensor(image, task).to(device), task)
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)
        self._record()
        return PredictInlineResponse(
            model_id=MODEL_ID, task=task, threshold=threshold, features_used=used,
            warnings=input_warnings(image, task), **result,
        )

    def predict_batch(  # pylint: disable=too-many-locals
        self, *, data_path: str, mlflow_run_id: str = "", model_key: str | None = None,
    ) -> PredictBatchResponse:
        """Classify every image of a directory or ZIP (local or s3://).

        Images inside a folder named like a class (healthy/sick; healthy|pcrhealthy, …) carry
        their true class; if all do, the response also has the test-set style metrics of
        predictor.py::save_metrics for each base model and the ensemble.
        """
        task = _require_task(model_key)
        models, device, user_tmp = self._resolve_models(task, mlflow_run_id)
        extract_dir: str | None = None
        try:
            with local_file_path(data_path) as local_path:
                image_dir = local_path
                if local_path.lower().endswith(".zip"):
                    extract_dir = tempfile.mkdtemp(prefix="ml6_batch_")
                    safe_extract_zip(local_path, extract_dir)
                    image_dir = _unwrap_single_dir(extract_dir)
                if not os.path.isdir(image_dir):
                    raise FileNotFoundError(
                        f"data_path no es un directorio ni un .zip: {data_path}")
                files = sorted(
                    os.path.relpath(os.path.join(root, f), image_dir)
                    for root, _, names in os.walk(image_dir) for f in names
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                )
                rows, y_true, votes = [], [], {name: [] for name in models}
                y_pred: list[int] = []
                for rel in files:
                    parent = os.path.basename(os.path.dirname(rel))
                    true_idx = class_index_of_folder(parent, task) if parent else None
                    try:
                        image = load_image(os.path.join(image_dir, rel))
                        result = run_ensemble(models, to_tensor(image, task).to(device), task)
                    except Exception as exc:  # pylint: disable=broad-except
                        rows.append({"filename": os.path.basename(rel), "path": rel,
                                     "error": str(exc)})
                        continue
                    row = {"filename": os.path.basename(rel), "path": rel, **result}
                    warnings = input_warnings(image, task)
                    if warnings:
                        row["warnings"] = warnings
                    if true_idx is not None:
                        row["true_class"] = CLASS_NAMES[task][true_idx]
                        y_true.append(true_idx)
                        y_pred.append(result["predicted_class_index"])
                        for name, cls in result["model_predictions"].items():
                            votes[name].append(CLASS_NAMES[task].index(cls))
                    rows.append(row)
        finally:
            if extract_dir:
                shutil.rmtree(extract_dir, ignore_errors=True)
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)

        evaluated = None
        if y_true and len(y_true) == len(rows):
            evaluated = {"ensemble": classification_metrics(y_true, y_pred, task)}
            for name, model_votes in votes.items():
                evaluated[name] = classification_metrics(y_true, model_votes, task)
        self._record()
        return PredictBatchResponse(
            model_id=MODEL_ID, task=task, n_images=len(rows), predictions=rows,
            evaluated_metrics=evaluated, output_path=None,
        )

    # ── train ──────────────────────────────────────────────────────────────

    def train(  # pylint: disable=too-many-locals
        self, *, data_path: str, mlflow_run_id: str, task: str = "",
    ) -> TrainResponse:
        """Retrain the three base models of ``task`` on a raw dataset and save them to MLflow.

        Starts from a deep copy of the served checkpoints (self._ensembles is never mutated),
        follows src/training/trainer.py (see training.py), evaluates the test split of the user's
        data per model and for the ensemble, and uploads ``model/<name>.pt`` + metadata.json to
        ``mlflow_run_id``. A failed upload raises ModelPersistenceError (502).
        """
        if task not in TASKS:  # TrainRequest already restricts it; direct callers get a 400
            raise ValueError(f"task debe ser 'thermal' o 'rgb'; recibido: {task!r}")
        self._require_loaded()
        hyper = TRAIN_HYPERPARAMS[task]
        tracker = BaseMLflowTracker(mlflow_run_id)
        tracker.log_params({
            "task": task, "base_models": ",".join(TASK_MODELS[task]), "optimizer": "adam",
            "loss": "cross_entropy", "scheduler": "step_lr", "step_size": TRAIN_STEP_SIZE,
            "gamma": TRAIN_GAMMA, "epochs": TRAIN_EPOCHS, "patience": TRAIN_PATIENCE,
            "min_delta": TRAIN_MIN_DELTA, "seed": TRAIN_SEED, "init": "served_checkpoint", **hyper,
        })

        t0 = time.perf_counter()
        extract_dir = tempfile.mkdtemp(prefix="ml6_train_")
        try:
            with local_file_path(data_path) as local_path:
                if os.path.isfile(local_path) and local_path.lower().endswith(".zip"):
                    safe_extract_zip(local_path, extract_dir)
                    data_root = _unwrap_single_dir(extract_dir)
                elif os.path.isdir(local_path):
                    data_root = local_path
                else:
                    raise FileNotFoundError(
                        f"data_path no existe o no es un directorio/.zip: {data_path}")
                splits = split_dataset(data_root, task)
                loaders = make_loaders(splits, task, hyper["batch_size"])
                y_test = [label for _, label in splits["test"]]

                trained, per_model_preds, model_report = {}, {}, {}
                devices = [self._device.index or 0] if self._device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(TRAIN_SEED)
                    for name in TASK_MODELS[task]:
                        model = copy.deepcopy(self._ensembles[task][name]).to(self._device)

                        def _log(epoch, values, _name=name):
                            tracker.log_metrics(
                                {f"{_name}_{k}": v for k, v in values.items()}, step=epoch)

                        model, best_val_acc, epochs = train_model(
                            model, loaders, hyper, self._device, _log)
                        trained[name] = model
                        per_model_preds[name] = predict_loader(model, loaders["test"], self._device)
                        report = classification_metrics(y_test, per_model_preds[name], task)
                        model_report[name] = {
                            "test": {k: v for k, v in report.items() if k != "per_class"},
                            "best_val_accuracy": round(best_val_acc, 4), "epochs_run": epochs,
                        }
                        logger.info("ml6 train(%s): %s done (%d epochs, val_acc=%.4f)",
                                    task, name, epochs, best_val_acc)
                ens = classification_metrics(y_test, ensemble_predictions(per_model_preds), task)
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

        final_metrics = {
            "accuracy": ens["accuracy"], "precision_weighted": ens["precision_weighted"],
            "recall_weighted": ens["recall_weighted"], "f1_weighted": ens["f1_weighted"],
            "n_train": len(splits["train"]), "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
        }
        upload_dir = tempfile.mkdtemp(prefix="ml6_mlflow_upload_")
        try:
            for name, model in trained.items():
                torch.save(model.state_dict(), os.path.join(upload_dir, MODEL_FILENAMES[name]))
            metadata_path = os.path.join(upload_dir, MLFLOW_METADATA_FILENAME)
            with open(metadata_path, "w", encoding="utf-8") as fh:
                json.dump({"task": task, "class_names": list(CLASS_NAMES[task]),
                           "model_names": list(TASK_MODELS[task])}, fh)
            upload_artifacts_to_mlflow(upload_dir, mlflow_run_id, final_metrics)
        except Exception as exc:
            logger.error("ml6 train(): MLflow upload failed: %s", exc)
            raise ModelPersistenceError(
                f"El ensemble reentrenado no se ha podido guardar en MLflow: {exc}"
            ) from exc
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)

        return TrainResponse(
            detail=(f"Reentrenamiento del ensemble '{task}' completado (3 modelos base, test "
                    "del split 70/15/15)"),
            task=task, accuracy=ens["accuracy"], precision_weighted=ens["precision_weighted"],
            recall_weighted=ens["recall_weighted"], f1_weighted=ens["f1_weighted"],
            per_class=ens["per_class"], models=model_report,
            n_train=len(splits["train"]), n_val=len(splits["val"]), n_test=len(splits["test"]),
            training_time_s=round(time.perf_counter() - t0, 2), mlflow_run_id=mlflow_run_id,
            upload_warning=None,
        )

    # ── stats ──────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the AI team's declared test metrics and runtime counters."""
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Detección de enfermedades en aves de galpón con dos ensembles de CNN por votación "
                "por mayoría: 'thermal' (imagen térmica del ave → healthy/sick; DenseNet-121, "
                "EfficientNetV2-S, ResNet-50) y 'rgb' (imagen RGB de heces → "
                "healthy/cocci/ncd/salmo; "
                "VGG19-BN, EfficientNet-B7, MobileNetV3-L). Herramienta de apoyo a la decisión "
                "veterinaria, no sustituye al diagnóstico."
            ),
            task_type="image_classification_ensemble",
            framework=FRAMEWORK,
            inputs=[
                InputField(name="model_key", type="str", format=list(TASKS),
                           description="Ensemble: 'thermal' o 'rgb' (obligatorio)"),
                InputField(
                    name="image", type="file",
                    format=sorted(e.lstrip(".") for e in IMAGE_EXTENSIONS),
                    description=("thermal: imagen de la cámara térmica a su resolución nativa "
                                 "240x320; rgb: imagen de heces (se redimensiona a 256x256). "
                                 "base64/path en inline; directorio o .zip en batch"),
                ),
            ],
            outputs=[
                OutputField(name="prediction", type="str",
                            description="Clase del ensemble (voto por mayoría)"),
                OutputField(name="predicted_class_index", type="int",
                            description="Índice de la clase"),
                OutputField(name="confidence", type="float",
                            description="Fracción de modelos que votan la clase (1/3, 2/3, 1)"),
                OutputField(name="model_predictions", type="dict",
                            description="Voto de cada modelo base"),
                OutputField(name="model_probabilities", type="dict",
                            description="Softmax de cada modelo base (informativo)"),
                OutputField(name="warnings", type="list",
                            description="Avisos de entrada fuera de distribución"),
            ],
            metrics={"reported_test": REPORTED_TEST_METRICS},
            runtime_stats=RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=None),
        )
        if mlflow_run_id:
            try:
                tracker = BaseMLflowTracker(mlflow_run_id)
                base.metrics["mlflow"] = {
                    "params": tracker.get_params(), "metrics": tracker.get_metrics(),
                }
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Could not fetch MLflow stats for run_id=%s: %s", mlflow_run_id, exc)
        return base
