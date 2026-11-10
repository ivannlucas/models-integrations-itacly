"""Ml5MeatCowBehaviour — Detectron2 + ByteTrack + SlowFast cow-behaviour recognition.

``train()`` retrains the SlowFast behaviour CLASSIFIER only (see train_dto.py for the
ZIP contract and training.py for the fixed training pipeline). The Faster R-CNN
DETECTOR always keeps using the fixed served artifact — it stays out of ``/train``'s
scope (bbox-annotation-only ZIP contract mismatch, see inbox/a05/manifest.yaml
training.reason).
"""
from __future__ import annotations

import gc
import logging
import os
import tempfile
import time
import zipfile
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.application.dto.train_dto import TrainResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import (
    InsufficientFramesError,
    InvalidVideoError,
    ModelNotLoadedError,
)
from app.domain.services.mlflow_artifacts import (
    removing_dir,
    require_train_tracker,
    upload_model_to_mlflow,
)
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.infrastructure.archive import safe_extract_zip
from app.infrastructure.artifact_store import _build_s3_client, download_s3_file
from app.plugins.ml5_meat_cow_behaviour.byte_tracker import ByteTracker
from app.plugins.ml5_meat_cow_behaviour.constants import (
    ALPHA,
    ANNOTATIONS_DIRNAME,
    CLASSIFIER_FILENAME,
    CLIP_LENGTH,
    CROP_SIZE,
    CVAT_ANNOTATIONS_FILENAME,
    DEFAULT_ANOMALY_THRESHOLD,
    FRAMEWORK,
    MODEL_ID,
    RAW_FRAMES_DIRNAME,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAINING_BEHAVIOR_TO_IDX,
    VERSION,
)
from app.plugins.ml5_meat_cow_behaviour.mlflow_utils import download_user_model_from_mlflow
from app.plugins.ml5_meat_cow_behaviour.model_loader import (
    build_classifier_module,
    load_model_bundle,
)
from app.plugins.ml5_meat_cow_behaviour.postprocessing import decode_logits
from app.plugins.ml5_meat_cow_behaviour.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml5_meat_cow_behaviour.preprocessing import (
    decode_frames_base64,
    extract_cow_roi,
    prepare_slowfast_tensor,
)
from app.plugins.ml5_meat_cow_behaviour.training import (
    build_clips,
    load_all_clips,
    split_clips,
    train_classifier,
)

logger = logging.getLogger(__name__)


def _resolve_video_path(data_path: str) -> tuple[str, str | None]:
    """Return (local_path, temp_path). Downloads from S3 if data_path is an s3:// URI."""
    if not data_path.startswith("s3://"):
        return data_path, None

    without_prefix = data_path[len("s3://"):]
    bucket, key = without_prefix.split("/", 1)
    suffix = os.path.splitext(key)[-1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        pass
    logger.info("Downloading video from S3: %s → %s", data_path, tmp.name)
    _build_s3_client().download_file(bucket, key, tmp.name)
    return tmp.name, tmp.name


def _detect(detector, frame) -> list[dict]:
    """Run the Faster R-CNN detector on one frame; returns ``[{"bbox", "score"}, ...]``."""
    outputs = detector(frame)
    instances = outputs["instances"].to("cpu")
    return [
        {"bbox": box.tolist(), "score": float(score)}
        for box, score in zip(
            instances.pred_boxes.tensor.numpy(),
            instances.scores.numpy(),
        )
    ]


def _is_zip(local_path: str) -> bool:
    """True for a ZIP, by extension or by content (an s3:// key may have no ".zip" suffix)."""
    return local_path.lower().endswith(".zip") or zipfile.is_zipfile(local_path)


def _load_training_clips(local_data_path: str, temp_dir: str) -> tuple[Path, list, list]:
    """Extract the ZIP, read its CVAT clips and split them; returns (raw_frames_dir, train, val)."""
    safe_extract_zip(local_data_path, temp_dir)

    root = Path(temp_dir)
    entries = list(root.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        root = entries[0]

    raw_frames_dir = root / RAW_FRAMES_DIRNAME
    annotations_dir = root / ANNOTATIONS_DIRNAME
    if not raw_frames_dir.is_dir() or not annotations_dir.is_dir():
        raise ValueError(
            "ZIP sin estructura válida. Esperado en la raíz: "
            f"'{RAW_FRAMES_DIRNAME}/<clip>/*.jpg' + "
            f"'{ANNOTATIONS_DIRNAME}/<clip>/annotations/{CVAT_ANNOTATIONS_FILENAME}'."
        )

    data = load_all_clips(str(raw_frames_dir), str(annotations_dir))
    if not data["images"]:
        raise ValueError("No se encontraron clips válidos (CVAT/COCO) en el ZIP.")

    splits = split_clips(data)
    train_clips = build_clips(data, splits["train"], CLIP_LENGTH)
    val_clips = build_clips(data, splits["val"], CLIP_LENGTH)
    if not train_clips or not val_clips:
        raise ValueError(
            "Dataset insuficiente tras el split train/val por clip "
            "(se necesitan varios clips para poblar ambos splits)."
        )
    return raw_frames_dir, train_clips, val_clips


class Ml5MeatCowBehaviourPlugin(PluginRuntimeMixin, ModelPluginPort):
    """Faster R-CNN (detection) + ByteTrack (tracking) + SlowFast (classification) pipeline."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin (counters and latency start at the mixin defaults)."""
        self._bundle: dict | None = None

    def load(self) -> None:
        """Load the detector and classifier into memory."""
        self._bundle = load_model_bundle()
        logger.info("Ml5MeatCowBehaviourPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        """Return True if the model bundle is loaded and ready for inference."""
        return self._bundle is not None

    def _require_bundle(self) -> dict:
        """Return the loaded bundle or raise :class:`ModelNotLoadedError`."""
        if self._bundle is None:
            raise ModelNotLoadedError("El modelo no está cargado.")
        return self._bundle

    @contextmanager
    def _request_bundle(self, mlflow_run_id: str):
        """Yield the bundle that serves this request.

        Without ``mlflow_run_id`` it is the shared served bundle. With one, it is a
        request-local copy whose ``classifier`` is the user's retrained SlowFast (the
        detector and decoding tables are shared, read-only). ``self._bundle`` is never
        written: the plugin instance is shared by concurrent requests, and swapping the
        user's classifier into it (as this method used to) let a concurrent request be
        served another user's model, or the base one get a user's.
        """
        bundle = self._require_bundle()
        result = download_user_model_from_mlflow(mlflow_run_id) if mlflow_run_id else None
        if result is None:
            yield bundle
            return
        state_dict, tmp_dir = result
        with removing_dir(tmp_dir):
            num_classes = len(TRAINING_BEHAVIOR_TO_IDX)
            user_classifier = build_classifier_module(num_classes, bundle["device"])
            user_classifier.load_state_dict(state_dict)
            user_classifier.eval()
            yield {**bundle, "classifier": user_classifier}

    @staticmethod
    def _classify_clip(bundle: dict, frames_np: np.ndarray, anomaly_threshold: float) -> dict:
        """Run the bundle's SlowFast on a clip of frames and decode the behaviour prediction."""
        slow_fast = prepare_slowfast_tensor(frames_np, bundle["device"], alpha=ALPHA)
        with torch.no_grad():
            logits = bundle["classifier"](slow_fast)
        return decode_logits(logits, bundle["idx_to_behavior"], anomaly_threshold)

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Run the full detection + tracking + classification pipeline over a video file."""
        local_path, temp_path = _resolve_video_path(data_path)
        try:
            with self._request_bundle(mlflow_run_id) as bundle:
                return self._run_video_pipeline(bundle, bundle["detector"], local_path, data_path)
        finally:
            if temp_path:
                os.unlink(temp_path)

    def _run_video_pipeline(
        self, bundle: dict, detector: object, local_path: str, original_path: str
    ) -> PredictBatchResponse:
        """Execute the frame-by-frame detection + tracking + classification loop."""
        cap = cv2.VideoCapture(local_path)  # pylint: disable=no-member
        if not cap.isOpened():
            raise InvalidVideoError(f"Cannot open video file: {original_path}")

        logger.info(
            "predict_batch — video=%s total_frames=%d",
            original_path, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),  # pylint: disable=no-member
        )

        tracker = ByteTracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
        track_buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=CLIP_LENGTH))
        track_last_result: dict[int, dict] = {}

        all_frame_results: list[dict] = []
        frame_idx = 0
        t0 = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_detections = [
                self._track_detection(bundle, frame, track, (track_buffers, track_last_result))
                for track in tracker.update(_detect(detector, frame))
            ]

            all_frame_results.append({"frame": frame_idx, "detections": frame_detections})
            frame_idx += 1

        cap.release()

        self._record_latency((time.perf_counter() - t0) * 1000)
        logger.info(
            "predict_batch done — %d frames processed, count=%d", frame_idx, self._predict_count
        )

        return PredictBatchResponse(
            model_id=MODEL_ID,
            predictions=all_frame_results,
            output_path=None,
        )

    def _track_detection(self, bundle: dict, frame, track: dict, state: tuple) -> dict:
        """Buffer this track's ROI, classify its clip once full, and build its detection row.

        ``state`` is (track_buffers, track_last_result), shared across the video's frames.
        """
        track_buffers, track_last_result = state
        roi = extract_cow_roi(frame, track["bbox"])
        track_buffers[track["track_id"]].append(roi)

        if len(track_buffers[track["track_id"]]) >= CLIP_LENGTH:
            frames_np = np.array(list(track_buffers[track["track_id"]]))
            result = self._classify_clip(
                bundle, frames_np, anomaly_threshold=DEFAULT_ANOMALY_THRESHOLD
            )
            track_last_result[track["track_id"]] = result
        else:
            result = track_last_result.get(
                track["track_id"],
                {
                    "prediction": (
                        list(bundle["behavior_to_idx"].keys())[0]
                        if bundle["behavior_to_idx"] else "unknown"
                    ),
                    "confidence": 1.0,
                    "is_anomaly": False,
                    "behavior_idx": 0,
                },
            )

        return {
            "track_id": track["track_id"],
            "bbox": track["bbox"],
            "score": track["score"],
            "behavior": result["prediction"],
            "behavior_confidence": result["confidence"],
            "is_anomaly": result["is_anomaly"],
        }

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Classify a pre-cropped clip of one cow given as base64 frames."""
        frames_b64: list[str] = features["frames_base64"]
        anomaly_threshold = threshold if threshold is not None else DEFAULT_ANOMALY_THRESHOLD

        # Belt-and-suspenders beyond PredictInlineRequest's Pydantic min_length=CLIP_LENGTH:
        # protects any direct (non-HTTP) caller of predict_inline that bypasses DTO validation.
        if len(frames_b64) < CLIP_LENGTH:
            raise InsufficientFramesError(
                f"Se requieren al menos {CLIP_LENGTH} frames, recibidos {len(frames_b64)}."
            )

        frames_b64_clip = frames_b64[-CLIP_LENGTH:]
        frames_np = decode_frames_base64(frames_b64_clip)

        t0 = time.perf_counter()
        with self._request_bundle(mlflow_run_id) as bundle:
            result = self._classify_clip(bundle, frames_np, anomaly_threshold)
        self._record_latency((time.perf_counter() - t0) * 1000)
        logger.info(
            "predict_inline done — behavior=%s confidence=%.4f anomaly=%s count=%d",
            result["prediction"], result["confidence"], result["is_anomaly"], self._predict_count,
        )

        return PredictInlineResponse(
            model_id=MODEL_ID,
            threshold=anomaly_threshold,
            prediction=result["prediction"],
            confidence=result["confidence"],
            features_used=["frames_base64"],
            is_anomaly=result["is_anomaly"],
            behavior_idx=result["behavior_idx"],
            xai_feature_values=result.get("all_probs", {}),
        )

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Retrain the SlowFast behaviour classifier from a ZIP of annotated clips.

        Only the classifier is in scope (see train_dto.py for the exact ZIP contract);
        the Faster R-CNN detector keeps using the fixed served artifact unconditionally.
        The retrained classifier is persisted ONLY to ``mlflow_run_id``'s own MLflow run
        (predict/stats with that same run_id swap it back in — see
        ``_request_bundle``); the served base artifact in ArtifactStore is never
        touched/overwritten, same pattern as every other plugin with real training
        support (e.g. ml10_dairy_disease_vector_detection).
        """
        tracker = require_train_tracker(mlflow_run_id)
        tmp_zip = download_s3_file(data_path) if data_path.startswith("s3://") else None
        local_data_path = tmp_zip or data_path
        try:
            if not _is_zip(local_data_path):
                raise ValueError("data_path debe ser un fichero .zip")

            temp_dir = tempfile.mkdtemp(prefix="ml5_train_")
            with removing_dir(temp_dir):
                return self._train_on_zip(local_data_path, temp_dir, tracker, mlflow_run_id)
        finally:
            if tmp_zip and os.path.exists(tmp_zip):
                os.unlink(tmp_zip)
            gc.collect()

    def _train_on_zip(
        self, local_data_path: str, temp_dir: str, tracker, mlflow_run_id: str
    ) -> TrainResponse:
        """Train the classifier on the extracted ZIP and persist it to the tracker's run."""
        raw_frames_dir, train_clips, val_clips = _load_training_clips(local_data_path, temp_dir)

        device = self._bundle["device"] if self._bundle else "cpu"

        tracker.log_params({
            "lr": 0.0005,
            "weight_decay": 1e-4,
            "batch_size": TRAIN_BATCH_SIZE,
            "max_epochs": TRAIN_EPOCHS,
            "optimizer": "AdamW",
            "scheduler": "warmup(2 epochs)+cosine, stepped per-batch",
            "loss": "FocalLoss(alpha=0.5, gamma=1.5)",
            "sampler": "WeightedRandomSampler",
            "model": "SlowFast-R50",
            "train_clips": len(train_clips),
            "val_clips": len(val_clips),
        })

        t0 = time.perf_counter()
        model, train_metrics = train_classifier(
            train_clips, val_clips, str(raw_frames_dir), CLIP_LENGTH, device,
            mlflow_log_metrics=tracker.log_metrics,
        )
        elapsed = time.perf_counter() - t0

        def _write(model_dir: str) -> None:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "behavior_to_idx": TRAINING_BEHAVIOR_TO_IDX,
            }
            torch.save(checkpoint, os.path.join(model_dir, CLASSIFIER_FILENAME))

        upload_model_to_mlflow(
            tracker, _write, prefix="ml5_mlflow_", artifact_path="classifier"
        )

        tracker.log_metrics({"training_time_min": round(elapsed / 60, 2)})

        return TrainResponse(
            detail="Entrenamiento del clasificador SlowFast completado.",
            metrics={**train_metrics, "time_min": round(elapsed / 60, 2)},
            mlflow_run_id=mlflow_run_id,
            upload_warning=None,
        )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        # Downloading the run's classifier checks it exists (422 otherwise); the behaviour
        # labels are the served model's either way.
        with self._request_bundle(mlflow_run_id) as bundle:
            behaviors = list(bundle["idx_to_behavior"].values())
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Reconocimiento de comportamiento bovino en vídeo mediante Faster R-CNN "
                "(detección) + ByteTrack (tracking) + SlowFast R50 (clasificación)."
            ),
            task_type="behavior_recognition",
            framework=FRAMEWORK,
            inputs=[
                InputField(
                    name="frames_base64",
                    type="array",
                    format=["jpg", "jpeg", "png"],
                    description=(
                        f"Array de al menos {CLIP_LENGTH} frames JPEG/PNG en base64 "
                        f"(ROI de vaca recortado, {CROP_SIZE}×{CROP_SIZE} px recomendado)"
                    ),
                ),
                InputField(
                    name="threshold",
                    type="float",
                    default=DEFAULT_ANOMALY_THRESHOLD,
                    description="Override del umbral de anomalía",
                ),
            ],
            outputs=[
                OutputField(
                    name="prediction",
                    type="str",
                    description=f"Comportamiento predicho (uno de: {behaviors or 'cargando…'})",
                ),
                OutputField(name="confidence", type="float",
                            description="Probabilidad softmax del comportamiento predicho [0, 1]"),
                OutputField(name="is_anomaly", type="bool",
                            description="True si confidence < threshold"),
            ],
            metrics={},
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=self._avg_latency_ms(1),
            ),
        )
