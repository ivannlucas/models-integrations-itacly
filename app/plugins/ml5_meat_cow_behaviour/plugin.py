"""Ml5MeatCowBehaviour — Detectron2 + ByteTrack + SlowFast cow-behaviour recognition.

The model serves externally-trained artifacts, so ``train()`` raises
``TrainingNotSupportedError`` (HTTP 501).
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import cv2
import numpy as np
import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import (
    InvalidVideoError,
    ModelNotLoadedError,
    TrainingNotSupportedError,
)
from app.plugins.ml5_meat_cow_behaviour.byte_tracker import ByteTracker
from app.plugins.ml5_meat_cow_behaviour.constants import (
    ALPHA,
    CLIP_LENGTH,
    CROP_SIZE,
    DEFAULT_ANOMALY_THRESHOLD,
    FRAMEWORK,
    MODEL_ID,
    VERSION,
)
from app.plugins.ml5_meat_cow_behaviour.model_loader import load_model_bundle
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

logger = logging.getLogger(__name__)


class Ml5MeatCowBehaviourPlugin(ModelPluginPort):
    """Faster R-CNN (detection) + ByteTrack (tracking) + SlowFast (classification) pipeline."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._bundle: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None
        self._total_latency_ms: float = 0.0

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

    def _classify_clip(self, frames_np: np.ndarray, anomaly_threshold: float) -> dict:
        """Run SlowFast on a clip of frames and decode the behaviour prediction."""
        bundle = self._require_bundle()
        slow_fast = prepare_slowfast_tensor(frames_np, bundle["device"], alpha=ALPHA)
        with torch.no_grad():
            logits = bundle["classifier"](slow_fast)
        return decode_logits(logits, bundle["idx_to_behavior"], anomaly_threshold)

    def _record_prediction(self, latency_ms: float) -> None:
        """Update runtime counters after a prediction."""
        self._total_latency_ms += latency_ms
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def predict_batch(self, *, data_path: str) -> PredictBatchResponse:  # pylint: disable=too-many-locals
        """Run the full detection + tracking + classification pipeline over a video file."""
        bundle = self._require_bundle()
        detector = bundle["detector"]

        cap = cv2.VideoCapture(data_path)
        if not cap.isOpened():
            raise InvalidVideoError(f"Cannot open video file: {data_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("predict_batch — video=%s total_frames=%d", data_path, total_frames)

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

            outputs = detector(frame)
            instances = outputs["instances"].to("cpu")
            detections = [
                {"bbox": box.tolist(), "score": float(score)}
                for box, score in zip(
                    instances.pred_boxes.tensor.numpy(),
                    instances.scores.numpy(),
                )
            ]

            tracks = tracker.update(detections)

            frame_detections: list[dict] = []
            for track in tracks:
                roi = extract_cow_roi(frame, track["bbox"])
                track_buffers[track["track_id"]].append(roi)

                if len(track_buffers[track["track_id"]]) >= CLIP_LENGTH:
                    frames_np = np.array(list(track_buffers[track["track_id"]]))
                    result = self._classify_clip(
                        frames_np, anomaly_threshold=DEFAULT_ANOMALY_THRESHOLD
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

                frame_detections.append({
                    "track_id": track["track_id"],
                    "bbox": track["bbox"],
                    "score": track["score"],
                    "behavior": result["prediction"],
                    "behavior_confidence": result["confidence"],
                    "is_anomaly": result["is_anomaly"],
                })

            all_frame_results.append({"frame": frame_idx, "detections": frame_detections})
            frame_idx += 1

        cap.release()

        self._record_prediction((time.perf_counter() - t0) * 1000)
        logger.info(
            "predict_batch done — %d frames processed, count=%d", frame_idx, self._predict_count
        )

        return PredictBatchResponse(
            model_id=MODEL_ID,
            predictions=all_frame_results,
            output_path=None,
        )

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
    ) -> PredictInlineResponse:
        """Classify a pre-cropped clip of one cow given as base64 frames."""
        self._require_bundle()
        frames_b64: list[str] = features["frames_base64"]
        anomaly_threshold = threshold if threshold is not None else DEFAULT_ANOMALY_THRESHOLD

        frames_b64_clip = frames_b64[-CLIP_LENGTH:]
        frames_np = decode_frames_base64(frames_b64_clip)

        t0 = time.perf_counter()
        result = self._classify_clip(frames_np, anomaly_threshold)
        self._record_prediction((time.perf_counter() - t0) * 1000)
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

    def train(self, *, data_path: str) -> PredictInlineResponse:
        """Training is not supported: the model uses externally trained artifacts (HTTP 501)."""
        _ = data_path
        raise TrainingNotSupportedError(
            "Este modelo usa artefactos externos; el reentrenamiento no está disponible."
        )

    def stats(self) -> StatsResponse:
        """Return model metadata and runtime statistics."""
        behaviors = list(self._bundle["idx_to_behavior"].values()) if self._bundle else []
        avg = self._total_latency_ms / self._predict_count if self._predict_count > 0 else None

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
                avg_latency_ms=round(avg, 1) if avg is not None else None,
            ),
        )
