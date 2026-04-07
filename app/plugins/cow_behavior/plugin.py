import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
import torch

from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import InsufficientFramesError, InvalidImageError, InvalidVideoError
from app.plugins.cow_behavior.byte_tracker import ByteTracker
from app.plugins.cow_behavior.model_loader import (
    ALPHA,
    CLASSIFIER_FILENAME,
    CLIP_LENGTH,
    CROP_SIZE,
    DETECTOR_FILENAME,
    load_artifacts,
)
from app.plugins.cow_behavior.postprocessing import decode_logits
from app.plugins.cow_behavior.preprocessing import (
    decode_frames_base64,
    extract_cow_roi,
    prepare_slowfast_tensor,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "cow-behavior"
MODEL_VERSION = "1.0.0"


class CowBehaviorPlugin(ModelPluginPort):
    def __init__(self) -> None:
        self._detector: Any = None
        self._classifier: Any = None
        self._behavior_to_idx: dict[str, int] = {}
        self._idx_to_behavior: dict[int, str] = {}
        self._num_classes: int = 0
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded: bool = False
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        (
            self._detector,
            self._classifier,
            self._behavior_to_idx,
            self._idx_to_behavior,
            self._num_classes,
        ) = load_artifacts()
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def _classify_clip(self, frames_np: np.ndarray, anomaly_threshold: float) -> dict:
        slow_fast = prepare_slowfast_tensor(frames_np, self._device, alpha=ALPHA)
        with torch.no_grad():
            logits = self._classifier(slow_fast)
        return decode_logits(logits, self._idx_to_behavior, anomaly_threshold)

    def predict_batch(self, *, data_path: str) -> dict:
        cap = cv2.VideoCapture(data_path)
        if not cap.isOpened():
            raise InvalidVideoError(f"Cannot open video file: {data_path}")

        tracker = ByteTracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
        track_buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=CLIP_LENGTH))
        track_last_result: dict[int, dict] = {}

        all_frame_results: list[dict] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            outputs = self._detector(frame)
            instances = outputs["instances"].to("cpu")
            detections = [
                {"bbox": box.tolist(), "score": float(score)}
                for box, score in zip(instances.pred_boxes.tensor.numpy(), instances.scores.numpy())
            ]
            tracks = tracker.update(detections)

            frame_detections: list[dict] = []
            for track in tracks:
                roi = extract_cow_roi(frame, track["bbox"])
                track_buffers[track["track_id"]].append(roi)
                if len(track_buffers[track["track_id"]]) >= CLIP_LENGTH:
                    frames_np = np.array(list(track_buffers[track["track_id"]]))
                    result = self._classify_clip(frames_np, anomaly_threshold=0.5)
                    track_last_result[track["track_id"]] = result
                else:
                    result = track_last_result.get(
                        track["track_id"],
                        {"prediction": list(self._behavior_to_idx.keys())[0] if self._behavior_to_idx else "unknown",
                         "confidence": 1.0, "is_anomaly": False, "behavior_idx": 0},
                    )
                frame_detections.append({
                    "track_id": track["track_id"], "bbox": track["bbox"],
                    "score": track["score"], "behavior": result["prediction"],
                    "behavior_confidence": result["confidence"], "is_anomaly": result["is_anomaly"],
                })
            all_frame_results.append({"frame": frame_idx, "detections": frame_detections})
            frame_idx += 1

        cap.release()
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info("predict_batch done — %d frames, count=%d", frame_idx, self._predict_count)

        return {"model_id": MODEL_NAME, "predictions": all_frame_results, "output_path": None}

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
    ) -> dict:
        frames_b64: list[str] = features["frames_base64"]
        anomaly_threshold = threshold if threshold is not None else 0.5

        if len(frames_b64) < CLIP_LENGTH:
            raise InsufficientFramesError(
                f"At least {CLIP_LENGTH} frames required, got {len(frames_b64)}."
            )

        frames_b64_clip = frames_b64[-CLIP_LENGTH:]
        frames_np = decode_frames_base64(frames_b64_clip)
        result = self._classify_clip(frames_np, anomaly_threshold)

        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info(
            "predict_inline done — behavior=%s confidence=%.4f count=%d",
            result["prediction"], result["confidence"], self._predict_count,
        )

        return {
            "model_id": MODEL_NAME,
            "threshold": anomaly_threshold,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "features_used": ["frames_base64"],
            "is_anomaly": result["is_anomaly"],
            "behavior_idx": result["behavior_idx"],
        }

    def stats(self) -> StatsResponse:
        behaviors = list(self._idx_to_behavior.values()) if self._loaded else []
        return StatsResponse(
            model_name=MODEL_NAME,
            model_type="Faster R-CNN ResNet-101 (detector) + ByteTrack + SlowFast R50 (classifier)",
            framework="torch + detectron2 + pytorchvideo",
            artifact_path=f"model-runtime-cow_behavior/artifacts/{DETECTOR_FILENAME} + {CLASSIFIER_FILENAME}",
            input_schema={
                "mode=inline": {
                    "frames_base64": f"list[str] — at least {CLIP_LENGTH} base64 JPEG/PNG frames",
                    "threshold": "float | null — anomaly threshold override (default 0.5)",
                },
                "mode=batch": {"data_path": "str — path to MP4/AVI video"},
            },
            output_schema={
                "batch": {"predictions": "list[{frame, detections: list[{track_id, bbox, behavior, ...}]}]"},
                "inline": {
                    "prediction": f"str — one of {behaviors}",
                    "confidence": "float", "is_anomaly": "bool",
                },
            },
            predict_count=self._predict_count,
            last_predict_at=self._last_predict_at,
        )
