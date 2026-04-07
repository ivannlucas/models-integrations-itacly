"""
ByteTrack implementation for cow tracking.
Vendored from the source repository (trackers/byte_tracker.py).
Based on: https://github.com/ifzhang/ByteTrack
"""
from collections import deque

import numpy as np


class KalmanFilter:
    """Simple Kalman filter for bounding box tracking."""

    def __init__(self) -> None:
        self.dt = 1.0
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = self.dt

        self.H = np.zeros((4, 8))
        for i in range(4):
            self.H[i, i] = 1.0

        self.P = np.eye(8) * 1000.0
        self.Q = np.eye(8) * 0.1
        self.R = np.eye(4) * 10.0
        self.x = np.zeros(8)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]

    def update(self, measurement: np.ndarray) -> np.ndarray:
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = measurement - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.x[:4]


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class Track:
    _count = 0

    def __init__(self, bbox: list[float], score: float) -> None:
        self.track_id = Track._count
        Track._count += 1

        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        x, y = x1 + w / 2, y1 + h / 2

        self.kalman = KalmanFilter()
        self.kalman.x[:4] = [x, y, w, h]

        self.bbox = bbox
        self.score = score
        self.state = TrackState.New
        self.time_since_update = 0
        self.hits = 1
        self.history: deque = deque(maxlen=30)
        self.history.append(bbox)

    def predict(self) -> None:
        x, y, w, h = self.kalman.predict()
        x1, y1 = x - w / 2, y - h / 2
        self.bbox = [x1, y1, x1 + w, y1 + h]

    def update(self, bbox: list[float], score: float) -> None:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        x, y = x1 + w / 2, y1 + h / 2
        self.kalman.update(np.array([x, y, w, h]))
        self.bbox = bbox
        self.score = score
        self.time_since_update = 0
        self.hits += 1
        self.state = TrackState.Tracked
        self.history.append(bbox)

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed


class ByteTracker:
    """ByteTrack: simple, fast, strong multi-object tracker."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate

        self.tracked_tracks: list[Track] = []
        self.frame_id = 0

    @staticmethod
    def iou(bbox1: list[float], bbox2: list[float]) -> float:
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _match(
        self, detections: list[dict], tracks: list[Track]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(tracks)))

        iou_matrix = np.zeros((len(detections), len(tracks)))
        for d, det in enumerate(detections):
            for t, track in enumerate(tracks):
                iou_matrix[d, t] = self.iou(det["bbox"], track.bbox)

        matches: list[tuple[int, int]] = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))

        while unmatched_dets and unmatched_tracks:
            best_iou = 0.0
            best_det, best_track = -1, -1
            for d in unmatched_dets:
                for t in unmatched_tracks:
                    if iou_matrix[d, t] > best_iou:
                        best_iou = iou_matrix[d, t]
                        best_det, best_track = d, t
            if best_iou >= self.match_thresh:
                matches.append((best_det, best_track))
                unmatched_dets.remove(best_det)
                unmatched_tracks.remove(best_track)
            else:
                break

        return matches, unmatched_dets, unmatched_tracks

    def update(self, detections: list[dict]) -> list[dict]:
        """Update tracker with new detections. Returns active track list."""
        self.frame_id += 1

        high_dets = [d for d in detections if d["score"] >= self.track_thresh]

        for track in self.tracked_tracks:
            track.predict()

        matches, unmatched_dets, unmatched_tracks = self._match(high_dets, self.tracked_tracks)

        for det_idx, track_idx in matches:
            self.tracked_tracks[track_idx].update(
                high_dets[det_idx]["bbox"], high_dets[det_idx]["score"]
            )

        for det_idx in unmatched_dets:
            new_track = Track(high_dets[det_idx]["bbox"], high_dets[det_idx]["score"])
            new_track.state = TrackState.Tracked
            self.tracked_tracks.append(new_track)

        removed_indices = []
        for i in unmatched_tracks:
            self.tracked_tracks[i].time_since_update += 1
            if self.tracked_tracks[i].time_since_update > self.track_buffer:
                self.tracked_tracks[i].mark_removed()
                removed_indices.append(i)
            else:
                self.tracked_tracks[i].mark_lost()

        self.tracked_tracks = [
            t for i, t in enumerate(self.tracked_tracks) if i not in removed_indices
        ]

        return [
            {"track_id": t.track_id, "bbox": t.bbox, "score": t.score}
            for t in self.tracked_tracks
            if t.state == TrackState.Tracked
        ]
