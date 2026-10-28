"""Input handling for ml1: YOLO crop+align (faithful port of
src/data_processing/preprocess.py::yolo_crop_alignment) and sensor contract validation
(faithful port of src/predict/predictor.py::_validate_sensor_dimensions / _align_sensors).
"""
from __future__ import annotations

import base64
import binascii
import logging
from pathlib import Path

import cv2
import numpy as np

from app.domain.services.exceptions import DataContractError, InvalidImageError
from app.plugins.ml1_wine_pest_disease_detection.constants import (
    IGNORED_GAS_COLS,
    IMAGE_SIZE,
    SENSOR_COLS,
    YOLO_CONF_THRESHOLD,
)

logger = logging.getLogger(__name__)


def decode_image_base64(image_base64: str) -> np.ndarray:
    """Decode a base64 image string into a BGR array (cv2 convention), as cv2.imread would
    return for a file on disk — the format yolo_crop_alignment expects."""
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError(f"image_base64 no es base64 válido: {exc}") from exc
    array = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise InvalidImageError("No se pudo decodificar la imagen (formato no soportado o datos corruptos).")
    return img_bgr


def yolo_crop_alignment(img_bgr: np.ndarray, yolo_model, yolo_device: str) -> np.ndarray:
    """Faithful port of VidDataProcessor.yolo_crop_alignment, given an already-decoded BGR array
    instead of a path (the plugin decodes once; batch and inline share this function).

    Takes results[0].boxes.data[0] — the first box YOLO returns at conf>=0.3, not an explicit
    confidence argmax (see inbox/a01/manifest.yaml known_issues: the AI team's own reference
    script does take the max-confidence box explicitly — two different behaviours in their own
    delivery. This plugin replicates preprocess.py, the one predictor.py actually uses in
    production).
    """
    if yolo_model is not None:
        results = yolo_model.predict(img_bgr, conf=YOLO_CONF_THRESHOLD, verbose=False, device=yolo_device)
        if len(results) > 0 and len(results[0].boxes) > 0:
            box = results[0].boxes.data[0]
            x1, y1, x2, y2 = (int(v) for v in box[:4])
            img_bgr = img_bgr[max(0, y1):y2, max(0, x1):x2]
            if img_bgr.size == 0:
                raise InvalidImageError("El recorte detectado por YOLO está vacío (caja degenerada).")

    img_res = cv2.resize(img_bgr, IMAGE_SIZE)
    img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0


def validate_and_align_sensors(sensors: dict) -> np.ndarray:
    """Faithful port of predictor.py::_validate_sensor_dimensions + _align_sensors for the
    production (exclude_gases=true) configuration: exige coincidencia EXACTA con SENSOR_COLS,
    ignorando silenciosamente CO2_ppm/VOC_ppb si el cliente los manda (el modelo no los usa)."""
    received = {k: v for k, v in sensors.items() if k not in IGNORED_GAS_COLS}
    expected = set(SENSOR_COLS)
    received_keys = set(received.keys())
    if received_keys != expected:
        missing = expected - received_keys
        unexpected = received_keys - expected
        parts = []
        if missing:
            parts.append(f"Faltan sensores requeridos: {sorted(missing)}")
        if unexpected:
            parts.append(f"Columnas no reconocidas: {sorted(unexpected)}")
        raise DataContractError("; ".join(parts))
    try:
        aligned = np.array([float(received[col]) for col in SENSOR_COLS], dtype=np.float32).reshape(1, -1)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"Valor de sensor no numérico: {exc}") from exc
    if not np.all(np.isfinite(aligned)):
        # A finite Python float (e.g. 1e100) can still overflow to +/-inf once cast to float32
        # above — sklearn's RandomForest.predict() rejects non-finite input with a bare
        # ValueError that predict_batch's per-row except does not catch, aborting the whole
        # batch. Reject it here instead, as a DataContractError scoped to this row.
        bad_cols = [col for col, val in zip(SENSOR_COLS, aligned[0]) if not np.isfinite(val)]
        raise DataContractError(f"Valor de sensor no finito (inf/NaN o desbordamiento): {bad_cols}")
    return aligned


def resolve_image_path_for_batch(raw_value, search_dirs: list[Path]) -> Path | None:
    """predictor.py's own link_image_path logic (src/main.py::predict): N/A / empty / a filename
    not found under any of search_dirs all resolve to "no image" (tabular-only fallback), never
    an error — a missing photo for one batch row must not fail the whole row.

    search_dirs is searched in order (ZIP root and an images/ subfolder, if present, are both
    valid locations per predict_dto.py's documented contract — a photo at the root must not be
    ignored just because the ZIP also happens to have an images/ subfolder for other rows)."""
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text or text.upper() == "N/A":
        return None
    filename = Path(text).name
    for directory in search_dirs:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None
