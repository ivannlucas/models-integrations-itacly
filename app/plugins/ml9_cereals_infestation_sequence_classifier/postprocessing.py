"""Turns model probabilities into the ml9 per-window output contract.

There is NO business rule layer here: the delivered pipeline has none (manifest
outputs.postproceso_negocio: null). The delivered postprocess.py only reorders/sorts columns, and
that is exactly what this module does, plus the class-name mapping and JSON-safe scalars.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.plugins.ml9_cereals_infestation_sequence_classifier._vendor.sequential import (
    predict_sequence_proba,
    transform_sequences,
)
from app.plugins.ml9_cereals_infestation_sequence_classifier.constants import (
    CLASS_LABELS,
    GROUP_COLUMN,
    PROBA_FIELD_BY_CLASS,
)

_SORT_COLUMNS = [GROUP_COLUMN, "timestamp_end", "window_index"]


def clean_scalar(value: Any) -> Any:  # pylint: disable=too-many-return-statements
    """Convert numpy/pandas scalars into plain JSON-serializable Python values."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        fv = float(value)
        return None if not math.isfinite(fv) else fv
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NaT or (value is not None and not isinstance(value, (str, bool)) and pd.isna(value)):
        return None
    return value


def serialize_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe list[dict] records."""
    return [{k: clean_scalar(v) for k, v in rec.items()} for rec in df.to_dict(orient="records")]


def run_inference(checkpoint: dict, scaler: Any, x_seq: np.ndarray, *, batch_size: int = 128) -> np.ndarray:
    """Scale the windows with the fitted StandardScaler and return softmax probabilities.

    Same two steps as the delivered predictor's run_inference(): transform_sequences() flattens the
    windows to (n*T, n_features) before scaling, then the checkpoint's model produces softmax
    probabilities on CPU.
    """
    if len(x_seq) == 0:
        return np.zeros((0, len(CLASS_LABELS)), dtype=np.float32)
    x_scaled = transform_sequences(scaler, x_seq)
    return predict_sequence_proba(checkpoint["model"], x_scaled, batch_size=batch_size, device="cpu")


def label_for(pred_class: int) -> str:
    """Return the human-readable class name, falling back to the raw index if unknown.

    A user model retrained with a different number of classes may emit indices outside the
    delivered 0/1/2 mapping — those are reported as "clase_<n>" instead of crashing.
    """
    return CLASS_LABELS.get(int(pred_class), f"clase_{int(pred_class)}")


def build_predictions_frame(
    window_meta: pd.DataFrame,
    proba: np.ndarray,
    y_seq: np.ndarray | None,
    *,
    has_target: bool,
) -> pd.DataFrame:
    """Assemble one output row per window: identity + predicted class + per-class probabilities."""
    out = window_meta.reset_index(drop=True).copy()
    pred = np.argmax(proba, axis=1) if len(proba) else np.zeros((0,), dtype=int)

    out["pred_class"] = pred.astype(int)
    out["pred_label"] = [label_for(p) for p in pred]
    for class_idx in range(proba.shape[1] if proba.ndim == 2 else 0):
        field = PROBA_FIELD_BY_CLASS.get(class_idx, f"proba_clase_{class_idx}")
        out[field] = proba[:, class_idx]
    out["confidence"] = proba.max(axis=1) if len(proba) else np.zeros((0,), dtype=float)

    if has_target and y_seq is not None and np.size(y_seq):
        out["y_true"] = np.asarray(y_seq).astype(int)

    sort_cols = [c for c in _SORT_COLUMNS if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def window_feature_values(
    x_seq: np.ndarray,
    feature_columns: list[str],
    window_pos: int,
) -> dict[str, float]:
    """Return the engineered feature values of the LAST step of the window at *window_pos*.

    This is what the platform's explainability service consumes as `xai_feature_values`: the 65
    derived variables (diffs, slopes, rolling means/stds, interactions, cyclic hour/day) that the
    model actually saw, not the four raw sensor readings. Values are pre-scaling — the same units
    the features are computed in — so an attribution is readable by a human.
    """
    if window_pos < 0 or window_pos >= len(x_seq):
        return {}
    last_step = np.asarray(x_seq[window_pos])[-1, :]
    return {
        name: clean_scalar(float(value))
        for name, value in zip(feature_columns, last_step)
    }


def class_distribution(pred_df: pd.DataFrame) -> dict[str, int]:
    """Return the count of scored windows per predicted class label (operational monitoring signal).

    The memoria (§10) recommends watching sustained shifts >20% in this distribution as a practical
    drift signal, so it is returned with every batch prediction.
    """
    if len(pred_df) == 0:
        return {name: 0 for name in CLASS_LABELS.values()}
    counts = pred_df["pred_label"].value_counts().to_dict()
    return {name: int(counts.get(name, 0)) for name in CLASS_LABELS.values()} | {
        name: int(count) for name, count in counts.items() if name not in CLASS_LABELS.values()
    }
