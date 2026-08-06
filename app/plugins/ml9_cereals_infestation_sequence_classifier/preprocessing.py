"""Builds sliding windows from raw cereal-storage telemetry (batch CSV or inline row list).

Reproduces, step by step and in the same order, the delivered inference path
(src/predict/predictor.py::predict_with_bundle):

    merge_runtime_config_with_bundle -> validate_and_normalize_input_frame
    -> build_sequence_feature_frame -> align_feature_columns -> build_sliding_windows

Keeping this order matters: features are computed over *all* submitted rows before windows are cut,
so the amount of history supplied changes the result (manifest known_issues
sensibilidad_al_historial_aportado).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.plugins.ml9_cereals_infestation_sequence_classifier._vendor.model_bundle import (
    align_feature_columns,
    merge_runtime_config_with_bundle,
)
from app.plugins.ml9_cereals_infestation_sequence_classifier._vendor.preprocess import (
    build_sequence_feature_frame,
    build_sliding_windows,
    validate_and_normalize_input_frame,
)
from app.plugins.ml9_cereals_infestation_sequence_classifier.constants import RAW_FIXED_COLUMNS, TARGET_COLUMN
from app.plugins.ml9_cereals_infestation_sequence_classifier.model_loader import RUNTIME_CFG


def build_raw_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build a raw telemetry DataFrame from a list of row dicts (inline predict)."""
    return pd.DataFrame(rows)


def resolve_runtime_config(bundle: dict) -> dict[str, Any]:
    """Return the effective config for this bundle (training snapshot + local device settings)."""
    return merge_runtime_config_with_bundle(RUNTIME_CFG, bundle)


def missing_required_columns(raw_df: pd.DataFrame) -> list[str]:
    """Return the mandatory input columns absent from *raw_df* (empty list if the contract holds)."""
    return [col for col in RAW_FIXED_COLUMNS if col not in raw_df.columns]


def build_windows(
    raw_df: pd.DataFrame,
    bundle: dict,
    *,
    require_target: bool | None = None,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Validate the input frame and turn it into model-ready sliding windows.

    Returns (window_payload, runtime_cfg, has_target). ``window_payload`` is the dict produced by
    the vendored build_sliding_windows(): X_seq, y_seq, group_seq, timestamps_seq, window_meta,
    feature_columns.
    """
    runtime_cfg = resolve_runtime_config(bundle)
    df = validate_and_normalize_input_frame(raw_df, runtime_cfg, require_target=False)

    target_col = runtime_cfg["data"].get("target_column", TARGET_COLUMN)
    has_target = target_col in df.columns if require_target is None else bool(require_target)

    seq_df, feature_columns = build_sequence_feature_frame(df, runtime_cfg, require_target=has_target)
    _, feature_columns = align_feature_columns(
        seq_df.loc[:, feature_columns],
        bundle.get("feature_columns"),
    )
    payload = build_sliding_windows(
        seq_df,
        runtime_cfg,
        feature_columns=feature_columns,
        require_target=has_target,
    )
    return payload, runtime_cfg, has_target


def last_window_index(payload: dict[str, Any]) -> int:
    """Return the row index (in window_meta) of the most recent window across all series.

    predict_inline always scores the latest point of the submitted history. Windows are produced
    grouped by sample_id in input order, so "most recent" is resolved by timestamp_end, not by
    position.
    """
    meta = payload["window_meta"]
    if len(meta) == 0:
        return -1
    return int(pd.to_datetime(meta["timestamp_end"]).idxmax())
