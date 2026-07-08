"""Builds telemetry windows from a batch CSV or an inline list of raw telemetry rows."""
from __future__ import annotations

import pandas as pd

from app.plugins.ml46_dairy_fouling_clog_detection._vendor.common import TrainConfig, FeatureArtifacts
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.data import load_telemetry
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.datasets import AssetSequence, WindowDataset, make_sequences
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.features import build_feature_matrix, engineer_row_features
from app.plugins.ml46_dairy_fouling_clog_detection.constants import SEQ_LEN


def build_raw_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build a raw telemetry DataFrame from a list of row dicts (inline predict)."""
    return pd.DataFrame(rows)


def prepare_sequences(
    raw_df: pd.DataFrame,
    train_cfg: TrainConfig,
    feature_artifacts: FeatureArtifacts,
) -> tuple[dict[str, AssetSequence], list[int], list[str]]:
    """Run the full feature-engineering pipeline and group rows into per-asset AssetSequences.

    Returns (sequences, feature_indices_for_no_clock, asset_ids).
    """
    telemetry_df = load_telemetry(raw_df, train_cfg, require_targets=False)
    telemetry_df = engineer_row_features(telemetry_df)
    telemetry_df, full_feature_names, _ = build_feature_matrix(telemetry_df, feature_artifacts, [], train_cfg)
    sequences = make_sequences(telemetry_df, train_cfg, full_feature_names)

    feature_to_idx = {name: i for i, name in enumerate(full_feature_names)}
    feature_indices = [feature_to_idx[name] for name in feature_artifacts.no_clock_feature_names]
    asset_ids = sorted(telemetry_df["asset_id"].astype(str).unique().tolist())
    return sequences, feature_indices, asset_ids


def make_window_dataset(
    sequences: dict[str, AssetSequence],
    asset_ids: list[str],
    feature_indices: list[int],
    train_cfg: TrainConfig,
    stride: int | None = None,
) -> WindowDataset:
    """Build a WindowDataset over every valid window ending in production/no-maintenance."""
    return WindowDataset(sequences, asset_ids, feature_indices, train_cfg, stride=stride)


def last_window_only(
    sequences: dict[str, AssetSequence],
    feature_indices: list[int],
    seq_len: int = SEQ_LEN,
) -> tuple[str, int]:
    """Return the (sequence_id, end_idx) of the single most recent valid window across all sequences.

    Used by predict_inline, which always scores the latest point in the submitted history.
    """
    best: tuple[str, int, pd.Timestamp] | None = None
    for sequence_id, seq in sequences.items():
        meta = seq.meta
        valid_end = (meta["phase"] == "production") & (meta["maintenance_active"].fillna(0).astype(int) == 0)
        for end_idx in range(len(seq.timestamps) - 1, seq_len - 2, -1):
            if bool(valid_end.iloc[end_idx]):
                ts = pd.Timestamp(meta["timestamp"].iloc[end_idx])
                if best is None or ts > best[2]:
                    best = (sequence_id, end_idx, ts)
                break
    if best is None:
        return "", -1
    return best[0], best[1]
