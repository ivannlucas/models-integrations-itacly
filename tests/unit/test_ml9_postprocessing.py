"""Unit tests for ml9 postprocessing.build_predictions_frame.

Covers the xai_feature_values wiring predict_batch needs (previously entirely absent — see
predict_batch in plugin.py): each output row must carry the engineered features of its OWN
window, not some other window's — build_predictions_frame sorts rows by
(sample_id, timestamp_end, window_index) after computing xai_feature_values in the pre-sort
(x_seq-aligned) order, so a naive implementation could easily attach the wrong row's features
once the sort reorders things.
"""
import numpy as np
import pandas as pd

from app.plugins.ml9_cereals_infestation_sequence_classifier.postprocessing import (
    build_predictions_frame,
)

# x_seq[i] carries feature values [i, i*10] at its (only) step, so a window's
# xai_feature_values is a direct fingerprint of which original index i it came from.
_X_SEQ = np.array([[[i, i * 10]] for i in range(3)], dtype="float32")
_FEATURE_COLUMNS = ["a", "b"]

# Deliberately NOT in (sample_id, timestamp_end, window_index) order, so build_predictions_frame
# must actually reorder rows — window_meta[i] is metadata for the window at x_seq[i].
_WINDOW_META = pd.DataFrame({
    "sample_id": ["S1", "S1", "S1"],
    "timestamp_end": ["2026-01-03", "2026-01-01", "2026-01-02"],
    "window_index": [2, 0, 1],
})
_PROBA = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])


def test_xai_feature_values_present_and_aligned_after_sort():
    out = build_predictions_frame(
        _WINDOW_META, _PROBA, None, has_target=False,
        x_seq=_X_SEQ, feature_columns=_FEATURE_COLUMNS,
    )

    assert list(out["window_index"]) == [0, 1, 2]  # confirms the sort actually happened
    # Row for window_index=0 must carry x_seq[1]'s features (window_meta row 1 -> x_seq[1]).
    assert out.loc[out["window_index"] == 0, "xai_feature_values"].iloc[0] == {"a": 1.0, "b": 10.0}
    assert out.loc[out["window_index"] == 1, "xai_feature_values"].iloc[0] == {"a": 2.0, "b": 20.0}
    assert out.loc[out["window_index"] == 2, "xai_feature_values"].iloc[0] == {"a": 0.0, "b": 0.0}


def test_xai_feature_values_omitted_when_not_requested():
    """predict_inline doesn't go through this path — x_seq/feature_columns stay optional."""
    out = build_predictions_frame(_WINDOW_META, _PROBA, None, has_target=False)
    assert "xai_feature_values" not in out.columns
