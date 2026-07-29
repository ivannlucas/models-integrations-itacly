"""Unit tests for ml40 postprocessing.run_inference."""
import numpy as np
import pandas as pd

from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.postprocessing import run_inference


class _FakeScaler:
    """Stand-in for a fitted sklearn scaler."""

    feature_names_in_ = ["sensor_a", "sensor_b"]

    def transform(self, x):
        return np.asarray(x, dtype="float64") * 0.1 - 5.0


class _FakeModel:
    """Records the frame it was scored on so the test can inspect it."""

    feature_names_in_ = ["sensor_a", "sensor_b"]

    def __init__(self):
        self.seen = None

    def predict(self, x):
        self.seen = x.copy()
        return np.zeros(len(x), dtype=int)

    def predict_proba(self, x):
        return np.tile([0.5, 0.5], (len(x), 1))


def test_run_inference_scales_int_columns_without_truncation():
    """Scaled (float) values must land in full precision even when the source columns
    are int64 — assigning back in place without an explicit cast risks the scaled floats
    being silently corrupted against the original integer dtype."""
    df = pd.DataFrame({"sensor_a": [13, 23, 33], "sensor_b": [11, 21, 31]}).astype("int64")
    model = _FakeModel()
    scaler = _FakeScaler()

    run_inference(model, scaler, df, system="refrigeracion")

    expected = df.astype("float64") * 0.1 - 5.0
    pd.testing.assert_frame_equal(
        model.seen[["sensor_a", "sensor_b"]], expected, check_dtype=True
    )
