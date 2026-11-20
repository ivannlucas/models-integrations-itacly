"""Tests for app/domain/services/shap_background.py and its use in the ml43/ml45 retrain upload.

Retraining refits scaler_x, so the SHAP background shipped with the base model (already scaled
with the base scaler_x) has to be re-expressed in the new scaler's space before it is uploaded
next to the retrained checkpoint — the original repos rebuild it from xai_background.csv with
the current scaler on every reprocessing.
"""
# pylint: disable=protected-access  # exercises the plugins' private _save_and_upload
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from app.domain.services.shap_background import rescale_background
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin import (
    Ml43CerealsDnslAnomalyFaultDetectionPlugin,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.plugin import (
    Ml45CerealsDnslCriticalPointDetectionPlugin,
)

N_WINDOWS, N_STEPS, N_FEATURES = 6, 12, 4


def _fitted(rows: np.ndarray) -> StandardScaler:
    return StandardScaler().fit(rows.reshape(-1, rows.shape[-1]))


def _scale(scaler: StandardScaler, windows: np.ndarray) -> np.ndarray:
    flat = scaler.transform(windows.reshape(-1, windows.shape[-1]))
    return flat.reshape(windows.shape).astype(np.float32)


@pytest.fixture(name="scenario")
def _scenario():
    """Raw background windows plus a base and a retrained scaler fitted on different data."""
    rng = np.random.default_rng(0)
    raw_background = rng.normal(50.0, 10.0, (N_WINDOWS, N_STEPS, N_FEATURES))
    base_scaler = _fitted(rng.normal(40.0, 5.0, (30, N_STEPS, N_FEATURES)))
    retrained_scaler = _fitted(rng.normal(70.0, 20.0, (30, N_STEPS, N_FEATURES)))
    return {
        "raw": raw_background,
        "base_scaler": base_scaler,
        "retrained_scaler": retrained_scaler,
        "shipped_background": _scale(base_scaler, raw_background),
    }


def test_rescale_matches_scaling_the_raw_background_with_the_new_scaler(scenario):
    """Same windows the original pipeline builds by scaling xai_background.csv anew."""
    rescaled = rescale_background(
        scenario["shipped_background"], scenario["base_scaler"], scenario["retrained_scaler"]
    )

    expected = _scale(scenario["retrained_scaler"], scenario["raw"])
    assert rescaled.shape == expected.shape
    assert rescaled.dtype == np.float32
    np.testing.assert_allclose(rescaled, expected, atol=1e-4)
    # And it really changed: the base model's copy is on a different scale.
    assert not np.allclose(scenario["shipped_background"], expected, atol=1e-2)


@pytest.mark.parametrize("missing", ["background", "fitted", "new"])
def test_rescale_is_a_no_op_when_something_is_missing(scenario, missing):
    background = None if missing == "background" else scenario["shipped_background"]
    fitted = None if missing == "fitted" else scenario["base_scaler"]
    new = None if missing == "new" else scenario["retrained_scaler"]

    assert rescale_background(background, fitted, new) is background


@pytest.mark.parametrize(
    "plugin_cls, plugin_module",
    [
        (
            Ml43CerealsDnslAnomalyFaultDetectionPlugin,
            "app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.plugin",
        ),
        (
            Ml45CerealsDnslCriticalPointDetectionPlugin,
            "app.plugins.ml45_cereals_dnsl_critical_point_detection.plugin",
        ),
    ],
)
def test_retrain_uploads_the_background_rescaled_with_the_new_scaler(
    scenario, plugin_cls, plugin_module
):
    """The xai_background.npy uploaded with a retrained model is in that model's input space."""
    plugin = plugin_cls()
    plugin._scaler_x = scenario["base_scaler"]
    plugin._xai_background = scenario["shipped_background"]

    uploaded = {}

    def _capture(local_dir, **_kwargs):
        uploaded["background"] = np.load(Path(local_dir) / "xai_background.npy")
        return "fake-run-id"

    with patch(f"{plugin_module}.upload_artifacts_to_mlflow", side_effect=_capture):
        run_id = plugin._save_and_upload(
            {"model_state_dict": {}, "model_cfg": {}},
            {"scaler_x": scenario["retrained_scaler"], "scaler_num": scenario["retrained_scaler"]},
            {},
            "",
        )

    assert run_id == "fake-run-id"
    np.testing.assert_allclose(
        uploaded["background"], _scale(scenario["retrained_scaler"], scenario["raw"]), atol=1e-4
    )
