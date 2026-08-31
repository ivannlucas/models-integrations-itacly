"""Unit tests for ml16 preprocessing/postprocessing pure logic (no artifacts needed)."""
import numpy as np
import pandas as pd
import pytest

from app.domain.services.exceptions import InsufficientRowsError
from app.plugins.ml16_meat_raw_material_price_alert.postprocessing import run_inference
from app.plugins.ml16_meat_raw_material_price_alert.preprocessing import (
    create_endogenous_features,
    ensure_month_column,
    prepare_inference_input,
)


def test_ensure_month_column_derives_from_fecha_when_missing():
    """'month' must be derived from 'fecha' (day-1-of-month) when the caller omits it."""
    df = pd.DataFrame({"fecha": ["2024-03-01", "2024-04-01"]})
    out = ensure_month_column(df)
    assert list(out["month"]) == [3, 4]


def test_ensure_month_column_keeps_supplied_value():
    """An explicitly supplied 'month' column must be trusted as-is, not overwritten."""
    df = pd.DataFrame({"fecha": ["2024-03-01"], "month": [99]})
    out = ensure_month_column(df)
    assert out["month"].iloc[0] == 99


def test_create_endogenous_features_month_cyclical_encoding():
    """month_sin/month_cos must match the exact sin/cos(2*pi*month/12) formula (no phase shift)."""
    df = pd.DataFrame({
        "indice_animales": [100.0, 110.0, 120.0, 130.0],
        "indice_insumos": [50.0, 55.0, 60.0, 65.0],
        "animales_afectados": [0, 1, 2, 3],
        "precip_total": [10.0, 20.0, 30.0, 40.0],
        "month": [1, 2, 3, 4],
    })
    out = create_endogenous_features(df)
    np.testing.assert_allclose(out["month_sin"], np.sin(2 * np.pi * df["month"] / 12))
    np.testing.assert_allclose(out["month_cos"], np.cos(2 * np.pi * df["month"] / 12))


def test_create_endogenous_features_spread_is_ratio():
    df = pd.DataFrame({
        "indice_animales": [400.0], "indice_insumos": [200.0],
        "animales_afectados": [0], "precip_total": [0.0], "month": [1],
    })
    out = create_endogenous_features(df)
    assert out["spread"].iloc[0] == pytest.approx(2.0)


def test_prepare_inference_input_raises_insufficient_rows_for_short_history():
    """Fewer than warmup(6) + lookback(3) + 1 = 10 rows must raise InsufficientRowsError."""
    df = pd.DataFrame([{
        "fecha": f"2024-0{i}-01", "month": i, "indice_animales": 400.0, "indice_insumos": 200.0,
        "precip_total": 10.0, "precip_max": 5.0, "wet_days": 2.0, "wash_days": 1.0,
        "animales_afectados": 0,
    } for i in range(1, 5)])
    train_config = {
        "input_cols_per_target": {"target_animales": ["indice_animales"], "target_insumos": ["indice_insumos"]},
        "lookback": 3, "horizon": 4,
    }
    with pytest.raises(InsufficientRowsError):
        prepare_inference_input(df, train_config, scalers={})


def test_prepare_inference_input_missing_required_column_raises_value_error():
    df = pd.DataFrame({"fecha": ["2024-01-01"] * 10})  # missing indice_animales etc.
    train_config = {"input_cols_per_target": {}, "lookback": 3, "horizon": 4}
    with pytest.raises(ValueError):
        prepare_inference_input(df, train_config, scalers={})


class _FakeModel:
    """Stand-in for a fitted XGBoost/LogReg classifier."""

    def __init__(self, probas):
        self._probas = np.asarray(probas)

    def predict_proba(self, x):
        n = len(x)
        return np.stack([1 - self._probas[:n], self._probas[:n]], axis=1)


def test_run_inference_without_bagging_collapses_uncertainty_range_to_point_estimate():
    """No bagging_models entry (missing/failed artifact) must yield proba_low == proba_high == proba,
    matching predictor.py's graceful degradation without a confidence range."""
    models = {"target_animales": _FakeModel([0.2, 0.8])}
    results = run_inference(
        models=models,
        bagging_models={"target_animales": []},
        thresholds={"target_animales": 0.5},
        x_flat_per_target={"target_animales": np.zeros((2, 3))},
    )
    res = results["target_animales"]
    np.testing.assert_allclose(res["proba_low"], res["proba"])
    np.testing.assert_allclose(res["proba_high"], res["proba"])
    assert list(res["pred"]) == [0, 1]


def test_run_inference_with_bagging_widens_range_around_point_estimate():
    """proba_low/high must be widened to at least cover the point estimate (predictor.py takes
    min(percentile10, proba) / max(percentile90, proba))."""
    models = {"target_animales": _FakeModel([0.5])}
    bag_a = _FakeModel([0.3])
    bag_b = _FakeModel([0.7])
    results = run_inference(
        models=models,
        bagging_models={"target_animales": [bag_a, bag_b]},
        thresholds={"target_animales": 0.5},
        x_flat_per_target={"target_animales": np.zeros((1, 3))},
    )
    res = results["target_animales"]
    assert res["proba_low"][0] <= res["proba"][0] <= res["proba_high"][0]
