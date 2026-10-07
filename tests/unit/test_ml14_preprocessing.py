"""Pure-logic tests for ml14's inference core (preprocessing.py::run_inference).

Requires torch (not installed in every sandbox) since run_inference builds a real tensor —
same constraint as other torch-backed plugin tests in this repo.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from app.domain.services.exceptions import InsufficientDataError
from app.plugins.ml14_wine_phyto_price_forecast import preprocessing


class _ZeroModel(torch.nn.Module):
    """Fake model: always predicts residual=0, so predicted_price == drift_baseline."""

    def forward(self, x):  # noqa: D102
        return torch.zeros(x.shape[0])


def _bundle(feature_columns: list[str]) -> dict:
    return {
        "model": _ZeroModel(),
        "feature_columns": feature_columns,
        "input_scaler_mean": np.zeros(len(feature_columns), dtype=np.float32),
        "input_scaler_scale": np.ones(len(feature_columns), dtype=np.float32),
        "target_scaler_mean": 0.0,
        "target_scaler_scale": 1.0,
    }


def _rows(n: int = 24, overflow_on_last: bool = False) -> list[dict]:
    base = {
        "PROTECCION_FITO": 125.0, "CARBURANTES": 130.0, "COPPER_EUR_TON": 8000.0,
        "GAS_EUR_MMBTU": 10.0, "COIL_EUR_BARRIL": 55.0, "DEXUSEU": 1.15,
    }
    start = pd.Timestamp("2025-05-18")
    rows = []
    for i in range(n):
        row = dict(base)
        row["date"] = (start + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d")
        rows.append(row)
    if overflow_on_last:
        rows[-1]["COPPER_EUR_TON"] = 1e250  # finite in float64, overflows to inf in float32
    return rows


def test_run_inference_normal_case_returns_finite_price():
    from app.plugins.ml14_wine_phyto_price_forecast.feature_engineering import build_modeling_features

    rows = _rows()
    feature_columns = list(build_modeling_features(rows).columns.difference(["date"]))
    result = preprocessing.run_inference(_bundle(feature_columns), rows)
    assert np.isfinite(result["predicted_price"])


def test_run_inference_rejects_float32_overflow_instead_of_returning_null():
    """A finite-but-huge value (1e250) overflows to inf when cast to float32 for the model
    input. Without a guard, this used to silently return predicted_price=null with HTTP 200
    (Pydantic serializes non-finite floats as null) instead of a clear 422."""
    from app.plugins.ml14_wine_phyto_price_forecast.feature_engineering import build_modeling_features

    rows = _rows(overflow_on_last=True)
    feature_columns = list(build_modeling_features(rows).columns.difference(["date"]))
    with pytest.raises(InsufficientDataError, match="desborda"):
        preprocessing.run_inference(_bundle(feature_columns), rows)
