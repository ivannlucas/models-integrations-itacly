"""ml17: every /train publishes the refit Ridge as the new base (spec 2026-10-14)."""
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest

from app.plugins.ml17_meat_market_price_analysis import plugin as plugin_mod
from app.plugins.ml17_meat_market_price_analysis.constants import (
    ARTIFACT_FOLDER_NAME,
    FEATURE_COLUMNS,
    MODEL_FILENAME,
)


@pytest.fixture
def train_csv(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    for col in FEATURE_COLUMNS:
        df[col] = rng.normal(100, 5, len(dates))
    df["month_sin"] = np.sin(2 * np.pi * dates.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dates.month / 12)
    path = tmp_path / "ml17.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_declares_its_base_folder():
    assert plugin_mod.Ml17MeatMarketPriceAnalysisPlugin.BASE_ARTIFACT_FOLDER == ARTIFACT_FOLDER_NAME


def test_train_without_run_publishes_the_new_base(train_csv, isolated_base_store, monkeypatch):
    tracker = MagicMock()
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = plugin_mod.Ml17MeatMarketPriceAnalysisPlugin().train(data_path=train_csv, mlflow_run_id="")
    assert resp.base_version.version == "v1" and resp.base_version.mlflow_run_id is None
    assert resp.base_version.metrics["mae"] == pytest.approx(resp.mae, abs=1e-4)
    tracker.upload_artifacts.assert_not_called()
    model = joblib.load(isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1" / MODEL_FILENAME)
    assert hasattr(model, "predict")


def test_train_with_run_also_uploads_to_mlflow(train_csv, monkeypatch):
    tracker = MagicMock()
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = plugin_mod.Ml17MeatMarketPriceAnalysisPlugin().train(data_path=train_csv, mlflow_run_id="run-1")
    assert tracker.upload_artifacts.call_args.kwargs["artifact_path"] == "model"
    tracker.log_params.assert_called_once()
    assert resp.base_version.mlflow_run_id == "run-1" and resp.upload_warning is None


def test_mlflow_failure_is_a_warning(train_csv, monkeypatch):
    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = plugin_mod.Ml17MeatMarketPriceAnalysisPlugin().train(data_path=train_csv, mlflow_run_id="run-1")
    assert "mlflow caído" in resp.upload_warning and resp.base_version.version == "v1"
