"""ml18: every /train publishes the retrained GRU as the new base (spec 2026-10-14)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.plugins.ml18_meat_spatial_price_forecast import mlflow_utils
from app.plugins.ml18_meat_spatial_price_forecast import plugin as plugin_mod
from app.plugins.ml18_meat_spatial_price_forecast import training
from app.plugins.ml18_meat_spatial_price_forecast.constants import ARTIFACT_FOLDER_NAME, MODEL_FILENAME

_METRICS = {
    **{f"{split}_{m}": 0.1 for split in ("train", "val", "test") for m in ("mae", "rmse", "mape_pct", "r2")},
    "n_train": 10, "n_val": 2, "n_test": 2, "epochs_run": 2, "best_epoch": 1, "training_time_s": 0.5,
}


@pytest.fixture
def fake_training(monkeypatch):
    monkeypatch.setattr(training, "train_gru", lambda path, on_epoch=None: SimpleNamespace(metrics=dict(_METRICS)))
    monkeypatch.setattr(training, "save_training_artifacts",
                        lambda directory, result: (directory / MODEL_FILENAME).write_bytes(b"gru"))


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "panel.csv"
    path.write_text("Fecha;CCAA\n")
    return str(path)


def test_declares_its_base_folder():
    assert plugin_mod.Ml18MeatSpatialPriceForecastPlugin.BASE_ARTIFACT_FOLDER == ARTIFACT_FOLDER_NAME


def test_train_without_run_publishes_the_new_base(fake_training, csv_path, isolated_base_store):
    resp = plugin_mod.Ml18MeatSpatialPriceForecastPlugin().train(data_path=csv_path, mlflow_run_id="")
    assert resp.base_version.version == "v1" and resp.upload_warning is None
    assert (isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1" / MODEL_FILENAME).read_bytes() == b"gru"


def test_mlflow_failure_is_a_warning(fake_training, csv_path, monkeypatch):
    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = plugin_mod.Ml18MeatSpatialPriceForecastPlugin().train(data_path=csv_path, mlflow_run_id="run-1")
    assert "mlflow caído" in resp.upload_warning and resp.base_version.mlflow_run_id == "run-1"
