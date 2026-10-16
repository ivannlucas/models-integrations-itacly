"""ml16: every /train publishes the retrained bundle as the new base (spec 2026-10-14)."""
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.plugins.ml16_meat_raw_material_price_alert import plugin as plugin_mod
from app.plugins.ml16_meat_raw_material_price_alert import training
from app.plugins.ml16_meat_raw_material_price_alert.constants import (
    ARTIFACT_FOLDER_NAME,
    BAGGING_FILENAMES,
    MODEL_FILENAMES,
    RAW_REQUIRED_COLUMNS,
    SCALER_FILENAMES,
    TARGETS,
    TRAIN_CONFIG_FILENAME,
)

_METRICS = {"threshold": 0.5, "accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.65, "auc": 0.75}


@pytest.fixture
def instance(monkeypatch):
    monkeypatch.setattr(training, "train_models", lambda df: {
        "models": {t: f"model-{t}" for t in TARGETS},
        "scalers": {t: f"scaler-{t}" for t in TARGETS},
        "bagging_models": {t: [f"bag-{t}"] for t in TARGETS},
        "thresholds": {t: 0.5 for t in TARGETS},
        "input_cols_per_target": {t: ["indice_animales"] for t in TARGETS},
        "metrics": {t: dict(_METRICS) for t in TARGETS},
        "n_train": 40, "n_test": 12,
    })
    plugin = plugin_mod.Ml16MeatRawMaterialPriceAlertPlugin()
    plugin._models = {t: object() for t in TARGETS}       # pylint: disable=protected-access
    plugin._scalers = {t: object() for t in TARGETS}      # pylint: disable=protected-access
    plugin._train_config = {"horizon": 4}                 # pylint: disable=protected-access
    return plugin


@pytest.fixture
def train_csv(tmp_path):
    cols = [c for c in RAW_REQUIRED_COLUMNS if c != "month"] + list(TARGETS)
    df = pd.DataFrame({c: [0.0] * 3 for c in cols})
    df["fecha"] = ["2024-01-01", "2024-02-01", "2024-03-01"]
    path = tmp_path / "ml16.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_declares_its_base_folder():
    assert plugin_mod.Ml16MeatRawMaterialPriceAlertPlugin.BASE_ARTIFACT_FOLDER == ARTIFACT_FOLDER_NAME


def test_train_without_run_publishes_the_new_base(instance, train_csv, isolated_base_store):
    resp = instance.train(data_path=train_csv, mlflow_run_id="")
    assert resp.base_version.version == "v1" and resp.upload_warning is None
    assert resp.base_version.metrics["target_animales_f1"] == 0.65
    v1 = isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1"
    for target in TARGETS:
        for names in (MODEL_FILENAMES, SCALER_FILENAMES, BAGGING_FILENAMES):
            assert (v1 / names[target]).exists()
    assert json.loads((v1 / TRAIN_CONFIG_FILENAME).read_text())["horizon"] == 4


def test_train_with_run_uploads_and_failure_is_a_warning(instance, train_csv, monkeypatch):
    tracker = MagicMock()
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = instance.train(data_path=train_csv, mlflow_run_id="run-1")
    assert tracker.upload_artifacts.call_args.kwargs["artifact_path"] == "model"
    assert resp.base_version.mlflow_run_id == "run-1"

    tracker.upload_artifacts.side_effect = ConnectionError("mlflow caído")
    resp = instance.train(data_path=train_csv, mlflow_run_id="run-2")
    assert "mlflow caído" in resp.upload_warning and resp.base_version.version == "v2"
