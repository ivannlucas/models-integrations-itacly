"""ml21: /train publishes the 6 models + metadata + the training CSV as panel (spec 2026-10-14)."""
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import sklearn.model_selection
from app.plugins.ml21_cereals_price_spatial import plugin as plugin_mod
from app.plugins.ml21_cereals_price_spatial import preprocessing
from app.plugins.ml21_cereals_price_spatial.constants import ARTIFACT_FOLDER_NAME, DATASET_FILENAME

PANEL_COLUMNS = ["date", "provincia", "cereal_predominante", "precio_provincial_lag_1", "x1"]


class _FastSearch:
    """GridSearchCV stand-in: fits the estimator once with few trees (same code path, seconds)."""

    def __init__(self, estimator, param_grid, **_kwargs):
        self.estimator = estimator
        self.best_params_ = {}
        self.best_score_ = 0.0

    def fit(self, X, y):
        if "n_estimators" in self.estimator.get_params():
            self.estimator.set_params(n_estimators=5)
        self.best_estimator_ = self.estimator.fit(X, y)
        return self


@pytest.fixture
def fast_training(monkeypatch):
    rng = np.random.default_rng(0)

    def fake_prepare(df, h, task):
        X = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.normal(size=60)})
        y = pd.Series(rng.normal(size=60)) if task == "regresion" else pd.Series([0, 1] * 30)
        return X.iloc[:48], X.iloc[48:], y.iloc[:48], y.iloc[48:]

    monkeypatch.setattr(preprocessing, "prepare_train_test", fake_prepare)
    monkeypatch.setattr(sklearn.model_selection, "GridSearchCV", _FastSearch)


@pytest.fixture
def instance():
    plugin = plugin_mod.Ml21CerealsPriceSpatialPlugin()
    plugin._panel_df = pd.DataFrame(columns=PANEL_COLUMNS)  # pylint: disable=protected-access
    return plugin


def _csv(tmp_path, columns=PANEL_COLUMNS):
    df = pd.DataFrame({c: [1.0, 2.0] for c in columns})
    if "date" in columns:
        df["date"] = ["2020-01-01", "2021-02-01"]
    for col in ("provincia", "cereal_predominante"):
        if col in columns:
            df[col] = ["Burgos", "Soria"]
    path = tmp_path / "panel.csv"
    df.to_csv(path, index=False)
    return path


def test_declares_its_base_folder():
    assert plugin_mod.Ml21CerealsPriceSpatialPlugin.BASE_ARTIFACT_FOLDER == ARTIFACT_FOLDER_NAME


def test_train_publishes_models_and_the_training_csv_as_panel(instance, fast_training, tmp_path,
                                                              isolated_base_store):
    csv = _csv(tmp_path)
    resp = instance.train(data_path=str(csv), mlflow_run_id="")
    assert resp.base_version.version == "v1" and resp.upload_warning is None
    v1 = isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1"
    raw_info = (v1 / "version.json").read_text()
    assert "NaN" not in raw_info and json.loads(raw_info)["metrics"]["mae_h1"] is not None
    assert (v1 / DATASET_FILENAME).read_bytes() == csv.read_bytes()
    metadata = json.loads((v1 / "model_metadata.json").read_text())
    assert set(metadata["selected_models"]) == {"H1", "H2", "H3"}
    for h in (1, 2, 3):
        assert (v1 / f"datagia_best_h{h}_reg.joblib").exists() and (v1 / f"datagia_best_h{h}_clf.joblib").exists()


def test_csv_without_the_panel_columns_is_rejected_before_training(instance, tmp_path, monkeypatch):
    monkeypatch.setattr(preprocessing, "prepare_train_test",
                        MagicMock(side_effect=AssertionError("must not train")))
    csv = _csv(tmp_path, columns=["date", "provincia", "cereal_predominante"])
    with pytest.raises(ValueError, match="precio_provincial_lag_1"):
        instance.train(data_path=str(csv), mlflow_run_id="")


def test_mlflow_failure_is_a_warning(instance, fast_training, tmp_path, monkeypatch):
    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = instance.train(data_path=str(_csv(tmp_path)), mlflow_run_id="run-1")
    assert "mlflow caído" in resp.upload_warning and resp.base_version.mlflow_run_id == "run-1"


def test_early_rejections_leave_no_temp_dir(instance, tmp_path, monkeypatch):
    import glob
    import tempfile
    pattern = f"{tempfile.gettempdir()}/ml21_train_*"
    before = set(glob.glob(pattern))
    for cols in (["date", "provincia", "cereal_predominante"], ["provincia", "cereal_predominante",
                                                                "precio_provincial_lag_1", "x1"]):
        with pytest.raises(ValueError):
            instance.train(data_path=str(_csv(tmp_path, columns=cols)), mlflow_run_id="")
    assert set(glob.glob(pattern)) == before


def test_mlflow_run_upload_excludes_the_training_csv(instance, fast_training, tmp_path,
                                                     monkeypatch, isolated_base_store):
    import os
    import shutil
    seen = {}

    def capture(directory, artifact_path=None):  # pylint: disable=unused-argument
        copy = tmp_path / "uploaded"
        shutil.copytree(directory, copy)
        seen["files"] = set(os.listdir(copy))

    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=capture))
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = instance.train(data_path=str(_csv(tmp_path)), mlflow_run_id="run-1")
    assert resp.upload_warning is None
    assert DATASET_FILENAME not in seen["files"]
    assert "model_metadata.json" in seen["files"] and len([f for f in seen["files"] if f.endswith(".joblib")]) == 6
    v1 = isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1"
    assert (v1 / DATASET_FILENAME).exists()
