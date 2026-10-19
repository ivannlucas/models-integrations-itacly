"""ml19: /train runs the official pipeline and publishes the new base (spec 2026-10-14)."""
import ast
import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.plugins.ml19_cereals_cost_forecast import mlflow_utils, training
from app.plugins.ml19_cereals_cost_forecast import plugin as plugin_mod
from app.plugins.ml19_cereals_cost_forecast._vendor import training_core as core
from app.plugins.ml19_cereals_cost_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
    DATASET_FE_FILENAME,
    MODEL_SELECTION_AUDIT_FILENAME,
    TRAINING_META_FILENAME,
)

CODE = Path(__file__).resolve().parents[2] / "inbox" / "a19" / "codigo"


def _dataset(n_months: int = 96, front: int = 3, seed: int = 0) -> pd.DataFrame:
    """dataset_v7_fe.csv-shaped frame: target_media plus features, the last rows without target."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-01-01", periods=n_months + front, freq="MS")
    price = 180 * np.exp(np.cumsum(rng.normal(0, 0.03, len(dates))))
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "target_media": price})
    nxt = df["target_media"].pct_change().shift(-1) * 100
    df["target_ret_next"] = nxt
    df["target_dir_next"] = (nxt > 0).astype(float)
    for i in range(6):   # features loosely related to next month's return, plus pure noise
        df[f"feat_{i}"] = nxt.fillna(0) * (0.5 if i < 3 else 0.0) + rng.normal(0, 1, len(df))
    df.loc[df.index[-front:], ["target_ret_next", "target_dir_next"]] = np.nan
    return df


@pytest.fixture
def fast(monkeypatch):
    """Ten trees instead of 200: same code path, seconds instead of a minute."""
    monkeypatch.setitem(core.XGB_PARAMS, "n_estimators", 10)
    monkeypatch.setitem(core.XGBC_PARAMS, "n_estimators", 10)
    monkeypatch.setitem(core.RF_PARAMS, "n_estimators", 10)


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "dataset_v7_fe.csv"
    _dataset().to_csv(path, index=False)
    return str(path)


def test_declares_its_base_folder():
    assert plugin_mod.Ml19CerealsCostForecastPlugin.BASE_ARTIFACT_FOLDER == ARTIFACT_FOLDER_NAME


def test_train_without_run_publishes_models_meta_and_dataset(csv, fast, isolated_base_store):
    resp = plugin_mod.Ml19CerealsCostForecastPlugin().train(data_path=csv, mlflow_run_id="")
    assert resp.base_version.version == "v1" and resp.upload_warning is None
    assert resp.train_end == "2019-12-31" and [h.horizon for h in resp.horizons] == [1, 2, 3]
    assert resp.data_end < "2022-01-01"   # rows without target only serve inference

    v1 = isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1"
    meta = json.loads((v1 / TRAINING_META_FILENAME).read_text())
    assert (v1 / MODEL_SELECTION_AUDIT_FILENAME).exists()
    for h, name in meta["best_reg"].items():
        artifact = pickle.loads((v1 / f"best_reg_h{h}m_{name.lower()}.pkl").read_bytes())
        assert artifact["features"] == meta["features"] and artifact["provenance"]["training_scope"] == "all_data"
    for h, name in meta["best_clf"].items():
        assert (v1 / f"best_clf_h{h}m_{name.lower()}.pkl").exists()
    # The training CSV, front rows included, becomes the inference dataset of the new base.
    assert (v1 / DATASET_FE_FILENAME).read_bytes() == Path(csv).read_bytes()
    assert resp.base_version.metrics["n_features"] == len(meta["features"])


def test_promoted_bundle_loads_and_predicts_the_latest_row(csv, fast, isolated_base_store):
    from app.plugins.ml19_cereals_cost_forecast import model_loader

    plugin_mod.Ml19CerealsCostForecastPlugin().train(data_path=csv, mlflow_run_id="")
    fresh = plugin_mod.Ml19CerealsCostForecastPlugin()
    fresh._bundle = model_loader.load_bundle_from_dir(  # pylint: disable=protected-access
        isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1")
    out = fresh.predict_inline(features={})
    assert out.date_label == pd.to_datetime(_dataset()["date"]).max().strftime("%Y-%m")
    assert len(out.horizons) == 3


def test_with_run_uploads_the_bundle_and_failure_is_a_warning(csv, fast, monkeypatch):
    tracker = MagicMock()
    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = plugin_mod.Ml19CerealsCostForecastPlugin().train(data_path=csv, mlflow_run_id="run-1")
    assert tracker.upload_artifacts.call_args.kwargs["artifact_path"] == "model"
    assert resp.base_version.mlflow_run_id == "run-1"

    tracker.upload_artifacts.side_effect = ConnectionError("mlflow caído")
    resp = plugin_mod.Ml19CerealsCostForecastPlugin().train(data_path=csv, mlflow_run_id="run-2")
    assert "mlflow caído" in resp.upload_warning and resp.base_version.version == "v2"


@pytest.mark.parametrize("drop", ["target_media", "target_ret_next", "date"])
def test_invalid_dataset_is_a_value_error_and_nothing_is_published(tmp_path, drop, isolated_base_store):
    path = tmp_path / "bad.csv"
    _dataset().drop(columns=[drop]).to_csv(path, index=False)
    with pytest.raises(ValueError, match=drop):
        plugin_mod.Ml19CerealsCostForecastPlugin().train(data_path=str(path), mlflow_run_id="")
    assert not (isolated_base_store / ARTIFACT_FOLDER_NAME / "versions").exists()


def test_dataset_without_walk_forward_period_is_rejected(tmp_path):
    path = tmp_path / "short.csv"
    _dataset(n_months=60).to_csv(path, index=False)   # 2014-01 .. 2018-12: nothing after TRAIN_END
    with pytest.raises(ValueError, match="2019-12-31"):
        training.validate_training_frame(pd.read_csv(path))


def test_predict_with_a_run_uses_that_runs_bundle(csv, fast, monkeypatch, tmp_path, isolated_base_store):
    plugin_mod.Ml19CerealsCostForecastPlugin().train(data_path=csv, mlflow_run_id="")
    v1 = isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1"
    tracker = MagicMock()
    tracker.download_artifacts.side_effect = lambda dst, artifact_path: str(v1)
    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker", MagicMock(return_value=tracker))
    plugin = plugin_mod.Ml19CerealsCostForecastPlugin()
    plugin._bundle = {"marker": "base"}  # pylint: disable=protected-access  # never used with a run
    out = plugin.predict_inline(features={}, mlflow_run_id="run-x")
    assert out.date_label == pd.to_datetime(_dataset()["date"]).max().strftime("%Y-%m")


@pytest.mark.skipif(not CODE.exists(), reason="inbox/a19/codigo (delivered code) not available")
def test_vendored_helpers_are_verbatim_copies_of_the_delivered_code():
    """The bodies in _vendor/training_core.py must match the original functions exactly."""
    vendored = {n.name: ast.dump(n) for n in ast.parse(Path(core.__file__).read_text()).body
                if isinstance(n, ast.FunctionDef)}
    sources = ("src/training/train.py", "src/training/feature_selection.py", "src/training/models.py",
               "src/training/walk_forward.py")
    original = {}
    for rel in sources:
        for node in ast.parse((CODE / rel).read_text(encoding="utf-8")).body:
            if isinstance(node, ast.FunctionDef) and node.name in vendored:
                original[node.name] = ast.dump(node)
    assert original.keys() == vendored.keys()
    assert {name for name in vendored if vendored[name] != original[name]} == set()
