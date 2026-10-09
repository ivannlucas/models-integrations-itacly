"""Tests for ml14 /train — the real training code (PyTorch GRU), not the FakePlugin wiring.

Port of the GRU part of src/training/compare_models.py from inbox/a14/codigo. On the delivered
dataset it reproduces model_comparison.json to 4 decimals and the delivered gru_model.pt to 2e-7
(see outputs/a14/verification_report.md); here a small synthetic weekly series runs the same code
in seconds.
"""
import shutil
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.domain.services.exceptions import ModelPersistenceError, UserModelUnavailableError
from app.plugins.ml14_wine_phyto_price_forecast import constants, mlflow_utils, training
from app.plugins.ml14_wine_phyto_price_forecast import plugin as plugin_mod
from app.plugins.ml14_wine_phyto_price_forecast.feature_engineering import build_modeling_features


def _raw(n_weeks=140, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_weeks)
    return pd.DataFrame({
        "date": pd.date_range("2018-01-07", periods=n_weeks, freq="W-SUN").strftime("%Y-%m-%d"),
        "PROTECCION_FITO": 100 + 0.1 * t + 2 * np.sin(t / 8) + rng.normal(0, 0.3, n_weeks),
        "CARBURANTES": 120 + rng.normal(0, 2, n_weeks),
        "COPPER_EUR_TON": 7000 + 10 * t + rng.normal(0, 50, n_weeks),
        "GAS_EUR_MMBTU": 10 + rng.normal(0, 0.5, n_weeks),
        "COIL_EUR_BARRIL": 60 + rng.normal(0, 2, n_weeks),
        "DEXUSEU": 1.1 + rng.normal(0, 0.01, n_weeks),
    })


def _features():
    return [c for c in build_modeling_features(_raw().to_dict("records")).columns if c != "date"]


@pytest.fixture
def raw_csv(tmp_path):
    path = tmp_path / "raw.csv"
    _raw().to_csv(path, index=False)
    return str(path)


@pytest.fixture
def instance():
    """The real plugin with only the feature contract loaded (the fixed artifacts live in S3)."""
    plugin = plugin_mod.Ml14WinePhytoPriceForecastPlugin()
    plugin._bundle = {"feature_columns": _features(), "model": object()}  # pylint: disable=protected-access
    return plugin


@pytest.fixture
def fast(monkeypatch):
    real = training.train_gru
    monkeypatch.setattr(training, "train_gru", lambda path, feats, **kw: real(path, feats, **{**kw, "epochs": 2}))


def test_hyperparameters_are_the_delivered_config():
    assert (constants.TRAIN_EPOCHS, constants.TRAIN_BATCH_SIZE, constants.TRAIN_LR) == (150, 32, 1e-3)
    assert (constants.TRAIN_PATIENCE, constants.TRAIN_N_SEEDS, constants.TRAIN_SEED_BASE) == (20, 3, 42)
    assert (constants.SEQ_LEN, constants.HORIZON_WEEKS, constants.TEST_RATIO) == (12, 16, 0.2)


def test_both_input_formats_give_the_same_modelling_frame(raw_csv, tmp_path):
    feats = _features()
    from_raw = training.load_training_frame(raw_csv, feats)
    final = tmp_path / "final.csv"
    from_raw.assign(FECHA=from_raw["FECHA"].dt.strftime("%Y-%m-%d")).to_csv(final, index=False)
    # Same values; only MONTH comes as int32 from feature engineering vs int64 from the CSV (both scaled as float).
    pd.testing.assert_frame_equal(training.load_training_frame(str(final), feats), from_raw, check_dtype=False)


def test_split_and_sequences_follow_the_original(raw_csv):
    df = training.prepare_horizon_dataset(training.load_training_frame(raw_csv, _features()))
    train_df, test_df = training.split_temporal(df)
    assert train_df["FECHA"].max() < test_df["FECHA"].min()
    assert len(test_df) == int(df["FECHA"].nunique() * 0.2)
    cols = [c for c in df.columns if c not in training.EXCLUDE_COLS]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train_df[cols].values)
    x, *_ = training.build_test_sequences_with_bridge(train_df, test_df, cols, scaler)
    assert x.shape == (len(test_df), 12, len(cols))   # the bridge gives one window per test row


def test_wrong_csv_is_a_clear_error(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="FECHA"):
        training.load_training_frame(str(path), _features())


def test_train_uploads_only_to_mlflow_and_the_model_is_served_by_run(instance, raw_csv, fast, monkeypatch, tmp_path):
    store = tmp_path / "mlflow_store"
    tracker = MagicMock()
    tracker.upload_artifacts.side_effect = lambda d, artifact_path: shutil.copytree(d, store / artifact_path)
    tracker.download_artifacts.side_effect = (
        lambda dest, artifact_path: str(shutil.copytree(store / artifact_path, f"{dest}/{artifact_path}"))
    )
    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker", MagicMock(return_value=tracker))
    local_dir = plugin_mod.model_loader._store.local_dir  # pylint: disable=protected-access
    before = set(local_dir.rglob("*")) if local_dir.exists() else set()

    resp = instance.train(data_path=raw_csv, mlflow_run_id="run-1")
    assert resp.mlflow_run_id == "run-1" and resp.n_seeds == 3 and resp.n_test > 0
    assert {p.name for p in (store / "model").iterdir()} == {"gru_model.pt", "scalers.json"}
    assert (set(local_dir.rglob("*")) if local_dir.exists() else set()) == before

    out = instance.predict_inline(features={"rows": _raw().tail(30).to_dict("records")}, mlflow_run_id="run-1")
    assert np.isfinite(out.predicted_price)


def test_train_fails_when_mlflow_upload_fails(instance, raw_csv, fast, monkeypatch):
    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker", MagicMock(return_value=tracker))
    with pytest.raises(ModelPersistenceError, match="mlflow caído"):
        instance.train(data_path=raw_csv, mlflow_run_id="run-1")


def test_predict_with_a_run_without_model_never_uses_the_base(instance, monkeypatch):
    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker",
                        MagicMock(return_value=MagicMock(download_artifacts=MagicMock(return_value=""))))
    with pytest.raises(UserModelUnavailableError, match="run-vacio"):
        instance.predict_inline(features={"rows": []}, mlflow_run_id="run-vacio")
