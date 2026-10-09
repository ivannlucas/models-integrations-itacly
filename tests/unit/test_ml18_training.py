"""Tests for ml18 /train — the real training code (Keras GRU), not the FakePlugin wiring.

Port of src/main.py::train() from inbox/a18/codigo. A full run on the delivered dataset reproduces
the original to 4 decimals (see outputs/a18/verification_report.md); here a small synthetic panel
shaped like dataset_modelado_desde_2008.csv runs the same code in seconds.
"""
import shutil
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.domain.services.exceptions import ModelPersistenceError, UserModelUnavailableError
from app.plugins.ml18_meat_spatial_price_forecast import constants, mlflow_utils, preprocessing, training
from app.plugins.ml18_meat_spatial_price_forecast import plugin as plugin_mod

pytest.importorskip("tensorflow")


def _panel(n_months=40, regions=("ANDALUCIA", "MURCIA", "BALEARES"), products=("CARNE POLLO",), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ccaa in regions:
        for prod in products:
            for i, fecha in enumerate(pd.date_range("2015-01-01", periods=n_months, freq="MS")):
                rows.append({
                    "Fecha": fecha.strftime("%Y-%m-%d"), "CCAA": ccaa, "Producto": prod,
                    "Poblacion": 1e6, "RentaHogar": 25000 + 10 * i, "CONSUMO X CAPITA": 1.0 + rng.normal(0, 0.05),
                    "PENETRACION (%)": 50.0, "PRECIO MEDIO KG": 4 + 0.02 * i + rng.normal(0, 0.05),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "panel.csv"
    _panel().to_csv(path, sep=";", index=False)
    return str(path)


@pytest.fixture
def fast(monkeypatch):
    """Two epochs instead of 60: same code path, seconds instead of minutes."""
    real = training.train_gru
    monkeypatch.setattr(training, "train_gru", lambda path, **kw: real(path, **{**kw, "epochs": 2}))


def test_hyperparameters_are_the_delivered_config():
    assert (constants.HIDDEN_UNITS, constants.DROPOUT, constants.DENSE_UNITS) == (96, 0.2, 32)
    assert (constants.LEARNING_RATE, constants.TRAIN_EPOCHS, constants.TRAIN_BATCH_SIZE) == (0.001, 60, 32)
    assert (constants.EARLY_STOPPING_PATIENCE, constants.TRAIN_RATIO, constants.VAL_RATIO) == (8, 0.70, 0.15)
    assert constants.LOOKBACK == 12


def test_split_is_temporal_and_scalers_fit_on_train_dates_only(csv_path):
    df = training.engineer_features(training.load_training_frame(csv_path))
    train_end, val_end = training.temporal_cutoffs(df)
    dates = sorted(df["Fecha"].unique())
    assert train_end == dates[int(40 * 0.70) - 1] and val_end == dates[int(40 * 0.85) - 1]
    scaler = training.fit_scaler_train_only(df, ["PRECIO MEDIO KG"], train_end)
    assert scaler.mean_[0] == pytest.approx(df.loc[df["Fecha"] <= train_end, "PRECIO MEDIO KG"].mean())


def test_every_window_is_built_and_split_by_target_date(csv_path):
    df = training.engineer_features(training.load_training_frame(csv_path))
    df[training.TARGET_SCALED] = df["PRECIO MEDIO KG"]
    x, _, dates = training.build_training_sequences(df)
    assert x.shape == (3 * (40 - 12), 12, len(constants.FEATURE_COLUMNS))
    train_end, val_end = training.temporal_cutoffs(df)
    parts = training.split_by_date(x, np.zeros(len(x)), dates, train_end, val_end)
    assert sum(len(p) for p in parts[::2]) == len(x)


def test_mojibake_price_header_is_resolved_like_the_original(tmp_path):
    path = tmp_path / "mojibake.csv"
    _panel().rename(columns={"PRECIO MEDIO KG": "PRECIO MEDIO KG."}).to_csv(path, sep=";", index=False)
    df = training.load_training_frame(str(path))
    assert df["PRECIO MEDIO KG"].notna().all()
    assert preprocessing.resolve_target_column(pd.DataFrame({"x": [1]})) is False


@pytest.mark.parametrize("months,match", [(3, "fechas distintas"), (12, "ninguna secuencia"), (14, "Split temporal")])
def test_too_short_history_is_a_clear_error(tmp_path, months, match):
    path = tmp_path / "short.csv"
    _panel(n_months=months).to_csv(path, sep=";", index=False)
    with pytest.raises(ValueError, match=match):
        training.train_gru(str(path), epochs=1)


def test_train_uploads_only_to_mlflow_and_the_model_is_served_by_run(csv_path, fast, monkeypatch, tmp_path):
    captured = tmp_path / "mlflow_store"
    tracker = MagicMock()
    tracker.upload_artifacts.side_effect = lambda d, artifact_path: shutil.copytree(d, captured / artifact_path)
    tracker.download_artifacts.side_effect = (
        lambda dest, artifact_path: str(shutil.copytree(captured / artifact_path, f"{dest}/{artifact_path}"))
    )
    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker", MagicMock(return_value=tracker))
    local_before = set(plugin_mod.model_loader._store.local_dir.rglob("*"))  # pylint: disable=protected-access

    instance = plugin_mod.Ml18MeatSpatialPriceForecastPlugin()   # no base model loaded: train never needs it
    resp = instance.train(data_path=csv_path, mlflow_run_id="run-1")
    assert resp.mlflow_run_id == "run-1" and resp.upload_warning is None
    assert resp.epochs_run == 2 and resp.n_test > 0
    assert set(plugin_mod.model_loader._store.local_dir.rglob("*")) == local_before  # pylint: disable=protected-access
    assert {"model.joblib", "x_scaler.json", "y_scaler.json", "metrics.json"} <= {p.name for p in (captured / "model").iterdir()}

    rows = _panel().query("CCAA == 'ANDALUCIA'").tail(12).to_dict("records")
    preds = instance.predict_inline(features={"rows": rows}, mlflow_run_id="run-1").predictions
    assert preds[0]["CCAA"] == "ANDALUCIA" and np.isfinite(preds[0]["predicted_price"])


def test_train_fails_when_mlflow_upload_fails(csv_path, fast, monkeypatch):
    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker", MagicMock(return_value=tracker))
    with pytest.raises(ModelPersistenceError, match="mlflow caído"):
        plugin_mod.Ml18MeatSpatialPriceForecastPlugin().train(data_path=csv_path, mlflow_run_id="run-1")


def test_predict_with_a_run_without_model_never_uses_the_base(monkeypatch):
    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker",
                        MagicMock(return_value=MagicMock(download_artifacts=MagicMock(return_value=""))))
    instance = plugin_mod.Ml18MeatSpatialPriceForecastPlugin()
    instance._bundle = {"model": object()}  # pylint: disable=protected-access  # a base model is loaded
    with pytest.raises(UserModelUnavailableError, match="run-vacio"):
        instance.predict_inline(features={"rows": []}, mlflow_run_id="run-vacio")
