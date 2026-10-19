"""ml24: /train publishes the retrained horizons with their series (and exogenous) as the new base.

The CatBoost training itself is replaced by tiny real models (the full procedure runs in
test_base_reload_per_model.py against the delivered artifacts): these tests cover what /train
decides and publishes — split years, series and exogenous handling, MLflow run, errors.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

pytest.importorskip("catboost")

from catboost import CatBoostRegressor  # noqa: E402

from app.plugins.ml24_dairy_spatial_price_forecast import mlflow_utils, preprocessing, training  # noqa: E402
from app.plugins.ml24_dairy_spatial_price_forecast import plugin as plugin_mod  # noqa: E402
from app.plugins.ml24_dairy_spatial_price_forecast.constants import (  # noqa: E402
    ARTIFACT_FOLDER_NAME,
    EXOG_MONTHLY_FILENAME,
    HORIZONS,
    MODEL_MANIFEST_FILENAME,
    MODEL_PRODUCTS,
    SERIES_FILENAME,
    TARGET_METRIC,
    model_filename,
)


def _series(start="2018-01", months=36, ccaa=("CASTILLA Y LEON", "GALICIA")) -> pd.DataFrame:
    rows = []
    for name in ccaa:
        for period in pd.period_range(start, periods=months, freq="M"):
            for product in MODEL_PRODUCTS:
                rows.append({"anio": period.year, "mes": period.month, "ccaa": name, "producto": product,
                             "metrica": TARGET_METRIC, "valor": 0.7, "es_reconstruido": False})
            rows.append({"anio": period.year, "mes": period.month, "ccaa": name, "producto": "BATIDOS DE LECHE",
                         "metrica": TARGET_METRIC, "valor": 1.0, "es_reconstruido": False})
    return pd.DataFrame(rows)


def _exog(periods=("2018-01-01", "2018-02-01")) -> pd.DataFrame:
    return pd.DataFrame({"period_dt": pd.to_datetime(list(periods)), "exog__milk_price": [30.0, 31.0]})


def _split(n=10):
    return {"mae": 0.01, "smape": 2.0, "mae_baseline": 0.02, "smape_baseline": 3.0, "accuracy_pct": 98.0, "n": n}


@pytest.fixture
def fake_training(monkeypatch):
    """Record the call and return tiny real CatBoost models in the delivered bundle format."""
    calls = {}

    def fake(series, reference, *, validation_year, test_year, task_type=None):
        calls.update(series=series, reference=reference, validation_year=validation_year, test_year=test_year)
        result = training.TrainingResult(task_type="CPU", training_time_s=0.1)
        for h in HORIZONS:
            model = CatBoostRegressor(iterations=2, depth=1, verbose=False, allow_writing_files=False)
            model.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])
            result.horizons.append(training.HorizonResult(
                horizon=h, model=model,
                metadata={"feature_cols": ["x"], "selected_anchor_features": [], "selected_context_features": [],
                          "selected_exog": [], "baseline_w_persist_1": 0.5, "baseline_w_persist_12": 0.5,
                          "ml_alpha": 0.0},
                metrics={"train": _split(), "validation": _split(), "test": _split()},
                stage_summary={},
            ))
        result.metrics = {f"h{h}_test_smape": 2.0 for h in HORIZONS}
        return result

    monkeypatch.setattr(training, "train_spatial_price_model", fake)
    return calls


@pytest.fixture
def instance():
    plugin = plugin_mod.Ml24DairySpatialPriceForecastPlugin()
    plugin._bundle = {"marker": "base"}  # pylint: disable=protected-access
    plugin._reference = {"series": None, "spatial_weights": "W", "exog": _exog()}  # pylint: disable=protected-access
    return plugin


@pytest.fixture
def series_csv(tmp_path):
    path = tmp_path / "series.csv"
    _series().to_csv(path, index=False)
    return str(path)


def test_declares_its_base_folder():
    assert plugin_mod.Ml24DairySpatialPriceForecastPlugin.BASE_ARTIFACT_FOLDER == ARTIFACT_FOLDER_NAME


def test_default_split_years_use_the_last_complete_year():
    # 2018-01 .. 2020-06: 2020 is still in progress, so test = 2019 and validation = 2018.
    assert training.default_split_years(_series(months=30)) == (2018, 2019)
    with pytest.raises(ValueError, match="12 meses"):
        training.default_split_years(_series(months=11))


def test_train_publishes_models_and_the_filtered_series(instance, fake_training, series_csv, isolated_base_store):
    resp = instance.train(data_path=series_csv)
    assert (fake_training["validation_year"], fake_training["test_year"]) == (2019, 2020)   # derived
    assert resp.base_version.version == "v1" and resp.exog_renewed is False
    assert "exógenas" in resp.upload_warning

    v1 = isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1"
    assert (v1 / MODEL_MANIFEST_FILENAME).exists() and all((v1 / model_filename(h)).exists() for h in HORIZONS)
    published = pd.read_csv(v1 / SERIES_FILENAME)
    assert set(published["producto"]) == set(MODEL_PRODUCTS)   # only what the panels use
    assert len(published) == len(preprocessing.read_series_csv(series_csv))
    assert not (v1 / EXOG_MONTHLY_FILENAME).exists()           # inherited from the parent version
    assert fake_training["reference"]["exog"] is instance._reference["exog"]  # pylint: disable=protected-access


def test_explicit_years_are_respected(instance, fake_training, series_csv):
    instance.train(data_path=series_csv, validation_year=2018)
    assert (fake_training["validation_year"], fake_training["test_year"]) == (2018, 2019)
    instance.train(data_path=series_csv, test_year=2019)
    assert (fake_training["validation_year"], fake_training["test_year"]) == (2018, 2019)


def test_new_exogenous_block_is_trained_with_and_published(
    instance, fake_training, series_csv, tmp_path, isolated_base_store,
):
    exog_path = tmp_path / "input_costs_monthly.csv"
    _exog(("2018-01-01", "2026-09-01")).to_csv(exog_path, index=False)
    resp = instance.train(data_path=series_csv, exog_path=str(exog_path))
    assert resp.exog_renewed is True and resp.upload_warning is None
    assert fake_training["reference"]["exog"]["period_dt"].max() == pd.Timestamp("2026-09-01")
    v1 = isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1"
    assert (v1 / EXOG_MONTHLY_FILENAME).read_bytes() == Path(exog_path).read_bytes()


def test_exogenous_without_the_current_columns_is_rejected_before_training(
    instance, fake_training, series_csv, tmp_path, isolated_base_store,
):
    exog_path = tmp_path / "bad_exog.csv"
    pd.DataFrame({"period_dt": ["2018-01-01"], "other": [1.0]}).to_csv(exog_path, index=False)
    with pytest.raises(ValueError, match="exog__milk_price"):
        instance.train(data_path=series_csv, exog_path=str(exog_path))
    assert not fake_training
    assert not (isolated_base_store / ARTIFACT_FOLDER_NAME / "versions").exists()


def test_with_run_uploads_and_failure_is_a_warning(instance, fake_training, series_csv, monkeypatch):
    tracker = MagicMock()
    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = instance.train(data_path=series_csv, mlflow_run_id="run-1")
    assert tracker.upload_artifacts.call_args.kwargs["artifact_path"] == "model"
    assert resp.base_version.mlflow_run_id == "run-1"

    tracker.upload_artifacts.side_effect = ConnectionError("mlflow caído")
    resp = instance.train(data_path=series_csv, mlflow_run_id="run-2")
    assert "mlflow caído" in resp.upload_warning and resp.base_version.version == "v2"
