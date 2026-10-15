"""Tests for ml15's real retraining code and prediction metadata (not the FakePlugin wiring).

training.py ports the AI team's frozen splits (features.py), trainer.py and get_stats.py. On the
delivered data/processed/model_ready_panel.csv it reproduces horizon_6_ridge.pkl and
models/metrics/final_evaluation.json exactly; that test needs inbox/a15/codigo (not committed) and
is skipped without it. The rest runs on small synthetic panels.
"""
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest

from app.plugins.ml15_wine_ipi_price_forecast import history, training
from app.plugins.ml15_wine_ipi_price_forecast import plugin as plugin_mod
from app.plugins.ml15_wine_ipi_price_forecast.constants import (
    ANCHOR_COLUMN, FEATURE_COLUMNS, MODEL_FILENAME, REFERENCE_FINANCIAL_FILENAME, REFERENCE_PRICES_FILENAME,
    TARGET_COLUMN, TEST_MONTHS, VALIDATION_MONTHS,
)

CODE = Path(__file__).resolve().parents[2] / "inbox" / "a15" / "codigo"


def _panel(n_months=60, start="2018-01-01", seed=0) -> pd.DataFrame:
    """Model-ready panel shaped like model_ready_panel.csv: target = IPI 6 months later."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_months, freq="MS")
    ipi = 100 + np.cumsum(rng.normal(0.2, 0.5, n_months + 6))
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    for col in FEATURE_COLUMNS:
        df[col] = rng.normal(100, 5, n_months)
    df[ANCHOR_COLUMN] = ipi[:n_months]
    target = ipi[6:n_months + 6].copy()
    target[-6:] = np.nan  # the last 6 origins have no observed target yet
    df[TARGET_COLUMN] = target
    return df


def test_split_follows_the_original_structure_anchored_to_the_last_origin():
    dates = training.split_dates(pd.Timestamp("2026-01-01"))
    assert dates == {
        "train_end": pd.Timestamp("2023-12-01"), "val_start": pd.Timestamp("2024-01-01"),
        "test_start": pd.Timestamp("2025-01-01"), "test_end": pd.Timestamp("2026-01-01"),
    }  # config.yaml::training of the delivered code
    assert (TEST_MONTHS, VALIDATION_MONTHS) == (13, 12)


def test_train_uses_only_rows_whose_target_is_before_validation():
    df = _panel()
    result = training.train_model(df)
    dates = training.split_dates(pd.Timestamp(df["date"].iloc[-1]))
    assert result["train_origin_end"] + pd.DateOffset(months=6) <= dates["train_end"]
    assert result["n_train"] == 60 - TEST_MONTHS - VALIDATION_MONTHS - 6
    assert result["n_test"] == TEST_MONTHS - 6  # test origins with an observed target
    assert result["metrics"]["rmse"] is not None and result["baseline_persistence"]["rmse"] is not None


def test_feature_nans_become_zero_and_extra_columns_are_ignored():
    df = _panel()
    df.loc[3, "copper_lag_14"] = np.nan
    df["unexpected_numeric"] = 1.0
    model = training.train_model(df)["model"]
    assert model.n_features_in_ == len(FEATURE_COLUMNS)


def test_no_evaluable_test_rows_gives_null_metrics_not_nan():
    df = _panel()
    df.loc[df.index[-TEST_MONTHS:], TARGET_COLUMN] = np.nan
    result = training.train_model(df)
    assert result["n_test"] == 0 and all(v is None for v in result["metrics"].values())


@pytest.mark.parametrize("bad,match", [("short", "Histórico insuficiente"), ("no_target", "ipi_national_t_plus_6")])
def test_invalid_training_csv_is_a_clear_error(bad, match):
    df = _panel(n_months=30) if bad == "short" else _panel().drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError, match=match):
        training.train_model(df)


def test_retraining_reproduces_the_delivered_artifact_and_final_evaluation():
    panel = CODE / "data" / "processed" / "model_ready_panel.csv"
    if not panel.exists():
        pytest.skip("inbox/a15/codigo not present")
    delivered = joblib.load(CODE / "models" / "artifacts" / MODEL_FILENAME)
    evaluation = json.loads((CODE / "models" / "metrics" / "final_evaluation.json").read_text())
    expected = evaluation["by_horizon"]["t+6"]["metrics"]

    result = training.train_model(pd.read_csv(panel))
    ours, theirs = result["model"], delivered["model"]
    assert result["n_train"] == delivered["n_training_rows"] == 85
    assert np.array_equal(ours.steps[-1][1].coef_, theirs.steps[-1][1].coef_)
    assert ours.steps[-1][1].intercept_ == theirs.steps[-1][1].intercept_
    for key in ("rmse", "mae", "mape_pct", "r2", "mda_pct"):
        assert result["metrics"][key] == pytest.approx(expected["model"][key], rel=1e-9)
        assert result["baseline_persistence"][key] == pytest.approx(expected["baseline_persistence"][key], rel=1e-9)


# ── plugin.train: MLflow only ──────────────────────────────────────────────────

@pytest.fixture
def instance():
    plugin = plugin_mod.Ml15WineIpiPriceForecastPlugin()
    plugin._payload = {  # pylint: disable=protected-access
        "model": object(), "feature_columns": list(FEATURE_COLUMNS), "target_column": TARGET_COLUMN,
        "anchor_column": ANCHOR_COLUMN, "horizon": 6, "model_name": "Ridge",
    }
    return plugin


def test_train_publishes_the_new_base_and_uploads_to_the_run(instance, monkeypatch, tmp_path, isolated_base_store):
    csv = tmp_path / "panel.csv"
    _panel().to_csv(csv, index=False)
    store = tmp_path / "mlflow"
    tracker = MagicMock()
    tracker.upload_artifacts.side_effect = lambda d, artifact_path: shutil.copytree(d, store / artifact_path)
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))

    resp = instance.train(data_path=str(csv), mlflow_run_id="run-1")
    assert resp.n_train_rows == 29 and resp.n_test_rows == 7 and resp.train_end_date == "2020-11-01"
    payload = joblib.load(store / "model" / MODEL_FILENAME)
    assert payload["feature_columns"] == list(FEATURE_COLUMNS) and payload["train_end_date"] == "2020-11-01"
    assert resp.base_version.version == "v1" and resp.base_version.mlflow_run_id == "run-1"
    v1 = isolated_base_store / "ml15_wine_ipi_price_forecast" / "versions" / "v1"
    assert joblib.load(v1 / MODEL_FILENAME)["train_end_date"] == "2020-11-01"
    assert "históricos exógenos" in resp.upload_warning   # no reference CSVs were given


def test_train_without_run_publishes_the_base(instance, tmp_path):
    csv = tmp_path / "panel.csv"
    _panel().to_csv(csv, index=False)
    resp = instance.train(data_path=str(csv), mlflow_run_id="")
    assert resp.base_version.version == "v1" and resp.base_version.mlflow_run_id is None


def test_mlflow_failure_is_a_warning(instance, monkeypatch, tmp_path):
    csv = tmp_path / "panel.csv"
    _panel().to_csv(csv, index=False)
    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = instance.train(data_path=str(csv), mlflow_run_id="run-1")
    assert "mlflow caído" in resp.upload_warning and resp.base_version.version == "v1"


@pytest.fixture
def reference_env(monkeypatch, tmp_path):
    """Synthetic 'current' reference CSVs + a tracked mkstemp dir (to detect leaked temp copies)."""
    dates = pd.date_range("2023-01-01", periods=6, freq="MS").strftime("%Y-%m-%d")
    current_prices = tmp_path / "cur_prices.csv"
    current_financial = tmp_path / "cur_fin.csv"
    pd.DataFrame({"date": dates, "spain": range(6)}).to_csv(current_prices, index=False)
    pd.DataFrame({"date": dates, "copper": range(6)}).to_csv(current_financial, index=False)
    monkeypatch.setattr(plugin_mod.model_loader, "reference_data_paths",
                        lambda: (str(current_prices), str(current_financial)))
    temp_dir = tmp_path / "tmpcopies"
    temp_dir.mkdir()
    real_mkstemp = plugin_mod.tempfile.mkstemp
    monkeypatch.setattr(
        plugin_mod.tempfile, "mkstemp",
        lambda suffix="", **kw: real_mkstemp(suffix=suffix, dir=str(temp_dir) if suffix == ".csv" else None, **kw),
    )
    return current_prices, current_financial, temp_dir


def test_reference_csvs_travel_with_the_new_base(instance, tmp_path, isolated_base_store, reference_env):
    current_prices, current_financial, temp_dir = reference_env
    csv = tmp_path / "panel.csv"
    _panel().to_csv(csv, index=False)

    resp = instance.train(data_path=str(csv), reference_prices_path=str(current_prices),
                          reference_financial_path=str(current_financial))
    v1 = isolated_base_store / "ml15_wine_ipi_price_forecast" / "versions" / "v1"
    assert (v1 / REFERENCE_PRICES_FILENAME).exists() and (v1 / REFERENCE_FINANCIAL_FILENAME).exists()
    assert resp.upload_warning is None
    assert list(temp_dir.iterdir()) == []


def test_reference_csv_with_wrong_columns_is_rejected_before_training(instance, tmp_path, monkeypatch, reference_env):
    current_prices, _, temp_dir = reference_env
    train_model = MagicMock(side_effect=AssertionError("must not train"))
    monkeypatch.setattr(plugin_mod.training, "train_model", train_model)
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-01-01"], "x": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="faltan las columnas"):
        instance.train(data_path=str(tmp_path / "panel.csv"), reference_prices_path=str(current_prices),
                       reference_financial_path=str(bad))
    train_model.assert_not_called()
    assert list(temp_dir.iterdir()) == []


def test_missing_reference_csv_is_a_value_error_naming_the_field(instance, tmp_path, reference_env):
    current_prices, _, temp_dir = reference_env
    with pytest.raises(ValueError, match="reference_financial_path"):
        instance.train(data_path=str(tmp_path / "panel.csv"), reference_prices_path=str(current_prices),
                       reference_financial_path=str(tmp_path / "nope.csv"))
    assert list(temp_dir.iterdir()) == []


def test_only_one_reference_csv_is_rejected(instance, tmp_path):
    with pytest.raises(ValueError, match="juntos"):
        instance.train(data_path=str(tmp_path / "panel.csv"), reference_prices_path="/tmp/p.csv")


# ── prediction metadata (y_true / feature_status / feature_notes) ─────────────

def _reference(monkeypatch):
    dates = pd.date_range("2023-01-01", periods=24, freq="MS")
    prices = pd.DataFrame({"date": dates, "spain": [100.0 + i for i in range(24)], "usa": [200.0] * 22 + [None, None]})
    fin_dates = pd.date_range("2020-01-01", periods=72, freq="MS")
    financial = pd.DataFrame({"date": fin_dates, "chem_sector_avg": 50.0, "copper_avg": 10.0,
                              "oil_brent_avg": 70.0, "eur_usd_avg": 1.1})
    monkeypatch.setattr(history, "load_reference_data",
                        lambda: (prices, history.reindex_financial_to_base_2020(financial)))


def test_prediction_metadata_follows_the_original(instance, monkeypatch):
    _reference(monkeypatch)
    model = MagicMock(predict=MagicMock(return_value=np.array([110.0])))
    instance._payload["model"] = model  # pylint: disable=protected-access

    known = instance.predict_inline(features={"origin_date": "2023-12"})
    assert known.origin_date == "2023-12-01" and known.y_true == 117.0 and known.feature_status == "exact"

    # usa for 2024-11 is missing -> last available (2024-10), like predictor.py's note.
    late = instance.predict_inline(features={"origin_date": "2024-12-01"})
    assert late.y_true is None and late.feature_status == "exogenous_last_available"
    assert late.feature_notes == (
        "usa_lag_1: se usa USA 2024-10-01 como último dato disponible para 2024-11-01; desfase=1 mes"
    )


def test_origin_date_wins_over_date_like_the_original(instance, monkeypatch):
    _reference(monkeypatch)
    instance._payload["model"] = MagicMock(predict=MagicMock(return_value=np.array([110.0])))  # pylint: disable=protected-access
    resp = instance.predict_inline(features={"origin_date": "2023-12-01", "date": "2023-08-01"})
    assert resp.origin_date == "2023-12-01"
