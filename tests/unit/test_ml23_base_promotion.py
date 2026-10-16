"""ml23: every /train publishes the refit GRU as the new base (spec 2026-10-14)."""
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from app.plugins.ml23_lactic_market_price_forecast import mlflow_utils  # noqa: E402
from app.plugins.ml23_lactic_market_price_forecast import plugin as plugin_mod  # noqa: E402
from app.plugins.ml23_lactic_market_price_forecast.constants import (  # noqa: E402
    ARTIFACT_FOLDER_NAME,
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    SCALER_FILENAME,
)


@pytest.fixture
def train_csv(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for producto, canal in (("LECHE ENTERA", "HIPER"), ("LECHE DESNATADA", "SUPER")):
        for i, fecha in enumerate(pd.date_range("2015-01-01", periods=60, freq="MS")):
            rows.append({"fecha": fecha.strftime("%Y-%m-%d"), "producto": producto, "canal": canal,
                         "target_precio_medio": 0.8 + 0.002 * i + rng.normal(0, 0.01),
                         "ipc_general_lag1": 100 + 0.1 * i, "mes": fecha.month})
    path = tmp_path / "ml23.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(plugin_mod, "TRAIN_EPOCHS", 2)
    monkeypatch.setattr(plugin_mod, "TRAIN_N_SEEDS", 1)


def test_declares_its_base_folder():
    assert plugin_mod.Ml23LacticMarketPriceForecastPlugin.BASE_ARTIFACT_FOLDER == ARTIFACT_FOLDER_NAME


def test_train_without_run_publishes_the_new_base(train_csv, fast, isolated_base_store):
    resp = plugin_mod.Ml23LacticMarketPriceForecastPlugin().train(data_path=train_csv, mlflow_run_id="")
    assert resp.base_version.version == "v1" and resp.upload_warning is None
    v1 = isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1"
    assert (v1 / MODEL_FILENAME).exists() and (v1 / SCALER_FILENAME).exists()
    assert json.loads((v1 / MANIFEST_FILENAME).read_text())["selected_model"] == "GRU"


def test_train_with_run_and_mlflow_failure_is_a_warning(train_csv, fast, monkeypatch):
    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = plugin_mod.Ml23LacticMarketPriceForecastPlugin().train(data_path=train_csv, mlflow_run_id="run-1")
    assert "mlflow caído" in resp.upload_warning
    assert resp.base_version.version == "v1" and resp.base_version.mlflow_run_id == "run-1"
