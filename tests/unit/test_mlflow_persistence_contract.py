"""Contract tests for the MLflow persistence rules shared by every trainable plugin.

1. ``mlflow_run_id`` is mandatory and non-empty on every /train (→ 422).
2. If the retrained model cannot be saved to MLflow, /train fails (→ 502) instead of
   answering 200 with a warning and silently discarding the model.
3. If predict/stats are asked for a run whose model cannot be loaded, they fail (→ 422)
   instead of silently serving the base model.

The router tests use the FakePlugin wiring; the ml15 tests run the real plugin code with
the training step and MLflow replaced, so they catch regressions the fakes cannot.
"""
import pandas as pd
import pytest

from app.domain.services.exceptions import ModelPersistenceError, UserModelUnavailableError
from app.domain.services.mlflow_tracker import require_user_model
from tests.conftest import TEST_REGISTRY

TRAINABLE = [entry for entry in TEST_REGISTRY if entry.train_request_type is not None]


# ── 1. mlflow_run_id obligatorio y no vacío ───────────────────────────────────

@pytest.mark.parametrize("run_id", ["", "   "])
@pytest.mark.parametrize("entry", TRAINABLE, ids=[entry.model_id for entry in TRAINABLE])
def test_train_rejects_empty_mlflow_run_id(client, entry, run_id):
    resp = client.post(f"{entry.prefix}/train", json={"data_path": "/tmp/x.csv", "mlflow_run_id": run_id})
    assert resp.status_code == 422
    assert any("mlflow_run_id" in err["loc"] for err in resp.json()["detail"])


# ── 2 y 3. Mapeo HTTP de los errores de MLflow ────────────────────────────────

ML10 = "ml10-dairy-disease-vector-detection"
ML10_PREFIX = "/models/ml10-dairy-disease-vector-detection"


def test_train_returns_502_when_model_cannot_be_saved(client, fake_plugins, monkeypatch):
    def failing_train(**_kwargs):
        raise ModelPersistenceError("El modelo reentrenado no se ha podido guardar en MLflow: boom")

    monkeypatch.setattr(fake_plugins[ML10], "train", failing_train)
    resp = client.post(f"{ML10_PREFIX}/train", json={"data_path": "/tmp/x.zip", "mlflow_run_id": "run-1"})
    assert resp.status_code == 502
    assert "no se ha podido guardar en MLflow" in resp.json()["detail"]


def test_predict_returns_422_when_user_model_unavailable(client, fake_plugins, lacteo_inline_payload):
    fake_plugins[ML10].raise_on_inline = UserModelUnavailableError("run-1 sin modelo")
    resp = client.post(f"{ML10_PREFIX}/predict", json={**lacteo_inline_payload, "mlflow_run_id": "run-1"})
    assert resp.status_code == 422
    assert "run-1 sin modelo" in resp.json()["detail"]


def test_stats_returns_422_when_user_model_unavailable(client, fake_plugins, monkeypatch):
    def failing_stats(mlflow_run_id=""):
        raise UserModelUnavailableError(f"{mlflow_run_id} sin modelo")

    monkeypatch.setattr(fake_plugins[ML10], "stats", failing_stats)
    resp = client.get(f"{ML10_PREFIX}/stats", params={"mlflow_run_id": "run-1"})
    assert resp.status_code == 422
    assert "run-1 sin modelo" in resp.json()["detail"]


def test_require_user_model_raises_only_when_a_run_was_requested():
    download = require_user_model(lambda run_id: None)
    with pytest.raises(UserModelUnavailableError, match="run-1"):
        download("run-1")
    assert download("") is None
    assert require_user_model(lambda run_id: ("model", "/tmp/x"))("run-1") == ("model", "/tmp/x")


# ── Plugin real (ml15) ────────────────────────────────────────────────────────

@pytest.fixture
def ml15(monkeypatch, tmp_path):
    from sklearn.preprocessing import StandardScaler

    from app.plugins.ml15_wine_ipi_price_forecast import mlflow_utils, plugin

    monkeypatch.setattr(plugin.training, "train_model", lambda df: {
        "model": StandardScaler(), "n_train": 10, "n_test": 2,
        "metrics": dict(rmse=1.0, mae=1.0, mape_pct=1.0, r2=0.5, mda_pct=100.0),
    })
    instance = plugin.Ml15WineIpiPriceForecastPlugin()
    instance._payload = {"feature_columns": []}  # pylint: disable=protected-access
    data = tmp_path / "train.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(data, index=False)
    return plugin, mlflow_utils, instance, str(data)


def test_ml15_train_fails_when_mlflow_upload_fails(ml15, monkeypatch):
    plugin, _, instance, data = ml15

    class UnreachableTracker:
        def __init__(self, run_id):
            self.run_id = run_id

        def log_params(self, *_args):
            pass

        def log_metrics(self, *_args):
            pass

        def upload_artifacts(self, *_args, **_kwargs):
            raise ConnectionError("MLflow unreachable")

    monkeypatch.setattr(plugin, "BaseMLflowTracker", UnreachableTracker)
    with pytest.raises(ModelPersistenceError, match="MLflow unreachable"):
        instance.train(data_path=data, mlflow_run_id="run-1")


def test_ml15_predict_fails_instead_of_serving_base_model(ml15, monkeypatch):
    _, mlflow_utils, instance, data = ml15

    class EmptyRunTracker:
        def __init__(self, run_id):
            self.run_id = run_id

        def download_artifacts(self, *_args, **_kwargs):
            return ""  # what BaseMLflowTracker returns for a run without model / MLflow down

    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker", EmptyRunTracker)
    with pytest.raises(UserModelUnavailableError, match="run-1"):
        instance.predict_batch(data_path=data, mlflow_run_id="run-1")
