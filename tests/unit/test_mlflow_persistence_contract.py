"""Contract tests for the MLflow persistence rules shared by every trainable plugin.

1. ``mlflow_run_id`` is mandatory and non-empty on every /train (→ 422), except on the models
   whose /train publishes the new base model (PROMOTING_MODEL_IDS), where it is optional.
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
from app.registry import REGISTRY
from tests.conftest import TEST_REGISTRY

# Models whose /train publishes the new base (BASE_ARTIFACT_FOLDER on the real plugin class):
# they accept /train without a run (spec D4). Read from the real registry so the list cannot drift.
PROMOTING_MODEL_IDS = {
    entry.model_id for entry in REGISTRY if getattr(entry.plugin_class, "BASE_ARTIFACT_FOLDER", None)
}
TRAINABLE = [entry for entry in TEST_REGISTRY
             if entry.train_request_type is not None and entry.model_id not in PROMOTING_MODEL_IDS]


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


@pytest.mark.parametrize("error", [
    FileNotFoundError("model/scaler.pkl"), EOFError(), ValueError("truncated"),
    RuntimeError("size mismatch for head.weight"), KeyError("target"),
])
def test_require_user_model_turns_bad_artifacts_into_422(error):
    def download(run_id):
        raise error
    with pytest.raises(UserModelUnavailableError, match="run-1"):
        require_user_model(download)("run-1")


@pytest.mark.parametrize("error", [AttributeError("x"), TypeError("unexpected keyword"), NameError("y")])
def test_require_user_model_lets_loader_bugs_surface_as_500(error):
    # A bug in our own loader is not a client error: it must not be disguised as a 422.
    def download(run_id):
        raise error
    with pytest.raises(type(error)):
        require_user_model(download)("run-1")


# ── Plugin real (ml15) ────────────────────────────────────────────────────────

@pytest.fixture
def ml15(monkeypatch, tmp_path):
    from sklearn.preprocessing import StandardScaler

    from app.plugins.ml15_wine_ipi_price_forecast import mlflow_utils, plugin

    metrics = dict(rmse=1.0, mae=1.0, mape_pct=1.0, r2=0.5, mda_pct=100.0)
    monkeypatch.setattr(plugin.training, "train_model", lambda df: {
        "model": StandardScaler(), "n_train": 10, "n_test": 2, "metrics": metrics,
        "baseline_persistence": metrics, "rmse_reduction_pct": 0.0, "mape_reduction_pct": 0.0,
        "train_end_date": pd.Timestamp("2023-12-01"), "train_origin_start": pd.Timestamp("2016-06-01"),
        "train_origin_end": pd.Timestamp("2023-06-01"), "test_start_date": pd.Timestamp("2025-01-01"),
        "test_end_date": pd.Timestamp("2026-01-01"), "training_time_s": 0.0,
    })
    instance = plugin.Ml15WineIpiPriceForecastPlugin()
    instance._payload = {"feature_columns": []}  # pylint: disable=protected-access
    data = tmp_path / "train.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(data, index=False)
    return plugin, mlflow_utils, instance, str(data)


def test_ml15_train_mlflow_failure_is_a_warning_and_the_base_is_published(ml15, monkeypatch):
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
    resp = instance.train(data_path=data, mlflow_run_id="run-1")
    assert "MLflow unreachable" in resp.upload_warning
    assert resp.base_version.version == "v1"


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


def test_promoting_models_are_the_documented_ones():
    """CLAUDE.md, the plugin-integration skill and the spec list these models by name."""
    assert PROMOTING_MODEL_IDS == {
        "ml13-wine-price-fluctuation-prediction", "ml14-wine-phyto-price-forecast",
        "ml15-wine-ipi-price-forecast", "ml16-meat-raw-material-price-alert",
        "ml17-meat-market-price-analysis", "ml18-meat-spatial-price-forecast",
        "ml19-cereals-cost-forecast", "ml21-cereals-price-spatial",
        "ml23-lactic-market-price-forecast", "ml24-dairy-spatial-price-forecast",
    }
