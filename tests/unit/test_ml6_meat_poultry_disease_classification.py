"""Endpoint tests for the ``ml6-meat-poultry-disease-classification`` model.

The endpoint tests use FakePlugin (wiring only). The last tests run the plugin's real pure
functions (majority vote, split, folder mapping) against the AI team's behaviour; numerical
correctness is in outputs/a06/verification_report.md.
"""
import os

import pytest

from app.domain.services.exceptions import InvalidImageError
from app.plugins.ml6_meat_poultry_disease_classification.exceptions import UnknownImageModalityError

MODEL_ID = "ml6-meat-poultry-disease-classification"
PREFIX = f"/models/{MODEL_ID}"

_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)
INLINE_PAYLOAD = {"mode": "inline", "model_key": "thermal", "image_base64": _TINY_PNG_B64}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == MODEL_ID
    assert body["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == MODEL_ID


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["task"] == "thermal"
    assert body["prediction"] in ("healthy", "sick")
    assert set(body["model_predictions"]) == {"densenet121", "efficientnet_v2s", "resnet50"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "model_key": "rgb", "data_path": "/tmp/feces.zip"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["predictions"][0]["prediction"] == "ncd"


def test_predict_unknown_model_key_returns_422(client):
    resp = client.post(f"{PREFIX}/predict", json={**INLINE_PAYLOAD, "model_key": "xray"})
    assert resp.status_code == 422


def test_predict_missing_model_key_maps_to_422(client, fake_plugins):
    fake_plugins[MODEL_ID].raise_on_inline = UnknownImageModalityError("model_key requerido")
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "image_base64": _TINY_PNG_B64})
    assert resp.status_code == 422


def test_predict_inline_invalid_image_maps_to_422(client, fake_plugins):
    fake_plugins[MODEL_ID].raise_on_inline = InvalidImageError("bad")
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422


def test_train_returns_200_with_metrics(client):
    """training.supported=true (inbox/a06/manifest.yaml)."""
    resp = client.post(
        f"{PREFIX}/train", json={"data_path": "/tmp/x", "mlflow_run_id": "test-run-id", "task": "thermal"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mlflow_run_id"] == "test-run-id"
    assert 0.0 <= body["accuracy"] <= 1.0
    assert body["n_test"] > 0


def test_train_without_mlflow_run_id_returns_422(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/x", "task": "thermal"})
    assert resp.status_code == 422


def test_train_with_empty_mlflow_run_id_returns_422(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/x", "mlflow_run_id": "  ", "task": "rgb"})
    assert resp.status_code == 422


def test_train_without_task_returns_422(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/x", "mlflow_run_id": "test-run-id"})
    assert resp.status_code == 422


# ── Real plugin code (pure functions) ─────────────────────────────────────

def test_majority_vote_three_way_tie_keeps_first_model_vote():
    """src/main.py uses Counter.most_common: on a 3-way tie the first model (vgg19_bn) wins
    (golden cases rgb_tie_001..003)."""
    from app.plugins.ml6_meat_poultry_disease_classification.postprocessing import majority_vote

    assert majority_vote([0, 1, 3]) == 0
    assert majority_vote([3, 0, 1]) == 3
    assert majority_vote([1, 2, 2]) == 2


def test_class_folder_mapping_accepts_raw_and_logical_names():
    from app.plugins.ml6_meat_poultry_disease_classification.training import class_index_of_folder

    assert class_index_of_folder("Healthy", "thermal") == 0      # Mendeley ships Healthy/ and Sick/
    assert class_index_of_folder("sick", "thermal") == 1
    assert class_index_of_folder("pcrncd", "rgb") == 2
    assert class_index_of_folder("salmo", "rgb") == 3
    assert class_index_of_folder("pcrncd", "thermal") is None


def _make_raw(tmp_path, counts):
    for folder, n in counts.items():
        os.makedirs(tmp_path / folder)
        for i in range(n):
            (tmp_path / folder / f"{folder}.{i}.jpg").write_bytes(b"")
    return str(tmp_path)


def test_split_matches_ensure_data_splits_sizes(tmp_path):
    """70/15/15 with int() truncation per class, like src/utils/io.py::ensure_data_splits."""
    from app.plugins.ml6_meat_poultry_disease_classification.training import split_dataset

    root = _make_raw(tmp_path, {"healthy": 86, "sick": 168})
    splits = split_dataset(root, "thermal")
    assert [len(splits[p]) for p in ("train", "val", "test")] == [177, 37, 40]
    assert splits["test"][0][1] == 0 and splits["test"][-1][1] == 1   # ImageFolder order
    again = split_dataset(root, "thermal")
    assert again == splits                                             # reproducible (seed 42)


def test_split_rejects_missing_class_and_too_few_images(tmp_path):
    from app.plugins.ml6_meat_poultry_disease_classification.training import split_dataset

    with pytest.raises(ValueError, match="faltan"):
        split_dataset(_make_raw(tmp_path / "a", {"healthy": 20}), "thermal")
    with pytest.raises(ValueError, match="muy pocas"):
        split_dataset(_make_raw(tmp_path / "b", {"healthy": 6, "sick": 20}), "thermal")


@pytest.mark.parametrize("payload", [
    {"mode": "inline", "model_key": "thermal", "image_base64": _TINY_PNG_B64, "mlflow_run_id": "run-1"},
    {"mode": "batch", "model_key": "rgb", "data_path": "/tmp/x.zip", "mlflow_run_id": "run-1"},
])
def test_predict_request_forwards_mlflow_run_id_to_plugin(payload):
    """The predict DTOs must declare mlflow_run_id: without it pydantic drops the field, the
    use case passes "" and the base model answers a request for the user's model (found during
    the a06 verification against the real server)."""
    from pydantic import TypeAdapter

    from app.application.use_cases.predict_model_use_case import PredictModelUseCase
    from app.plugins.ml6_meat_poultry_disease_classification.predict_dto import (
        PredictBatchResponse,
        PredictInlineResponse,
        PredictRequest,
    )

    seen = {}

    class _Spy:
        def predict_inline(self, *, features, model_key=None, threshold=None, mlflow_run_id=""):
            seen.update(run=mlflow_run_id, key=model_key)
            return PredictInlineResponse(
                model_id=MODEL_ID, task="thermal", prediction="sick", predicted_class_index=1,
                confidence=1.0, model_predictions={}, model_probabilities={}, features_used=[])

        def predict_batch(self, *, data_path, mlflow_run_id="", model_key=None):
            seen.update(run=mlflow_run_id, key=model_key)
            return PredictBatchResponse(model_id=MODEL_ID, task="rgb", n_images=0, predictions=[])

    request = TypeAdapter(PredictRequest).validate_python(payload)
    PredictModelUseCase(_Spy()).execute(request)
    assert seen == {"run": "run-1", "key": payload["model_key"]}


@pytest.fixture
def ml6_train_stubbed(monkeypatch, tmp_path):
    """Real Ml6 plugin.train() with the long steps (data loading, epochs) and MLflow stubbed."""
    import torch

    from app.plugins.ml6_meat_poultry_disease_classification import plugin as ml6

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(2, 2)

    instance = ml6.Ml6MeatPoultryDiseaseClassificationPlugin()
    base = {name: _Tiny() for name in ("densenet121", "efficientnet_v2s", "resnet50")}
    instance._ensembles = {"thermal": base, "rgb": {}}  # pylint: disable=protected-access
    instance._device = torch.device("cpu")  # pylint: disable=protected-access
    splits = {"train": [("a", 0)] * 3, "val": [("b", 1)], "test": [("c", 0), ("d", 1)]}
    monkeypatch.setattr(ml6, "split_dataset", lambda root, task: splits)
    monkeypatch.setattr(ml6, "make_loaders", lambda splits, task, bs: {"test": None})
    monkeypatch.setattr(ml6, "train_model", lambda model, loaders, hp, dev, log: (model, 0.5, 3))
    monkeypatch.setattr(ml6, "predict_loader", lambda model, loader, dev: [0, 1])

    class _Tracker:
        def __init__(self, run_id=""):
            self.run_id = run_id

        def log_params(self, params):
            pass

        def log_metrics(self, metrics, step=0):
            pass

    monkeypatch.setattr(ml6, "BaseMLflowTracker", _Tracker)
    (tmp_path / "data").mkdir()
    return ml6, instance, base, str(tmp_path / "data")


def test_real_train_uploads_ensemble_and_never_touches_served_models(ml6_train_stubbed, monkeypatch):
    ml6, instance, base, data_path = ml6_train_stubbed
    uploaded = {}

    def fake_upload(artifact_dir, run_id, metrics):
        uploaded["files"] = sorted(os.listdir(artifact_dir))
        uploaded["run"], uploaded["metrics"] = run_id, metrics

    monkeypatch.setattr(ml6, "upload_artifacts_to_mlflow", fake_upload)
    resp = instance.train(data_path=data_path, mlflow_run_id="run-9", task="thermal")
    assert uploaded["files"] == ["densenet121.pt", "efficientnet_v2s.pt", "metadata.json", "resnet50.pt"]
    assert uploaded["run"] == "run-9" and resp.mlflow_run_id == "run-9"
    assert resp.accuracy == 1.0 and (resp.n_train, resp.n_val, resp.n_test) == (3, 1, 2)
    assert instance._ensembles["thermal"] is not None  # pylint: disable=protected-access
    assert all(instance._ensembles["thermal"][n] is m for n, m in base.items())  # served models untouched


def test_real_train_fails_with_502_error_when_mlflow_upload_fails(ml6_train_stubbed, monkeypatch):
    from app.domain.services.exceptions import ModelPersistenceError

    ml6, instance, _, data_path = ml6_train_stubbed

    def broken_upload(*_args, **_kwargs):
        raise ConnectionError("mlflow down")

    monkeypatch.setattr(ml6, "upload_artifacts_to_mlflow", broken_upload)
    with pytest.raises(ModelPersistenceError, match="mlflow down"):
        instance.train(data_path=data_path, mlflow_run_id="run-9", task="thermal")
