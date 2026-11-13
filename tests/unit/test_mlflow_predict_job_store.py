"""MLflowPredictJobStore against a real local MLflow (sqlite) and a mocked client."""
from __future__ import annotations

import gzip
import json
import os
import pickle
from unittest.mock import MagicMock

import pytest
from mlflow.tracking import MlflowClient

from app.domain.services.exceptions import (
    TrainJobResultUnreadableError,
    TrainJobStoreUnavailableError,
)
from app.domain.services.predict_job import PredictJobRecord, PredictJobStatus
from app.infrastructure.predict_jobs.mlflow_job_store import (
    JOB_ID_TAG,
    RESULT_DIR,
    RESULT_FILE,
    TAG_PREFIX,
    MLflowPredictJobStore,
)

EXPERIMENT = "test-async-predict"


@pytest.fixture(name="mlflow_uri", scope="module")
def _mlflow_uri(tmp_path_factory):
    root = tmp_path_factory.mktemp("mlflow")
    uri = f"sqlite:///{root}/mlflow.db"
    MlflowClient(tracking_uri=uri).create_experiment(
        EXPERIMENT, artifact_location=(root / "artifacts").as_uri()
    )
    return uri


def _record(job_id, status=PredictJobStatus.QUEUED, **kw):
    return PredictJobRecord(job_id, "m5", status, 1000.0, kw.pop("updated_at", 1000.0), **kw)


def test_round_trip_in_a_real_mlflow(mlflow_uri):
    store = MLflowPredictJobStore(tracking_uri=mlflow_uri, experiment_name=EXPERIMENT)
    assert store.get("task-1") is None
    store.create(_record("task-1"), {"data_path": "s3://b/v.mp4", "mlflow_run_id": ""})

    # A fresh store (another replica, or after a restart) finds the job by its tag.
    other = MLflowPredictJobStore(tracking_uri=mlflow_uri, experiment_name=EXPERIMENT)
    assert other.get("task-1") == _record("task-1")
    other.heartbeat("task-1", 1050.0)
    assert store.get("task-1").updated_at == 1050.0

    result = {"model_id": "m5", "predictions": [{"frame": i, "behaviour": "ñu"} for i in range(500)]}
    other.save(_record("task-1", PredictJobStatus.SUCCEEDED, updated_at=1100.0, result=result))
    got = MLflowPredictJobStore(tracking_uri=mlflow_uri, experiment_name=EXPERIMENT).get("task-1")
    assert got.status is PredictJobStatus.SUCCEEDED and got.result == result

    run = MlflowClient(tracking_uri=mlflow_uri).get_run(other._find_run_id("task-1"))  # pylint: disable=protected-access
    assert run.info.run_name == "task-1" and run.info.status == "FINISHED"
    assert run.data.tags[JOB_ID_TAG] == "task-1"
    assert run.data.tags[f"{TAG_PREFIX}data_path"] == "s3://b/v.mp4"
    assert f"{TAG_PREFIX}mlflow_run_id" not in run.data.tags  # empty info is not tagged


def test_failed_job_closes_the_run_failed(mlflow_uri):
    store = MLflowPredictJobStore(tracking_uri=mlflow_uri, experiment_name=EXPERIMENT)
    store.create(_record("task-2"), {})
    store.save(_record("task-2", PredictJobStatus.FAILED, error="x" * 9000,
                       error_type="InvalidVideoError", http_status=422))
    got = store.get("task-2")
    assert (got.error_type, got.http_status, len(got.error)) == ("InvalidVideoError", 422, 4000)
    run = MlflowClient(tracking_uri=mlflow_uri).get_run(store._find_run_id("task-2"))  # pylint: disable=protected-access
    assert run.info.status == "FAILED"


def test_missing_experiment_is_no_job_and_is_created_on_first_job(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path}/m.db"
    store = MLflowPredictJobStore(tracking_uri=uri, experiment_name="nuevo")
    assert store.get("task-1") is None
    monkeypatch.chdir(tmp_path)  # default artifact root of a new experiment is ./mlruns
    store.create(_record("task-1"), {})
    assert store.get("task-1").status is PredictJobStatus.QUEUED


def test_store_is_picklable_without_its_client(mlflow_uri):
    store = MLflowPredictJobStore(tracking_uri=mlflow_uri, experiment_name=EXPERIMENT)
    _ = store.client
    clone = pickle.loads(pickle.dumps(store))
    assert clone._client is None  # pylint: disable=protected-access
    clone.create(_record("task-3"), {})
    assert store.get("task-3") is not None


def _mocked_store(download):
    client = MagicMock()
    client.get_run.return_value.data.tags = {
        f"{TAG_PREFIX}status": "succeeded", f"{TAG_PREFIX}model_id": "m5",
        f"{TAG_PREFIX}submitted_at": "1.0", f"{TAG_PREFIX}updated_at": "2.0",
    }
    client.download_artifacts.side_effect = download
    store = MLflowPredictJobStore(tracking_uri="http://x", experiment_name="e", client=client)
    store._runs["task-1"] = "run-1"  # pylint: disable=protected-access
    return store, client


def test_result_is_read_by_downloading_the_folder():
    def download(run_id, path, dst):
        assert (run_id, path) == ("run-1", RESULT_DIR)  # never the lone file (S3 proxy bug)
        os.makedirs(os.path.join(dst, RESULT_DIR))
        with gzip.open(os.path.join(dst, RESULT_DIR, RESULT_FILE), "wt") as fh:
            json.dump({"ok": 1}, fh)
        return os.path.join(dst, RESULT_DIR)

    store, _ = _mocked_store(download)
    assert store.get("task-1").result == {"ok": 1}


def test_result_downloaded_as_a_folder_is_unreadable_not_unavailable():
    def download(run_id, path, dst):
        os.makedirs(os.path.join(dst, RESULT_DIR, RESULT_FILE))
        return os.path.join(dst, RESULT_DIR)

    store, _ = _mocked_store(download)
    with pytest.raises(TrainJobResultUnreadableError, match="una carpeta"):
        store.get("task-1")


def test_network_errors_are_unavailable():
    import requests  # pylint: disable=import-outside-toplevel
    store, client = _mocked_store(None)
    client.get_run.side_effect = requests.exceptions.ConnectionError("down")
    with pytest.raises(TrainJobStoreUnavailableError):
        store.get("task-1")
