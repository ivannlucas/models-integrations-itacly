"""MLflowTrainJobStore against a mocked MlflowClient."""
from __future__ import annotations

import json
import os
import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import requests
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST

from app.domain.services.exceptions import TrainJobRunNotFoundError, TrainJobStoreUnavailableError
from app.domain.services.train_job import TrainJobRecord, TrainJobStatus
from app.infrastructure.train_jobs.mlflow_job_store import RESULT_ARTIFACT, TAG_PREFIX, MLflowTrainJobStore


def _run(tags: dict):
    return SimpleNamespace(data=SimpleNamespace(tags=tags))


def _tags(status: str, **extra) -> dict:
    tags = {
        f"{TAG_PREFIX}status": status,
        f"{TAG_PREFIX}model_id": "m47",
        f"{TAG_PREFIX}submitted_at": "1000.0",
        f"{TAG_PREFIX}updated_at": "1010.5",
        "mlflow.runName": "otra-etiqueta",
    }
    tags.update({f"{TAG_PREFIX}{k}": v for k, v in extra.items()})
    return tags


class TestSave:
    def test_queued_sets_tags_with_status_last_and_no_artifact(self):
        client = MagicMock()
        MLflowTrainJobStore(client=client).save(TrainJobRecord("run-1", "m47", TrainJobStatus.QUEUED, 1000.0, 1000.0))
        client.log_text.assert_not_called()
        assert client.set_tag.call_args_list[-1] == call("run-1", f"{TAG_PREFIX}status", "queued")
        assert call("run-1", f"{TAG_PREFIX}model_id", "m47") in client.set_tag.call_args_list
        assert call("run-1", f"{TAG_PREFIX}submitted_at", "1000.0") in client.set_tag.call_args_list

    def test_succeeded_writes_result_before_status(self):
        client = MagicMock()
        record = TrainJobRecord("run-1", "m47", TrainJobStatus.SUCCEEDED, 1.0, 2.0, result={"detail": "ok", "f1": 0.9})
        MLflowTrainJobStore(client=client).save(record)
        names = [c[0] for c in client.mock_calls]
        assert names.index("log_text") < len(names) - 1
        assert client.mock_calls[-1] == call.set_tag("run-1", f"{TAG_PREFIX}status", "succeeded")
        client.log_text.assert_called_once_with("run-1", json.dumps({"detail": "ok", "f1": 0.9}, ensure_ascii=False), RESULT_ARTIFACT)

    def test_failed_truncates_long_error(self):
        client = MagicMock()
        record = TrainJobRecord("run-1", "m47", TrainJobStatus.FAILED, 1.0, 2.0, error="x" * 10000, error_type="ValueError")
        MLflowTrainJobStore(client=client).save(record)
        error_tag = next(c for c in client.set_tag.call_args_list if c.args[1] == f"{TAG_PREFIX}error")
        assert len(error_tag.args[2]) == 4000
        assert call("run-1", f"{TAG_PREFIX}error_type", "ValueError") in client.set_tag.call_args_list

    def test_saving_on_a_run_that_does_not_exist_is_a_client_error(self):
        client = MagicMock()
        client.set_tag.side_effect = MlflowException("no run", error_code=RESOURCE_DOES_NOT_EXIST)
        with pytest.raises(TrainJobRunNotFoundError, match="run-x"):
            MLflowTrainJobStore(client=client).save(TrainJobRecord("run-x", "m47", TrainJobStatus.QUEUED, 1.0, 1.0))

    def test_save_with_mlflow_down_is_store_unavailable(self):
        client = MagicMock()
        client.set_tag.side_effect = requests.exceptions.ConnectionError("refused")
        with pytest.raises(TrainJobStoreUnavailableError):
            MLflowTrainJobStore(client=client).save(TrainJobRecord("run-1", "m47", TrainJobStatus.QUEUED, 1.0, 1.0))

    def test_heartbeat_only_touches_updated_at(self):
        client = MagicMock()
        MLflowTrainJobStore(client=client).heartbeat("run-1", 1234.5)
        client.set_tag.assert_called_once_with("run-1", f"{TAG_PREFIX}updated_at", "1234.5")


class TestGet:
    def test_missing_run_returns_none(self):
        client = MagicMock()
        client.get_run.side_effect = MlflowException("no run", error_code=RESOURCE_DOES_NOT_EXIST)
        assert MLflowTrainJobStore(client=client).get("run-x") is None

    def test_other_mlflow_errors_become_store_unavailable(self):
        client = MagicMock()
        client.get_run.side_effect = MlflowException("boom", error_code=INTERNAL_ERROR)
        with pytest.raises(TrainJobStoreUnavailableError, match="boom"):
            MLflowTrainJobStore(client=client).get("run-1")

    def test_connection_errors_become_store_unavailable(self):
        client = MagicMock()
        client.get_run.side_effect = requests.exceptions.ConnectionError("refused")
        with pytest.raises(TrainJobStoreUnavailableError, match="refused"):
            MLflowTrainJobStore(client=client).get("run-1")

    def test_run_without_job_tags_returns_none(self):
        client = MagicMock()
        client.get_run.return_value = _run({"mlflow.runName": "pre-creado por la plataforma"})
        assert MLflowTrainJobStore(client=client).get("run-1") is None

    def test_running_job_is_parsed_without_downloading(self):
        client = MagicMock()
        client.get_run.return_value = _run(_tags("running"))
        record = MLflowTrainJobStore(client=client).get("run-1")
        assert record == TrainJobRecord("run-1", "m47", TrainJobStatus.RUNNING, 1000.0, 1010.5)
        client.download_artifacts.assert_not_called()

    def test_failed_job_carries_error(self):
        client = MagicMock()
        client.get_run.return_value = _run(_tags("failed", error="sin datos", error_type="ValueError"))
        record = MLflowTrainJobStore(client=client).get("run-1")
        assert (record.status, record.error, record.error_type) == (TrainJobStatus.FAILED, "sin datos", "ValueError")

    def test_succeeded_job_loads_result_artifact(self):
        client = MagicMock()
        client.get_run.return_value = _run(_tags("succeeded"))

        def fake_download(run_id, path, dst_path):
            local = os.path.join(dst_path, path)
            os.makedirs(os.path.dirname(local), exist_ok=True)
            with open(local, "w", encoding="utf-8") as fh:
                json.dump({"detail": "ok"}, fh)
            return local

        client.download_artifacts.side_effect = fake_download
        record = MLflowTrainJobStore(client=client).get("run-1")
        assert record.result == {"detail": "ok"}
        client.download_artifacts.assert_called_once()
        assert client.download_artifacts.call_args.args[:2] == ("run-1", RESULT_ARTIFACT)


def test_pickling_drops_the_client():
    store = MLflowTrainJobStore(tracking_uri="http://mlflow:5000", client=MagicMock())
    clone = pickle.loads(pickle.dumps(store))
    assert clone._client is None
    assert clone._tracking_uri == "http://mlflow:5000"
