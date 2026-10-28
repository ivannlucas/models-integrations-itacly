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

from app.domain.services.exceptions import (
    TrainJobResultUnreadableError,
    TrainJobRunNotFoundError,
    TrainJobStoreUnavailableError,
)
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
        record = TrainJobRecord("run-1", "m47", TrainJobStatus.FAILED, 1.0, 2.0, error="x" * 10000,
                                error_type="ValueError", http_status=400)
        MLflowTrainJobStore(client=client).save(record)
        error_tag = next(c for c in client.set_tag.call_args_list if c.args[1] == f"{TAG_PREFIX}error")
        assert len(error_tag.args[2]) == 4000
        assert call("run-1", f"{TAG_PREFIX}error_type", "ValueError") in client.set_tag.call_args_list
        assert call("run-1", f"{TAG_PREFIX}http_status", "400") in client.set_tag.call_args_list
        assert client.set_tag.call_args_list[-1] == call("run-1", f"{TAG_PREFIX}status", "failed")

    def test_non_failed_job_writes_no_http_status(self):
        client = MagicMock()
        MLflowTrainJobStore(client=client).save(TrainJobRecord("run-1", "m47", TrainJobStatus.RUNNING, 1.0, 2.0))
        assert not any(c.args[1] == f"{TAG_PREFIX}http_status" for c in client.set_tag.call_args_list)

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
        client.get_run.return_value = _run(_tags("failed", error="sin datos", error_type="ValueError", http_status="400"))
        record = MLflowTrainJobStore(client=client).get("run-1")
        assert (record.status, record.error, record.error_type) == (TrainJobStatus.FAILED, "sin datos", "ValueError")
        assert record.http_status == 400

    def test_failed_job_saved_before_http_status_existed_has_none(self):
        client = MagicMock()
        client.get_run.return_value = _run(_tags("failed", error="sin datos", error_type="ValueError"))
        assert MLflowTrainJobStore(client=client).get("run-1").http_status is None

    def test_succeeded_job_loads_result_from_the_async_train_folder(self):
        client = _proxy_client(result_text=json.dumps({"detail": "ok"}))
        record = MLflowTrainJobStore(client=client).get("run-1")
        assert record.result == {"detail": "ok"}
        client.download_artifacts.assert_called_once()
        assert client.download_artifacts.call_args.args[:2] == ("run-1", "async_train")

    def test_lone_file_download_that_yields_an_empty_dir_is_not_used(self, tmp_path):
        """Seen in int (ml48 via the platform's S3 proxy): downloading async_train/result.json alone
        returns an empty directory; the folder download returns the real file."""
        client = _proxy_client(result_text=json.dumps({"detail": "ok"}))
        lone = client.download_artifacts.side_effect("run-1", RESULT_ARTIFACT, str(tmp_path))
        assert os.path.isdir(lone) and os.listdir(lone) == []  # the double reproduces the proxy
        assert MLflowTrainJobStore(client=client).get("run-1").result == {"detail": "ok"}

    @pytest.mark.parametrize(
        "result_text,what", [(None, "nada"), ("<dir>", "una carpeta"), ("{roto", "no se puede leer")],
    )
    def test_unreadable_result_is_a_clear_error_not_store_unavailable(self, result_text, what):
        client = _proxy_client(result_text=result_text)
        with pytest.raises(TrainJobResultUnreadableError, match=what) as info:
            MLflowTrainJobStore(client=client).get("run-1")
        assert "run-1" in str(info.value) and RESULT_ARTIFACT in str(info.value)

    def test_missing_result_artifact_is_unreadable_not_missing_run(self):
        client = _proxy_client(result_text=None)
        client.download_artifacts.side_effect = MlflowException("no artifact", error_code=RESOURCE_DOES_NOT_EXIST)
        with pytest.raises(TrainJobResultUnreadableError, match="no tiene"):
            MLflowTrainJobStore(client=client).get("run-1")

    def test_local_file_errors_during_download_are_unreadable_not_503(self):
        client = _proxy_client(result_text=None)
        client.download_artifacts.side_effect = IsADirectoryError(21, "Is a directory")
        with pytest.raises(TrainJobResultUnreadableError, match="IsADirectoryError"):
            MLflowTrainJobStore(client=client).get("run-1")

    def test_network_errors_during_download_stay_store_unavailable(self):
        client = _proxy_client(result_text=None)
        client.download_artifacts.side_effect = requests.exceptions.ConnectionError("refused")
        with pytest.raises(TrainJobStoreUnavailableError):
            MLflowTrainJobStore(client=client).get("run-1")


def _proxy_client(result_text):
    """Succeeded run whose artifacts behave like the platform's S3 proxy.

    ``download_artifacts("async_train/result.json")`` yields an empty directory (the proxy lists the
    file as a folder); ``download_artifacts("async_train")`` yields the folder holding result.json
    (``result_text`` None = no file, "<dir>" = a directory instead of the file).
    """
    client = MagicMock()
    client.get_run.return_value = _run(_tags("succeeded"))

    def fake_download(run_id, path, dst_path):
        local = os.path.join(dst_path, path)
        os.makedirs(local, exist_ok=True)
        if path == "async_train" and result_text is not None:
            target = os.path.join(local, "result.json")
            if result_text == "<dir>":
                os.makedirs(target)
            else:
                with open(target, "w", encoding="utf-8") as fh:
                    fh.write(result_text)
        return local

    client.download_artifacts.side_effect = fake_download
    return client


def test_pickling_drops_the_client():
    store = MLflowTrainJobStore(tracking_uri="http://mlflow:5000", client=MagicMock())
    clone = pickle.loads(pickle.dumps(store))
    assert clone._client is None
    assert clone._tracking_uri == "http://mlflow:5000"
