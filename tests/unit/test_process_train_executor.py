"""ProcessTrainExecutor with real `spawn` child processes and a file-backed store."""
from __future__ import annotations

import os
import time

from app.domain.services.train_job import TrainJobRecord, TrainJobStatus
from app.infrastructure.train_jobs.process_executor import ProcessTrainExecutor, run_train_job
from tests.unit.train_job_fakes import (
    EchoTrainPlugin,
    FailingTrainPlugin,
    FileJobStore,
    NumpyMetricsTrainPlugin,
    SlowTrainPlugin,
)


def _queue(store, job_id):
    now = time.time()
    store.save(TrainJobRecord(job_id, "fake-model", TrainJobStatus.QUEUED, now, now))


def _wait(predicate, timeout=60.0, what="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(f"timeout waiting for {what}")


def _terminal(store, job_id):
    return _wait(lambda: (r := store.get(job_id)) is not None and r.status.is_terminal and r, what=f"{job_id} terminal")


def _submit(executor, store, job_id, plugin_class, data_path="s3://b/d.csv"):
    _queue(store, job_id)
    executor.submit(
        job_id=job_id, plugin_class=plugin_class,
        train_kwargs={"data_path": data_path, "mlflow_run_id": job_id}, store=store,
    )


def test_trains_in_a_child_process_and_stores_the_result(tmp_path):
    store, executor = FileJobStore(str(tmp_path)), ProcessTrainExecutor(heartbeat_s=0.2)
    _submit(executor, store, "run-1", EchoTrainPlugin)
    record = _terminal(store, "run-1")
    assert record.status is TrainJobStatus.SUCCEEDED, record.error
    assert record.result["detail"] == "ok"
    assert record.result["mlflow_run_id"] == "run-1"
    assert record.result["metrics"]["data_path"] == "s3://b/d.csv"
    assert record.result["metrics"]["pid"] != os.getpid()


def test_failure_is_stored_with_its_type(tmp_path):
    store, executor = FileJobStore(str(tmp_path)), ProcessTrainExecutor(heartbeat_s=0.2)
    _submit(executor, store, "run-2", FailingTrainPlugin, data_path="s3://b/missing.csv")
    record = _terminal(store, "run-2")
    assert record.status is TrainJobStatus.FAILED
    assert record.error_type == "FileNotFoundError"
    assert "s3://b/missing.csv" in record.error


def test_numpy_metrics_are_stored_as_plain_json(tmp_path):
    store, executor = FileJobStore(str(tmp_path)), ProcessTrainExecutor(heartbeat_s=0.2)
    _submit(executor, store, "run-3", NumpyMetricsTrainPlugin)
    record = _terminal(store, "run-3")
    assert record.status is TrainJobStatus.SUCCEEDED, record.error
    assert record.result["metrics"] == {"f1": 0.5, "n": 3, "cm": [[1, 0], [0, 2]]}


def test_busy_while_alive_and_free_afterwards(tmp_path):
    store, executor = FileJobStore(str(tmp_path)), ProcessTrainExecutor(heartbeat_s=0.2)
    _submit(executor, store, "run-4", SlowTrainPlugin, data_path="2")
    assert executor.is_busy()
    _terminal(store, "run-4")
    _wait(lambda: not executor.is_busy(), timeout=10, what="executor free")


def test_heartbeat_advances_updated_at_while_running(tmp_path):
    store, executor = FileJobStore(str(tmp_path)), ProcessTrainExecutor(heartbeat_s=0.2)
    _submit(executor, store, "run-5", SlowTrainPlugin, data_path="3")
    first = _wait(lambda: (r := store.get("run-5")).status is TrainJobStatus.RUNNING and r, what="running")
    time.sleep(1.0)
    second = store.get("run-5")
    assert second.status is TrainJobStatus.RUNNING
    assert second.updated_at > first.updated_at
    assert _terminal(store, "run-5").status is TrainJobStatus.SUCCEEDED


class _LoseOnSecondGetStore(FileJobStore):
    """Simulates GET marking the job lost while train() runs (2nd read = the pre-final-save check)."""

    def __init__(self, root):
        super().__init__(root)
        self.gets = 0

    def get(self, job_id):
        self.gets += 1
        if self.gets == 2:
            record = super().get(job_id)
            self.save(TrainJobRecord(job_id, record.model_id, TrainJobStatus.FAILED, record.submitted_at,
                                     time.time(), error="lost", error_type="TrainJobLost"))
        return super().get(job_id)


def test_late_success_does_not_overwrite_a_job_already_marked_lost(tmp_path):
    store = _LoseOnSecondGetStore(str(tmp_path))
    _queue(store, "run-6")
    run_train_job(job_id="run-6", plugin_class=EchoTrainPlugin,
                  train_kwargs={"data_path": "x", "mlflow_run_id": "run-6"}, store=store, heartbeat_s=60)
    record = FileJobStore(str(tmp_path)).get("run-6")
    assert record.status is TrainJobStatus.FAILED
    assert record.error_type == "TrainJobLost"


def test_job_marked_lost_while_queued_is_not_trained(tmp_path):
    store = FileJobStore(str(tmp_path))
    now = time.time()
    store.save(TrainJobRecord("run-7", "fake-model", TrainJobStatus.FAILED, now, now,
                              error="lost", error_type="TrainJobLost"))
    run_train_job(job_id="run-7", plugin_class=FailingTrainPlugin,
                  train_kwargs={"data_path": "x", "mlflow_run_id": "run-7"}, store=store, heartbeat_s=60)
    record = store.get("run-7")
    assert record.status is TrainJobStatus.FAILED
    assert record.error_type == "TrainJobLost"  # FailingTrainPlugin never ran (it would say FileNotFoundError)


def test_run_train_job_without_record_does_nothing(tmp_path):
    store = FileJobStore(str(tmp_path))
    run_train_job(job_id="ghost", plugin_class=EchoTrainPlugin, train_kwargs={}, store=store, heartbeat_s=60)
    assert store.get("ghost") is None
