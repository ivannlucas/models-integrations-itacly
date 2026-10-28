"""ProcessTrainExecutor with real `spawn` child processes and a file-backed store."""
from __future__ import annotations

import logging
import os
import time

import pytest

from app.domain.services.train_job import TrainJobRecord, TrainJobStatus
from app.infrastructure.train_jobs import process_executor
from app.infrastructure.train_jobs.process_executor import ProcessTrainExecutor, run_train_job
from tests.unit.train_job_fakes import (
    TRAIN_ERRORS,
    EchoTrainPlugin,
    FailingTrainPlugin,
    FileJobStore,
    InMemoryJobStore,
    NiceReportingTrainPlugin,
    NumpyMetricsTrainPlugin,
    RaisingTrainPlugin,
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
    assert record.http_status == 400
    assert "s3://b/missing.csv" in record.error


@pytest.mark.parametrize("data_path", sorted(TRAIN_ERRORS))
def test_failure_stores_the_http_status_of_the_synchronous_train(tmp_path, data_path):
    exc, expected = TRAIN_ERRORS[data_path]
    store = FileJobStore(str(tmp_path))
    _queue(store, "run-h")
    run_train_job(job_id="run-h", plugin_class=RaisingTrainPlugin,
                  train_kwargs={"data_path": data_path, "mlflow_run_id": "run-h"}, store=store, heartbeat_s=60)
    record = store.get("run-h")
    assert record.status is TrainJobStatus.FAILED
    assert record.error_type == type(exc).__name__  # unchanged: still the exact class name
    assert record.http_status == expected


def test_success_has_no_http_status(tmp_path):
    store = FileJobStore(str(tmp_path))
    _queue(store, "run-ok")
    run_train_job(job_id="run-ok", plugin_class=EchoTrainPlugin,
                  train_kwargs={"data_path": "x", "mlflow_run_id": "run-ok"}, store=store, heartbeat_s=60)
    assert store.get("run-ok").http_status is None


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


def test_hung_training_is_stopped_after_max_duration_and_frees_the_model(tmp_path):
    store = FileJobStore(str(tmp_path))
    executor = ProcessTrainExecutor(heartbeat_s=0.2, max_duration_s=2)
    _submit(executor, store, "run-8", SlowTrainPlugin, data_path="120")
    proc = executor._procs["run-8"]
    record = _terminal(store, "run-8")
    assert record.status is TrainJobStatus.FAILED
    assert record.error_type == "TrainJobTimeout"
    assert record.http_status == 500
    assert "duración máxima" in record.error
    _wait(lambda: not proc.is_alive(), timeout=15, what="child stopped")
    assert not executor.is_busy()


def test_training_child_runs_at_lower_cpu_priority(tmp_path):
    store, executor = FileJobStore(str(tmp_path)), ProcessTrainExecutor(heartbeat_s=0.2, nice=5)
    _submit(executor, store, "run-9", NiceReportingTrainPlugin)
    record = _terminal(store, "run-9")
    assert record.status is TrainJobStatus.SUCCEEDED, record.error
    assert record.result["metrics"]["nice"] == min(os.nice(0) + 5, 19)


def test_renice_failure_is_logged_not_raised(monkeypatch, caplog):
    def deny(_):
        raise PermissionError("not allowed")

    monkeypatch.setattr(process_executor.os, "nice", deny)
    with caplog.at_level(logging.WARNING):
        process_executor._lower_priority(10)
    assert "priority" in caplog.text


class _FakeProc:
    """Process double for the watchdog: ``dies_on`` is the signal method that ends it."""

    pid = 4242

    def __init__(self, dies_on: str = "terminate", finishes: bool = False) -> None:
        self.alive = not finishes
        self.dies_on = dies_on
        self.calls: list[str] = []

    def join(self, timeout=None):
        self.calls.append("join")

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.calls.append("terminate")
        if self.dies_on == "terminate":
            self.alive = False

    def kill(self):
        self.calls.append("kill")
        self.alive = False


def _queued_store(status=TrainJobStatus.RUNNING, **extra):
    store = InMemoryJobStore()
    store.save(TrainJobRecord("run-w", "fake-model", status, 1.0, 1.0, **extra))
    return store


def test_watchdog_does_nothing_when_the_child_ends_in_time():
    proc, store = _FakeProc(finishes=True), _queued_store()
    process_executor._watch(proc, "run-w", store, 5)
    assert proc.calls == ["join"]
    assert store.get("run-w").status is TrainJobStatus.RUNNING


def test_watchdog_kills_a_child_that_ignores_sigterm():
    proc, store = _FakeProc(dies_on="kill"), _queued_store()
    process_executor._watch(proc, "run-w", store, 5)
    assert proc.calls == ["join", "terminate", "join", "kill", "join"]
    record = store.get("run-w")
    assert (record.status, record.error_type, record.http_status) == (TrainJobStatus.FAILED, "TrainJobTimeout", 500)


def test_watchdog_never_overwrites_a_terminal_job():
    proc = _FakeProc()
    store = _queued_store(TrainJobStatus.FAILED, error="lost", error_type="TrainJobLost", http_status=500)
    process_executor._watch(proc, "run-w", store, 5)
    assert "terminate" in proc.calls  # the child is still stopped
    assert store.get("run-w").error_type == "TrainJobLost"


@pytest.mark.parametrize("workers,warns", [("4", True), ("1", False), ("", False)])
def test_more_than_one_worker_is_warned_once(monkeypatch, caplog, workers, warns):
    monkeypatch.setenv("WEB_CONCURRENCY", workers)
    process_executor._warn_if_multiple_workers.cache_clear()
    with caplog.at_level(logging.WARNING):
        ProcessTrainExecutor()
        ProcessTrainExecutor()
    process_executor._warn_if_multiple_workers.cache_clear()
    assert caplog.text.count("1 worker and 1 replica") == (1 if warns else 0)
