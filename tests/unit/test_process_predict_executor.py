"""ProcessPredictExecutor with real `spawn` child processes and a file-backed store."""
from __future__ import annotations

import os
import time

from app.domain.services.predict_job import PredictJobRecord, PredictJobStatus
from app.infrastructure.predict_jobs.process_executor import ProcessPredictExecutor, run_predict_job
from tests.unit.predict_job_fakes import (
    EXTRA_PREDICT_EXCEPTIONS,
    EchoPredictPlugin,
    FakeBatchRequest,
    FilePredictJobStore,
    InMemoryPredictJobStore,
    SlowPredictPlugin,
)


def _queue(store, job_id):
    now = time.time()
    store.save(PredictJobRecord(job_id, "fake-model", PredictJobStatus.QUEUED, now, now))


def _terminal(store, job_id, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = store.get(job_id)
        if record is not None and record.status.is_terminal:
            return record
        time.sleep(0.1)
    raise AssertionError(f"timeout waiting for {job_id}")


def test_predicts_in_a_child_process_and_stores_the_json_result(tmp_path):
    store, executor = FilePredictJobStore(str(tmp_path)), ProcessPredictExecutor(heartbeat_s=0.2)
    _queue(store, "t-1")
    executor.submit(
        job_id="t-1", plugin_class=EchoPredictPlugin,
        request=FakeBatchRequest(data_path="s3://b/v.mp4", mlflow_run_id="user-run"),
        store=store, extra_predict_exceptions=EXTRA_PREDICT_EXCEPTIONS,
    )
    record = _terminal(store, "t-1")
    assert record.status is PredictJobStatus.SUCCEEDED, record.error
    rows = record.result["predictions"]
    assert rows[0]["frame"] == 0 and rows[0]["run"] == "user-run"
    assert rows[0]["pid"] != os.getpid()


def test_concurrency_limit_and_watchdog(tmp_path):
    store = FilePredictJobStore(str(tmp_path))
    executor = ProcessPredictExecutor(heartbeat_s=0.2, max_duration_s=3.0, max_concurrent=1)
    assert not executor.is_busy()
    _queue(store, "t-slow")
    executor.submit(
        job_id="t-slow", plugin_class=SlowPredictPlugin,
        request=FakeBatchRequest(data_path="x"), store=store,
    )
    assert executor.is_busy()
    record = _terminal(store, "t-slow")
    assert (record.status, record.error_type, record.http_status) == (
        PredictJobStatus.FAILED, "PredictJobTimeout", 500
    )
    deadline = time.time() + 20
    while executor.is_busy() and time.time() < deadline:
        time.sleep(0.1)
    assert not executor.is_busy()


def test_run_predict_job_never_overwrites_a_terminal_job():
    store = InMemoryPredictJobStore()
    store.save(PredictJobRecord("t-1", "fake-model", PredictJobStatus.FAILED, 1.0, 1.0,
                                error_type="PredictJobLost"))
    run_predict_job(job_id="t-1", plugin_class=EchoPredictPlugin,
                    request=FakeBatchRequest(data_path="x"), store=store)
    assert store.get("t-1").error_type == "PredictJobLost"
