"""SubmitPredictJobUseCase / GetPredictJobUseCase (async batch prediction)."""
from __future__ import annotations

import pytest

from app.application.use_cases.predict_job_use_cases import (
    GetPredictJobUseCase,
    SubmitPredictJobUseCase,
)
from app.domain.services.exceptions import PredictJobBusyError
from app.domain.services.predict_job import (
    PredictJobRecord,
    PredictJobStatus,
    predict_error_http_status,
    validate_job_id,
)
from tests.unit.predict_job_fakes import (
    EXTRA_PREDICT_EXCEPTIONS,
    PREDICT_ERRORS,
    EchoPredictPlugin,
    FakeBatchRequest,
    InMemoryPredictJobStore,
    RecordingPredictExecutor,
)

MODEL = "fake-model"


def _submit(store=None, executor=None, clock=lambda: 1000.0):
    store = store if store is not None else InMemoryPredictJobStore()
    executor = executor if executor is not None else RecordingPredictExecutor()
    use_case = SubmitPredictJobUseCase(
        EchoPredictPlugin(), MODEL, store, executor,
        extra_predict_exceptions=EXTRA_PREDICT_EXCEPTIONS, clock=clock,
    )
    return use_case, store, executor


REQUEST = FakeBatchRequest(data_path="s3://b/v.mp4", mlflow_run_id="user-run")


def test_submit_queues_the_job_and_starts_it():
    use_case, store, executor = _submit()
    record = use_case.execute(REQUEST, job_id="task-1")
    assert record.status is PredictJobStatus.QUEUED
    assert (record.job_id, record.model_id, record.submitted_at) == ("task-1", MODEL, 1000.0)
    assert store.get("task-1") == record
    assert store.infos["task-1"] == {"data_path": "s3://b/v.mp4", "mlflow_run_id": "user-run"}
    [sub] = executor.submitted
    assert sub["plugin_class"] is EchoPredictPlugin
    assert sub["request"] is REQUEST
    assert sub["extra_predict_exceptions"] == EXTRA_PREDICT_EXCEPTIONS


def test_submit_is_idempotent_by_job_id_even_when_busy():
    use_case, _, executor = _submit()
    first = use_case.execute(REQUEST, job_id="task-1")
    executor.busy = True
    assert use_case.execute(REQUEST, job_id="task-1") == first
    assert len(executor.submitted) == 1


def test_submit_without_job_id_generates_one():
    use_case, _, _ = _submit()
    a, b = use_case.execute(REQUEST), use_case.execute(REQUEST)
    assert a.job_id != b.job_id and validate_job_id(a.job_id) == a.job_id


def test_submit_busy_is_409():
    use_case, store, _ = _submit(executor=RecordingPredictExecutor(busy=True))
    with pytest.raises(PredictJobBusyError):
        use_case.execute(REQUEST, job_id="task-1")
    assert store.get("task-1") is None


@pytest.mark.parametrize("bad", ["a b", "x'y", "../etc", "ñ", "a" * 129])
def test_submit_rejects_unsafe_job_ids(bad):
    use_case, _, _ = _submit()
    with pytest.raises(ValueError):
        use_case.execute(REQUEST, job_id=bad)


def test_submit_rejects_a_job_id_of_another_model():
    use_case, store, _ = _submit()
    store.save(PredictJobRecord("task-1", "other", PredictJobStatus.QUEUED, 1.0, 1.0))
    with pytest.raises(ValueError, match="other"):
        use_case.execute(REQUEST, job_id="task-1")


def test_submit_failure_to_start_is_a_failed_job():
    use_case, store, _ = _submit(executor=RecordingPredictExecutor(fail_on_submit=OSError("fork")))
    record = use_case.execute(REQUEST, job_id="task-1")
    assert record.status is PredictJobStatus.FAILED
    assert (record.error_type, record.http_status) == ("OSError", 500)
    assert store.get("task-1") == record


@pytest.mark.parametrize("key", sorted(PREDICT_ERRORS))
def test_error_http_status_matches_the_synchronous_predict(key):
    exc, code = PREDICT_ERRORS[key]
    assert predict_error_http_status(exc, EXTRA_PREDICT_EXCEPTIONS) == code


def _get(store, now):
    return GetPredictJobUseCase(
        MODEL, store, clock=lambda: now, stale_after_s=600, max_duration_s=43200
    )


def _store_with(status, submitted_at=1000.0, updated_at=1000.0, model_id=MODEL):
    store = InMemoryPredictJobStore()
    store.save(PredictJobRecord("task-1", model_id, status, submitted_at, updated_at))
    return store


def test_get_unknown_other_model_or_invalid_is_none():
    assert _get(InMemoryPredictJobStore(), 1000.0).execute("task-1") is None
    assert _get(_store_with(PredictJobStatus.RUNNING, model_id="x"), 1000.0).execute("task-1") is None
    assert _get(InMemoryPredictJobStore(), 1000.0).execute("bad id") is None


def test_get_running_job_with_recent_heartbeat_is_unchanged():
    store = _store_with(PredictJobStatus.RUNNING, updated_at=1500.0)
    assert _get(store, 2000.0).execute("task-1").status is PredictJobStatus.RUNNING


def test_get_marks_a_silent_job_lost_and_persists_it():
    store = _store_with(PredictJobStatus.RUNNING, updated_at=1000.0)
    record = _get(store, 1601.0).execute("task-1")
    assert (record.status, record.error_type, record.http_status) == (
        PredictJobStatus.FAILED, "PredictJobLost", 500
    )
    assert "predicción" in record.error
    assert store.get("task-1") == record


def test_get_marks_an_overlong_job_timed_out_even_if_heart_beating():
    store = _store_with(PredictJobStatus.RUNNING, submitted_at=0.0, updated_at=43300.0)
    record = _get(store, 43301.0).execute("task-1")
    assert (record.status, record.error_type) == (PredictJobStatus.FAILED, "PredictJobTimeout")
    assert "12 h" in record.error


def test_get_never_changes_a_terminal_job():
    store = _store_with(PredictJobStatus.SUCCEEDED, updated_at=0.0)
    assert _get(store, 10**9).execute("task-1").status is PredictJobStatus.SUCCEEDED
