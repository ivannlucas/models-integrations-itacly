"""Async training use cases: submit (idempotent, busy, validation) and get (stale detection)."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.application.use_cases.train_job_use_cases import GetTrainJobUseCase, SubmitTrainJobUseCase
from app.application.use_cases.train_model_use_case import build_train_kwargs
from app.domain.services.exceptions import TrainJobBusyError
from app.domain.services.train_job import TrainJobRecord, TrainJobStatus
from tests.unit.train_job_fakes import EchoTrainPlugin, InMemoryJobStore, RecordingExecutor, SystemTrainPlugin

MODEL_ID = "fake-model"


def _request(run_id: str = "run-1", data_path: str = "s3://bucket/train.csv", **extra):
    return SimpleNamespace(mlflow_run_id=run_id, data_path=data_path, **extra)


def _submit(store=None, executor=None, plugin=None, now=1000.0):
    store = store if store is not None else InMemoryJobStore()
    executor = executor if executor is not None else RecordingExecutor()
    plugin = plugin if plugin is not None else EchoTrainPlugin()
    return SubmitTrainJobUseCase(plugin, MODEL_ID, store, executor, clock=lambda: now), store, executor


class TestBuildTrainKwargs:
    def test_basic_kwargs(self):
        assert build_train_kwargs(EchoTrainPlugin(), _request()) == {
            "data_path": "s3://bucket/train.csv", "mlflow_run_id": "run-1",
        }

    def test_forwards_system_only_when_declared(self):
        kwargs = build_train_kwargs(SystemTrainPlugin(), _request(system="aireado"))
        assert kwargs["system"] == "aireado"


class TestSubmit:
    def test_creates_queued_job_and_submits_plugin_class(self):
        use_case, store, executor = _submit()
        record = use_case.execute(_request())
        assert record == TrainJobRecord(
            job_id="run-1", model_id=MODEL_ID, status=TrainJobStatus.QUEUED, submitted_at=1000.0, updated_at=1000.0,
        )
        assert store.get("run-1") == record
        assert executor.submitted == [{
            "job_id": "run-1", "plugin_class": EchoTrainPlugin,
            "train_kwargs": {"data_path": "s3://bucket/train.csv", "mlflow_run_id": "run-1"},
        }]

    @pytest.mark.parametrize("run_id", ["", "   "])
    def test_empty_run_id_is_rejected(self, run_id):
        use_case, store, executor = _submit()
        with pytest.raises(ValueError, match="mlflow_run_id"):
            use_case.execute(_request(run_id=run_id))
        assert store.records == {} and executor.submitted == []

    def test_resubmitting_same_job_returns_existing_and_does_not_start_again(self):
        use_case, store, executor = _submit()
        first = use_case.execute(_request())
        store.save(replace(first, status=TrainJobStatus.RUNNING, updated_at=1010.0))
        executor.busy = True  # the first one is still alive: a redelivery must NOT get a 409
        again = use_case.execute(_request())
        assert again.status is TrainJobStatus.RUNNING
        assert len(executor.submitted) == 1

    def test_job_id_of_another_model_is_rejected(self):
        use_case, store, _ = _submit()
        store.save(TrainJobRecord("run-1", "other-model", TrainJobStatus.SUCCEEDED, 1.0, 2.0))
        with pytest.raises(ValueError, match="other-model"):
            use_case.execute(_request())

    def test_busy_executor_raises_and_stores_nothing(self):
        use_case, store, _ = _submit(executor=RecordingExecutor(busy=True))
        with pytest.raises(TrainJobBusyError):
            use_case.execute(_request(run_id="run-2"))
        assert store.get("run-2") is None

    def test_submit_failure_marks_job_failed(self):
        use_case, store, _ = _submit(executor=RecordingExecutor(fail_on_submit=OSError("no fork")))
        record = use_case.execute(_request())
        assert record.status is TrainJobStatus.FAILED
        assert record.error_type == "OSError"
        assert "no fork" in record.error
        assert store.get("run-1") == record


class TestGet:
    def _store_with(self, status, updated_at):
        store = InMemoryJobStore()
        store.save(TrainJobRecord("run-1", MODEL_ID, status, 1000.0, updated_at))
        return store

    def test_unknown_job_returns_none(self):
        assert GetTrainJobUseCase(MODEL_ID, InMemoryJobStore()).execute("nope") is None

    def test_job_of_another_model_returns_none(self):
        store = InMemoryJobStore()
        store.save(TrainJobRecord("run-1", "other-model", TrainJobStatus.RUNNING, 1.0, 1.0))
        assert GetTrainJobUseCase(MODEL_ID, store).execute("run-1") is None

    def test_fresh_running_job_is_returned_as_is(self):
        store = self._store_with(TrainJobStatus.RUNNING, updated_at=1500.0)
        record = GetTrainJobUseCase(MODEL_ID, store, clock=lambda: 1600.0, stale_after_s=600).execute("run-1")
        assert record.status is TrainJobStatus.RUNNING

    @pytest.mark.parametrize("status", [TrainJobStatus.QUEUED, TrainJobStatus.RUNNING])
    def test_silent_job_is_marked_lost_and_persisted(self, status):
        store = self._store_with(status, updated_at=1000.0)
        record = GetTrainJobUseCase(MODEL_ID, store, clock=lambda: 1601.0, stale_after_s=600).execute("run-1")
        assert record.status is TrainJobStatus.FAILED
        assert record.error_type == "TrainJobLost"
        assert store.get("run-1").status is TrainJobStatus.FAILED

    def test_terminal_job_is_never_marked_lost(self):
        store = self._store_with(TrainJobStatus.SUCCEEDED, updated_at=1000.0)
        record = GetTrainJobUseCase(MODEL_ID, store, clock=lambda: 99999.0, stale_after_s=600).execute("run-1")
        assert record.status is TrainJobStatus.SUCCEEDED

    def test_stale_window_defaults_to_env(self, monkeypatch):
        monkeypatch.setenv("TRAIN_JOB_STALE_AFTER_S", "10")
        store = self._store_with(TrainJobStatus.RUNNING, updated_at=1000.0)
        record = GetTrainJobUseCase(MODEL_ID, store, clock=lambda: 1011.0).execute("run-1")
        assert record.status is TrainJobStatus.FAILED
