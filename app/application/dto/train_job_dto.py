"""Response body of the async training endpoints (POST /train?wait=false, GET /train/{job_id})."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.services.train_job import TrainJobRecord


def _utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


class TrainJobResponse(BaseModel):
    """State of an async training job. ``job_id`` is the MLflow run id sent in the TrainRequest."""

    job_id: str
    model_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    submitted_at: datetime
    updated_at: datetime
    result: dict | None = Field(default=None, description="TrainResponse del plugin, solo si succeeded")
    error: str | None = Field(default=None, description="Mensaje de error, solo si failed")
    error_type: str | None = Field(
        default=None,
        description="Clase del error (ValueError, FileNotFoundError, TrainingNotSupportedError, TrainJobLost…)",
    )

    @classmethod
    def from_record(cls, record: TrainJobRecord) -> "TrainJobResponse":
        return cls(
            job_id=record.job_id,
            model_id=record.model_id,
            status=record.status.value,
            submitted_at=_utc(record.submitted_at),
            updated_at=_utc(record.updated_at),
            result=record.result,
            error=record.error,
            error_type=record.error_type,
        )
