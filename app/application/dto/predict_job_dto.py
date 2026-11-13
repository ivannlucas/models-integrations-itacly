"""Response body of the async prediction endpoints.

POST /predict?wait=false and GET /predict/jobs/{job_id}.
"""
from __future__ import annotations

from pydantic import Field

from app.application.dto.train_job_dto import TrainJobResponse


class PredictJobResponse(TrainJobResponse):
    """State of an async batch prediction. ``job_id`` is the one sent in ``?job_id=``."""

    result: dict | None = Field(
        default=None,
        description="Cuerpo que habría devuelto POST /predict síncrono, solo si succeeded",
    )
    error_type: str | None = Field(
        default=None,
        description="Clase del error (InvalidVideoError, PredictJobLost, PredictJobTimeout…)",
    )
    http_status: int | None = Field(
        default=None,
        description=(
            "Código que habría devuelto POST /predict síncrono (413, 422, 500), solo si failed"
        ),
    )
