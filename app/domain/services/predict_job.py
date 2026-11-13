"""State of an asynchronous batch prediction job (POST /predict?wait=false).

Same lifecycle and fields as an async training job (``TrainJobRecord``): the job only differs in
what the child process runs and where the store keeps it. ``job_id`` is chosen by the client (the
orchestrator sends its task id), since a prediction has no MLflow run created beforehand.
"""
from __future__ import annotations

import re

from app.domain.services.exceptions import ArchiveLimitExceededError, UserModelUnavailableError
from app.domain.services.train_job import TrainJobRecord, TrainJobStatus

PredictJobStatus = TrainJobStatus
PredictJobRecord = TrainJobRecord

LOST_ERROR_TYPE = "PredictJobLost"
TIMEOUT_ERROR_TYPE = "PredictJobTimeout"

# Goes into an MLflow search filter and a URL path segment: keep it to a safe alphabet.
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def validate_job_id(job_id: str) -> str:
    """Return ``job_id`` stripped, or raise ValueError (→ 400) if its format is not allowed."""
    job_id = (job_id or "").strip()
    if not _JOB_ID_RE.match(job_id):
        raise ValueError(
            "job_id no válido: de 1 a 128 caracteres entre letras, dígitos, '.', '_' y '-'"
        )
    return job_id


def predict_error_http_status(
    exc: BaseException, extra_predict_exceptions: tuple[type[BaseException], ...] = ()
) -> int:
    """HTTP status the synchronous /predict answers for ``exc`` (same order as its handler)."""
    if isinstance(exc, ArchiveLimitExceededError):
        return 413
    if isinstance(exc, UserModelUnavailableError) or (
        extra_predict_exceptions and isinstance(exc, extra_predict_exceptions)
    ):
        return 422
    return 500


def max_duration_error(max_duration_s: float) -> str:
    """User-facing message of a job stopped for exceeding PREDICT_JOB_MAX_DURATION_S."""
    limit = f"{max_duration_s / 3600:g} h" if max_duration_s >= 3600 else f"{max_duration_s:g} s"
    return f"La predicción superó la duración máxima ({limit}) y se ha detenido."


def lost_error(stale_after_s: float) -> str:
    """User-facing message of a job whose process stopped heart-beating."""
    return (
        f"El proceso de predicción dejó de dar señales hace más de {int(stale_after_s)} s "
        "(reinicio del pod o proceso terminado por el sistema)."
    )
