"""Factory for creating FastAPI routers per model plugin."""
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.application.dto.stats_dto import StatsResponse
from app.application.dto.train_dto import TrainRequest as _DefaultTrainRequest, TrainResponse as _DefaultTrainResponse
from app.domain.services.exceptions import TrainingNotSupportedError

logger = logging.getLogger(__name__)


def make_model_router(
    model_id: str,
    version: str,
    predict_request_type: Any,
    predict_response_type: Any,
    train_request_type: Any,
    train_response_type: Any,
    extra_predict_exceptions: tuple[type[Exception], ...] = (),
) -> APIRouter:
    """Create a FastAPI router for a model plugin.

    All endpoints share the same contract (health / stats / predict / train).
    The concrete request/response Pydantic types are injected per model so that
    Swagger docs reflect each model's actual schema.

    Adding a new model only requires:
    1. Implementing ModelPluginPort in app/plugins/<name>/plugin.py
    2. Adding a ModelEntry to app/registry.py
    """
    router = APIRouter()
    _train_req = train_request_type or _DefaultTrainRequest
    _train_resp = train_response_type or _DefaultTrainResponse

    def _get_container(request: Request):
        """Return the model's container or raise a 503 with the real load failure reason.

        Models that failed to load during startup are never added to
        ``app.state.containers`` — indexing it directly would raise an unhandled
        ``KeyError`` (a bare 500 with no diagnostic detail) for every endpoint.
        """
        container = request.app.state.containers.get(model_id)
        if container is None:
            reason = getattr(request.app.state, "load_errors", {}).get(model_id, "unknown error")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model '{model_id}' failed to load and is unavailable: {reason}",
            )
        return container

    @router.get("/health")
    async def health(request: Request) -> dict:
        """Return health status for the model, including load state and version."""
        container = request.app.state.containers.get(model_id)
        if container is None:
            reason = getattr(request.app.state, "load_errors", {}).get(model_id, "unknown error")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "error",
                    "model": model_id,
                    "version": version,
                    "loaded": False,
                    "error": reason,
                },
            )
        return {
            "status": "ok",
            "model": model_id,
            "version": version,
            "loaded": container.service.is_loaded(),
        }

    @router.get("/stats")
    async def stats(request: Request, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        if mlflow_run_id:
            logger.info("Stats requested with mlflow_run_id=%s for model '%s'", mlflow_run_id, model_id)
        return _get_container(request).stats_use_case.execute(mlflow_run_id=mlflow_run_id)

    @router.post("/predict")
    def predict(request: Request, body: predict_request_type) -> predict_response_type:
        """Run prediction (inline or batch) and return typed response."""
        container = _get_container(request)
        try:
            return container.predict_use_case.execute(body)
        except Exception as exc:
            if extra_predict_exceptions and isinstance(exc, extra_predict_exceptions):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
                ) from exc
            logger.exception("Unexpected error during prediction for model '%s'", model_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc

    @router.post("/train")
    def train(request: Request, body: _train_req) -> _train_resp:
        """Trigger model training with the provided data."""
        container = _get_container(request)
        try:
            return container.train_use_case.execute(body)
        except TrainingNotSupportedError as exc:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)
            ) from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error during training for model '%s'", model_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc

    return router
