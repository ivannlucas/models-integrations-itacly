import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from app.application.dto.stats_dto import StatsResponse
from app.application.dto.train_dto import TrainRequest as _DefaultTrainRequest
from app.domain.services.exceptions import TrainingNotSupportedError

logger = logging.getLogger(__name__)


def make_model_router(
    model_id: str,
    version: str,
    predict_request_type: Any,
    predict_response_type: Any,
    extra_predict_exceptions: tuple[type[Exception], ...] = (),
    train_request_type: Any = None,
    train_response_type: Any = None,
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

    @router.get("/health")
    async def health(request: Request) -> dict:
        """Return liveness status and whether the model artifacts are loaded."""
        container = request.app.state.containers[model_id]
        return {
            "status": "ok",
            "model": model_id,
            "version": version,
            "loaded": container.service.is_loaded(),
        }

    @router.get("/stats", response_model=StatsResponse)
    async def stats(request: Request) -> StatsResponse:
        """Return model metadata, input/output schema, and runtime statistics."""
        return request.app.state.containers[model_id].stats_use_case.execute()

    @router.post("/predict", response_model=predict_response_type)
    async def predict(request: Request, body: predict_request_type) -> predict_response_type:
        """Run inline or batch prediction and return the model's response."""
        container = request.app.state.containers[model_id]
        try:
            return container.predict_use_case.execute(body)
        except Exception as exc:
            if extra_predict_exceptions and isinstance(exc, extra_predict_exceptions):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
                ) from exc
            logger.exception("Unexpected error during prediction for model '%s'", model_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc

    _train_req_type = train_request_type or _DefaultTrainRequest
    _train_resp_type = train_response_type

    @router.post("/train", response_model=_train_resp_type)
    async def train(request: Request, body: _train_req_type) -> Any:
        """Trigger model training with the CSV at *body.data_path*."""
        container = request.app.state.containers[model_id]
        try:
            return container.train_use_case.execute(data_path=body.data_path)
        except TrainingNotSupportedError as exc:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error during training for model '%s'", model_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc

    return router
