import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.infrastructure.http.dependencies.container import ModelContainer
from app.infrastructure.http.router_factory import make_model_router
from app.registry import REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# When MODEL is set (e.g. by Kubernetes via configmap), only that plugin is loaded.
# Without MODEL, all registry entries are loaded (useful for local development).
_model_filter = os.getenv("MODEL")
if _model_filter:
    logger.info("MODEL env var set — loading only '%s'.", _model_filter)
_active_entries = [e for e in REGISTRY if not _model_filter or e.model_id == _model_filter]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading %d model(s)...", len(_active_entries))
    app.state.containers = {}
    for entry in _active_entries:
        try:
            container = ModelContainer(
                plugin=entry.plugin_class(),
                batch_response_cls=entry.batch_response_class,
                inline_response_cls=entry.inline_response_class,
            )
            container.init()
            app.state.containers[entry.model_id] = container
            logger.info("Model '%s' loaded successfully.", entry.model_id)
        except Exception:
            logger.exception("Failed to load model '%s' — it will be unavailable.", entry.model_id)
    logger.info("Startup complete. %d/%d models ready.", len(app.state.containers), len(_active_entries))
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Luce ML Models API",
    version="1.0.0",
    description=(
        "Central API for all Luce ML model runtimes. "
        "Each model is available under /models/<model-id>/."
    ),
    lifespan=lifespan,
)

for _entry in _active_entries:
    _router = make_model_router(
        model_id=_entry.model_id,
        version=_entry.version,
        predict_request_type=_entry.predict_request_type,
        predict_response_type=_entry.predict_response_type,
        extra_predict_exceptions=_entry.extra_predict_exceptions,
        train_request_type=_entry.train_request_type,
        train_response_type=_entry.train_response_type,
    )
    app.include_router(_router, prefix=_entry.prefix, tags=[_entry.model_id])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
