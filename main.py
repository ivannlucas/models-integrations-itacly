import logging
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading %d model(s)...", len(REGISTRY))
    app.state.containers = {}
    for entry in REGISTRY:
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
    logger.info("Startup complete. %d/%d models ready.", len(app.state.containers), len(REGISTRY))
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

for _entry in REGISTRY:
    _router = make_model_router(
        model_id=_entry.model_id,
        version=_entry.version,
        predict_request_type=_entry.predict_request_type,
        predict_response_type=_entry.predict_response_type,
        extra_predict_exceptions=_entry.extra_predict_exceptions,
    )
    app.include_router(_router, prefix=_entry.prefix, tags=[_entry.model_id])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
