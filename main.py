"""Application entry point for the Luce ML Models API.

Creates and configures the FastAPI application, loading model plugins
based on the registry and the optional MODEL environment variable.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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
    """Manage the application lifespan: load models on startup, clean up on shutdown."""
    logger.info("Starting up — loading %d model(s)...", len(_active_entries))
    app.state.containers = {}
    app.state.load_errors = {}
    for entry in _active_entries:
        try:
            container = ModelContainer(plugin=entry.plugin_class())
            container.init()
            app.state.containers[entry.model_id] = container
            logger.info("Model '%s' loaded successfully.", entry.model_id)
        except Exception as exc:
            logger.exception("Failed to load model '%s' — it will be unavailable.", entry.model_id)
            app.state.load_errors[entry.model_id] = str(exc)
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


@app.get("/health", tags=["health"])
async def health() -> JSONResponse:
    """Global health check used by Kubernetes liveness and readiness probes.

    Returns a non-2xx status if any active model failed to load, so K8s probes
    actually detect and act on it — a pod stuck permanently reporting HTTP 200
    while every model call 503s is what caused failures to go unnoticed before.
    """
    containers = getattr(app.state, "containers", {})
    load_errors = getattr(app.state, "load_errors", {})
    models_status = {mid: c.service.is_loaded() for mid, c in containers.items()}
    all_active_loaded = len(containers) == len(_active_entries) and all(models_status.values())
    return JSONResponse(
        status_code=200 if all_active_loaded else 503,
        content={
            "status": "ok" if all_active_loaded else "error",
            "models": models_status,
            "load_errors": load_errors,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a diagnosable JSON body for any otherwise-unhandled exception.

    Without this, e.g. indexing a missing model container raises a bare
    KeyError that FastAPI's default handling turns into an opaque
    "Internal Server Error" with no detail — making load/routing bugs
    indistinguishable from real prediction errors to callers.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


for _entry in _active_entries:
    _router = make_model_router(
        model_id=_entry.model_id,
        version=_entry.version,
        predict_request_type=_entry.predict_request_type,
        predict_response_type=_entry.predict_response_type,
        train_request_type=_entry.train_request_type,
        train_response_type=_entry.train_response_type,
        extra_predict_exceptions=_entry.extra_predict_exceptions,
    )
    app.include_router(_router, prefix=_entry.prefix, tags=[_entry.model_id])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
