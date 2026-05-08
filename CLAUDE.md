# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run dev server (auto-reload)
python main.py

# Run with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker build -t modelsinference-svc .
docker run -p 8000:8000 -e APP_ENV=dev modelsinference-svc

# Tests
python -m pytest
python -m pytest --cov=app
python -m pytest tests/unit/test_foo.py::TestClass::test_method  # single test

# Lint
flake8 app/                # max-line-length=120, E501 ignored
```

## Architecture

This is a **FastAPI ML inference service** with a plugin architecture. All models are served through a single app instance under `/models/<model-id>/`.

### Request lifecycle

```
HTTP → FastAPI router (router_factory.py)
      → ModelContainer (container.py) [DI wiring]
        → PredictModelUseCase (predict_model_use_case.py)
          → ModelPluginPort (abstract port)
            → Concrete plugin (app/plugins/<name>/plugin.py)
```

### Key files

| File | Role |
|------|------|
| `main.py` | App entry point; loads REGISTRY, mounts routers, initializes containers in lifespan |
| `app/registry.py` | **Only file to edit when adding a model** — declares all `ModelEntry` items |
| `app/infrastructure/http/router_factory.py` | Generates `/health`, `/stats`, `/predict`, `/train` endpoints for any model |
| `app/infrastructure/http/dependencies/container.py` | `ModelContainer` wires plugin → service → use cases |
| `app/infrastructure/artifact_store.py` | Resolves artifact file paths; downloads from S3 when `S3_BUCKET` env var is set |
| `app/domain/ports/model_plugin_port.py` | Abstract interface every plugin must implement |

### Plugin structure

Each plugin lives in `app/plugins/<name>/` with these files:

- `plugin.py` — implements `ModelPluginPort` (load, is_loaded, predict_batch, predict_inline, stats)
- `model_loader.py` — loads artifacts via `ArtifactStore`
- `preprocessing.py` — feature transformation before inference
- `postprocessing.py` — output transformation after inference
- `predict_dto.py` — Pydantic models for `PredictRequest`/`PredictResponse` (batch + inline variants)

### Adding a new model

1. Create `app/plugins/<name>/` with the five files above.
2. Add a `ModelEntry` to `REGISTRY` in `app/registry.py` — that's the only central change required.

### Predict modes

Every model supports two modes, discriminated by the `mode` field in the request body:

- `"batch"` — pass `data_path` (path to CSV inside the container); returns list of predictions
- `"inline"` — pass individual feature fields; returns a single prediction dict

### Artifact storage

Artifacts are stored under `artifacts/<model_name>/`. If `S3_BUCKET` is set, missing files are auto-downloaded from `s3://<S3_BUCKET>/<S3_PREFIX>/<model_name>/<file>` (default prefix: `artifacts`).

### Environment variables

| Variable | Description |
|----------|-------------|
| `APP_ENV` | `dev` / `test` / `prod` |
| `HOST` | Bind host (default `0.0.0.0`) |
| `PORT` | Bind port (default `8000`) |
| `LOG_LEVEL` | Logging level |
| `S3_BUCKET` | If set, enables auto-download of artifacts from S3 |
| `S3_PREFIX` | S3 key prefix (default `artifacts`) |

### API docs

Swagger UI is available at `http://localhost:8000/docs` when the app is running.

### Registered models

| model-id | Plugin class |
|----------|-------------|
| `wine-price-fluctuation` | `WinePriceFluctuationPlugin` |
| `cereal-price-forecast` | `CerealPriceForecastPlugin` |
| `meat-price-forecast` | `MeatPriceForecastPlugin` |
| `cnn-fungal-detection` | `CnnFungalDetectionPlugin` |
| `cnn-thermal-scm` | `CnnThermalScmPlugin` |
| `cow-behavior` | `CowBehaviorPlugin` |
| `wine-sulphite` | `WineSulphitePlugin` |