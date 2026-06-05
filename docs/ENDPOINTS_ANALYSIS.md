# Endpoints Analysis — Luce ML Models API

This document inventories every endpoint currently exposed by the service,
the tests that cover them, and the sample payloads used by those tests.

- **Base URL (local):** `http://localhost:8000`
- **Swagger UI:** `http://localhost:8000/docs`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`
- **Test suite:** `tests/unit/` (51 tests, ~11 s on a cold run)
- **Shared fixtures:** `tests/conftest.py` (fake plugins + sample payloads)

---

## 1. Architecture recap

Every model registered in `app/registry.py` gets the same four routes, mounted
under `/models/<model-id>/`:

| Method | Path                          | Status codes              | Purpose                                |
|--------|-------------------------------|---------------------------|----------------------------------------|
| GET    | `/models/<id>/health`         | 200                       | Liveness + loaded flag                 |
| GET    | `/models/<id>/stats`          | 200                       | Runtime metadata + prediction counters |
| POST   | `/models/<id>/predict`        | 200 / 422 / 500           | Inference (inline or batch)            |
| POST   | `/models/<id>/train`          | 501                       | Always "Not implemented"               |

The `/predict` route accepts two request shapes discriminated by the `mode`
field (`"inline"` or `"batch"`).

Exception mapping (see `router_factory.py`):
- Known domain exceptions declared in `ModelEntry.extra_predict_exceptions`
  → **HTTP 422** with the original message.
- Anything else → **HTTP 500** (logged with stack trace).

---

## 2. Registered models

| model-id                  | Prefix                                  | Version | Domain exceptions (422)                                         |
|---------------------------|-----------------------------------------|---------|-----------------------------------------------------------------|
| `wine-price-fluctuation`  | `/models/wine-price-fluctuation`        | 1.0.0   | `InsufficientDataError`                                         |
| `cereal-price-forecast`   | `/models/cereal-price-forecast`         | 1.0.0   | `UnsupportedProductError`                                       |
| `meat-price-forecast`     | `/models/meat-price-forecast`           | 1.0.0   | `InsufficientRowsError`                                         |
| `cnn-fungal-detection`    | `/models/cnn-fungal-detection`          | 1.0.0   | `InvalidImageError`                                             |
| `cnn-thermal-scm`         | `/models/cnn-thermal-scm`               | 1.0.0   | `InvalidImageError`                                             |
| `cow-behavior`            | `/models/cow-behavior`                  | 1.0.0   | `InvalidVideoError`, `InvalidImageError`, `InsufficientFramesError` |
| `wine-sulphite`           | `/models/wine-sulphite`                 | 1.2.0   | `NoValidSimulationPointError`                                   |

---

## 3. Per-model endpoint coverage

For every model, the test suite covers:

- ✅ `GET /health` — 200, `loaded: true`, correct `version` and `model`.
- ✅ `GET /stats` — 200, `model_name` and `predict_count` fields.
- ✅ `POST /predict` (mode=`inline`) — 200 with expected schema.
- ✅ `POST /predict` (mode=`batch`) — 200 with expected schema.
- ✅ `POST /predict` — input validation (422) for at least one required/constrained field.
- ✅ `POST /predict` — domain-exception mapping (422) for every exception declared in `extra_predict_exceptions`.
- ✅ `POST /train` — 501.

Additional cross-cutting assertion (wine-price-fluctuation only, but behaviour is shared):
- ✅ `stats.predict_count` increments after a successful `/predict` call.

### 3.1 `wine-price-fluctuation`

Predicts whether red-wine price will rise by ≥ 2.5% in the next 4 weeks.

**Test file:** `tests/unit/test_wine_price_fluctuation.py` (9 tests)

| Test                                                   | What it proves                                           |
|--------------------------------------------------------|----------------------------------------------------------|
| `test_health`                                          | Router mounted, version 1.0.0, plugin reports loaded     |
| `test_stats`                                           | Stats schema + zero predict_count at cold start          |
| `test_predict_inline`                                  | Inline mode returns a valid `PredictInlineResponse`      |
| `test_predict_inline_rejects_fewer_than_22_records`    | Pydantic `min_length=22` enforced → 422                  |
| `test_predict_batch`                                   | Batch mode returns `predictions` list                    |
| `test_predict_domain_exception_maps_to_422`            | `InsufficientDataError` → 422                            |
| `test_predict_unknown_exception_maps_to_500`           | Unknown exception → 500                                  |
| `test_train_returns_501`                               | `/train` always 501                                      |
| `test_stats_increments_after_predict`                  | Counter wiring                                           |

**Sample inline payload** (24 weekly records, minimum is 22):

```json
{
  "mode": "inline",
  "records": [
    {"campaign": "2023/2024", "week": 1, "price_red": 40.1},
    {"campaign": "2023/2024", "week": 2, "price_red": 40.2}
    /* ...24 total... */
  ]
}
```

**Sample batch payload:**
```json
{"mode": "batch", "data_path": "/tmp/wine_prices.csv"}
```

---

### 3.2 `cereal-price-forecast`

Forecasts market price (EUR/tonne) for 5 cereal products.

**Test file:** `tests/unit/test_cereal_price_forecast.py` (7 tests)

| Test                                               | What it proves                                 |
|----------------------------------------------------|------------------------------------------------|
| `test_health`                                      | Router + loaded                                |
| `test_stats`                                       | Stats schema                                   |
| `test_predict_inline`                              | Inline with pre-computed features              |
| `test_predict_batch`                               | Batch mode                                     |
| `test_predict_unsupported_product_maps_to_422`     | `UnsupportedProductError` → 422                |
| `test_predict_inline_missing_product_name`         | Required field validation → 422                |
| `test_train_returns_501`                           | `/train` → 501                                 |

**Sample inline payload:**

```json
{
  "mode": "inline",
  "product_name": "Milling wheat",
  "market_name": "Valladolid",
  "week_begin_date": "2024-01-15",
  "Year": 2024.0,
  "Month": 1.0,
  "Quarter": 1.0,
  "Week_of_Year": 3.0
}
```

Valid `product_name` values: `"Durum wheat"`, `"Milling wheat"`,
`"Feed barley"`, `"Malting barley"`, `"Feed maize"`.

**Sample batch payload:** `{"mode": "batch", "data_path": "/tmp/cereal.csv"}`

---

### 3.3 `meat-price-forecast`

Forecasts weekly IPC indices for 5 meat targets (bovino, porcino, ovino, ave, carne).

**Test file:** `tests/unit/test_meat_price_forecast.py` (7 tests)

| Test                                              | What it proves                                 |
|---------------------------------------------------|------------------------------------------------|
| `test_health`                                     | Router + loaded                                |
| `test_stats`                                      | Stats schema                                   |
| `test_predict_inline`                             | Inline returns predictions for 5 targets       |
| `test_predict_inline_rejects_fewer_than_4_rows`   | `min_length=4` enforced → 422                  |
| `test_predict_batch`                              | Batch mode                                     |
| `test_predict_insufficient_rows_maps_to_422`      | `InsufficientRowsError` → 422                  |
| `test_train_returns_501`                          | `/train` → 501                                 |

**Sample inline payload** (4 weekly rows, the minimum):

```json
{
  "mode": "inline",
  "rows": [
    {"date": "2024-01-01", "bovino": 130.0, "porcino": 128.0, "ovino": 135.0, "ave": 120.0, "carne": 129.0},
    {"date": "2024-01-08", "bovino": 131.0, "porcino": 129.0, "ovino": 136.0, "ave": 121.0, "carne": 130.0},
    {"date": "2024-01-15", "bovino": 132.0, "porcino": 130.0, "ovino": 137.0, "ave": 122.0, "carne": 131.0},
    {"date": "2024-01-22", "bovino": 133.0, "porcino": 131.0, "ovino": 138.0, "ave": 123.0, "carne": 132.0}
  ],
  "include_lstm": false
}
```

**Sample batch payload:** `{"mode": "batch", "data_path": "/tmp/meat.csv"}`

---

### 3.4 `cnn-fungal-detection`

Binary CNN classifier (healthy vs fungal) on leaf imagery.

**Test file:** `tests/unit/test_cnn_fungal_detection.py` (7 tests)

| Test                                        | What it proves                        |
|---------------------------------------------|---------------------------------------|
| `test_health`                               | Router + loaded                       |
| `test_stats`                                | Stats schema                          |
| `test_predict_inline`                       | Inline with `image_path`              |
| `test_predict_batch`                        | Batch over a directory/zip            |
| `test_predict_invalid_image_maps_to_422`    | `InvalidImageError` → 422             |
| `test_predict_inline_missing_image_path`    | Missing required field → 422          |
| `test_train_returns_501`                    | `/train` → 501                        |

**Sample inline payload:** `{"mode": "inline", "image_path": "/tmp/fake_image.jpg"}`

**Sample batch payload:** `{"mode": "batch", "data_path": "/tmp/images"}`

---

### 3.5 `cnn-thermal-scm`

Binary CNN classifier (Healthy vs SCM) on thermal imagery.

**Test file:** `tests/unit/test_cnn_thermal_scm.py` (6 tests)

| Test                                        | What it proves                                    |
|---------------------------------------------|---------------------------------------------------|
| `test_health`                               | Router + loaded                                   |
| `test_stats`                                | Stats schema                                      |
| `test_predict_inline`                       | Inline returns probabilities that sum to ~1.0     |
| `test_predict_batch`                        | Batch over directory                              |
| `test_predict_invalid_image_maps_to_422`    | `InvalidImageError` → 422                         |
| `test_train_returns_501`                    | `/train` → 501                                    |

**Sample inline payload:** `{"mode": "inline", "image_path": "/tmp/fake_thermal.jpg"}`

**Sample batch payload:** `{"mode": "batch", "data_path": "/tmp/thermal"}`

---

### 3.6 `cow-behavior`

Video clip classifier with anomaly detection (uses ByteTrack + CLIP_LENGTH=32).

**Test file:** `tests/unit/test_cow_behavior.py` (8 tests)

| Test                                                 | What it proves                                       |
|------------------------------------------------------|------------------------------------------------------|
| `test_health`                                        | Router + loaded                                      |
| `test_stats`                                         | Stats schema                                         |
| `test_predict_inline`                                | Inline with 32 base64 frames                         |
| `test_predict_inline_rejects_fewer_than_32_frames`   | `min_length=32` enforced → 422                       |
| `test_predict_batch`                                 | Batch over a video file                              |
| `test_predict_invalid_video_maps_to_422`             | `InvalidVideoError` → 422                            |
| `test_predict_insufficient_frames_maps_to_422`       | `InsufficientFramesError` → 422                      |
| `test_train_returns_501`                             | `/train` → 501                                       |

**Sample inline payload:** 32 dummy base64 strings (content unused in unit tests):

```json
{"mode": "inline", "frames_base64": ["AAAA", "AAAA", ...], "detection_threshold": 0.5}
```

**Sample batch payload:**

```json
{
  "mode": "batch",
  "data_path": "/tmp/cow.mp4",
  "detection_threshold": 0.5,
  "anomaly_threshold": 0.5
}
```

---

### 3.7 `wine-sulphite`

Dual regressor that recommends an SO2 dose to maximise predicted sensory quality.

**Test file:** `tests/unit/test_wine_sulphite.py` (7 tests)

| Test                                                     | What it proves                               |
|----------------------------------------------------------|----------------------------------------------|
| `test_health`                                            | Router + loaded, version 1.2.0               |
| `test_stats`                                             | Stats schema                                 |
| `test_predict_inline`                                    | Recommendation returned with all SO2 fields  |
| `test_predict_inline_missing_required_field`             | Missing `pH` → 422                           |
| `test_predict_batch`                                     | Batch over CSV                               |
| `test_predict_no_valid_simulation_point_maps_to_422`     | `NoValidSimulationPointError` → 422          |
| `test_train_returns_501`                                 | `/train` → 501                               |

**Sample inline payload** (canonical red-wine row):

```json
{
  "mode": "inline",
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.66,
  "citric_acid": 0.0,
  "residual_sugar": 1.8,
  "chlorides": 0.075,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "min_molecular": 0.6,
  "max_total": 200.0,
  "delta_max": 40.0
}
```

**Sample batch payload:** `{"mode": "batch", "data_path": "/tmp/wine.csv"}`

---

## 4. Test design notes

### 4.1 Why fake plugins?

Loading real artifacts in unit tests would:

- Require ~GB of model files on disk (or S3 credentials).
- Drag in TensorFlow, Detectron2, Torch, XGBoost on every test run.
- Make "unit" tests actually integration tests — slow and flaky.

Every test therefore routes through `FakePlugin` (`tests/conftest.py`). The fake
implements `ModelPluginPort`, returns canonical dicts matching each model's
`PredictInlineResponse` / `PredictBatchResponse`, and exposes two knobs —
`raise_on_inline` / `raise_on_batch` — that let a single test force an error
path without affecting the others.

### 4.2 What is actually exercised?

- `make_model_router` routing and HTTP status mapping.
- Pydantic request validation (discriminator, required fields, `min_length`,
  etc.) — this is real, not mocked.
- Pydantic response validation — each fake payload must pass the real
  `inline_response_class` / `batch_response_class` or the test fails.
- Exception mapping in `router_factory.predict` (known → 422, unknown → 500).
- `ModelPluginPort` contract per plugin (every method called at least once).
- DI wiring: `PredictModelUseCase` dispatching on `mode`, `stats_use_case`
  delegating to `plugin.stats()`, `train_use_case` raising.

### 4.3 What is NOT exercised (out of scope for unit tests)

- Real model inference (mathematical correctness).
- Artifact loading from disk / S3 (`ArtifactStore`).
- Preprocessing and postprocessing modules of each plugin.
- Lifespan startup (`main.lifespan`) — the test app is built with pre-loaded
  fake containers.
- Concurrency / performance.

These belong in integration tests (with real artifacts staged) or plugin-level
unit tests (`tests/unit/plugins/<name>/test_preprocessing.py` etc.) — neither
exists yet.

---

## 5. Running the suite

```bash
# full suite
python -m pytest

# single file
python -m pytest tests/unit/test_wine_sulphite.py

# single test
python -m pytest tests/unit/test_wine_sulphite.py::test_predict_inline

# with coverage on the app package
python -m pytest --cov=app --cov-report=term-missing
```

---

## 6. How to extend when adding a new model

1. Add a `ModelEntry` to `app/registry.py` (see `CLAUDE.md`).
2. In `tests/conftest.py`:
   - Write `_<name>_inline` and `_<name>_batch` factory functions returning
     dicts compatible with the new plugin's response schemas.
   - Register them in `FAKE_FACTORIES[...]`.
   - Add a sample inline payload fixture (e.g. `<name>_inline_payload`).
3. Create `tests/unit/test_<name>.py` following the template used by the
   existing files: 6–9 tests covering health, stats, inline, batch, input
   validation, each declared domain exception, and train.
4. Append a new section to this document (section 3.x) and update the table
   in section 2.

The same fake-plugin machinery handles any future model — no conftest
refactoring required.
