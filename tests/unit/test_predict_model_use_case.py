"""Regression tests for PredictModelUseCase's JSON-safety normalization.

Plugins occasionally leave numpy scalars in loosely-typed (``dict[str, Any]``) response
fields — Pydantic only coerces concretely-typed fields, so a stray numpy.int64/float64
used to reach ``execute()``'s caller unchanged and only blow up later, at FastAPI's
JSON-encoding step, with ``PydanticSerializationError: Unable to serialize unknown type``.
``PredictModelUseCase.execute()`` now normalizes every plugin's response before returning
it, so this is tested once here instead of per-plugin.
"""
from __future__ import annotations

import json

import numpy as np

from app.application.use_cases.predict_model_use_case import PredictModelUseCase
from app.plugins.ml45_cereals_dnsl_critical_point_detection.predict_dto import (
    PredictBatchRequest as Ml45BatchRequest,
    PredictBatchResponse as Ml45BatchResponse,
)
from app.plugins.ml28_meat_neuroevolutionary_raw_materials_prediction.predict_dto import (
    PredictBatchRequest as Ml28BatchRequest,
    PredictBatchResponse as Ml28BatchResponse,
)
from tests.conftest import FakePlugin


def _ml45_batch_with_numpy_leak(_plugin, *, data_path):
    _ = data_path
    return Ml45BatchResponse(
        model_id="ml45-cereals-dnsl-critical-point-detection",
        predictions=[
            {
                "window_index": 1,
                # Mirrors the real bug: a pandas groupby key (numpy.int64) landing in a
                # dict[str, Any] response field unconverted.
                "cycle_id": np.int64(42),
                "anomaly_probability": np.float64(0.87),
                "predicted_anomaly_label": "Fallo",
            }
        ],
        output_path=None,
    )


def _ml28_batch_with_numpy_leak(_plugin, *, data_path):
    _ = data_path
    return Ml28BatchResponse(
        model_id="ml28-meat-neuroevolutionary-raw-materials-prediction",
        predictions=[
            {
                "raw_material_id": "RM-1",
                # pandas.DataFrame.to_dict(orient="records") preserves numpy dtypes verbatim.
                "purchase_trigger_flag": np.int64(1),
                "order_quantity_tons": np.float64(12.5),
            }
        ],
        summary={"row_count": np.int64(1)},
        output_path=None,
    )


def test_execute_batch_normalizes_numpy_types_for_ml45():
    plugin = FakePlugin(
        model_id="ml45-cereals-dnsl-critical-point-detection",
        inline_factory=lambda *a, **k: None,
        batch_factory=_ml45_batch_with_numpy_leak,
    )
    result = PredictModelUseCase(plugin).execute(Ml45BatchRequest(data_path="s3://bucket/file.csv"))

    row = result.predictions[0]
    assert type(row["cycle_id"]) is int
    assert type(row["anomaly_probability"]) is float
    # The real bug only ever surfaced here — at JSON-encoding time.
    json.dumps(result.model_dump(mode="json"))


def test_execute_batch_normalizes_numpy_types_for_ml28():
    plugin = FakePlugin(
        model_id="ml28-meat-neuroevolutionary-raw-materials-prediction",
        inline_factory=lambda *a, **k: None,
        batch_factory=_ml28_batch_with_numpy_leak,
    )
    result = PredictModelUseCase(plugin).execute(Ml28BatchRequest(data_path="s3://bucket/file.csv"))

    row = result.predictions[0]
    assert type(row["purchase_trigger_flag"]) is int
    assert type(row["order_quantity_tons"]) is float
    assert type(result.summary["row_count"]) is int
    json.dumps(result.model_dump(mode="json"))
