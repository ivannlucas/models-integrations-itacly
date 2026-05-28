"""Shared fixtures for endpoint unit tests.

Strategy
--------
We don't want to load real ML artifacts during unit tests, so every test uses
a fake ``ModelPluginPort`` implementation that returns canonical dicts
matching each model's Pydantic response schema.

Fake plugins are injected into ``app.state.containers`` directly, and routers
are mounted using the real ``make_model_router`` factory and the real
``REGISTRY`` entries. This exercises the full HTTP surface (routing, schema
validation, exception mapping) without touching disk or any ML framework.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.application.dto.stats_dto import StatsResponse
from app.application.use_cases.get_stats_use_case import GetStatsUseCase
from app.application.use_cases.predict_model_use_case import PredictModelUseCase
from app.application.use_cases.train_model_use_case import TrainModelUseCase
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.model_runtime_service import ModelRuntimeService
from app.infrastructure.http.router_factory import make_model_router
from app.registry import REGISTRY


# ── Fake plugin ────────────────────────────────────────────────────────────

class FakePlugin(ModelPluginPort):
    """Deterministic fake plugin.

    ``inline_factory`` and ``batch_factory`` receive the plugin instance plus
    the call kwargs and must return a dict compatible with the model's
    ``PredictInlineResponse`` / ``PredictBatchResponse``.

    ``raise_on_inline`` / ``raise_on_batch`` can be set to force an error
    path for a single call (used by exception-mapping tests).
    """

    def __init__(
        self,
        *,
        model_id: str,
        inline_factory: Callable[..., dict],
        batch_factory: Callable[..., dict],
    ) -> None:
        self._model_id = model_id
        self._inline_factory = inline_factory
        self._batch_factory = batch_factory
        self._loaded = False
        self._predict_count = 0
        self._last_predict_at: str | None = None
        self.raise_on_inline: Exception | None = None
        self.raise_on_batch: Exception | None = None

    def load(self) -> None:
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
    ) -> dict:
        if self.raise_on_inline is not None:
            exc, self.raise_on_inline = self.raise_on_inline, None
            raise exc
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        return self._inline_factory(
            self, features=features, model_key=model_key, threshold=threshold
        )

    def predict_batch(self, *, data_path: str) -> dict:
        if self.raise_on_batch is not None:
            exc, self.raise_on_batch = self.raise_on_batch, None
            raise exc
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        return self._batch_factory(self, data_path=data_path)

    def stats(self) -> StatsResponse:
        return StatsResponse(
            model_name=self._model_id,
            model_type="fake",
            framework="fake",
            artifact_path=f"/fake/{self._model_id}",
            input_schema={"fake": "input"},
            output_schema={"fake": "output"},
            predict_count=self._predict_count,
            last_predict_at=self._last_predict_at,
        )


# ── Fake response factories per model ──────────────────────────────────────

def _wine_pf_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> dict:
    return {
        "model_id": "wine-price-fluctuation",
        "threshold": threshold,
        "prediction": 1,
        "confidence": 0.72,
        "features_used": ["rsi", "bollinger", "momentum"],
        "model_type": "xgboost",
        "prediction_date": "2024-01-15",
        "xai_feature_values": {"rsi": 55.3, "bollinger": 1.1},
    }


def _wine_pf_batch(plugin: FakePlugin, *, data_path: str) -> dict:
    return {
        "model_id": "wine-price-fluctuation",
        "predictions": [
            {"row": 0, "prediction": 1, "confidence": 0.72},
            {"row": 1, "prediction": 0, "confidence": 0.31},
        ],
        "output_path": None,
    }


def _cereal_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> dict:
    return {
        "model_id": "cereal-price-forecast",
        "threshold": threshold,
        "prediction": 245.7,
        "confidence": None,
        "features_used": ["Year", "Month", "prec", "tmed"],
        "product_name": features.get("product_name", "Milling wheat"),
        "market_name": features.get("market_name", "Unknown"),
        "week_begin_date": features.get("week_begin_date", "Unknown"),
        "model_version": "1.0",
        "xai_feature_values": {"Year": 2024.0, "Month": 1.0},
    }


def _cereal_batch(plugin: FakePlugin, *, data_path: str) -> dict:
    return {
        "model_id": "cereal-price-forecast",
        "predictions": [
            {"row": 0, "product": "Milling wheat", "prediction": 245.7},
            {"row": 1, "product": "Feed barley", "prediction": 208.3},
        ],
        "output_path": None,
    }


def _meat_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> dict:
    return {
        "model_id": "meat-price-forecast",
        "threshold": threshold,
        "prediction": {
            "bovino": {"rf": 130.4, "lstm": None},
            "porcino": {"rf": 128.1, "lstm": None},
            "ovino": {"rf": 135.2, "lstm": None},
            "ave": {"rf": 120.8, "lstm": None},
            "carne": {"rf": 129.7, "lstm": None},
        },
        "confidence": None,
        "features_used": ["bovino_lag_1", "porcino_lag_1"],
        "prediction_date": "2024-01-15",
        "rows_used": 4,
        "xai_feature_values": None,
    }


def _meat_batch(plugin: FakePlugin, *, data_path: str) -> dict:
    return {
        "model_id": "meat-price-forecast",
        "predictions": [
            {"date": "2024-01-15", "target": "bovino", "rf": 130.4},
            {"date": "2024-01-15", "target": "porcino", "rf": 128.1},
        ],
        "output_path": None,
    }


def _cnn_fungal_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> dict:
    return {
        "model_id": "cnn-fungal-detection",
        "threshold": threshold,
        "prediction": "healthy",
        "confidence": 0.93,
        "features_used": ["pixel_tensor"],
        "probabilities": {"healthy": 0.93, "fungal": 0.07},
    }


def _cnn_fungal_batch(plugin: FakePlugin, *, data_path: str) -> dict:
    return {
        "model_id": "cnn-fungal-detection",
        "predictions": [
            {"file": "img_0001.jpg", "prediction": "healthy", "confidence": 0.93},
            {"file": "img_0002.jpg", "prediction": "fungal", "confidence": 0.81},
        ],
        "output_path": None,
    }


def _cnn_thermal_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> dict:
    return {
        "model_id": "cnn-thermal-scm",
        "threshold": threshold,
        "prediction": "Healthy",
        "confidence": 0.88,
        "features_used": ["thermal_image"],
        "predicted_class_index": 0,
        "probability_healthy": 0.88,
        "probability_scm": 0.12,
    }


def _cnn_thermal_batch(plugin: FakePlugin, *, data_path: str) -> dict:
    return {
        "model_id": "cnn-thermal-scm",
        "predictions": [
            {"file": "thermal_001.jpg", "prediction": "Healthy", "probability_scm": 0.12},
            {"file": "thermal_002.jpg", "prediction": "SCM", "probability_scm": 0.77},
        ],
        "output_path": None,
    }


def _cow_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> dict:
    return {
        "model_id": "cow-behavior",
        "threshold": threshold,
        "prediction": "walking",
        "confidence": 0.81,
        "features_used": ["frame_embedding"],
        "is_anomaly": False,
        "behavior_idx": 2,
    }


def _cow_batch(plugin: FakePlugin, *, data_path: str) -> dict:
    return {
        "model_id": "cow-behavior",
        "predictions": [
            {"clip_idx": 0, "behavior": "walking", "is_anomaly": False},
            {"clip_idx": 1, "behavior": "lying", "is_anomaly": False},
        ],
        "output_path": None,
    }


def _wine_so2_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> dict:
    return {
        "model_id": "wine-sulphite",
        "threshold": threshold,
        "prediction": True,
        "confidence": 6.2,
        "features_used": [
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "density", "pH", "sulphates", "alcohol",
            "free sulfur dioxide", "total sulfur dioxide",
        ],
        "recommended_free_so2": 32.0,
        "recommended_bound_so2": 68.0,
        "recommended_total_so2": 100.0,
        "recommended_molecular_so2": 0.7,
        "predicted_quality": 6.2,
        "baseline_predicted_quality": 5.8,
        "recommendation_reason": "Intervention improves predicted quality by more than MAE threshold.",
        "intervention_recommended": True,
        "mae_quality": 0.427,
        "mae_bound": 14.5,
    }


def _wine_so2_batch(plugin: FakePlugin, *, data_path: str) -> dict:
    return {
        "model_id": "wine-sulphite",
        "predictions": [
            {
                "row": 0,
                "intervention_recommended": True,
                "recommended_free_so2": 32.0,
                "predicted_quality": 6.2,
                "recommendation_reason": "ok",
            }
        ],
        "output_path": None,
    }


FAKE_FACTORIES: dict[str, tuple[Callable, Callable]] = {
    "wine-price-fluctuation": (_wine_pf_inline, _wine_pf_batch),
    "cereal-price-forecast": (_cereal_inline, _cereal_batch),
    "meat-price-forecast": (_meat_inline, _meat_batch),
    "cnn-fungal-detection": (_cnn_fungal_inline, _cnn_fungal_batch),
    "cnn-thermal-scm": (_cnn_thermal_inline, _cnn_thermal_batch),
    "cow-behavior": (_cow_inline, _cow_batch),
    "wine-sulphite": (_wine_so2_inline, _wine_so2_batch),
}


# ── App / client fixtures ──────────────────────────────────────────────────

class _FakeContainer:
    """Lightweight stand-in for ModelContainer used only in tests."""

    def __init__(self, plugin: FakePlugin) -> None:
        self._plugin = plugin
        self.service = ModelRuntimeService(plugin)


def _build_container(plugin: FakePlugin, entry) -> Any:
    container = _FakeContainer(plugin)
    container.predict_use_case = PredictModelUseCase(
        plugin, entry.batch_response_class, entry.inline_response_class
    )
    container.stats_use_case = GetStatsUseCase(plugin)
    container.train_use_case = TrainModelUseCase()
    return container


@pytest.fixture
def fake_plugins() -> dict[str, FakePlugin]:
    """One fake plugin per REGISTRY entry, already ``load()``-ed."""
    plugins: dict[str, FakePlugin] = {}
    for entry in REGISTRY:
        inline_factory, batch_factory = FAKE_FACTORIES[entry.model_id]
        plugin = FakePlugin(
            model_id=entry.model_id,
            inline_factory=inline_factory,
            batch_factory=batch_factory,
        )
        plugin.load()
        plugins[entry.model_id] = plugin
    return plugins


@pytest.fixture
def app(fake_plugins: dict[str, FakePlugin]) -> FastAPI:
    """A FastAPI app wired to fake containers — no real ML artifacts touched."""
    application = FastAPI(title="Luce ML Models API (test)")
    application.state.containers = {}
    for entry in REGISTRY:
        application.state.containers[entry.model_id] = _build_container(
            fake_plugins[entry.model_id], entry
        )
        application.include_router(
            make_model_router(
                model_id=entry.model_id,
                version=entry.version,
                predict_request_type=entry.predict_request_type,
                predict_response_type=entry.predict_response_type,
                extra_predict_exceptions=entry.extra_predict_exceptions,
            ),
            prefix=entry.prefix,
            tags=[entry.model_id],
        )
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ── Sample payloads ────────────────────────────────────────────────────────

@pytest.fixture
def wine_pf_inline_payload() -> dict:
    records = [
        {"campaign": "2023/2024", "week": w, "price_red": 40.0 + w * 0.1}
        for w in range(1, 25)
    ]
    return {"mode": "inline", "records": records}


@pytest.fixture
def cereal_inline_payload() -> dict:
    return {
        "mode": "inline",
        "product_name": "Milling wheat",
        "market_name": "Valladolid",
        "week_begin_date": "2024-01-15",
        "Year": 2024.0,
        "Month": 1.0,
        "Quarter": 1.0,
        "Week_of_Year": 3.0,
    }


@pytest.fixture
def meat_inline_payload() -> dict:
    rows = [
        {
            "date": f"2024-01-{day:02d}",
            "bovino": 130.0 + i,
            "porcino": 128.0 + i,
            "ovino": 135.0 + i,
            "ave": 120.0 + i,
            "carne": 129.0 + i,
        }
        for i, day in enumerate([1, 8, 15, 22])
    ]
    return {"mode": "inline", "rows": rows, "include_lstm": False}


@pytest.fixture
def cnn_fungal_inline_payload() -> dict:
    return {"mode": "inline", "image_path": "/tmp/fake_image.jpg"}


@pytest.fixture
def cnn_thermal_inline_payload() -> dict:
    return {"mode": "inline", "image_path": "/tmp/fake_thermal.jpg"}


@pytest.fixture
def cow_inline_payload() -> dict:
    # 32 frames is the minimum clip length
    frames = ["AAAA"] * 32
    return {"mode": "inline", "frames_base64": frames, "detection_threshold": 0.5}


@pytest.fixture
def wine_so2_inline_payload() -> dict:
    return {
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
        "delta_max": 40.0,
    }
