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
from pydantic import BaseModel

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.application.dto.train_dto import TrainResponse
from app.application.use_cases.get_stats_use_case import GetStatsUseCase
from app.application.use_cases.predict_model_use_case import PredictModelUseCase
from app.application.use_cases.train_model_use_case import TrainModelUseCase
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import TrainingNotSupportedError
from app.domain.services.model_runtime_service import ModelRuntimeService
from app.infrastructure.http.router_factory import make_model_router
from app.plugins.ml25_wine_sulphites.predict_dto import (
    PredictBatchResponse as WineSO2BatchResp,
    PredictInlineResponse as WineSO2InlineResp,
)
from app.plugins.modelo10_lacteo.predict_dto import (
    PredictBatchResponse as LacteoBatchResp,
    PredictInlineResponse as LacteoInlineResp,
)
from app.plugins.ml8_cereals_img_anomaly_detector.predict_dto import (
    PredictBatchResponse as Ml8CerealsBatchResp,
    PredictInlineResponse as Ml8CerealsInlineResp,
)
from app.plugins.ml8_cereals_img_anomaly_detector.train_dto import (
    TrainResponse as Ml8CerealsTrainResp,
)
from app.plugins.ml5_meat_cow_behaviour.predict_dto import (
    PredictBatchResponse as Ml5CowBatchResp,
    PredictInlineResponse as Ml5CowInlineResp,
)
from app.registry import REGISTRY


# ── Fake plugin ────────────────────────────────────────────────────────────

class FakePlugin(ModelPluginPort):
    """Deterministic fake plugin.

    ``inline_factory`` and ``batch_factory`` receive the plugin instance plus
    the call kwargs and must return the model's typed ``PredictInlineResponse``
    / ``PredictBatchResponse`` (mirroring how real plugins build their DTOs).

    ``raise_on_inline`` / ``raise_on_batch`` can be set to force an error
    path for a single call (used by exception-mapping tests).
    """

    def __init__(
        self,
        *,
        model_id: str,
        inline_factory: Callable[..., BaseModel],
        batch_factory: Callable[..., BaseModel],
        train_factory: Callable[..., BaseModel] | None = None,
    ) -> None:
        """Initialize the fake plugin with the given model ID and factories."""
        self._model_id = model_id
        self._inline_factory = inline_factory
        self._batch_factory = batch_factory
        self._train_factory = train_factory
        self._loaded = False
        self._predict_count = 0
        self._last_predict_at: str | None = None
        self.raise_on_inline: Exception | None = None
        self.raise_on_batch: Exception | None = None

    def load(self) -> None:
        """Simulate loading model artifacts from disk."""
        self._loaded = True

    def is_loaded(self) -> bool:
        """Return True if the model is ready for inference."""
        return self._loaded

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
    ) -> BaseModel:
        """Run inline inference on a single feature dict and return the typed inline response."""
        if self.raise_on_inline is not None:
            exc, self.raise_on_inline = self.raise_on_inline, None
            raise exc
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        return self._inline_factory(
            self, features=features, model_key=model_key, threshold=threshold
        )

    def predict_batch(self, *, data_path: str) -> BaseModel:
        """Run batch inference on a CSV file and return the typed batch response."""
        if self.raise_on_batch is not None:
            exc, self.raise_on_batch = self.raise_on_batch, None
            raise exc
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        return self._batch_factory(self, data_path=data_path)

    def stats(self) -> StatsResponse:
        """Return model metadata and runtime statistics."""
        return StatsResponse(
            model_name=self._model_id,
            version="0.0.0",
            description="Fake plugin for testing",
            task_type="fake",
            framework="fake",
            inputs=[InputField(name="fake_input", type="float", description="Fake input field")],
            outputs=[OutputField(name="fake_output", type="float", description="Fake output field")],
            metrics={},
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=None,
            ),
        )

    def train(self, *, data_path: str) -> BaseModel:
        """Train the model with the provided data."""
        if self._train_factory is None:
            raise TrainingNotSupportedError(
                "Training is not supported by this runtime. Use the data science pipeline instead."
            )
        return self._train_factory(self, data_path=data_path)


# ── Fake response factories per model ──────────────────────────────────────

def _wine_so2_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> WineSO2InlineResp:
    """Return a fake inline prediction response for the wine sulphite intervention model."""
    return WineSO2InlineResp(
        model_id="wine-sulphite",
        threshold=threshold,
        prediction=True,
        confidence=6.2,
        features_used=[
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "density", "pH", "sulphates", "alcohol",
            "free sulfur dioxide", "total sulfur dioxide",
        ],
        recommended_free_so2=32.0,
        recommended_bound_so2=68.0,
        recommended_total_so2=100.0,
        recommended_molecular_so2=0.7,
        predicted_quality=6.2,
        baseline_predicted_quality=5.8,
        recommendation_reason="Intervention improves predicted quality by more than MAE threshold.",
        intervention_recommended=True,
        mae_quality=0.427,
        mae_bound=14.5,
    )


def _wine_so2_batch(plugin: FakePlugin, *, data_path: str) -> WineSO2BatchResp:
    """Return a fake batch prediction response for the wine sulphite intervention model."""
    return WineSO2BatchResp(
        model_id="wine-sulphite",
        predictions=[
            {
                "row": 0,
                "intervention_recommended": True,
                "recommended_free_so2": 32.0,
                "predicted_quality": 6.2,
                "recommendation_reason": "ok",
            }
        ],
        output_path=None,
    )


def _lacteo_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> LacteoInlineResp:
    """Return a fake inline prediction response for the Modelo10Lacteo plugin."""
    return LacteoInlineResp(
        model_id="modelo10-lacteo",
        prediction="fly",
        confidence=0.91,
        vectors_count=1,
        detections=[
            {"species": "fly", "det_conf": 0.85, "cls_conf": 0.91, "bbox": {"x1": 30, "y1": 40, "x2": 80, "y2": 90}},
        ],
        species_summary={"fly": 1},
    )


def _lacteo_batch(plugin: FakePlugin, *, data_path: str) -> LacteoBatchResp:
    """Return a fake batch prediction response for the Modelo10Lacteo plugin."""
    return LacteoBatchResp(
        model_id="modelo10-lacteo",
        predictions=[
            {
                "filename": "test_cow.jpg",
                "prediction": "tick",
                "confidence": 0.88,
                "vectors_count": 2,
                "detections": [
                    {"species": "tick", "det_conf": 0.82, "cls_conf": 0.88, "bbox": {"x1": 10, "y1": 20, "x2": 50, "y2": 60}},
                    {"species": "tick", "det_conf": 0.75, "cls_conf": 0.81, "bbox": {"x1": 100, "y1": 120, "x2": 150, "y2": 160}},
                ],
                "species_summary": {"tick": 2},
            }
        ],
        output_path=None,
    )


def _lacteo_train(plugin: FakePlugin, *, data_path: str) -> TrainResponse:
    """Return a fake training response for the Modelo10Lacteo plugin."""
    return TrainResponse(
        detail="Training completed successfully",
        metrics={
            "train_samples": 100,
            "val_samples": 20,
            "classes": ["fly", "mos", "tick"],
            "epochs_run": 5,
            "best_val_acc": 95.0,
            "time_min": 2.5,
        },
    )


def _ml8_cereals_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml8CerealsInlineResp:
    """Return a fake inline prediction response for the ml8 cereals model."""
    return Ml8CerealsInlineResp(
        model_id="ml8-cereals-img-anomaly-detector",
        categoria="sano",
        cereal="trigo",
        confianza_categoria=0.95,
        confianza_cereal=0.91,
        probabilidades_categoria={"sano": 0.95, "hongos": 0.02, "insectos": 0.02, "otros": 0.01},
        probabilidades_cereal={"trigo": 0.91, "maiz": 0.04, "arroz": 0.03, "sorgo": 0.02},
    )


def _ml8_cereals_batch(plugin: FakePlugin, *, data_path: str) -> Ml8CerealsBatchResp:
    """Return a fake batch prediction response for the ml8 cereals model."""
    return Ml8CerealsBatchResp(
        model_id="ml8-cereals-img-anomaly-detector",
        predictions=[
            {
                "filename": "img_001.jpg",
                "model_id": "ml8-cereals-img-anomaly-detector",
                "categoria": "sano",
                "cereal": "trigo",
                "confianza_categoria": 0.95,
                "confianza_cereal": 0.91,
                "probabilidades_categoria": {"sano": 0.95, "hongos": 0.02, "insectos": 0.02, "otros": 0.01},
                "probabilidades_cereal": {"trigo": 0.91, "maiz": 0.04, "arroz": 0.03, "sorgo": 0.02},
            }
        ],
        output_path=None,
    )


def _ml8_cereals_train(plugin: FakePlugin, *, data_path: str) -> Ml8CerealsTrainResp:
    """Return a fake training response for the ml8 cereals model."""
    return Ml8CerealsTrainResp(
        detail="Entrenamiento completado",
        train_samples=80,
        val_samples=20,
        fase1_epochs=3,
        fase2_epochs=2,
        fase1_time_min=0.5,
        fase2_time_min=0.2,
        best_val_acc_cat=91.2,
        best_val_acc_cer=88.5,
        upload_warning=None,
    )


def _ml5_cow_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml5CowInlineResp:
    """Return a fake inline prediction response for the ml5 cow-behaviour model."""
    return Ml5CowInlineResp(
        model_id="ml5-meat-cow-behaviour",
        threshold=threshold if threshold is not None else 0.5,
        prediction="grazing",
        confidence=0.87,
        features_used=["frames_base64"],
        is_anomaly=False,
        behavior_idx=0,
        xai_feature_values={"grazing": 0.87, "walking": 0.08, "drinking": 0.05},
    )


def _ml5_cow_batch(plugin: FakePlugin, *, data_path: str) -> Ml5CowBatchResp:
    """Return a fake batch prediction response for the ml5 cow-behaviour model."""
    return Ml5CowBatchResp(
        model_id="ml5-meat-cow-behaviour",
        predictions=[
            {
                "frame": 0,
                "detections": [
                    {
                        "track_id": 0,
                        "bbox": [10.0, 20.0, 110.0, 220.0],
                        "score": 0.95,
                        "behavior": "grazing",
                        "behavior_confidence": 0.87,
                        "is_anomaly": False,
                    }
                ],
            }
        ],
        output_path=None,
    )


FAKE_FACTORIES: dict[str, tuple[Callable, Callable]] = {
    "wine-sulphite": (_wine_so2_inline, _wine_so2_batch),
    "modelo10-lacteo": (_lacteo_inline, _lacteo_batch),
    "ml8-cereals-img-anomaly-detector": (_ml8_cereals_inline, _ml8_cereals_batch),
    "ml5-meat-cow-behaviour": (_ml5_cow_inline, _ml5_cow_batch),
}

TRAIN_FACTORIES: dict[str, Callable] = {
    "modelo10-lacteo": _lacteo_train,
    "ml8-cereals-img-anomaly-detector": _ml8_cereals_train,
}


# ── App / client fixtures ──────────────────────────────────────────────────

class _FakeContainer:
    """Lightweight stand-in for ModelContainer used only in tests."""

    def __init__(self, plugin: FakePlugin) -> None:
        """Initialize the fake container with a plugin."""
        self._plugin = plugin
        self.service = ModelRuntimeService(plugin)


def _build_container(plugin: FakePlugin) -> Any:
    """Build a fake container with the appropriate use cases for the given plugin."""
    container = _FakeContainer(plugin)
    container.predict_use_case = PredictModelUseCase(plugin)
    container.stats_use_case = GetStatsUseCase(plugin)
    container.train_use_case = TrainModelUseCase(plugin)
    return container


@pytest.fixture
def fake_plugins() -> dict[str, FakePlugin]:
    """One fake plugin per REGISTRY entry, already ``load()``-ed."""
    plugins: dict[str, FakePlugin] = {}
    for entry in REGISTRY:
        inline_factory, batch_factory = FAKE_FACTORIES[entry.model_id]
        train_factory = TRAIN_FACTORIES.get(entry.model_id)
        plugin = FakePlugin(
            model_id=entry.model_id,
            inline_factory=inline_factory,
            batch_factory=batch_factory,
            train_factory=train_factory,
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
            fake_plugins[entry.model_id]
        )
        application.include_router(
            make_model_router(
                model_id=entry.model_id,
                version=entry.version,
                predict_request_type=entry.predict_request_type,
                predict_response_type=entry.predict_response_type,
                extra_predict_exceptions=entry.extra_predict_exceptions,
                train_request_type=entry.train_request_type,
                train_response_type=entry.train_response_type,
            ),
            prefix=entry.prefix,
            tags=[entry.model_id],
        )
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """A TestClient for the app fixture."""
    return TestClient(app)


# ── Sample payloads ────────────────────────────────────────────────────────

@pytest.fixture
def wine_so2_inline_payload() -> dict:
    """Return a sample inline payload for the wine sulphite intervention model."""
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


@pytest.fixture
def lacteo_inline_payload() -> dict:
    """Return a sample inline payload for the Modelo10Lacteo plugin."""
    return {"mode": "inline", "image_base64": "dGVzdC1pbWFnZQ=="}
