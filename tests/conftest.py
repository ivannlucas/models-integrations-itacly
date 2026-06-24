"""Shared fixtures for endpoint unit tests.

Strategy
--------
We don't want to load real ML artifacts during unit tests, so every test uses
a fake ``ModelPluginPort`` implementation that returns canonical dicts
matching each model's Pydantic response schema.

Fake plugins are injected into ``app.state.containers`` directly, and routers
are mounted using the real ``make_model_router`` factory and a local
``TEST_REGISTRY`` (avoids importing heavy plugin modules like cv2/torch).
This exercises the full HTTP surface (routing, schema validation, exception
mapping) without touching disk or any ML framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.application.dto.train_dto import TrainResponse as BaseTrainResponse
from app.application.use_cases.get_stats_use_case import GetStatsUseCase
from app.application.use_cases.predict_model_use_case import PredictModelUseCase
from app.application.use_cases.train_model_use_case import TrainModelUseCase
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import (
    InsufficientFramesError,
    InvalidImageError,
    InvalidVideoError,
    NoValidSimulationPointError,
    TrainingNotSupportedError,
)
from app.domain.services.model_runtime_service import ModelRuntimeService
from app.infrastructure.http.router_factory import make_model_router
from app.plugins.ml25_wine_sulphites.predict_dto import (
    PredictBatchResponse as WineSO2BatchResp,
    PredictInlineResponse as WineSO2InlineResp,
    PredictRequest as WineSO2_Request,
    PredictResponse as WineSO2_Response,
)
from app.plugins.ml25_wine_sulphites.train_dto import (
    TrainRequest as WineSO2_TrainReq,
    TrainResponse as WineSO2_TrainResp,
)
from app.plugins.modelo10_lacteo.predict_dto import (
    PredictBatchResponse as LacteoBatchResp,
    PredictInlineResponse as LacteoInlineResp,
    PredictRequest as Lacteo10_Request,
    PredictResponse as Lacteo10_Response,
)
from app.plugins.ml8_cereals_img_anomaly_detector.predict_dto import (
    PredictBatchResponse as Ml8CerealsBatchResp,
    PredictInlineResponse as Ml8CerealsInlineResp,
    PredictRequest as Ml8Cereals_Request,
    PredictResponse as Ml8Cereals_Response,
)
from app.plugins.ml8_cereals_img_anomaly_detector.train_dto import (
    TrainRequest as Ml8Cereals_TrainReq,
    TrainResponse as Ml8CerealsTrainResp,
)
from app.plugins.ml2_fungal_cnn_disease_detection.predict_dto import (
    PredictBatchResponse as Ml2FungalBatchResp,
    PredictInlineResponse as Ml2FungalInlineResp,
    PredictRequest as Ml2Fungal_Request,
    PredictResponse as Ml2Fungal_Response,
)
from app.plugins.ml5_meat_cow_behaviour.predict_dto import (
    PredictBatchResponse as Ml5CowBatchResp,
    PredictInlineResponse as Ml5CowInlineResp,
    PredictRequest as Ml5Cow_Request,
    PredictResponse as Ml5Cow_Response,
)
from app.plugins.ml7_cereals_grain_pest_detection.predict_dto import (
    PredictBatchResponse as Ml7GrainBatchResp,
    PredictInlineResponse as Ml7GrainInlineResp,
    PredictRequest as Ml7Grain_Request,
    PredictResponse as Ml7Grain_Response,
)
from app.plugins.ml30_meat_traceability_detection.predict_dto import (
    PredictBatchResponse as Ml30TraceBatchResp,
    PredictInlineResponse as Ml30TraceInlineResp,
    PredictRequest as Ml30Trace_Request,
    PredictResponse as Ml30Trace_Response,
)
from app.plugins.ml30_meat_traceability_detection.train_dto import (
    TrainRequest as Ml30Trace_TrainReq,
    TrainResponse as Ml30TraceTrainResp,
)
from app.plugins.ml31_cereals_residue_optimizer.predict_dto import (
    PredictBatchResponse as Ml31ResidueBatchResp,
    PredictInlineResponse as Ml31ResidueInlineResp,
    PredictRequest as Ml31Residue_Request,
    PredictResponse as Ml31Residue_Response,
)
from app.plugins.ml31_cereals_residue_optimizer.train_dto import (
    TrainRequest as Ml31Residue_TrainReq,
    TrainResponse as Ml31ResidueTrainResp,
)

# ── ModelEntry dataclass (local copy — avoids importing app.registry which loads real plugins) ───


@dataclass
class ModelEntry:
    """Defines the metadata and types for a model plugin."""
    model_id: str
    prefix: str
    version: str
    plugin_class: type
    predict_request_type: Any
    predict_response_type: Any
    train_request_type: Any | None = None
    train_response_type: Any | None = None
    extra_predict_exceptions: tuple[type[Exception], ...] = field(default_factory=tuple)


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
        mlflow_run_id: str = "",
    ) -> BaseModel:
        """Run inline inference on a single feature dict and return the typed inline response."""
        _ = mlflow_run_id
        if self.raise_on_inline is not None:
            exc, self.raise_on_inline = self.raise_on_inline, None
            raise exc
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        return self._inline_factory(
            self, features=features, model_key=model_key, threshold=threshold
        )

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> BaseModel:
        """Run batch inference on a CSV file and return the typed batch response."""
        _ = mlflow_run_id
        if self.raise_on_batch is not None:
            exc, self.raise_on_batch = self.raise_on_batch, None
            raise exc
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        return self._batch_factory(self, data_path=data_path)

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
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

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> BaseModel:
        """Train the model with the provided data."""
        _ = mlflow_run_id
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


def _lacteo_train(plugin: FakePlugin, *, data_path: str) -> BaseTrainResponse:
    """Return a fake training response for the Modelo10Lacteo plugin."""
    return BaseTrainResponse(
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


def _ml2_fungal_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml2FungalInlineResp:
    """Fake inline response for the ml2 fungal CNN model."""
    return Ml2FungalInlineResp(
        model_id="ml2-fungal-cnn-disease-detection",
        prediction="healthy",
        confidence=0.93,
        probabilities={"black_rot": 0.02, "downy_mildew": 0.02, "healthy": 0.93,
                       "powdery_mildew": 0.02, "trunk_disease": 0.01},
    )


def _ml2_fungal_batch(plugin: FakePlugin, *, data_path: str) -> Ml2FungalBatchResp:
    """Fake batch response for the ml2 fungal CNN model."""
    return Ml2FungalBatchResp(
        model_id="ml2-fungal-cnn-disease-detection",
        predictions=[{"filename": "leaf_001.jpg", "model_id": "ml2-fungal-cnn-disease-detection",
                      "prediction": "powdery_mildew", "confidence": 0.88,
                      "probabilities": {"black_rot": 0.03, "downy_mildew": 0.04, "healthy": 0.03,
                                        "powdery_mildew": 0.88, "trunk_disease": 0.02}}],
        output_path=None,
    )


def _ml7_grain_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml7GrainInlineResp:
    """Fake inline response for the ml7 grain pest detector."""
    return Ml7GrainInlineResp(
        model_id="ml7-cereals-grain-pest-detection",
        prediction="sz",
        confidence=0.81,
        total_detections=2,
        species_counts={"sz": 2},
        detections=[{"class": "sz", "class_name": "Sitophilus spp.", "confidence": 0.81,
                     "bbox": [10.0, 20.0, 50.0, 60.0]}],
        annotated_image="ZmFrZQ==",
        threshold=threshold,
        features_used=["image_base64"],
    )


def _ml7_grain_batch(plugin: FakePlugin, *, data_path: str) -> Ml7GrainBatchResp:
    """Fake batch response for the ml7 grain pest detector."""
    return Ml7GrainBatchResp(
        model_id="ml7-cereals-grain-pest-detection",
        predictions=[{"filename": "img_001.jpg", "prediction": "sz", "confidence": 0.81,
                      "total_detections": 2, "species_counts": {"sz": 2}}],
        output_path=None,
    )


def _ml30_trace_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml30TraceInlineResp:
    """Fake inline response for the ml30 traceability model."""
    return Ml30TraceInlineResp(
        model_id="ml30-meat-traceability-detection",
        pred_traceability_incident=1,
        pred_score=0.82,
        confidence=0.82,
        model_name="ml30-meat-traceability-detection",
        xai_feature_values={"sensor_temp_c": 7.5},
    )


def _ml30_trace_batch(plugin: FakePlugin, *, data_path: str) -> Ml30TraceBatchResp:
    """Fake batch response for the ml30 traceability model."""
    return Ml30TraceBatchResp(
        model_id="ml30-meat-traceability-detection",
        predictions=[{"row_id": 0, "pred_traceability_incident": 1, "pred_score": 0.82,
                      "model_name": "ml30-meat-traceability-detection"}],
        output_path=None,
    )


def _ml30_trace_train(plugin: FakePlugin, *, data_path: str) -> Ml30TraceTrainResp:
    """Fake training response for the ml30 traceability model."""
    return Ml30TraceTrainResp(
        detail="Training completed", accuracy=0.87, f1=0.6, roc_auc=0.72,
        n_train=800, n_test=200, training_time_s=12.3, upload_warning=None,
    )


def _ml31_residue_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml31ResidueInlineResp:
    """Fake inline response for the ml31 residue optimizer."""
    return Ml31ResidueInlineResp(
        model_id="ml31-cereals-residue-optimizer",
        prediction=1234.5,
        confidence=None,
        xai_feature_values={"Sup_Secano_ha": 100.0, "Lluvia_Primavera_mm": 180.0},
    )


def _ml31_residue_batch(plugin: FakePlugin, *, data_path: str) -> Ml31ResidueBatchResp:
    """Fake batch response for the ml31 residue optimizer."""
    return Ml31ResidueBatchResp(
        model_id="ml31-cereals-residue-optimizer",
        predictions=[{"row": 0, "prediction": 1234.5, "Cultivo": "Trigo"}],
        output_path=None,
    )


def _ml31_residue_train(plugin: FakePlugin, *, data_path: str) -> Ml31ResidueTrainResp:
    """Fake training response for the ml31 residue optimizer."""
    return Ml31ResidueTrainResp(
        detail="Entrenamiento completado", r2_test=0.83, n_train=800, n_test=200,
    )


FAKE_FACTORIES: dict[str, tuple[Callable, Callable]] = {
    "ml31-cereals-residue-optimizer": (_ml31_residue_inline, _ml31_residue_batch),
    "ml30-meat-traceability-detection": (_ml30_trace_inline, _ml30_trace_batch),
    "ml7-cereals-grain-pest-detection": (_ml7_grain_inline, _ml7_grain_batch),
    "ml2-fungal-cnn-disease-detection": (_ml2_fungal_inline, _ml2_fungal_batch),
    "wine-sulphite": (_wine_so2_inline, _wine_so2_batch),
    "modelo10-lacteo": (_lacteo_inline, _lacteo_batch),
    "ml8-cereals-img-anomaly-detector": (_ml8_cereals_inline, _ml8_cereals_batch),
    "ml5-meat-cow-behaviour": (_ml5_cow_inline, _ml5_cow_batch),
}

TRAIN_FACTORIES: dict[str, Callable] = {
    "modelo10-lacteo": _lacteo_train,
    "ml8-cereals-img-anomaly-detector": _ml8_cereals_train,
    "ml30-meat-traceability-detection": _ml30_trace_train,
    "ml31-cereals-residue-optimizer": _ml31_residue_train,
}


# ── Test registry (avoids importing heavy plugin modules like cv2/torch) ───

TEST_REGISTRY: list[ModelEntry] = [
    ModelEntry(
        model_id="wine-sulphite",
        prefix="/models/wine-sulphite",
        version="1.2.0",
        plugin_class=FakePlugin,
        predict_request_type=WineSO2_Request,
        predict_response_type=WineSO2_Response,
        extra_predict_exceptions=(NoValidSimulationPointError,),
        train_request_type=WineSO2_TrainReq,
        train_response_type=WineSO2_TrainResp,
    ),
    ModelEntry(
        model_id="modelo10-lacteo",
        prefix="/models/modelo10-lacteo",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Lacteo10_Request,
        predict_response_type=Lacteo10_Response,
        extra_predict_exceptions=(),
    ),
    ModelEntry(
        model_id="ml8-cereals-img-anomaly-detector",
        prefix="/models/ml8-cereals-img-anomaly-detector",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml8Cereals_Request,
        predict_response_type=Ml8Cereals_Response,
        extra_predict_exceptions=(InvalidImageError,),
        train_request_type=Ml8Cereals_TrainReq,
        train_response_type=Ml8CerealsTrainResp,
    ),
    ModelEntry(
        model_id="ml5-meat-cow-behaviour",
        prefix="/models/ml5-meat-cow-behaviour",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml5Cow_Request,
        predict_response_type=Ml5Cow_Response,
        extra_predict_exceptions=(InvalidVideoError, InvalidImageError, InsufficientFramesError),
    ),
    ModelEntry(
        model_id="ml2-fungal-cnn-disease-detection",
        prefix="/models/ml2-fungal-cnn-disease-detection",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml2Fungal_Request,
        predict_response_type=Ml2Fungal_Response,
        extra_predict_exceptions=(InvalidImageError,),
    ),
    ModelEntry(
        model_id="ml7-cereals-grain-pest-detection",
        prefix="/models/ml7-cereals-grain-pest-detection",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml7Grain_Request,
        predict_response_type=Ml7Grain_Response,
        extra_predict_exceptions=(InvalidImageError,),
    ),
    ModelEntry(
        model_id="ml30-meat-traceability-detection",
        prefix="/models/ml30-meat-traceability-detection",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml30Trace_Request,
        predict_response_type=Ml30Trace_Response,
        extra_predict_exceptions=(),
        train_request_type=Ml30Trace_TrainReq,
        train_response_type=Ml30TraceTrainResp,
    ),
    ModelEntry(
        model_id="ml31-cereals-residue-optimizer",
        prefix="/models/ml31-cereals-residue-optimizer",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml31Residue_Request,
        predict_response_type=Ml31Residue_Response,
        extra_predict_exceptions=(),
        train_request_type=Ml31Residue_TrainReq,
        train_response_type=Ml31ResidueTrainResp,
    ),
]


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
    """One fake plugin per TEST_REGISTRY entry, already ``load()``-ed."""
    plugins: dict[str, FakePlugin] = {}
    for entry in TEST_REGISTRY:
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
    for entry in TEST_REGISTRY:
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
