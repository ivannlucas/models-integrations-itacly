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

import inspect
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
    InfeasibleOptimizationError,
    InsufficientCycleHistoryError,
    InsufficientFramesError,
    InsufficientRowsError,
    InsufficientSequenceHistoryError,
    InsufficientSensorWindowError,
    InsufficientTelemetryHistoryError,
    InsufficientWindowHistoryError,
    InvalidImageError,
    InvalidVideoError,
    NoValidSimulationPointError,
    PuConstraintViolationError,
    ThermalSafetyViolationError,
    TrainingNotSupportedError,
    UnknownDiagnosisSystemError,
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
    PredictOptimizeResponse as Ml31ResidueOptimizeResp,
    PredictRequest as Ml31Residue_Request,
    PredictResponse as Ml31Residue_Response,
)
from app.plugins.ml33_cereals_reuse_strategy_optimizer.predict_dto import (
    PredictBatchResponse as Ml33ReuseBatchResp,
    PredictInlineResponse as Ml33ReuseInlineResp,
    PredictRequest as Ml33Reuse_Request,
    PredictResponse as Ml33Reuse_Response,
)
from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.predict_dto import (
    PredictBatchResponse as Ml4ThermalBatchResp,
    PredictInlineResponse as Ml4ThermalInlineResp,
    PredictRequest as Ml4Thermal_Request,
    PredictResponse as Ml4Thermal_Response,
)
from app.plugins.ml23_lactic_market_price_forecast.predict_dto import (
    PredictBatchResponse as Ml23BatchResp,
    PredictInlineResponse as Ml23InlineResp,
    PredictRequest as Ml23_Request,
    PredictResponse as Ml23_Response,
)
from app.plugins.ml17_meat_market_price_analysis.predict_dto import (
    PredictBatchResponse as Ml17BatchResp,
    PredictInlineResponse as Ml17InlineResp,
    PredictRequest as Ml17_Request,
    PredictResponse as Ml17_Response,
)
from app.plugins.ml35_dairy_ann_cleaning_cost.predict_dto import (
    PredictBatchResponse as Ml35DairyBatchResp,
    PredictInlineResponse as Ml35DairyInlineResp,
    PredictRequest as Ml35Dairy_Request,
    PredictResponse as Ml35Dairy_Response,
)
from app.plugins.ml35_dairy_ann_cleaning_cost.train_dto import (
    TrainRequest as Ml35Dairy_TrainReq,
    TrainResponse as Ml35DairyTrainResp,
)
from app.plugins.ml34_dairy_pasteurization_energy_ga.predict_dto import (
    PredictBatchResponse as Ml34DairyBatchResp,
    PredictInlineResponse as Ml34DairyInlineResp,
    PredictOptimizeResponse as Ml34DairyOptimizeResp,
    PredictRequest as Ml34Dairy_Request,
    PredictResponse as Ml34Dairy_Response,
)
from app.plugins.ml34_dairy_pasteurization_energy_ga.train_dto import (
    TrainRequest as Ml34Dairy_TrainReq,
    TrainResponse as Ml34DairyTrainResp,
)
from app.plugins.ml46_dairy_fouling_clog_detection.predict_dto import (
    PredictBatchResponse as Ml46DairyBatchResp,
    PredictInlineResponse as Ml46DairyInlineResp,
    PredictRequest as Ml46Dairy_Request,
    PredictResponse as Ml46Dairy_Response,
)
from app.plugins.ml46_dairy_fouling_clog_detection.train_dto import (
    TrainRequest as Ml46Dairy_TrainReq,
    TrainResponse as Ml46DairyTrainResp,
)
from app.plugins.ml9_cereals_infestation_sequence_classifier.predict_dto import (
    PredictBatchResponse as Ml9CerealsBatchResp,
    PredictInlineResponse as Ml9CerealsInlineResp,
    PredictRequest as Ml9Cereals_Request,
    PredictResponse as Ml9Cereals_Response,
)
from app.plugins.ml9_cereals_infestation_sequence_classifier.train_dto import (
    TrainRequest as Ml9Cereals_TrainReq,
    TrainResponse as Ml9CerealsTrainResp,
)
from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.predict_dto import (
    PredictBatchResponse as M47BatchResp,
    PredictInlineResponse as M47InlineResp,
    PredictRequest as M47_Request,
    PredictResponse as M47_Response,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.predict_dto import (
    PredictBatchResponse as Ml45BatchResp,
    PredictInlineResponse as Ml45InlineResp,
    PredictRequest as Ml45_Request,
    PredictResponse as Ml45_Response,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.train_dto import (
    TrainRequest as Ml45_TrainReq,
    TrainResponse as Ml45TrainResp,
)
from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.predict_dto import (
    PredictBatchResponse as Ml40MeatBatchResp,
    PredictInlineResponse as Ml40MeatInlineResp,
    PredictRequest as Ml40Meat_Request,
    PredictResponse as Ml40Meat_Response,
)
from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.train_dto import (
    TrainRequest as Ml40Meat_TrainReq,
    TrainResponse as Ml40MeatTrainResp,
)
from app.plugins.ml28_meat_neuroevolutionary_raw_materials_prediction.predict_dto import (
    PredictBatchResponse as Ml28MeatBatchResp,
    PredictInlineResponse as Ml28MeatInlineResp,
    PredictRequest as Ml28Meat_Request,
    PredictResponse as Ml28Meat_Response,
)
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.predict_dto import (
    PredictBatchResponse as Ml43BatchResp,
    PredictInlineResponse as Ml43InlineResp,
    PredictRequest as Ml43_Request,
    PredictResponse as Ml43_Response,
)
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.train_dto import (
    TrainRequest as Ml43_TrainReq,
    TrainResponse as Ml43TrainResp,
)
from app.plugins.ml3_wine_disease_pest_forecast.predict_dto import (
    PredictBatchResponse as Ml3WineBatchResp,
    PredictInlineResponse as Ml3WineInlineResp,
    PredictRequest as Ml3Wine_Request,
    PredictResponse as Ml3Wine_Response,
)
from app.plugins.ml3_wine_disease_pest_forecast.train_dto import (
    TrainRequest as Ml3Wine_TrainReq,
    TrainResponse as Ml3WineTrainResp,
)
from app.plugins.m21_cereal_price_spatial.predict_dto import (
    PredictBatchResponse as M21BatchResp,
    PredictInlineResponse as M21InlineResp,
    PredictRequest as M21_Request,
    PredictResponse as M21_Response,
)
from app.plugins.m21_cereal_price_spatial.train_dto import (
    TrainRequest as M21_TrainReq,
    TrainResponse as M21TrainResp,
)
from app.plugins.ml16_meat_raw_material_price_alert.predict_dto import (
    PredictBatchResponse as Ml16BatchResp,
    PredictInlineResponse as Ml16InlineResp,
    PredictRequest as Ml16_Request,
    PredictResponse as Ml16_Response,
)
from app.plugins.ml16_meat_raw_material_price_alert.train_dto import (
    TrainRequest as Ml16_TrainReq,
    TrainResponse as Ml16TrainResp,
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
        data_path: str | None = None,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> BaseModel:
        """Run inline inference on a single feature dict and return the typed inline response."""
        _ = mlflow_run_id, data_path
        if self.raise_on_inline is not None:
            exc, self.raise_on_inline = self.raise_on_inline, None
            raise exc
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        return self._inline_factory(
            self, features=features, model_key=model_key, threshold=threshold
        )

    def predict_batch(
        self, *, data_path: str, model_key: str | None = None, mlflow_run_id: str = "",
    ) -> BaseModel:
        """Run batch inference on a CSV file and return the typed batch response."""
        _ = mlflow_run_id
        if self.raise_on_batch is not None:
            exc, self.raise_on_batch = self.raise_on_batch, None
            raise exc
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        kwargs: dict[str, Any] = {"data_path": data_path}
        # Only ml34's batch_factory declares model_key (GA-vs-MLP dispatch) — every
        # other factory is (plugin, *, data_path), so pass it conditionally.
        if "model_key" in inspect.signature(self._batch_factory).parameters:
            kwargs["model_key"] = model_key
        return self._batch_factory(self, **kwargs)

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


def _ml31_residue_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml31ResidueOptimizeResp:
    """Fake inline response for the ml31 LP residue optimizer."""
    return Ml31ResidueOptimizeResp(
        model_id="ml31-cereals-residue-optimizer",
        reference_year=2023,
        optimization_mode="minimize_residue",
        crop_allocation={
            "Trigo duro": {"secano_ha": 10.0, "regadio_ha": 2.0,
                           "production_t": 5.0, "residue_t": 3.0, "benefit_eur": 120.0}
        },
        total_production_t=98.0,
        total_residue_t=45.0,
        total_benefit_eur=1000.0,
        baseline_total_production_t=100.0,
        baseline_total_residue_t=50.0,
        baseline_total_benefit_eur=1000.0,
        residue_reduction_pct=10.0,
        benefit_change_eur=0.0,
        benefit_change_pct=0.0,
        production_change_pct=-2.0,
        solver_status="OPTIMAL",
        solve_time_seconds=0.01,
        verdict="PASADO",
    )


def _ml31_residue_batch(plugin: FakePlugin, *, data_path: str) -> Ml31ResidueBatchResp:
    """Fake batch response for the ml31 LP residue optimizer."""
    return Ml31ResidueBatchResp(
        model_id="ml31-cereals-residue-optimizer",
        predictions=[{"row": 0, "reference_year": 2023, "optimization_mode": "minimize_residue",
                      "status": "OPTIMAL", "optimized_residue_t": 45.0,
                      "optimized_benefit_eur": 1000.0, "residue_reduction_pct": 10.0,
                      "verdict": "PASADO"}],
        output_path=None,
    )


def _ml33_reuse_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml33ReuseInlineResp:
    """Fake inline response for the ml33 cereal reuse-strategy MILP optimizer."""
    n_lots = len(features.get("lots", [])) or 1
    return Ml33ReuseInlineResp(
        model_id="ml33-cereals-reuse-strategy-optimizer",
        results=[
            {
                "row": i,
                "ai_assigned_strategy": "Animal feed",
                "ai_assignment_source": "exact_min_emissions",
                "ai_is_fallback": False,
                "estimated_emissions_kg": 120.5,
            }
            for i in range(n_lots)
        ],
        distribution={"counts": {"Animal feed": n_lots}, "percentages": {"Animal feed": 100.0}},
        capacity_fallback_count=0,
        total_estimated_emissions_kg=120.5 * n_lots,
    )


def _ml33_reuse_batch(plugin: FakePlugin, *, data_path: str) -> Ml33ReuseBatchResp:
    """Fake batch response for the ml33 cereal reuse-strategy MILP optimizer."""
    return Ml33ReuseBatchResp(
        model_id="ml33-cereals-reuse-strategy-optimizer",
        n_rows=1,
        predictions=[{
            "row": 0, "ai_assigned_strategy": "Composting", "ai_assignment_source": "exact_min_emissions",
            "ai_is_fallback": False, "estimated_emissions_kg": 85.3,
        }],
        distribution={"counts": {"Composting": 1}, "percentages": {"Composting": 100.0}},
        capacity_fallback_count=0,
        total_estimated_emissions_kg=85.3,
        output_path=None,
    )


def _ml4_thermal_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml4ThermalInlineResp:
    """Fake inline response for the ml4 thermal mastitis model."""
    return Ml4ThermalInlineResp(
        model_id="ml4-lactic-cnn-thermal-early-disease-detection",
        threshold=threshold,
        prediction="SCM",
        confidence=0.91,
        features_used=["image_base64"],
        predicted_class_index=1,
        probability_healthy=0.09,
        probability_scm=0.91,
    )


def _ml4_thermal_batch(plugin: FakePlugin, *, data_path: str) -> Ml4ThermalBatchResp:
    """Fake batch response for the ml4 thermal mastitis model."""
    return Ml4ThermalBatchResp(
        model_id="ml4-lactic-cnn-thermal-early-disease-detection",
        predictions=[{"filename": "udder_001.jpg", "prediction": "Healthy", "confidence": 0.88,
                      "predicted_class_index": 0, "probability_healthy": 0.88, "probability_scm": 0.12}],
        output_path=None,
    )


def _ml23_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml23InlineResp:
    """Fake inline response for the ml23 GRU dairy price forecast model."""
    return Ml23InlineResp(
        model_id="ml23-lactic-market-price-forecast",
        prediction=0.9187,
        confidence=None,
        horizon=6,
        features_used=["year", "mes", "precio_lag_1", "current_price"],
        model_version="1.0.0",
        xai_feature_values={"precio_lag_1": 0.9234, "current_price": 0.9187},
    )


def _ml23_batch(plugin: FakePlugin, *, data_path: str) -> Ml23BatchResp:
    """Fake batch response for the ml23 GRU dairy price forecast model."""
    return Ml23BatchResp(
        model_id="ml23-lactic-market-price-forecast",
        predictions=[
            {"fecha": "2023-01-01", "producto": "leche_entera", "canal": "T.ESPAÑA",
             "current_price": 0.9234, "y_pred": 0.9187, "model_id": "ml23-lactic-market-price-forecast",
             "horizon": 6},
        ],
        output_path=None,
    )


def _ml17_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml17InlineResp:
    """Fake inline response for the ml17 Ridge pork price forecast model."""
    return Ml17InlineResp(
        model_id="ml17-meat-market-price-analysis",
        line="official_v1_4",
        prediction=185.32,
        y_pred=185.32,
        confidence=None,
        base_date="2023-01-01",
        xai_feature_values={
            "target_price_pigmeat_class_e_es": 173.82,
            "eurostat_pigmeat_slaughter_tonnes_es": 381.88,
            "eurostat_pigmeat_slaughter_tonnes_eu": 1795.36,
            "cereal_feed_barley_price_monthly": 149.72,
            "cereal_feed_maize_price_monthly": 175.73,
            "mapa_porcino_otras_razas_price_monthly": 117.16,
            "month_sin": 0.5,
            "month_cos": 0.866,
        },
    )


def _ml17_batch(plugin: FakePlugin, *, data_path: str) -> Ml17BatchResp:
    """Fake batch response for the ml17 Ridge pork price forecast model."""
    return Ml17BatchResp(
        model_id="ml17-meat-market-price-analysis",
        line="official_v1_4",
        predictions=[
            {"row": 0, "date": "2023-01-01", "y_pred": 185.32,
             "model_id": "ml17-meat-market-price-analysis", "line": "official_v1_4"},
        ],
        output_path=None,
    )


def _ml35_dairy_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml35DairyInlineResp:
    """Fake inline prediction response for the ml35 dairy ANN plugin."""
    return Ml35DairyInlineResp(
        model_id="ml35-dairy-ann-cleaning-cost",
        consumo_agua_l=24500.0,
        pu_logrado=45.3,
    )


def _ml35_dairy_batch(plugin: FakePlugin, *, data_path: str) -> Ml35DairyBatchResp:
    """Fake batch prediction response for the ml35 dairy ANN plugin."""
    return Ml35DairyBatchResp(
        model_id="ml35-dairy-ann-cleaning-cost",
        predictions=[{"row": 0, "consumo_agua_l": 24500.0, "pu_logrado": 45.3}],
        output_path=None,
    )


def _ml35_dairy_train(plugin: FakePlugin, *, data_path: str) -> Ml35DairyTrainResp:
    """Fake training response for the ml35 dairy ANN plugin."""
    return Ml35DairyTrainResp(detail="Fine-tuning completado", mae=320.0, r2=0.993, n_samples=500)


def _ml34_dairy_inline(plugin: FakePlugin, *, features: dict, model_key, threshold):
    """Fake inline/optimize prediction response for the ml34 pasteurization plugin."""
    if model_key == "optimize":
        return Ml34DairyOptimizeResp(
            model_id="ml34-dairy-pasteurization-energy-ga",
            IA_F_flow=5422.10,
            IA_T_servicio=80.40,
            IA_E_consumo=412.9016,
            IA_T_out=72.30,
            IA_consumo_especifico=0.076152,
            IA_factible=True,
            fitness_final=0.076152,
            seed=1,
        )
    return Ml34DairyInlineResp(
        model_id="ml34-dairy-pasteurization-energy-ga",
        E_consumo_pred=392.9284,
        T_out_pred=72.6027,
    )


def _ml34_dairy_batch(plugin: FakePlugin, *, data_path: str, model_key: str | None = None) -> Ml34DairyBatchResp:
    """Fake batch prediction response for the ml34 pasteurization plugin.

    Mirrors the real predict_batch's GA-vs-MLP dispatch on model_key.
    """
    if model_key == "optimize":
        return Ml34DairyBatchResp(
            model_id="ml34-dairy-pasteurization-energy-ga",
            predictions=[{
                "row": 0,
                "T_in_leche": 6.78,
                "Delta_P": 0.481,
                "t_ciclo": 80.0,
                "IA_F_flow": 5422.10,
                "IA_T_servicio": 80.40,
                "IA_E_consumo": 412.9016,
                "IA_T_out": 72.30,
                "IA_consumo_especifico": 0.076152,
                "IA_factible": True,
                "fitness_final": 0.076152,
                "seed": 1,
            }],
            output_path=None,
        )
    return Ml34DairyBatchResp(
        model_id="ml34-dairy-pasteurization-energy-ga",
        predictions=[{"row": 0, "E_consumo_pred": 392.9284, "T_out_pred": 72.6027}],
        output_path=None,
    )


def _ml34_dairy_train(plugin: FakePlugin, *, data_path: str) -> Ml34DairyTrainResp:
    """Fake fine-tuning response for the ml34 pasteurization plugin."""
    return Ml34DairyTrainResp(
        detail="Fine-tuning completado",
        rmse_E_consumo=5.38, mae_E_consumo=4.26, r2_E_consumo=0.9779,
        rmse_T_out_leche=0.0643, mae_T_out_leche=0.0473, r2_T_out_leche=0.3759,
        n_samples=500, epochs_executed=42,
    )


def _ml46_dairy_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml46DairyInlineResp:
    """Fake inline prediction response for the ml46 dairy fouling/clog detection plugin."""
    return Ml46DairyInlineResp(
        model_id="ml46-dairy-fouling-clog-detection",
        asset_id="asset_00",
        timestamp="2026-01-11T08:00:00+00:00",
        pred_severity=0.000176,
        pred_stage=0,
        pred_stage_name="stable",
        p_stage0=0.999998,
        p_stage1=0.0000001,
        p_stage2=0.0000015,
        p_foul_h=0.0000014,
        p_actionable_foul_h=0.0000083,
        p_clog_h=0.00000036,
        pred_tte_foul_min=166.77,
        pred_tte_clog_min=86.79,
        pred_ttu_min=244.25,
        operator_status="Normal",
        priority="low",
        recommended_action="operación normal",
        activated_predicates="none",
        is_alert=False,
        model_name="ml46-dairy-fouling-clog-detection",
        xai_feature_values={"p_foul_h": 0.0000014, "p_clog_h": 0.00000036},
    )


def _ml46_dairy_batch(plugin: FakePlugin, *, data_path: str) -> Ml46DairyBatchResp:
    """Fake batch prediction response for the ml46 dairy fouling/clog detection plugin."""
    return Ml46DairyBatchResp(
        model_id="ml46-dairy-fouling-clog-detection",
        predictions=[{
            "asset_id": "asset_00", "timestamp": "2026-01-11T08:00:00+00:00",
            "pred_severity": 0.000176, "pred_stage": 0, "pred_stage_name": "stable",
            "operator_status": "Normal",
        }],
        alerts=[],
        output_path=None,
    )


def _ml46_dairy_train(plugin: FakePlugin, *, data_path: str) -> Ml46DairyTrainResp:
    """Fake fine-tuning response for the ml46 dairy fouling/clog detection plugin."""
    return Ml46DairyTrainResp(
        detail="Fine-tuning completado sobre el checkpoint no_clock servido.",
        n_windows=500, epochs=8,
        severity_rmse=0.00012, severity_mae=0.00008,
        stage_accuracy=0.96, stage_macro_f1=0.94,
        watch_foul_auc=0.95, watch_foul_ap=0.46,
        clog_h_auc=0.95, clog_h_ap=0.21,
        tte_foul_mae_min=69.3, tte_clog_mae_min=32.5, ttu_mae_min=109.6,
        upload_warning=None,
    )


def _ml9_cereals_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml9CerealsInlineResp:
    """Fake inline prediction response for the ml9 cereal infestation sequence classifier."""
    confidence = 0.9982911
    return Ml9CerealsInlineResp(
        model_id="ml9-cereals-infestation-sequence-classifier",
        sample_id="S_0_0061",
        window_index=4,
        timestamp_start="2026-01-03T00:00:00",
        timestamp_end="2026-01-04T23:00:00",
        pred_class=0,
        pred_label="sano",
        proba_sano=confidence,
        proba_insectos=0.0016914126,
        proba_moho_critico=0.0000174922,
        confidence=confidence,
        low_confidence=bool(threshold is not None and confidence < threshold),
        n_rows_used=len(features.get("rows", [])),
        n_windows_available=37,
        y_true=None,
        model_name="ml9-cereals-infestation-sequence-classifier",
        xai_feature_values={"proba_sano": confidence, "confidence": confidence, "window_size": 48},
    )


def _ml9_cereals_batch(plugin: FakePlugin, *, data_path: str) -> Ml9CerealsBatchResp:
    """Fake batch prediction response for the ml9 cereal infestation sequence classifier."""
    return Ml9CerealsBatchResp(
        model_id="ml9-cereals-infestation-sequence-classifier",
        n_windows=37,
        n_series=1,
        predictions=[{
            "sample_id": "S_0_0061", "window_index": 4,
            "timestamp_start": "2026-01-03T00:00:00", "timestamp_end": "2026-01-04T23:00:00",
            "pred_class": 0, "pred_label": "sano",
            "proba_sano": 0.9982911, "proba_insectos": 0.0016914126, "proba_moho_critico": 0.0000174922,
            "confidence": 0.9982911,
        }],
        class_distribution={"sano": 37, "insectos": 0, "moho_critico": 0},
        evaluated_metrics=None,
        output_path=None,
    )


def _ml9_cereals_train(plugin: FakePlugin, *, data_path: str) -> Ml9CerealsTrainResp:
    """Fake fine-tuning response for the ml9 cereal infestation sequence classifier."""
    return Ml9CerealsTrainResp(
        detail="Fine-tuning completado sobre el checkpoint servido (GRU).",
        n_series_train=195, n_series_validation=45, n_series_test=60,
        n_windows_train=7215, n_windows_validation=1665, n_windows_test=2220,
        epochs_run=7,
        accuracy=0.9414, balanced_accuracy=0.9452, f1_macro=0.9436,
        precision_macro=0.9422, recall_macro=0.9452, log_loss=0.1830,
        validation_f1_macro=0.9543,
        baseline_f1_macro=0.9436,
        artifact_path="artifacts/ml9_cereals_infestation_sequence_classifier/user_final_winner.pt",
        upload_warning=None,
    )


def _m47_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> M47InlineResp:
    """Fake inline response for the m47 DNSL model."""
    return M47InlineResp(
        model_id="m47-dnsl-fallas-maquinaria-pasteurizado",
        Enfriador_Fouling=0,
        Valvula_Switch=0,
        Bomba_Leakage=0,
        Acumulador_Gas=0,
        Confianza_Fouling=0.99,
        Confianza_Valvula=0.98,
        Confianza_Bomba=0.97,
        Confianza_Acumulador=0.96,
        model_name="m47-dnsl-fallas-maquinaria-pasteurizado",
    )


def _m47_batch(plugin: FakePlugin, *, data_path: str) -> M47BatchResp:
    """Fake batch response for the m47 DNSL model."""
    return M47BatchResp(
        model_id="m47-dnsl-fallas-maquinaria-pasteurizado",
        predictions=[{
            "Cycle_ID": 1,
            "Enfriador_Fouling": 0,
            "Válvula_Switch": 0,
            "Bomba_Leakage": 0,
            "Acumulador_Gas": 0,
            "Enfriador_Fouling_Texto": "SANO",
            "Válvula_Switch_Texto": "SANO",
            "Bomba_Leakage_Texto": "SANO",
            "Acumulador_Gas_Texto": "SANO",
            "Confianza_Fouling": 0.99,
            "Confianza_Valvula": 0.98,
            "Confianza_Bomba": 0.97,
            "Confianza_Acumulador": 0.96,
            "model_name": "m47-dnsl-fallas-maquinaria-pasteurizado",
        }],
        output_path=None,
    )


def _ml45_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml45InlineResp:
    """Fake inline response for the m45 grain-dryer PCC detection model."""
    return Ml45InlineResp(
        model_id="ml45-cereals-dnsl-critical-point-detection",
        window_index=1,
        timestamp_init="2029-04-15 00:00:00",
        timestamp_end="2029-04-15 03:59:00",
        predicted_anomaly_class=0,
        predicted_anomaly_label="No Fallo",
        anomaly_probability=0.43,
        decision_threshold=0.73,
        **{
            "Estado interpretativo": "Vigilancia",
            "Evidencia": "No se identifica un perfil catalogado de criticidad, pero hay indicios de anomalia.",
            "Probabilidad de anomalia": 0.43,
            "Umbral de detección de anomalias": 0.73,
            "Margen respecto al umbral": 0.3,
            "Recomendacion": "Se recomienda vigilancia reforzada y seguimiento.",
        },
    )


def _ml45_batch(plugin: FakePlugin, *, data_path: str) -> Ml45BatchResp:
    """Fake batch response for the m45 grain-dryer PCC detection model."""
    return Ml45BatchResp(
        model_id="ml45-cereals-dnsl-critical-point-detection",
        predictions=[{
            "window_index": 1,
            "cycle_id": 2400,
            "timestamp_init": "2029-04-15 00:00:00",
            "timestamp_end": "2029-04-15 03:59:00",
            "predicted_anomaly_class": 0,
            "predicted_anomaly_label": "No Fallo",
            "anomaly_probability": 0.43,
            "decision_threshold": 0.73,
            "Estado interpretativo": "Vigilancia",
            "Evidencia": "No se identifica un perfil catalogado de criticidad, pero hay indicios de anomalia.",
            "Probabilidad de anomalia": 0.43,
            "Umbral de detección de anomalias": 0.73,
            "Margen respecto al umbral": 0.3,
            "Recomendacion": "Se recomienda vigilancia reforzada y seguimiento.",
        }],
        output_path=None,
    )


def _ml45_train(plugin: FakePlugin, *, data_path: str) -> Ml45TrainResp:
    """Fake fine-tuning response for the m45 grain-dryer PCC detection model."""
    return Ml45TrainResp(
        detail="Fine-tuning completado",
        accuracy=0.92, f1=0.87, auc=0.91, n_windows=40, n_epochs=30,
    )


def _ml40_meat_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml40MeatInlineResp:
    """Fake inline prediction response for the ml40 refrigeration/aeration fault diagnosis plugin."""
    return Ml40MeatInlineResp(
        model_id="ml40-meat-refrigeration-aeration-fault-diagnosis",
        system="aireado",
        run_id=0,
        prediction=0,
        prediction_name="NORMAL",
        confidence=0.9987,
        n_rows_used=100,
        model_health="ESTABLE",
        model_name="ml40-meat-refrigeration-aeration-fault-diagnosis",
        xai_feature_values={"prediction": 0, "confidence": 0.9987, "n_rows": 100},
    )


def _ml40_meat_batch(plugin: FakePlugin, *, data_path: str) -> Ml40MeatBatchResp:
    """Fake batch prediction response for the ml40 refrigeration/aeration fault diagnosis plugin."""
    return Ml40MeatBatchResp(
        model_id="ml40-meat-refrigeration-aeration-fault-diagnosis",
        system="refrigeracion",
        predictions=[
            {"run_id": 1, "fault_id": 0, "prediction": 0, "prediction_name": "NORMAL",
             "confidence": 0.9539},
        ],
        n_runs=1,
        avg_confidence=0.9539,
        model_health="ESTABLE",
        output_path=None,
    )


def _ml40_meat_train(plugin: FakePlugin, *, data_path: str) -> Ml40MeatTrainResp:
    """Fake retraining response for the ml40 refrigeration/aeration fault diagnosis plugin."""
    return Ml40MeatTrainResp(
        detail="Reentrenamiento completado para el sistema aireado con el procedimiento original.",
        system="aireado",
        n_samples=24000,
        n_runs_train=240,
        n_runs_test=60,
        accuracy=1.0,
        f1_macro=1.0,
        precision_macro=1.0,
        recall_macro=1.0,
        upload_warning=None,
    )


def _ml28_meat_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml28MeatInlineResp:
    """Fake inline response for the ml28 meat raw-material procurement rules engine."""
    return Ml28MeatInlineResp(
        model_id="ml28-meat-neuroevolutionary-raw-materials-prediction",
        date="2025-01-26", raw_material_id="RM_BEEF_TRIM_A", destination_profile="cooked_standard_line",
        current_inventory_tons=36.0, expected_requirement_tons=24.0, lead_time_days=6.0,
        safety_coverage_days=11.0, expected_yield_rate=0.88, expected_waste_rate=0.02,
        unit_purchase_cost=3.88, shelf_life_days=28,
        purchase_trigger_proba=0.8588, purchase_trigger_flag=1, recommended_action="BUY",
        quantity_optimizer_recommendation_tons=29.817, order_quantity_tons=29.817,
        decision_reason="Projected stock after lead time is below safety stock. Purchase triggered and quantity optimized under current policy.",
        projected_stock_after_lead_time_tons=15.429, safety_stock_tons=37.714, coverage_gap_tons=22.286,
        risk_level="HIGH", baseline_order_quantity_tons=40.75, delta_order_vs_baseline_tons=-10.933,
        excess_tons=41.337, stockout_tons=0.0,
    )


def _ml28_meat_batch(plugin: FakePlugin, *, data_path: str) -> Ml28MeatBatchResp:
    """Fake batch response for the ml28 meat raw-material procurement rules engine."""
    return Ml28MeatBatchResp(
        model_id="ml28-meat-neuroevolutionary-raw-materials-prediction",
        predictions=[{
            "date": "2025-01-05", "raw_material_id": "RM_BEEF_TRIM_A", "destination_profile": "cooked_standard_line",
            "current_inventory_tons": 58.0, "expected_requirement_tons": 22.0, "lead_time_days": 5.0,
            "safety_coverage_days": 10.0, "expected_yield_rate": 0.89, "expected_waste_rate": 0.02,
            "unit_purchase_cost": 3.80, "shelf_life_days": 28,
            "purchase_trigger_proba": 0.2658, "purchase_trigger_flag": 0, "recommended_action": "DO_NOT_BUY",
            "quantity_optimizer_recommendation_tons": 0.0, "order_quantity_tons": 0.0,
            "decision_reason": "Coverage remains above safety threshold; purchase blocked.",
            "projected_stock_after_lead_time_tons": 42.286, "safety_stock_tons": 31.429, "coverage_gap_tons": 0.0,
            "risk_level": "LOW", "baseline_order_quantity_tons": 3.767, "delta_order_vs_baseline_tons": -3.767,
            "excess_tons": 35.56, "stockout_tons": 0.0,
        }],
        summary={"row_count": 1, "triggered_orders": 0, "aggregate_excess_reduction_pct": 20.959, "stockout_guardrail_pass": True},
        output_path=None,
    )


def _ml43_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml43InlineResp:
    """Fake inline response for the ml43 cereal dryer DNF anomaly/fault detection model (CU43+CU44)."""
    return Ml43InlineResp(
        model_id="ml43-cereals-dnsl-anomaly-fault-detection",
        predicted_anomaly_class=0,
        predicted_anomaly_label="No Fallo",
        anomaly_probability=0.0512,
        decision_threshold=threshold if threshold is not None else 0.41,
        xai_feature_values={"temp_zona1": 80.0, "temp_zona2": 82.0},
        corrective_actions=None,
        xai_error=None,
        model_name="ml43-cereals-dnsl-anomaly-fault-detection",
    )


def _ml43_batch(plugin: FakePlugin, *, data_path: str) -> Ml43BatchResp:
    """Fake batch response for the ml43 cereal dryer DNF anomaly/fault detection model (CU43+CU44)."""
    return Ml43BatchResp(
        model_id="ml43-cereals-dnsl-anomaly-fault-detection",
        predictions=[{
            "window_index": 1,
            "cycle_id": "2400",
            "predicted_anomaly_class": 0,
            "predicted_anomaly_label": "No Fallo",
            "anomaly_probability": 0.0512,
            "decision_threshold": 0.41,
            "xai_feature_values": {"temp_zona1": 80.0, "temp_zona2": 82.0},
            "corrective_actions": None,
            "xai_error": None,
        }],
        output_path=None,
    )


def _ml43_train(plugin: FakePlugin, *, data_path: str) -> Ml43TrainResp:
    """Fake training response for the ml43 cereal dryer DNF anomaly/fault detection model."""
    return Ml43TrainResp(
        detail="Entrenamiento completado",
        accuracy=0.989, macro_f1=0.9498, macro_precision=0.9415, macro_recall=0.9585,
        fallo_f1=0.9054, fallo_precision=0.8876, fallo_recall=0.9240,
        n_train=28, n_test=7, n_windows_total=35,
        upload_warning=None,
    )


def _ml3_wine_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml3WineInlineResp:
    """Fake inline response for the ml3 wine disease/pest Deep Ensemble plugin."""
    return Ml3WineInlineResp(
        model_id="ml3-wine-disease-pest-forecast",
        id_serie=552,
        fecha_evaluacion="2021-06-18 00:00:00",
        diagnostico_ia="ALTICA",
        confianza_clasificacion=0.960752,
        grado_severidad=0.758976,
        tratamiento_recomendado=(
            "Químico: Generalmente controlado por tratamientos para Lobesia. "
            "Específicos: Lambda cihalotrin y clorpirifos."
        ),
        probabilidades_clases={
            "ALTICA": 0.960752, "BLACK_ROT": 0.0, "BOTRYTIS": 0.0, "EMPOASCA": 0.0,
            "ERINOSIS": 0.0, "ESCA": 0.0, "HEALTHY": 0.0, "LOBESIA": 0.0,
            "MILDIU": 0.0, "OIDIO": 0.039248, "RED_MITE": 0.0,
        },
        model_name="ml3-wine-disease-pest-forecast",
        xai_feature_values={"Temp_Amb_C": 16.6, "Hum_Rel_Pct": 91.0},
    )


def _ml3_wine_batch(plugin: FakePlugin, *, data_path: str) -> Ml3WineBatchResp:
    """Fake batch response for the ml3 wine disease/pest Deep Ensemble plugin."""
    return Ml3WineBatchResp(
        model_id="ml3-wine-disease-pest-forecast",
        predictions=[
            {
                "id_serie": 552,
                "fecha_evaluacion": "2021-06-18 00:00:00",
                "diagnostico_ia": "ALTICA",
                "confianza_clasificacion": 0.960752,
                "grado_severidad": 0.758976,
                "tratamiento_recomendado": "Diagnóstico: Planta sana.",
            }
        ],
        output_path=None,
    )


def _ml3_wine_train(plugin: FakePlugin, *, data_path: str) -> Ml3WineTrainResp:
    """Fake retraining response for the ml3 wine disease/pest Deep Ensemble plugin."""
    return Ml3WineTrainResp(
        detail="Reentrenamiento completo del Deep Ensemble (LSTM + CNN + BiGRU) desde cero.",
        n_windows_train=100000,
        n_windows_val=21400,
        n_windows_test=21400,
        epochs_executed=50,
        accuracy=0.7052,
        precision_macro=0.7494,
        recall_macro=0.7198,
        f1_macro=0.7259,
        f1_weighted=0.7055,
        mae=0.0610,
        mse=0.00876,
        r2=0.8903,
        upload_warning=None,
    )


def _m21_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> M21InlineResp:
    """Fake inline response for the m21 ESP-CEREAL spatial cereal price model."""
    return M21InlineResp(
        model_id="m21-cereal-price-spatial",
        province="Burgos",
        cereal="trigo",
        month="2024-01",
        geo_risk=False,
        timing_label="SEGUIMIENTO ACTIVO",
        causal_drivers=["Trigo Internacional (al alza, z=1.23)", "EUR/USD (a la baja, z=0.87)", "Precipitacion (al alza, z=0.65)"],
        card_text="FICHA DE DECISION DATAGIA v1.1\n...",
        predictions={
            "H1": {"horizon": 1, "signal": "NEUTRAL/ESPERA", "confidence": 55.0, "expected_return": 0.012, "prob_up": 0.55},
            "H2": {"horizon": 2, "signal": "ALCISTA", "confidence": 68.0, "expected_return": 0.025, "prob_up": 0.68},
            "H3": {"horizon": 3, "signal": "NEUTRAL/ESPERA", "confidence": 52.0, "expected_return": 0.008, "prob_up": 0.52},
        },
        model_version="1.0.0",
        xai_feature_values={"corn_intl_eur_lag_1": 185.0, "wheat_intl_eur_lag_1": 210.0},
    )


def _m21_batch(plugin: FakePlugin, *, data_path: str) -> M21BatchResp:
    """Fake batch response for the m21 ESP-CEREAL spatial cereal price model."""
    return M21BatchResp(
        model_id="m21-cereal-price-spatial",
        predictions=[
            {
                "row": 0,
                "provincia": "Burgos",
                "cereal_predominante": "trigo",
                "ret_h1": 0.012, "ret_h2": 0.025, "ret_h3": 0.008,
                "prob_up_h1": 0.55, "prob_up_h2": 0.68, "prob_up_h3": 0.52,
                "signal_h1": "NEUTRAL/ESPERA", "signal_h2": "ALCISTA", "signal_h3": "NEUTRAL/ESPERA",
                "confidence_h1": 55.0, "confidence_h2": 68.0, "confidence_h3": 52.0,
            }
        ],
        output_path=None,
    )


def _m21_train(plugin: FakePlugin, *, data_path: str) -> M21TrainResp:
    """Fake training response for the m21 ESP-CEREAL spatial cereal price model."""
    return M21TrainResp(
        detail="Training completado — 6 modelos entrenados (3H × reg+clf)",
        mae_h1=0.0508, mae_h2=0.0674, mae_h3=0.1012,
        pearson_h1=0.3785, pearson_h2=0.2653, pearson_h3=0.2093,
        da_h1=0.5591, da_h2=0.5656, da_h3=0.6250,
        auc_h1=0.7116, auc_h2=0.7441, auc_h3=0.7523,
        n_train=1500, n_test=500,
        upload_warning=None,
    )


def _ml16_inline(plugin: FakePlugin, *, features: dict, model_key, threshold) -> Ml16InlineResp:
    """Fake inline response for the ml16 meat raw-material price alert model."""
    return Ml16InlineResp(
        model_id="ml16-meat-raw-material-price-alert",
        fecha="2024-12-01",
        target_animales_pred=1,
        target_animales_proba=0.9339,
        target_animales_proba_low=0.7613,
        target_animales_proba_high=0.9453,
        target_insumos_pred=1,
        target_insumos_proba=0.3345,
        target_insumos_proba_low=0.2865,
        target_insumos_proba_high=0.3879,
        n_rows_used=len(features.get("rows", [])),
        n_predictions_available=47,
        model_name="ml16-meat-raw-material-price-alert",
        xai_feature_values={"target_animales_proba": 0.9339, "target_insumos_proba": 0.3345},
    )


def _ml16_batch(plugin: FakePlugin, *, data_path: str) -> Ml16BatchResp:
    """Fake batch response for the ml16 meat raw-material price alert model."""
    return Ml16BatchResp(
        model_id="ml16-meat-raw-material-price-alert",
        predictions=[{
            "fecha": "2024-01-01",
            "target_animales_pred": 0, "target_animales_proba": 0.2503,
            "target_animales_proba_low": 0.16, "target_animales_proba_high": 0.6625,
            "target_insumos_pred": 1, "target_insumos_proba": 0.3581,
            "target_insumos_proba_low": 0.3351, "target_insumos_proba_high": 0.3982,
        }],
        n_predictions=1,
        output_path=None,
    )


def _ml16_train(plugin: FakePlugin, *, data_path: str) -> Ml16TrainResp:
    """Fake retraining response for the ml16 meat raw-material price alert model."""
    return Ml16TrainResp(
        detail="Reentrenamiento completado (XGBoost + LogisticRegression, procedimiento original).",
        n_train_rows=35,
        n_test_rows=12,
        target_animales_threshold=0.48,
        target_animales_accuracy=0.833,
        target_animales_precision=0.833,
        target_animales_recall=0.833,
        target_animales_f1=0.833,
        target_animales_auc=0.917,
        target_insumos_threshold=0.30,
        target_insumos_accuracy=0.667,
        target_insumos_precision=0.429,
        target_insumos_recall=1.0,
        target_insumos_f1=0.6,
        target_insumos_auc=0.741,
        upload_warning=None,
    )


FAKE_FACTORIES: dict[str, tuple[Callable, Callable]] = {
    "ml9-cereals-infestation-sequence-classifier": (_ml9_cereals_inline, _ml9_cereals_batch),
    "ml46-dairy-fouling-clog-detection": (_ml46_dairy_inline, _ml46_dairy_batch),
    "ml40-meat-refrigeration-aeration-fault-diagnosis": (_ml40_meat_inline, _ml40_meat_batch),
    "ml28-meat-neuroevolutionary-raw-materials-prediction": (_ml28_meat_inline, _ml28_meat_batch),
    "ml35-dairy-ann-cleaning-cost": (_ml35_dairy_inline, _ml35_dairy_batch),
    "ml34-dairy-pasteurization-energy-ga": (_ml34_dairy_inline, _ml34_dairy_batch),
    "ml17-meat-market-price-analysis": (_ml17_inline, _ml17_batch),
    "ml23-lactic-market-price-forecast": (_ml23_inline, _ml23_batch),
    "ml4-lactic-cnn-thermal-early-disease-detection": (_ml4_thermal_inline, _ml4_thermal_batch),
    "ml31-cereals-residue-optimizer": (_ml31_residue_inline, _ml31_residue_batch),
    "ml33-cereals-reuse-strategy-optimizer": (_ml33_reuse_inline, _ml33_reuse_batch),
    "ml30-meat-traceability-detection": (_ml30_trace_inline, _ml30_trace_batch),
    "ml7-cereals-grain-pest-detection": (_ml7_grain_inline, _ml7_grain_batch),
    "ml2-fungal-cnn-disease-detection": (_ml2_fungal_inline, _ml2_fungal_batch),
    "wine-sulphite": (_wine_so2_inline, _wine_so2_batch),
    "modelo10-lacteo": (_lacteo_inline, _lacteo_batch),
    "ml8-cereals-img-anomaly-detector": (_ml8_cereals_inline, _ml8_cereals_batch),
    "ml5-meat-cow-behaviour": (_ml5_cow_inline, _ml5_cow_batch),
    "m47-dnsl-fallas-maquinaria-pasteurizado": (_m47_inline, _m47_batch),
    "ml45-cereals-dnsl-critical-point-detection": (_ml45_inline, _ml45_batch),
    "ml43-cereals-dnsl-anomaly-fault-detection": (_ml43_inline, _ml43_batch),
    "ml3-wine-disease-pest-forecast": (_ml3_wine_inline, _ml3_wine_batch),
    "m21-cereal-price-spatial": (_m21_inline, _m21_batch),
    "ml16-meat-raw-material-price-alert": (_ml16_inline, _ml16_batch),
}

TRAIN_FACTORIES: dict[str, Callable] = {
    "ml9-cereals-infestation-sequence-classifier": _ml9_cereals_train,
    "ml46-dairy-fouling-clog-detection": _ml46_dairy_train,
    "ml40-meat-refrigeration-aeration-fault-diagnosis": _ml40_meat_train,
    "ml35-dairy-ann-cleaning-cost": _ml35_dairy_train,
    "ml34-dairy-pasteurization-energy-ga": _ml34_dairy_train,
    "modelo10-lacteo": _lacteo_train,
    "ml8-cereals-img-anomaly-detector": _ml8_cereals_train,
    "ml30-meat-traceability-detection": _ml30_trace_train,
    "ml45-cereals-dnsl-critical-point-detection": _ml45_train,
    "ml43-cereals-dnsl-anomaly-fault-detection": _ml43_train,
    "ml3-wine-disease-pest-forecast": _ml3_wine_train,
    "m21-cereal-price-spatial": _m21_train,
    "ml16-meat-raw-material-price-alert": _ml16_train,
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
        extra_predict_exceptions=(InfeasibleOptimizationError,),
    ),
    ModelEntry(
        model_id="ml33-cereals-reuse-strategy-optimizer",
        prefix="/models/ml33-cereals-reuse-strategy-optimizer",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml33Reuse_Request,
        predict_response_type=Ml33Reuse_Response,
        extra_predict_exceptions=(),
    ),
    ModelEntry(
        model_id="ml4-lactic-cnn-thermal-early-disease-detection",
        prefix="/models/ml4-lactic-cnn-thermal-early-disease-detection",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml4Thermal_Request,
        predict_response_type=Ml4Thermal_Response,
        extra_predict_exceptions=(InvalidImageError,),
    ),
    ModelEntry(
        model_id="ml23-lactic-market-price-forecast",
        prefix="/models/ml23-lactic-market-price-forecast",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml23_Request,
        predict_response_type=Ml23_Response,
        extra_predict_exceptions=(),
    ),
    ModelEntry(
        model_id="ml17-meat-market-price-analysis",
        prefix="/models/ml17-meat-market-price-analysis",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml17_Request,
        predict_response_type=Ml17_Response,
        extra_predict_exceptions=(),
    ),
    ModelEntry(
        model_id="ml35-dairy-ann-cleaning-cost",
        prefix="/models/ml35-dairy-ann-cleaning-cost",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml35Dairy_Request,
        predict_response_type=Ml35Dairy_Response,
        extra_predict_exceptions=(PuConstraintViolationError,),
        train_request_type=Ml35Dairy_TrainReq,
        train_response_type=Ml35DairyTrainResp,
    ),
    ModelEntry(
        model_id="ml34-dairy-pasteurization-energy-ga",
        prefix="/models/ml34-dairy-pasteurization-energy-ga",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml34Dairy_Request,
        predict_response_type=Ml34Dairy_Response,
        extra_predict_exceptions=(ThermalSafetyViolationError,),
        train_request_type=Ml34Dairy_TrainReq,
        train_response_type=Ml34DairyTrainResp,
    ),
    ModelEntry(
        model_id="ml9-cereals-infestation-sequence-classifier",
        prefix="/models/ml9-cereals-infestation-sequence-classifier",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml9Cereals_Request,
        predict_response_type=Ml9Cereals_Response,
        extra_predict_exceptions=(InsufficientSequenceHistoryError,),
        train_request_type=Ml9Cereals_TrainReq,
        train_response_type=Ml9CerealsTrainResp,
    ),
    ModelEntry(
        model_id="ml46-dairy-fouling-clog-detection",
        prefix="/models/ml46-dairy-fouling-clog-detection",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml46Dairy_Request,
        predict_response_type=Ml46Dairy_Response,
        extra_predict_exceptions=(InsufficientTelemetryHistoryError,),
        train_request_type=Ml46Dairy_TrainReq,
        train_response_type=Ml46DairyTrainResp,
    ),
    ModelEntry(
        model_id="m47-dnsl-fallas-maquinaria-pasteurizado",
        prefix="/models/m47-dnsl-fallas-maquinaria-pasteurizado",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=M47_Request,
        predict_response_type=M47_Response,
        extra_predict_exceptions=(),
    ),
    ModelEntry(
        model_id="ml45-cereals-dnsl-critical-point-detection",
        prefix="/models/ml45-cereals-dnsl-critical-point-detection",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml45_Request,
        predict_response_type=Ml45_Response,
        extra_predict_exceptions=(InsufficientWindowHistoryError,),
        train_request_type=Ml45_TrainReq,
        train_response_type=Ml45TrainResp,
    ),
    ModelEntry(
        model_id="ml40-meat-refrigeration-aeration-fault-diagnosis",
        prefix="/models/ml40-meat-refrigeration-aeration-fault-diagnosis",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml40Meat_Request,
        predict_response_type=Ml40Meat_Response,
        extra_predict_exceptions=(InsufficientCycleHistoryError, UnknownDiagnosisSystemError),
        train_request_type=Ml40Meat_TrainReq,
        train_response_type=Ml40MeatTrainResp,
    ),
    ModelEntry(
        model_id="ml28-meat-neuroevolutionary-raw-materials-prediction",
        prefix="/models/ml28-meat-neuroevolutionary-raw-materials-prediction",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml28Meat_Request,
        predict_response_type=Ml28Meat_Response,
        extra_predict_exceptions=(),
    ),
    ModelEntry(
        model_id="ml43-cereals-dnsl-anomaly-fault-detection",
        prefix="/models/ml43-cereals-dnsl-anomaly-fault-detection",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml43_Request,
        predict_response_type=Ml43_Response,
        extra_predict_exceptions=(InsufficientSensorWindowError,),
        train_request_type=Ml43_TrainReq,
        train_response_type=Ml43TrainResp,
    ),
    ModelEntry(
        model_id="ml3-wine-disease-pest-forecast",
        prefix="/models/ml3-wine-disease-pest-forecast",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml3Wine_Request,
        predict_response_type=Ml3Wine_Response,
        extra_predict_exceptions=(),
        train_request_type=Ml3Wine_TrainReq,
        train_response_type=Ml3WineTrainResp,
    ),
    ModelEntry(
        model_id="m21-cereal-price-spatial",
        prefix="/models/m21-cereal-price-spatial",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=M21_Request,
        predict_response_type=M21_Response,
        extra_predict_exceptions=(),
        train_request_type=M21_TrainReq,
        train_response_type=M21TrainResp,
    ),
    ModelEntry(
        model_id="ml16-meat-raw-material-price-alert",
        prefix="/models/ml16-meat-raw-material-price-alert",
        version="1.0.0",
        plugin_class=FakePlugin,
        predict_request_type=Ml16_Request,
        predict_response_type=Ml16_Response,
        extra_predict_exceptions=(InsufficientRowsError,),
        train_request_type=Ml16_TrainReq,
        train_response_type=Ml16TrainResp,
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
