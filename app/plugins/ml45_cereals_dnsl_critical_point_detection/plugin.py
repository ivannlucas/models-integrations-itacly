from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone

import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml45_cereals_dnsl_critical_point_detection.constants import (
    FRAMEWORK,
    MODEL_FILENAME,
    MODEL_ID,
    SENSOR_COLUMNS,
    VERSION,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.mlflow_utils import (
    download_user_model_from_mlflow,
    upload_artifacts_to_mlflow,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.model_loader import (
    _store,
    load_artifacts,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.postprocessing import (
    build_explainer,
    explain_window,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.postprocessing import (
    build_predictions,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.preprocessing import (
    build_windows_from_csv,
    build_windows_from_sensor_arrays,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.train_dto import TrainResponse
from app.plugins.ml45_cereals_dnsl_critical_point_detection.trainer import fine_tune

logger = logging.getLogger(__name__)


class Ml45CerealsDnslCriticalPointDetectionPlugin(ModelPluginPort):
    """Deep Neuro-Fuzzy (LSTM+attention || fuzzy TSK rules) grain-dryer fault detector.

    Predicts per-window (240 consecutive readings) binary anomaly, then classifies into
    Normal/Vigilancia/Criticidad detectada via the built-in PCC (Puntos Críticos de Control)
    monitor — SHAP + fuzzy-rule explainability ported inline (not delegated to the external
    explainability microservice), per this model's own functional design.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_cfg: dict | None = None
        self._scaler_x = None
        self._scaler_num = None
        self._xai_background = None
        self._threshold: float = 0.73
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        bucket = os.environ.get("STORAGE_BUCKET")
        if bucket:
            _store.download_all_if_needed()

        model, model_cfg, scaler_x, scaler_num, xai_background, threshold = load_artifacts()
        self._model = model
        self._model_cfg = model_cfg
        self._scaler_x = scaler_x
        self._scaler_num = scaler_num
        self._xai_background = xai_background
        self._threshold = threshold
        logger.info("m45 plugin loaded: %s (threshold=%.3f)", MODEL_ID, threshold)

    def is_loaded(self) -> bool:
        return self._model is not None

    def _require_loaded(self) -> None:
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _load_model_for_predict(self, mlflow_run_id: str):
        if not mlflow_run_id:
            self._require_loaded()
            return None
        logger.info("Using user-trained model from MLflow run_id=%s", mlflow_run_id)
        loaded = download_user_model_from_mlflow(mlflow_run_id)
        if loaded is None:
            logger.warning("MLflow download failed for %s, falling back to standard model", mlflow_run_id)
            self._require_loaded()
            return None
        model, model_cfg, scaler_x, scaler_num, xai_background, threshold, temp_dir = loaded
        return {
            "model": model,
            "model_cfg": model_cfg,
            "scaler_x": scaler_x,
            "scaler_num": scaler_num,
            "xai_background": xai_background if xai_background is not None else self._xai_background,
            "threshold": threshold,
            "temp_dir": temp_dir,
        }

    def predict_inline(
        self,
        *,
        data_path: str | None = None,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        _ = model_key, threshold
        mlflow_ctx = self._load_model_for_predict(mlflow_run_id)
        ctx = mlflow_ctx or {
            "model": self._model,
            "model_cfg": self._model_cfg,
            "scaler_x": self._scaler_x,
            "scaler_num": self._scaler_num,
            "xai_background": self._xai_background,
            "threshold": self._threshold,
            "temp_dir": None,
        }
        try:
            if data_path:
                with local_file_path(data_path) as local_path:
                    sequences, stats, stats_cols, ts_windows, entity_ids, _ = build_windows_from_csv(
                        local_path, ctx["scaler_x"], ctx["scaler_num"], require_target=False,
                    )
            else:
                sensor_arrays = {col: features[col] for col in SENSOR_COLUMNS}
                sequences, stats, stats_cols, ts_windows, entity_ids, _ = build_windows_from_sensor_arrays(
                    sensor_arrays,
                    features["timestamp"],
                    features.get("cycle_id"),
                    ctx["scaler_x"],
                    ctx["scaler_num"],
                )

            if len(sequences) > 1:
                logger.warning(
                    "predict_inline received enough rows for %d windows; only the first "
                    "window is used. Use predict_batch to get a prediction per window.",
                    len(sequences),
                )

            explainer = build_explainer(ctx["model"], stats_cols, ctx["model_cfg"])
            row = explain_window(
                explainer,
                x_window=sequences[0],
                s_stats=stats[0],
                xai_background=ctx["xai_background"],
                threshold=ctx["threshold"],
            )
            self._record()
            return PredictInlineResponse(
                model_id=MODEL_ID,
                window_index=1,
                timestamp_init=str(ts_windows[0][0]),
                timestamp_end=str(ts_windows[0][1]),
                **row,
            )
        finally:
            if mlflow_ctx and mlflow_ctx["temp_dir"]:
                shutil.rmtree(mlflow_ctx["temp_dir"], ignore_errors=True)

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        mlflow_ctx = self._load_model_for_predict(mlflow_run_id)
        ctx = mlflow_ctx or {
            "model": self._model,
            "model_cfg": self._model_cfg,
            "scaler_x": self._scaler_x,
            "scaler_num": self._scaler_num,
            "xai_background": self._xai_background,
            "threshold": self._threshold,
            "temp_dir": None,
        }
        try:
            with local_file_path(data_path) as local_path:
                sequences, stats, stats_cols, ts_windows, entity_ids, _ = build_windows_from_csv(
                    local_path, ctx["scaler_x"], ctx["scaler_num"], require_target=False,
                )

            predictions = build_predictions(
                ctx["model"],
                sequences,
                stats,
                stats_cols,
                ts_windows,
                entity_ids,
                ctx["xai_background"],
                ctx["threshold"],
                ctx["model_cfg"],
            )
            self._record()
            return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)
        finally:
            if mlflow_ctx and mlflow_ctx["temp_dir"]:
                shutil.rmtree(mlflow_ctx["temp_dir"], ignore_errors=True)

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> TrainResponse:
        self._require_loaded()
        with local_file_path(data_path) as local_path:
            sequences, stats, _stats_cols, _ts_windows, _entity_ids, y_labels = build_windows_from_csv(
                local_path, self._scaler_x, self._scaler_num, require_target=True,
            )

        fine_model, metrics = fine_tune(
            self._model, self._model_cfg, sequences, stats, y_labels, self._threshold,
        )

        _store.local_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": fine_model.state_dict(), "model_cfg": self._model_cfg},
            _store.local_dir / MODEL_FILENAME,
        )

        if mlflow_run_id:
            import tempfile

            mlflow_tmp = tempfile.mkdtemp(prefix="m45_mlflow_")
            try:
                torch.save(
                    {"model_state_dict": fine_model.state_dict(), "model_cfg": self._model_cfg},
                    os.path.join(mlflow_tmp, MODEL_FILENAME),
                )
                upload_artifacts_to_mlflow(mlflow_tmp, mlflow_run_id=mlflow_run_id, metrics=metrics)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("MLflow artifact upload failed: %s", exc)
            finally:
                shutil.rmtree(mlflow_tmp, ignore_errors=True)

        self.load()
        logger.info(
            "train() done — acc=%.4f f1=%.4f auc=%.4f n_windows=%d mlflow=%s",
            metrics["accuracy"], metrics["f1"], metrics["auc"], metrics["n_windows"], bool(mlflow_run_id),
        )
        return TrainResponse(
            detail="Fine-tuning completado",
            accuracy=round(metrics["accuracy"], 4),
            f1=round(metrics["f1"], 4),
            auc=round(metrics["auc"], 4) if metrics["auc"] == metrics["auc"] else metrics["auc"],
            n_windows=metrics["n_windows"],
            n_epochs=metrics["n_epochs"],
        )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        inputs = [
            InputField(name=col, type="float", description=f"Sensor {col} time-series (per row)")
            for col in SENSOR_COLUMNS
        ]
        outputs = [
            OutputField(name="predicted_anomaly_class", type="int", description="0=No Fallo, 1=Fallo"),
            OutputField(name="anomaly_probability", type="float", description="Fused fuzzy+LSTM anomaly probability"),
            OutputField(name="Estado interpretativo", type="string", description="Normal | Vigilancia | Criticidad detectada"),
        ]

        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Detección de anomalías en secadoras de grano de flujo mixto mediante un modelo "
                "híbrido Deep Neuro-Fuzzy (LSTM+atención || reglas fuzzy TSK), con capa XAI "
                "(SHAP + reglas fuzzy) y monitor de Puntos Críticos de Control (PCC)."
            ),
            task_type="classification_binary_timeseries_windowed",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics={
                "accuracy": 0.9254166666666667,
                "macro_f1": 0.8669441348101212,
                "fallo_f1": 0.7787391841779975,
                "fallo_auc": 0.9122309027777777,
                "fallo_precision": 0.9574468085106383,
                "fallo_recall": 0.65625,
                "selected_threshold": 0.73,
            },
            runtime_stats=RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=None),
        )
        if mlflow_run_id:
            try:
                tracker = BaseMLflowTracker(mlflow_run_id)
                base.metrics["mlflow"] = {
                    "params": tracker.get_params(),
                    "metrics": tracker.get_metrics(),
                }
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("MLflow stats fetch failed for run_id=%s: %s", mlflow_run_id, exc)
        return base
