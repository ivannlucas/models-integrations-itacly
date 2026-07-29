from __future__ import annotations

import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import ARTIFACTS_ROOT, local_file_path
from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.constants import (
    APPLY_DIGITAL_TWIN,
    ARTIFACT_FOLDER_NAME,
    COMPONENT_NAMES,
    FEATURE_COLUMNS_FILENAME,
    FRAMEWORK,
    MODEL_FILENAME,
    MODEL_ID,
    SCALER_FILENAME,
    SENSOR_COLUMNS,
    TS1_MEAN_FILENAME,
    VERSION,
)
from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.mlflow_utils import (
    download_user_model_from_mlflow,
    upload_artifacts_to_mlflow,
)
from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.model_loader import (
    load_artifacts_from_dir,
)
from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.postprocessing import (
    format_batch_row,
    format_inline_response,
    run_inference,
)
from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.preprocessing import (
    build_dataframe_from_csv,
    build_dataframe_from_sensors,
)
from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.train_dto import TrainResponse
from app.plugins.m47_dnsl_fallas_maquinaria_pasteurizado.trainer import (
    save_training_artifacts,
    train_model_from_csv,
)

logger = logging.getLogger(__name__)

_ARTIFACT_DIR = ARTIFACTS_ROOT / ARTIFACT_FOLDER_NAME


def _cleanup_artifact_files():
    for fname in [MODEL_FILENAME, SCALER_FILENAME, FEATURE_COLUMNS_FILENAME, TS1_MEAN_FILENAME]:
        p = _ARTIFACT_DIR / fname
        if p.exists():
            p.unlink()
            logger.debug("Deleted artifact file: %s", p)


class M47DnsFallMaquinariaPasteurizadoPlugin(ModelPluginPort):
    def __init__(self) -> None:
        self._model = None
        self._scaler = None
        self._feature_columns: list[str] = []
        self._ts1_mean_train: float = 45.0
        self._device: torch.device | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        bucket = os.environ.get("STORAGE_BUCKET")
        if bucket:
            from app.infrastructure.artifact_store import ArtifactStore

            store = ArtifactStore(ARTIFACT_FOLDER_NAME)
            store.download_all_if_needed()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, scaler, feature_cols, ts1_mean_train = load_artifacts_from_dir(_ARTIFACT_DIR)
        self._model = model.to(self._device)
        self._scaler = scaler
        self._feature_columns = feature_cols
        self._ts1_mean_train = ts1_mean_train

        # NOTA: los artefactos se quedan en disco para evitar
        # re-descargar de S3 en cada restart.
        logger.info("M47 plugin loaded: %s (device=%s)", MODEL_ID, self._device)

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
        model, scaler, feature_cols, ts1_mean_train, user_temp_dir = loaded
        model = model.to(self._device)
        return {
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "ts1_mean_train": ts1_mean_train,
            "temp_dir": user_temp_dir,
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
        _ = threshold, model_key
        mlflow_ctx = self._load_model_for_predict(mlflow_run_id)
        ctx = mlflow_ctx or {
            "model": self._model,
            "scaler": self._scaler,
            "feature_cols": self._feature_columns,
            "ts1_mean_train": self._ts1_mean_train,
            "temp_dir": None,
        }
        try:
            self._require_loaded() if mlflow_ctx is None else None

            if data_path:
                with local_file_path(data_path) as local_path:
                    x_df, cycle_ids = build_dataframe_from_csv(
                        local_path, ctx["ts1_mean_train"], APPLY_DIGITAL_TWIN,
                    )
                if cycle_ids is not None and cycle_ids.nunique() > 1:
                    first = cycle_ids.unique()[0]
                    x_df = x_df[cycle_ids == first].drop(columns=["Cycle_ID"], errors="ignore")
                elif cycle_ids is not None:
                    x_df = x_df.drop(columns=["Cycle_ID"], errors="ignore")
            else:
                sensor_data = {col: features[col] for col in SENSOR_COLUMNS}
                x_df = build_dataframe_from_sensors(
                    sensor_data,
                    features.get("Time_Segundos"),
                    features.get("Cycle_ID"),
                    ctx["ts1_mean_train"],
                    APPLY_DIGITAL_TWIN,
                )

            resultados = run_inference(ctx["model"], ctx["scaler"], ctx["feature_cols"], x_df, self._device)
            if resultados is None:
                raise ValueError("Feature columns mismatch during inference")
            self._record()
            dto = format_inline_response(MODEL_ID, resultados)
            return PredictInlineResponse(**dto)
        finally:
            if mlflow_ctx and mlflow_ctx["temp_dir"]:
                shutil.rmtree(mlflow_ctx["temp_dir"], ignore_errors=True)

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        mlflow_ctx = self._load_model_for_predict(mlflow_run_id)
        ctx = mlflow_ctx or {
            "model": self._model,
            "scaler": self._scaler,
            "feature_cols": self._feature_columns,
            "ts1_mean_train": self._ts1_mean_train,
            "temp_dir": None,
        }
        try:
            if mlflow_ctx is None:
                self._require_loaded()
            with local_file_path(data_path) as local_path:
                x_df, cycle_ids = build_dataframe_from_csv(
                    local_path, ctx["ts1_mean_train"], APPLY_DIGITAL_TWIN,
                )

            predictions = []
            if cycle_ids is not None and cycle_ids.nunique() > 1:
                for cid in cycle_ids.unique():
                    mask = cycle_ids == cid
                    cycle_df = x_df[mask].drop(columns=["Cycle_ID"], errors="ignore")
                    if cycle_df.empty:
                        continue
                    result = run_inference(ctx["model"], ctx["scaler"], ctx["feature_cols"], cycle_df, self._device)
                    if result:
                        predictions.append(format_batch_row(result, cid))
            else:
                result = run_inference(ctx["model"], ctx["scaler"], ctx["feature_cols"], x_df, self._device)
                if result:
                    predictions.append(format_batch_row(result, None))

            self._record()
            return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)
        finally:
            if mlflow_ctx and mlflow_ctx["temp_dir"]:
                shutil.rmtree(mlflow_ctx["temp_dir"], ignore_errors=True)

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        logger.info("Training m47 model from data_path=%s, mlflow_run_id=%s", data_path, mlflow_run_id)
        with local_file_path(data_path) as local_path:
            model, scaler, feature_cols, ts1_mean_train, metrics = train_model_from_csv(local_path)

        temp_dir = Path(tempfile.mkdtemp(prefix="m47_train_"))
        try:
            save_training_artifacts(temp_dir, model, scaler, feature_cols, ts1_mean_train)
            new_run_id = upload_artifacts_to_mlflow(str(temp_dir), mlflow_run_id=mlflow_run_id, metrics=metrics)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info("Training complete. MLflow run_id=%s", new_run_id)
        return TrainResponse(
            detail="Entrenamiento completado exitosamente",
            exact_match=metrics["exact_match"],
            accuracy=metrics["accuracy"],
            f1_macro=metrics["f1_macro"],
            n_train=metrics["n_train"],
            n_test=metrics["n_test"],
            training_time_s=metrics["training_time_s"],
            upload_warning=None,
        )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        inputs = [
            InputField(name=col, type="float", description=f"Sensor {col} time-series")
            for col in SENSOR_COLUMNS
        ]
        outputs = []
        for comp in COMPONENT_NAMES:
            outputs.append(
                OutputField(name=comp, type="int", description="0=SANO, 1=WARNING, 2=CRÍTICO")
            )
        for comp in ["Fouling", "Valvula", "Bomba", "Acumulador"]:
            outputs.append(
                OutputField(name=f"Confianza_{comp}", type="float", description="Confidence 0-1")
            )

        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Clasificación multi-etiqueta de 4 componentes de maquinaria de pasteurizado "
                "(Enfriador, Válvula, Bomba, Acumulador) mediante Deep Neurosymbolic Learning "
                "(1D-CNN + Physics-Informed Loss)."
            ),
            task_type="multi_label_classification",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics={
                "exact_match": 0.9879,
                "accuracy": 0.9970,
                "precision_macro": 0.9969,
                "recall_macro": 0.9972,
                "f1_macro": 0.9970,
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
            except Exception as exc:
                logger.warning("MLflow stats fetch failed for run_id=%s: %s", mlflow_run_id, exc)
        return base
