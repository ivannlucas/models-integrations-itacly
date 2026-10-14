from __future__ import annotations

import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError, ModelPersistenceError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import ARTIFACTS_ROOT, local_file_path
from app.plugins.ml48_dairy_dnsl_critical_point_detection import xai
from app.plugins.ml48_dairy_dnsl_critical_point_detection.constants import (
    APPLY_DIGITAL_TWIN,
    ARTIFACT_FOLDER_NAME,
    COMPONENT_NAMES,
    FRAMEWORK,
    HEAD_NAMES,
    MODEL_ID,
    REPORTED_METRICS,
    SENSOR_COLUMNS,
    VERSION,
)
from app.plugins.ml48_dairy_dnsl_critical_point_detection.mlflow_utils import (
    download_user_model_from_mlflow,
    upload_artifacts_to_mlflow,
)
from app.plugins.ml48_dairy_dnsl_critical_point_detection.model_loader import (
    load_artifacts_from_dir,
    load_shap_background,
)
from app.plugins.ml48_dairy_dnsl_critical_point_detection.postprocessing import (
    format_batch_row,
    format_inline_response,
    run_inference,
)
from app.plugins.ml48_dairy_dnsl_critical_point_detection.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml48_dairy_dnsl_critical_point_detection.preprocessing import (
    build_dataframe_from_csv,
    build_dataframe_from_sensors,
)
from app.plugins.ml48_dairy_dnsl_critical_point_detection.train_dto import TrainResponse
from app.plugins.ml48_dairy_dnsl_critical_point_detection.trainer import (
    fine_tune,
    save_training_artifacts,
    train_full,
)

logger = logging.getLogger(__name__)

_ARTIFACT_DIR = ARTIFACTS_ROOT / ARTIFACT_FOLDER_NAME


class Ml48DairyDnslCriticalPointDetectionPlugin(ModelPluginPort):
    def __init__(self) -> None:
        self._model = None
        self._scaler = None
        self._feature_columns: list[str] = []
        self._ts1_mean_train: float = 45.0
        self._shap_background: np.ndarray | None = None
        self._device: torch.device | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        bucket = os.environ.get("STORAGE_BUCKET")
        if bucket:
            from app.infrastructure.artifact_store import ArtifactStore

            ArtifactStore(ARTIFACT_FOLDER_NAME).download_all_if_needed()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, scaler, feature_cols, ts1_mean_train = load_artifacts_from_dir(_ARTIFACT_DIR)
        self._model = model.to(self._device)
        self._scaler = scaler
        self._feature_columns = feature_cols
        self._ts1_mean_train = ts1_mean_train
        self._shap_background = load_shap_background(_ARTIFACT_DIR)
        logger.info("ml48 plugin loaded: %s (device=%s)", MODEL_ID, self._device)

    def is_loaded(self) -> bool:
        return self._model is not None

    def _require_loaded(self) -> None:
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _context(self, mlflow_run_id: str) -> dict:
        """Contexto de inferencia: modelo estándar o modelo del usuario (MLflow). temp_dir != None => limpiar."""
        if mlflow_run_id:
            logger.info("Using user-trained model from MLflow run_id=%s", mlflow_run_id)
            # Never falls back to the base model: a run without a loadable model raises (422).
            model, scaler, feature_cols, ts1_mean_train, shap_bg, temp_dir = download_user_model_from_mlflow(
                mlflow_run_id,
            )
            return {
                "model": model.to(self._device), "scaler": scaler, "feature_cols": feature_cols,
                "ts1_mean_train": ts1_mean_train, "shap_background": shap_bg, "temp_dir": temp_dir,
            }
        self._require_loaded()
        return {
            "model": self._model, "scaler": self._scaler, "feature_cols": self._feature_columns,
            "ts1_mean_train": self._ts1_mean_train, "shap_background": self._shap_background, "temp_dir": None,
        }

    @staticmethod
    def _digital_twin(flag: bool | None) -> bool:
        return APPLY_DIGITAL_TWIN if flag is None else flag

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
        ctx = self._context(mlflow_run_id)
        try:
            use_dt = self._digital_twin(features.get("apply_digital_twin"))
            cycle_id = features.get("Cycle_ID")

            if data_path:
                with local_file_path(data_path) as local_path:
                    x_df, cycle_ids = build_dataframe_from_csv(local_path, ctx["ts1_mean_train"], use_dt)
                if cycle_ids is not None and cycle_ids.nunique() > 1:
                    first = cycle_ids.unique()[0]
                    logger.warning(
                        "predict_inline received a CSV with %d distinct Cycle_ID values; only Cycle_ID=%s "
                        "is used. Use predict_batch to get a prediction per cycle.",
                        cycle_ids.nunique(), first,
                    )
                    label = f"Cycle_ID={first} (1 of {cycle_ids.nunique()} in file)"
                    cycle_id = int(first)
                    x_df = x_df[cycle_ids == first].drop(columns=["Cycle_ID"], errors="ignore")
                elif cycle_ids is not None:
                    label = "single-cycle-csv"
                    cycle_id = int(cycle_ids.iloc[0])
                    x_df = x_df.drop(columns=["Cycle_ID"], errors="ignore")
                else:
                    label = "inline-csv"
            else:
                sensor_data = {col: features[col] for col in SENSOR_COLUMNS}
                x_df = build_dataframe_from_sensors(
                    sensor_data, features.get("Time_Segundos"), cycle_id, ctx["ts1_mean_train"], use_dt,
                )
                label = f"Cycle_ID={cycle_id}"

            resultados = run_inference(ctx["model"], ctx["scaler"], ctx["feature_cols"], x_df, self._device, label=label)
            if resultados is None:
                raise ValueError("Feature columns mismatch during inference")

            xai_result, xai_error = None, None
            if features.get("include_xai", True):
                xai_result, xai_error = xai.explain_cycle(
                    ctx["model"], resultados["input_tensor"], resultados["predicciones"], resultados["confianzas"],
                    ctx["feature_cols"], cycle_id=cycle_id,
                    include_shap=bool(features.get("include_shap", False)),
                    shap_background=ctx["shap_background"],
                    include_cam=bool(features.get("include_cam", False)),
                )
            self._record()
            return PredictInlineResponse(
                **format_inline_response(MODEL_ID, resultados), xai_result=xai_result, xai_error=xai_error,
            )
        finally:
            if ctx["temp_dir"]:
                shutil.rmtree(ctx["temp_dir"], ignore_errors=True)

    def predict_batch(
        self,
        *,
        data_path: str,
        mlflow_run_id: str = "",
        apply_digital_twin: bool | None = None,
        include_xai: bool = True,
        include_shap: bool = False,
        n_samples: int = 50,
    ) -> PredictBatchResponse:
        ctx = self._context(mlflow_run_id)
        try:
            use_dt = self._digital_twin(apply_digital_twin)
            with local_file_path(data_path) as local_path:
                x_df, cycle_ids = build_dataframe_from_csv(local_path, ctx["ts1_mean_train"], use_dt)

            if cycle_ids is not None and cycle_ids.nunique() > 1:
                groups = [(cid, x_df[cycle_ids == cid].drop(columns=["Cycle_ID"], errors="ignore"))
                          for cid in cycle_ids.unique()]
            else:
                cid = cycle_ids.iloc[0] if cycle_ids is not None and len(cycle_ids) else None
                groups = [(cid, x_df.drop(columns=["Cycle_ID"], errors="ignore"))]

            predictions = []
            shap_budget = max(int(n_samples), 0) if include_shap else 0
            for cid, cycle_df in groups:
                if cycle_df.empty:
                    continue
                label = f"Cycle_ID={cid}" if cid is not None else "single-cycle"
                result = run_inference(ctx["model"], ctx["scaler"], ctx["feature_cols"], cycle_df, self._device, label=label)
                if not result:
                    continue
                row = format_batch_row(result, cid)
                # XAI por ciclo (contrato de 43/44): xai_result / xai_error en cada fila. Un fallo de XAI de un ciclo
                # se queda en su fila y no afecta a la prediccion ni a la XAI del resto del lote.
                xai_result, xai_error = None, None
                if include_xai:
                    with_shap = shap_budget > 0
                    xai_result, xai_error = xai.explain_cycle(
                        ctx["model"], result["input_tensor"], result["predicciones"], result["confianzas"],
                        ctx["feature_cols"], cycle_id=cid, include_shap=with_shap,
                        shap_background=ctx["shap_background"],
                    )
                    if with_shap:
                        shap_budget -= 1
                row["xai_result"] = xai_result
                row["xai_error"] = xai_error
                predictions.append(row)

            self._record()
            return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)
        finally:
            if ctx["temp_dir"]:
                shutil.rmtree(ctx["temp_dir"], ignore_errors=True)

    def train(self, *, data_path: str, mlflow_run_id: str, mode: str = "fine_tune") -> TrainResponse:
        """Retrain with one of the AI team's two procedures (see trainer.py), like ml47.

        fine_tune (default — the platform sends a few hundred plant cycles at most) recalibrates a
        copy of the served model; full trains a new CNN from scratch. Either way the result, with its
        own SHAP background, lives only in its MLflow run — the served model is never touched.
        """
        logger.info("ml48 train mode=%s data_path=%s mlflow_run_id=%s", mode, data_path, mlflow_run_id)
        with local_file_path(data_path) as local_path:
            if mode == "full":
                result = train_full(local_path)
            else:
                self._require_loaded()
                result = fine_tune(
                    local_path, base_model=self._model, scaler=self._scaler,
                    feature_cols=self._feature_columns, ts1_mean_train=self._ts1_mean_train,
                )
        metrics = result.metrics

        temp_dir = Path(tempfile.mkdtemp(prefix="ml48_train_"))
        try:
            save_training_artifacts(temp_dir, result)
            try:
                new_run_id = upload_artifacts_to_mlflow(
                    str(temp_dir), mlflow_run_id=mlflow_run_id, metrics={**metrics, "mode_full": float(mode == "full")},
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("MLflow artifact upload failed: %s", exc)
                raise ModelPersistenceError(f"El modelo reentrenado no se ha podido guardar en MLflow: {exc}") from exc
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info("ml48 training complete (mode=%s). MLflow run_id=%s", mode, new_run_id)
        detail = (
            "Reentrenamiento completo desde cero (procedimiento train del equipo de IA)."
            if mode == "full" else
            "Calibración a planta de las 4 cabezas del modelo servido (procedimiento fine_tune del equipo de IA)."
        )
        return TrainResponse(
            detail=detail,
            mode=mode,
            exact_match=metrics["exact_match"],
            accuracy=metrics["accuracy"],
            precision_macro=metrics["precision_macro"],
            recall_macro=metrics["recall_macro"],
            f1_macro=metrics["f1_macro"],
            n_train=metrics["n_train"],
            n_val=metrics["n_val"],
            n_test=metrics["n_test"],
            epochs_run=metrics["epochs_run"],
            training_time_s=metrics["training_time_s"],
            mlflow_run_id=new_run_id or mlflow_run_id,
            upload_warning=None,
        )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        inputs = [InputField(name=col, type="float", description=f"Sensor {col} time-series") for col in SENSOR_COLUMNS]
        outputs = [OutputField(name=c, type="int", description="0=SANO, 1=WARNING, 2=CRÍTICO") for c in COMPONENT_NAMES]
        outputs += [OutputField(name=f"Confianza_{c}", type="float", description="Confidence 0-1")
                    for c in ["Fouling", "Valvula", "Bomba", "Acumulador"]]
        outputs.append(OutputField(
            name="xai_result", type="object",
            description=(
                "XAI del ciclo (contrato de modelo-43): Estado_del_sistema, Bloques_principales, Variables_clave, "
                "Acciones_sugeridas / Mensaje_operativo y Detalle_componentes (ventana crítica Grad-CAM, riesgo, acción)."
            ),
        ))
        outputs.append(OutputField(
            name="xai_error", type="string", description="Motivo si la XAI de este ciclo falló; null si fue correcta.",
        ))

        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "XAI para detección de puntos críticos de control sobre el DNSL (1D-CNN + Physics-Informed Loss) "
                f"de 4 componentes de pasteurizado ({', '.join(HEAD_NAMES)}): Grad-CAM temporal, SHAP opcional, "
                "CCP y recomendaciones prescriptivas."
            ),
            task_type="multi_label_classification_xai",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics=dict(REPORTED_METRICS),
            runtime_stats=RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=None),
        )
        if mlflow_run_id:
            try:
                tracker = BaseMLflowTracker(mlflow_run_id)
                base.metrics["mlflow"] = {"params": tracker.get_params(), "metrics": tracker.get_metrics()}
            except Exception as exc:
                logger.warning("MLflow stats fetch failed for run_id=%s: %s", mlflow_run_id, exc)
        return base
