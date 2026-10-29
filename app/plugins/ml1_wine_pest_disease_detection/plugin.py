"""Ml1WinePestDiseaseDetectionPlugin — VID-Deep_Ensemble vineyard pest/disease image detection.

Hybrid pipeline (faithful port of the delivered src/predict/predictor.py, production config:
Desacoplado + Sin Gases): YOLOv8 crops the leaf/cluster, a soft-voted Deep Ensemble of 3 CNN
backbones (MobileNetV3Small, EfficientNetB0, DenseNet121) classifies one of 15
pests/pathologies, and a decoupled tabular MLP head (trained jointly with each backbone)
estimates a 0-1 epidemiological risk index from 9 IIoT sensor readings. Without an image, the
service falls back to a RandomForest regressor for risk only (the RandomForest classifier is
NOT used in this fallback — see inbox/a01/manifest.yaml known_issues for why: its own audited
accuracy is 11.6% over 15 classes, and the memoria's claim that it IS used here does not match
the delivered code — so it is not even loaded).

Only the 2 RandomForest tabular experts are retrainable via train() — see training.py and
manifest.yaml training.scope_note for why the CNN Deep Ensemble is out of scope.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import (
    DataContractError,
    InvalidImageError,
    ModelNotLoadedError,
    ModelPersistenceError,
)
from app.infrastructure.archive import safe_extract_zip
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml1_wine_pest_disease_detection import inference, model_loader, preprocessing, training
from app.plugins.ml1_wine_pest_disease_detection.constants import (
    FRAMEWORK,
    IGNORED_GAS_COLS,
    METRICS_REPORTED,
    MODEL_ID,
    SENSOR_COLS,
    VERSION,
)
from app.plugins.ml1_wine_pest_disease_detection.mlflow_utils import (
    download_user_model_from_mlflow,
    upload_artifacts_to_mlflow,
)
from app.plugins.ml1_wine_pest_disease_detection.predict_dto import (
    BatchPrediction,
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml1_wine_pest_disease_detection.train_dto import TrainResponse

logger = logging.getLogger(__name__)


class Ml1WinePestDiseaseDetectionPlugin(ModelPluginPort):
    """YOLOv8 + 3-backbone Deep Ensemble + RandomForest fallback for vineyard pest/disease diagnosis."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._bundle: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load YOLO, the 3-backbone Deep Ensemble and the RandomForest risk regressor."""
        self._bundle = model_loader.load_model_bundle()
        logger.info("ml1 plugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        """Return True once the full bundle is loaded."""
        return self._bundle is not None

    def _require_loaded(self) -> dict:
        if self._bundle is None:
            raise ModelNotLoadedError("El modelo no está cargado.")
        return self._bundle

    def _record_prediction(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _resolve_rf_riesgo(self, mlflow_run_id: str) -> tuple[dict, str | None]:
        """Return ({"rf_riesgo"}, temp_dir) — the base regressor from self._bundle, or the
        user-retrained one from MLflow. Only the RF varies; YOLO and the CNN ensemble always
        come from the fixed base bundle (see training.scope_note). Never assigns to
        self._bundle — a concurrent request must never see another user's retrained RF."""
        bundle = self._require_loaded()
        if not mlflow_run_id:
            return {"rf_riesgo": bundle["rf_riesgo"]}, None
        return download_user_model_from_mlflow(mlflow_run_id)

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Diagnose one sample: image (optional) + 9 sensor readings."""
        _ = model_key, threshold
        bundle = self._require_loaded()
        rf_override, user_tmp = self._resolve_rf_riesgo(mlflow_run_id)
        try:
            sensors_aligned = preprocessing.validate_and_align_sensors(features.get("sensors") or {})
            image_base64 = features.get("image_base64")

            if image_base64:
                img_bgr = preprocessing.decode_image_base64(image_base64)
                img_rgb = preprocessing.yolo_crop_alignment(
                    img_bgr, bundle["yolo_model"], bundle["yolo_device"],
                )
                result = inference.run_inference(bundle, img_rgb, sensors_aligned)
            else:
                result = inference.run_inference_tabular_only(
                    {**bundle, **rf_override}, sensors_aligned,
                )

            self._record_prediction()
            logger.info(
                "predict_inline done — metodo=%s count=%d mlflow=%s",
                result["metodo"], self._predict_count, bool(mlflow_run_id),
            )
            return PredictInlineResponse(model_id=MODEL_ID, **result)
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Diagnose every row of a ZIP's CSV (Path_Imagen + 9 sensor columns).

        A row's image missing/unreadable/undetectable falls back to the tabular-only path for
        that row alone (matching src/main.py::predict's link_image_path contingency logic) —
        one bad photo must never abort the batch nor the other rows.
        """
        bundle = self._require_loaded()
        rf_override, user_tmp = self._resolve_rf_riesgo(mlflow_run_id)
        extract_dir = tempfile.mkdtemp(prefix="ml1_batch_")
        try:
            with local_file_path(data_path) as local_path:
                try:
                    safe_extract_zip(local_path, extract_dir)
                except ValueError as exc:
                    raise DataContractError(str(exc)) from exc

            root = Path(extract_dir)
            csv_candidates = sorted(root.rglob("*.csv"))
            if not csv_candidates:
                raise DataContractError("El ZIP no contiene ningún CSV de entrada.")
            df = pd.read_csv(csv_candidates[0])
            if "Path_Imagen" not in df.columns:
                raise DataContractError("El CSV debe traer una columna 'Path_Imagen' (N/A si no hay foto).")

            nested_images = root / "images"
            search_dirs = [nested_images, root] if nested_images.is_dir() else [root]

            predictions: list[BatchPrediction] = []
            for idx, row in df.iterrows():
                raw_name = row.get("Path_Imagen", "N/A")
                nombre_imagen = str(raw_name) if pd.notna(raw_name) else "N/A"
                try:
                    # Like predictor.py:317, only the sensor columns are read — any other column
                    # (an ID, the label, a cached path...) is ignored, not a contract error.
                    sensors = {col: row[col] for col in (*SENSOR_COLS, *IGNORED_GAS_COLS) if col in row.index}
                    sensors_aligned = preprocessing.validate_and_align_sensors(sensors)
                    image_path = preprocessing.resolve_image_path_for_batch(raw_name, search_dirs)

                    if image_path is not None:
                        img_bgr = _read_image_bgr(image_path)
                        img_rgb = preprocessing.yolo_crop_alignment(
                            img_bgr, bundle["yolo_model"], bundle["yolo_device"],
                        )
                        result = inference.run_inference(bundle, img_rgb, sensors_aligned)
                    else:
                        result = inference.run_inference_tabular_only(
                            {**bundle, **rf_override}, sensors_aligned,
                        )
                    predictions.append(BatchPrediction(id=int(idx), nombre_imagen=nombre_imagen, **result))
                except (DataContractError, InvalidImageError) as exc:
                    predictions.append(
                        BatchPrediction(
                            id=int(idx), nombre_imagen=nombre_imagen,
                            enfermedad="", confianza=0.0, riesgo=0.0, metodo="",
                            error=str(exc),
                        )
                    )

            n_predictions = sum(1 for p in predictions if p.error is None)
            self._record_prediction()
            logger.info(
                "predict_batch done — %d filas, %d predicciones, mlflow=%s",
                len(predictions), n_predictions, bool(mlflow_run_id),
            )
            return PredictBatchResponse(
                model_id=MODEL_ID, predictions=predictions,
                n_rows=len(predictions), n_predictions=n_predictions, output_path=None,
            )
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)

    # ── train (solo los 2 RandomForest tabulares — ver manifest training.scope_note) ──────────

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Retrain the RandomForest classifier/regressor pair on a client-supplied sensor CSV.

        Both are fit (the classifier only to report the accuracy/F1 metrics of the manifest
        contract), but only the regressor is uploaded: it is the only RF predict ever uses.

        Fresh sklearn estimators are fit from scratch (never mutating the served ones); the
        result is uploaded ONLY to the given MLflow run. If the upload fails, the whole
        retraining result is lost by design (nothing else persists it) — so this raises
        ModelPersistenceError (-> 502) rather than returning 200 with a misleading warning.
        """
        self._require_loaded()
        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)
        result = training.train_tabular_experts(df)

        mlflow_tmp = tempfile.mkdtemp(prefix="ml1_mlflow_")
        try:
            joblib.dump(result.rf_riesgo, f"{mlflow_tmp}/rf_riesgo_expert.joblib")
            upload_artifacts_to_mlflow(
                mlflow_tmp, mlflow_run_id,
                metrics={
                    "accuracy": result.accuracy, "f1_macro": result.f1_macro,
                    "f1_weighted": result.f1_weighted, "mae_riesgo": result.mae_riesgo,
                    "n_train": result.n_train, "n_test": result.n_test,
                },
            )
        except Exception as exc:
            logger.error("ml1: fallo al subir el modelo reentrenado a MLflow: %s", exc)
            raise ModelPersistenceError(
                f"El modelo reentrenado no se ha podido guardar en MLflow: {exc}"
            ) from exc
        finally:
            shutil.rmtree(mlflow_tmp, ignore_errors=True)

        logger.info(
            "ml1 train() done — accuracy=%.4f f1_macro=%.4f f1_weighted=%.4f mae_riesgo=%.4f n_train=%d n_test=%d",
            result.accuracy, result.f1_macro, result.f1_weighted, result.mae_riesgo,
            result.n_train, result.n_test,
        )
        return TrainResponse(
            detail="Expertos RandomForest (clasificación + riesgo) reentrenados desde cero con el procedimiento del equipo de IA",
            accuracy=round(result.accuracy, 4),
            f1_macro=round(result.f1_macro, 4),
            f1_weighted=round(result.f1_weighted, 4),
            mae_riesgo=round(result.mae_riesgo, 4),
            n_train=result.n_train,
            n_test=result.n_test,
            upload_warning=None,
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and the real audited metrics."""
        metrics = dict(METRICS_REPORTED)
        if mlflow_run_id:
            from app.domain.services.mlflow_tracker import BaseMLflowTracker  # pylint: disable=import-outside-toplevel
            tracker = BaseMLflowTracker(mlflow_run_id)
            metrics["mlflow"] = {"params": tracker.get_params(), "metrics": tracker.get_metrics()}

        inputs = [
            InputField(name="image_base64", type="str | null",
                       description="Imagen de hoja/racimo en base64, opcional (sin ella: modo de contingencia tabular)."),
            InputField(name="sensors", type="dict[str, float]",
                       description="9 sensores IIoT de los últimos 5 días (producción, exclude_gases=true)."),
        ]
        outputs = [
            OutputField(name="enfermedad", type="str", description="Una de 15 clases, o IMAGEN_NO_DISPONIBLE sin foto."),
            OutputField(name="confianza", type="float", description="Probabilidad softmax [0,1]; 0.0 sin imagen."),
            OutputField(name="riesgo", type="float", description="Índice de riesgo epidemiológico [0,1]."),
            OutputField(name="metodo", type="str", description="HIBRIDO_ENSAMBLE | RIESGO_TABULAR_EXPERT."),
        ]
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Diagnóstico fitopatológico multimodal de la vid (VID-Deep_Ensemble): detección "
                "YOLOv8 de hoja/racimo + Deep Ensemble (MobileNetV3/EfficientNetB0/DenseNet121) "
                "para 15 plagas/patologías, con estimación de riesgo epidemiológico 0-1 a partir "
                "de 9 sensores IIoT. Ver inbox/a01/manifest.yaml para el contrato completo y las "
                "discrepancias memoria-vs-código documentadas."
            ),
            task_type="classification_regression_multimodal",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics=metrics,
            runtime_stats=RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=None),
        )


def _read_image_bgr(path: Path):
    """cv2.imread wrapper raising InvalidImageError instead of silently returning None."""
    import cv2  # pylint: disable=import-outside-toplevel

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise InvalidImageError(f"No se pudo leer la imagen en: {path}")
    return img_bgr
