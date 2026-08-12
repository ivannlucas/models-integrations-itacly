"""Ml3WineDiseasePestForecastPlugin — Deep Ensemble (LSTM+CNN+BiGRU) vineyard diagnosis.

Predicts vine diseases/pests (11 classes) and infection severity (0-1) from a 168-hour
window of ambient + soil + air sensors per series, using the delivered Deep Ensemble
Learning architecture with Soft Voting and an editable treatment knowledge base. See
inbox/a03/manifest.yaml for the full contract, golden cases and known issues (e.g. the
delivered model pads <168-row windows, so at least 168 rows are required for a
non-degraded prediction).
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml3_wine_disease_pest_forecast import (
    model_loader,
    postprocessing,
    preprocessing,
    training,
)
from app.plugins.ml3_wine_disease_pest_forecast.constants import (
    BATCH_SIZE,
    EPOCHS,
    FRAMEWORK,
    LABEL_ENCODER_FILENAME,
    MODEL_FILENAMES,
    MODEL_ID,
    RAW_FIXED_COLUMNS,
    REPORTED_METRICS,
    SCALER_FILENAME,
    SERIES_COLUMN,
    TRAIN_RANDOM_SEED,
    USER_LABEL_ENCODER_FILENAME,
    USER_MODEL_FILENAMES,
    USER_SCALER_FILENAME,
    VERSION,
    WINDOW_SIZE,
)
from app.plugins.ml3_wine_disease_pest_forecast.mlflow_utils import download_user_model_from_mlflow
from app.plugins.ml3_wine_disease_pest_forecast.model_loader import _store
from app.plugins.ml3_wine_disease_pest_forecast.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml3_wine_disease_pest_forecast.train_dto import TrainResponse

logger = logging.getLogger(__name__)


class Ml3WineDiseasePestForecastPlugin(ModelPluginPort):
    """Deep Ensemble vineyard diagnosis over 168-hour per-series sensor windows."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._bundle: tuple[list, Any, Any, list[str]] | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        """Load the fixed ensemble (3 Keras checkpoints + scaler + label_encoder)."""
        self._bundle = model_loader.load_artifacts()
        logger.info("Ml3 plugin loaded: %s (window=%d)", MODEL_ID, WINDOW_SIZE)

    def is_loaded(self) -> bool:
        """Return True once the fixed bundle is loaded."""
        return self._bundle is not None

    def _require_loaded(self) -> None:
        if self._bundle is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record_prediction(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    # ── shared inference core ─────────────────────────────────────────────────

    def _resolve_user_bundle(
        self, mlflow_run_id: str,
    ) -> tuple[tuple[list, Any, Any, list[str]], str] | None:
        """Download a user-retrained ensemble from MLflow, or None to use the fixed one."""
        if not mlflow_run_id:
            return None
        logger.info("Using user-retrained model from MLflow run_id=%s", mlflow_run_id)
        loaded = download_user_model_from_mlflow(mlflow_run_id)
        if loaded is None:
            return None
        models, scaler, le, class_names, tmp = loaded
        return (models, scaler, le, class_names), tmp

    def _bundle_for(
        self, mlflow_run_id: str,
    ) -> tuple[tuple[list, Any, Any, list[str]], str | None]:
        """Resolve the bundle to use (user-retrained or fixed) plus a tempdir to clean up."""
        user_tmp = None
        bundle = self._bundle
        if mlflow_run_id:
            resolved = self._resolve_user_bundle(mlflow_run_id)
            if resolved is not None:
                bundle, user_tmp = resolved
        return bundle, user_tmp

    def _score_window(
        self, series_name: Any, series_df: pd.DataFrame, bundle: tuple[list, Any, Any, list[str]],
    ) -> dict[str, Any]:
        """Build the 168-row tensor, run the ensemble and serialize the prediction record."""
        models, scaler, le, class_names = bundle
        x_tensor, last_fecha, window_df = preprocessing.build_window_tensor(series_df, scaler)
        pred = postprocessing.predict_ensemble(models, x_tensor, le, class_names)
        record: dict[str, Any] = {
            "id_serie": series_name,
            "fecha_evaluacion": None if last_fecha is None else str(last_fecha),
            "diagnostico_ia": pred["diagnostico_ia"],
            "confianza_clasificacion": round(pred["confianza_clasificacion"], 6),
            "grado_severidad": round(pred["grado_severidad"], 6),
            "tratamiento_recomendado": postprocessing.build_treatment(pred["diagnostico_ia"]),
            "probabilidades_clases": pred["probabilidades_clases"],
            "xai_feature_values": postprocessing.raw_snapshot(window_df),
        }
        return record

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Score every series in a CSV/parquet (one prediction per 168-hour window)."""
        self._require_loaded()
        bundle, user_tmp = self._bundle_for(mlflow_run_id)
        try:
            with local_file_path(data_path) as local_path:
                if str(local_path).lower().endswith(".parquet"):
                    raw_df = pd.read_parquet(local_path)
                else:
                    raw_df = pd.read_csv(local_path)
            preprocessing.validate_raw_columns(raw_df)

            predictions = []
            for series_name, series_df in preprocessing.prepare_series_groups(raw_df):
                predictions.append(self._score_window(series_name, series_df, bundle))

            self._record_prediction()
            logger.info(
                "predict_batch done — n_series=%d mlflow=%s count=%d",
                len(predictions), bool(mlflow_run_id), self._predict_count,
            )
            return PredictBatchResponse(
                model_id=MODEL_ID, predictions=predictions, output_path=None,
            )
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Score a single series submitted as a list of hourly row dicts (>= 168 rows)."""
        self._require_loaded()
        bundle, user_tmp = self._bundle_for(mlflow_run_id)
        try:
            rows: list[dict] = features["rows"]
            raw_df = preprocessing.build_raw_dataframe(rows)
            preprocessing.validate_raw_columns(raw_df)

            if len(raw_df) < WINDOW_SIZE:
                logger.warning(
                    "predict_inline recibió %d filas (< %d): el modelo entregado hace padding "
                    "repitiendo la primera fila (predicción degradada, ver manifest known_issues).",
                    len(raw_df), WINDOW_SIZE,
                )

            series_name = "Única"
            if SERIES_COLUMN in raw_df.columns:
                series_name = raw_df[SERIES_COLUMN].iloc[0]
            record = self._score_window(series_name, raw_df, bundle)

            self._record_prediction()
            logger.info(
                "predict_inline done — id_serie=%s prediction=%s conf=%.4f count=%d",
                record["id_serie"], record["diagnostico_ia"], record["confianza_clasificacion"],
                self._predict_count,
            )
            return PredictInlineResponse(
                model_id=MODEL_ID,
                id_serie=record["id_serie"],
                fecha_evaluacion=record["fecha_evaluacion"],
                diagnostico_ia=record["diagnostico_ia"],
                confianza_clasificacion=record["confianza_clasificacion"],
                grado_severidad=record["grado_severidad"],
                tratamiento_recomendado=record["tratamiento_recomendado"],
                probabilidades_clases=record["probabilidades_clases"],
                model_name=MODEL_ID,
                xai_feature_values=record["xai_feature_values"],
            )
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)

    # ── train (full retraining with the delivered procedure) ─────────────────

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> TrainResponse:
        """Retrain the full Deep Ensemble from a labeled raw CSV with the delivered procedure.

        Trains into fresh objects — the served fixed S3 artifacts are never mutated. User
        artifacts are saved locally under user_* names and, when mlflow_run_id is given,
        uploaded to MLflow under artifact_path="model" with the canonical filenames so that
        mlflow_utils can rebuild the bundle.
        """
        self._require_loaded()
        raw_df = training.load_retraining_input(data_path)

        _store.local_dir.mkdir(parents=True, exist_ok=True)
        model_paths = {
            "M1_LSTM": str(_store.local_dir / USER_MODEL_FILENAMES[0]),
            "M2_CNN": str(_store.local_dir / USER_MODEL_FILENAMES[1]),
            "M3_BiGRU": str(_store.local_dir / USER_MODEL_FILENAMES[2]),
            "scaler": str(_store.local_dir / USER_SCALER_FILENAME),
            "label_encoder": str(_store.local_dir / USER_LABEL_ENCODER_FILENAME),
        }
        result = training.run_retraining(raw_df, model_paths)
        metrics = result["metrics"]

        upload_warning = None
        if mlflow_run_id:
            tracker = BaseMLflowTracker(mlflow_run_id)
            try:
                tracker.log_params({"window_size": WINDOW_SIZE, "epochs": EPOCHS,
                                    "batch_size": BATCH_SIZE, "seed": TRAIN_RANDOM_SEED})
                tracker.log_metrics(metrics)
                mlflow_tmp = tempfile.mkdtemp(prefix="ml3_mlflow_")
                try:
                    for src, dst in zip(USER_MODEL_FILENAMES, MODEL_FILENAMES):
                        shutil.copy2(_store.local_dir / src, f"{mlflow_tmp}/{dst}")
                    shutil.copy2(
                        _store.local_dir / USER_SCALER_FILENAME, f"{mlflow_tmp}/{SCALER_FILENAME}",
                    )
                    shutil.copy2(
                        _store.local_dir / USER_LABEL_ENCODER_FILENAME,
                        f"{mlflow_tmp}/{LABEL_ENCODER_FILENAME}",
                    )
                    tracker.upload_artifacts(mlflow_tmp, artifact_path="model")
                finally:
                    shutil.rmtree(mlflow_tmp, ignore_errors=True)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("MLflow artifact upload failed: %s", exc)
                upload_warning = f"Modelo guardado localmente, pero falló la subida a MLflow: {exc}"

        logger.info(
            "ml3 train() done — n_train_windows=%d n_val_windows=%d n_test_windows=%d "
            "f1_macro=%.4f mlflow=%s",
            result["n_windows_train"], result["n_windows_val"], result["n_windows_test"],
            metrics["f1_macro"], bool(mlflow_run_id),
        )
        return TrainResponse(
            detail=(
                "Reentrenamiento completo del Deep Ensemble (LSTM + CNN + BiGRU) desde cero "
                "con el procedimiento original de train.py."
            ),
            n_windows_train=result["n_windows_train"],
            n_windows_val=result["n_windows_val"],
            n_windows_test=result["n_windows_test"],
            epochs_executed=max(result["epochs_executed"]),
            accuracy=metrics["accuracy"],
            precision_macro=metrics["precision_macro"],
            recall_macro=metrics["recall_macro"],
            f1_macro=metrics["f1_macro"],
            f1_weighted=metrics["f1_weighted"],
            mae=metrics["mae"],
            mse=metrics["mse"],
            r2=metrics["r2"],
            upload_warning=upload_warning,
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and the reported ensemble metrics."""
        inputs = [
            InputField(
                name="Fecha",
                type="datetime",
                description="Marca horaria de cada registro (necesaria para Hora_Sin/Hora_Cos).",
            ),
            InputField(
                name="ID_Serie",
                type="int",
                description=(
                    "Identificador de la serie temporal (opcional en inline con una serie)."
                ),
            ),
        ]
        inputs += [
            InputField(
                name=col,
                type="float",
                description="Sensor crudo por hora del contrato de entrada (ventana mínima 168 h).",
            )
            for col in RAW_FIXED_COLUMNS
        ]
        outputs = [
            OutputField(name="id_serie", type="int", description="Serie evaluada"),
            OutputField(name="fecha_evaluacion", type="datetime",
                        description="Última hora de la ventana de 168 h predicha"),
            OutputField(name="diagnostico_ia", type="str",
                        description="Clase vencedora (11 clases) tras Soft Voting del ensemble"),
            OutputField(name="confianza_clasificacion", type="float",
                        description="Probabilidad media (0-1) de la clase vencedora"),
            OutputField(name="grado_severidad", type="float",
                        description="Severidad media (0-1) de las 3 regresiones sigmoid"),
            OutputField(name="tratamiento_recomendado", type="str",
                        description=(
                            "Protocolo de la base de conocimiento (editable) para la clase"
                        )),
            OutputField(name="probabilidades_clases", type="dict",
                        description="Probabilidad media por clase"),
        ]
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Deep Ensemble Learning (LSTM + CNN + BiGRU) para diagnóstico de enfermedades y "
                "plagas de la vid (11 clases: HEALTHY, LOBESIA, EMPOASCA, ALTICA, RED_MITE, "
                "ERINOSIS, ESCA, OIDIO, MILDIU, BOTRYTIS, BLACK_ROT) y severidad de infección "
                "(0-1), sobre la última ventana de 168 horas por serie. Soft Voting + regresión "
                "media + base de conocimiento de tratamientos. Dataset simulado/vid_simulator — "
                "ver manifest."
            ),
            task_type="multiclass_classification_and_regression_timeseries",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics=REPORTED_METRICS,
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=None,
            ),
        )
