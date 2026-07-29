"""Ml40MeatRefrigerationAerationFaultDiagnosisPlugin — dual RF + neurosymbolic fault diagnosis.

Diagnoses failures in meat-industry refrigeration (13 classes) and sausage-curing aeration
(4 classes) equipment: a RandomForest per subsystem corrected by hard physics rules and a
per-cycle majority vote (DNSL). See inbox/a40/manifest.yaml for the full contract, golden
cases and known issues (synthetic/simulated datasets, batch-dependent NC/CF rule, 171 MB
refrigeration artifact).
"""
from __future__ import annotations

import logging
import math
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
import yaml

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError, UnknownDiagnosisSystemError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis import (
    model_loader,
    postprocessing,
    preprocessing,
    training,
)
from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.constants import (
    CLASS_MAPPINGS,
    CYCLE_COLUMNS,
    FRAMEWORK,
    METRICS_REPORTED,
    MIN_HISTORY_MINUTES,
    MODEL_FILENAMES,
    MODEL_ID,
    MODEL_PARAMS,
    RAW_INPUT_COLUMNS,
    SCALER_FILENAMES,
    STATS_FILENAMES,
    TARGET_COLUMN,
    THRESHOLDS_FILENAMES,
    USER_MODEL_FILENAMES,
    USER_SCALER_FILENAMES,
    USER_STATS_FILENAMES,
    VERSION,
)
from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.mlflow_utils import (
    download_user_model_from_mlflow,
)
from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.model_loader import _store
from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml40_meat_refrigeration_aeration_fault_diagnosis.train_dto import TrainResponse

logger = logging.getLogger(__name__)


def _clean_scalar(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain JSON-serializable Python types."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
    return value


def _serialize_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe list[dict] records."""
    return [{k: _clean_scalar(v) for k, v in rec.items()} for rec in df.to_dict(orient="records")]


class Ml40MeatRefrigerationAerationFaultDiagnosisPlugin(ModelPluginPort):
    """Dual RandomForest + neurosymbolic rules + per-cycle vote (refrigeracion / aireado)."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._bundles: dict[str, dict] | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        """Load both subsystems' artifacts (models, scaler, thresholds, drift stats)."""
        self._bundles = model_loader.load_artifacts()
        logger.info("Ml40 plugin loaded: %s (systems=%s)", MODEL_ID, sorted(self._bundles))

    def is_loaded(self) -> bool:
        """Return True once both subsystem bundles are loaded."""
        return self._bundles is not None

    def _require_loaded(self) -> None:
        if self._bundles is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record_prediction(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    # ── shared inference core ─────────────────────────────────────────────────

    def _diagnose(self, df: pd.DataFrame, system: str | None, bundle_override: tuple[str, dict] | None) -> tuple[pd.DataFrame, str]:
        """Run the full pipeline (features → RF → rules → vote → collapse) on a frame.

        Returns (per-run diagnosis frame, system).
        """
        engineered, system = preprocessing.prepare_input(df, system)
        if bundle_override is not None:
            user_system, bundle = bundle_override
            if user_system != system:
                raise UnknownDiagnosisSystemError(
                    f"El modelo reentrenado de MLflow es del sistema '{user_system}' pero los "
                    f"datos de entrada corresponden al sistema '{system}'."
                )
        else:
            bundle = self._bundles[system]

        y_ml, y_probs = postprocessing.run_inference(bundle["model"], bundle["scaler"], engineered, system)
        y_ns = postprocessing.apply_neurosymbolic_rules(engineered, y_ml, system, bundle["thresholds"])
        y_final = postprocessing.apply_run_voting(engineered, y_ns)

        result = engineered.copy()
        result["prediction"] = y_final
        result["confidence"] = y_probs.max(axis=1)
        return postprocessing.collapse_by_run(result, system), system

    def _resolve_user_bundle(self, mlflow_run_id: str) -> tuple[tuple[str, dict], str] | None:
        """Download a user-retrained bundle from MLflow, or None to use fixed artifacts.

        Returns ((system, bundle), temp_dir); the caller must rmtree(temp_dir) in a finally.
        """
        if not mlflow_run_id:
            return None
        logger.info("Using user-retrained model from MLflow run_id=%s", mlflow_run_id)
        loaded = download_user_model_from_mlflow(mlflow_run_id)
        if loaded is None:
            return None
        system, bundle, tmp = loaded
        return (system, bundle), tmp

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Diagnose every cycle in a CSV (raw sensor rows or engineered splits-style)."""
        self._require_loaded()
        user_tmp = None
        bundle_override = None
        if mlflow_run_id:
            resolved = self._resolve_user_bundle(mlflow_run_id)
            if resolved is not None:
                bundle_override, user_tmp = resolved
        try:
            with local_file_path(data_path) as local_path:
                raw_df = pd.read_csv(local_path)
            runs_df, system = self._diagnose(raw_df, None, bundle_override)
            avg_conf = float(runs_df["confidence"].mean())
            health = postprocessing.health_status(avg_conf)
            self._record_prediction()
            logger.info(
                "predict_batch done — system=%s n_runs=%d avg_conf=%.4f health=%s mlflow=%s",
                system, len(runs_df), avg_conf, health, bool(mlflow_run_id),
            )
            return PredictBatchResponse(
                model_id=MODEL_ID,
                system=system,
                predictions=_serialize_records(runs_df),
                n_runs=len(runs_df),
                avg_confidence=avg_conf,
                model_health=health,
                output_path=None,
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
        """Diagnose a single cycle submitted as a list of per-minute row dicts."""
        self._require_loaded()
        user_tmp = None
        bundle_override = None
        if mlflow_run_id:
            resolved = self._resolve_user_bundle(mlflow_run_id)
            if resolved is not None:
                bundle_override, user_tmp = resolved
        try:
            rows: list[dict] = features["rows"]
            raw_df = pd.DataFrame(rows)
            if "run_id" not in raw_df.columns:
                raw_df["run_id"] = 0  # single-cycle payload
            if "time_min" not in raw_df.columns:
                raw_df["time_min"] = range(len(raw_df))

            runs_df, system = self._diagnose(raw_df, features.get("system"), bundle_override)
            row = runs_df.iloc[0]
            confidence = float(row["confidence"])
            health = postprocessing.health_status(confidence)
            self._record_prediction()
            logger.info(
                "predict_inline done — system=%s run=%s prediction=%s conf=%.4f count=%d",
                system, row["run_id"], row["prediction_name"], confidence, self._predict_count,
            )
            return PredictInlineResponse(
                model_id=MODEL_ID,
                system=system,
                run_id=int(row["run_id"]),
                prediction=int(row["prediction"]),
                prediction_name=str(row["prediction_name"]),
                confidence=confidence,
                n_rows_used=len(raw_df),
                model_health=health,
                model_name=MODEL_ID,
                xai_feature_values={
                    "prediction": int(row["prediction"]),
                    "confidence": confidence,
                    "n_rows": len(raw_df),
                },
            )
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)

    # ── train (retraining with the original procedure) ───────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> TrainResponse:
        """Retrain one subsystem's RandomForest from a labeled raw CSV.

        Follows the AI team's original trainers exactly (hyperparams, split, weights,
        scaler). Trains into fresh objects — the served fixed artifacts are never mutated
        nor overwritten (user artifacts are saved under user_* filenames locally and, when
        mlflow_run_id is given, uploaded to MLflow under artifact_path="model" with the
        canonical filenames so mlflow_utils can rebuild the bundle).
        """
        self._require_loaded()
        with local_file_path(data_path) as local_path:
            raw_df = pd.read_csv(local_path)
        system = preprocessing.detect_system(raw_df.columns)

        required = CYCLE_COLUMNS + [TARGET_COLUMN] + RAW_INPUT_COLUMNS[system]
        missing = [c for c in required if c not in raw_df.columns]
        if missing:
            raise ValueError(f"CSV de entrenamiento ({system}) — faltan columnas requeridas: {missing}")

        preprocessing.validate_history(raw_df, system)
        engineered = preprocessing.apply_feature_engineering(
            raw_df.sort_values(["run_id", "time_min"]), system
        ).reset_index(drop=True)
        if engineered.empty:
            raise ValueError(
                "Tras la ingeniería de variables no queda ninguna fila de entrenamiento: "
                f"se requieren ciclos de al menos {MIN_HISTORY_MINUTES[system]} minutos."
            )

        thresholds = self._bundles[system]["thresholds"]
        result = training.train_system(engineered, system, thresholds)
        metrics = result["metrics"]

        # Persist locally under user_* names — the fixed S3 artifacts are never overwritten.
        _store.local_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(result["model"], _store.local_dir / USER_MODEL_FILENAMES[system])
        if result["scaler"] is not None:
            joblib.dump(result["scaler"], _store.local_dir / USER_SCALER_FILENAMES[system])
        with open(_store.local_dir / USER_STATS_FILENAMES[system], "w", encoding="utf-8") as fh:
            yaml.dump(result["stats"], fh)

        upload_warning = None
        if mlflow_run_id:
            tracker = BaseMLflowTracker(mlflow_run_id)
            try:
                tracker.log_params({"system": system, **MODEL_PARAMS[system]})
                tracker.log_metrics({**metrics, "n_samples": result["n_samples"]})
                mlflow_tmp = tempfile.mkdtemp(prefix="ml40_mlflow_")
                try:
                    joblib.dump(result["model"], f"{mlflow_tmp}/{MODEL_FILENAMES[system]}")
                    if result["scaler"] is not None:
                        joblib.dump(result["scaler"], f"{mlflow_tmp}/{SCALER_FILENAMES[system]}")
                    with open(f"{mlflow_tmp}/{STATS_FILENAMES[system]}", "w", encoding="utf-8") as fh:
                        yaml.dump(result["stats"], fh)
                    with open(f"{mlflow_tmp}/{THRESHOLDS_FILENAMES[system]}", "w", encoding="utf-8") as fh:
                        yaml.dump(thresholds, fh)  # rules thresholds travel with the bundle
                    tracker.upload_artifacts(mlflow_tmp, artifact_path="model")
                finally:
                    shutil.rmtree(mlflow_tmp, ignore_errors=True)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("MLflow artifact upload failed: %s", exc)
                upload_warning = f"Modelo guardado localmente, pero falló la subida a MLflow: {exc}"

        logger.info(
            "ml40 train() done — system=%s n_samples=%d f1_macro=%.4f mlflow=%s",
            system, result["n_samples"], metrics["f1_macro"], bool(mlflow_run_id),
        )
        return TrainResponse(
            detail=f"Reentrenamiento completado para el sistema {system} con el procedimiento original.",
            system=system,
            n_samples=result["n_samples"],
            n_runs_train=result["n_runs_train"],
            n_runs_test=result["n_runs_test"],
            accuracy=metrics["accuracy"],
            f1_macro=metrics["f1_macro"],
            precision_macro=metrics["precision_macro"],
            recall_macro=metrics["recall_macro"],
            upload_warning=upload_warning,
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the dual input/output contract and reported metrics."""
        inputs = [
            InputField(
                name="system",
                type="str",
                description="Subsistema: refrigeracion o aireado (autodetectable por columnas).",
            ),
            InputField(name="run_id", type="int", description="Identificador de ciclo (ambos sistemas)."),
            InputField(name="time_min", type="int", description="Minuto dentro del ciclo (ambos sistemas)."),
        ]
        for system in ("refrigeracion", "aireado"):
            inputs += [
                InputField(
                    name=col,
                    type="float",
                    description=f"Sensor crudo del sistema {system} "
                                f"(histórico mínimo {MIN_HISTORY_MINUTES[system]} min por ciclo).",
                )
                for col in RAW_INPUT_COLUMNS[system]
            ]
        outputs = [
            OutputField(name="run_id", type="int", description="Ciclo diagnosticado"),
            OutputField(name="prediction", type="int",
                        description="Clase de fallo final (RF + reglas físicas + voto por ciclo)"),
            OutputField(name="prediction_name", type="str",
                        description="Nombre de la clase (refrigeracion: 13 clases; aireado: 4 clases)"),
            OutputField(name="confidence", type="float", description="Confianza media del ciclo (0-1)"),
            OutputField(name="model_health", type="str",
                        description="ESTABLE/DEGRADADO según confianza media vs. 75% (stateless)"),
        ]
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Sistema DNSL de diagnóstico de fallas para equipos cárnicos: dos RandomForest "
                "(refrigeración, 13 clases; aireado de embutidos, 4 clases) corregidos por reglas "
                "termodinámicas/psicrométricas duras y consolidados por voto mayoritario por "
                "ciclo (run_id). Datasets de origen simulado/sintético — ver manifest."
            ),
            task_type="classification_multiclass_timeseries",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics={
                **METRICS_REPORTED,
                "class_mappings": {
                    system: {str(k): v for k, v in mapping.items()}
                    for system, mapping in CLASS_MAPPINGS.items()
                },
            },
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=None,
            ),
        )
        if mlflow_run_id:
            try:
                tracker = BaseMLflowTracker(mlflow_run_id)
                base.metrics["mlflow"] = {
                    "params": tracker.get_params(),
                    "metrics": tracker.get_metrics(),
                }
                logger.info("Stats enriched with MLflow data for run_id=%s", mlflow_run_id)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Could not fetch MLflow stats for run_id=%s: %s", mlflow_run_id, exc)
        return base
