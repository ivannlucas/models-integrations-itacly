"""Ml16MeatRawMaterialPriceAlertPlugin — hybrid XGBoost + LogisticRegression price-alert classifier.

Predicts, from a monthly historical series of the Spanish meat sector, whether the raw-material
price indices for livestock (target_animales, XGBoost) and feed cereals (target_insumos,
LogisticRegression) will rise >= 2.5% over the next 4 months. See inbox/a16/manifest.yaml for
the full input/output contract, the memoria-vs-artifact metrics discrepancy, and the date-shift
convention used here (output 'fecha' is the target month, last input month + 4 — not the input
month itself, which is the convention trainer.py used for predicciones_test.csv).
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone

import joblib
import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml16_meat_raw_material_price_alert import model_loader, postprocessing, preprocessing, training
from app.plugins.ml16_meat_raw_material_price_alert.constants import (
    BAGGING_FILENAMES,
    FRAMEWORK,
    METRICS_REPORTED,
    MODEL_FILENAMES,
    MODEL_ID,
    RAW_REQUIRED_COLUMNS,
    SCALER_FILENAMES,
    TARGETS,
    TRAIN_CONFIG_FILENAME,
    USER_BAGGING_FILENAMES,
    USER_MODEL_FILENAMES,
    USER_SCALER_FILENAMES,
    USER_TRAIN_CONFIG_FILENAME,
    VERSION,
)
from app.plugins.ml16_meat_raw_material_price_alert.mlflow_utils import download_user_model_from_mlflow
from app.plugins.ml16_meat_raw_material_price_alert.model_loader import _store
from app.plugins.ml16_meat_raw_material_price_alert.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml16_meat_raw_material_price_alert.train_dto import TrainResponse

logger = logging.getLogger(__name__)


class Ml16MeatRawMaterialPriceAlertPlugin(ModelPluginPort):
    """Dual-target price-alert classifier: XGBoost (animales) + LogisticRegression (insumos)."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._models: dict | None = None
        self._scalers: dict | None = None
        self._bagging_models: dict | None = None
        self._train_config: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load both targets' fixed artifacts (models, scalers, optional bagging, train_config)."""
        artifacts = model_loader.load_artifacts()
        self._models = artifacts["models"]
        self._scalers = artifacts["scalers"]
        self._bagging_models = artifacts["bagging_models"]
        self._train_config = artifacts["train_config"]
        logger.info("Ml16 plugin loaded: %s (targets=%s)", MODEL_ID, list(self._models))

    def is_loaded(self) -> bool:
        """Return True once both targets' models/scalers/train_config are loaded."""
        return self._models is not None and self._scalers is not None and self._train_config is not None

    def _require_loaded(self) -> None:
        if not self.is_loaded():
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record_prediction(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _resolve_user_bundle(self, mlflow_run_id: str):
        """Download a user-retrained bundle from MLflow, or None on failure/absence.

        Returns ((models, scalers, bagging_models, train_config), temp_dir); the caller must
        shutil.rmtree(temp_dir) in a finally block.
        """
        if not mlflow_run_id:
            return None
        logger.info("Using user-retrained model from MLflow run_id=%s", mlflow_run_id)
        loaded = download_user_model_from_mlflow(mlflow_run_id)
        if loaded is None:
            logger.warning(
                "No se pudo recuperar el modelo de MLflow run_id=%s; se usa el artefacto fijo.",
                mlflow_run_id,
            )
            return None
        models, scalers, bagging_models, train_config, tmp = loaded
        return (models, scalers, bagging_models, train_config), tmp

    # ── shared inference core ─────────────────────────────────────────────────

    def _run(self, df: pd.DataFrame, bundle_override: tuple | None) -> tuple[dict, dict]:
        """Prepare input and run inference. Returns (run_inference results, prepared input dict)."""
        if bundle_override is not None:
            models, scalers, bagging_models, train_config = bundle_override
        else:
            models = self._models
            scalers = self._scalers
            bagging_models = self._bagging_models
            train_config = self._train_config

        prepared = preprocessing.prepare_inference_input(df, train_config, scalers)
        results = postprocessing.run_inference(
            models, bagging_models, train_config["final_thresholds"], prepared["x_flat_per_target"],
        )
        return results, prepared

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Score every valid target month in a monthly historical CSV."""
        self._require_loaded()
        user_tmp = None
        bundle_override = None
        if mlflow_run_id:
            resolved = self._resolve_user_bundle(mlflow_run_id)
            if resolved is not None:
                bundle_override, user_tmp = resolved
        try:
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)
            results, prepared = self._run(df, bundle_override)
            records = postprocessing.build_predictions_records(results, prepared["fechas"])
            self._record_prediction()
            logger.info(
                "predict_batch done — %d predicciones, mlflow=%s", len(records), bool(mlflow_run_id),
            )
            return PredictBatchResponse(
                model_id=MODEL_ID, predictions=records, n_predictions=len(records), output_path=None,
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
        """Score the most recent target month from a submitted monthly historical series."""
        _ = model_key, threshold  # no usados: los umbrales por target vienen de train_config.json
        self._require_loaded()
        user_tmp = None
        bundle_override = None
        if mlflow_run_id:
            resolved = self._resolve_user_bundle(mlflow_run_id)
            if resolved is not None:
                bundle_override, user_tmp = resolved
        try:
            rows: list[dict] = features["rows"]
            df = preprocessing.build_raw_dataframe(rows)
            results, prepared = self._run(df, bundle_override)
            records = postprocessing.build_predictions_records(results, prepared["fechas"])
            last = records[-1]

            self._record_prediction()
            logger.info(
                "predict_inline done — fecha=%s animales_pred=%s insumos_pred=%s count=%d",
                last["fecha"], last["target_animales_pred"], last["target_insumos_pred"],
                self._predict_count,
            )
            return PredictInlineResponse(
                model_id=MODEL_ID,
                fecha=last["fecha"],
                target_animales_pred=last["target_animales_pred"],
                target_animales_proba=last["target_animales_proba"],
                target_animales_proba_low=last["target_animales_proba_low"],
                target_animales_proba_high=last["target_animales_proba_high"],
                target_insumos_pred=last["target_insumos_pred"],
                target_insumos_proba=last["target_insumos_proba"],
                target_insumos_proba_low=last["target_insumos_proba_low"],
                target_insumos_proba_high=last["target_insumos_proba_high"],
                n_rows_used=len(df),
                n_predictions_available=len(records),
                model_name=MODEL_ID,
                xai_feature_values={
                    "target_animales_proba": last["target_animales_proba"],
                    "target_insumos_proba": last["target_insumos_proba"],
                },
            )
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)

    # ── train (retraining with the original procedure) ───────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> TrainResponse:
        """Retrain both targets from scratch on a labeled CSV shaped like
        dataset_clasificacion_base.csv (target_animales/target_insumos already computed — this
        plugin does not reproduce create_targets() nor the raw MAPA/GEE/RASVE ETL).

        Follows the AI team's original procedure exactly (config/config.yaml hyperparams,
        walk-forward CV threshold search with manual override for target_insumos, bootstrap
        bagging). Trains into fresh model objects — the served fixed artifacts are never
        mutated nor overwritten; user artifacts are saved under user_* filenames locally and,
        when mlflow_run_id is given, uploaded to MLflow under artifact_path="model" with the
        canonical filenames so mlflow_utils can rebuild the bundle.
        """
        self._require_loaded()
        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)

        missing = [c for c in RAW_REQUIRED_COLUMNS if c != "month" and c not in df.columns]
        missing += [c for c in TARGETS if c not in df.columns]
        if missing:
            raise ValueError(f"El CSV de entrenamiento no trae las columnas requeridas: {missing}")
        df = preprocessing.ensure_month_column(df)

        result = training.train_models(df)

        train_config = {
            "final_thresholds": result["thresholds"],
            "input_cols_per_target": result["input_cols_per_target"],
            "lookback": training.DEFAULT_LOOKBACK,
            "test_size": result["n_test"],
            "horizon": (self._train_config or {}).get("horizon", 4),
            "n_bagging": len(next(iter(result["bagging_models"].values()), [])),
        }

        # Persist locally under user_* names — the fixed S3 artifacts are never overwritten.
        _store.local_dir.mkdir(parents=True, exist_ok=True)
        for target in TARGETS:
            joblib.dump(result["models"][target], _store.local_dir / USER_MODEL_FILENAMES[target])
            joblib.dump(result["scalers"][target], _store.local_dir / USER_SCALER_FILENAMES[target])
            joblib.dump(result["bagging_models"][target], _store.local_dir / USER_BAGGING_FILENAMES[target])
        with open(_store.local_dir / USER_TRAIN_CONFIG_FILENAME, "w", encoding="utf-8") as fh:
            json.dump(train_config, fh, indent=2, ensure_ascii=False)

        upload_warning = None
        if mlflow_run_id:
            tracker = BaseMLflowTracker(mlflow_run_id)
            try:
                tracker.log_params({
                    "seed": training.DEFAULT_SEED,
                    "lookback": training.DEFAULT_LOOKBACK,
                    "n_bagging": training.DEFAULT_N_BAGGING,
                })
                flat_metrics = {
                    f"{target}_{k}": v for target, m in result["metrics"].items() for k, v in m.items()
                }
                tracker.log_metrics(flat_metrics)
                mlflow_tmp = tempfile.mkdtemp(prefix="ml16_mlflow_")
                try:
                    for target in TARGETS:
                        joblib.dump(result["models"][target], f"{mlflow_tmp}/{MODEL_FILENAMES[target]}")
                        joblib.dump(result["scalers"][target], f"{mlflow_tmp}/{SCALER_FILENAMES[target]}")
                        joblib.dump(result["bagging_models"][target], f"{mlflow_tmp}/{BAGGING_FILENAMES[target]}")
                    with open(f"{mlflow_tmp}/{TRAIN_CONFIG_FILENAME}", "w", encoding="utf-8") as fh:
                        json.dump(train_config, fh, indent=2, ensure_ascii=False)
                    tracker.upload_artifacts(mlflow_tmp, artifact_path="model")
                finally:
                    shutil.rmtree(mlflow_tmp, ignore_errors=True)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("MLflow artifact upload failed: %s", exc)
                upload_warning = f"Modelo guardado localmente, pero falló la subida a MLflow: {exc}"

        m = result["metrics"]
        logger.info(
            "ml16 train() done — n_train=%d n_test=%d animales_f1=%.3f insumos_f1=%.3f mlflow=%s",
            result["n_train"], result["n_test"], m["target_animales"]["f1"], m["target_insumos"]["f1"],
            bool(mlflow_run_id),
        )
        return TrainResponse(
            detail="Reentrenamiento completado (XGBoost + LogisticRegression, procedimiento original).",
            n_train_rows=result["n_train"],
            n_test_rows=result["n_test"],
            target_animales_threshold=m["target_animales"]["threshold"],
            target_animales_accuracy=m["target_animales"]["accuracy"],
            target_animales_precision=m["target_animales"]["precision"],
            target_animales_recall=m["target_animales"]["recall"],
            target_animales_f1=m["target_animales"]["f1"],
            target_animales_auc=m["target_animales"]["auc"],
            target_insumos_threshold=m["target_insumos"]["threshold"],
            target_insumos_accuracy=m["target_insumos"]["accuracy"],
            target_insumos_precision=m["target_insumos"]["precision"],
            target_insumos_recall=m["target_insumos"]["recall"],
            target_insumos_f1=m["target_insumos"]["f1"],
            target_insumos_auc=m["target_insumos"]["auc"],
            upload_warning=upload_warning,
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and the real hold-out metrics."""
        inputs = [
            InputField(name="fecha", type="date", description="Primer día del mes (YYYY-MM-01)."),
            InputField(name="month", type="int", description="Mes 1-12. Se deriva de 'fecha' si falta."),
            InputField(name="indice_animales", type="float",
                       description="Índice mensual de precios del sector cárnico (EUR)."),
            InputField(name="indice_insumos", type="float",
                       description="Índice mensual de precios de cereales para pienso (EUR)."),
            InputField(name="precip_total", type="float",
                       description="Precipitación total mensual (mm), promedio nacional."),
            InputField(name="precip_max", type="float", default=None,
                       description="Precipitación máxima mensual (mm). No influye en la predicción."),
            InputField(name="wet_days", type="float",
                       description="Días con lluvia al mes, promedio nacional."),
            InputField(name="wash_days", type="float", default=None,
                       description="Días de lavado al mes. No influye en la predicción."),
            InputField(name="animales_afectados", type="int",
                       description="Animales afectados por epidemias veterinarias en el mes (0 si no hay focos)."),
        ]
        outputs = [
            OutputField(name="fecha", type="date",
                        description="Mes objetivo de la predicción (último mes de entrada + 4)."),
            OutputField(name="target_animales_pred", type="int",
                        description="1 = alerta de encarecimiento (>= 2.5% en 4 meses)"),
            OutputField(name="target_animales_proba", type="float", description="Probabilidad XGBoost"),
            OutputField(name="target_insumos_pred", type="int",
                        description="1 = alerta de encarecimiento (>= 2.5% en 4 meses)"),
            OutputField(name="target_insumos_proba", type="float", description="Probabilidad LogisticRegression"),
        ]
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Modelo híbrido de alerta temprana de precios del sector cárnico: XGBoost para "
                "el índice de animales y Regresión Logística para el índice de insumos (cereales "
                "de pienso), sobre ventanas mensuales de lookback=3 con horizonte de predicción "
                "de 4 meses. Clasificación binaria (sube >= 2.5% / no sube), no regresión de "
                "precio exacto — descartada por el equipo de IA por inestabilidad frente a "
                "datos no vistos (ver memoria)."
            ),
            task_type="classification_binary_multi_target_timeseries_windowed",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics={
                **METRICS_REPORTED,
                "lookback": (self._train_config or {}).get("lookback", 3),
                "horizon": (self._train_config or {}).get("horizon", 4),
                "n_bagging": (self._train_config or {}).get("n_bagging", 12),
                "date_convention_warning": (
                    "'fecha' en la salida es el mes OBJETIVO (t + horizon), no el mes de entrada "
                    "— ver inbox/a16/manifest.yaml known_issues."
                ),
            },
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=None,  # una llamada puntúa varios meses; la latencia por mes no es comparable
            ),
        )
        if mlflow_run_id:
            try:
                tracker = BaseMLflowTracker(mlflow_run_id)
                base.metrics["mlflow"] = {"params": tracker.get_params(), "metrics": tracker.get_metrics()}
                logger.info("Stats enriched with MLflow data for run_id=%s", mlflow_run_id)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Could not fetch MLflow stats for run_id=%s: %s", mlflow_run_id, exc)
        return base
