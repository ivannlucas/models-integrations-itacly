"""Ml24DairySpatialPriceForecastPlugin — spatial dairy price forecast per CCAA, h = 1, 3, 6 months.

Predicts the monthly average retail price of TOTAL LECHE LIQUIDA (MAPA household consumption
panel) for the 17 Comunidades Autonomas with a hybrid model: a baseline calibrated between t-1
persistence and t-12 seasonality, blended (ml_alpha) with a CatBoostRegressor per horizon that
uses spatial (border-weighted) signals and EU dairy-market exogenous variables. Faithful port of
the AI team's src/predict/predictor.py and src/training/train.py; see inbox/a24/manifest.yaml
for the contract, the golden-dataset verification and the known issues (notably the
out-of-sample degradation after 2021, outputs/a24/verification_report.md). The base model is
versioned: every /train publishes the retrained horizons with the series (and, when given, the
monthly exogenous block) they were trained on as the new base.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.mlflow_artifacts import mlflow_run_summary, removing_dir
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.infrastructure.artifact_store import local_file_path
from app.infrastructure.base_model_publisher import persist_retrained_bundle
from app.plugins.ml24_dairy_spatial_price_forecast import (
    inference,
    model_loader,
    preprocessing,
    training,
)
from app.plugins.ml24_dairy_spatial_price_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
    FRAMEWORK,
    HORIZONS,
    METRICS_REPORTED,
    MODEL_ID,
    SERIES_REQUIRED_COLUMNS,
    TRAINING_CFG,
    VERSION,
)
from app.plugins.ml24_dairy_spatial_price_forecast.mlflow_utils import (
    download_user_model_from_mlflow,
    upload_artifacts_to_mlflow,
)
from app.plugins.ml24_dairy_spatial_price_forecast.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml24_dairy_spatial_price_forecast.train_dto import (
    HorizonMetrics,
    HorizonSplitMetrics,
    TrainResponse,
)

logger = logging.getLogger(__name__)


def _copy_to_temp_csv(path: str) -> str:
    """Copy *path* (local or s3://) to a temp .csv that outlives the download; caller unlinks it."""
    with local_file_path(path) as src:
        fd, local = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            shutil.copyfile(src, local)
        except BaseException:
            os.unlink(local)
            raise
    return local


def _split_years(series, validation_year: int | None, test_year: int | None) -> tuple[int, int]:
    """(validation_year, test_year): the given ones, the missing one derived, or both from the
    series (training.default_split_years)."""
    if test_year is None:
        test_year = (
            validation_year + 1 if validation_year is not None
            else training.default_split_years(series)[1]
        )
    if validation_year is None:
        validation_year = test_year - 1
    return validation_year, test_year


def _mlflow_params(result, validation_year: int, test_year: int, *, exog_renewed: bool) -> dict:
    """Params logged to the MLflow run of a retraining."""
    return {
        "model": "CatBoostRegressor + baseline calibrado (t-1, t-12)",
        "horizons": ",".join(str(h) for h in HORIZONS),
        "iterations": TRAINING_CFG["catboost_iterations"],
        "depth": TRAINING_CFG["catboost_depth"],
        "learning_rate": TRAINING_CFG["catboost_learning_rate"],
        "l2_leaf_reg": TRAINING_CFG["catboost_l2_leaf_reg"],
        "loss_function": TRAINING_CFG["loss_function"],
        "publication_lag_months": TRAINING_CFG["publication_lag_months"],
        "random_state": TRAINING_CFG["random_state"],
        "validation_year": validation_year,
        "test_year": test_year,
        "task_type": result.task_type,
        "exog_renewed": exog_renewed,
    }


def _per_horizon_metrics(result) -> dict[str, HorizonMetrics]:
    """TrainResponse h1/h3/h6 fields: split metrics and the calibrated blend of each horizon."""
    return {
        f"h{item.horizon}": HorizonMetrics(
            train=HorizonSplitMetrics(**item.metrics["train"]),
            validation=HorizonSplitMetrics(**item.metrics["validation"]),
            test=HorizonSplitMetrics(**item.metrics["test"]),
            baseline_w_persist_1=float(item.metadata["baseline_w_persist_1"]),
            baseline_w_persist_12=float(item.metadata["baseline_w_persist_12"]),
            ml_alpha=float(item.metadata["ml_alpha"]),
            n_features=len(item.metadata["feature_cols"]),
        )
        for item in result.horizons
    }


class Ml24DairySpatialPriceForecastPlugin(PluginRuntimeMixin, ModelPluginPort):
    """CatBoost + calibrated-baseline plugin for the spatial dairy price forecast (h=1/3/6)."""

    BASE_ARTIFACT_FOLDER = ARTIFACT_FOLDER_NAME   # every /train publishes the new base model

    def __init__(self) -> None:
        """Initialize an unloaded plugin (runtime counters and latency: mixin defaults)."""
        self._bundle: dict | None = None
        self._reference: dict | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the three horizons (.cbm + metadata) and the reference data via ArtifactStore."""
        self._bundle = model_loader.load_model_bundle()
        self._reference = model_loader.load_reference_data()
        logger.info(
            "ml24 plugin loaded: %s (%d filas de serie para los 5 productos del modelo)",
            MODEL_ID, len(self._reference["series"]),
        )

    def is_loaded(self) -> bool:
        """Return True once models and reference data are loaded."""
        return self._bundle is not None and self._reference is not None

    def _resolve_bundle(self, mlflow_run_id: str) -> tuple[dict, str | None]:
        """Return (bundle, temp_dir) as locals — never stored on self (shared across requests).

        A run without a loadable model raises UserModelUnavailableError (→ 422): never fall
        back silently to the base model. The caller must remove temp_dir (``removing_dir``).
        """
        self._require_loaded()
        if mlflow_run_id:
            return download_user_model_from_mlflow(mlflow_run_id)
        return self._bundle, None

    def _record_prediction(self, started: float) -> None:
        """Count one served prediction (mixin) and accumulate its latency since *started*."""
        self._record_latency((time.perf_counter() - started) * 1000.0)

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Forecast from the base series: optional ccaa / horizons filters, as_of_period cut-off."""
        _ = model_key, threshold
        started = time.perf_counter()
        ccaa = preprocessing.resolve_ccaa(features.get("ccaa"))
        bundle, user_tmp = self._resolve_bundle(mlflow_run_id)
        with removing_dir(user_tmp):
            series = preprocessing.truncate_series(self._reference["series"],
                                                   features.get("as_of_period"))
            predictions, freshness, warnings = inference.run_inference(
                bundle, self._reference, series, ccaa=ccaa, horizons=features.get("horizons"),
            )
        self._record_prediction(started)
        logger.info("predict_inline done — %d predicciones, count=%d", len(predictions),
                    self._predict_count)
        return PredictInlineResponse(
            model_id=MODEL_ID, predictions=predictions, n_predictions=len(predictions),
            data_freshness=freshness, warnings=warnings,
        )

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Forecast every CCAA x horizon from a long-format series CSV (the original's --input)."""
        started = time.perf_counter()
        bundle, user_tmp = self._resolve_bundle(mlflow_run_id)
        with removing_dir(user_tmp):
            with local_file_path(data_path) as local_path:
                series = preprocessing.read_series_csv(local_path)
            predictions, freshness, warnings = inference.run_inference(bundle, self._reference,
                                                                       series)
        self._record_prediction(started)
        logger.info("predict_batch done — %d predicciones, count=%d", len(predictions),
                    self._predict_count)
        return PredictBatchResponse(
            model_id=MODEL_ID, predictions=predictions, n_predictions=len(predictions),
            data_freshness=freshness, warnings=warnings, output_path=None,
        )

    # ── train ─────────────────────────────────────────────────────────────────

    def train(
        self,
        *,
        data_path: str,
        mlflow_run_id: str = "",
        validation_year: int | None = None,
        test_year: int | None = None,
        exog_path: str = "",
    ) -> TrainResponse:
        """Retrain h=1/3/6 from scratch with the AI team's procedure (see training.py).

        A new model is trained (self._bundle is never mutated) and published as the new base model
        version together with the series it was trained on (its five model products) and, if
        exog_path is given, the new monthly exogenous block; otherwise the current one is used and
        inherited (upload_warning says so). With mlflow_run_id the bundle is also saved to that
        run. Without validation_year/test_year they are derived from the series
        (training.default_split_years). Invalid or insufficient data raise ValueError (-> 400).
        """
        self._require_loaded()
        reference = dict(self._reference)
        exog_local = None
        with local_file_path(data_path) as local_path:
            series = preprocessing.read_series_csv(local_path)
        try:
            if exog_path:
                exog_local = _copy_to_temp_csv(exog_path)
                reference["exog"] = preprocessing.read_exog_csv(
                    exog_local, list(self._reference["exog"].columns)
                )
            validation_year, test_year = _split_years(series, validation_year, test_year)
            result = training.train_spatial_price_model(
                series, reference, validation_year=validation_year, test_year=test_year,
            )
            params = _mlflow_params(result, validation_year, test_year,
                                    exog_renewed=bool(exog_path))
            artifact_tmp = Path(tempfile.mkdtemp(prefix="ml24_train_"))
            with removing_dir(str(artifact_tmp)):
                training.save_training_artifacts(artifact_tmp, result)
                training.save_base_reference_data(artifact_tmp,
                                                  preprocessing.validate_series(series), exog_local)
                persisted = persist_retrained_bundle(
                    artifact_tmp, artifact_folder=ARTIFACT_FOLDER_NAME, data_path=data_path,
                    metrics=result.metrics, mlflow_run_id=mlflow_run_id,
                    upload_to_mlflow=lambda: upload_artifacts_to_mlflow(
                        str(artifact_tmp), mlflow_run_id, metrics=result.metrics, params=params),
                )
        finally:
            if exog_local:
                os.unlink(exog_local)

        warnings = [persisted.upload_warning] if persisted.upload_warning else []
        if not exog_path:
            warnings.append(
                "No se han renovado las exógenas mensuales (exog_path): se ha entrenado con las "
                "vigentes "
                "y el modelo base nuevo las hereda."
            )
        return TrainResponse(
            detail=(
                "spatial_price_model reentrenado desde cero (h=1/3/6) con el procedimiento del "
                "equipo de IA. "
                f"Modelo base {persisted.base_version['version']}."
            ),
            **_per_horizon_metrics(result),
            validation_year=validation_year,
            test_year=test_year,
            task_type=result.task_type,
            training_time_s=result.training_time_s,
            mlflow_run_id=mlflow_run_id,
            exog_renewed=bool(exog_path),
            upload_warning=" ".join(warnings) or None,
            base_version=persisted.base_version,
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _metrics_with_run(mlflow_run_id: str) -> dict:
        metrics = dict(METRICS_REPORTED)
        if mlflow_run_id:
            metrics["mlflow"] = mlflow_run_summary(mlflow_run_id)
        return metrics

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and the reported test metrics.

        With mlflow_run_id, the params and metrics of that retraining run are added under
        metrics["mlflow"] (read-only; the base metrics stay as reported).
        """
        inputs = [
            InputField(name="ccaa", type="list[str]", default=None,
                       description="(inline) Códigos de 3 letras o nombres oficiales de CCAA; "
                                   "vacío = las 17."),
            InputField(name="horizons", type="list[int]", format=["1", "3", "6"], default=None,
                       description="(inline) Horizontes en meses; vacío = los tres."),
            InputField(name="as_of_period", type="str", format=["YYYY-MM"], default=None,
                       description="(inline) Corta la serie base en ese mes (backtesting)."),
            InputField(name="data_path", type="str",
                       description="(batch) CSV largo con columnas "
                                   f"{list(SERIES_REQUIRED_COLUMNS)}."),
        ]
        outputs = [
            OutputField(name="predictions", type="list[dict]",
                        description="Una fila por CCAA x horizonte: feature_period, period, ccaa, "
                                    "ccaa_code, horizon_months, y_pred (EUR/kg o litro), "
                                    "y_baseline_calibrated, y_persist_1, y_persist_12, "
                                    "selected_*."),
            OutputField(name="data_freshness", type="dict",
                        description="Último mes con dato MAPA y exógeno, y su antigüedad en "
                                    "meses."),
            OutputField(name="warnings", type="list[str]",
                        description="Desfase MAPA > 24 meses o CCAA omitidas por histórico "
                                    "insuficiente."),
        ]
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Predicción espacial del precio medio mensual de TOTAL LECHE LIQUIDA (EUR/kg o "
                "litro, panel "
                "de consumo en hogares del MAPA) por comunidad autónoma, a 1, 3 y 6 meses. Modelo "
                "híbrido: "
                "baseline calibrado (t-1, t-12) interpolado con CatBoostRegressor sobre señales "
                "espaciales y "
                "exógenas del mercado lácteo europeo. Ver inbox/a24/manifest.yaml y "
                "outputs/a24/verification_report.md (degradación fuera de muestra desde 2022)."
            ),
            task_type="regression_timeseries_spatial",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics=self._metrics_with_run(mlflow_run_id),
            runtime_stats=RuntimeStats(total_predictions=self._predict_count,
                                       avg_latency_ms=self._avg_latency_ms()),
        )
