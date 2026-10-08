"""Ml13WinePriceFluctuationPredictionPlugin — weekly bulk red-wine price-rise alert classifier.

Predicts, from a weekly MAPA bulk-wine price series, the probability that the mean price of the
next 4 weeks exceeds the current week's price by more than 2.5%. Despite the project title
("redes neuronales recurrentes"), the delivered production model is a tabular LogisticRegression
over 6 technical indicators (models/prod/model_config.json = logreg); the GRU branch is
experimental and not integrated. See inbox/a13/manifest.yaml for the full contract and known
issues (sklearn 1.3.2 pickles, bulletin year convention, CV vs hold-out metrics gap).
"""
from __future__ import annotations

import copy
import json
import logging
import math
import shutil
import tempfile
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError, ModelPersistenceError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml13_wine_price_fluctuation_prediction import model_loader, preprocessing, training
from app.plugins.ml13_wine_price_fluctuation_prediction.constants import (
    BULLETIN_COLUMN,
    CAMPAIGN_COLUMN,
    DECISION_THRESHOLD,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_FILENAME,
    FRAMEWORK,
    LOGREG_PARAMS,
    METRICS_REPORTED,
    MIN_INFERENCE_WEEKS,
    MLFLOW_ARTIFACT_PATH,
    MODEL_CONFIG_FILENAME,
    MODEL_FILENAME,
    MODEL_ID,
    N_FOLDS,
    PRICE_COLUMN_CANDIDATES,
    RANDOM_SEED,
    RETURN_THRESHOLD,
    SCALER_FILENAME,
    TARGET_WINDOW,
    TEST_SIZE,
    VERSION,
    WEEK_COLUMN,
    XGB_PARAMS,
)
from app.plugins.ml13_wine_price_fluctuation_prediction.mlflow_utils import (
    download_user_model_from_mlflow,
)
from app.plugins.ml13_wine_price_fluctuation_prediction.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml13_wine_price_fluctuation_prediction.train_dto import TrainResponse

logger = logging.getLogger(__name__)


def _clean(value):
    """NaN/NaT/numpy scalars -> JSON-friendly Python values."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _week_or_none(value) -> int | None:
    """ISO week as int, or None when the merged row has no week (NaN)."""
    value = _clean(value)
    return None if value is None else int(value)


class Ml13WinePriceFluctuationPredictionPlugin(ModelPluginPort):
    """Price-rise alert (>2.5% over the next 4 weeks) for bulk red wine — LogisticRegression."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._bundle: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the fixed AI-team bundle (model, scaler, feature_schema, model_config)."""
        self._bundle = model_loader.load_artifacts()
        logger.info("Ml13 plugin loaded: %s (model_type=%s)", MODEL_ID, self._bundle["model_type"])

    def is_loaded(self) -> bool:
        """Return True once the fixed bundle is loaded."""
        return self._bundle is not None

    def _require_loaded(self) -> None:
        if not self.is_loaded():
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record_prediction(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _resolve_bundle(self, mlflow_run_id: str) -> tuple[dict, str | None]:
        """Return (bundle, temp_dir). temp_dir is set only for an MLflow user bundle and the caller
        must shutil.rmtree it in a finally block. A run without a loadable model raises
        UserModelUnavailableError (→ 422): never fall back silently to the base model."""
        if mlflow_run_id:
            logger.info("Using user-retrained model from MLflow run_id=%s", mlflow_run_id)
            return download_user_model_from_mlflow(mlflow_run_id)
        return self._bundle, None

    # ── shared inference core ─────────────────────────────────────────────────

    @staticmethod
    def _score(df_raw_input: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """inference.py::predict_from_csv. Returns (features with pred_proba_up, merged output)."""
        df_raw, df_features, price_col = preprocessing.prepare_inference_frame(df_raw_input)
        x_scaled = bundle["scaler"].transform(df_features[FEATURE_COLUMNS].values)
        df_features["pred_proba_up"] = bundle["model"].predict_proba(x_scaled)[:, 1]
        return df_features, preprocessing.merge_predictions(df_raw, df_features, price_col)

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Score every week of a raw weekly price CSV (one output row per input row)."""
        self._require_loaded()
        bundle, user_tmp = self._resolve_bundle(mlflow_run_id)
        try:
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)
            _, df_out = self._score(df, bundle)
            records = []
            for fecha, row in df_out.iterrows():
                rec = {"fecha": None if pd.isna(fecha) else str(pd.Timestamp(fecha).date())}
                rec.update({k: _clean(v) for k, v in row.items()})
                records.append(rec)
            n_pred = int(df_out["pred_proba_up"].notna().sum())
            self._record_prediction()
            logger.info(
                "predict_batch done — %d filas, %d predicciones, mlflow=%s",
                len(records), n_pred, bool(mlflow_run_id),
            )
            return PredictBatchResponse(
                model_id=MODEL_ID,
                model_type=bundle["model_type"],
                predictions=records,
                n_rows=len(records),
                n_predictions=n_pred,
                decision_threshold=DECISION_THRESHOLD,
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
        """Score the most recent week of a submitted weekly price history."""
        _ = model_key  # un único modelo de producción
        self._require_loaded()
        decision_threshold = DECISION_THRESHOLD if threshold is None else float(threshold)
        bundle, user_tmp = self._resolve_bundle(mlflow_run_id)
        try:
            df = pd.DataFrame(features["rows"])
            df_features, df_out = self._score(df, bundle)
            last_fecha = df_features.index[-1]
            last = df_features.iloc[-1]
            out_row = df_out.loc[[last_fecha]].iloc[-1]
            proba = float(last["pred_proba_up"])
            self._record_prediction()
            logger.info(
                "predict_inline done — fecha=%s proba=%.4f count=%d",
                last_fecha.date(), proba, self._predict_count,
            )
            return PredictInlineResponse(
                model_id=MODEL_ID,
                fecha=str(last_fecha.date()),
                campaign=_clean(out_row.get(CAMPAIGN_COLUMN)),
                week=_week_or_none(out_row.get(WEEK_COLUMN)),
                price=float(last["price"]),
                pred_proba_up=proba,
                alerta_subida=int(proba >= decision_threshold),
                decision_threshold=decision_threshold,
                horizon_weeks=TARGET_WINDOW,
                return_threshold=RETURN_THRESHOLD,
                n_rows_used=len(df),
                n_predictions_available=len(df_features),
                model_name=MODEL_ID,
                model_type=bundle["model_type"],
                xai_feature_values={c: float(last[c]) for c in FEATURE_COLUMNS},
            )
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)

    # ── train (original procedure, from scratch) ──────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:  # pylint: disable=too-many-locals
        """Retrain with the AI team's original procedure on a raw weekly price CSV.

        Walk-forward CV (LogReg vs XGBoost) -> Smart Score selection -> final fit on train+val ->
        hold-out metrics on the last 24 weeks. Fresh objects are trained; the served fixed
        artifacts are never mutated nor overwritten. The retrained bundle lives only in its MLflow
        run (artifact_path="model", canonical filenames so mlflow_utils can rebuild it); if it
        cannot be uploaded, ModelPersistenceError (→ 502).
        """
        self._require_loaded()
        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)

        cols = set(df.columns)
        has_time_key = {CAMPAIGN_COLUMN, WEEK_COLUMN} <= cols or BULLETIN_COLUMN in cols
        has_price = bool(cols.intersection(PRICE_COLUMN_CANDIDATES))
        if not (has_time_key and has_price):
            raise ValueError(
                "El CSV de entrenamiento necesita 'campaign'+'week' (o 'bulletin') y una "
                f"columna de precio {PRICE_COLUMN_CANDIDATES}. "
                f"Columnas recibidas: {list(df.columns)}"
            )
        df_price = preprocessing.clean_and_index_data(preprocessing.normalize_time_keys(df))
        result = training.train_models(df_price)

        model_config = {"model_type": result["model_type"]}
        schema = copy.deepcopy(self._bundle["schema"])
        schema["feature_columns"] = FEATURE_COLUMNS
        schema.setdefault("scaler", {})["fitted_on"] = "train+validation set (user retrain)"

        cv_m = result["cv"]["metrics"]
        scores = result["cv"]["scores"]
        test_m = result["test_metrics"]
        tracker = BaseMLflowTracker(mlflow_run_id)
        mlflow_tmp = tempfile.mkdtemp(prefix="ml13_mlflow_")
        try:
            tracker.log_params({
                "best_model_type": result["model_type"],
                "return_threshold": RETURN_THRESHOLD,
                "target_window": TARGET_WINDOW,
                "test_size": TEST_SIZE,
                "n_folds": N_FOLDS,
                "random_seed": RANDOM_SEED,
                **{f"logreg_{k}": v for k, v in LOGREG_PARAMS.items()},
                **{f"xgb_{k}": v for k, v in XGB_PARAMS.items()},
            })
            tracker.log_metrics({
                **{f"test_{k}": v for k, v in test_m.items()},
                **{f"cv_{name}_{k}": v for name, m in cv_m.items() for k, v in m.items()},
                **{f"smart_score_{k}": v for k, v in scores.items()},
                "n_trainval": result["n_trainval"],
                "n_test": result["n_test"],
            })
            joblib.dump(result["model"], f"{mlflow_tmp}/{MODEL_FILENAME}")
            joblib.dump(result["scaler"], f"{mlflow_tmp}/{SCALER_FILENAME}")
            with open(f"{mlflow_tmp}/{FEATURE_SCHEMA_FILENAME}", "w", encoding="utf-8") as fh:
                json.dump(schema, fh, indent=2, ensure_ascii=False)
            with open(f"{mlflow_tmp}/{MODEL_CONFIG_FILENAME}", "w", encoding="utf-8") as fh:
                json.dump(model_config, fh, indent=2)
            tracker.upload_artifacts(mlflow_tmp, artifact_path=MLFLOW_ARTIFACT_PATH)
        except Exception as exc:
            logger.error("MLflow artifact upload failed: %s", exc)
            raise ModelPersistenceError(
                f"El modelo reentrenado no se ha podido guardar en MLflow: {exc}"
            ) from exc
        finally:
            shutil.rmtree(mlflow_tmp, ignore_errors=True)

        return TrainResponse(
            detail=(
                "Reentrenamiento completado (procedimiento original; modelo seleccionado: "
                f"{result['model_type']})."
            ),
            best_model_type=result["model_type"],
            n_trainval_rows=result["n_trainval"],
            n_test_rows=result["n_test"],
            test_period=f"{result['test_period'][0]}..{result['test_period'][1]}",
            auc=test_m["auc"],
            accuracy=test_m["accuracy"],
            f1=test_m["f1"],
            precision=test_m["precision"],
            recall=test_m["recall"],
            cv_logreg_auc_mean=cv_m["logreg"]["auc_mean"],
            cv_logreg_auc_std=cv_m["logreg"]["auc_std"],
            cv_logreg_f1_mean=cv_m["logreg"]["f1_mean"],
            cv_xgboost_auc_mean=cv_m["xgboost"]["auc_mean"],
            cv_xgboost_auc_std=cv_m["xgboost"]["auc_std"],
            cv_xgboost_f1_mean=cv_m["xgboost"]["f1_mean"],
            smart_score_logreg=scores["logreg"],
            smart_score_xgboost=scores["xgboost"],
            upload_warning=None,
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and the reported metrics."""
        inputs = [
            InputField(
                name="campaign", type="str",
                description="Campaña vitivinícola 'AAAA/AAAA' (agosto-julio).",
            ),
            InputField(
                name="week", type="int",
                description="Semana ISO 1-53 (>=31 -> primer año de campaña).",
            ),
            InputField(
                name="bulletin", type="str", default=None,
                description=(
                    "Alternativa a campaign+week: 'SEMANA {semana}/{año_fin}' "
                    "(año_fin = segundo año de campaña)."
                ),
            ),
            InputField(
                name="price_red", type="float",
                description="Precio medio semanal del vino tinto a granel (EUR/hl).",
            ),
        ]
        outputs = [
            OutputField(
                name="fecha", type="date", description="Lunes de la semana ISO de referencia t.",
            ),
            OutputField(
                name="pred_proba_up", type="float",
                description=(
                    "Probabilidad de que la media del precio en t+1..t+4 supere el precio_t "
                    "en más de un 2,5%."
                ),
            ),
            OutputField(
                name="alerta_subida", type="int",
                description="1 si pred_proba_up >= umbral de decisión (0.5 por defecto).",
            ),
        ]
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Alerta temprana de subida del precio del vino tinto a granel (boletines MAPA): "
                "Regresión Logística sobre 6 indicadores técnicos semanales (retorno logarítmico, "
                "distancia a SMA-12, RSI-14, posición en Bandas de Bollinger-20 y estacionalidad "
                "seno/coseno). Clasificación binaria con horizonte de 4 semanas y umbral de subida "
                f"del 2,5%. Requiere al menos {MIN_INFERENCE_WEEKS} semanas de histórico."
            ),
            task_type="classification_binary_timeseries_tabular",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics={
                **METRICS_REPORTED,
                "model_type_served": (self._bundle or {}).get("model_type"),
                "horizon_weeks": TARGET_WINDOW,
                "return_threshold": RETURN_THRESHOLD,
                "decision_threshold": DECISION_THRESHOLD,
                "min_history_weeks": MIN_INFERENCE_WEEKS,
            },
            runtime_stats=RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=None),
        )
        if mlflow_run_id:
            try:
                tracker = BaseMLflowTracker(mlflow_run_id)
                base.metrics["mlflow"] = {
                    "params": tracker.get_params(), "metrics": tracker.get_metrics(),
                }
                logger.info("Stats enriched with MLflow data for run_id=%s", mlflow_run_id)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Could not fetch MLflow stats for run_id=%s: %s", mlflow_run_id, exc)
        return base
