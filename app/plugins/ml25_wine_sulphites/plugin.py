"""Plugin for recommending optimal free SO2 doses for wine preservation."""
import logging
import shutil
import time
import types
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelPersistenceError, NoValidSimulationPointError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.plugins.ml25_wine_sulphites.constants import BASE_FREE_SO2_P99, MODEL_ID, VERSION
from app.plugins.ml25_wine_sulphites.model_loader import load_artifacts
from app.plugins.ml25_wine_sulphites.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml25_wine_sulphites.train_dto import TrainResponse
from app.plugins.ml25_wine_sulphites.mlflow_utils import download_user_predictor_from_mlflow
from app.plugins.ml25_wine_sulphites.postprocessing import (
    apply_operational_constraints,
    compute_molecular_so2,
    decode_bound_predictions,
    select_recommendation,
)
from app.plugins.ml25_wine_sulphites.preprocessing import (
    FEATURES_BOUND,
    FEATURES_QUAL,
    TARGET_QUAL,
    build_simulation_grid,
    map_request_to_wine_dict,
    normalize_training_columns,
    require_training_columns,
)

logger = logging.getLogger(__name__)

# Wine analysis inputs accepted from the inline request, in snake_case (matches
# PredictInlineRequest field names) — echoed back as xai_feature_values.
WINE_ANALYSIS_KEYS = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "density", "pH", "sulphates", "alcohol",
    "free_sulfur_dioxide", "total_sulfur_dioxide",
]


class Ml25WineSulphitesPlugin(ModelPluginPort):
    """Plugin that recommends optimal free SO2 doses for wine preservation."""

    def __init__(self) -> None:
        """Initialise the plugin with empty models and zeroed runtime counters."""
        self._model_qual: Any = None
        self._model_bound: Any = None
        self._metadata: dict = {}
        self._loaded: bool = False
        self._predict_count: int = 0
        self._last_predict_at: str | None = None
        self._total_latency_ms: float = 0.0

    def load(self) -> None:
        """Load quality and bound SO2 models plus metadata from the artifact store."""
        self._model_qual, self._model_bound, self._metadata = load_artifacts()
        self._loaded = True

    def is_loaded(self) -> bool:
        """Return True if both models and metadata have been loaded."""
        return self._loaded

    def _run_inference(self, features: dict, ctx: tuple) -> dict:
        """Run the full dual-model simulation pipeline for *features* and return raw results."""
        model_qual, model_bound, metadata = ctx
        mae_quality: float = (
            metadata.get("metrics", {}).get("quality_cv", {}).get("mae_mean", 0.427)
        )
        mae_bound: float = (
            metadata.get("metrics", {}).get("bound_cv", {}).get("mae_mean", 14.5)
        )

        free_so2_p99 = metadata.get("simulation", {}).get(
            "free_so2_p99", BASE_FREE_SO2_P99
        )

        req_ns = types.SimpleNamespace(**features)
        base_wine = map_request_to_wine_dict(req_ns)
        free_targets, qual_rows, bound_rows = build_simulation_grid(
            base_wine, features["delta_max"], free_so2_p99=free_so2_p99
        )

        raw_bound_pred = model_bound.predict(bound_rows)
        pred_bounds = decode_bound_predictions(raw_bound_pred, free_targets)

        pred_totals = np.maximum(free_targets + pred_bounds, free_targets)
        qual_rows = qual_rows.copy()
        qual_rows["total sulfur dioxide"] = pred_totals

        pred_qualities = model_qual.predict(qual_rows[FEATURES_QUAL])
        molecular_so2 = compute_molecular_so2(free_targets, base_wine["pH"])

        baseline_idx = int(np.argmin(np.abs(free_targets - base_wine["free sulfur dioxide"])))
        baseline_quality = float(pred_qualities[baseline_idx])

        try:
            valid_free, valid_bounds, valid_totals, valid_moleculars, valid_qualities = (
                apply_operational_constraints(
                    free_targets,
                    pred_bounds,
                    pred_totals,
                    molecular_so2,
                    pred_qualities,
                    features["min_molecular"],
                    features["max_total"],
                )
            )
        except ValueError as exc:
            raise NoValidSimulationPointError(str(exc)) from exc

        rec_idx, reason, intervention = select_recommendation(
            valid_free,
            valid_bounds,
            valid_totals,
            valid_moleculars,
            valid_qualities,
            baseline_quality,
            mae_quality,
        )

        return {
            "mae_quality": mae_quality,
            "mae_bound": mae_bound,
            "valid_free": valid_free,
            "valid_bounds": valid_bounds,
            "valid_totals": valid_totals,
            "valid_moleculars": valid_moleculars,
            "valid_qualities": valid_qualities,
            "baseline_quality": baseline_quality,
            "rec_idx": rec_idx,
            "reason": reason,
            "intervention": intervention,
        }

    def _resolve_model(self, mlflow_run_id: str) -> tuple[tuple, str | None]:
        """Return ((model_qual, model_bound, metadata), temp_dir) to serve this request.

        The user's retrained models (if mlflow_run_id) are returned, never assigned to self:
        the plugin instance is shared by concurrent requests, so swapping them into self leaked
        a user's model to other requests — and, with two overlapping runs, left it in place
        as the "base" model. temp_dir is None for the base model.
        """
        if not mlflow_run_id:
            return (self._model_qual, self._model_bound, self._metadata), None
        model_qual, model_bound, metadata, user_temp_dir = download_user_predictor_from_mlflow(mlflow_run_id)
        return (model_qual, model_bound, metadata), user_temp_dir

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Run inference on every row of the CSV at *data_path* and return all predictions."""
        import os
        import tempfile

        _tmp_csv: str | None = None
        local_data_path = data_path

        # ── MLflow: user-trained model for this request only (never assigned to self) ──
        if mlflow_run_id:
            logger.info("predict_batch — using user-trained model from MLflow run_id=%s", mlflow_run_id)
        ctx, user_temp_dir = self._resolve_model(mlflow_run_id)

        if data_path.startswith("s3://"):
            import boto3
            from botocore.client import Config as BotoConfig
            without_prefix = data_path[5:]
            bucket, _, s3_key = without_prefix.partition("/")
            s3 = boto3.client(
                "s3",
                endpoint_url=os.environ.get("CUSTOM_S3_ENDPOINT"),
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                config=BotoConfig(signature_version="s3v4"),
                region_name=os.environ.get("CUSTOM_REGION", "us-east-1"),
            )
            fd, _tmp_csv = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
            logger.info("Downloading batch data from s3://%s/%s", bucket, s3_key)
            s3.download_file(bucket, s3_key, _tmp_csv)
            local_data_path = _tmp_csv

        try:
            df = pd.read_csv(local_data_path, sep=None, engine="python")
            df.columns = [c.strip().replace(" ", "_") for c in df.columns]

            predictions = []
            t0 = time.perf_counter()
            for idx, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict.setdefault("min_molecular", 0.6)
                row_dict.setdefault("max_total", 200.0)
                row_dict.setdefault("delta_max", 40.0)
                try:
                    res = self._run_inference(row_dict, ctx)
                    i = res["rec_idx"]
                    prediction_row = {
                        "row": int(idx),
                        "intervention_recommended": res["intervention"],
                        "recommended_free_so2": float(res["valid_free"][i]),
                        "recommended_bound_so2": float(res["valid_bounds"][i]),
                        "recommended_total_so2": float(res["valid_totals"][i]),
                        "recommended_molecular_so2": float(res["valid_moleculars"][i]),
                        "predicted_quality": float(res["valid_qualities"][i]),
                        "recommendation_reason": res["reason"],
                        "xai_feature_values": {
                            k: float(row_dict[k]) for k in WINE_ANALYSIS_KEYS if k in row_dict
                        },
                    }
                    # Full dose-simulation path, first row only — attaching it to every row
                    # would multiply the response size by up to ~40 points/row for large
                    # batches; the platform shows it as a representative sample, same as it
                    # already does for XAI (see buildXaiContextFromBatchTabular using firstRow).
                    if idx == 0:
                        prediction_row["simulation_trajectory"] = [
                            {
                                "free_so2": float(res["valid_free"][j]),
                                "bound_so2": float(res["valid_bounds"][j]),
                                "total_so2": float(res["valid_totals"][j]),
                                "molecular_so2": float(res["valid_moleculars"][j]),
                                "predicted_quality": float(res["valid_qualities"][j]),
                                "is_recommended": j == i,
                            }
                            for j in range(len(res["valid_free"]))
                        ]
                    predictions.append(prediction_row)
                except Exception as exc:
                    predictions.append({"row": int(idx), "error": str(exc)})

            self._total_latency_ms += (time.perf_counter() - t0) * 1000
            self._predict_count += 1
            self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
            logger.info(
                "predict_batch done — %d predictions count=%d mlflow=%s",
                len(predictions), self._predict_count, bool(mlflow_run_id),
            )

            return PredictBatchResponse(
                model_id=MODEL_ID,
                predictions=predictions,
                output_path=None,
            )
        finally:
            if _tmp_csv:
                os.unlink(_tmp_csv)
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Run a single-sample inference and return the sulphite recommendation."""
        if mlflow_run_id:
            logger.info("predict_inline — using user-trained model from MLflow run_id=%s", mlflow_run_id)
        ctx, user_temp_dir = self._resolve_model(mlflow_run_id)
        try:
            t0 = time.perf_counter()
            res = self._run_inference(features, ctx)
            self._total_latency_ms += (time.perf_counter() - t0) * 1000
            i = res["rec_idx"]

            self._predict_count += 1
            self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
            logger.info(
                "predict_inline done — intervention=%s rec_free=%.1f quality=%.3f count=%d mlflow=%s",
                res["intervention"],
                float(res["valid_free"][i]),
                float(res["valid_qualities"][i]),
                self._predict_count,
                bool(mlflow_run_id),
            )

            # Echo the wine analysis inputs actually used for this prediction, so the
            # platform's XAI panel can show them next to the SHAP contributions instead
            # of only the recommendation output fields (see WINE_ANALYSIS_KEYS above).
            xai_feature_values = {
                k: float(features[k]) for k in WINE_ANALYSIS_KEYS if k in features
            }

            # Full dose-simulation path (every candidate free SO2 dose evaluated, not just
            # the recommended point) — computed by _run_inference above but previously
            # discarded here. Exposed so the platform can plot quality/total SO2 vs dose.
            simulation_trajectory = [
                {
                    "free_so2": float(res["valid_free"][j]),
                    "bound_so2": float(res["valid_bounds"][j]),
                    "total_so2": float(res["valid_totals"][j]),
                    "molecular_so2": float(res["valid_moleculars"][j]),
                    "predicted_quality": float(res["valid_qualities"][j]),
                    "is_recommended": j == i,
                }
                for j in range(len(res["valid_free"]))
            ]

            return PredictInlineResponse(
                model_id=MODEL_ID,
                threshold=threshold,
                prediction=res["intervention"],
                confidence=float(res["valid_qualities"][i]),
                features_used=list(FEATURES_QUAL),
                recommended_free_so2=float(res["valid_free"][i]),
                recommended_bound_so2=float(res["valid_bounds"][i]),
                recommended_total_so2=float(res["valid_totals"][i]),
                recommended_molecular_so2=float(res["valid_moleculars"][i]),
                predicted_quality=float(res["valid_qualities"][i]),
                baseline_predicted_quality=res["baseline_quality"],
                recommendation_reason=res["reason"],
                intervention_recommended=res["intervention"],
                mae_quality=res["mae_quality"],
                mae_bound=res["mae_bound"],
                xai_feature_values=xai_feature_values,
                simulation_trajectory=simulation_trajectory,
            )
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:  # pylint: disable=too-many-locals
        """Train dual RandomForest models from the CSV at *data_path* and upload them to MLflow."""
        # pylint: disable=import-outside-toplevel
        import os
        import tempfile
        # pylint: enable=import-outside-toplevel

        if mlflow_run_id:
            logger.info("Training with MLflow tracking, run_id=%s", mlflow_run_id)
            tracker = BaseMLflowTracker(mlflow_run_id)
            # Hiperparámetros y protocolo del equipo de IA (inbox/a25/codigo/modules/
            # wine_quality/src/training/train_rf.py: RF_MODEL_PARAMS + KFold(5,
            # shuffle=True, seed=42) sobre el dataset completo) — no un 80/20 inventado.
            tracker.log_params({
                "n_estimators": 300,
                "random_state": 42,
                "n_jobs": -1,
                "cv_splits": 5,
                "protocol": "official_cv5_full_dataset",
                "model_qual": "RandomForestRegressor",
                "model_bound": "RandomForestRegressor",
            })

        _tmp_path: str | None = None
        local_data_path = data_path

        if data_path.startswith("s3://"):
            import boto3
            from botocore.client import Config as BotoConfig

            without_prefix = data_path[5:]
            bucket, _, s3_key = without_prefix.partition("/")
            s3 = boto3.client(
                "s3",
                endpoint_url=os.environ.get("CUSTOM_S3_ENDPOINT"),
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                config=BotoConfig(signature_version="s3v4"),
                region_name=os.environ.get("CUSTOM_REGION", "us-east-1"),
            )
            fd, _tmp_path = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
            logger.info("Downloading training data from s3://%s/%s", bucket, s3_key)
            s3.download_file(bucket, s3_key, _tmp_path)
            local_data_path = _tmp_path

        try:
            result = self._train_from_local(local_data_path, mlflow_run_id=mlflow_run_id, tracker=tracker if mlflow_run_id else None)
            return result
        finally:
            if _tmp_path is not None:
                try:
                    os.unlink(_tmp_path)
                except OSError:
                    pass

    def _train_from_local(  # pylint: disable=too-many-locals,too-many-statements
        self,
        data_path: str,
        mlflow_run_id: str = "",
        tracker: BaseMLflowTracker | None = None,
    ) -> TrainResponse:
        """Core training logic operating on a local CSV file.

        Replicates the real delivered protocol (inbox/a25/codigo/modules/wine_quality/
        src/training/train_rf.py: ``train_dual_models`` + ``cv_regression_metrics``)
        instead of an invented 80/20 chronological holdout: RF hyperparameters
        n_estimators=300, random_state=42, n_jobs=-1; metrics reported are the mean
        of a 5-fold KFold(shuffle=True, random_state=42) cross-validation over the
        *entire* cleaned dataset (protocol "official_cv5_full_dataset"); the models
        actually served are then refit on 100% of the data — CV never withholds
        data from the shipped model, it only estimates its generalization error.
        """
        # pylint: disable=import-outside-toplevel
        import json
        import joblib
        import tempfile
        from sklearn.base import clone
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import KFold
        from app.plugins.ml25_wine_sulphites.constants import (
            QUALITY_RF_MODEL_FILENAME,
            BOUND_RF_MODEL_FILENAME,
            METADATA_FILENAME,
        )
        # pylint: enable=import-outside-toplevel

        t0 = time.perf_counter()
        df = pd.read_csv(data_path, sep=None, engine="python")
        df.columns = [c.strip() for c in df.columns]
        df = normalize_training_columns(df)
        require_training_columns(df)

        # Misma limpieza de consistencia química que wine_quality.common.load_wine_data:
        # descarta filas físicamente imposibles (total < free, negativos) y con NaN en
        # columnas críticas — el 80/20 anterior entrenaba sobre datos sin este filtro.
        df = df[
            (df["free sulfur dioxide"] >= 0)
            & (df["total sulfur dioxide"] >= 0)
            & (df["total sulfur dioxide"] >= df["free sulfur dioxide"])
        ].copy()
        df["bound_so2"] = (df["total sulfur dioxide"] - df["free sulfur dioxide"]).clip(lower=0)
        df = df.dropna(subset=FEATURES_QUAL + [TARGET_QUAL, "bound_so2"]).reset_index(drop=True)

        rf_params = {"n_estimators": 300, "random_state": 42, "n_jobs": -1}
        seed = 42
        n_splits = 5

        # pylint: disable=invalid-name
        X_qual = df[FEATURES_QUAL]
        y_qual = df[TARGET_QUAL].astype(float)
        X_bound = df[FEATURES_BOUND]
        y_bound = df["bound_so2"]
        # pylint: enable=invalid-name

        def _cv_mae_r2(x_all, y_all, log_transform: bool) -> tuple[float, float]:
            """5-fold CV MAE/R2 mean, matching train_rf.py's cv_regression_metrics."""
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            template = RandomForestRegressor(**rf_params)
            maes, r2s = [], []
            for tr_idx, va_idx in kf.split(x_all, y_all):
                model = clone(template)
                x_tr, x_va = x_all.iloc[tr_idx], x_all.iloc[va_idx]
                y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]
                if log_transform:
                    model.fit(x_tr, np.log1p(y_tr))
                    pred = np.maximum(np.expm1(model.predict(x_va)), 0.0)
                else:
                    model.fit(x_tr, y_tr)
                    pred = model.predict(x_va)
                maes.append(mean_absolute_error(y_va, pred))
                r2s.append(r2_score(y_va, pred))
            return float(np.mean(maes)), float(np.mean(r2s))

        mae_qual, r2_qual = _cv_mae_r2(X_qual, y_qual, log_transform=False)
        mae_bound, r2_bound = _cv_mae_r2(X_bound, y_bound, log_transform=True)

        # El modelo que se sirve en producción se ajusta sobre el 100% de los datos
        # limpios — la CV de arriba solo estima el error de generalización, nunca retira
        # datos del modelo final (protocolo "official_cv5_full_dataset" del equipo de IA).
        model_qual = RandomForestRegressor(**rf_params)
        model_qual.fit(X_qual, y_qual)

        model_bound = RandomForestRegressor(**rf_params)
        model_bound.fit(X_bound, np.log1p(y_bound))

        metadata = {
            "metrics": {
                "quality_cv": {"mae_mean": round(mae_qual, 4), "r2_mean": round(r2_qual, 4)},
                "bound_cv": {"mae_mean": round(mae_bound, 4), "r2_mean": round(r2_bound, 4)},
            },
            "protocol": "official_cv5_full_dataset",
            "cv_splits": n_splits,
            "seed": seed,
            "n_train": int(len(df)),
            "n_test": 0,
            "features_qual": list(FEATURES_QUAL),
            "features_bound": list(FEATURES_BOUND),
            "simulation": {
                # Percentiles del free SO2 de entrenamiento — acotan el rango de dosis
                # simulado en build_simulation_grid (ver wine_quality.common.build_free_grid,
                # SimulationConfig.sim_free_p_low/p_high=1/99) para no extrapolar fuera de
                # la distribución real del dataset con el que se entrenó este modelo.
                "free_so2_p1": float(np.percentile(df["free sulfur dioxide"], 1.0)),
                "free_so2_p99": float(np.percentile(df["free sulfur dioxide"], 99.0)),
            },
        }

        # The retrained model lives only in its own MLflow run (predict with that
        # mlflow_run_id); the served base model and its local artifacts are never replaced.
        upload_warning = None
        if not tracker:
            raise ModelPersistenceError("Sin run de MLflow: el modelo reentrenado no se ha guardado.")
        if tracker:
            tracker.log_metrics({
                "mae_quality": mae_qual,
                "mae_bound": mae_bound,
                "n_train": int(len(df)),
                "n_test": 0,
            })
            try:
                mlflow_tmp = tempfile.mkdtemp(prefix="wine_mlflow_")
                joblib.dump(model_qual, os.path.join(mlflow_tmp, QUALITY_RF_MODEL_FILENAME))
                joblib.dump(model_bound, os.path.join(mlflow_tmp, BOUND_RF_MODEL_FILENAME))
                with open(os.path.join(mlflow_tmp, METADATA_FILENAME), "w") as fh:
                    json.dump(metadata, fh)
                tracker.upload_artifacts(mlflow_tmp, artifact_path="model")
                shutil.rmtree(mlflow_tmp, ignore_errors=True)
            except Exception as exc:
                logger.error("MLflow artifact upload failed: %s", exc)
                raise ModelPersistenceError(f"El modelo reentrenado no se ha podido guardar en MLflow: {exc}") from exc

        elapsed = time.perf_counter() - t0
        logger.info("train() done — mae_qual=%.4f mae_bound=%.4f elapsed=%.1fs mlflow=%s",
                    mae_qual, mae_bound, elapsed, bool(mlflow_run_id))

        return TrainResponse(
            detail="Training completed",
            mae_quality=round(mae_qual, 4),
            mae_bound_so2=round(mae_bound, 4),
            n_train=int(len(df)),
            n_test=0,
            training_time_s=round(elapsed, 1),
            upload_warning=upload_warning,
            mlflow_run_id=mlflow_run_id,
        )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Build and return the full stats response including input/output schema and runtime metrics."""
        avg = self._total_latency_ms / self._predict_count if self._predict_count > 0 else None
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Recomendación de dosis óptima de SO2 libre en vinos mediante "
                "RandomForestRegressor dual (calidad + SO2 combinado)"
            ),
            task_type="regression",
            framework="sklearn",
            inputs=[
                InputField(name="fixed_acidity", type="float", description="Fixed acidity (g/dm³)"),
                InputField(name="volatile_acidity", type="float", description="Volatile acidity (g/dm³)"),
                InputField(name="citric_acid", type="float", description="Citric acid (g/dm³)"),
                InputField(name="residual_sugar", type="float", description="Residual sugar (g/dm³)"),
                InputField(name="chlorides", type="float", description="Chlorides (g/dm³)"),
                InputField(name="density", type="float", description="Density (g/cm³)"),
                InputField(name="pH", type="float", description="pH"),
                InputField(name="sulphates", type="float", description="Sulphates (g/dm³)"),
                InputField(name="alcohol", type="float", description="Alcohol (% vol.)"),
                InputField(name="free_sulfur_dioxide", type="float", description="Current free SO2 (mg/L)"),
                InputField(name="total_sulfur_dioxide", type="float", description="Current total SO2 (mg/L)"),
                InputField(name="min_molecular", type="float", default=0.6, description="Minimum molecular SO2 (mg/L)"),
                InputField(name="max_total", type="float", default=200.0, description="Maximum legal total SO2 (mg/L)"),
                InputField(
                    name="delta_max", type="float", default=40.0,
                    description="Maximum free SO2 increment to explore (mg/L)",
                ),
            ],
            outputs=[
                OutputField(name="recommended_free_so2", type="float", description="Recommended free SO2 dose (mg/L)"),
                OutputField(name="recommended_bound_so2", type="float", description="Estimated bound SO2 (mg/L)"),
                OutputField(name="recommended_total_so2", type="float", description="Estimated total SO2 (mg/L)"),
                OutputField(name="recommended_molecular_so2", type="float", description="Molecular SO2 (mg/L)"),
                OutputField(name="predicted_quality", type="float", description="Predicted sensory quality (0–10)"),
                OutputField(
                    name="baseline_predicted_quality", type="float",
                    description="Predicted sensory quality at the current free SO2 (0–10)",
                ),
                OutputField(
                    name="intervention_recommended", type="bool",
                    description="True if sulphite intervention is recommended",
                ),
                OutputField(
                    name="recommendation_reason", type="string",
                    description="Quality gain vs. decision threshold behind the recommendation",
                ),
            ],
            metrics={
                "mae_quality": self._metadata.get("metrics", {}).get("quality_cv", {}).get("mae_mean", 0.427),
                "mae_bound_so2_mg_l": self._metadata.get("metrics", {}).get("bound_cv", {}).get("mae_mean", 14.5),
            },
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=round(avg, 1) if avg is not None else None,
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
            except Exception as exc:
                logger.warning("Could not fetch MLflow stats for run_id=%s: %s", mlflow_run_id, exc)
        return base
