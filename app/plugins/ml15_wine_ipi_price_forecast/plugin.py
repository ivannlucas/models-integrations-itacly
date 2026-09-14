"""Ml15WineIpiPriceForecastPlugin — national phytosanitary price index (IPI) forecast, t+6.

Predicts the Spanish national phytosanitary price index (vitivinicultura sector, base 2020=100)
six months ahead via a Ridge regression (sklearn Pipeline: StandardScaler + Ridge(alpha=25.0)).
See inbox/a15/manifest.yaml for the full input/output contract, the golden-dataset verification,
and known issues (in particular: despite the "rnn" in the original folder name, the delivered
production model is Ridge, not a recurrent network — see constants.py::METRICS_REPORTED).

The "capa regional contextual" (regional_layer mode) documented in the memoria is NOT
implemented here — it is a separate, heavier post-process (RECAN/IPC/clima by CCAA) explicitly
scoped out in the manifest known_issues; this plugin only serves the national t+6 forecast.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import MissingRequiredFeatureError, ModelNotLoadedError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml15_wine_ipi_price_forecast import model_loader, preprocessing, training
from app.plugins.ml15_wine_ipi_price_forecast.constants import (
    ALPHA,
    CALENDAR_DERIVED_COLUMNS,
    FRAMEWORK,
    METRICS_REPORTED,
    MODEL_FILENAME,
    MODEL_ID,
    USER_MODEL_FILENAME,
    VERSION,
)
from app.plugins.ml15_wine_ipi_price_forecast.mlflow_utils import download_user_model_from_mlflow
from app.plugins.ml15_wine_ipi_price_forecast.model_loader import _store
from app.plugins.ml15_wine_ipi_price_forecast.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml15_wine_ipi_price_forecast.train_dto import TrainResponse

logger = logging.getLogger(__name__)


class Ml15WineIpiPriceForecastPlugin(ModelPluginPort):
    """Ridge regression plugin for the national phytosanitary price index (t+6)."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._payload: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the fixed Ridge artifact via ArtifactStore."""
        self._payload = model_loader.load_artifact()
        logger.info("ml15 plugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        """Return True once the artifact payload is loaded."""
        return self._payload is not None

    def _require_loaded(self) -> None:
        if not self.is_loaded():
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record_prediction(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _resolve_payload(self, mlflow_run_id: str) -> tuple[dict, str | None]:
        """Return (payload, temp_dir_to_cleanup). temp_dir is None for the fixed artifact."""
        if not mlflow_run_id:
            return self._payload, None
        logger.info("Using user-retrained model from MLflow run_id=%s", mlflow_run_id)
        loaded = download_user_model_from_mlflow(mlflow_run_id)
        if loaded is None:
            logger.warning(
                "No se pudo recuperar el modelo de MLflow run_id=%s; se usa el artefacto fijo.",
                mlflow_run_id,
            )
            return self._payload, None
        payload, tmp = loaded
        return payload, tmp

    @staticmethod
    def _predict_row(payload: dict, row: dict) -> dict[str, Any]:
        """Run the model on a single raw row dict; shared by predict_inline/predict_batch.

        Returns feature_columns/horizon/y_pred/y_anchor/origin_date/target_date/X — the caller
        picks what it needs. Raises MissingRequiredFeatureError if a feature is missing.
        """
        feature_columns = list(payload["feature_columns"])
        horizon = int(payload["horizon"])
        try:
            X = preprocessing.build_feature_frame(row, feature_columns, CALENDAR_DERIVED_COLUMNS)
        except ValueError as exc:
            raise MissingRequiredFeatureError(str(exc)) from exc

        origin_date, target_date = preprocessing.resolve_origin_target_dates(row, horizon)
        return {
            "feature_columns": feature_columns,
            "horizon": horizon,
            "X": X,
            "y_pred": float(payload["model"].predict(X)[0]),
            "y_anchor": float(X.iloc[0][str(payload["anchor_column"])]),
            "origin_date": origin_date,
            "target_date": target_date,
        }

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Predict the national IPI at t+6 from a single feature dict."""
        _ = model_key, threshold
        self._require_loaded()
        payload, tmp = self._resolve_payload(mlflow_run_id)
        try:
            r = self._predict_row(payload, features)
            self._record_prediction()
            logger.info(
                "predict_inline done — origin_date=%s y_pred=%.4f count=%d",
                r["origin_date"], r["y_pred"], self._predict_count,
            )
            return PredictInlineResponse(
                model_id=MODEL_ID,
                origin_date=r["origin_date"],
                target_date=r["target_date"],
                horizon=r["horizon"],
                y_pred=r["y_pred"],
                y_anchor=r["y_anchor"],
                model_name=str(payload.get("model_name", "Ridge")),
                xai_feature_values={col: float(r["X"].iloc[0][col]) for col in r["feature_columns"]},
            )
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Predict the national IPI at t+6 for every row in a CSV (one row = one prediction)."""
        self._require_loaded()
        payload, tmp = self._resolve_payload(mlflow_run_id)
        try:
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)

            predictions: list[dict[str, Any]] = []
            for idx, row in df.iterrows():
                try:
                    r = self._predict_row(payload, row.to_dict())
                    predictions.append({
                        "row": int(idx),
                        "origin_date": r["origin_date"],
                        "target_date": r["target_date"],
                        "horizon": r["horizon"],
                        "y_pred": r["y_pred"],
                        "y_anchor": r["y_anchor"],
                        "model_id": MODEL_ID,
                    })
                except ValueError as exc:
                    logger.warning("Error en fila %s: %s", idx, exc)
                    predictions.append({"row": int(idx), "error": str(exc)})

            self._record_prediction()
            logger.info(
                "predict_batch done — %d filas, mlflow=%s", len(predictions), bool(mlflow_run_id),
            )
            return PredictBatchResponse(
                model_id=MODEL_ID, predictions=predictions, n_predictions=len(predictions), output_path=None,
            )
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    # ── train (retraining with the original procedure) ────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> TrainResponse:
        """Retrain a fresh Pipeline(StandardScaler + Ridge(alpha=25.0)) from a labeled CSV.

        Follows the AI team's original procedure exactly (same model_kind/alpha as
        config.yaml::production_deployment — see manifest.training). Trains into a fresh
        artifact — the served fixed artifact is never mutated nor overwritten; the user
        artifact is saved locally under USER_MODEL_FILENAME and, when mlflow_run_id is given,
        uploaded to MLflow under artifact_path="model" with the canonical filename so
        mlflow_utils can reload it later.
        """
        self._require_loaded()
        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)

        result = training.train_model(df)
        model = result["model"]
        metrics = result["metrics"]

        new_payload = dict(self._payload)
        new_payload.update({
            "model": model,
            "model_params": {"alpha": ALPHA},
            "trained_at": datetime.now(tz=timezone.utc).isoformat(),
            "n_training_rows": result["n_train"],
        })

        # Persist locally under user_* name — the fixed S3 artifact is never overwritten.
        _store.local_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(new_payload, _store.local_dir / USER_MODEL_FILENAME)

        upload_warning = None
        if mlflow_run_id:
            tracker = BaseMLflowTracker(mlflow_run_id)
            mlflow_tmp = None
            try:
                tracker.log_params({"model_kind": "ridge", "alpha": ALPHA})
                tracker.log_metrics(metrics)
                mlflow_tmp = tempfile.mkdtemp(prefix="ml15_mlflow_")
                joblib.dump(new_payload, f"{mlflow_tmp}/{MODEL_FILENAME}")
                tracker.upload_artifacts(mlflow_tmp, artifact_path="model")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("MLflow artifact upload failed: %s", exc)
                upload_warning = f"Modelo guardado localmente, pero falló la subida a MLflow: {exc}"
            finally:
                if mlflow_tmp:
                    shutil.rmtree(mlflow_tmp, ignore_errors=True)

        logger.info(
            "ml15 train() done — n_train=%d n_test=%d rmse=%.4f mae=%.4f r2=%.4f mlflow=%s",
            result["n_train"], result["n_test"], metrics["rmse"], metrics["mae"], metrics["r2"],
            bool(mlflow_run_id),
        )
        return TrainResponse(
            detail="Reentrenamiento completado (Ridge, procedimiento original).",
            n_train_rows=result["n_train"],
            n_test_rows=result["n_test"],
            rmse=metrics["rmse"],
            mae=metrics["mae"],
            mape_pct=metrics["mape_pct"],
            r2=metrics["r2"],
            mda_pct=metrics["mda_pct"],
            upload_warning=upload_warning,
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and the real hold-out metrics."""
        feature_columns = list(self._payload["feature_columns"]) if self._payload else []
        inputs = [
            InputField(
                name="ipi_national_current", type="float",
                description="Último IPI fitosanitario nacional observado en el mes de origen (índice base 2020=100).",
            ),
            *[
                InputField(
                    name=f"ipi_national_lag_{i}", type="float",
                    description=f"IPI fitosanitario nacional observado {i} mes(es) antes del mes de origen.",
                )
                for i in range(1, 7)
            ],
            InputField(name="chem_sector_lag_11", type="float", description="Índice sectorial químico retardado 11 meses (base 2020=100)."),
            InputField(name="copper_lag_14", type="float", description="Precio del cobre retardado 14 meses (base 2020=100)."),
            InputField(name="eur_usd_lag_17", type="float", description="Tipo de cambio EUR/USD retardado 17 meses (base 2020=100)."),
            InputField(name="oil_brent_lag_12", type="float", description="Precio del petróleo Brent retardado 12 meses (base 2020=100)."),
            InputField(name="usa_lag_1", type="float", description="Índice de precios de pesticidas de EE.UU. retardado 1 mes (base 2020=100)."),
            InputField(
                name="month_sin", type="float", default=None,
                description="sin(2*pi*mes/12) — se deriva de 'date'/'origin_date' si se omite.",
            ),
            InputField(
                name="month_cos", type="float", default=None,
                description="cos(2*pi*mes/12) — se deriva de 'date'/'origin_date' si se omite.",
            ),
            InputField(
                name="quarter", type="int", default=None,
                description="Trimestre natural (1-4) — se deriva de 'date'/'origin_date' si se omite.",
            ),
            InputField(
                name="is_spring_risk", type="int", default=None,
                description="1 si el mes es abril/mayo/junio (riesgo fúngico primaveral), 0 en otro caso — se deriva si se omite.",
            ),
            InputField(
                name="date", type="str", default=None,
                description="Fecha de origen (YYYY-MM-DD), alias 'origin_date'. Opcional si ya se aportan las 4 variables de calendario.",
            ),
        ]
        outputs = [
            OutputField(name="y_pred", type="float", description="IPI fitosanitario nacional predicho a t+6 (índice base 2020=100)."),
            OutputField(name="y_anchor", type="float", description="ipi_national_current usado como ancla."),
            OutputField(name="target_date", type="date", description="origin_date + 6 meses, si origin_date es derivable."),
        ]
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Predicción mensual del IPI fitosanitario nacional español (sector vitivinícola, "
                "base 2020=100) a horizonte t+6 mediante Ridge Regression (sklearn, alpha=25.0) "
                "sobre 16 features autorregresivas/exógenas/estacionales. Único horizonte "
                "productivo aceptado tras evaluación multi-horizonte (t+1/t+2/t+3/t+6) — ver "
                "inbox/a15/manifest.yaml. No incluye la capa regional contextual por CCAA "
                "(regional_layer) documentada en la memoria — fuera de alcance de este plugin."
            ),
            task_type="regression_timeseries",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics={**METRICS_REPORTED, "feature_columns": feature_columns},
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=None,
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
