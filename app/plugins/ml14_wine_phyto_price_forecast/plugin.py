"""Ml14WinePhytoPriceForecastPlugin — GRU wine phytosanitary price forecast, t+16 weeks.

Predicts the Spanish national phytosanitary price index (MAPA, sector vitivinícola, base
2020=100) 16 weeks ahead via a GRU trained on a drift-residual decomposition (PyTorch,
input_size=39, hidden_size=64, num_layers=2) — the architecture predictor.py::_pick_best_model()
selects by RMSE on the reported test split. See inbox/a14/manifest.yaml for the full
input/output contract, the golden-dataset verification and known issues (in particular: an
earlier, unofficial local copy of this code — with an internal audit claiming a different,
LSTM-based result — was used by mistake before the client confirmed this GRU-based delivery is
the officially approved one).
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError, ModelPersistenceError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml14_wine_phyto_price_forecast import model_loader, preprocessing, training
from app.plugins.ml14_wine_phyto_price_forecast.constants import (
    FRAMEWORK,
    HORIZON_WEEKS,
    METRICS_REPORTED,
    MIN_HISTORY_ROWS,
    MODEL_ID,
    MODEL_NAME,
    RAW_VALUE_COLS,
    SEQ_LEN,
    TEST_RATIO,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_N_SEEDS,
    TRAIN_SEED_BASE,
    VERSION,
)
from app.plugins.ml14_wine_phyto_price_forecast.mlflow_utils import (
    download_user_model_from_mlflow,
    upload_artifacts_to_mlflow,
)
from app.plugins.ml14_wine_phyto_price_forecast.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml14_wine_phyto_price_forecast.train_dto import TrainResponse

logger = logging.getLogger(__name__)


class Ml14WinePhytoPriceForecastPlugin(ModelPluginPort):
    """GRU plugin for the national phytosanitary price index (t+16 weeks)."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._bundle: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the fixed GRU artifact and refit its scalers from the bundled reference dataset."""
        self._bundle = model_loader.load_artifact_bundle()
        logger.info("ml14 plugin loaded: %s (model=%s)", MODEL_ID, MODEL_NAME)

    def is_loaded(self) -> bool:
        """Return True once the artifact bundle is loaded."""
        return self._bundle is not None

    def _require_loaded(self) -> None:
        if not self.is_loaded():
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _resolve_bundle(self, mlflow_run_id: str) -> tuple[dict, str | None]:
        """Return (bundle, temp_dir) as locals — never stored on self (shared across requests).

        A run without a loadable model raises UserModelUnavailableError (→ 422): never fall
        back silently to the base model. The caller must rmtree temp_dir in a finally block.
        """
        if mlflow_run_id:
            return download_user_model_from_mlflow(mlflow_run_id)
        self._require_loaded()
        return self._bundle, None

    def _record_prediction(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Predict the national phytosanitary price index at t+16 weeks from a raw history."""
        _ = model_key, threshold
        bundle, user_tmp = self._resolve_bundle(mlflow_run_id)
        try:
            r = preprocessing.run_inference(bundle, features["rows"])
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)
        self._record_prediction()
        logger.info(
            "predict_inline done — last_observed_date=%s predicted_price=%.4f count=%d",
            r["last_observed_date"], r["predicted_price"], self._predict_count,
        )
        return PredictInlineResponse(
            model_id=MODEL_ID,
            predicted_price=r["predicted_price"],
            current_price=r["current_price"],
            drift_baseline=r["drift_baseline"],
            horizon_weeks=r["horizon_weeks"],
            last_observed_date=r["last_observed_date"],
            prediction_date=r["prediction_date"],
            model_used=MODEL_NAME,
            gap_warning=r["gap_warning"],
            n_rows_used=r["n_rows_used"],
            xai_feature_values=r["xai_feature_values"],
        )

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Predict from a CSV with a raw weekly history — one prediction anchored at its last row.

        Mirrors the delivered CLI exactly: the whole CSV is treated as one continuous history,
        not a sliding-window multi-prediction batch (the delivered code has no such mode — see
        inbox/a14/manifest.yaml known_issues).
        """
        bundle, user_tmp = self._resolve_bundle(mlflow_run_id)
        try:
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)
            rows: list[dict[str, Any]] = df.to_dict(orient="records")
            r = preprocessing.run_inference(bundle, rows)
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)
        r["model_id"] = MODEL_ID
        r["model_used"] = MODEL_NAME
        self._record_prediction()
        logger.info(
            "predict_batch done — last_observed_date=%s predicted_price=%.4f count=%d",
            r["last_observed_date"], r["predicted_price"], self._predict_count,
        )
        return PredictBatchResponse(
            model_id=MODEL_ID, predictions=[r], n_predictions=1, output_path=None,
        )

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Refit the deployed GRU with the AI team's procedure (see training.py).

        The original reads a fixed dataset path and also trains XGBoost/LSTM to compare; here the
        data comes from data_path and only the selected GRU is refit. The served model is never
        touched; the result lives only in its MLflow run (ModelPersistenceError → 502 if the
        upload fails).
        """
        self._require_loaded()   # the served model defines the feature contract
        with local_file_path(data_path) as local_path:
            result = training.train_gru(local_path, list(self._bundle["feature_columns"]))

        artifact_tmp = tempfile.mkdtemp(prefix="ml14_train_")
        try:
            model_loader.save_user_bundle(Path(artifact_tmp), result.model, result.feature_columns,
                                          result.input_scaler, result.target_scaler)
            upload_artifacts_to_mlflow(
                artifact_tmp, mlflow_run_id, metrics=result.metrics,
                params={"model": MODEL_NAME, "horizon_weeks": HORIZON_WEEKS, "seq_len": SEQ_LEN,
                        "epochs_max": TRAIN_EPOCHS, "batch_size": TRAIN_BATCH_SIZE,
                        "test_ratio": TEST_RATIO, "n_seeds": TRAIN_N_SEEDS,
                        "canonical_seed": TRAIN_SEED_BASE, "target_mode": "drift_residual_h"},
            )
        except Exception as exc:
            logger.error("ml14: MLflow upload failed: %s", exc)
            raise ModelPersistenceError(
                f"El modelo reentrenado no se ha podido guardar en MLflow: {exc}"
            ) from exc
        finally:
            shutil.rmtree(artifact_tmp, ignore_errors=True)

        return TrainResponse(
            detail=("GRU reentrenada con el procedimiento del equipo de IA (semilla 42; métricas "
                    "media ± std de 3 semillas en el test 80/20)"),
            **result.metrics, mlflow_run_id=mlflow_run_id, upload_warning=None,
        )

    @staticmethod
    def _metrics_with_run(mlflow_run_id: str) -> dict:
        metrics = dict(METRICS_REPORTED)
        if mlflow_run_id:
            tracker = BaseMLflowTracker(mlflow_run_id)
            metrics["mlflow"] = {"params": tracker.get_params(), "metrics": tracker.get_metrics()}
        return metrics

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and the real hold-out metrics.

        With mlflow_run_id, the params and metrics of that retraining run are added under
        metrics["mlflow"] (read-only; the base metrics stay as reported).
        """
        inputs = [
            InputField(
                name="rows", type="list[dict]",
                description=(
                    f"Histórico semanal (W-SUN) continuo, mínimo {MIN_HISTORY_ROWS} filas, sin "
                    "huecos ni fechas duplicadas. Cada fila: date (YYYY-MM-DD) + "
                    f"{', '.join(RAW_VALUE_COLS)}."
                ),
            ),
        ]
        outputs = [
            OutputField(name="predicted_price", type="float", description="PROTECCION_FITO predicho a horizon_weeks vista (índice base 2020=100)."),
            OutputField(name="current_price", type="float", description="PROTECCION_FITO de la última fila del histórico aportado."),
            OutputField(name="drift_baseline", type="float", description="Baseline lineal de drift — referencia frente a la que el GRU compara en el test reportado (ver manifest)."),
            OutputField(name="horizon_weeks", type="int", description=f"Siempre {HORIZON_WEEKS}."),
            OutputField(name="prediction_date", type="date", description="last_observed_date + horizon_weeks."),
        ]
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Predicción semanal del índice MAPA de precios de protección fitosanitaria "
                "(sector vitivinícola, base 2020=100) a horizonte de 16 semanas mediante GRU "
                "(PyTorch) sobre un residual respecto a un baseline lineal de drift, con 39 "
                "features autorregresivas/exógenas/estacionales. "
                "Ver inbox/a14/manifest.yaml para el contrato completo y las métricas de test "
                "frente al baseline."
            ),
            task_type="regression_timeseries",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics=self._metrics_with_run(mlflow_run_id),
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=None,
            ),
        )
