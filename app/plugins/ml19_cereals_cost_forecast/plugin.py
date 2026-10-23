"""Ml19CerealsCostForecastPlugin — DATAGIA-19 multi-horizon cereal price signal.

Predicts, for h=1/2/3 months, the expected return (%) and the up/down direction of the average
price of bulk cereals (trigo, cebada, maiz) in Spain, combined into an ensemble LONG/SHORT/FLAT
signal with ALTA/BAJA confidence. Production models: Ridge (h1 regression), XGBoost (h2/h3
regression), LogReg (h1 classification), XGBoost (h2/h3 classification).

See inbox/a19/manifest.yaml for the full contract and the "DECISION DE ARQUITECTURA" note: this
plugin answers against the reference dataset bundled with the model artifacts (the delivered
src/predict/predict.py never takes raw client data either — see manifest for why). The base model
is versioned: every /train publishes the retrained models together with the dataset_v7_fe.csv
they were trained on as the new base, so predictions advance with each update of the data.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import DataContractError, ModelNotLoadedError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.infrastructure.base_model_publisher import persist_retrained_bundle
from app.plugins.ml19_cereals_cost_forecast import inference, model_loader, training
from app.plugins.ml19_cereals_cost_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
    DATASET_FE_FILENAME,
    FRAMEWORK,
    METRICS_REPORTED,
    MODEL_ID,
    VERSION,
)
from app.plugins.ml19_cereals_cost_forecast.mlflow_utils import (
    download_user_model_from_mlflow,
    upload_artifacts_to_mlflow,
)
from app.plugins.ml19_cereals_cost_forecast.predict_dto import (
    BatchPrediction,
    ClfResult,
    HorizonResult,
    PredictBatchResponse,
    PredictInlineResponse,
    RegResult,
)
from app.plugins.ml19_cereals_cost_forecast.train_dto import (
    HorizonTrainMetrics,
    SelectedClfMetrics,
    SelectedRegMetrics,
    TrainResponse,
)

logger = logging.getLogger(__name__)


def _build_horizons(result: dict[int, dict], date_label: str) -> list[HorizonResult]:
    """Convert inference.predict_next_month()'s dict[int, dict] into typed HorizonResult rows."""
    horizons: list[HorizonResult] = []
    for h in sorted(result):
        r = result[h]
        horizons.append(
            HorizonResult(
                horizon=h,
                predicted_for=inference.shift_month_label(date_label, h),
                reg=RegResult(**r["reg"]) if "reg" in r else None,
                clf=ClfResult(**r["clf"]) if "clf" in r else None,
                ensemble_signal=r.get("ensemble_signal"),
                ensemble_str=r.get("ensemble_str"),
                confidence=r.get("confidence"),
            )
        )
    return horizons


class Ml19CerealsCostForecastPlugin(ModelPluginPort):
    """Multi-horizon cereal price signal (regression + classification ensemble)."""

    BASE_ARTIFACT_FOLDER = ARTIFACT_FOLDER_NAME   # every /train publishes the new base model

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._bundle: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the fixed artifact bundle (6 model+scaler pairs + the frozen reference dataset)."""
        self._bundle = model_loader.load_artifact_bundle()
        logger.info(
            "ml19 plugin loaded: %s (reg=%s, clf=%s)",
            MODEL_ID, self._bundle["best_reg"], self._bundle["best_clf"],
        )

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
        """Return (bundle, temp_dir) as locals — never stored on self (shared across requests).

        With mlflow_run_id, the retrained bundle of that run (models + its own dataset). A run
        without a loadable model raises UserModelUnavailableError (-> 422): never fall back
        silently to the base model (repo review finding, PR #20). The caller must rmtree
        temp_dir in a finally block.
        """
        self._require_loaded()
        if mlflow_run_id:
            return download_user_model_from_mlflow(mlflow_run_id)
        return self._bundle, None

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Score one month (the latest available, or 'date' if given) of the bundled dataset."""
        _ = model_key, threshold
        date_str = features.get("date") if features else None
        bundle, user_tmp = self._resolve_bundle(mlflow_run_id)
        try:
            row, date_label = inference.get_row_from_dataset(bundle["dataset"], bundle["features"], date_str)
            result = inference.predict_next_month(row, bundle)
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)
        self._record_prediction()
        logger.info("predict_inline done — date_label=%s count=%d", date_label, self._predict_count)

        return PredictInlineResponse(
            model_id=MODEL_ID,
            date_label=date_label,
            mapa_month_used=inference.mapa_month_used(date_label),
            train_cutoff=bundle["meta"]["train_end"],
            horizons=_build_horizons(result, date_label),
        )

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Score every month listed in a CSV's 'date' column (YYYY-MM) against the bundled
        reference dataset. Unknown months are reported per-row as an error, not silently
        dropped — one bad month in a batch must not hide the others nor pass as a null result."""
        self._require_loaded()
        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path, dtype=str)
        if "date" not in df.columns:
            raise DataContractError("El CSV de entrada debe traer una columna 'date' (YYYY-MM).")
        bundle, user_tmp = self._resolve_bundle(mlflow_run_id)
        try:
            predictions = self._predict_batch_rows(df, bundle)
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)

        n_predictions = sum(1 for p in predictions if p.error is None)
        self._record_prediction()
        logger.info(
            "predict_batch done — %d filas, %d predicciones, %d errores",
            len(predictions), n_predictions, len(predictions) - n_predictions,
        )
        return PredictBatchResponse(
            model_id=MODEL_ID,
            train_cutoff=bundle["meta"]["train_end"],
            predictions=predictions,
            n_rows=len(predictions),
            n_predictions=n_predictions,
            output_path=None,
        )

    @staticmethod
    def _predict_batch_rows(df: pd.DataFrame, bundle: dict) -> list[BatchPrediction]:
        """One BatchPrediction per CSV row; a bad month is a per-row error, never the whole batch."""

        predictions: list[BatchPrediction] = []
        for raw_date in df["date"]:
            # Repo review finding (PR #20): a blank/NaN cell in the CSV's 'date' column (pandas
            # parses it as float NaN even with dtype=str) must not reach date_requested — a
            # Pydantic str field rejects a float and raises ValidationError, which is not
            # DataContractError and was escaping this try/except entirely, aborting the whole
            # batch (including the valid rows already processed) instead of just this row.
            date_str = None if pd.isna(raw_date) else str(raw_date)
            try:
                if date_str is None:
                    raise DataContractError("La columna 'date' tiene una celda vacía en esta fila.")
                row, date_label = inference.get_row_from_dataset(bundle["dataset"], bundle["features"], date_str)
                result = inference.predict_next_month(row, bundle)
                predictions.append(
                    BatchPrediction(
                        date_requested=date_str,
                        date_label=date_label,
                        mapa_month_used=inference.mapa_month_used(date_label),
                        horizons=_build_horizons(result, date_label),
                    )
                )
            except DataContractError as exc:
                predictions.append(
                    BatchPrediction(date_requested=date_str or "", error=str(exc))
                )
        return predictions

    # ── train (official pipeline; every /train publishes the new base) ───────

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> TrainResponse:
        """Retrain with the AI team's official pipeline on an updated dataset_v7_fe.csv.

        Port of src/training/train.py::run_multihorizon_training (training.py; bit-identical to
        the original in the same environment). self._bundle is never mutated. The bundle — the 6
        selected models, training_meta.json, model_selection_audit.json and the dataset itself,
        which becomes the inference dataset — is published as the new base model version and,
        if mlflow_run_id is given, also saved to that run. Invalid data -> ValueError (400).
        """
        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)
            bundle_dir = Path(tempfile.mkdtemp(prefix="ml19_train_"))
            try:
                shutil.copyfile(local_path, bundle_dir / DATASET_FE_FILENAME)
            except BaseException:
                shutil.rmtree(bundle_dir, ignore_errors=True)
                raise
        try:
            result = training.train_multihorizon(df)
            training.save_training_artifacts(bundle_dir, result)
            metrics = training.selected_metrics(result)
            params = {
                "train_end": result.meta["train_end"],
                "selection_reg": training.REG_SELECTION_METRIC,
                "selection_clf": training.CLF_SELECTION_METRIC,
                "best_reg": ",".join(f"h{h}:{m}" for h, m in result.best_reg.items()),
                "best_clf": ",".join(f"h{h}:{m}" for h, m in result.best_clf.items()),
                "n_features": len(result.features),
                "lasso_alpha": result.lasso_alpha,
            }
            persisted = persist_retrained_bundle(
                bundle_dir, artifact_folder=ARTIFACT_FOLDER_NAME, data_path=data_path,
                metrics=metrics, mlflow_run_id=mlflow_run_id,
                upload_to_mlflow=lambda: upload_artifacts_to_mlflow(
                    str(bundle_dir), mlflow_run_id, metrics=metrics, params=params),
            )
        finally:
            shutil.rmtree(bundle_dir, ignore_errors=True)

        horizons = []
        for h in sorted(set(result.best_reg) & set(result.best_clf)):
            reg_name, clf_name = result.best_reg[h], result.best_clf[h]
            reg_m, clf_m = result.reg_results[(reg_name, h)], result.clf_results[(clf_name, h)]
            horizons.append(HorizonTrainMetrics(
                horizon=h,
                reg=SelectedRegMetrics(model_name=reg_name, **{k: float(reg_m[k]) for k in (
                    "corr", "dir_acc", "sharpe", "mae", "rmse", "r2")}),
                clf=SelectedClfMetrics(model_name=clf_name, **{k: float(clf_m[k]) for k in (
                    "acc", "f1", "auc", "brier", "sharpe")}),
            ))
        logger.info("ml19 train() done — base %s, %d features, %.1fs",
                    persisted.base_version["version"], len(result.features), result.training_time_s)
        return TrainResponse(
            detail=(
                "Reentrenamiento completado (pipeline oficial: selección STRICT, walk-forward y "
                f"modelos finales sobre todos los datos). Modelo base {persisted.base_version['version']}."
            ),
            horizons=horizons,
            n_features=len(result.features),
            lasso_alpha=float(result.lasso_alpha),
            train_end=result.meta["train_end"],
            n_rows=result.n_rows,
            data_start=result.data_start,
            data_end=result.data_end,
            training_time_s=result.training_time_s,
            mlflow_run_id=mlflow_run_id,
            upload_warning=persisted.upload_warning,
            base_version=persisted.base_version,
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _metrics_with_run(mlflow_run_id: str) -> dict:
        metrics = dict(METRICS_REPORTED)
        if mlflow_run_id:
            tracker = BaseMLflowTracker(mlflow_run_id)
            metrics["mlflow"] = {"params": tracker.get_params(), "metrics": tracker.get_metrics()}
        return metrics

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and the reported walk-forward metrics.

        With mlflow_run_id, the params and metrics of that retraining run are added under
        metrics["mlflow"] (read-only; the reported metrics stay as delivered).
        """
        inputs = [
            InputField(
                name="date", type="str | null",
                description=(
                    "Mes a consultar dentro del dataset de referencia empaquetado (YYYY-MM). "
                    "Si se omite, se usa el mes mas reciente disponible."
                ),
            ),
        ]
        outputs = [
            OutputField(
                name="horizons", type="list[dict]",
                description=(
                    "Una entrada por horizonte (1, 2, 3 meses): retorno esperado (regresion), "
                    "probabilidad de subida (clasificacion) y senal ensemble LONG/SHORT/FLAT "
                    "con confianza ALTA/BAJA."
                ),
            ),
        ]
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Senal de trading mensual (magnitud de retorno + direccion) para el precio "
                "medio de cereales a granel en España (trigo, cebada, maiz), a 1/2/3 meses "
                "vista, mediante un ensemble de Ridge/XGBoost (regresion) y LogReg/XGBoost "
                "(clasificacion). Responde contra el dataset de referencia de la version vigente "
                "del modelo base, que cada reentreno renueva -- ver inbox/a19/manifest.yaml."
            ),
            task_type="regression_classification_ensemble",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics=self._metrics_with_run(mlflow_run_id),
            runtime_stats=RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=None),
        )
