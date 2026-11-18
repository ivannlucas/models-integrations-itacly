"""Ml37WinePressPredictiveMaintenancePlugin — LSTM multitask predictive maintenance.

Faithful port of the delivered src/predict/predictor.py + src/neurosymbolic/ (production
config: run_id no_virtual_v1__rand_scool). A "cycle" is a ~60s window of 14 physical sensor
readings; the model classifies 4 components (cooler/valve/pump/accumulator) independently and a
neurosymbolic rule layer fuses the model's predictions with 9 physical threshold rules into
per-component alerts (see inbox/a37/manifest.yaml).

IMPORTANT (see manifest known_issues): the underlying dataset is NOT a real vineyard — it is the
public UCI "Condition Monitoring of Hydraulic Systems" dataset, used as a technical proxy for
press/pumping/filtration/bottling equipment.
"""
from __future__ import annotations

import json
import logging
import os

import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import DataContractError, ModelNotLoadedError
from app.domain.services.mlflow_artifacts import (
    mlflow_run_summary,
    removing_dir,
    require_train_tracker,
    upload_model_to_mlflow,
)
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml37_wine_press_predictive_maintenance import inference, model_loader, preprocessing, rules, training
from app.plugins.ml37_wine_press_predictive_maintenance.constants import (
    ARTIFACT_FOLDER_NAME,
    CLASS_MAPPINGS_FILENAME,
    FRAMEWORK,
    METRICS_REPORTED,
    MODEL_FILENAME,
    MODEL_ID,
    PHYSICAL_SENSORS,
    SCALER_FILENAME,
    TARGET_NAMES,
    VERSION,
)
from app.plugins.ml37_wine_press_predictive_maintenance.mlflow_utils import (
    download_user_model_from_mlflow,
    log_run_metadata,
)
from app.plugins.ml37_wine_press_predictive_maintenance.predict_dto import (
    BatchCyclePrediction,
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml37_wine_press_predictive_maintenance.train_dto import TrainResponse

logger = logging.getLogger(__name__)

REQUIRED_BATCH_COLUMNS = ("cycle_id", "t", *PHYSICAL_SENSORS)


class Ml37WinePressPredictiveMaintenancePlugin(PluginRuntimeMixin, ModelPluginPort):
    """LSTM multitask classifier (4 heads) + neurosymbolic rule fusion."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin (runtime counters start at the mixin defaults)."""
        self._bundle: dict | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the LSTM multitask model, scaler and class mappings."""
        self._bundle = model_loader.load_model_bundle()
        logger.info("ml37 plugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        """Return True once the bundle is loaded."""
        return self._bundle is not None

    def _require_loaded(self) -> dict:
        if self._bundle is None:
            raise ModelNotLoadedError("El modelo no está cargado.")
        return self._bundle

    def _resolve_bundle(self, mlflow_run_id: str) -> tuple[dict, str | None]:
        """Return (bundle, temp_dir) — the base bundle, or the user-retrained one from MLflow.
        Never assigns to self._bundle — a concurrent request must never see another user's
        retrained model."""
        if not mlflow_run_id:
            return self._require_loaded(), None
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
        """Diagnose one cycle's physical-sensor time series."""
        _ = model_key, threshold
        bundle, user_tmp = self._resolve_bundle(mlflow_run_id)
        with removing_dir(user_tmp):
            readings = features.get("readings") or []
            result = _predict_cycle(bundle, readings)
            self._record()
            logger.info(
                "predict_inline done — count=%d mlflow=%s", self.predict_count, bool(mlflow_run_id)
            )
            return PredictInlineResponse(model_id=MODEL_ID, **result)

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Diagnose every cycle of a long-format CSV/parquet (cycle_id, t, 14 sensores).

        A cycle with a contract error (missing sensor, non-finite value, duplicate timestamps)
        falls to that cycle's `error` field alone — one bad cycle must never abort the batch.
        """
        bundle, user_tmp = self._resolve_bundle(mlflow_run_id)
        with removing_dir(user_tmp):
            df = _read_batch_table(data_path)
            predictions = []
            for cycle_id, group in df.groupby("cycle_id", sort=False):
                predictions.append(_predict_batch_cycle(bundle, str(cycle_id), group))

            n_predictions = sum(1 for p in predictions if p.error is None)
            self._record()
            logger.info(
                "predict_batch done — %d ciclos, %d predicciones, mlflow=%s",
                len(predictions), n_predictions, bool(mlflow_run_id),
            )
            return PredictBatchResponse(
                model_id=MODEL_ID, predictions=predictions,
                n_rows=len(predictions), n_predictions=n_predictions, output_path=None,
            )

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Retrain the LSTM multitask model from scratch on a client-supplied long-format table.

        Faithful to trainer.py: a fresh model (no fine-tuning step in the original), the same
        random_stratified_cooler split (70/15/15, seed 42) and the same hyperparameters/
        callbacks. The result is uploaded ONLY to mlflow_run_id; a failed upload raises
        ModelPersistenceError (502) rather than returning 200 with a misleading warning.
        """
        tracker = require_train_tracker(mlflow_run_id)
        self._require_loaded()
        with local_file_path(data_path) as local_path:
            df = _read_table(local_path)
        result = training.train_lstm_multitask(df)

        def _write(model_dir: str) -> None:
            result.model.save(os.path.join(model_dir, MODEL_FILENAME))
            with open(os.path.join(model_dir, SCALER_FILENAME), "w", encoding="utf-8") as fh:
                json.dump(result.scaler, fh)
            with open(os.path.join(model_dir, CLASS_MAPPINGS_FILENAME), "w", encoding="utf-8") as fh:
                json.dump(result.class_mappings, fh)
            log_run_metadata(tracker, {
                "accuracy": result.accuracy, "balanced_accuracy": result.balanced_accuracy,
                "f1_macro": result.f1_macro, "n_train": result.n_train, "n_val": result.n_val,
                "n_test": result.n_test, "epochs_run": result.epochs_run,
            })

        upload_model_to_mlflow(tracker, _write, prefix=f"{ARTIFACT_FOLDER_NAME}_mlflow_")

        logger.info(
            "ml37 train() done — n_train=%d n_val=%d n_test=%d epochs_run=%d",
            result.n_train, result.n_val, result.n_test, result.epochs_run,
        )
        return TrainResponse(
            detail="Modelo LSTM multitarea reentrenado desde cero con el procedimiento del equipo de IA",
            accuracy=result.accuracy, balanced_accuracy=result.balanced_accuracy,
            f1_macro=result.f1_macro, n_train=result.n_train, n_val=result.n_val,
            n_test=result.n_test, epochs_run=result.epochs_run, upload_warning=None,
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and the real audited metrics."""
        metrics = dict(METRICS_REPORTED)
        if mlflow_run_id:
            metrics["mlflow"] = mlflow_run_summary(mlflow_run_id)

        inputs = [
            InputField(name="readings", type="list[dict[str, float]]",
                       description=f"Serie temporal de un ciclo (t + {len(PHYSICAL_SENSORS)} "
                       "sensores físicos), remuestreada internamente a 600 pasos."),
        ]
        outputs = [
            OutputField(name="predictions", type="dict[str, int]",
                        description="Código real UCI predicho por head (cooler/valve/pump/accumulator)."),
            OutputField(name="risk", type="dict[str, float]",
                        description="Incertidumbre predictiva [0,1] por head — NO es riesgo físico."),
            OutputField(name="alerts", type="list[dict]",
                        description="Alertas fusionadas reglas+modelo por componente "
                        "(INFO/WARNING/CRITICAL/EMERGENCY)."),
        ]
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Mantenimiento predictivo de prensas y equipos de embotellado vitivinícola: un "
                "LSTM multitarea clasifica la condición de 4 componentes (cooler/valve/pump/"
                "accumulator) a partir de un ciclo de 14 sensores físicos, y una capa "
                "neurosimbólica fusiona esas predicciones con 9 reglas físicas en alertas de "
                "severidad. Ver inbox/a37/manifest.yaml: el dataset es un proxy técnico (UCI "
                "Hydraulic Systems), no telemetría real de bodega."
            ),
            task_type="classification_multitask_sequence + neurosymbolic_rule_fusion",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics=metrics,
            runtime_stats=self._runtime_stats(),
        )


def _predict_cycle(bundle: dict, readings: list[dict]) -> dict:
    """Run the full pipeline (resample -> schema tensor -> scale -> model -> rules) for one
    cycle's readings — shared by predict_inline and predict_batch's per-cycle loop."""
    schema_tensor, stats = preprocessing.cycle_from_readings(readings)
    X_scaled = preprocessing.apply_scaler(schema_tensor[None, ...], bundle["scaler"])
    row = inference.run_inference(bundle, X_scaled)[0]
    predictions = inference.model_predictions_for_rules(row)
    alert_result = rules.evaluate_cycle_alerts(stats, predictions)
    return {
        "predictions": {t: row[f"pred_{t}"] for t in TARGET_NAMES},
        "risk": {t: row[f"risk_{t}"] for t in TARGET_NAMES},
        "confidence_level": {t: row[f"sem_{t}"] for t in TARGET_NAMES},
        "probabilities": {
            t: {
                int(k[len(f"p_{t}_"):]): v for k, v in row.items() if k.startswith(f"p_{t}_")
            }
            for t in TARGET_NAMES
        },
        "alerts": alert_result["alerts"],
        "rule_evidence": alert_result["rule_evidence"],
    }


def _read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _read_batch_table(data_path: str) -> pd.DataFrame:
    with local_file_path(data_path) as local_path:
        df = _read_table(local_path)
    missing = [c for c in REQUIRED_BATCH_COLUMNS if c not in df.columns]
    if missing:
        raise DataContractError(f"Faltan columnas obligatorias en el fichero de entrada: {missing}")
    return df


def _predict_batch_cycle(bundle: dict, cycle_id: str, group: pd.DataFrame) -> BatchCyclePrediction:
    """Diagnose one cycle's group of rows; a contract error becomes that cycle's ``error``."""
    try:
        readings = group.sort_values("t").to_dict("records")
        result = _predict_cycle(bundle, readings)
        return BatchCyclePrediction(cycle_id=cycle_id, **result)
    except DataContractError as exc:
        return BatchCyclePrediction(cycle_id=cycle_id, error=str(exc))
