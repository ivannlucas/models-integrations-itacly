"""Ml46DairyFoulingClogDetectionPlugin — causal TCN multitask model over 120-min telemetry windows.

Detects milk heat-exchanger fouling stage/severity and clogging risk (CU07 / DNSL). See
inbox/a46/manifest.yaml for the full input/output contract and the discrepancy between the
delivered memoria and the actually-shipped artifacts (10 assets, no_clock scenario served).
"""
from __future__ import annotations

import logging
import math
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import InsufficientTelemetryHistoryError, ModelNotLoadedError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml46_dairy_fouling_clog_detection import model_loader
from app.plugins.ml46_dairy_fouling_clog_detection import preprocessing
from app.plugins.ml46_dairy_fouling_clog_detection import postprocessing
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.data import align_future_labels, derive_maintenance_from_telemetry, load_telemetry
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.datasets import collate_window_batch
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.evaluation import predict_loader, window_metrics_from_preds
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.features import build_feature_matrix, engineer_row_features
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.training_loop import train_one_epoch
from app.plugins.ml46_dairy_fouling_clog_detection._vendor.utils_common import set_seed
from app.plugins.ml46_dairy_fouling_clog_detection.constants import (
    FEATURE_ARTIFACTS_FILENAME,
    MODEL_FILENAME,
    MODEL_ID,
    POLICY_THRESHOLDS_FILENAME,
    RAW_FIXED_COLUMNS,
    RAW_OPTIONAL_COLUMNS,
    RAW_RECOMMENDED_COLUMNS,
    SCENARIO,
    SEQ_LEN,
    TRAINING_CONFIG_FILENAME,
    TRAIN_HARD_REQUIRED_COLUMNS,
    VERSION,
)
from app.plugins.ml46_dairy_fouling_clog_detection.mlflow_utils import download_user_model_from_mlflow
from app.plugins.ml46_dairy_fouling_clog_detection.model_loader import _store, build_model
from app.plugins.ml46_dairy_fouling_clog_detection.predict_dto import PredictBatchResponse, PredictInlineResponse
from app.plugins.ml46_dairy_fouling_clog_detection.train_dto import TrainResponse

logger = logging.getLogger(__name__)


def _clean_scalar(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain JSON-serializable Python types."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        fv = float(value)
        return None if not math.isfinite(fv) else fv
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _serialize_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-safe list[dict] records."""
    return [{k: _clean_scalar(v) for k, v in rec.items()} for rec in df.to_dict(orient="records")]


def _safe_metric(value: Any) -> float | None:
    """Return None for NaN/inf metrics (e.g. AUC/AP undefined with a single class), float otherwise."""
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


def _raw_xai_columns() -> list[str]:
    """Raw telemetry columns the explicabilidad service re-derives the 76 no_clock features
    from (see dairy_fouling_clog/features.py + common.py in that repo). Excludes the
    clock-source columns (batch_elapsed_min, time_since_last_*) — the served no_clock model
    was trained without them, so they aren't part of what's being explained."""
    return [c for c in RAW_FIXED_COLUMNS if c not in ("timestamp", "asset_id")] + RAW_RECOMMENDED_COLUMNS + RAW_OPTIONAL_COLUMNS


def _attach_xai_feature_values(explained: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Join each scored window back to its raw telemetry row (by asset_id + timestamp) and
    attach the raw feature values as an "xai_feature_values" column, so batch predictions can
    be sent to SHAP with the same real inputs the model saw (not the partial ~14-column subset
    that survives into AssetSequence.meta)."""
    explained = explained.copy()
    cols = [c for c in _raw_xai_columns() if c in raw_df.columns]
    if not cols:
        explained["xai_feature_values"] = None
        return explained

    lookup = raw_df[["asset_id", "timestamp"] + cols].copy()
    lookup["asset_id"] = lookup["asset_id"].astype(str)
    lookup["timestamp"] = pd.to_datetime(lookup["timestamp"], utc=True, errors="coerce")
    lookup = lookup.dropna(subset=["timestamp"]).drop_duplicates(subset=["asset_id", "timestamp"], keep="last")

    merged = explained[["asset_id", "timestamp"]].merge(lookup, on=["asset_id", "timestamp"], how="left")

    def _row_values(row: pd.Series) -> dict[str, Any] | None:
        xai: dict[str, Any] = {}
        for c in cols:
            val = row.get(c)
            if val is None or pd.isna(val):
                continue
            xai[c] = _clean_scalar(val)
        return xai or None

    explained["xai_feature_values"] = merged[cols].apply(_row_values, axis=1)
    return explained


class Ml46DairyFoulingClogDetectionPlugin(ModelPluginPort):
    """Causal TCN (no_clock scenario) over 120-minute telemetry windows per asset/cycle."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._model = None
        self._train_cfg = None
        self._feature_artifacts = None
        self._policy: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        """Load the no_clock TCN checkpoint plus its feature artifacts and alert policy."""
        self._model, self._train_cfg, self._feature_artifacts, self._policy = model_loader.load_artifacts()
        logger.info("Ml46DairyFoulingClogDetectionPlugin loaded: %s (scenario=%s)", MODEL_ID, SCENARIO)

    def is_loaded(self) -> bool:
        """Return True once the model bundle is loaded."""
        return self._model is not None

    def _require_loaded(self) -> None:
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record_prediction(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Score every valid 120-min window in a raw telemetry CSV (one or more assets)."""
        user_tmp = None
        saved = (self._model, self._train_cfg, self._feature_artifacts, self._policy)
        if mlflow_run_id:
            logger.info("predict_batch — using user fine-tuned model from MLflow run_id=%s", mlflow_run_id)
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._model, self._train_cfg, self._feature_artifacts, self._policy, user_tmp = loaded
        try:
            self._require_loaded()
            with local_file_path(data_path) as local_path:
                raw_df = pd.read_csv(local_path)
            sequences, feature_indices, asset_ids = preprocessing.prepare_sequences(raw_df, self._train_cfg, self._feature_artifacts)
            pred_df = postprocessing.run_inference(
                self._model, sequences, feature_indices, asset_ids, self._train_cfg,
                batch_size=256, stride=self._train_cfg.stride,
            )
            if len(pred_df) == 0:
                logger.warning("predict_batch — no valid production windows found in %s", data_path)
                self._record_prediction()
                return PredictBatchResponse(model_id=MODEL_ID, predictions=[], alerts=[], output_path=None)

            explained, alerts = postprocessing.explain_and_alert(
                pred_df, self._policy, self._feature_artifacts.predicate_thresholds, self._train_cfg,
            )
            explained = _attach_xai_feature_values(explained, raw_df)
            self._record_prediction()
            logger.info("predict_batch done — %d windows scored, %d alert episodes, mlflow=%s",
                        len(explained), len(alerts), bool(mlflow_run_id))
            return PredictBatchResponse(
                model_id=MODEL_ID,
                predictions=_serialize_records(explained),
                alerts=_serialize_records(alerts),
                output_path=None,
            )
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)
                self._model, self._train_cfg, self._feature_artifacts, self._policy = saved

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Score the most recent 120-min window from a submitted telemetry history."""
        user_tmp = None
        saved = (self._model, self._train_cfg, self._feature_artifacts, self._policy)
        if mlflow_run_id:
            logger.info("predict_inline — using user fine-tuned model from MLflow run_id=%s", mlflow_run_id)
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._model, self._train_cfg, self._feature_artifacts, self._policy, user_tmp = loaded
        try:
            self._require_loaded()
            rows: list[dict] = features["rows"]
            if len(rows) < self._train_cfg.seq_len:
                raise InsufficientTelemetryHistoryError(
                    f"Se requieren al menos {self._train_cfg.seq_len} filas de telemetría "
                    f"(1 por minuto); se recibieron {len(rows)}."
                )

            raw_df = preprocessing.build_raw_dataframe(rows)
            sequences, feature_indices, _ = preprocessing.prepare_sequences(raw_df, self._train_cfg, self._feature_artifacts)
            sequence_id, end_idx = preprocessing.last_window_only(sequences, feature_indices, seq_len=self._train_cfg.seq_len)
            if end_idx == -1:
                raise InsufficientTelemetryHistoryError(
                    "No hay ninguna ventana válida de 120 minutos en producción (sin mantenimiento "
                    "activo) dentro del historial recibido."
                )

            pred_df = postprocessing.run_inference_single_window(
                self._model, sequences, feature_indices, self._train_cfg, sequence_id, end_idx,
            )
            policy = dict(self._policy)
            if threshold is not None:
                policy["watch_foul_prob_thr"] = float(threshold)

            explained, alerts = postprocessing.explain_and_alert(
                pred_df, policy, self._feature_artifacts.predicate_thresholds, self._train_cfg,
            )
            row = explained.iloc[0]
            is_alert = bool(len(alerts) > 0)
            self._record_prediction()
            logger.info(
                "predict_inline done — asset=%s stage=%s status=%s alert=%s count=%d",
                row["asset_id"], row["pred_stage_name"], row["operator_status"], is_alert, self._predict_count,
            )
            return PredictInlineResponse(
                model_id=MODEL_ID,
                asset_id=str(row["asset_id"]),
                timestamp=_clean_scalar(row["timestamp"]),
                pred_severity=float(row["pred_severity"]),
                pred_stage=int(row["pred_stage"]),
                pred_stage_name=str(row["pred_stage_name"]),
                p_stage0=float(row["p_stage0"]),
                p_stage1=float(row["p_stage1"]),
                p_stage2=float(row["p_stage2"]),
                p_foul_h=float(row["p_foul_h"]),
                p_actionable_foul_h=float(row["p_actionable_foul_h"]),
                p_clog_h=float(row["p_clog_h"]),
                pred_tte_foul_min=float(row["pred_tte_foul_min"]),
                pred_tte_clog_min=float(row["pred_tte_clog_min"]),
                pred_ttu_min=float(row["pred_ttu_min"]),
                operator_status=str(row["operator_status"]),
                priority=str(row["priority"]),
                recommended_action=str(row["recommended_action"]),
                activated_predicates=str(row["activated_predicates"]),
                is_alert=is_alert,
                model_name=MODEL_ID,
                xai_feature_values={
                    "p_stage0": float(row["p_stage0"]),
                    "p_stage1": float(row["p_stage1"]),
                    "p_stage2": float(row["p_stage2"]),
                    "p_foul_h": float(row["p_foul_h"]),
                    "p_actionable_foul_h": float(row["p_actionable_foul_h"]),
                    "p_clog_h": float(row["p_clog_h"]),
                },
            )
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)
                self._model, self._train_cfg, self._feature_artifacts, self._policy = saved

    # ── train (fine-tuning) ──────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> TrainResponse:  # pylint: disable=too-many-locals
        """Fine-tune the served no_clock checkpoint on the caller's own labeled CSV.

        Reuses the loaded feature_artifacts (medians/IQR/baselines/class-weights) as-is — it
        does NOT refit them, matching the fine-tuning pattern (clone + continue training, never
        re-derive the scaling contract from a possibly small/unrepresentative user CSV). Same
        optimizer/loss/hyperparameters as the AI team's original code
        (models/artifacts/training_config.json). See manifest known_issues: this does not
        recalibrate the alert-policy thresholds, and does not reproduce the original repo's
        from-scratch dual-scenario asset-split retrain.
        """
        self._require_loaded()
        with local_file_path(data_path) as local_path:
            raw_df = pd.read_csv(local_path)
        missing = [c for c in TRAIN_HARD_REQUIRED_COLUMNS if c not in raw_df.columns]
        if missing:
            raise ValueError(f"CSV falta columnas requeridas: {missing}")

        telemetry_df = load_telemetry(raw_df, self._train_cfg, require_targets=True)
        maintenance_df = derive_maintenance_from_telemetry(telemetry_df)
        telemetry_df = align_future_labels(telemetry_df, maintenance_df, self._train_cfg)
        telemetry_df = engineer_row_features(telemetry_df)
        telemetry_df, full_feature_names, _ = build_feature_matrix(
            telemetry_df, self._feature_artifacts, [], self._train_cfg,
        )

        from app.plugins.ml46_dairy_fouling_clog_detection._vendor.datasets import make_sequences, WindowDataset

        sequences = make_sequences(telemetry_df, self._train_cfg, full_feature_names)
        feature_to_idx = {name: i for i, name in enumerate(full_feature_names)}
        feature_indices = [feature_to_idx[name] for name in self._feature_artifacts.no_clock_feature_names]
        asset_ids = sorted(telemetry_df["asset_id"].astype(str).unique().tolist())

        dataset = WindowDataset(sequences, asset_ids, feature_indices, self._train_cfg)
        if len(dataset) == 0:
            raise ValueError(
                "No se encontraron ventanas válidas de entrenamiento en el CSV: se necesitan al "
                f"menos {self._train_cfg.seq_len} filas consecutivas en producción (sin "
                "mantenimiento activo) por activo/ciclo."
            )

        set_seed(self._train_cfg.seed)
        train_loader = DataLoader(dataset, batch_size=self._train_cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_window_batch)
        eval_loader = DataLoader(dataset, batch_size=self._train_cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_window_batch)

        # Clone weights into a new instance — self._model stays usable for concurrent predicts.
        fine_model = build_model(self._train_cfg, self._feature_artifacts)
        fine_model.load_state_dict(self._model.state_dict())

        stage_weights = torch.tensor(self._feature_artifacts.stage_class_weights, dtype=torch.float32)
        foul_pos_weight = torch.tensor(self._feature_artifacts.foul_pos_weight, dtype=torch.float32)
        actionable_foul_pos_weight = torch.tensor(self._feature_artifacts.actionable_foul_pos_weight, dtype=torch.float32)
        clog_pos_weight = torch.tensor(self._feature_artifacts.clog_pos_weight, dtype=torch.float32)
        optimizer = Adam(fine_model.parameters(), lr=self._train_cfg.lr, weight_decay=self._train_cfg.weight_decay)

        if mlflow_run_id:
            tracker: BaseMLflowTracker | None = BaseMLflowTracker(mlflow_run_id)
            tracker.log_params({
                "epochs": self._train_cfg.epochs, "lr": self._train_cfg.lr,
                "batch_size": self._train_cfg.batch_size, "optimizer": "Adam", "scenario": SCENARIO,
            })
        else:
            tracker = None

        for epoch in range(1, self._train_cfg.epochs + 1):
            stats = train_one_epoch(
                fine_model, train_loader, optimizer, self._train_cfg,
                stage_weights, foul_pos_weight, actionable_foul_pos_weight, clog_pos_weight,
            )
            logger.info("ml46 fine-tune epoch=%d train_loss=%.4f", epoch, stats.get("train_loss", float("nan")))
            if tracker:
                tracker.log_metrics({"train_loss": stats.get("train_loss", float("nan"))}, step=epoch)

        fine_model.eval()
        pred_df = predict_loader(fine_model, eval_loader, sequences, self._train_cfg)
        metrics = window_metrics_from_preds(pred_df, self._train_cfg)

        _store.local_dir.mkdir(parents=True, exist_ok=True)
        torch.save(fine_model.state_dict(), _store.local_dir / MODEL_FILENAME)

        upload_warning = None
        if tracker:
            try:
                tracker.log_metrics({k: v for k, v in metrics.items() if math.isfinite(v)})
                mlflow_tmp = tempfile.mkdtemp(prefix="ml46_mlflow_")
                torch.save(fine_model.state_dict(), f"{mlflow_tmp}/{MODEL_FILENAME}")
                shutil.copy2(_store.path(FEATURE_ARTIFACTS_FILENAME), f"{mlflow_tmp}/{FEATURE_ARTIFACTS_FILENAME}")
                shutil.copy2(_store.path(TRAINING_CONFIG_FILENAME), f"{mlflow_tmp}/{TRAINING_CONFIG_FILENAME}")
                shutil.copy2(_store.path(POLICY_THRESHOLDS_FILENAME), f"{mlflow_tmp}/{POLICY_THRESHOLDS_FILENAME}")
                tracker.upload_artifacts(mlflow_tmp, artifact_path="model")
                shutil.rmtree(mlflow_tmp, ignore_errors=True)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("MLflow artifact upload failed: %s", exc)
                upload_warning = f"Fine-tuning guardado localmente, pero falló la subida a MLflow: {exc}"

        self.load()
        logger.info(
            "ml46 train() done — n_windows=%d epochs=%d stage_acc=%.4f mlflow=%s",
            len(dataset), self._train_cfg.epochs, metrics.get("stage_accuracy", float("nan")), bool(mlflow_run_id),
        )
        return TrainResponse(
            detail="Fine-tuning completado sobre el checkpoint no_clock servido.",
            n_windows=len(dataset),
            epochs=self._train_cfg.epochs,
            severity_rmse=float(metrics["severity_rmse"]),
            severity_mae=float(metrics["severity_mae"]),
            stage_accuracy=float(metrics["stage_accuracy"]),
            stage_macro_f1=float(metrics["stage_macro_f1"]),
            watch_foul_auc=_safe_metric(metrics.get("watch_foul_auc")),
            watch_foul_ap=_safe_metric(metrics.get("watch_foul_ap")),
            clog_h_auc=_safe_metric(metrics.get("clog_h_auc")),
            clog_h_ap=_safe_metric(metrics.get("clog_h_ap")),
            tte_foul_mae_min=float(metrics["tte_foul_mae_min"]),
            tte_clog_mae_min=float(metrics["tte_clog_mae_min"]),
            ttu_mae_min=float(metrics["ttu_mae_min"]),
            upload_warning=upload_warning,
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract, and real test-split metrics."""
        inputs = [
            InputField(name=name, type="float" if name not in ("timestamp", "asset_id") else "string",
                       description=f"Columna obligatoria del contrato mínimo común (ventana de {SEQ_LEN} min).")
            for name in RAW_FIXED_COLUMNS
        ] + [
            InputField(name=name, type="string" if name in ("phase",) else "int",
                       description="Recomendado — mejora la delimitación de producción/CIP/mantenimiento.")
            for name in RAW_RECOMMENDED_COLUMNS
        ] + [
            InputField(name=name, type="string", default=None, description="Opcional — contexto adicional del activo/producto.")
            for name in RAW_OPTIONAL_COLUMNS
        ]
        outputs = [
            OutputField(name="pred_severity", type="float", description="Severidad física de ensuciamiento (Rf_m2K_W)"),
            OutputField(name="pred_stage_name", type="str", description="Estado: stable / incipient / advanced"),
            OutputField(name="p_foul_h", type="float", description="Probabilidad de inicio de ensuciamiento en 30 min"),
            OutputField(name="p_actionable_foul_h", type="float", description="Probabilidad de ensuciamiento no planificado en 120 min"),
            OutputField(name="p_clog_h", type="float", description="Probabilidad de inicio de obstrucción en 15 min"),
            OutputField(name="pred_tte_foul_min", type="float", description="Minutos estimados hasta inicio de ensuciamiento (cap 240)"),
            OutputField(name="pred_tte_clog_min", type="float", description="Minutos estimados hasta inicio de obstrucción (cap 120)"),
            OutputField(name="pred_ttu_min", type="float", description="Minutos estimados hasta próxima intervención no planificada (cap 360)"),
            OutputField(name="operator_status", type="str", description="Normal / Watch fouling / Fouling accionable / Obstrucción probable"),
        ]
        avg_latency = None  # no per-call latency tracked at window granularity (batch scores many windows per call)
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "TCN causal multitarea (escenario no_clock) que detecta ensuciamiento e "
                "incrustación/obstrucción en intercambiadores de calor de placas lácteos a partir "
                "de ventanas de 120 minutos de telemetría de proceso. Dataset de entrenamiento "
                "100% sintético (10 activos, 198.615 filas) — ver manifest known_issues."
            ),
            task_type="timeseries_multitask",
            framework="pytorch/pandas/numpy/scikit-learn",
            inputs=inputs,
            outputs=outputs,
            metrics={
                "scenario": SCENARIO,
                "dataset": "test_split",
                "test_assets": ["asset_00", "asset_06"],
                "n_total_assets": 10,
                "n_telemetry_rows_total": 198615,
                "severity_rmse": 0.00011608393649876462,
                "severity_mae": 8.135631146805206e-05,
                "stage_accuracy": 0.9651557655396427,
                "stage_macro_f1": 0.9471419241121192,
                "watch_foul_auc": 0.9527048491980619,
                "watch_foul_ap": 0.46423455917949,
                "actionable_foul_auc": 0.9716387856257745,
                "actionable_foul_ap": 0.2400946212680447,
                "clog_h_auc": 0.9551856255780156,
                "clog_h_ap": 0.2126166553023802,
                "tte_foul_mae_min": 69.29759333205156,
                "tte_clog_mae_min": 32.53357991520313,
                "ttu_mae_min": 109.62735871918072,
                "synthetic_data_warning": (
                    "Dataset y métricas provienen de un simulador propio, sin validar contra "
                    "telemetría real de planta — ver inbox/a46/manifest.yaml known_issues."
                ),
            },
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=avg_latency,
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
