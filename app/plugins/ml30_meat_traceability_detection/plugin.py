"""Ml30MeatTraceability — traceability-incident scoring (sklearn preprocessor + Torch MLP)."""
from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime, timezone

import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml30_meat_traceability_detection.constants import (
    DEFAULT_THRESHOLD,
    FEATURE_COLUMNS,
    FRAMEWORK,
    MODEL_ID,
    NUMERIC_FEATURES,
    VERSION,
)
from app.plugins.ml30_meat_traceability_detection.model_loader import load_artifacts
from app.plugins.ml30_meat_traceability_detection.postprocessing import run_inference
from app.plugins.ml30_meat_traceability_detection.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml30_meat_traceability_detection.preprocessing import (
    build_dataframe_from_csv,
    build_dataframe_from_features,
)
from app.plugins.ml30_meat_traceability_detection.train_dto import TrainResponse
from app.plugins.ml30_meat_traceability_detection.mlflow_utils import download_user_model_from_mlflow

logger = logging.getLogger(__name__)

_ID_COLUMNS = ("row_id", "event_uid")


def _to_native(value):
    """Convert a numpy scalar to a native Python type so pydantic can serialize it."""
    return value.item() if hasattr(value, "item") else value


def _xai_values_from_row(row, feature_columns: list[str]) -> dict[str, float] | None:
    """Build xai_feature_values from a features dict (inline) or DataFrame row (batch)."""
    xai: dict[str, float] = {}
    for col in feature_columns:
        val = row.get(col)
        if val is not None and pd.notna(val):
            try:
                xai[col] = float(val)
            except (TypeError, ValueError):
                pass
    return xai or None


class Ml30MeatTraceabilityDetectionPlugin(ModelPluginPort):
    """Binary classifier scoring traceability incidents in a meat-processing plant."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._preprocessor = None
        self._mlp = None
        self._feature_columns: list[str] = []
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        """Load the preprocessor and MLP artifacts."""
        self._preprocessor, self._mlp, self._feature_columns = load_artifacts()
        logger.info("Ml30MeatTraceabilityDetectionPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        """Return True if the model is loaded."""
        return self._mlp is not None

    def _require_loaded(self) -> None:
        """Raise ModelNotLoadedError if the model is not loaded."""
        if self._mlp is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record(self) -> None:
        """Update runtime counters after a prediction."""
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Score a single traceability event."""
        user_temp_dir = None
        saved_preprocessor = self._preprocessor
        saved_mlp = self._mlp
        if mlflow_run_id:
            logger.info("predict_inline — using user-trained model from MLflow run_id=%s", mlflow_run_id)
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._preprocessor, self._mlp, self._feature_columns, user_temp_dir = loaded
        try:
            self._require_loaded()
            df = build_dataframe_from_features(features)
            _, y_score = run_inference(self._preprocessor, self._mlp, df[self._feature_columns])

            score = float(y_score[0])
            thr = threshold if threshold is not None else DEFAULT_THRESHOLD
            pred = int(score >= thr)
            confidence = score if pred == 1 else 1.0 - score

            self._record()
            return PredictInlineResponse(
                model_id=MODEL_ID,
                pred_traceability_incident=pred,
                pred_score=score,
                confidence=confidence,
                model_name=MODEL_ID,
                xai_feature_values=_xai_values_from_row(features, self._feature_columns),
            )
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)
                self._preprocessor = saved_preprocessor
                self._mlp = saved_mlp

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Score every row of a CSV of traceability events."""
        user_temp_dir = None
        saved_preprocessor = self._preprocessor
        saved_mlp = self._mlp
        if mlflow_run_id:
            logger.info("predict_batch — using user-trained model from MLflow run_id=%s", mlflow_run_id)
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._preprocessor, self._mlp, self._feature_columns, user_temp_dir = loaded
        try:
            self._require_loaded()
            with local_file_path(data_path) as local_path:
                df = build_dataframe_from_csv(local_path)
            id_cols = [c for c in _ID_COLUMNS if c in df.columns]
            y_pred, y_score = run_inference(self._preprocessor, self._mlp, df[self._feature_columns])

            predictions: list[dict] = []
            for i in range(len(df)):
                df_row = df.iloc[i]
                row = {c: _to_native(df_row[c]) for c in id_cols}
                row["pred_traceability_incident"] = int(y_pred[i])
                row["pred_score"] = float(y_score[i])
                row["model_name"] = MODEL_ID
                row["xai_feature_values"] = _xai_values_from_row(df_row, self._feature_columns)
                predictions.append(row)

            self._record()
            return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)
        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)
                self._preprocessor = saved_preprocessor
                self._mlp = saved_mlp

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> TrainResponse:  # pylint: disable=too-many-locals
        """Fine-tune the MLP on a CSV (reusing the fitted preprocessor) and persist artifacts."""
        import pickle  # pylint: disable=import-outside-toplevel
        import tempfile  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
        from torch import nn  # pylint: disable=import-outside-toplevel
        from sklearn.metrics import (  # pylint: disable=import-outside-toplevel
            accuracy_score, f1_score, roc_auc_score,
        )

        from app.plugins.ml30_meat_traceability_detection.constants import (  # pylint: disable=import-outside-toplevel
            MODEL_FILENAME, PREPROCESSOR_FILENAME,
        )
        from app.plugins.ml30_meat_traceability_detection.model_loader import (  # pylint: disable=import-outside-toplevel
            _store, build_torch_mlp, load_payload,
        )

        if mlflow_run_id:
            logger.info("Training with MLflow tracking, run_id=%s", mlflow_run_id)
            tracker = BaseMLflowTracker(mlflow_run_id)
            tracker.log_params({
                "learning_rate": 0.01,
                "weight_decay": 1e-4,
                "batch_size": 128,
                "epochs": 80,
                "optimizer": "Adam",
                "loss": "BCEWithLogitsLoss",
                "test_split": 0.2,
            })
        else:
            tracker = None

        self._require_loaded()
        t0 = time.perf_counter()
        with local_file_path(data_path) as local_path:
            df = build_dataframe_from_csv(local_path)
        target = "target_traceability_incident"
        if target not in df.columns:
            raise ValueError(f"CSV must contain '{target}' column for training")

        x_raw = df[FEATURE_COLUMNS]
        y = df[target].astype(int).to_numpy()
        split = int(len(x_raw) * 0.8)
        x_train = self._preprocessor.transform(x_raw.iloc[:split])
        x_test = self._preprocessor.transform(x_raw.iloc[split:])
        y_train, y_test = y[:split], y[split:]

        mlp = build_torch_mlp(load_payload())
        mlp.train()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1),
            ),
            batch_size=128, shuffle=True,
        )
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(80):
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(mlp(xb), yb).backward()
                optimizer.step()

        mlp.eval()
        with torch.no_grad():
            y_score = torch.sigmoid(
                mlp(torch.tensor(x_test, dtype=torch.float32)).reshape(-1)
            ).numpy()
        y_pred = (y_score >= 0.5).astype(int)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_score)) if len(set(y_test)) > 1 else 0.0

        # ── MLflow: log metrics and upload artifacts ────────────────────────
        if tracker:
            tracker.log_metrics({
                "test_accuracy": acc,
                "test_f1": f1,
                "test_roc_auc": auc,
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
            })
            try:
                mlflow_tmp = tempfile.mkdtemp(prefix="ml30_mlflow_")
                torch.save(mlp.state_dict(), os.path.join(mlflow_tmp, MODEL_FILENAME))
                with open(os.path.join(mlflow_tmp, PREPROCESSOR_FILENAME), "wb") as fh:
                    pickle.dump(self._preprocessor, fh)
                tracker.upload_artifacts(mlflow_tmp, artifact_path="model")
                shutil.rmtree(mlflow_tmp, ignore_errors=True)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("MLflow artifact upload failed: %s", exc)

        _store.local_dir.mkdir(parents=True, exist_ok=True)
        torch.save(mlp.state_dict(), _store.local_dir / MODEL_FILENAME)

        self.load()
        elapsed = time.perf_counter() - t0
        logger.info("train() done — acc=%.4f f1=%.4f auc=%.4f mlflow=%s",
                    acc, f1, auc, bool(mlflow_run_id))
        return TrainResponse(
            detail="Training completed",
            accuracy=round(acc, 4),
            f1=round(f1, 4),
            roc_auc=round(auc, 4),
            n_train=int(len(x_train)),
            n_test=int(len(x_test)),
            training_time_s=round(elapsed, 1),
        )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        inputs = [
            InputField(name=c, type="float" if c in NUMERIC_FEATURES else "str", description=c)
            for c in FEATURE_COLUMNS
        ]
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Scoring de incidencias de trazabilidad cárnica en planta. Clasificación "
                "binaria (0=sin incidencia, 1=incidencia) mediante MLP neuroevolutivo."
            ),
            task_type="binary_classification",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=[
                OutputField(name="pred_traceability_incident", type="int",
                            description="0 = sin incidencia, 1 = incidencia"),
                OutputField(name="pred_score", type="float",
                            description="Probabilidad de incidencia (0.0–1.0)"),
                OutputField(name="confidence", type="float",
                            description="Probabilidad de la clase predicha"),
            ],
            metrics={"test_accuracy": 0.8707, "test_f1": 0.6016, "test_roc_auc": 0.7185},
            runtime_stats=RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=None),
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
