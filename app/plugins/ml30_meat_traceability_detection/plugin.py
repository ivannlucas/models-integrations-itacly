"""Ml30MeatTraceability — traceability-incident scoring (sklearn preprocessor + Torch MLP)."""
from __future__ import annotations

import logging
import os
import pickle
import time

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.mlflow_artifacts import (
    add_mlflow_run_to_stats,
    removing_dir,
    require_train_tracker,
    upload_model_to_mlflow,
)
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.domain.services.xai_values import xai_feature_values
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml30_meat_traceability_detection.constants import (
    DEFAULT_THRESHOLD,
    FEATURE_COLUMNS,
    FRAMEWORK,
    MODEL_FILENAME,
    MODEL_ID,
    NUMERIC_FEATURES,
    PREPROCESSOR_FILENAME,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LEARNING_RATE,
    TRAIN_SEED,
    TRAIN_TEST_SPLIT,
    TRAIN_WEIGHT_DECAY,
    VERSION,
)
from app.plugins.ml30_meat_traceability_detection.contract import enforce_data_contract
from app.plugins.ml30_meat_traceability_detection.model_loader import (
    build_torch_mlp,
    load_artifacts,
    load_payload,
)
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
from app.plugins.ml30_meat_traceability_detection.mlflow_utils import (
    download_user_model_from_mlflow,
)

logger = logging.getLogger(__name__)

_ID_COLUMNS = ("row_id", "event_uid")
_TARGET = "target_traceability_incident"


class _TrainTestSplit(NamedTuple):
    """Preprocessed chronological train/test split of the training CSV."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def _to_native(value):
    """Convert a numpy scalar to a native Python type so pydantic can serialize it."""
    return value.item() if hasattr(value, "item") else value


def _batch_rows(df: pd.DataFrame, y_pred, y_score, feature_columns: list[str]) -> list[dict]:
    """One predict_batch output row per CSV row (ids, prediction, score and XAI values)."""
    id_cols = [c for c in _ID_COLUMNS if c in df.columns]
    predictions: list[dict] = []
    for i in range(len(df)):
        df_row = df.iloc[i]
        row = {c: _to_native(df_row[c]) for c in id_cols}
        row["pred_traceability_incident"] = int(y_pred[i])
        row["pred_score"] = float(y_score[i])
        row["model_name"] = MODEL_ID
        row["xai_feature_values"] = xai_feature_values(df_row, feature_columns) or None
        predictions.append(row)
    return predictions


def _fit_mlp(x_train: np.ndarray, y_train: np.ndarray):
    """Train a fresh MLP with the original best-genome hyperparameters (seeded)."""
    import torch  # pylint: disable=import-outside-toplevel
    from torch import nn  # pylint: disable=import-outside-toplevel

    # Seeded like the original TorchMLPClassifier.fit (torch + DataLoader generator), so
    # retraining on the same CSV yields the same model.
    torch.manual_seed(TRAIN_SEED)
    generator = torch.Generator()
    generator.manual_seed(TRAIN_SEED)
    mlp = build_torch_mlp(load_payload())
    mlp.train()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1),
        ),
        batch_size=TRAIN_BATCH_SIZE, shuffle=True, generator=generator,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        mlp.parameters(), lr=TRAIN_LEARNING_RATE, weight_decay=TRAIN_WEIGHT_DECAY
    )
    for _ in range(TRAIN_EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(mlp(xb), yb).backward()
            optimizer.step()

    mlp.eval()
    return mlp


def _test_metrics(mlp, x_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float, float]:
    """Accuracy, F1 and ROC-AUC of the retrained MLP on the held-out tail of the CSV."""
    import torch  # pylint: disable=import-outside-toplevel
    from sklearn.metrics import (  # pylint: disable=import-outside-toplevel
        accuracy_score, f1_score, roc_auc_score,
    )

    with torch.no_grad():
        y_score = torch.sigmoid(
            mlp(torch.tensor(x_test, dtype=torch.float32)).reshape(-1)
        ).numpy()
    y_pred = (y_score >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_score)) if len(set(y_test)) > 1 else 0.0
    return acc, f1, auc


class Ml30MeatTraceabilityDetectionPlugin(PluginRuntimeMixin, ModelPluginPort):
    """Binary classifier scoring traceability incidents in a meat-processing plant."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._preprocessor = None
        self._mlp = None
        self._feature_columns: list[str] = []

    def load(self) -> None:
        """Load the preprocessor and MLP artifacts."""
        self._preprocessor, self._mlp, self._feature_columns = load_artifacts()
        logger.info("Ml30MeatTraceabilityDetectionPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        """Return True if the model is loaded."""
        return self._mlp is not None

    def _resolve_model(self, mlflow_run_id: str) -> tuple[Any, Any, list[str], str | None]:
        """Return (preprocessor, mlp, feature_columns, temp_dir) to serve this request.

        The user's retrained model (if mlflow_run_id) is returned, never assigned to self:
        the plugin instance is shared by concurrent requests, so swapping it into self leaked
        a user's model to other requests — and, with two overlapping runs, left it in place
        as the "base" model. temp_dir is None for the base model.
        """
        if not mlflow_run_id:
            self._require_loaded()
            return self._preprocessor, self._mlp, self._feature_columns, None
        return download_user_model_from_mlflow(mlflow_run_id)

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Score a single traceability event."""
        if mlflow_run_id:
            logger.info(
                "predict_inline — using user-trained model from MLflow run_id=%s", mlflow_run_id
            )
        preprocessor, mlp, feature_columns, user_temp_dir = self._resolve_model(mlflow_run_id)
        with removing_dir(user_temp_dir):
            df = build_dataframe_from_features(features)
            enforce_data_contract(
                df, preprocessor, [k for k, v in features.items() if v is not None], logger
            )
            _, y_score = run_inference(preprocessor, mlp, df[feature_columns])

            score = float(y_score[0])
            pred = int(score >= (threshold if threshold is not None else DEFAULT_THRESHOLD))
            confidence = score if pred == 1 else 1.0 - score

            self._record()
            return PredictInlineResponse(
                model_id=MODEL_ID,
                pred_traceability_incident=pred,
                pred_score=score,
                confidence=confidence,
                model_name=MODEL_ID,
                xai_feature_values=xai_feature_values(features, feature_columns) or None,
            )

    def predict_batch(
        self, *, data_path: str, mlflow_run_id: str = ""
    ) -> PredictBatchResponse:
        """Score every row of a CSV of traceability events."""
        if mlflow_run_id:
            logger.info(
                "predict_batch — using user-trained model from MLflow run_id=%s", mlflow_run_id
            )
        preprocessor, mlp, feature_columns, user_temp_dir = self._resolve_model(mlflow_run_id)
        with removing_dir(user_temp_dir):
            with local_file_path(data_path) as local_path:
                raw_columns = pd.read_csv(local_path, nrows=0).columns
                df = build_dataframe_from_csv(local_path)
            warnings = enforce_data_contract(df, preprocessor, raw_columns, logger)
            y_pred, y_score = run_inference(preprocessor, mlp, df[feature_columns])
            predictions = _batch_rows(df, y_pred, y_score, feature_columns)

            self._record()
            return PredictBatchResponse(
                model_id=MODEL_ID, predictions=predictions, output_path=None, warnings=warnings,
            )

    def _load_train_frame(self, data_path: str) -> pd.DataFrame:
        """Read the labelled training CSV and check it against the served data contract."""
        with local_file_path(data_path) as local_path:
            raw_columns = pd.read_csv(local_path, nrows=0).columns
            df = build_dataframe_from_csv(local_path)
        if _TARGET not in df.columns:
            raise ValueError(f"CSV must contain '{_TARGET}' column for training")
        enforce_data_contract(df, self._preprocessor, raw_columns, logger)
        return df

    def _train_test_split(self, df: pd.DataFrame) -> _TrainTestSplit:
        """Chronological split (last TRAIN_TEST_SPLIT rows held out) through the preprocessor."""
        x_raw = df[FEATURE_COLUMNS]
        y = df[_TARGET].astype(int).to_numpy()
        split = int(len(x_raw) * (1 - TRAIN_TEST_SPLIT))
        x_train = self._preprocessor.transform(x_raw.iloc[:split])
        x_test = self._preprocessor.transform(x_raw.iloc[split:])
        return _TrainTestSplit(x_train, x_test, y[:split], y[split:])

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Retrain the MLP from scratch with the original best-genome hyperparameters (reusing
        the fitted preprocessor) and persist artifacts."""
        # The retrained model lives only in its own MLflow run (predict with that
        # mlflow_run_id); the served base model and its local artifacts are never replaced.
        tracker = require_train_tracker(mlflow_run_id)
        logger.info("Training with MLflow tracking, run_id=%s", mlflow_run_id)
        tracker.log_params({
            "learning_rate": TRAIN_LEARNING_RATE,
            "weight_decay": TRAIN_WEIGHT_DECAY,
            "batch_size": TRAIN_BATCH_SIZE,
            "epochs": TRAIN_EPOCHS,
            "seed": TRAIN_SEED,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss",
            "test_split": TRAIN_TEST_SPLIT,
        })

        self._require_loaded()
        t0 = time.perf_counter()
        df = self._load_train_frame(data_path)

        split = self._train_test_split(df)
        mlp = _fit_mlp(split.x_train, split.y_train)
        acc, f1, auc = _test_metrics(mlp, split.x_test, split.y_test)

        tracker.log_metrics({
            "test_accuracy": acc,
            "test_f1": f1,
            "test_roc_auc": auc,
            "n_train": int(len(split.x_train)),
            "n_test": int(len(split.x_test)),
        })

        def _write(model_dir: str) -> None:
            import torch  # pylint: disable=import-outside-toplevel

            torch.save(mlp.state_dict(), os.path.join(model_dir, MODEL_FILENAME))
            with open(os.path.join(model_dir, PREPROCESSOR_FILENAME), "wb") as fh:
                pickle.dump(self._preprocessor, fh)

        upload_model_to_mlflow(tracker, _write, prefix="ml30_mlflow_")

        elapsed = time.perf_counter() - t0
        logger.info("train() done — acc=%.4f f1=%.4f auc=%.4f mlflow=%s",
                    acc, f1, auc, bool(mlflow_run_id))
        return TrainResponse(
            detail="Training completed",
            accuracy=round(acc, 4),
            f1=round(f1, 4),
            roc_auc=round(auc, 4),
            n_train=int(len(split.x_train)),
            n_test=int(len(split.x_test)),
            training_time_s=round(elapsed, 1),
            upload_warning=None,
            mlflow_run_id=mlflow_run_id,
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
            runtime_stats=self._runtime_stats(),
        )
        return add_mlflow_run_to_stats(base, mlflow_run_id)
