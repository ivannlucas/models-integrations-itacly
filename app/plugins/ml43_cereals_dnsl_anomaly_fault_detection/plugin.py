"""Ml43CerealsDnslAnomalyFaultDetectionPlugin — Deep Neuro-Fuzzy anomaly detection for cereal ovens.

Detects anomalies/faults in cereal baking cycles from 180-row sensor windows
(13 sensors), combining a BiLSTM branch with an ANFIS-style fuzzy-rule branch
(late fusion) plus SHAP + fuzzy-rule explanations (XAI). Vendored model code
is kept in sync with a43-44-neurofuzzy-anomalias-fallas/src/ (the training
repo this runtime serves).
"""
from __future__ import annotations

import logging
import math
import pickle
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import (
    InsufficientSensorWindowError,
    ModelPersistenceError,
)
from app.domain.services.mlflow_artifacts import removing_dir
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection import (
    model_loader,
    postprocessing,
    preprocessing,
)
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection._vendor.dnf_loss import DNFLoss
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection._vendor.preprocess import (
    split_train_val_test_by_id,
    stats_windows,
)
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection._vendor.train_metrics import (
    alpha_entropy_mean,
    compute_class_weights,
    optimize_threshold_by_f1,
    rule_corr_mean,
)
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.constants import (
    DECISION_THRESHOLD,
    DEFAULT_MODEL_CFG,
    DNF_LOSS_KWARGS,
    FRAMEWORK,
    ID_COLUMN,
    MODEL_FILENAME,
    MODEL_ID,
    NORMAL_TOKENS,
    SCALER_FILENAME,
    SENSOR_COLUMNS,
    SEQ_LENGTH,
    STATS_CREATION,
    TARGET_COLUMN,
    TEST_METRICS,
    TRAIN_BATCH_SIZE,
    TRAIN_EXTERNAL_TEST_PCT,
    TRAIN_EXTERNAL_VAL_PCT,
    TRAIN_GRAD_CLIP,
    TRAIN_LR,
    TRAIN_MIN_DELTA,
    TRAIN_NUM_EPOCHS,
    TRAIN_PATIENCE,
    TRAIN_SCHEDULER_ETA_MIN,
    TRAIN_SCHEDULER_T_MAX,
    TRAIN_THRESHOLD_SEARCH_POINTS,
    TRAIN_WARMUP_EPOCHS,
    TRAIN_WEIGHT_DECAY,
    UNIFIED_METRIC_KEYS,
    VERSION,
    XAI_BACKGROUND_FILENAME,
    monitor_score,
)
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.mlflow_utils import (
    download_user_model_from_mlflow,
    get_calibrated_threshold,
    upload_artifacts_to_mlflow,
)
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.predict_dto import PredictBatchResponse
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.train_dto import TrainResponse

logger = logging.getLogger(__name__)


def _load_pickle_file(path: Path):
    """Unpickle *path* (used to check that a just-saved pickle artifact is readable)."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _validate_saved_artifact(path: Path, loader) -> None:
    """Raise ValueError if a just-saved training artifact is missing, empty, or unreadable.

    Feedback 3 (modelo 43-44 audit, point 8 — previously investigated but left
    unimplemented pending confirmation): torch.save/pickle.dump/np.save have no return
    value to check, so a process killed or a disk filled mid-write can silently leave a
    0-byte or truncated file on disk. upload_artifacts_to_mlflow's own broad except only
    guards the *upload* step — a corrupt local file never reaches it, gets uploaded as-is,
    and the run is still reported as "Entrenamiento completado" with no warning. Catching
    this immediately after each save, before the upload is even attempted, turns that into
    a clear, immediate error instead.
    """
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(
            f"El artefacto de entrenamiento '{path.name}' se guardó vacío (0 bytes) o no "
            "se generó — entrenamiento abortado, no se subió nada a MLflow."
        )
    try:
        loader(path)
    except Exception as exc:
        raise ValueError(
            f"El artefacto de entrenamiento '{path.name}' quedó corrupto tras guardarse "
            f"({exc}) — entrenamiento abortado, no se subió nada a MLflow."
        ) from exc


_NO_CYCLE_ID_WARNING = (
    "El CSV no tiene columna 'cycle_id': todas las filas se trataron como un único "
    "ciclo continuo. Si el archivo realmente contiene varios ciclos del horno, "
    "añade 'cycle_id' para que las ventanas no mezclen datos de ciclos distintos."
)


def _score_windows(ctx: dict, x_arr: np.ndarray, cycle_ids, window_timestamps) -> list[dict]:
    """Scale the windows, run the model and format one prediction (with XAI) per window."""
    x_scaled = postprocessing.scale_sequences(x_arr, ctx["scaler_x"])
    stats_scaled = postprocessing.compute_scaled_stats(x_arr, ctx["scaler_num"])
    scores = postprocessing.run_model(ctx["model"], x_scaled, stats_scaled)

    return postprocessing.format_batch_predictions(
        x_arr, scores, ctx["threshold"],
        postprocessing.XaiInputs(ctx["explainer"], ctx["xai_background"], x_scaled, stats_scaled),
        windows=(cycle_ids, window_timestamps),
    )


def _no_window_warning(has_cycle_id_column: bool) -> str:
    """Warning returned when the CSV has no valid SEQ_LENGTH-row window."""
    return (
        f"No se encontró ninguna ventana válida de {SEQ_LENGTH} registros "
        f"consecutivos. "
        + (
            "El CSV no tiene columna 'cycle_id': se trató como un único ciclo "
            "continuo — si contiene varios ciclos reales, añade 'cycle_id' para "
            "separarlos. "
            if not has_cycle_id_column
            else ""
        )
        + f"Se requieren al menos {SEQ_LENGTH} filas por ciclo."
    )


def _load_train_splits(data_path: str) -> dict:
    """Read the labeled CSV, split it by whole cycle_id and window each split.

    Returns ``{"train": (x_raw, y), "val": (x_raw, y), "test": (x_raw, y)}``.
    """
    with local_file_path(data_path) as local_path:
        df = pd.read_csv(local_path)
    df.columns = df.columns.str.lower()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"CSV falta columna objetivo '{TARGET_COLUMN}'. Requerida para entrenamiento."
        )

    # Split RAW rows by whole cycle_id (never by row/window count) — see
    # _vendor/preprocess.py::split_train_val_test_by_id. A window is a sliding
    # SEQ_LENGTH-row slice within a single cycle, so splitting whole cycles first
    # and windowing each split independently guarantees no window straddles two
    # splits, and reproduces the real repo's train_pct/val_pct/test_pct exactly.
    raw_splits, _ = split_train_val_test_by_id(
        df, id_col=ID_COLUMN, target_column=TARGET_COLUMN,
        val_size=TRAIN_EXTERNAL_VAL_PCT / 100.0, test_size=TRAIN_EXTERNAL_TEST_PCT / 100.0,
        normal_tokens=NORMAL_TOKENS,
    )

    x_train_raw, y_train, _, _ = preprocessing.prepare_batch_sequences(raw_splits["train"])
    x_val_raw, y_val, _, _ = preprocessing.prepare_batch_sequences(raw_splits["val"])
    x_test_raw, y_test, _, _ = preprocessing.prepare_batch_sequences(raw_splits["test"])
    n_train, n_val, n_test = x_train_raw.shape[0], x_val_raw.shape[0], x_test_raw.shape[0]
    if n_train + n_val + n_test < 10:
        raise ValueError(
            f"Muy pocas secuencias para entrenar: {n_train + n_val + n_test}. "
            f"Se necesitan al menos 10 ventanas de {SEQ_LENGTH} filas en total."
        )
    if n_train == 0 or n_val == 0 or n_test == 0:
        raise ValueError(
            f"El split train/val/test por cycle_id dejó una partición vacía "
            f"(train={n_train}, val={n_val}, test={n_test}) — sube un CSV con más "
            f"ciclos distintos."
        )
    return {
        "train": (x_train_raw, y_train),
        "val": (x_val_raw, y_val),
        "test": (x_test_raw, y_test),
    }


def _scale_sequences(splits: dict, scaler_cls) -> tuple[tuple, object]:
    """Fit the per-sensor scaler on the train windows and apply it to every split."""
    x_train_raw, x_val_raw, x_test_raw = (splits[k][0] for k in ("train", "val", "test"))
    n_feat = x_train_raw.shape[-1]

    scaler_x = scaler_cls()
    x_train = (
        scaler_x.fit_transform(x_train_raw.reshape(-1, n_feat))
        .reshape(x_train_raw.shape)
        .astype(np.float32)
    )
    x_val = (
        scaler_x.transform(x_val_raw.reshape(-1, n_feat))
        .reshape(x_val_raw.shape)
        .astype(np.float32)
    )
    x_test = (
        scaler_x.transform(x_test_raw.reshape(-1, n_feat))
        .reshape(x_test_raw.shape)
        .astype(np.float32)
    )
    return (x_train, x_val, x_test), scaler_x


def _scale_stats(splits: dict, scaler_cls) -> tuple[tuple, object]:
    """Fit the window-statistics scaler on train and apply it to every split."""
    df_stats_train, df_stats_val, df_stats_test = (
        stats_windows(splits[k][0], feature_names=SENSOR_COLUMNS, stats_creation=STATS_CREATION)
        for k in ("train", "val", "test")
    )
    scaler_num = scaler_cls()
    s_train = scaler_num.fit_transform(df_stats_train.values).astype(np.float32)
    s_val = scaler_num.transform(df_stats_val.values).astype(np.float32)
    s_test = scaler_num.transform(df_stats_test.values).astype(np.float32)
    return (s_train, s_val, s_test), scaler_num


def _scale_splits(splits: dict) -> tuple[dict, object, object]:
    """Scale sequences and window stats of every split (scalers fitted on train only).

    Returns ``({"train"|"val"|"test": (x_scaled, stats_scaled)}, scaler_x, scaler_num)``.
    """
    from sklearn.preprocessing import StandardScaler  # pylint: disable=import-outside-toplevel

    xs, scaler_x = _scale_sequences(splits, StandardScaler)
    ss, scaler_num = _scale_stats(splits, StandardScaler)
    scaled = {k: (xs[i], ss[i]) for i, k in enumerate(("train", "val", "test"))}
    return scaled, scaler_x, scaler_num


def _forward_step(model, criterion, x_batch, s_batch, y_batch):
    """Training loss of one batch (DNFLoss over the model's outputs)."""
    out = model(x_batch, s_batch)
    anomaly_score = torch.nan_to_num(
        out["anomaly_score"], nan=0.0, posinf=20.0, neginf=-20.0
    )
    rule_activations = out["rule_activations"]
    loss, _ = criterion(
        anomaly_score=anomaly_score,
        logit_anomaly_dl=out["logit_anomaly_dl"],
        logit_anomaly_fuzzy=out["logit_anomaly_fuzzy"],
        rule_activations=rule_activations,
        alpha_probs=out["alpha_probs"],
        y_anomaly=y_batch,
    )
    return loss


def _evaluate_val(model, x_val_t, s_val_t, y_val_arr):
    """Full forward pass over the validation split — returns the rich metric
    dict monitor_score() needs (anomaly_f1, fuzzy_f1, dead_rules_ratio,
    alpha_entropy_mean, rule_corr_mean), plus raw probabilities."""
    model.eval()
    with torch.no_grad():
        out_v = model(x_val_t, s_val_t)
        probs = torch.sigmoid(out_v["anomaly_score"]).view(-1).numpy()
        preds = (probs >= 0.5).astype(int)
        fuzzy_probs = torch.sigmoid(out_v["logit_anomaly_fuzzy"]).view(-1).numpy()
        fuzzy_preds = (fuzzy_probs >= 0.5).astype(int)
        rule_activations = out_v["rule_activations"]
        rule_activation_mean = rule_activations.mean(dim=0).numpy()
        dead_rules = int((rule_activation_mean < 1e-4).sum())
        dead_rules_ratio = float(dead_rules / max(len(rule_activation_mean), 1))
        val_out = {
            "anomaly_f1": f1_score(y_val_arr, preds, zero_division=0),
            "fuzzy_f1": f1_score(y_val_arr, fuzzy_preds, zero_division=0),
            "dead_rules_ratio": dead_rules_ratio,
            "alpha_entropy_mean": alpha_entropy_mean(out_v["alpha_probs"]),
            "rule_corr_mean": rule_corr_mean(rule_activations),
        }
    return val_out, probs


def _train_epoch(model, criterion, optimizer, train_t: tuple) -> None:
    """One pass over the training windows (``train_t`` = x, stats, y tensors)."""
    x_t, s_t, y_t = train_t
    n_train = x_t.shape[0]
    model.train()
    # shuffle_train=False in the real config — batches iterate in the same
    # fixed (unshuffled) order every epoch.
    for start in range(0, n_train, TRAIN_BATCH_SIZE):
        end = start + TRAIN_BATCH_SIZE
        optimizer.zero_grad(set_to_none=True)
        loss = _forward_step(model, criterion, x_t[start:end], s_t[start:end], y_t[start:end])
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_GRAD_CLIP)
        optimizer.step()


def _fit(model, criterion, train_t: tuple, val_t: tuple) -> dict:
    """Adam + CosineAnnealingLR with warmup and early stopping on monitor_score().

    ``train_t`` = (x, stats, y) tensors; ``val_t`` = (x, stats) tensors + int labels.
    Returns the best state_dict (or the last one if no epoch improved).
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_SCHEDULER_T_MAX, eta_min=TRAIN_SCHEDULER_ETA_MIN,
    )

    best_score = -np.inf
    best_state = None
    epochs_without_improve = 0

    for epoch in range(TRAIN_NUM_EPOCHS):
        _train_epoch(model, criterion, optimizer, train_t)

        val_out, _ = _evaluate_val(model, *val_t)
        score = monitor_score(val_out)
        scheduler.step()

        logger.info(
            "Epoch %d/%d — val_anomaly_f1=%.4f val_fuzzy_f1=%.4f monitor_score=%.4f lr=%.2e",
            epoch + 1, TRAIN_NUM_EPOCHS, val_out["anomaly_f1"], val_out["fuzzy_f1"],
            score, optimizer.param_groups[0]["lr"],
        )

        improved = score > (best_score + TRAIN_MIN_DELTA)
        if improved:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch < TRAIN_WARMUP_EPOCHS:
            continue
        epochs_without_improve = 0 if improved else epochs_without_improve + 1
        if epochs_without_improve >= TRAIN_PATIENCE:
            logger.info(
                "Early stopping en epoch %d (patience=%d).", epoch + 1, TRAIN_PATIENCE
            )
            break

    if best_state is None:
        best_state = model.state_dict()
    return best_state


def _train_and_calibrate(model_cfg: dict, splits: dict, scaled: dict) -> tuple:
    """Build and train the model, keep its best checkpoint and calibrate the threshold.

    Returns ``(model, best_state, calibrated_threshold)``.
    """
    model = model_loader.build_model(model_cfg)

    y_train = splits["train"][1]
    pos_weight = compute_class_weights(y_train)
    criterion = DNFLoss(anomaly_pos_weight=pos_weight, **DNF_LOSS_KWARGS)
    y_val_arr = splits["val"][1].astype(int)
    val_t = (torch.from_numpy(scaled["val"][0]), torch.from_numpy(scaled["val"][1]), y_val_arr)
    best_state = _fit(
        model,
        criterion,
        (
            torch.from_numpy(scaled["train"][0]),
            torch.from_numpy(scaled["train"][1]),
            torch.from_numpy(y_train.astype(np.float32)),
        ),
        val_t,
    )
    model.load_state_dict(best_state)

    # Threshold calibration — 101-point uniform grid search over [0, 1] on the
    # VALIDATION split (never test), matching the real repo's
    # optimize_threshold_by_f1 exactly so a threshold trained here is comparable
    # to one trained locally on the same data.
    _, val_probs = _evaluate_val(model, *val_t)
    threshold_metrics = optimize_threshold_by_f1(
        y_true=y_val_arr, prob_anomaly=val_probs, n_points=TRAIN_THRESHOLD_SEARCH_POINTS,
    )
    calibrated_threshold = float(threshold_metrics["best_threshold"])
    logger.info(
        "Calibrated decision_threshold=%.4f (val F1-optimal, 101-point grid)",
        calibrated_threshold,
    )
    return model, best_state, calibrated_threshold


def _test_metrics(model, test_xs: tuple, y_test, calibrated_threshold: float) -> dict:
    """Final evaluation on the held-out test split with the selected checkpoint/threshold."""
    x_test, s_test = test_xs
    # Final test evaluation — the ONLY time the test split is used, on the
    # checkpoint and threshold already selected above (never used to choose either).
    model.eval()
    with torch.no_grad():
        out_t = model(torch.from_numpy(x_test), torch.from_numpy(s_test))
        test_probs = torch.sigmoid(out_t["anomaly_score"]).view(-1).numpy()
    test_preds = (test_probs >= calibrated_threshold).astype(int)
    y_test_arr = y_test.astype(int)
    test_f1 = float(f1_score(y_test_arr, test_preds, zero_division=0))

    # F1 (Macro) / Recall (Macro) — modelo 43-44 audit, Fase 2. Unweighted mean across
    # BOTH classes (Normal=0, Fallo=1), computed on the same held-out test predictions
    # above. labels=[0, 1] is explicit so a class absent from a (small) test fold still
    # counts as 0 for that class instead of silently being dropped from the average.
    test_f1_macro = float(
        f1_score(y_test_arr, test_preds, labels=[0, 1], average="macro", zero_division=0)
    )
    test_recall_macro = float(
        recall_score(y_test_arr, test_preds, labels=[0, 1], average="macro", zero_division=0)
    )

    # accuracy/auc/precision/recall (binary, "Fallo" positive class) — modelo 43-44
    # audit, Fase 5 follow-up: stats(mlflow_run_id=...) has always reported these
    # under the anomaly_acc/anomaly_f1/anomaly_auc/anomaly_precision/anomaly_recall
    # keys, but train() never computed nor logged them, so a retrain's /stats could
    # never surface real values under those exact tiles — only decision_threshold
    # happened to work, by naming coincidence.
    test_auc = (
        float(roc_auc_score(y_test_arr, test_probs))
        if len(np.unique(y_test_arr)) > 1
        else float("nan")
    )
    return {
        "f1": test_f1,
        "f1_macro": test_f1_macro,
        "recall_macro": test_recall_macro,
        "accuracy": float(accuracy_score(y_test_arr, test_preds)),
        "precision": float(precision_score(y_test_arr, test_preds, zero_division=0)),
        "recall": float(recall_score(y_test_arr, test_preds, zero_division=0)),
        "auc": test_auc,
    }


def _rounded_metrics(test: dict, calibrated_threshold: float, counts: dict) -> dict:
    """Metric values reported by train() (TrainResponse fields and MLflow metrics)."""
    test_auc = test["auc"]
    return {
        "accuracy": round(test["accuracy"], 4),
        "fallo_auc": round(test_auc, 4) if not math.isnan(test_auc) else test_auc,
        "fallo_precision": round(test["precision"], 4),
        "fallo_recall": round(test["recall"], 4),
        "macro_f1": round(test["f1_macro"], 4),
        "macro_recall": round(test["recall_macro"], 4),
        "decision_threshold": round(calibrated_threshold, 4),
        **counts,
    }


def _train_response(reported: dict, counts: dict, new_run_id: str | None) -> TrainResponse:
    """Build the TrainResponse from the reported metrics and window counts."""
    split_val_pct = TRAIN_EXTERNAL_VAL_PCT
    split_test_pct = TRAIN_EXTERNAL_TEST_PCT
    split_train_pct = 100.0 - split_val_pct - split_test_pct

    return TrainResponse(
        detail="Entrenamiento completado",
        accuracy=reported["accuracy"],
        fallo_auc=reported["fallo_auc"],
        fallo_precision=reported["fallo_precision"],
        fallo_recall=reported["fallo_recall"],
        macro_f1=reported["macro_f1"],
        macro_recall=reported["macro_recall"],
        decision_threshold=reported["decision_threshold"],
        n_windows_train=int(counts["n_windows_train"]),
        n_windows_val=int(counts["n_windows_val"]),
        n_windows_test=int(counts["n_windows_test"]),
        n_windows_total=int(sum(counts.values())),
        split_train_pct=round(split_train_pct, 1),
        split_val_pct=round(split_val_pct, 1),
        split_test_pct=round(split_test_pct, 1),
        mlflow_run_id=new_run_id,
        upload_warning=None,
    )


class Ml43CerealsDnslAnomalyFaultDetectionPlugin(PluginRuntimeMixin, ModelPluginPort):  # pylint: disable=too-many-instance-attributes
    """Deep Neuro-Fuzzy (BiLSTM + fuzzy rules) anomaly detector for cereal ovens."""

    def __init__(self) -> None:
        self._model = None
        self._model_cfg: dict = DEFAULT_MODEL_CFG.copy()
        self._scaler_x = None
        self._scaler_num = None
        self._xai_background = None
        self._explainer = None
        self._threshold: float = DECISION_THRESHOLD
        self._total_latency_ms: float = 0.0

    def load(self) -> None:
        (
            self._model,
            self._model_cfg,
            self._scaler_x,
            self._scaler_num,
            self._xai_background,
            self._explainer,
        ) = model_loader.load_artifacts()
        logger.info("Ml43CerealsDnslAnomalyFaultDetectionPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        return self._model is not None

    def _record(self, latency_ms: float = 0.0) -> None:
        # modelo 43-44 audit: latency was already measured with perf_counter() (see
        # predict_batch) but only ever logged, never accumulated — stats() hardcoded
        # avg_latency_ms=None. Same _total_latency_ms/_predict_count pattern documented
        # in IMPLEMENTATION.md and already used by other plugins (e.g. ml25_wine_sulphites).
        self._predict_count += 1
        self._total_latency_ms += latency_ms
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _load_model_for_predict(self, mlflow_run_id: str) -> dict | None:
        """Resolve the model/scalers/explainer to use — user-trained (MLflow) or served."""
        if not mlflow_run_id:
            self._require_loaded()
            return None
        logger.info("Using user-trained model from MLflow run_id=%s", mlflow_run_id)
        loaded = download_user_model_from_mlflow(mlflow_run_id)
        model, model_cfg, scaler_x, scaler_num, xai_background, explainer, temp_dir = loaded
        # The trained weights and the trained decision_threshold must travel together — a
        # run's probabilities are only meaningful against the cutoff it was calibrated
        # with. Without this, predict_batch used to classify Fallo/No Fallo with the
        # served model's fixed 0.41 constant even when genuinely scoring with different,
        # user-trained weights.
        threshold = get_calibrated_threshold(mlflow_run_id, default=self._threshold)
        return {
            "model": model, "model_cfg": model_cfg, "scaler_x": scaler_x, "scaler_num": scaler_num,
            "xai_background": xai_background, "explainer": explainer, "temp_dir": temp_dir,
            "threshold": threshold,
        }

    def _served_ctx(self) -> dict:
        return {
            "model": self._model, "model_cfg": self._model_cfg,
            "scaler_x": self._scaler_x, "scaler_num": self._scaler_num,
            "xai_background": self._xai_background, "explainer": self._explainer, "temp_dir": None,
            "threshold": self._threshold,
        }

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictBatchResponse:
        """Not supported — ml43-cereals-dnsl-anomaly-fault-detection needs a real 180-row window.

        There is no meaningful single-snapshot prediction for this model (a
        synthetic repeated-snapshot window used to exist here but was removed:
        it never reflected real transient dynamics and could not be compared to
        genuine batch/CSV predictions). ``ModelPluginPort.predict_inline`` is
        abstract so this method must exist, but in practice it is unreachable
        through the HTTP API — ``PredictRequest`` only accepts ``mode="batch"``
        bodies, so FastAPI/Pydantic rejects anything else before this runs.
        """
        _ = features, model_key, threshold, mlflow_run_id
        raise InsufficientSensorWindowError(
            "ml43-cereals-dnsl-anomaly-fault-detection no soporta predicción inline (una sola "
            "lectura). Usa el modo batch con un CSV de al menos 180 registros consecutivos."
        )

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Score every valid 180-row window in a raw sensor CSV (one or more cycles)."""
        mlflow_ctx = self._load_model_for_predict(mlflow_run_id)
        ctx = mlflow_ctx or self._served_ctx()
        with removing_dir(mlflow_ctx["temp_dir"] if mlflow_ctx else None):
            t0 = time.perf_counter()
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)
            has_cycle_id_column = ID_COLUMN in df.columns.str.lower().tolist()

            x_arr, _, cycle_ids, window_timestamps = preprocessing.prepare_batch_sequences(df)
            if x_arr.shape[0] == 0:
                logger.warning("predict_batch — no valid windows found in %s", data_path)
                self._record((time.perf_counter() - t0) * 1000)
                return PredictBatchResponse(
                    model_id=MODEL_ID,
                    predictions=[],
                    output_path=None,
                    warning=_no_window_warning(has_cycle_id_column),
                )

            predictions = _score_windows(ctx, x_arr, cycle_ids, window_timestamps)

            latency_ms = (time.perf_counter() - t0) * 1000
            self._record(latency_ms)
            logger.info(
                "predict_batch done — %d windows, %d failures (threshold=%.2f), latency_ms=%.1f "
                "count=%d",
                len(predictions),
                sum(1 for p in predictions if p["predicted_anomaly_label"] == "Fallo"),
                ctx["threshold"],
                latency_ms,
                self._predict_count,
            )
            return PredictBatchResponse(
                model_id=MODEL_ID,
                predictions=predictions,
                output_path=None,
                warning=None if has_cycle_id_column else _NO_CYCLE_ID_WARNING,
            )

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Train a fresh model from a labeled CSV and upload it to MLflow.

        Does not replace the served checkpoint — pass the returned MLflow
        run_id back to /predict or /stats to use the newly trained model.

        Faithfully ports a43-44-neurofuzzy-anomalias-fallas's real training pipeline
        (config.yaml's training:/data_processing.external_data_split: sections) —
        modelo 43-44 audit, Fase 5. Earlier versions of this method used a much
        simplified stand-in (plain BCE loss, no class weighting, fixed 50 epochs, no
        LR schedule, a 70/15/15 window-count split) that trained a materially
        different, less-converged model — its F1-optimal decision_threshold was not
        comparable to the one the same CSV produces in the real repo (confirmed:
        0.77 here vs 0.49 there on an identical dataset). This version reproduces the
        real repo's split-by-cycle-id, DNFLoss, Adam+CosineAnnealingLR, warmup +
        early-stopping-on-a-composite-monitor-score, and 101-point threshold search,
        so results trained via this service are comparable to results trained
        locally on the same data.
        """
        logger.info(
            "Starting ml43-cereals-dnsl-anomaly-fault-detection training from %s", data_path
        )
        splits = _load_train_splits(data_path)
        counts = {
            "n_windows_train": splits["train"][0].shape[0],
            "n_windows_val": splits["val"][0].shape[0],
            "n_windows_test": splits["test"][0].shape[0],
        }
        scaled, scaler_x, scaler_num = _scale_splits(splits)

        model_cfg = DEFAULT_MODEL_CFG.copy()
        model, best_state, calibrated_threshold = _train_and_calibrate(model_cfg, splits, scaled)

        test = _test_metrics(model, scaled["test"], splits["test"][1], calibrated_threshold)
        logger.info(
            "Final test evaluation — test_f1=%.4f test_f1_macro=%.4f test_recall_macro=%.4f "
            "(n_train=%d, n_val=%d, n_test=%d)",
            test["f1"], test["f1_macro"], test["recall_macro"],
            counts["n_windows_train"], counts["n_windows_val"], counts["n_windows_test"],
        )

        # Key names unified with ml45_cereals_dnsl_critical_point_detection and with
        # both real training repos' own results.json (modelo 43-45 audit, metrics
        # unification) — these ARE the final display names now, so stats() no longer
        # needs a separate legacy-key alias map: it just copies these straight through.
        reported = _rounded_metrics(test, calibrated_threshold, counts)
        new_run_id = self._save_and_upload(
            {"model_state_dict": best_state, "model_cfg": model_cfg},
            {"scaler_x": scaler_x, "scaler_num": scaler_num},
            reported,
            mlflow_run_id,
        )

        return _train_response(reported, counts, new_run_id)

    def _save_and_upload(
        self, checkpoint: dict, scalers: dict, reported: dict, mlflow_run_id: str
    ) -> str:
        """Write and validate the retrained bundle, then upload it with its metrics to MLflow.

        A corrupt/empty local artifact raises ``ValueError`` before any upload; an upload
        failure becomes ``ModelPersistenceError``. Returns the run_id used.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="ml43_cereals_train_"))
        try:
            torch.save(checkpoint, temp_dir / MODEL_FILENAME)
            _validate_saved_artifact(
                temp_dir / MODEL_FILENAME, lambda p: torch.load(p, weights_only=False)
            )

            with open(temp_dir / SCALER_FILENAME, "wb") as f:
                pickle.dump(scalers, f)
            _validate_saved_artifact(
                temp_dir / SCALER_FILENAME, _load_pickle_file
            )

            if self._xai_background is not None:
                np.save(temp_dir / XAI_BACKGROUND_FILENAME, self._xai_background)
                _validate_saved_artifact(temp_dir / XAI_BACKGROUND_FILENAME, np.load)

            try:
                # MLflow rejects NaN/Inf metric values (auc is NaN when the test split
                # happens to contain only one class) — log only finite numbers so a
                # small/unbalanced retrain CSV can't turn "training succeeded, upload
                # failed" into an opaque MLflow client error.
                loggable_metrics = {
                    k: v for k, v in reported.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
                }
                new_run_id = upload_artifacts_to_mlflow(
                    str(temp_dir), mlflow_run_id=mlflow_run_id, metrics=loggable_metrics,
                )
                logger.info("Training complete. MLflow run_id=%s", new_run_id)
            except Exception as exc:
                logger.error(
                    "MLflow artifact upload failed (model trained but not persisted): %s", exc
                )
                raise ModelPersistenceError(
                    f"El modelo reentrenado no se ha podido guardar en MLflow: {exc}"
                ) from exc
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return new_run_id

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and test-split metrics."""
        inputs = [
            InputField(name=name, type="float", description=desc)
            for name, desc in [
                ("temp_zona1", "Temperatura zona 1 (°C)"),
                ("temp_zona2", "Temperatura zona 2 (°C)"),
                ("temp_zona3", "Temperatura zona 3 (°C)"),
                ("temp_salida_gases", "Temperatura salida de gases (°C)"),
                ("presion_camara", "Presión de cámara (Pa)"),
                ("presion_ventilacion", "Presión de ventilación (Pa)"),
                ("potencia_kw", "Potencia eléctrica (kW)"),
                ("flujo_gas", "Flujo de gas (m³/h)"),
                ("humedad_relativa", "Humedad relativa (%)"),
                ("temp_ambiente", "Temperatura ambiente (°C)"),
                ("setpoint_temp", "Temperatura de consigna (°C)"),
                ("posicion_valvula", "Posición de válvula (%)"),
                ("velocidad_ventilador", "Velocidad del ventilador (RPM)"),
            ]
        ] + [
            InputField(
                name="timestamp", type="datetime", description="Marca temporal de la medida"
            ),
            InputField(
                name="cycle_id", type="string", description="Identificador del ciclo de horneado"
            ),
        ]
        outputs = [
            OutputField(
                name="predicted_anomaly_label", type="string", description="'Fallo' o 'No Fallo'"
            ),
            OutputField(
                name="anomaly_probability",
                type="float",
                description="Probabilidad de anomalía [0, 1]",
            ),
            OutputField(
                name="decision_threshold", type="float", description="Umbral de decisión utilizado"
            ),
            OutputField(
                name="cycle_id",
                type="string",
                description="Identificador del ciclo asociado a la ventana",
            ),
            OutputField(
                name="window_index", type="int", description="Índice de ventana temporal (1..N)"
            ),
            OutputField(
                name="timestamp_init",
                type="string",
                description="Timestamp de la primera lectura de la ventana",
            ),
            OutputField(
                name="timestamp_end",
                type="string",
                description="Timestamp de la última lectura de la ventana",
            ),
            OutputField(
                name="Estado del sistema",
                type="string",
                description=(
                    "Estado interpretativo global de la ventana (Normal / Normal con señales / "
                    "Alerta no confirmada / Anomalía confirmada)"
                ),
            ),
            OutputField(
                name="Subsistemas con alteraciones",
                type="string",
                description="Subsistemas del horno identificados con comportamiento alterado",
            ),
            OutputField(
                name="Variables con alteraciones",
                type="string",
                description="Variables/sensores concretos que motivan el estado interpretativo",
            ),
            OutputField(
                name="Acciones correctivas recomendadas",
                type="string",
                description="Acciones sugeridas para corregir la alteración detectada",
            ),
        ]

        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Detección de fallos en hornos cerealistas y generación de acciones "
                "correctivas (XAI). Modelo híbrido Deep Neuro-Fuzzy (BiLSTM + reglas "
                f"fuzzy) que clasifica ventanas temporales de {SEQ_LENGTH} medidas del "
                "horno como Normal o Fallo, con explicabilidad SHAP + reglas fuzzy y "
                "generación de acciones correctivas."
            ),
            task_type="binary_classification",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics={**TEST_METRICS, "decision_threshold": self._threshold},
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=(
                    round(self._total_latency_ms / self._predict_count, 1)
                    if self._predict_count > 0 else None
                ),
            ),
        )
        if mlflow_run_id:
            try:
                tracker = BaseMLflowTracker(mlflow_run_id)
                mlflow_metrics = tracker.get_metrics()
                # train() now logs metrics to MLflow under exactly the unified display key
                # names (accuracy, fallo_auc, fallo_precision, fallo_recall, macro_f1,
                # macro_recall, decision_threshold, n_windows_train/val/test) — modelo 43-45
                # audit, metrics unification. So this OVERWRITES base.metrics key-by-key with
                # the real per-run values instead of adding differently-named keys alongside
                # the base/reference ones (which used to leave both an old and a new key
                # showing the same number under two different labels). Only known keys are
                # copied — unlike the previous generic "copy every numeric key" loop, this
                # can't be polluted by unrelated metrics a future MLflow run might log.
                for key in UNIFIED_METRIC_KEYS:
                    value = mlflow_metrics.get(key)
                    if isinstance(value, (int, float)):
                        base.metrics[key] = value
                for key in ("n_windows_train", "n_windows_val", "n_windows_test"):
                    value = mlflow_metrics.get(key)
                    if isinstance(value, (int, float)):
                        base.metrics[key] = value
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Could not fetch MLflow stats for run_id=%s: %s", mlflow_run_id, exc)
        return base
