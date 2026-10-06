from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import copy

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml45_cereals_dnsl_critical_point_detection._vendor.loss import DNFLoss
from app.plugins.ml45_cereals_dnsl_critical_point_detection._vendor.model import (
    ParallelDeepNeuroFuzzyModel,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection._vendor.preprocess import (
    split_train_val_test_by_id,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection._vendor.train_metrics import (
    alpha_entropy_mean,
    compute_class_weights,
    optimize_threshold_by_f1,
    rule_corr_mean,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.constants import (
    DNF_LOSS_KWARGS,
    FRAMEWORK,
    ID_COLUMN,
    MODEL_FILENAME,
    MODEL_ID,
    NORMAL_TOKENS,
    SCALER_FILENAME,
    SENSOR_COLUMNS,
    SEQUENCE_LENGTH,
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
from app.plugins.ml45_cereals_dnsl_critical_point_detection.mlflow_utils import (
    download_user_model_from_mlflow,
    upload_artifacts_to_mlflow,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.model_loader import (
    _store,
    load_artifacts,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.postprocessing import (
    build_explainer,
    explain_window,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.postprocessing import (
    build_predictions,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.preprocessing import (
    build_raw_windows_from_dataframe,
    build_windows_from_csv,
    build_windows_from_sensor_arrays,
)
from app.plugins.ml45_cereals_dnsl_critical_point_detection.train_dto import TrainResponse

logger = logging.getLogger(__name__)


def _validate_saved_artifact(path: Path, loader) -> None:
    """Raise ValueError if a just-saved training artifact is missing, empty, or unreadable.

    torch.save/joblib.dump/np.save have no return value to check, so a process killed or a
    disk filled mid-write can silently leave a 0-byte or truncated file on disk — this is
    what previously let a fine-tuned model reach MLflow (and the platform's model listing)
    as a 0-byte artifact with no visible error. Same pattern as ml43_cereals_dnsl_anomaly_fault_detection's
    plugin.py::_validate_saved_artifact.
    """
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(
            f"El artefacto de entrenamiento '{path.name}' se guardó vacío (0 bytes) o no "
            "se generó — entrenamiento abortado, no se subió nada a MLflow."
        )
    try:
        loader(path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise ValueError(
            f"El artefacto de entrenamiento '{path.name}' quedó corrupto tras guardarse "
            f"({exc}) — entrenamiento abortado, no se subió nada a MLflow."
        ) from exc


class Ml45CerealsDnslCriticalPointDetectionPlugin(ModelPluginPort):
    """Deep Neuro-Fuzzy (LSTM+attention || fuzzy TSK rules) grain-dryer fault detector.

    Predicts per-window (240 consecutive readings) binary anomaly, then classifies into
    Normal/Vigilancia/Criticidad detectada via the built-in PCC (Puntos Críticos de Control)
    monitor — SHAP + fuzzy-rule explainability ported inline (not delegated to the external
    explainability microservice), per this model's own functional design.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_cfg: dict | None = None
        self._scaler_x = None
        self._scaler_num = None
        self._xai_background = None
        self._threshold: float = 0.73
        self._predict_count: int = 0
        self._last_predict_at: str | None = None
        self._total_latency_ms: float = 0.0

    def load(self) -> None:
        bucket = os.environ.get("STORAGE_BUCKET")
        if bucket:
            _store.download_all_if_needed()

        model, model_cfg, scaler_x, scaler_num, xai_background, threshold = load_artifacts()
        self._model = model
        self._model_cfg = model_cfg
        self._scaler_x = scaler_x
        self._scaler_num = scaler_num
        self._xai_background = xai_background
        self._threshold = threshold
        logger.info("m45 plugin loaded: %s (threshold=%.3f)", MODEL_ID, threshold)

    def is_loaded(self) -> bool:
        return self._model is not None

    def _require_loaded(self) -> None:
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record(self, latency_ms: float = 0.0) -> None:
        self._predict_count += 1
        self._total_latency_ms += latency_ms
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _load_model_for_predict(self, mlflow_run_id: str):
        if not mlflow_run_id:
            self._require_loaded()
            return None
        logger.info("Using user-trained model from MLflow run_id=%s", mlflow_run_id)
        loaded = download_user_model_from_mlflow(mlflow_run_id)
        if loaded is None:
            logger.warning("MLflow download failed for %s, falling back to standard model", mlflow_run_id)
            self._require_loaded()
            return None
        model, model_cfg, scaler_x, scaler_num, xai_background, threshold, temp_dir = loaded
        return {
            "model": model,
            "model_cfg": model_cfg,
            "scaler_x": scaler_x,
            "scaler_num": scaler_num,
            "xai_background": xai_background if xai_background is not None else self._xai_background,
            "threshold": threshold,
            "temp_dir": temp_dir,
        }

    def predict_inline(
        self,
        *,
        data_path: str | None = None,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        _ = model_key, threshold
        mlflow_ctx = self._load_model_for_predict(mlflow_run_id)
        ctx = mlflow_ctx or {
            "model": self._model,
            "model_cfg": self._model_cfg,
            "scaler_x": self._scaler_x,
            "scaler_num": self._scaler_num,
            "xai_background": self._xai_background,
            "threshold": self._threshold,
            "temp_dir": None,
        }
        try:
            t0 = time.perf_counter()
            if data_path:
                with local_file_path(data_path) as local_path:
                    sequences, stats, stats_cols, ts_windows, entity_ids, _ = build_windows_from_csv(
                        local_path, ctx["scaler_x"], ctx["scaler_num"], require_target=False,
                    )
            else:
                sensor_arrays = {col: features[col] for col in SENSOR_COLUMNS}
                sequences, stats, stats_cols, ts_windows, entity_ids, _ = build_windows_from_sensor_arrays(
                    sensor_arrays,
                    features["timestamp"],
                    features.get("cycle_id"),
                    ctx["scaler_x"],
                    ctx["scaler_num"],
                )

            if len(sequences) > 1:
                logger.warning(
                    "predict_inline received enough rows for %d windows; only the first "
                    "window is used. Use predict_batch to get a prediction per window.",
                    len(sequences),
                )

            explainer = build_explainer(ctx["model"], stats_cols, ctx["model_cfg"])
            row = explain_window(
                explainer,
                x_window=sequences[0],
                s_stats=stats[0],
                xai_background=ctx["xai_background"],
                threshold=ctx["threshold"],
            )
            self._record((time.perf_counter() - t0) * 1000)
            return PredictInlineResponse(
                model_id=MODEL_ID,
                window_index=1,
                timestamp_init=str(ts_windows[0][0]),
                timestamp_end=str(ts_windows[0][1]),
                **row,
            )
        finally:
            if mlflow_ctx and mlflow_ctx["temp_dir"]:
                shutil.rmtree(mlflow_ctx["temp_dir"], ignore_errors=True)

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        mlflow_ctx = self._load_model_for_predict(mlflow_run_id)
        ctx = mlflow_ctx or {
            "model": self._model,
            "model_cfg": self._model_cfg,
            "scaler_x": self._scaler_x,
            "scaler_num": self._scaler_num,
            "xai_background": self._xai_background,
            "threshold": self._threshold,
            "temp_dir": None,
        }
        try:
            t0 = time.perf_counter()
            with local_file_path(data_path) as local_path:
                sequences, stats, stats_cols, ts_windows, entity_ids, _ = build_windows_from_csv(
                    local_path, ctx["scaler_x"], ctx["scaler_num"], require_target=False,
                )

            predictions = build_predictions(
                ctx["model"],
                sequences,
                stats,
                stats_cols,
                ts_windows,
                entity_ids,
                ctx["xai_background"],
                ctx["threshold"],
                ctx["model_cfg"],
            )
            self._record((time.perf_counter() - t0) * 1000)
            return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)
        finally:
            if mlflow_ctx and mlflow_ctx["temp_dir"]:
                shutil.rmtree(mlflow_ctx["temp_dir"], ignore_errors=True)

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Train a fresh model from a labeled CSV and upload it to MLflow.

        Does not replace the served checkpoint — pass the returned mlflow_run_id back to
        /predict or /stats to use the newly trained model. Reusing the plugin's own
        in-memory model for later predictions would only work in a single-process,
        never-restarted deployment: any other worker/replica, or this same one after a
        restart, would still have the original checkpoint in memory, and load()'s S3 sync
        (ArtifactStore, which only ever mirrors the "fixed" original artifacts) is not
        aware of a retrain at all — so predictions would silently keep coming from the
        original model. Persisting to MLflow and requiring mlflow_run_id on predict/stats
        is the one path that is correct regardless of how many replicas/workers are
        serving this model.

        Faithfully ports a45-dnsl-cereals-deteccion-puntos-criticos's real training
        pipeline (config.yaml's training:/data_processing.external_data_split:
        sections) — modelo 43-44-45 audit, Fase 5. Earlier versions of this method
        fine-tuned the served checkpoint's weights for a fixed 30 epochs with no
        threshold recalibration — a simplification invented for this service; the real
        repo has no "fine-tune" concept at all, only scripts/train.py's full
        from-scratch pipeline (confirmed near byte-identical to ml43_cereals_dnsl_anomaly_fault_detection's own
        trainer.py/metrics.py). This version reproduces the real repo's
        split-by-cycle-id, DNFLoss, Adam+CosineAnnealingLR, warmup +
        early-stopping-on-a-composite-monitor-score, and 101-point threshold search, so
        results trained via this service are comparable to results trained locally on
        the same data.
        """
        self._require_loaded()
        logger.info("Starting ml45 training from %s", data_path)
        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)
        df.columns = df.columns.str.lower()

        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"CSV falta columna objetivo '{TARGET_COLUMN}'. Requerida para entrenamiento.")

        # Split RAW rows by whole cycle_id (never by row/window count) — see
        # _vendor/preprocess.py::split_train_val_test_by_id. A window is a sliding
        # SEQUENCE_LENGTH-row slice within a single cycle, so splitting whole cycles
        # first and windowing each split independently guarantees no window straddles
        # two splits, and reproduces the real repo's train_pct/val_pct/test_pct exactly.
        raw_splits, _ = split_train_val_test_by_id(
            df, id_col=ID_COLUMN, target_column=TARGET_COLUMN,
            val_size=TRAIN_EXTERNAL_VAL_PCT / 100.0, test_size=TRAIN_EXTERNAL_TEST_PCT / 100.0,
            normal_tokens=NORMAL_TOKENS,
        )

        x_train_raw, stats_train_df, _, _, _, y_train = build_raw_windows_from_dataframe(
            raw_splits["train"], require_target=True,
        )
        x_val_raw, stats_val_df, _, _, _, y_val = build_raw_windows_from_dataframe(
            raw_splits["val"], require_target=True,
        )
        x_test_raw, stats_test_df, _, _, _, y_test = build_raw_windows_from_dataframe(
            raw_splits["test"], require_target=True,
        )
        n_train, n_val, n_test = x_train_raw.shape[0], x_val_raw.shape[0], x_test_raw.shape[0]
        if n_train + n_val + n_test < 10:
            raise ValueError(
                f"Muy pocas secuencias para entrenar: {n_train + n_val + n_test}. "
                f"Se necesitan al menos 10 ventanas de {SEQUENCE_LENGTH} filas en total."
            )
        if n_train == 0 or n_val == 0 or n_test == 0:
            raise ValueError(
                f"El split train/val/test por cycle_id dejó una partición vacía "
                f"(train={n_train}, val={n_val}, test={n_test}) — sube un CSV con más "
                f"ciclos distintos."
            )
        n_windows = n_train + n_val + n_test
        n_feat = x_train_raw.shape[-1]

        scaler_x = StandardScaler()
        x_train = scaler_x.fit_transform(x_train_raw.reshape(-1, n_feat)).reshape(x_train_raw.shape).astype(np.float32)
        x_val = scaler_x.transform(x_val_raw.reshape(-1, n_feat)).reshape(x_val_raw.shape).astype(np.float32)
        x_test = scaler_x.transform(x_test_raw.reshape(-1, n_feat)).reshape(x_test_raw.shape).astype(np.float32)

        scaler_num = StandardScaler()
        s_train = scaler_num.fit_transform(stats_train_df.values).astype(np.float32)
        s_val = scaler_num.transform(stats_val_df.values).astype(np.float32)
        s_test = scaler_num.transform(stats_test_df.values).astype(np.float32)

        model_cfg = copy.deepcopy(self._model_cfg)
        model = ParallelDeepNeuroFuzzyModel(model_cfg)

        pos_weight = compute_class_weights(y_train)
        criterion = DNFLoss(anomaly_pos_weight=pos_weight, **DNF_LOSS_KWARGS)
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TRAIN_SCHEDULER_T_MAX, eta_min=TRAIN_SCHEDULER_ETA_MIN,
        )

        x_t = torch.from_numpy(x_train)
        s_t = torch.from_numpy(s_train)
        y_t = torch.from_numpy(y_train.astype(np.float32))
        x_val_t = torch.from_numpy(x_val)
        s_val_t = torch.from_numpy(s_val)
        y_val_arr = y_val.astype(int)

        def _forward_step(x_batch, s_batch, y_batch):
            out = model(x_batch, s_batch)
            anomaly_score = torch.nan_to_num(out["anomaly_score"], nan=0.0, posinf=20.0, neginf=-20.0)
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

        def _evaluate_val():
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

        best_score = -np.inf
        best_state = None
        epochs_without_improve = 0
        epochs_run = 0

        for epoch in range(TRAIN_NUM_EPOCHS):
            model.train()
            # shuffle_train=False in the real config — batches iterate in the same
            # fixed (unshuffled) order every epoch.
            for start in range(0, n_train, TRAIN_BATCH_SIZE):
                end = start + TRAIN_BATCH_SIZE
                optimizer.zero_grad(set_to_none=True)
                loss = _forward_step(x_t[start:end], s_t[start:end], y_t[start:end])
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_GRAD_CLIP)
                optimizer.step()

            epochs_run = epoch + 1
            val_out, _ = _evaluate_val()
            score = monitor_score(val_out)
            scheduler.step()

            logger.info(
                "Epoch %d/%d — val_anomaly_f1=%.4f val_fuzzy_f1=%.4f monitor_score=%.4f lr=%.2e",
                epoch + 1, TRAIN_NUM_EPOCHS, val_out["anomaly_f1"], val_out["fuzzy_f1"],
                score, optimizer.param_groups[0]["lr"],
            )

            improved = score > (best_score + TRAIN_MIN_DELTA)
            if epoch < TRAIN_WARMUP_EPOCHS:
                if improved:
                    best_score = score
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                if improved:
                    best_score = score
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1
                if epochs_without_improve >= TRAIN_PATIENCE:
                    logger.info("Early stopping en epoch %d (patience=%d).", epoch + 1, TRAIN_PATIENCE)
                    break

        if best_state is None:
            best_state = model.state_dict()
        model.load_state_dict(best_state)

        # Threshold calibration — 101-point uniform grid search over [0, 1] on the
        # VALIDATION split (never test), matching the real repo's
        # optimize_threshold_by_f1 exactly so a threshold trained here is comparable
        # to one trained locally on the same data.
        _, val_probs = _evaluate_val()
        threshold_metrics = optimize_threshold_by_f1(
            y_true=y_val_arr, prob_anomaly=val_probs, n_points=TRAIN_THRESHOLD_SEARCH_POINTS,
        )
        calibrated_threshold = float(threshold_metrics["best_threshold"])
        logger.info("Calibrated decision_threshold=%.4f (val F1-optimal, 101-point grid)", calibrated_threshold)
        # download_user_model_from_mlflow() reads the per-run threshold from exactly
        # this key (model_cfg["training_kwargs"]["threshold"]) — this is the ONLY place
        # that value is set; without it, every retrain's checkpoint kept propagating
        # the served model's original threshold, which is the bug this whole rewrite
        # exists to fix.
        model_cfg.setdefault("training_kwargs", {})["threshold"] = calibrated_threshold

        # Final test evaluation — the ONLY time the test split is used, on the
        # checkpoint and threshold already selected above (never used to choose either).
        model.eval()
        with torch.no_grad():
            out_t = model(torch.from_numpy(x_test), torch.from_numpy(s_test))
            test_probs = torch.sigmoid(out_t["anomaly_score"]).view(-1).numpy()
        test_preds = (test_probs >= calibrated_threshold).astype(int)
        y_test_arr = y_test.astype(int)

        test_accuracy = float(accuracy_score(y_test_arr, test_preds))
        test_f1 = float(f1_score(y_test_arr, test_preds, zero_division=0))
        test_precision = float(precision_score(y_test_arr, test_preds, zero_division=0))
        test_recall = float(recall_score(y_test_arr, test_preds, zero_division=0))
        test_f1_macro = float(f1_score(y_test_arr, test_preds, labels=[0, 1], average="macro", zero_division=0))
        test_recall_macro = float(recall_score(y_test_arr, test_preds, labels=[0, 1], average="macro", zero_division=0))
        test_auc = (
            float(roc_auc_score(y_test_arr, test_probs)) if len(np.unique(y_test_arr)) > 1 else float("nan")
        )

        logger.info(
            "Final test evaluation — test_f1=%.4f test_f1_macro=%.4f test_recall_macro=%.4f "
            "(n_train=%d, n_val=%d, n_test=%d)",
            test_f1, test_f1_macro, test_recall_macro, n_train, n_val, n_test,
        )

        temp_dir = Path(tempfile.mkdtemp(prefix="m45_train_"))
        upload_warning = None
        new_run_id: str | None = None
        try:
            checkpoint = {"model_state_dict": best_state, "model_cfg": model_cfg}
            torch.save(checkpoint, temp_dir / MODEL_FILENAME)
            _validate_saved_artifact(temp_dir / MODEL_FILENAME, lambda p: torch.load(p, weights_only=False))

            # download_user_model_from_mlflow() requires the scaler (and, if present, the XAI
            # background) alongside the model checkpoint in the same MLflow artifact path —
            # previously only the checkpoint was uploaded, so a predict call that DID pass a
            # freshly-trained mlflow_run_id would still fail (joblib.load on a scaler.pkl that
            # was never uploaded).
            joblib.dump({"scaler_x": scaler_x, "scaler_num": scaler_num}, temp_dir / SCALER_FILENAME)
            _validate_saved_artifact(temp_dir / SCALER_FILENAME, joblib.load)

            if self._xai_background is not None:
                np.save(temp_dir / XAI_BACKGROUND_FILENAME, self._xai_background)
                _validate_saved_artifact(temp_dir / XAI_BACKGROUND_FILENAME, np.load)

            try:
                # Key names unified with ml43_cereals_dnsl_anomaly_fault_detection and with both real training
                # repos' own results.json (modelo 43-45 audit, metrics unification) — these
                # ARE the final display names now, so stats() no longer needs a separate
                # legacy-key alias map: it just copies these straight through. Also fixes the
                # "Price fluctuation classifier" card bug (ws-models.js's customEvalHtml
                # duck-types on bare accuracy+f1 keys, which TrainResponse used to expose).
                loggable_metrics = {
                    "accuracy": test_accuracy,
                    "fallo_auc": test_auc,
                    "fallo_precision": test_precision,
                    "fallo_recall": test_recall,
                    "macro_f1": test_f1_macro,
                    "macro_recall": test_recall_macro,
                    "decision_threshold": round(calibrated_threshold, 4),
                    "n_windows_train": n_train, "n_windows_val": n_val, "n_windows_test": n_test,
                }
                # MLflow rejects NaN/Inf metric values (auc is NaN when the held-out eval
                # split happens to contain only one class) — log only finite numbers so a
                # small/unbalanced retrain CSV can't turn "training succeeded, upload failed"
                # into an opaque MLflow client error.
                loggable_metrics = {
                    k: v for k, v in loggable_metrics.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
                }
                new_run_id = upload_artifacts_to_mlflow(
                    str(temp_dir), mlflow_run_id=mlflow_run_id, metrics=loggable_metrics,
                )
                logger.info("Training complete. MLflow run_id=%s", new_run_id)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("MLflow artifact upload failed (model trained but not persisted): %s", exc)
                upload_warning = f"Entrenamiento completado, pero falló la subida a MLflow: {exc}"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        split_val_pct = TRAIN_EXTERNAL_VAL_PCT
        split_test_pct = TRAIN_EXTERNAL_TEST_PCT
        split_train_pct = 100.0 - split_val_pct - split_test_pct

        return TrainResponse(
            detail="Entrenamiento completado",
            accuracy=round(test_accuracy, 4),
            fallo_auc=round(test_auc, 4) if test_auc == test_auc else test_auc,
            fallo_precision=round(test_precision, 4),
            fallo_recall=round(test_recall, 4),
            macro_f1=round(test_f1_macro, 4),
            macro_recall=round(test_recall_macro, 4),
            decision_threshold=round(calibrated_threshold, 4),
            n_windows_train=int(n_train),
            n_windows_val=int(n_val),
            n_windows_test=int(n_test),
            n_windows_total=int(n_windows),
            split_train_pct=round(split_train_pct, 1),
            split_val_pct=round(split_val_pct, 1),
            split_test_pct=round(split_test_pct, 1),
            n_epochs=epochs_run,
            mlflow_run_id=new_run_id,
            upload_warning=upload_warning,
        )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        inputs = [
            InputField(name=col, type="float", description=f"Sensor {col} time-series (per row)")
            for col in SENSOR_COLUMNS
        ] + [
            InputField(name="timestamp", type="datetime", description="Marca temporal de la lectura"),
            InputField(name="cycle_id", type="string", description="Identificador del ciclo/entidad de secado"),
        ]
        outputs = [
            OutputField(name="predicted_anomaly_label", type="string", description="'Fallo' o 'No Fallo'"),
            OutputField(name="anomaly_probability", type="float", description="Probabilidad de anomalía fusionada (fuzzy + LSTM), [0, 1]"),
            OutputField(name="decision_threshold", type="float", description="Umbral de decisión utilizado"),
            OutputField(name="window_index", type="int", description="Índice de ventana temporal (1..N)"),
            OutputField(name="cycle_id", type="string", description="Identificador del ciclo asociado a la ventana (cuando el CSV lo incluye)"),
            OutputField(name="timestamp_init", type="string", description="Timestamp de la primera lectura de la ventana"),
            OutputField(name="timestamp_end", type="string", description="Timestamp de la última lectura de la ventana"),
            OutputField(name="Estado interpretativo", type="string", description="Normal | Vigilancia | Criticidad detectada"),
            OutputField(name="Evidencia", type="string", description="Motivo textual del estado interpretativo (PCC más relevante detectado)"),
            OutputField(name="Recomendacion", type="string", description="Acción recomendada para el estado interpretativo de la ventana"),
        ]

        # Base/fixed-model reference metrics — replaced below, key by key, with the real
        # per-run metrics logged by train() when the caller supplies the mlflow_run_id of a
        # retrain (see train()'s upload_artifacts_to_mlflow call).
        metrics = {**TEST_METRICS, "decision_threshold": self._threshold}

        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Detección de anomalías en secadoras de grano de flujo mixto mediante un modelo "
                "híbrido Deep Neuro-Fuzzy (LSTM+atención || reglas fuzzy TSK), con capa XAI "
                "(SHAP + reglas fuzzy) y monitor de Puntos Críticos de Control (PCC)."
            ),
            task_type="classification_binary_timeseries_windowed",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics=metrics,
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
