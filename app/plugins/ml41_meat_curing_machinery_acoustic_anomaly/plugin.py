"""Ml41MeatCuringMachineryAcousticAnomalyPlugin — Audio-MAE (ViT-Tiny) acoustic anomaly
detection for curing-chamber machinery (fan/pump/slider/valve), MIMII-based.

See inbox/a41/manifest.yaml for the full provenance of every design decision referenced
in comments below (architecture per machine, thresholds, golden cases, known_issues).
"""
from __future__ import annotations

import csv
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from app.application.dto.stats_dto import InputField, OutputField, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import (
    InvalidAudioError,
    ModelNotLoadedError,
    UnsupportedMachineConfigurationError,
    UserModelUnavailableError,
)
from app.domain.services.mlflow_artifacts import (
    add_mlflow_run_to_stats,
    removing_dir,
    require_train_tracker,
    upload_dir_to_mlflow,
)
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.infrastructure.archive import safe_extract_zip
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.constants import (
    ARTIFACT_FOLDER_NAME,
    ARCHITECTURE_BY_MACHINE,
    CHECKPOINT_FILENAME,
    MACHINE_IDS,
    MACHINES,
    MAHA_STATS_FILENAME,
    METRICS_REPORTED,
    MODEL_ID,
    SNRS,
    TRAINING_COMMON,
    TRAINING_HYPERPARAMS_BY_MACHINE,
    TRAINING_SPLIT_RATIO,
    TRAINING_SPLIT_SEED,
    VERSION,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.inference import (
    fit_mahalanobis,
    get_cls_embedding_single,
    mahalanobis_score,
    run_inference_single,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.mlflow_utils import (
    download_user_model_from_mlflow,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.model_loader import (
    CheckpointCache,
    LoadedCombination,
    artifacts_available,
    build_model,
    ensure_artifacts_downloaded,
    _safe_device,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.postprocessing import (
    build_inline_result,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.preprocessing import (
    audio_base64_to_logmel,
    normalize_logmel,
    wav_to_logmel,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.train_dto import (
    CombinationTrainMetrics,
    TrainResponse,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.trainer import (
    evaluate_against_abnormal,
    fine_tune,
)

if TYPE_CHECKING:
    from app.domain.services.mlflow_tracker import BaseMLflowTracker

logger = logging.getLogger(__name__)


@dataclass
class _FittedCombination:
    """A combination freshly trained with the original procedure, before evaluation/staging."""

    model: Any
    device: torch.device
    norm: tuple[float, float]  # (norm_mean, norm_std) of the train split
    train_data: np.ndarray
    val_data: np.ndarray
    best_val_loss: float
    maha: tuple  # (maha_mean, maha_inv_cov, maha_pca)


def _split_normal_clips(combo: tuple, hyperparams: dict) -> tuple:
    """Load the normal clips of ``combo`` and split/normalize them as the original.

    preprocess.py::process_machine_id: RandomState(split_seed=42) permutation, first
    int(0.8 * n) clips to train, the rest to val. Normalization stats come from the train
    split (MIMIIDataset). Returns ``(train_data, val_data, (norm_mean, norm_std))``.
    """
    machine, machine_id, snr, combo_root = combo
    normal_wavs = sorted((combo_root / "normal").glob("*.wav"))
    if not normal_wavs:
        raise ValueError(f"No .wav files under {combo_root / 'normal'}")

    specs = np.stack([wav_to_logmel(str(p)) for p in normal_wavs], axis=0)  # (N,1,mels,frames)

    # pylint: disable-next=no-member  # legacy RandomState, as the original
    idx = np.random.RandomState(TRAINING_SPLIT_SEED).permutation(len(specs))
    cut = int(TRAINING_SPLIT_RATIO * len(specs))
    train_raw, val_raw = specs[idx[:cut]], specs[idx[cut:]]
    if len(val_raw) == 0 or len(train_raw) < hyperparams["batch_size"]:
        raise ValueError(
            f"{machine}/{machine_id}/{snr}: {len(specs)} clips normales no bastan. "
            f"El split 80/20 deja {len(train_raw)} de entrenamiento y {len(val_raw)} de "
            "validación, y el original "
            f"descarta el último lote incompleto (drop_last): hacen falta al menos "
            f"{hyperparams['batch_size']} clips de entrenamiento y 1 de validación."
        )

    norm_mean, norm_std = float(train_raw.mean()), float(train_raw.std())
    return (
        normalize_logmel(train_raw, norm_mean, norm_std),
        normalize_logmel(val_raw, norm_mean, norm_std),
        (norm_mean, norm_std),
    )


def _fit_maha(model, train_data: np.ndarray, device, machine: str) -> tuple:
    """Recalibrate Mahalanobis stats on the (fine-tuned) normal training embeddings."""
    model.eval()
    embeddings = []
    for i in range(len(train_data)):
        spec = torch.from_numpy(train_data[i:i + 1]).float()
        embeddings.append(get_cls_embedding_single(model, spec, device))
    embeddings = np.stack(embeddings, axis=0)
    pca_components_n = ARCHITECTURE_BY_MACHINE[machine]["pca_components"]
    return fit_mahalanobis(embeddings, n_pca=pca_components_n)


def _scores(fitted: _FittedCombination, data: np.ndarray) -> tuple[list, list]:
    """(mse scores, Mahalanobis scores) of every normalized clip in ``data``."""
    mse_scores, maha_scores = [], []
    for i in range(len(data)):
        spec = torch.from_numpy(data[i:i + 1]).float()
        mse, maha = run_inference_single(fitted.model, spec, fitted.device), mahalanobis_score(
            get_cls_embedding_single(fitted.model, spec, fitted.device), *fitted.maha
        )
        mse_scores.append(mse)
        maha_scores.append(maha)
    return mse_scores, maha_scores


def _score_abnormal(fitted: _FittedCombination, combo_root: Path) -> tuple[dict, int] | None:
    """Evaluate against the combination's abnormal clips, if any: (metrics, n_abnormal)."""
    abnormal_dir = combo_root / "abnormal"
    if not (abnormal_dir.is_dir() and any(abnormal_dir.glob("*.wav"))):
        return None
    abnormal_wavs = sorted(abnormal_dir.glob("*.wav"))
    abn_specs = np.stack([wav_to_logmel(str(p)) for p in abnormal_wavs], axis=0)
    abn_specs_norm = normalize_logmel(abn_specs, *fitted.norm)

    normal_mse, normal_maha = _scores(fitted, fitted.val_data)
    abnormal_mse, abnormal_maha = _scores(fitted, abn_specs_norm)

    eval_metrics = evaluate_against_abnormal(
        np.array(normal_maha), np.array(abnormal_maha),
        np.array(normal_mse), np.array(abnormal_mse),
    )
    return eval_metrics, len(abnormal_wavs)


def _stage_combination(
    combo_upload_dir: Path, combo: tuple, fitted: _FittedCombination, threshold: float | None
) -> None:
    """Write best.pth + maha_stats.npz of a retrained combination for the MLflow upload."""
    machine, machine_id, snr, _combo_root = combo
    combo_upload_dir.mkdir(parents=True, exist_ok=True)
    norm_mean, norm_std = fitted.norm
    maha_mean, maha_inv_cov, maha_pca = fitted.maha
    checkpoint_out = {
        "model_state_dict": fitted.model.state_dict(),
        "norm_mean": norm_mean, "norm_std": norm_std,
        "machine": machine, "machine_id": machine_id, "snr": snr,
        # Explicit None means retrained but uncalibrated. Only legacy checkpoints
        # without this key may use the original benchmark thresholds.
        "threshold": threshold,
    }
    torch.save(checkpoint_out, combo_upload_dir / CHECKPOINT_FILENAME)
    np.savez(
        combo_upload_dir / MAHA_STATS_FILENAME,
        mean=maha_mean, inv_cov=maha_inv_cov,
        pca_mean=maha_pca[0] if maha_pca else np.zeros(0, dtype=np.float32),
        pca_v=maha_pca[1] if maha_pca else np.zeros((0, 0), dtype=np.float32),
        has_pca=np.array(maha_pca is not None),
    )


class Ml41MeatCuringMachineryAcousticAnomalyPlugin(PluginRuntimeMixin, ModelPluginPort):
    """Acoustic anomaly detection on curing machinery (AudioMAE + Mahalanobis score)."""

    def __init__(self) -> None:
        self._device: torch.device | None = None
        self._cache: CheckpointCache | None = None
        self._ready: bool = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> None:
        ensure_artifacts_downloaded()
        self._device = _safe_device()
        self._cache = CheckpointCache()
        self._ready = artifacts_available()
        if not self._ready:
            logger.warning(
                "Ml41 plugin loaded but no combination checkpoints were found under %s",
                ARTIFACT_FOLDER_NAME,
            )
        logger.info("Ml41MeatCuringMachineryAcousticAnomalyPlugin loaded: ready=%s device=%s",
                    self._ready, self._device)

    def is_loaded(self) -> bool:
        return self._ready

    def _require_loaded(self) -> None:
        if not self._ready:
            raise ModelNotLoadedError("El modelo no está cargado (no hay checkpoints disponibles).")

    # ── scoring ──────────────────────────────────────────────────────────────

    @staticmethod
    def _as_dict(loaded) -> dict:
        """Normalize a LoadedCombination (local/cached) or plain dict (from MLflow) to a dict."""
        if isinstance(loaded, LoadedCombination):
            return {
                "model": loaded.model, "norm_mean": loaded.norm_mean, "norm_std": loaded.norm_std,
                "maha_mean": loaded.maha_mean, "maha_inv_cov": loaded.maha_inv_cov,
                "maha_pca": loaded.maha_pca, "device": loaded.device,
                "threshold": loaded.threshold,
            }
        return loaded

    def _prediction_threshold(self, loaded, override: float | None = None) -> float:
        """Use the selected model's calibration, or an explicit inline override."""
        value = override if override is not None else self._as_dict(loaded)["threshold"]
        if value is None:
            raise UnsupportedMachineConfigurationError(
                "This checkpoint has no calibrated decision threshold. Train with normal "
                "and abnormal reference audio, or supply an explicit inline threshold."
            )
        if not np.isfinite(value) or value < 0:
            raise UnsupportedMachineConfigurationError(
                "The decision threshold must be finite and non-negative."
            )
        return float(value)

    def _score_spectrogram(self, loaded, spec_raw: np.ndarray) -> tuple[float, float]:
        """Given a raw (unnormalized) log-Mel spectrogram, return (mse_score, maha_score)."""
        ctx = self._as_dict(loaded)
        spec_norm = normalize_logmel(spec_raw, ctx["norm_mean"], ctx["norm_std"])
        tensor = torch.from_numpy(spec_norm).float().unsqueeze(0)  # (1, 1, n_mels, target_frames)

        mse_score = run_inference_single(ctx["model"], tensor, ctx["device"])
        embedding = get_cls_embedding_single(ctx["model"], tensor, ctx["device"])
        maha = mahalanobis_score(embedding, ctx["maha_mean"], ctx["maha_inv_cov"], ctx["maha_pca"])
        return mse_score, maha

    def _resolve_combination(self, machine: str, machine_id: str, snr: str, mlflow_run_id: str):
        """Return (loaded_combination, temp_dir_or_None). loaded_combination is either a
        LoadedCombination (local/cached) or a plain dict (from MLflow)."""
        if mlflow_run_id:
            downloaded = download_user_model_from_mlflow(mlflow_run_id)
            if downloaded:
                combinations, tmp = downloaded
                key = (machine, machine_id, snr)
                if key in combinations:
                    return combinations[key], tmp
                shutil.rmtree(tmp, ignore_errors=True)
                raise UserModelUnavailableError(
                    f"El run de MLflow '{mlflow_run_id}' no contiene un modelo reentrenado para "
                    f"la combinación {machine}/{machine_id}/{snr}. No se usa el modelo base en "
                    "su lugar: reentrena esa combinación o predice sin mlflow_run_id."
                )
        self._require_loaded()
        loaded = self._cache.get(machine, machine_id, snr, self._device)
        return loaded, None

    # ── predict_inline ───────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        _ = model_key  # not used: (machine, machine_id, snr) already select the combination
        machine, machine_id, snr = features["machine"], features["machine_id"], features["snr"]
        loaded, temp_dir = self._resolve_combination(machine, machine_id, snr, mlflow_run_id)
        with removing_dir(temp_dir):
            threshold_used = self._prediction_threshold(loaded, threshold)
            spec_raw = audio_base64_to_logmel(features["audio_base64"])
            mse_score, maha = self._score_spectrogram(loaded, spec_raw)

            self._record()
            return PredictInlineResponse(**build_inline_result(
                model_id=MODEL_ID, machine=machine, machine_id=machine_id, snr=snr,
                mse_score=mse_score, maha_score=maha, threshold=threshold_used,
            ))

    # ── predict_batch ────────────────────────────────────────────────────────

    def predict_batch(
        self, *, data_path: str, mlflow_run_id: str = "", threshold: float | None = None,
    ) -> PredictBatchResponse:
        predictions: list[dict] = []
        with local_file_path(data_path) as local_zip:
            with tempfile.TemporaryDirectory() as tmp_dir:
                safe_extract_zip(local_zip, tmp_dir)

                manifest_path = next(Path(tmp_dir).rglob("manifest.csv"), None)
                if manifest_path is None:
                    raise ValueError(
                        "El ZIP debe contener un manifest.csv (filename,machine,machine_id,snr)"
                    )

                with open(manifest_path, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))

                for row in rows:
                    filename = row["filename"]
                    try:
                        predictions.append(self._predict_manifest_row(
                            row, manifest_path.parent, mlflow_run_id, threshold
                        ))
                    except Exception as exc:  # pylint: disable=broad-exception-caught  # per-row error
                        logger.warning("Error processing %s: %s", filename, exc)
                        predictions.append({"filename": filename, "error": str(exc)})

        self._record()
        return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)

    def _predict_manifest_row(
        self, row: dict, base_dir: Path, mlflow_run_id: str, threshold: float | None
    ) -> dict:
        """Score one manifest.csv row of predict_batch (its WAV lives under ``base_dir``)."""
        filename = row["filename"]
        machine, machine_id, snr = row["machine"], row["machine_id"], row["snr"]
        wav_path = base_dir / filename
        if not wav_path.exists():
            raise InvalidAudioError(f"Referenced WAV not found in ZIP: {filename}")
        loaded, temp_dir = self._resolve_combination(machine, machine_id, snr, mlflow_run_id)
        with removing_dir(temp_dir):
            threshold_used = self._prediction_threshold(loaded, threshold)
            mse_score, maha = self._score_spectrogram(loaded, wav_to_logmel(str(wav_path)))
            return {
                **build_inline_result(
                    model_id=MODEL_ID, machine=machine, machine_id=machine_id, snr=snr,
                    mse_score=mse_score, maha_score=maha, threshold=threshold_used,
                ),
                "filename": filename,
            }

    # ── train ────────────────────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Retrain every (machine, machine_id, snr) combination
        found under data_path's {snr}_{machine}/{machine_id}/{normal,abnormal}/*.wav layout.

        'normal' is required per combination trained; 'abnormal' is optional (if present,
        full auc/fnr/fpr/recall/threshold metrics are computed for that combination — see
        train_dto.py).

        Each combination is trained from scratch with the AI team's procedure
        (trainer.py::run_training, see _train_one_combination), and the retrained
        combinations live only in their own MLflow run (artifact_path="model", laid out as
        {machine}/{machine_id}/{snr}/ so mlflow_utils can rebuild them with that
        mlflow_run_id). The served base checkpoints, their local copy and the in-memory
        cache are never mutated nor overwritten.
        """
        tracker = require_train_tracker(mlflow_run_id)
        per_combination: list[CombinationTrainMetrics] = []

        with local_file_path(data_path) as local_zip:
            extract_dir = Path(tempfile.mkdtemp(prefix="ml41_train_"))
            staging_dir = Path(tempfile.mkdtemp(prefix="ml41_mlflow_"))
            try:
                safe_extract_zip(local_zip, extract_dir)

                combos_found = self._discover_combinations(extract_dir)
                if not combos_found:
                    raise ValueError(
                        "No se encontraron combinaciones válidas. Estructura esperada: "
                        "{snr}_{machine}/{machine_id}/normal/*.wav"
                    )

                for combo in combos_found:
                    per_combination.append(
                        self._train_one_combination(combo, tracker, staging_dir)
                    )

                upload_dir_to_mlflow(tracker, str(staging_dir))
            finally:
                shutil.rmtree(extract_dir, ignore_errors=True)
                shutil.rmtree(staging_dir, ignore_errors=True)

        return TrainResponse(
            detail=f"Entrenamiento completado para {len(per_combination)} combinación(es)",
            per_combination=per_combination,
            upload_warning=None,
        )

    @staticmethod
    def _discover_combinations(root: Path) -> list[tuple[str, str, str, Path]]:
        """Find {snr}_{machine}/{machine_id}/normal/*.wav under root."""
        found = []
        for snr_machine_dir in root.rglob("*"):
            if not snr_machine_dir.is_dir():
                continue
            name = snr_machine_dir.name
            matched_machine = next((m for m in MACHINES if name.endswith(f"_{m}")), None)
            if matched_machine is None:
                continue
            snr_part = name[: -(len(matched_machine) + 1)]
            if snr_part not in SNRS:
                continue
            for machine_id_dir in snr_machine_dir.iterdir():
                if machine_id_dir.name in MACHINE_IDS and (machine_id_dir / "normal").is_dir():
                    found.append((matched_machine, machine_id_dir.name, snr_part, machine_id_dir))
        return found

    def _train_one_combination(
        self, combo: tuple, tracker: BaseMLflowTracker, staging_dir: Path
    ) -> CombinationTrainMetrics:
        """Train, evaluate and stage one ``(machine, machine_id, snr, combo_root)`` combination
        (mirrors trainer.py::run_training)."""
        machine, machine_id, snr, combo_root = combo
        fitted = self._fit_new_model(combo, tracker)

        metrics_kwargs = {
            "machine": machine, "machine_id": machine_id, "snr": snr,
            "best_val_loss": round(fitted.best_val_loss, 6), "n_train": len(fitted.train_data),
            "n_val": len(fitted.val_data),
        }

        abnormal = _score_abnormal(fitted, combo_root)
        if abnormal is not None:
            eval_metrics, n_abnormal = abnormal
            metrics_kwargs.update(eval_metrics)
            metrics_kwargs["n_abnormal_samples"] = n_abnormal
            tracker.log_metrics(
                {f"{machine}_{machine_id}_{snr}_{k}": v for k, v in eval_metrics.items()}
            )

        tracker.log_metrics({f"{machine}_{machine_id}_{snr}_best_val_loss": fitted.best_val_loss})

        # Stage for MLflow only — the fixed base checkpoint is never written.
        _stage_combination(
            staging_dir / machine / machine_id / snr, combo, fitted,
            metrics_kwargs.get("threshold"),
        )
        return CombinationTrainMetrics(**metrics_kwargs)

    def _fit_new_model(self, combo: tuple, tracker: BaseMLflowTracker) -> _FittedCombination:
        """Train a NEW model for ``combo`` with the original procedure.

        trainer.py::run_training (the original never fine-tunes: with an existing best.pth it
        skips training, and with force=True it trains a NEW model). Normalization stats come
        from the train split (MIMIIDataset), seed 42 is set BEFORE the model is created.
        """
        machine, machine_id, snr = combo[:3]
        hyperparams = TRAINING_HYPERPARAMS_BY_MACHINE[machine]
        train_data, val_data, norm = _split_normal_clips(combo, hyperparams)
        device = self._device or _safe_device()

        torch.manual_seed(TRAINING_COMMON["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(TRAINING_COMMON["seed"])
        np.random.seed(TRAINING_COMMON["seed"])
        model = build_model(machine, device)
        logger.info("Training %s/%s/%s from scratch (original procedure)", machine, machine_id, snr)

        tracker.log_params(
            {f"{machine}_{machine_id}_{snr}_{k}": v for k, v in hyperparams.items()}
        )

        best_val_loss = fine_tune(model, train_data, val_data, device, hyperparams)[0]
        maha = _fit_maha(model, train_data, device, machine)
        return _FittedCombination(
            model=model, device=device, norm=norm, train_data=train_data, val_data=val_data,
            best_val_loss=best_val_loss, maha=maha,
        )

    # ── stats ────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Detección de anomalías acústicas (Audio-MAE ViT-Tiny) en maquinaria de "
                "curado de embutidos: 48 combinaciones fan/pump/slider/valve × 4 unidades × "
                "3 niveles de ruido (dataset MIMII). Score de decisión: distancia de "
                "Mahalanobis en el espacio latente CLS proyectado por PCA."
            ),
            task_type="anomaly_detection_audio",
            framework="pytorch",
            inputs=[
                InputField(
                    name="audio_base64",
                    type="file",
                    format=["wav"],
                    description=(
                        "Grabación WAV de la máquina (base64 inline, ZIP+manifest.csv en batch)"
                    ),
                ),
                InputField(name="machine", type="str", description="fan | pump | slider | valve"),
                InputField(
                    name="machine_id", type="str", description="id_00 | id_02 | id_04 | id_06"
                ),
                InputField(name="snr", type="str", description="-6_dB | 0_dB | 6_dB"),
            ],
            outputs=[
                OutputField(
                    name="mse_score",
                    type="float",
                    description="Error de reconstrucción (informativo, no determinista)",
                ),
                OutputField(
                    name="maha_score",
                    type="float",
                    description="Distancia de Mahalanobis (señal de decisión, determinista)",
                ),
                OutputField(name="predicted_label", type="int", description="0=normal, 1=anómalo"),
                OutputField(
                    name="threshold_used",
                    type="float",
                    description="Umbral aplicado para esta combinación",
                ),
            ],
            metrics=dict(METRICS_REPORTED),
            runtime_stats=self._runtime_stats(),
        )
        return add_mlflow_run_to_stats(base, mlflow_run_id)
