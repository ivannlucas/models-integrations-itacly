"""Pipeline de entrenamiento para el modelo Deep Neuro-Fuzzy."""

import ast
import copy
import os
from typing import Dict, Optional, Tuple
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from src.training.loss import DNFLoss
from src.training.metrics import (
    binary_metrics_np,
    compute_class_weights,
    optimize_threshold_by_f1,
)
from src.training.model import ParallelDeepNeuroFuzzyModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =========================================================
# Configuración del modelo
# =========================================================


def model_config(
    model_cfg: dict
) -> Tuple[dict, ParallelDeepNeuroFuzzyModel]:
    """Crea el modelo a partir de la configuración y muestra resumen de arquitectura.

    Args:
        model_cfg: Diccionario de configuración del modelo.
        model_title: Título para el resumen impreso.

    Returns:
        Tupla con (config actualizada, modelo instanciado).
    """
    model_seed = int(model_cfg.get("seed", 42))
    model = ParallelDeepNeuroFuzzyModel(model_cfg, seed=model_seed)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dl_params = (
        sum(p.numel() for p in model.lstm_branch.parameters())
        + sum(p.numel() for p in model.attention_layer.parameters())
        + sum(p.numel() for p in model.dl_projection.parameters())
        + sum(p.numel() for p in model.anomaly_dl.parameters())
    )

    fuzzy_params = (
        sum(p.numel() for p in model.fuzzy_layer.parameters())
        + sum(p.numel() for p in model.rule_layer.parameters())
    )

    logger.info(f"Total de parámetros: {total_params:,}")
    logger.info(f"- Parámetros entrenables: {trainable_params:,}")
    logger.info(f"- Parámetros rama fuzzy: {fuzzy_params:,}")
    logger.info(f"- Parámetros rama DL: {dl_params:,}")

    return model_cfg, model


# =========================================================
# DataLoaders
# =========================================================


def prepare_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    stats_train: np.ndarray,
    stats_val: np.ndarray,
    stats_test: np.ndarray,
    batch_size: int = 64,
    shuffle_train: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """Construye DataLoaders para train/val/test.

    Args:
        X_train: Secuencias de entrenamiento [N, T, F].
        y_train: Etiquetas de entrenamiento [N].
        X_val: Secuencias de validación.
        y_val: Etiquetas de validación.
        X_test: Secuencias de test.
        y_test: Etiquetas de test.
        stats_train: Estadísticas de entrenamiento para rama fuzzy.
        stats_val: Estadísticas de validación.
        stats_test: Estadísticas de test.
        batch_size: Tamaño de batch.
        shuffle_train: Si mezclar datos de entrenamiento.
        num_workers: Workers para DataLoader.
        pin_memory: Pin memory para GPU.

    Returns:
        Dict con DataLoaders para 'train', 'val', 'test'.
    """

    def _validate_shapes(
        X: np.ndarray,
        y: np.ndarray,
        stats: np.ndarray,
        split_name: str,
    ) -> None:
        if X.ndim != 3:
            raise ValueError(
                f"{split_name}: X debe ser 3D [N,T,F]. Recibido: {X.shape}"
            )
        if y.ndim != 1:
            raise ValueError(
                f"{split_name}: y debe ser 1D [N]. Recibido: {y.shape}"
            )
        if len(X) != len(y):
            raise ValueError(
                f"{split_name}: X e y deben tener el mismo número de muestras."
            )
        if stats.ndim != 2:
            raise ValueError(
                f"{split_name}: stats debe ser 2D [N,S]. Recibido: {stats.shape}"
            )
        if len(stats) != len(X):
            raise ValueError(
                f"{split_name}: stats y X deben tener el mismo número de muestras."
            )

    def _make_loader(
        X: np.ndarray,
        y: np.ndarray,
        stats: np.ndarray,
        shuffle: bool,
        split_name: str,
    ) -> DataLoader:
        _validate_shapes(X, y, stats, split_name)
        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.long)
        S_t = torch.as_tensor(stats, dtype=torch.float32)
        if len(S_t) != len(X_t):
            raise ValueError("stats y X deben tener el mismo número de muestras")
        dataset = TensorDataset(X_t, S_t, y_t)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return {
        "train": _make_loader(X_train, y_train, stats_train, shuffle_train, "train"),
        "val": _make_loader(X_val, y_val, stats_val, False, "val"),
        "test": _make_loader(X_test, y_test, stats_test, False, "test"),
    }


# =========================================================
# Batch utils
# =========================================================


def unpack_batch(
    batch: tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Desempaqueta un batch en (x_window, s_stats, y).

    Args:
        batch: Tupla del DataLoader con 3 elementos.

    Returns:
        Tupla con (x_window, s_stats, y).
    """
    if len(batch) == 3:
        x_window, s_stats, y = batch
    else:
        raise ValueError(f"Batch no soportado con longitud {len(batch)}")
    return x_window, s_stats, y


# =========================================================
# Forward homogéneo del modelo
# =========================================================


def forward_model(
    model: torch.nn.Module,
    x_window: torch.Tensor,
    s_stats: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Ejecuta forward pass y valida la salida del modelo.

    Args:
        model: Modelo PyTorch.
        x_window: Tensor de secuencias [B, T, F].
        s_stats: Tensor de estadísticas [B, F_stats].

    Returns:
        Dict con salidas del modelo homogeneizadas.

    Raises:
        TypeError: Si el modelo no devuelve un dict.
        KeyError: Si faltan claves obligatorias.
    """
    out = model(x_window, s_stats)

    if not isinstance(out, dict):
        raise TypeError(
            "El modelo debe devolver un dict homogéneo con las claves esperadas."
        )

    if "anomaly_score" not in out:
        raise KeyError("La salida del modelo debe incluir 'anomaly_score'")

    return {
        "anomaly_score": out["anomaly_score"],
        "logit_anomaly_dl": out.get("logit_anomaly_dl"),
        "logit_anomaly_fuzzy": out.get("logit_anomaly_fuzzy"),
        "rule_activations": out.get("rule_activations"),
        "alpha_probs": out.get("alpha_probs"),
        "membership": out.get("membership"),
        "dl_embed": out.get("dl_embed"),
        "attention_weights": out.get("attention_weights"),
        "output_fuzzy": out.get("output_anomaly"),
    }


# =========================================================
# Métricas auxiliares del bloque fuzzy
# =========================================================


def alpha_entropy_mean(
    alpha_probs: Optional[torch.Tensor], eps: float = 1e-8
) -> float:
    """Entropía media de alpha_probs [R, F, M].

    Args:
        alpha_probs: Tensor de probabilidades alpha.
        eps: Epsilon numérico.

    Returns:
        Entropía media como float.
    """
    if alpha_probs is None:
        return 0.0
    p = alpha_probs.clamp_min(eps)
    entropy = -(p * torch.log(p)).sum(dim=-1)
    return float(entropy.mean().detach().cpu())


def rule_corr_mean(
    rule_activations: Optional[torch.Tensor], eps: float = 1e-8
) -> float:
    """Correlación media absoluta fuera de diagonal entre reglas.

    Args:
        rule_activations: Tensor [B, R].
        eps: Epsilon numérico.

    Returns:
        Correlación media como float.
    """
    if rule_activations is None or rule_activations.dim() != 2:
        return 0.0

    B, R = rule_activations.shape
    if B < 2 or R < 2:
        return 0.0

    x = rule_activations - rule_activations.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    x = x / std
    corr = (x.T @ x) / B

    mask = ~torch.eye(R, dtype=torch.bool, device=corr.device)
    return float(corr[mask].abs().mean().detach().cpu())


# =========================================================
# Step de entrenamiento/evaluación
# =========================================================


def step_dnfl(
    model: torch.nn.Module,
    criterion: Any,
    x_window: torch.Tensor,
    y_class: torch.Tensor,
    s_stats: torch.Tensor,
) -> Dict[str, Any]:
    """Ejecuta forward pass, loss y predicciones para un batch.

    Args:
        model: Modelo PyTorch.
        criterion: Función de loss.
        x_window: Tensor de secuencias [B, T, F].
        y_class: Etiquetas de anomalía [B].
        s_stats: Tensor de estadísticas [B, F_stats].

    Returns:
        Dict con loss, predicciones y salidas intermedias.
    """
    out = forward_model(model, x_window, s_stats)

    anomaly_score = torch.nan_to_num(
        out["anomaly_score"], nan=0.0, posinf=20.0, neginf=-20.0
    )

    logit_anomaly_dl = out["logit_anomaly_dl"]
    logit_anomaly_fuzzy = out["logit_anomaly_fuzzy"]
    rule_activations = out["rule_activations"]
    alpha_probs = out["alpha_probs"]

    if rule_activations is None:
        rule_activations = torch.ones(
            (anomaly_score.size(0), 1),
            device=anomaly_score.device,
            dtype=anomaly_score.dtype,
        )

    y_anomaly = (y_class > 0).float()

    loss, loss_dict = criterion(
        anomaly_score=anomaly_score,
        logit_anomaly_dl=logit_anomaly_dl,
        logit_anomaly_fuzzy=logit_anomaly_fuzzy,
        rule_activations=rule_activations,
        alpha_probs=alpha_probs,
        y_anomaly=y_anomaly,
    )

    probs_anom = torch.sigmoid(anomaly_score.view(-1))
    preds_anom = (probs_anom >= 0.5).long()

    preds_dl = None
    preds_fuzzy = None

    if logit_anomaly_dl is not None:
        preds_dl = (torch.sigmoid(logit_anomaly_dl.view(-1)) >= 0.5).long()
    if logit_anomaly_fuzzy is not None:
        preds_fuzzy = (
            (torch.sigmoid(logit_anomaly_fuzzy.view(-1)) >= 0.5).long()
        )

    return {
        "loss": loss,
        "loss_dict": loss_dict,
        "anomaly_score": anomaly_score,
        "preds_anom": preds_anom,
        "probs_anom": probs_anom,
        "preds_dl": preds_dl,
        "preds_fuzzy": preds_fuzzy,
        "y_anomaly": y_anomaly.long(),
        "rule_activations": rule_activations,
        "alpha_probs": alpha_probs,
    }


# =========================================================
# Epoch core
# =========================================================


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Any,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    """Ejecuta una época completa de entrenamiento o evaluación.

    Args:
        model: Modelo PyTorch.
        loader: DataLoader.
        criterion: Función de loss.
        device: Dispositivo (CPU/CUDA).
        optimizer: Optimizador (None para evaluación).
        grad_clip: Valor máximo de norma de gradiente.

    Returns:
        Dict con métricas de la época.
    """
    is_train = optimizer is not None
    model.train(is_train)

    losses_total = []
    losses_anom_final = []
    losses_anom_dl = []
    losses_anom_fuzzy = []
    losses_diversity = []
    losses_alpha_entropy = []
    losses_rule_structure = []
    losses_rule_usage = []

    y_anom_all = []
    pred_anom_all = []
    prob_anom_all = []

    pred_dl_all = []
    pred_fuzzy_all = []

    alpha_entropy_vals = []
    rule_corr_vals = []
    batch_sizes = []
    rule_sum = None
    num_skipped_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in loader:
            x_window, s_stats, y_class = unpack_batch(batch)

            x_window = x_window.to(device, non_blocking=True)
            y_class = y_class.to(device, non_blocking=True)
            if s_stats is not None:
                s_stats = s_stats.to(device, non_blocking=True)

            result = step_dnfl(
                model=model,
                criterion=criterion,
                x_window=x_window,
                y_class=y_class,
                s_stats=s_stats,
            )

            loss = result["loss"]
            if not torch.isfinite(loss):
                num_skipped_batches += 1
                continue

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if grad_clip is not None:
                    clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                optimizer.step()

            loss_dict = result["loss_dict"]

            losses_total.append(loss_dict.get("loss_total", float(loss.detach().cpu())))
            losses_anom_final.append(loss_dict.get("loss_anom_final", 0.0))
            losses_anom_dl.append(loss_dict.get("loss_anom_dl", 0.0))
            losses_anom_fuzzy.append(loss_dict.get("loss_anom_fuzzy", 0.0))
            losses_diversity.append(loss_dict.get("loss_diversity", 0.0))
            losses_alpha_entropy.append(loss_dict.get("loss_alpha_entropy", 0.0))
            losses_rule_structure.append(loss_dict.get("loss_rule_structure", 0.0))
            losses_rule_usage.append(loss_dict.get("loss_rule_usage", 0.0))

            y_anom_all.append(result["y_anomaly"].detach().cpu().numpy())
            pred_anom_all.append(result["preds_anom"].detach().cpu().numpy())
            prob_anom_all.append(result["probs_anom"].detach().cpu().numpy())

            if result["preds_dl"] is not None:
                pred_dl_all.append(
                    result["preds_dl"].detach().cpu().numpy()
                )
            if result["preds_fuzzy"] is not None:
                pred_fuzzy_all.append(
                    result["preds_fuzzy"].detach().cpu().numpy()
                )

            alpha_entropy_vals.append(alpha_entropy_mean(result["alpha_probs"]))
            rule_corr_vals.append(rule_corr_mean(result["rule_activations"]))

            ra = result["rule_activations"].detach()
            batch_mean = ra.mean(dim=0).cpu()
            if rule_sum is None:
                rule_sum = batch_mean * ra.size(0)
            else:
                rule_sum += batch_mean * ra.size(0)
            batch_sizes.append(ra.size(0))

    if len(losses_total) == 0:
        return {
            "loss_total": float("nan"),
            "num_skipped_batches": num_skipped_batches,
        }

    y_anom_all = np.concatenate(y_anom_all)
    pred_anom_all = np.concatenate(pred_anom_all)
    prob_anom_all = np.concatenate(prob_anom_all)

    metrics_anom = binary_metrics_np(y_anom_all, pred_anom_all)

    try:
        anomaly_auc = float(roc_auc_score(y_anom_all, prob_anom_all))
    except Exception:
        anomaly_auc = float("nan")

    dl_acc = float("nan")
    dl_f1 = float("nan")
    if len(pred_dl_all) > 0:
        pred_dl_all = np.concatenate(pred_dl_all)
        dl_metrics = binary_metrics_np(y_anom_all, pred_dl_all)
        dl_acc = dl_metrics["acc"]
        dl_f1 = dl_metrics["f1"]

    fuzzy_acc = float("nan")
    fuzzy_f1 = float("nan")
    if len(pred_fuzzy_all) > 0:
        pred_fuzzy_all = np.concatenate(pred_fuzzy_all)
        fuzzy_metrics = binary_metrics_np(y_anom_all, pred_fuzzy_all)
        fuzzy_acc = fuzzy_metrics["acc"]
        fuzzy_f1 = fuzzy_metrics["f1"]

    total_samples = sum(batch_sizes)
    rule_activation_mean = (
        (rule_sum / max(total_samples, 1)).numpy()
        if rule_sum is not None
        else np.array([])
    )

    dead_rules = (
        int((rule_activation_mean < 1e-4).sum())
        if rule_activation_mean.size > 0
        else 0
    )
    dead_rules_ratio = (
        float(dead_rules / len(rule_activation_mean))
        if rule_activation_mean.size > 0
        else 0.0
    )

    return {
        "loss_total": float(np.mean(losses_total)),
        "loss_anom_final": float(np.mean(losses_anom_final)),
        "loss_anom_dl": float(np.mean(losses_anom_dl)),
        "loss_anom_fuzzy": float(np.mean(losses_anom_fuzzy)),
        "loss_diversity": float(np.mean(losses_diversity)),
        "loss_alpha_entropy": float(np.mean(losses_alpha_entropy)),
        "loss_rule_structure": float(np.mean(losses_rule_structure)),
        "loss_rule_usage": float(np.mean(losses_rule_usage)),
        "anomaly_acc": metrics_anom["acc"],
        "anomaly_f1": metrics_anom["f1"],
        "anomaly_auc": anomaly_auc,
        "fuzzy_acc": fuzzy_acc,
        "fuzzy_f1": fuzzy_f1,
        "dl_acc": dl_acc,
        "dl_f1": dl_f1,
        "alpha_entropy_mean": float(np.mean(alpha_entropy_vals)),
        "rule_corr_mean": float(np.mean(rule_corr_vals)),
        "dead_rules": dead_rules,
        "dead_rules_ratio": dead_rules_ratio,
        "rule_activation_mean": rule_activation_mean,
        "num_skipped_batches": num_skipped_batches,
    }


# =========================================================
# Wrappers train / val
# =========================================================


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    """Wrapper de entrenamiento por época."""
    return run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        optimizer=optimizer,
        grad_clip=grad_clip,
    )


def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Any,
    device: torch.device,
) -> Dict[str, float]:
    """Wrapper de validación por época."""
    return run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        grad_clip=None,
    )


def resolve_training_device(train_cfg: dict) -> torch.device:
    """Resuelve el dispositivo de entrenamiento con fallback seguro a CPU.

    Soporta training_kwargs.device en {'auto', 'cuda', 'cpu'}.
    """
    requested = str(train_cfg.get("device", "auto")).strip().lower()
    cuda_available = torch.cuda.is_available()

    if requested in {"", "auto"}:
        use_cuda = cuda_available
    elif requested in {"cuda", "gpu"}:
        if not cuda_available:
            logger.warning(
                "Aviso: training_kwargs.device='cuda' pero CUDA no está disponible. "
                "Se usará CPU."
            )
            use_cuda = False
        else:
            use_cuda = True
    elif requested == "cpu":
        use_cuda = False
    else:
        logger.warning(
            f"Aviso: training_kwargs.device='{requested}' no es válido. "
            "Se usará modo 'auto'."
        )
        use_cuda = cuda_available

    device = torch.device("cuda" if use_cuda else "cpu")
    cuda_build = torch.version.cuda if torch.version.cuda is not None else "cpu-only"

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(
            f"Dispositivo de entrenamiento: CUDA ({gpu_name}) "
            f"| torch={torch.__version__} | build CUDA={cuda_build}"
        )
    else:
        reason = (
            "forzado por configuración"
            if requested == "cpu"
            else "CUDA no disponible"
        )
        logger.info(
            f"Dispositivo de entrenamiento: CPU ({reason}) "
            f"| torch={torch.__version__} | build CUDA={cuda_build}"
        )

    return device


# =========================================================
# Evaluación final en test
# =========================================================


def evaluate_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Evalúa rendimiento final y devuelve arrays para análisis posterior.

    Args:
        model: Modelo PyTorch.
        loader: DataLoader de test.
        device: Dispositivo.
        threshold: Umbral para convertir probabilidades a predicciones binarias.

    Returns:
        Tupla con (dict de métricas, dict de arrays de salida).
    """
    model.eval()

    y_anom_all = []
    pred_anom_all = []
    prob_anom_all = []
    anomaly_logits_all = []
    rule_activations_all = []
    attention_weights_all = []
    alpha_probs_last = None

    with torch.no_grad():
        for batch in loader:
            x_window, s_stats, y_class = unpack_batch(batch)

            x_window = x_window.to(device, non_blocking=True)
            y_class = y_class.to(device, non_blocking=True)
            if s_stats is not None:
                s_stats = s_stats.to(device, non_blocking=True)

            out = forward_model(model, x_window, s_stats)

            anomaly_score = out["anomaly_score"]

            probs_anom = torch.sigmoid(anomaly_score.view(-1))
            preds_anom = (probs_anom >= threshold).long()

            y_anom = (y_class > 0).long()

            y_anom_all.append(y_anom.cpu().numpy())
            pred_anom_all.append(preds_anom.cpu().numpy())
            prob_anom_all.append(probs_anom.cpu().numpy())

            anomaly_logits_all.append(anomaly_score.view(-1).cpu().numpy())

            if out["rule_activations"] is not None:
                rule_activations_all.append(out["rule_activations"].cpu().numpy())
            if out["attention_weights"] is not None:
                attention_weights_all.append(out["attention_weights"].cpu().numpy())
            if out["alpha_probs"] is not None:
                alpha_probs_last = out["alpha_probs"].detach().cpu().numpy()

    y_anom_all = np.concatenate(y_anom_all)
    pred_anom_all = np.concatenate(pred_anom_all)
    prob_anom_all = np.concatenate(prob_anom_all)

    metrics_anom = binary_metrics_np(y_anom_all, pred_anom_all)

    try:
        anomaly_auc = float(roc_auc_score(y_anom_all, prob_anom_all))
    except Exception:
        anomaly_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_anom_all, pred_anom_all, labels=[0, 1]).ravel()
    specificity = float(tn / max(tn + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    precision = float(tp / max(tp + fp, 1))

    metrics = {
        "anomaly_acc": metrics_anom["acc"],
        "anomaly_f1": metrics_anom["f1"],
        "anomaly_auc": anomaly_auc,
        "anomaly_precision": precision,
        "anomaly_recall": recall,
        "anomaly_specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    outputs = {
        "y_anomaly": y_anom_all,
        "pred_anomaly": pred_anom_all,
        "prob_anomaly": prob_anom_all,
        "anomaly_logits": np.concatenate(anomaly_logits_all),
        "rule_activations": (
            np.concatenate(rule_activations_all)
            if len(rule_activations_all) > 0
            else None
        ),
        "attention_weights": (
            np.concatenate(attention_weights_all)
            if len(attention_weights_all) > 0
            else None
        ),
        "alpha_probs": alpha_probs_last,
    }

    return metrics, outputs


# =========================================================
# Pipeline principal de entrenamiento
# =========================================================


def run_train_pipeline(
    split_data: dict,
    model_cfg: dict,
    split_data_df_stats: dict,
    save_prefix: str = "dnf_model",
    save_dir: str = "models/artifacts",
) -> dict:
    """Entrena el modelo Deep Neuro Fuzzy para detección binaria de anomalía y evalúa en test."""
    required_keys = {"train", "val", "test"}

    # -----------------------------------------------------
    # Validaciones de entrada
    # -----------------------------------------------------
    if set(split_data["X"].keys()) != required_keys:
        raise ValueError(
            f"split_data['X'] debe tener claves {required_keys}. "
            f"Recibido: {set(split_data['X'].keys())}"
        )
    if set(split_data["y"].keys()) != required_keys:
        raise ValueError(
            f"split_data['y'] debe tener claves {required_keys}. "
            f"Recibido: {set(split_data['y'].keys())}"
        )
    if set(split_data_df_stats.keys()) != required_keys:
        raise ValueError(
            f"split_data_df_stats debe tener claves {required_keys}. "
            f"Recibido: {set(split_data_df_stats.keys())}"
        )

    # -----------------------------------------------------
    # Helpers internos
    # -----------------------------------------------------
    def _to_numpy(x: Any) -> np.ndarray:
        """Convierte DataFrame/Series a numpy."""
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return x.values
        return np.asarray(x)

    # -----------------------------------------------------
    # Datos
    # -----------------------------------------------------
    X_train = _to_numpy(split_data["X"]["train"])
    X_val = _to_numpy(split_data["X"]["val"])
    X_test = _to_numpy(split_data["X"]["test"])

    y_train = _to_numpy(split_data["y"]["train"]).reshape(-1)
    y_val = _to_numpy(split_data["y"]["val"]).reshape(-1)
    y_test = _to_numpy(split_data["y"]["test"]).reshape(-1)

    S_train = _to_numpy(split_data_df_stats["train"])
    S_val = _to_numpy(split_data_df_stats["val"])
    S_test = _to_numpy(split_data_df_stats["test"])

    if X_train.ndim != 3 or X_val.ndim != 3 or X_test.ndim != 3:
        raise ValueError("Todas las particiones de X deben ser tensores 3D [N,T,F].")

    for split_name, X_part, y_part, S_part in [
        ("train", X_train, y_train, S_train),
        ("val", X_val, y_val, S_val),
        ("test", X_test, y_test, S_test),
    ]:
        if len(X_part) != len(y_part):
            raise ValueError(
                f"{split_name}: número de muestras distinto entre X e y "
                f"({len(X_part)} vs {len(y_part)})."
            )
        if len(X_part) != len(S_part):
            raise ValueError(
                f"{split_name}: número de muestras distinto entre X y S_stats "
                f"({len(X_part)} vs {len(S_part)})."
            )

    # -----------------------------------------------------
    # Config modelo
    # -----------------------------------------------------
    model_cfg_local = copy.deepcopy(model_cfg)
    model_cfg_local["input_features"] = int(X_train.shape[-1])
    model_cfg_local["sequence_length"] = int(X_train.shape[1])
    model_cfg_local["n_stats_features"] = int(S_train.shape[1])

    model_cfg_final, model = model_config(
        model_cfg_local,
    )

    train_cfg = model_cfg_final["training_kwargs"]

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------
    device = resolve_training_device(train_cfg)
    model = model.to(device)

    pin_memory_cfg = train_cfg.get("pin_memory", None)
    pin_memory = (
        bool(pin_memory_cfg)
        if pin_memory_cfg is not None
        else device.type == "cuda"
    )

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------
    dataloaders = prepare_dataloaders(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        stats_train=S_train,
        stats_val=S_val,
        stats_test=S_test,
        batch_size=int(train_cfg["batch_size"]),
        shuffle_train=bool(train_cfg.get("shuffle_train", False)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # -----------------------------------------------------
    # Peso positivo para la clase de anomalía
    # -----------------------------------------------------
    pos_weight_tensor = compute_class_weights(
        y_train=y_train, device=device,
    )

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------
    dnf_loss_kwargs = copy.deepcopy(train_cfg.get("dnf_loss", {}))

    criterion = DNFLoss(
        anomaly_pos_weight=pos_weight_tensor,
        **dnf_loss_kwargs,
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    # -----------------------------------------------------
    # Scheduler
    # -----------------------------------------------------
    scheduler = None
    scheduler_mode = None

    scheduler_cfg = train_cfg.get("scheduler", {})
    sched_type = scheduler_cfg.get("type", "cosine")

    if sched_type == "cosine_warmup":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(scheduler_cfg.get("T_0", 30)),
            T_mult=int(scheduler_cfg.get("T_mult", 2)),
            eta_min=float(scheduler_cfg.get("eta_min", 1e-6)),
        )
        scheduler_mode = "epoch_float"
    elif sched_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 8)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )
        scheduler_mode = "plateau"
    elif sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.get("T_max", train_cfg["num_epochs"])),
            eta_min=float(scheduler_cfg.get("eta_min", 1e-6)),
        )
        scheduler_mode = "epoch"

    logger.info(
        f"Scheduler configurado: "
        f"{scheduler.__class__.__name__ if scheduler is not None else 'None'} "
        f"| mode={scheduler_mode}"
    )
    logger.info("=" * 30)

    # -----------------------------------------------------
    # Historial
    # -----------------------------------------------------
    history = {
        "train_loss": [],
        "train_anomaly_acc": [],
        "train_anomaly_f1": [],
        "train_anomaly_auc": [],
        "val_loss": [],
        "val_anomaly_acc": [],
        "val_anomaly_f1": [],
        "val_anomaly_auc": [],
        "train_alpha_entropy": [],
        "val_alpha_entropy": [],
        "train_rule_corr": [],
        "val_rule_corr": [],
        "train_dead_rules": [],
        "val_dead_rules": [],
        "train_fuzzy_acc": [],
        "train_fuzzy_f1": [],
        "val_fuzzy_acc": [],
        "val_fuzzy_f1": [],
        "train_dl_acc": [],
        "train_dl_f1": [],
        "val_dl_acc": [],
        "val_dl_f1": [],
        "train_rule_activation_mean": [],
        "val_rule_activation_mean": [],
        "monitor_score": [],
    }

    # -----------------------------------------------------
    # Early stopping
    # -----------------------------------------------------
    best_score = -np.inf
    best_epoch = None
    best_epoch_summary = None
    epochs_without_improve = 0
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_{save_prefix}.pt")

    num_epochs = int(train_cfg["num_epochs"])
    grad_clip = train_cfg.get("grad_clip", None)
    grad_clip = None if grad_clip is None else float(grad_clip)

    min_delta = float(train_cfg.get("min_delta", 1e-4))
    patience = int(train_cfg.get("patience", 15))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    monitor_metric = train_cfg.get("monitor_metric", "val_anomaly_f1")

    def _compute_monitor_score(monitor_metric, train_out, val_out):
        metric_map = {
            "val_anomaly_f1": float(val_out["anomaly_f1"]),
            "val_anomaly_auc": float(val_out["anomaly_auc"]),
            "neg_val_loss": -float(val_out["loss_total"]),
            "fuzzy_f1": float(val_out["fuzzy_f1"]),
            "dl_f1": float(val_out["dl_f1"]),
            "dead_rules_ratio": float(val_out["dead_rules_ratio"]),
            "rule_corr": float(val_out["rule_corr_mean"]),
        }

        if monitor_metric in metric_map:
            return float(metric_map[monitor_metric])

        monitor_context = dict(metric_map)

        for key, value in train_out.items():
            if np.isscalar(value):
                monitor_context[f"train_{key}"] = float(value)

        for key, value in val_out.items():
            if np.isscalar(value):
                monitor_context[f"val_{key}"] = float(value)
                monitor_context[key] = float(value)

        try:
            formatted_score = monitor_metric.format(**monitor_context)
            return float(formatted_score)
        except Exception:
            pass

        try:
            expr = ast.parse(monitor_metric, mode="eval")
        except SyntaxError as exc:
            raise ValueError(
                f"monitor_metric '{monitor_metric}' no tiene sintaxis válida."
            ) from exc

        allowed_binops = (
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
        )
        allowed_unaryops = (ast.UAdd, ast.USub)

        def _eval_ast(node):
            if isinstance(node, ast.Expression):
                return _eval_ast(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return float(node.value)
                raise ValueError(
                    "Solo se permiten constantes numéricas en monitor_metric."
                )
            if isinstance(node, ast.Name):
                if node.id in monitor_context:
                    return float(monitor_context[node.id])
                raise ValueError(
                    f"Variable '{node.id}' no disponible para monitor_metric."
                )
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, allowed_unaryops
            ):
                value = _eval_ast(node.operand)
                return +value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and isinstance(
                node.op, allowed_binops
            ):
                left = _eval_ast(node.left)
                right = _eval_ast(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Pow):
                    return left ** right
                if isinstance(node.op, ast.Mod):
                    return left % right
                if isinstance(node.op, ast.FloorDiv):
                    return left // right
            raise ValueError("Expresión de monitor_metric no permitida.")

        return float(_eval_ast(expr))

    # -----------------------------------------------------
    # Loop principal
    # -----------------------------------------------------
    for epoch in range(num_epochs):
        train_out = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
        )
        val_out = validate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Historial train
        history["train_loss"].append(float(train_out["loss_total"]))
        history["train_anomaly_acc"].append(float(train_out["anomaly_acc"]))
        history["train_anomaly_f1"].append(float(train_out["anomaly_f1"]))
        history["train_anomaly_auc"].append(float(train_out["anomaly_auc"]))
        history["train_alpha_entropy"].append(float(train_out["alpha_entropy_mean"]))
        history["train_rule_corr"].append(float(train_out["rule_corr_mean"]))
        history["train_dead_rules"].append(int(train_out["dead_rules"]))
        history["train_fuzzy_acc"].append(
            float(train_out["fuzzy_acc"])
            if np.isfinite(train_out["fuzzy_acc"]) else np.nan
        )
        history["train_fuzzy_f1"].append(
            float(train_out["fuzzy_f1"])
            if np.isfinite(train_out["fuzzy_f1"]) else np.nan
        )
        history["train_dl_acc"].append(
            float(train_out["dl_acc"])
            if np.isfinite(train_out["dl_acc"]) else np.nan
        )
        history["train_dl_f1"].append(
            float(train_out["dl_f1"])
            if np.isfinite(train_out["dl_f1"]) else np.nan
        )
        history["train_rule_activation_mean"].append(
            train_out["rule_activation_mean"].tolist()
            if isinstance(train_out["rule_activation_mean"], np.ndarray) else []
        )

        # Historial val
        history["val_loss"].append(float(val_out["loss_total"]))
        history["val_anomaly_acc"].append(float(val_out["anomaly_acc"]))
        history["val_anomaly_f1"].append(float(val_out["anomaly_f1"]))
        history["val_anomaly_auc"].append(float(val_out["anomaly_auc"]))
        history["val_alpha_entropy"].append(float(val_out["alpha_entropy_mean"]))
        history["val_rule_corr"].append(float(val_out["rule_corr_mean"]))
        history["val_dead_rules"].append(int(val_out["dead_rules"]))
        history["val_fuzzy_acc"].append(
            float(val_out["fuzzy_acc"])
            if np.isfinite(val_out["fuzzy_acc"]) else np.nan
        )
        history["val_fuzzy_f1"].append(
            float(val_out["fuzzy_f1"])
            if np.isfinite(val_out["fuzzy_f1"]) else np.nan
        )
        history["val_dl_acc"].append(
            float(val_out["dl_acc"])
            if np.isfinite(val_out["dl_acc"]) else np.nan
        )
        history["val_dl_f1"].append(
            float(val_out["dl_f1"])
            if np.isfinite(val_out["dl_f1"]) else np.nan
        )
        history["val_rule_activation_mean"].append(
            val_out["rule_activation_mean"].tolist()
            if isinstance(val_out["rule_activation_mean"], np.ndarray) else []
        )

        score = _compute_monitor_score(monitor_metric, train_out, val_out)

        history["monitor_score"].append(score)

        if scheduler is not None:
            if scheduler_mode == "plateau":
                scheduler.step(score)
            elif scheduler_mode == "epoch":
                scheduler.step()
            elif scheduler_mode == "epoch_float":
                scheduler.step(epoch + 1)

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_summary = (
            f"Epoch {epoch+1:03d}/{num_epochs} | "
            f"val_f1={val_out['anomaly_f1']:.4f} "
            f"val_auc={val_out['anomaly_auc']:.4f} "
            f"val_acc={val_out['anomaly_acc']:.4f} | "
            f"alpha_ent={val_out['alpha_entropy_mean']:.4f} "
            f"rule_corr={val_out['rule_corr_mean']:.4f} "
            f"dead_rules={val_out['dead_rules']} "
            f"dead_rules_ratio={val_out['dead_rules_ratio']:.4f} | "
            f"fuzzy_f1={val_out['fuzzy_f1']:.4f} "
            f"dl_f1={val_out['dl_f1']:.4f} | "
            f"monitor_score={score:.4f} | "
            f"lr={current_lr:.2e}"
        )
        logger.info(epoch_summary)

        if epoch < warmup_epochs:
            logger.info(
                f"  (Early stopping en espera - warmup {epoch+1}/{warmup_epochs})"
            )
            if score > (best_score + min_delta):
                best_score = score
                best_epoch = epoch + 1
                best_epoch_summary = epoch_summary
                torch.save(model.state_dict(), save_path)
        else:
            if score > (best_score + min_delta):
                best_score = score
                best_epoch = epoch + 1
                best_epoch_summary = epoch_summary
                epochs_without_improve = 0
                torch.save(model.state_dict(), save_path)
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= patience:
                logger.info(
                    f"Early stopping en epoch {epoch+1} (patience={patience})."
                )
                break

    # -----------------------------------------------------
    # Cargar mejor modelo
    # -----------------------------------------------------
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    # -----------------------------------------------------
    # Optimización de umbral por F1 en validación
    # -----------------------------------------------------
    _, val_outputs_default = evaluate_test(
        model=model,
        loader=val_loader,
        device=device,
    )

    threshold_metrics = optimize_threshold_by_f1(
        y_true=val_outputs_default["y_anomaly"],
        prob_anomaly=val_outputs_default["prob_anomaly"],
        n_points=101,
    )
    best_threshold = float(threshold_metrics["best_threshold"])

    model_cfg_final["training_kwargs"]["threshold"] = best_threshold

    return {
        "model_cfg": model_cfg_final,
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "history": history,
        "best_threshold": best_threshold,
        "pos_weight": pos_weight_tensor.detach().cpu().numpy(),
        "best_epoch": best_epoch,
        "best_epoch_summary": best_epoch_summary,
        "num_epochs": num_epochs,
    }
