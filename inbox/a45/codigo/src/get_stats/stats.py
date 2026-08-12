"""Estadísticas y métricas del modelo de anomalía."""

import json
import os
from typing import Any, Dict, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from src.training.trainer import evaluate_test



# =========================================================
# Visualización del historial de entrenamiento
# =========================================================

def _history_series(
    history_dict: Mapping[str, Sequence[Any]],
    primary_key: str,
    fallback_key: Optional[str] = None,
) -> Sequence[Any]:
    """Devuelve una serie del historial con fallback opcional."""
    if primary_key in history_dict:
        return history_dict[primary_key]
    if fallback_key is not None and fallback_key in history_dict:
        return history_dict[fallback_key]
    return []


def plot_training_history(
    history_dict: Mapping[str, Sequence[float]], output_dir: str = "models/metrics"
) -> None:
    """Visualiza historial para clasificación binaria de anomalía."""
    import os
    from src.utils.logging import get_logger

    logger = get_logger(__name__)

    os.makedirs(output_dir, exist_ok=True)
    logger.info("=" * 30)
    logger.info("VISUALIZACIÓN DEL ENTRENAMIENTO")
    logger.info("=" * 30)

    train_loss = _history_series(history_dict, "train_loss")
    val_loss = _history_series(history_dict, "val_loss")

    train_acc = _history_series(history_dict, "train_anomaly_acc", "train_acc")
    val_acc = _history_series(history_dict, "val_anomaly_acc", "val_acc")

    train_f1 = _history_series(history_dict, "train_anomaly_f1", "train_f1")
    val_f1 = _history_series(history_dict, "val_anomaly_f1", "val_f1")

    val_auc = _history_series(history_dict, "val_anomaly_auc")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(train_loss, label="Train", linewidth=2)
    axes[0, 0].plot(val_loss, label="Val", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training vs Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(train_acc, label="Train", linewidth=2)
    axes[0, 1].plot(val_acc, label="Val", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Training vs Validation Anomaly Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(train_f1, label="Train", linewidth=2)
    axes[1, 0].plot(val_f1, label="Val", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].set_title("Training vs Validation Anomaly F1")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(val_auc, linewidth=2, color="green")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("AUC-ROC")
    axes[1, 1].set_title("Validation Anomaly AUC-ROC")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Figura guardada: {output_path}")
    plt.close()


# =========================================================
# Optimización de umbral y análisis
# =========================================================

def threshold_diagnostics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    selected_threshold: float = 0.5,
    n_points: int = 101,
    output_dir: str = "models/metrics",
) -> Dict[str, Any]:
    """Grafica F1 vs threshold y evalúa el umbral seleccionado (en validación)."""
    import os
    from src.utils.logging import get_logger

    logger = get_logger(__name__)

    os.makedirs(output_dir, exist_ok=True)
    logger.info("=" * 30)
    logger.info("COMPARACIÓN DE UMBRAL train-test (solo diagnóstico, no se usa en test)")
    logger.info("=" * 30)

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    selected_threshold = float(selected_threshold)

    thresholds = np.linspace(0.0, 1.0, n_points)
    f1_scores = []

    for threshold in thresholds:
        preds = (y_prob >= threshold).astype(int)
        f1_scores.append(f1_score(y_true, preds, zero_division=0))

    best_idx_test = int(np.argmax(f1_scores))
    best_threshold_test = float(thresholds[best_idx_test])
    best_f1_test = float(f1_scores[best_idx_test])

    logger.info("Umbral operativo (seleccionado en validación):")
    logger.info("  - Threshold: %.3f", selected_threshold)

    logger.info("")
    logger.info("Referencia diagnóstica (óptimo sobre test, no usado en producción):")
    logger.info("  - Mejor threshold teórico sobre test: %.3f", best_threshold_test)
    logger.info("  - Mejor F1 alcanzable sobre test: %.4f", best_f1_test)
    logger.info("")
    logger.info("NOTA:")
    logger.info("  Este threshold se calcula unicamente con fines analiticos")
    logger.info("  y no se utiliza en produccion para evitar sobreajuste al test.")
    logger.info("=" * 30)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(thresholds, f1_scores, linewidth=2, color="blue")
    ax.axvline(
        selected_threshold,
        color="red",
        linestyle="--",
        label=f"Train threshold: {selected_threshold:.3f}",
    )
    ax.axvline(
        best_threshold_test,
        color="gray",
        linestyle=":",
        alpha=0.8,
        label=f"Best on test: {best_threshold_test:.3f}",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score vs Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "threshold_diagnostics.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Figura guardada: {output_path}")
    plt.close()

    return {
        "thresholds": thresholds,
        "f1_scores": f1_scores,
        "selected_threshold": selected_threshold,
        "best_threshold_test": best_threshold_test,
        "best_f1_test": best_f1_test,
    }


# =========================================================
# Guardado de resultados
# =========================================================

def save_model_results(
    model: torch.nn.Module,
    model_cfg: Dict[str, Any],
    history: Mapping[str, Sequence[float]],
    test_metrics: Mapping[str, float],
    threshold_used: float,
    output_path: str = "dnf_model_results.json",
) -> Dict[str, Any]:
    """Guarda resultados estructurados del modelo en JSON."""
    from src.utils.logging import get_logger

    logger = get_logger(__name__)

    total_params_local = sum(p.numel() for p in model.parameters())
    trainable_params_local = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_info = {
        "best_threshold": float(threshold_used),
        "model_config": model_cfg,
        "test_metrics": {
            "accuracy": float(test_metrics["accuracy"]),
            "fallo_f1": float(test_metrics["fallo_f1"]),
            "fallo_auc": float(test_metrics["fallo_auc"]),
            "fallo_precision": float(test_metrics.get("fallo_precision", 0)),
            "fallo_recall": float(test_metrics.get("fallo_recall", 0)),
            "nofallo_specificity": float(test_metrics.get("nofallo_specificity", 0)),
            "macro_f1": float(test_metrics.get("macro_f1", 0)),
            "macro_precision": float(test_metrics.get("macro_precision", 0)),
            "macro_recall": float(test_metrics.get("macro_recall", 0)),
            "nofallo_f1": float(test_metrics.get("nofallo_f1", 0)),
            "nofallo_precision": float(test_metrics.get("nofallo_precision", 0)),
            "nofallo_recall": float(test_metrics.get("nofallo_recall", 0)),
            "tp": int(test_metrics.get("tp", 0)),
            "fp": int(test_metrics.get("fp", 0)),
            "tn": int(test_metrics.get("tn", 0)),
            "fn": int(test_metrics.get("fn", 0)),
            "selected_threshold": float(threshold_used),
            "total_params": int(total_params_local),
            "trainable_params": int(trainable_params_local),
        },
        "training_history": {
            "train_loss": [float(x) for x in _history_series(history, "train_loss")],
            "val_loss": [float(x) for x in _history_series(history, "val_loss")],
            "train_anomaly_acc": [
                float(x)
                for x in _history_series(history, "train_anomaly_acc", "train_acc")
            ],
            "val_anomaly_acc": [
                float(x)
                for x in _history_series(history, "val_anomaly_acc", "val_acc")
            ],
            "train_anomaly_f1": [
                float(x)
                for x in _history_series(history, "train_anomaly_f1", "train_f1")
            ],
            "val_anomaly_f1": [
                float(x)
                for x in _history_series(history, "val_anomaly_f1", "val_f1")
            ],
            "val_anomaly_auc": [
                float(x) for x in _history_series(history, "val_anomaly_auc")
            ],
        },
    }

    with open(output_path, "w") as f:
        json.dump(model_info, f, indent=4)

    logger.info("Resultados guardados en '%s'", output_path)

    return model_info


# =========================================================
# Pipeline de métricas — 5 bloques
# =========================================================

def run_metrics_pipeline(
    model: torch.nn.Module,
    history: Mapping[str, Sequence[float]],
    test_loader: Any,
    model_cfg: Dict[str, Any],
    model_path: Optional[str] = None,
    default_threshold: float = 0.5,
    output_path: str = "dnf_model_results.json",
    train_out: Optional[Dict[str, Any]] = None,
    output_dir: str = "models/metrics",
    auto_use_train_outputs: bool = True,
    auto_use_train_threshold: bool = True,
    best_epoch: Optional[int] = None,
) -> Dict[str, Any]:
    required = [
        "plot_training_history",
        "evaluate_test",
        "threshold_diagnostics",
        "save_model_results",
    ]

    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError(
            "Faltan funciones necesarias para ejecutar el pipeline de métricas: "
            + ", ".join(missing)
            + ". Ejecuta primero las celdas de definiciones de métricas."
        )

    os.makedirs(output_dir, exist_ok=True)

    from src.utils.logging import get_logger

    logger = get_logger(__name__)

    # Generar gráficos de historial (sin reporting en consola)
    plot_training_history(history, output_dir=output_dir)
    device = next(model.parameters()).device

    if (
        model_path is not None
        and isinstance(model_path, str)
        and len(model_path) > 0
        and os.path.exists(model_path)
    ):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

    # Obtener outputs de test
    if auto_use_train_outputs and isinstance(train_out, dict):
        outputs_default = train_out.get("test_outputs")
    else:
        outputs_default = None

    if outputs_default is None:
        _, outputs_default = evaluate_test(
            model=model,
            loader=test_loader,
            device=device,
        )

    y_true = np.asarray(outputs_default["y_anomaly"]).astype(int)
    y_prob = np.asarray(outputs_default["prob_anomaly"]).astype(float)

    # Determinar threshold
    threshold_from_train = None
    if isinstance(train_out, dict):
        threshold_from_train = train_out.get("best_threshold")
    if threshold_from_train is None and isinstance(model_cfg, dict):
        threshold_from_train = (
            model_cfg.get("training_kwargs", {}).get("threshold")
            if isinstance(model_cfg.get("training_kwargs", {}), dict)
            else None
        )

    if auto_use_train_threshold and threshold_from_train is not None:
        threshold_used = float(threshold_from_train)
        threshold_source = "train_validation"
    else:
        threshold_used = float(default_threshold)
        threshold_source = "default"

    y_pred_used = (y_prob >= threshold_used).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_used, labels=[0, 1]).ravel()
    try:
        auc_value = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc_value = float("nan")

    prec_fallo = float(tp / max(tp + fp, 1))
    rec_fallo = float(tp / max(tp + fn, 1))
    prec_nofallo = float(tn / max(tn + fn, 1))
    rec_nofallo = float(tn / max(tn + fp, 1))
    f1_nofallo = float(
        2 * prec_nofallo * rec_nofallo / max(prec_nofallo + rec_nofallo, 1e-9)
    )
    f1_fallo = float(f1_score(y_true, y_pred_used, zero_division=0))
    macro_precision = float((prec_nofallo + prec_fallo) / 2)
    macro_recall = float((rec_nofallo + rec_fallo) / 2)
    macro_f1 = float((f1_nofallo + f1_fallo) / 2)

    metrics_used = {
        "accuracy": float(accuracy_score(y_true, y_pred_used)),
        "fallo_f1": f1_fallo,
        "fallo_auc": auc_value,
        "fallo_precision": prec_fallo,
        "fallo_recall": rec_fallo,
        "nofallo_specificity": float(tn / max(tn + fp, 1)),
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "nofallo_f1": f1_nofallo,
        "nofallo_precision": prec_nofallo,
        "nofallo_recall": rec_nofallo,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    # ================================================================
    # BLOQUE 2: Entrenamiento y Validación
    # ================================================================
    val_f1_series = _history_series(history, "val_anomaly_f1", "val_f1")
    val_auc_series = _history_series(history, "val_anomaly_auc")
    val_acc_series = _history_series(history, "val_anomaly_acc", "val_acc")
    monitor_score_series = _history_series(history, "monitor_score")

    if val_f1_series:
        n_epochs = len(val_f1_series)

        # Usar best_epoch de monitor_metric si está disponible,
        # sino fallback a argmax(monitor_score) para compatibilidad
        if best_epoch is not None and 1 <= best_epoch <= n_epochs:
            best_val_idx = best_epoch - 1  # 0-indexed
        else:
            best_val_idx = int(np.argmax(monitor_score_series))

        best_val_f1 = float(val_f1_series[best_val_idx])
        best_val_auc = float(val_auc_series[best_val_idx]) if val_auc_series else float("nan")
        best_val_acc = float(val_acc_series[best_val_idx]) if val_acc_series else float("nan")

        logger.info("=" * 30)
        logger.info("ENTRENAMIENTO Y VALIDACION")
        logger.info("=" * 30)
        logger.info("Mejor epoch seleccionada: %d/%d", best_val_idx + 1, n_epochs)
        logger.info("")
        logger.info("Métricas de validación (mejor epoch):")
        logger.info("  - Validation F1:    %.4f", best_val_f1)
        logger.info("  - Validation AUC:   %.4f", best_val_auc)
        logger.info("  - Validation Acc:   %.4f", best_val_acc)
        logger.info("")
        logger.info("Threshold seleccionado sobre validación:")
        logger.info("  - Threshold óptimo: %.3f", threshold_used)
    else:
        best_val_f1 = float("nan")
        best_val_auc = float("nan")
        best_val_acc = float("nan")

    # ================================================================
    # BLOQUE 3: Evaluación Final en Test
    # ================================================================
    # (Métricas ya calculadas en metrics_used: anomaly_*, macro_*, nofallo_*)

    logger.info("")
    logger.info("=" * 30)
    logger.info("EVALUACION FINAL EN TEST")
    logger.info("=" * 30)

    # 3A. Métricas globales (nivel dataset)
    logger.info("")
    logger.info("Métricas globales:")
    logger.info("  - Accuracy:        %.4f", metrics_used["accuracy"])
    logger.info("  - AUC-ROC:         %.4f", metrics_used["fallo_auc"])
    logger.info("  - Macro Precision: %.4f", metrics_used["macro_precision"])
    logger.info("  - Macro Recall:    %.4f", metrics_used["macro_recall"])
    logger.info("  - Macro F1:        %.4f", metrics_used["macro_f1"])

    # 3B. Métricas por clase
    logger.info("")
    logger.info("Clase No fallo:")
    logger.info("  - Precision: %.4f", metrics_used["nofallo_precision"])
    logger.info("  - Recall:    %.4f", metrics_used["nofallo_recall"])
    logger.info("  - F1-score:  %.4f", metrics_used["nofallo_f1"])

    logger.info("")
    logger.info("Clase Fallo:")
    logger.info("  - Precision: %.4f", metrics_used["fallo_precision"])
    logger.info("  - Recall:    %.4f", metrics_used["fallo_recall"])
    logger.info("  - F1-score:  %.4f", metrics_used["fallo_f1"])

    # 3C. Matriz de confusión (conteos)
    logger.info("")
    logger.info("Matriz de confusión:")
    logger.info("  - TP=%d", tp)
    logger.info("  - FP=%d", fp)
    logger.info("  - TN=%d", tn)
    logger.info("  - FN=%d", fn)

    # ================================================================
    # BLOQUE 4: Diagnóstico de Threshold
    # ================================================================
    threshold_report = threshold_diagnostics(
        y_true=y_true,
        y_prob=y_prob,
        selected_threshold=threshold_used,
        output_dir=output_dir,
    )

    # ================================================================
    # Visualizaciones
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred_used,
        labels=[0, 1],
        display_labels=["No fallo", "Fallo"],
        ax=axes[0, 0],
        cmap="Blues",
    )
    axes[0, 0].set_title(f"Matriz de Confusion (thr={threshold_used:.2f})")

    RocCurveDisplay.from_predictions(y_true, y_prob, ax=axes[0, 1])
    axes[0, 1].set_title("Curva ROC (Fallo)")

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    axes[1, 0].plot(recall_vals, precision_vals, linewidth=2)
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Curva Precision-Recall (Fallo)")
    axes[1, 0].grid(True, alpha=0.3)

    bins = np.linspace(0.0, 1.0, 41)
    axes[1, 1].hist(y_prob[y_true == 0], bins=bins, alpha=0.65, label="No fallo")
    axes[1, 1].hist(y_prob[y_true == 1], bins=bins, alpha=0.65, label="Fallo")
    axes[1, 1].axvline(
        threshold_used,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Threshold={threshold_used:.2f}",
    )
    axes[1, 1].set_xlabel("Score de Fallo")
    axes[1, 1].set_ylabel("Frecuencia")
    axes[1, 1].set_title("Distribucion de Scores de Fallo")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    axes[1, 1].legend(loc="best")

    plt.tight_layout()
    metrics_fig_path = os.path.join(output_dir, "evaluation_metrics.png")
    plt.savefig(metrics_fig_path, dpi=150, bbox_inches="tight")
    logger.info(f"Figura guardada: {metrics_fig_path}")
    plt.close()

    # Guardar resultados
    model_info_out = save_model_results(
        model=model,
        model_cfg=model_cfg,
        history=history,
        test_metrics=metrics_used,
        threshold_used=threshold_used,
        output_path=output_path,
    )

    logger.info("")
    logger.info("=" * 30)
    logger.info("PIPELINE DE METRICAS COMPLETADO EXITOSAMENTE")
    logger.info("=" * 30)

    return {
        "history_plotted": True,
        "evaluation": {
            "test_targets": y_true,
            "test_preds_proba": y_prob,
            "test_preds": y_pred_used,
            **metrics_used,
            "threshold_used": float(threshold_used),
            "threshold_source": threshold_source,
        },
        "threshold_report": threshold_report,
        "model_info": model_info_out,
        "paths": {
            "model_path": model_path,
            "results_json": output_path,
        },
    }
