"""Production reporting engine for DATAGIA final models.

Generates mirrored train/test diagnostics, plots, and presentation-ready reports
for the 6 selected final models (H1/H2/H3 x regression/classification).
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to avoid Tkinter crashes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, roc_auc_score

from config.config import DATA_PROCESSED_DIR, MODEL_ARTIFACTS_DIR, MODEL_METRICS_DIR, PROJECT_ROOT
from src.data_processing.prepare_data import CUT_DATE, get_prepared_data
from src.utils.logging import get_logger

logger = get_logger(__name__)

ARTIFACT_DIR = MODEL_ARTIFACTS_DIR
METRICS_DIR = MODEL_METRICS_DIR
MODEL_METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
DATASET_PATH = DATA_PROCESSED_DIR / "dataset_entrenamiento_final.csv"
HORIZONS = (1, 2, 3)

sns.set_theme(style="whitegrid", context="talk")


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _file_sha256(path: Path) -> str | None:
    """SHA-256 del archivo con line endings normalizados a LF.

    Normalizar CRLF -> LF antes de hashear garantiza el mismo resultado
    en Windows (autocrlf=true) y Linux/Mac, de modo que el hash almacenado
    en full_report.json coincide con el blob que cualquier auditor descarga.
    """
    if not path.exists():
        return None
    with path.open("rb") as f:
        content = f.read()
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def _load_selected_models() -> dict[str, dict[str, str]]:
    model_map: dict[str, dict[str, str]] = {}
    if MODEL_METADATA_PATH.exists():
        with MODEL_METADATA_PATH.open(encoding="utf-8") as f:
            md = json.load(f)
        selected = md.get("selected_models", {})
        for h in HORIZONS:
            h_key = f"H{h}"
            if h_key in selected:
                reg_path = selected[h_key]["regression"]["model_path"]
                clf_path = selected[h_key]["classification"]["model_path"]
                model_map[h_key] = {"reg": reg_path, "clf": clf_path}

    for h in HORIZONS:
        h_key = f"H{h}"
        if h_key not in model_map:
            model_map[h_key] = {
                "reg": str(ARTIFACT_DIR / f"datagia_best_h{h}_reg.joblib"),
                "clf": str(ARTIFACT_DIR / f"datagia_best_h{h}_clf.joblib"),
            }

    return model_map


def _resolve_model_path(path_value: str) -> Path:
    r"""Resolve model path in a cross-platform way.

    Handles absolute Windows paths (C:\Users\...) stored in metadata by
    extracting only the filename and searching in ARTIFACT_DIR.
    """
    p = Path(path_value)

    # Always try ARTIFACT_DIR first using just the filename
    candidate = ARTIFACT_DIR / p.name
    if candidate.exists():
        return candidate

    # Fallback: try other locations
    if p.is_absolute():
        if p.exists():
            return p
        candidate = ARTIFACT_DIR / p.name
        if candidate.exists():
            return candidate

    candidates = [
        PROJECT_ROOT / p,
        p,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return ARTIFACT_DIR / p.name


def _load_aligned_dates_for_horizon(h: int) -> tuple[pd.Series, pd.Series]:
    """Recreate aligned temporal indexes for train/test sets in Hh."""
    df = pd.read_csv(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values(["date", "provincia", "cereal_predominante"], kind="mergesort")
    df = df.reset_index(drop=True)

    tgt = f"precio_provincial_TARGET_H{h}"
    base = "precio_provincial_lag_1"

    valid = pd.to_numeric(df[tgt], errors="coerce").notna() & pd.to_numeric(
        df[base], errors="coerce"
    ).notna() & pd.to_numeric(df[base], errors="coerce").ne(0)

    filtered = df.loc[valid].copy()
    train_dates = filtered.loc[filtered["date"] < CUT_DATE, "date"].reset_index(drop=True)
    test_dates = filtered.loc[filtered["date"] >= CUT_DATE, "date"].reset_index(drop=True)
    return train_dates, test_dates


def _extract_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    importances: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        importances = np.abs(coef).ravel()
    elif hasattr(model, "estimators_"):
        parts: list[np.ndarray] = []
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                parts.append(np.asarray(est.feature_importances_, dtype=float))
            elif hasattr(est, "coef_"):
                parts.append(np.abs(np.asarray(est.coef_, dtype=float).ravel()))
        if parts:
            importances = np.mean(np.vstack(parts), axis=0)
    elif hasattr(model, "calibrated_classifiers_"):
        parts = []
        for cal in model.calibrated_classifiers_:
            est = getattr(cal, "estimator", None)
            if est is None:
                continue
            if hasattr(est, "feature_importances_"):
                parts.append(np.asarray(est.feature_importances_, dtype=float))
            elif hasattr(est, "coef_"):
                parts.append(np.abs(np.asarray(est.coef_, dtype=float).ravel()))
        if parts:
            importances = np.mean(np.vstack(parts), axis=0)

    if importances is None or len(importances) != len(feature_names):
        return pd.DataFrame(columns=["feature", "importance"])

    out = pd.DataFrame({"feature": feature_names, "importance": importances})
    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    return out


def _plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    logger.info("Generando grafica feature importance -> %s", output_path)

    if importance_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Feature importance no disponible", ha="center", va="center")
        ax.set_title(title)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    top10 = importance_df.head(10).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=top10, x="importance", y="feature", ax=ax, hue="feature", palette="viridis", legend=False)
    ax.set_title(title)
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Variable")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_residuals_temporal(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    logger.info("Generando grafica residuos temporales -> %s", output_path)

    residual = np.abs(y_true - y_pred)
    df = pd.DataFrame({"date": pd.to_datetime(dates), "abs_error": residual})
    series = df.groupby("date", as_index=False)["abs_error"].mean().sort_values("date")
    series["rolling_3m"] = series["abs_error"].rolling(window=3, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(series["date"], series["abs_error"], label="Error absoluto medio", alpha=0.5)
    ax.plot(series["date"], series["rolling_3m"], label="Tendencia 3M", linewidth=2)
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), color="red", alpha=0.12, label="Crisis 2022")
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Error absoluto")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_reliability_diagram(
    y_train: np.ndarray,
    p_train: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    logger.info("Generando reliability diagram -> %s", output_path)

    train_frac, train_mean = calibration_curve(y_train, p_train, n_bins=10, strategy="quantile")
    test_frac, test_mean = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Calibracion perfecta")
    ax.plot(train_mean, train_frac, marker="o", label="Train")
    ax.plot(test_mean, test_frac, marker="s", label="Test")
    ax.set_title(title)
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    logger.info("Generando matriz de confusion -> %s", output_path)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Baja", "Sube"],
        yticklabels=["Baja", "Sube"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Pearson_R": _safe_pearson(y_true, y_pred),
        "DA": float(np.mean((y_true > 0).astype(int) == (y_pred > 0).astype(int))),
    }


def _clf_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    return {
        "Accuracy": acc,
        "AUC": _safe_auc(y_true, y_prob),
        # DA is kept for backward compatibility with existing reports.
        # In classification it is intentionally the same value as Accuracy.
        "DA": acc,
    }


def _build_summary_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Resumen de Performance - DATAGIA")
    lines.append("")
    lines.append(f"Generado UTC: {report['generated_at_utc']}")
    lines.append("")

    lines.append("## Regresion (Train vs Test)")
    lines.append("")
    lines.append("| Horizon | MAE Train | MAE Test | Gap MAE | Pearson Train | Pearson Test | DA Train | DA Test |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for h in HORIZONS:
        r = report["horizons"][f"H{h}"]["regression"]
        lines.append(
            "| H{} | {:.6f} | {:.6f} | {:.6f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                h,
                r["train"]["MAE"],
                r["test"]["MAE"],
                r["gaps"]["MAE_test_minus_train"],
                r["train"]["Pearson_R"],
                r["test"]["Pearson_R"],
                r["train"]["DA"],
                r["test"]["DA"],
            )
        )

    lines.append("")
    lines.append("## Clasificacion (Train vs Test)")
    lines.append("")
    lines.append("Nota: en clasificacion, DA coincide numericamente con Accuracy (DA = Accuracy).")
    lines.append("")
    lines.append("| Horizon | Acc Train | Acc Test | Gap Acc | AUC Train | AUC Test | Gap AUC | DA (=Acc) Train | DA (=Acc) Test |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for h in HORIZONS:
        c = report["horizons"][f"H{h}"]["classification"]
        lines.append(
            "| H{} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                h,
                c["train"]["Accuracy"],
                c["test"]["Accuracy"],
                c["gaps"]["Accuracy_test_minus_train"],
                c["train"]["AUC"],
                c["test"]["AUC"],
                c["gaps"]["AUC_test_minus_train"],
                c["train"]["DA"],
                c["test"]["DA"],
            )
        )

    lines.append("")
    lines.append("## Hallazgos de Robustez")
    lines.append("")
    lines.append("- Gap bajo entre Train/Test sugiere buena generalizacion.")
    lines.append("- Gap alto (sobre todo en AUC o MAE) sugiere posible sobreajuste.")
    lines.append("- Revisar la pendiente de residuos en 2022 para detectar sensibilidad a shocks.")

    return "\n".join(lines)


def generate_full_report() -> dict[str, Any]:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    model_paths = _load_selected_models()
    model_metadata: dict[str, Any] = {}
    if MODEL_METADATA_PATH.exists():
        with MODEL_METADATA_PATH.open(encoding="utf-8") as f:
            model_metadata = json.load(f)

    full_report_path = METRICS_DIR / "full_report.json"
    summary_md_path = METRICS_DIR / "summary_performance.md"
    error_path = METRICS_DIR / "full_report_error.json"

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts_dir": ARTIFACT_DIR.name,
        "metrics_dir": METRICS_DIR.name,
        "evidence_bundle_id": model_metadata.get("evidence_bundle_id"),
        "evidence_manifest_sha256": model_metadata.get("evidence_manifest_sha256"),
        "source_model_metadata_path": MODEL_METADATA_PATH.name,
        "source_model_metadata_sha256": _file_sha256(MODEL_METADATA_PATH),
        "horizons": {},
        "plots": [],
        "timing_seconds": {},
    }

    # ------------------------------------------------------------------ #
    # FASE 1 – Carga de modelos, datos y calculo de metricas              #
    # El JSON numerico se escribe al terminar esta fase, antes de generar #
    # cualquier grafico. Asi full_report.json siempre refleja las metricas #
    # aunque falle alguna visualizacion posterior.                         #
    # ------------------------------------------------------------------ #
    plot_cache: dict[str, Any] = {}
    t0_total = time.perf_counter()

    for h in HORIZONS:
        h_key = f"H{h}"
        t0_h = time.perf_counter()
        logger.info("===== [INICIO FASE 1] Diagnostico %s =====", h_key)

        t0 = time.perf_counter()
        reg_model = joblib.load(_resolve_model_path(model_paths[h_key]["reg"]))
        clf_model = joblib.load(_resolve_model_path(model_paths[h_key]["clf"]))
        logger.info("[%s] Modelos cargados en %.2fs", h_key, time.perf_counter() - t0)

        t0 = time.perf_counter()
        X_train_r, X_test_r, y_train_r, y_test_r = get_prepared_data(h, "regresion")
        logger.info("[%s] Datos regresion listos en %.2fs", h_key, time.perf_counter() - t0)

        t0 = time.perf_counter()
        X_train_c, X_test_c, y_train_c, y_test_c = get_prepared_data(h, "clasificacion")
        logger.info("[%s] Datos clasificacion listos en %.2fs", h_key, time.perf_counter() - t0)

        t0 = time.perf_counter()
        _, test_dates = _load_aligned_dates_for_horizon(h)
        logger.info("[%s] Fechas alineadas en %.2fs", h_key, time.perf_counter() - t0)

        t0 = time.perf_counter()
        pred_train_r = reg_model.predict(X_train_r)
        pred_test_r = reg_model.predict(X_test_r)
        pred_train_c = clf_model.predict(X_train_c)
        pred_test_c = clf_model.predict(X_test_c)
        prob_train_c = clf_model.predict_proba(X_train_c)[:, 1]
        prob_test_c = clf_model.predict_proba(X_test_c)[:, 1]
        logger.info("[%s] Predicciones completadas en %.2fs", h_key, time.perf_counter() - t0)

        t0 = time.perf_counter()
        reg_train = _reg_metrics(y_train_r.to_numpy(), pred_train_r)
        reg_test = _reg_metrics(y_test_r.to_numpy(), pred_test_r)
        clf_train = _clf_metrics(y_train_c.to_numpy(), pred_train_c, prob_train_c)
        clf_test = _clf_metrics(y_test_c.to_numpy(), pred_test_c, prob_test_c)
        logger.info("[%s] Metricas calculadas en %.2fs", h_key, time.perf_counter() - t0)

        reg_info = {
            "train": reg_train,
            "test": reg_test,
            "gaps": {
                "MAE_test_minus_train": reg_test["MAE"] - reg_train["MAE"],
                "Pearson_test_minus_train": reg_test["Pearson_R"] - reg_train["Pearson_R"],
                "DA_test_minus_train": reg_test["DA"] - reg_train["DA"],
            },
        }

        clf_info = {
            "train": clf_train,
            "test": clf_test,
            "gaps": {
                "Accuracy_test_minus_train": clf_test["Accuracy"] - clf_train["Accuracy"],
                "AUC_test_minus_train": clf_test["AUC"] - clf_train["AUC"],
                "DA_test_minus_train": clf_test["DA"] - clf_train["DA"],
            },
        }

        report["horizons"][h_key] = {
            "regression": reg_info,
            "classification": clf_info,
            "model_paths": model_paths[h_key],
        }

        elapsed_h = time.perf_counter() - t0_h
        report["timing_seconds"][f"{h_key}_fase1_metricas"] = round(elapsed_h, 3)
        logger.info("[%s] FASE 1 completada en %.2fs", h_key, elapsed_h)

        # Guardar lo necesario para la fase de graficos (evita recargar datos)
        plot_cache[h_key] = {
            "reg_imp": _extract_feature_importance(reg_model, X_train_r.columns.tolist()),
            "clf_imp": _extract_feature_importance(clf_model, X_train_c.columns.tolist()),
            "test_dates": test_dates,
            "y_test_r": y_test_r.to_numpy(),
            "pred_test_r": pred_test_r,
            "y_train_c": y_train_c.to_numpy(),
            "prob_train_c": prob_train_c,
            "y_test_c": y_test_c.to_numpy(),
            "prob_test_c": prob_test_c,
            "pred_test_c": pred_test_c,
        }

    # Escritura atomica del JSON de metricas ANTES de generar graficos.
    # Garantiza que full_report.json siempre existe y es valido aunque
    # alguna visualizacion posterior falle o tarde demasiado.
    try:
        tmp_path = full_report_path.with_suffix(".json.tmp")
        logger.info("Guardando reporte numerico (pre-graficos) -> %s", full_report_path)
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(_to_jsonable(report), f, indent=2, ensure_ascii=False)
        tmp_path.replace(full_report_path)
        logger.info("full_report.json escrito correctamente (metricas garantizadas).")
        if error_path.exists():
            error_path.unlink()
    except Exception as exc:
        error_state = {
            "status": "ERROR",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_bundle_id": model_metadata.get("evidence_bundle_id"),
            "source_model_metadata_sha256": _file_sha256(MODEL_METADATA_PATH),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        logger.error("Error al escribir reporte numerico: %s", exc)
        with error_path.open("w", encoding="utf-8") as f:
            json.dump(error_state, f, indent=2, ensure_ascii=False)
        raise

    # ------------------------------------------------------------------ #
    # FASE 2 – Generacion de graficos                                     #
    # Cada grafico va en su propio try/except. Un fallo de visualizacion  #
    # se registra como WARNING pero no impide el reporte numerico.        #
    # ------------------------------------------------------------------ #
    plot_errors: list[str] = []

    for h in HORIZONS:
        h_key = f"H{h}"
        t0_h = time.perf_counter()
        logger.info("===== [INICIO FASE 2] Graficos %s =====", h_key)
        d = plot_cache[h_key]

        reg_imp_path = METRICS_DIR / f"feature_importance_{h_key}_reg.png"
        clf_imp_path = METRICS_DIR / f"feature_importance_{h_key}_clf.png"
        residual_path = METRICS_DIR / f"residuos_temporales_{h_key}_reg_test.png"
        rel_path = METRICS_DIR / f"reliability_{h_key}_clf.png"
        cm_path = METRICS_DIR / f"confusion_matrix_{h_key}_clf_test.png"

        try:
            _plot_feature_importance(d["reg_imp"], f"Top 10 Variables {h_key} Regresion", reg_imp_path)
        except Exception as exc:
            logger.warning("[%s] feature_importance_reg fallido: %s", h_key, exc)
            plot_errors.append(f"{h_key}/feature_importance_reg: {exc}")

        try:
            _plot_feature_importance(d["clf_imp"], f"Top 10 Variables {h_key} Clasificacion", clf_imp_path)
        except Exception as exc:
            logger.warning("[%s] feature_importance_clf fallido: %s", h_key, exc)
            plot_errors.append(f"{h_key}/feature_importance_clf: {exc}")

        try:
            _plot_residuals_temporal(
                dates=d["test_dates"],
                y_true=d["y_test_r"],
                y_pred=d["pred_test_r"],
                title=f"Residuos Temporales {h_key} (Test)",
                output_path=residual_path,
            )
        except Exception as exc:
            logger.warning("[%s] residuos_temporales fallido: %s", h_key, exc)
            plot_errors.append(f"{h_key}/residuos_temporales: {exc}")

        try:
            _plot_reliability_diagram(
                y_train=d["y_train_c"],
                p_train=d["prob_train_c"],
                y_test=d["y_test_c"],
                p_test=d["prob_test_c"],
                title=f"Reliability Diagram {h_key}",
                output_path=rel_path,
            )
        except Exception as exc:
            logger.warning("[%s] reliability_diagram fallido: %s", h_key, exc)
            plot_errors.append(f"{h_key}/reliability_diagram: {exc}")

        try:
            _plot_confusion(
                y_true=d["y_test_c"],
                y_pred=d["pred_test_c"],
                title=f"Matriz de Confusion {h_key} Clasificacion (Test)",
                output_path=cm_path,
            )
        except Exception as exc:
            logger.warning("[%s] confusion_matrix fallido: %s", h_key, exc)
            plot_errors.append(f"{h_key}/confusion_matrix: {exc}")

        elapsed_h = time.perf_counter() - t0_h
        report["timing_seconds"][f"{h_key}_fase2_graficos"] = round(elapsed_h, 3)
        logger.info("[%s] FASE 2 completada en %.2fs", h_key, elapsed_h)

        report["plots"].extend(
            [
                reg_imp_path.name,
                clf_imp_path.name,
                residual_path.name,
                rel_path.name,
                cm_path.name,
            ]
        )

    elapsed_total = time.perf_counter() - t0_total
    report["timing_seconds"]["total"] = round(elapsed_total, 3)
    if plot_errors:
        report["plot_warnings"] = plot_errors
    logger.info("Duracion total: %.2fs", elapsed_total)

    # Escritura final del JSON con la lista de graficos y tiempos totales.
    try:
        tmp_path = full_report_path.with_suffix(".json.tmp")
        logger.info("Guardando reporte final -> %s", full_report_path)
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(_to_jsonable(report), f, indent=2, ensure_ascii=False)
        tmp_path.replace(full_report_path)

        logger.info("Guardando resumen markdown -> %s", summary_md_path)
        summary_text = _build_summary_markdown(report)
        summary_md_path.write_text(summary_text, encoding="utf-8")

        if error_path.exists():
            error_path.unlink()

        if plot_errors:
            logger.warning(
                "Reporte generado con %d advertencia(s) en graficos: %s",
                len(plot_errors),
                plot_errors,
            )
        else:
            logger.info("Reporte completo generado correctamente.")
        return report

    except Exception as exc:
        error_state = {
            "status": "ERROR",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_bundle_id": model_metadata.get("evidence_bundle_id"),
            "source_model_metadata_sha256": _file_sha256(MODEL_METADATA_PATH),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        logger.error("Error al escribir reporte final: %s", exc)
        with error_path.open("w", encoding="utf-8") as f:
            json.dump(error_state, f, indent=2, ensure_ascii=False)
        raise


def main() -> None:
    generate_full_report()


if __name__ == "__main__":
    main()
