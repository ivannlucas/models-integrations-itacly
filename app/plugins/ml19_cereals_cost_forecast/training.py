"""ml19 /train — port of the AI team's official pipeline (src/training/train.py::run_multihorizon_training).

Same steps, same order, same hyperparameters, with the stage helpers vendored verbatim in
_vendor/training_core.py:

1. Rows without target_ret_next (latest MAPA month + production front) are dropped: they only
   serve inference.
2. Multi-horizon targets ret_{h}m / dir_{h}m (h = 1, 2, 3).
3. STRICT feature selection on train (date <= TRAIN_END = 2019-12-31) against ret_1m, shared by
   every horizon.
4. Lasso alpha calibrated once on train (TimeSeriesSplit(5), max Pearson r).
5-6. Expanding-window walk-forward from TRAIN_END for Ridge/Lasso/XGBoost (regression) and
   LogReg/RandomForest/XGBoost (classification).
7. Best model per horizon: max Sharpe (regression) / max AUC (classification).
8. Production artifacts: the selected models refit on ALL data with their scaler
   (_save_best_models), plus training_meta.json and model_selection_audit.json.

TRAIN_END stays fixed like in the original: it only bounds feature selection and the start of
the walk-forward test. The production models are always refit on every row with a target, so a
retraining on an updated dataset_v7_fe.csv learns the new months.

Plugin adaptations, all deliberate:

* The dataset comes from /train's ``data_path`` (the dataset_v7_fe.csv the ETL produced); the
  original reads the fixed data/processed/auto/dataset_v7_fe.csv.
* ``_seed_everything(42)`` (random/np.random seed + PYTHONHASHSEED) is not called: every
  estimator receives random_state=42 explicitly and nothing in the pipeline draws from the
  global RNGs; the server is shared. Verified bit-identical to the original without it.
* No .sha256 sidecars, walk-forward traces, results/splits CSVs or console report: they are
  diagnostics of the original repo and do not change the artifacts.
"""
# pylint: disable=line-too-long  # repo limit is 120 (.flake8), enforced by flake8
from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from app.plugins.ml19_cereals_cost_forecast._vendor import training_core as core
from app.plugins.ml19_cereals_cost_forecast.constants import (
    DATE_COL,
    MODEL_SELECTION_AUDIT_FILENAME,
    TRAINING_META_FILENAME,
    clf_artifact_filename,
    reg_artifact_filename,
)

logger = logging.getLogger(__name__)

REG_SELECTION_METRIC = "sharpe"
CLF_SELECTION_METRIC = "auc"
# Columns train.py reads besides the features (targets come from target_media).
REQUIRED_TRAIN_COLUMNS = (DATE_COL, "target_media", core.TARGET_REG, core.TARGET_CLF)


@dataclass
class TrainingResult:
    """Production artifacts and walk-forward metrics of one retraining."""

    features: list[str]
    lasso_alpha: float
    best_reg: dict[int, str]
    best_clf: dict[int, str]
    reg_results: dict[tuple[str, int], dict]
    clf_results: dict[tuple[str, int], dict]
    artifacts: dict[str, dict] = field(default_factory=dict)   # filename -> artifact dict
    meta: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    n_rows: int = 0
    data_start: str = ""
    data_end: str = ""
    training_time_s: float = 0.0


def _reg_factories(lasso_alpha: float) -> dict:
    return {
        "Ridge": lambda: Ridge(alpha=core.RIDGE_ALPHA),
        "Lasso": lambda: Lasso(alpha=lasso_alpha, max_iter=20000),
        "XGBoost": lambda: xgb.XGBRegressor(**core.XGB_PARAMS),
    }


CLF_FACTORIES = {
    "LogReg": lambda: LogisticRegression(C=core.LR_C, max_iter=1000),
    "RandomForest": lambda: RandomForestClassifier(**core.RF_PARAMS),
    "XGBoost": lambda: xgb.XGBClassifier(**core.XGBC_PARAMS),
}


def validate_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fail with a clear 400 before pandas/sklearn fail with an obscure message."""
    missing = [c for c in REQUIRED_TRAIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"El CSV de entrenamiento no trae las columnas {missing}. Debe ser el dataset_v7_fe.csv "
            "que genera el pipeline del equipo de IA (python -m src.main data_processing)."
        )
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if df[DATE_COL].isna().any():
        raise ValueError("La columna 'date' tiene valores vacíos o que no son fechas.")
    with_target = df.dropna(subset=[core.TARGET_REG])
    if (with_target[DATE_COL] <= core.TRAIN_END).sum() == 0 or (with_target[DATE_COL] > core.TRAIN_END).sum() == 0:
        raise ValueError(
            f"El dataset debe tener filas con target antes y después de {core.TRAIN_END} (selección de "
            "features y walk-forward del procedimiento original)."
        )
    return df


def train_multihorizon(df: pd.DataFrame) -> TrainingResult:
    """run_multihorizon_training with the default horizons and selection metrics."""
    started = time.perf_counter()
    df = validate_training_frame(df)
    reg_horizons, clf_horizons = core.REG_HORIZONS, core.CLF_HORIZONS
    all_horizons = sorted(set(reg_horizons) | set(clf_horizons))
    train_end = core.TRAIN_END

    # -- 1. rows without target only serve inference
    df = df.dropna(subset=[core.TARGET_REG]).reset_index(drop=True)
    df[core.TARGET_CLF] = df[core.TARGET_CLF].astype(int)

    # -- 2. multi-horizon targets
    df = core.build_multihorizon_targets(df, all_horizons)

    # -- 3. STRICT feature selection on train, shared by every horizon
    ref_target = f"ret_{min(reg_horizons)}m"
    df_ref = df.dropna(subset=[ref_target]).copy()
    train_mask_ref = df_ref["date"] <= train_end
    features, _ = core.select_features(
        df_ref, min_corr=core.MIN_CORR, target_col=ref_target, strict=True, train_mask=train_mask_ref,
    )
    if not features:
        raise ValueError("La selección STRICT no ha dejado ninguna feature: el dataset no es válido para entrenar.")

    # -- 4. Lasso alpha, once, on train
    df_ref_clean = df_ref.dropna(subset=features + [ref_target]).copy()
    train_df = df_ref_clean[df_ref_clean["date"] <= train_end]
    lasso_alpha = core.best_lasso_alpha(StandardScaler().fit_transform(train_df[features].values),
                                        train_df[ref_target].values)

    # -- 5. walk-forward regression
    reg_models = _reg_factories(lasso_alpha)
    reg_results: dict[tuple[str, int], dict] = {}
    for h in reg_horizons:
        target_col = f"ret_{h}m"
        for mname, factory in reg_models.items():
            y_te, preds, dates = core.walk_forward_regression(df, features, target_col, train_end, factory)
            if len(y_te) == 0:
                continue
            metrics = core.eval_regression(y_te, preds, f"{mname}_h{h}m")
            metrics.update(y_test=y_te, preds=preds, dates=dates)
            reg_results[(mname, h)] = metrics

    # -- 6. walk-forward classification
    clf_results: dict[tuple[str, int], dict] = {}
    for h in clf_horizons:
        target_col = f"dir_{h}m"
        for mname, factory in CLF_FACTORIES.items():
            y_te, preds, probs, dates = core.walk_forward_classification(df, features, target_col, train_end, factory)
            if len(y_te) == 0:
                continue
            metrics = core.eval_classification(y_te, preds, probs, f"{mname}_h{h}m")
            metrics.update(y_test=y_te, dates=dates)
            clf_results[(mname, h)] = metrics

    # -- 7. best model per horizon and task
    best_reg = {
        h: core._select_best_model(  # pylint: disable=protected-access
            {m: reg_results[(m, h)] for m in reg_models if (m, h) in reg_results}, REG_SELECTION_METRIC, "reg")
        for h in reg_horizons if any((m, h) in reg_results for m in reg_models)
    }
    best_clf = {
        h: core._select_best_model(  # pylint: disable=protected-access
            {m: clf_results[(m, h)] for m in CLF_FACTORIES if (m, h) in clf_results}, CLF_SELECTION_METRIC, "clf")
        for h in clf_horizons if any((m, h) in clf_results for m in CLF_FACTORIES)
    }

    result = TrainingResult(
        features=features, lasso_alpha=lasso_alpha, best_reg=best_reg, best_clf=best_clf,
        reg_results=reg_results, clf_results=clf_results, n_rows=int(len(df)),
        data_start=str(df["date"].min().date()), data_end=str(df["date"].max().date()),
    )
    # -- 8. production artifacts on ALL data
    _build_production_artifacts(df, result, train_end, reg_models)
    result.training_time_s = round(time.perf_counter() - started, 2)
    return result


def _provenance(wf_metrics: dict, keys: tuple[str, ...], n_samples: int) -> dict:
    return {
        "training_scope": "all_data",
        "metrics_source": "walk_forward",
        "walk_forward_metrics": {k: float(wf_metrics.get(k, 0)) for k in keys},
        "note": "Este modelo fue reentrenado sobre ALL data para produccion. Las metricas walk-forward corresponden a la validacion out-of-sample.",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples_all_data": n_samples,
    }


def _build_production_artifacts(df: pd.DataFrame, result: TrainingResult, train_end: str, reg_models: dict) -> None:
    """train.py::_save_best_models + _save_model_selection_audit, kept in memory."""
    features = result.features
    for h, mname in result.best_reg.items():
        target_col = f"ret_{h}m"
        df_c = df.dropna(subset=features + [target_col]).copy()
        x_all, y_all = df_c[features].values, df_c[target_col].values
        sc = StandardScaler()
        mdl = reg_models[mname]()
        mdl.fit(sc.fit_transform(x_all), y_all)
        result.artifacts[reg_artifact_filename(h, mname)] = {
            "model": mdl, "scaler": sc, "features": features, "task": "reg", "horizon": h, "model_name": mname,
            "provenance": _provenance(result.reg_results.get((mname, h), {}),
                                      ("corr", "dir_acc", "sharpe", "mae", "rmse", "r2"), len(x_all)),
        }
    for h, mname in result.best_clf.items():
        target_col = f"dir_{h}m"
        df_c = df.dropna(subset=features + [target_col]).copy()
        x_all, y_all = df_c[features].values, df_c[target_col].values.astype(int)
        sc = StandardScaler()
        mdl = CLF_FACTORIES[mname]()
        mdl.fit(sc.fit_transform(x_all), y_all)
        result.artifacts[clf_artifact_filename(h, mname)] = {
            "model": mdl, "scaler": sc, "features": features, "task": "clf", "horizon": h, "model_name": mname,
            "provenance": _provenance(result.clf_results.get((mname, h), {}),
                                      ("acc", "f1", "auc", "brier", "sharpe"), len(x_all)),
        }
    result.meta = {
        "features": features,
        "lasso_alpha": result.lasso_alpha,
        "train_end": train_end,
        "ridge_alpha": core.RIDGE_ALPHA,
        "xgb_params": core.XGB_PARAMS,
        "best_reg": {str(h): m for h, m in result.best_reg.items()},
        "best_clf": {str(h): m for h, m in result.best_clf.items()},
        "selection_metrics": {"regression": REG_SELECTION_METRIC, "classification": CLF_SELECTION_METRIC},
        "production_model_scope": {
            "regression": ["Ridge", "Lasso", "XGBoost"],
            "classification": ["LogReg", "RandomForest", "XGBoost"],
        },
        "experimental_models": {
            "DRNN": "Disponible en flujo legacy (run_walk_forward include_drnn=True); excluido del ranking oficial por estabilidad/rendimiento.",
        },
        "provenance": {
            "metrics_source": "walk_forward",
            "artifact_training_scope": "all_data",
            "note": (
                "Las metricas reportadas en reg_results.csv, clf_results.csv y "
                "model_selection_audit.json provienen de la validacion walk-forward "
                "(out-of-sample). Los artefactos .pkl son modelos reentrenados sobre "
                "ALL data para maximizar la informacion disponible en produccion. "
                "Las metricas walk-forward se incluyen tambien en el bloque 'provenance' "
                "de cada artefacto .pkl para trazabilidad directa."
            ),
        },
    }
    reg_rows = [{
        "task": "reg", "horizon": int(h), "model_name": m, "selected": bool(result.best_reg.get(h) == m),
        "selection_metric": REG_SELECTION_METRIC, "selection_value": float(v[REG_SELECTION_METRIC]),
        "corr": float(v["corr"]), "dir_acc": float(v["dir_acc"]), "mae": float(v["mae"]),
        "rmse": float(v["rmse"]), "r2": float(v["r2"]),
    } for (m, h), v in result.reg_results.items()]
    clf_rows = [{
        "task": "clf", "horizon": int(h), "model_name": m, "selected": bool(result.best_clf.get(h) == m),
        "selection_metric": CLF_SELECTION_METRIC, "selection_value": float(v[CLF_SELECTION_METRIC]),
        "acc": float(v["acc"]), "f1": float(v["f1"]), "brier": float(v["brier"]), "sharpe": float(v["sharpe"]),
    } for (m, h), v in result.clf_results.items()]
    result.audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "criteria": {"regression": f"max_{REG_SELECTION_METRIC}", "classification": f"max_{CLF_SELECTION_METRIC}"},
        "best_reg": result.meta["best_reg"],
        "best_clf": result.meta["best_clf"],
        "regression_candidates": reg_rows,
        "classification_candidates": clf_rows,
    }


def selected_metrics(result: TrainingResult) -> dict[str, Any]:
    """Flat walk-forward metrics of the selected model per horizon (version.json / MLflow)."""
    flat: dict[str, Any] = {}
    for h, m in result.best_reg.items():
        v = result.reg_results[(m, h)]
        for key in ("corr", "dir_acc", "sharpe", "mae", "rmse", "r2"):
            flat[f"reg_h{h}m_{key}"] = float(v[key])
    for h, m in result.best_clf.items():
        v = result.clf_results[(m, h)]
        for key in ("acc", "f1", "auc", "brier", "sharpe"):
            flat[f"clf_h{h}m_{key}"] = float(v[key])
    flat["n_features"] = len(result.features)
    flat["lasso_alpha"] = float(result.lasso_alpha)
    return flat


def save_training_artifacts(directory: Path, result: TrainingResult) -> None:
    """Write the bundle with the base artifact's filenames (pickle like io.py::save_pickle)."""
    directory.mkdir(parents=True, exist_ok=True)
    for filename, artifact in result.artifacts.items():
        with open(directory / filename, "wb") as fh:
            pickle.dump(artifact, fh)
    with open(directory / TRAINING_META_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(result.meta, fh, indent=2, ensure_ascii=False, default=str)
    with open(directory / MODEL_SELECTION_AUDIT_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(result.audit, fh, indent=2, ensure_ascii=False, default=str)
