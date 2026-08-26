"""Master training engine with exhaustive model search.

Runs broad hyperparameter/model search for each horizon and task, then saves
only the best 6 models (H1/H2/H3 x regression/classification) in artifacts.
"""
from __future__ import annotations

import json
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

from config.config import MODEL_ARTIFACTS_DIR, RANDOM_SEED
from src.data_processing.prepare_data import get_prepared_data
from src.utils.logging import get_logger

logger = get_logger(__name__)

RANDOM_STATE = int(RANDOM_SEED)
HORIZONS = (1, 2, 3)
ARTIFACT_DIR = MODEL_ARTIFACTS_DIR


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


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


def _stable_metadata_digest(metadata: dict[str, Any]) -> str:
    payload = json.dumps(_to_jsonable(metadata), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metadata_for_digest(metadata: dict[str, Any]) -> dict[str, Any]:
    clean = dict(metadata)
    clean.pop("evidence_manifest_sha256", None)
    return clean


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Pearson": _safe_pearson(y_true, y_pred),
        "DA": float(np.mean((y_true > 0).astype(int) == (y_pred > 0).astype(int))),
        "AUC": float("nan"),
    }


def _evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    return {
        "MAE": float("nan"),
        "Pearson": float("nan"),
        "DA": float(np.mean(y_true == y_pred)),
        "AUC": auc,
    }


def _regression_candidates(h: int) -> list[dict[str, Any]]:
    xgb_base = {
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    if h == 1:
        xgb_base.update({"reg_lambda": 10.0, "reg_alpha": 3.0})
    else:
        xgb_base.update({"reg_lambda": 3.0, "reg_alpha": 0.5})

    return [
        {
            "name": "ridge",
            "search": GridSearchCV,
            "estimator": Ridge(),
            "params": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
            "search_kwargs": {},
        },
        {
            "name": "rf",
            "search": GridSearchCV,
            "estimator": RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": [300, 500],
                "max_depth": [2, 3, 4],
                "min_samples_leaf": [1, 3, 5],
            },
            "search_kwargs": {},
        },
        {
            "name": "extra_trees",
            "search": GridSearchCV,
            "estimator": ExtraTreesRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": [300, 500],
                "max_depth": [2, 3, 4],
                "min_samples_leaf": [1, 3, 5],
            },
            "search_kwargs": {},
        },
        {
            "name": "xgb",
            "search": RandomizedSearchCV,
            "estimator": XGBRegressor(**xgb_base),
            "params": {
                "n_estimators": [250, 350, 450, 600],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "reg_lambda": [3.0, 6.0, 10.0],
                "reg_alpha": [0.5, 1.0, 3.0],
            },
            "search_kwargs": {"n_iter": 25, "random_state": RANDOM_STATE},
        },
    ]


def _classification_candidates(h: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "logistic",
            "search": GridSearchCV,
            "estimator": LogisticRegression(max_iter=3000, class_weight="balanced"),
            "params": {
                "C": [0.001, 0.01, 0.1, 1.0, 10.0],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"],
            },
            "search_kwargs": {},
        },
        {
            "name": "rf",
            "search": GridSearchCV,
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": [300, 500],
                "max_depth": [2, 3, 4],
                "min_samples_leaf": [1, 3, 5],
            },
            "search_kwargs": {},
        },
        {
            "name": "xgb",
            "search": RandomizedSearchCV,
            "estimator": XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": [250, 350, 450, 600],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "reg_lambda": [3.0, 6.0, 10.0],
                "reg_alpha": [0.5, 1.0, 3.0],
            },
            "search_kwargs": {"n_iter": 25, "random_state": RANDOM_STATE},
        },
    ]


def _run_search(
    estimator: Any,
    params: dict[str, list[Any]],
    cv: TimeSeriesSplit,
    scoring: str,
    search_cls: Any,
    search_kwargs: dict[str, Any],
    X_train: Any,
    y_train: Any,
) -> Any:
    common = {
        "estimator": estimator,
        "cv": cv,
        "n_jobs": -1,
        "verbose": 0,
        "scoring": scoring,
        "refit": True,
    }

    if search_cls is GridSearchCV:
        search = search_cls(param_grid=params, **common, **search_kwargs)
    else:
        search = search_cls(param_distributions=params, **common, **search_kwargs)

    search.fit(X_train, y_train)
    return search


def _pick_best_regression(results: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(results, key=lambda r: (r["metrics_test"]["MAE"], -r["metrics_test"]["DA"]))[0]


def _pick_best_classification(results: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(results, key=lambda r: (-r["metrics_test"]["AUC"], -r["metrics_test"]["DA"]))[0]


def _save_experiment_results(
    results: list[dict[str, Any]],
    horizon: int,
    task: str,
    artifact_dir: Path,
) -> None:
    """Guarda resultados completos de todos los experimentos evaluados.

    Args:
        results: Lista con todos los resultados de los experimentos.
        horizon: Horizonte temporal (1, 2 o 3).
        task: Tarea ('regresion' o 'clasificacion').
        artifact_dir: Directorio donde guardar los resultados.
    """
    # Crear copia de resultados sin el objeto del estimador (no serializable)
    results_for_json = []
    for r in results:
        r_copy = r.copy()
        r_copy.pop("estimator", None)
        results_for_json.append(r_copy)

    # Guardar resultados completos
    all_results_path = artifact_dir / f"experiment_results_h{horizon}_{task}.json"
    with all_results_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results_for_json), f, indent=2, ensure_ascii=False)

    # Guardar resumen (best + top 5)
    sorted_results = sorted(
        results,
        key=lambda r: (
            r["metrics_test"]["MAE"] if task == "regresion" else float("inf") if r["metrics_test"].get("AUC") is None else -r["metrics_test"]["AUC"]
        ),
    )
    # Crear copia del resumen sin el objeto del estimador
    best_copy = sorted_results[0].copy()
    best_copy.pop("estimator", None)
    top_5_copies = []
    for r in sorted_results[:5]:
        r_copy = r.copy()
        r_copy.pop("estimator", None)
        top_5_copies.append(r_copy)

    summary = {
        "best": best_copy,
        "top_5": top_5_copies,
        "total_experiments": len(results),
    }
    summary_path = artifact_dir / f"experiment_summary_h{horizon}_{task}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(summary), f, indent=2, ensure_ascii=False)


def _train_regression_for_h(h: int, artifact_dir: Path) -> tuple[Any, dict[str, Any], list[str]]:
    X_train, X_test, y_train, y_test = get_prepared_data(h, "regresion")
    features = X_train.columns.tolist()
    cv = TimeSeriesSplit(n_splits=5)

    all_results: list[dict[str, Any]] = []
    best_model = None

    for cfg in _regression_candidates(h):
        logger.info("H%s REG -> buscando %s", h, cfg["name"])
        search = _run_search(
            estimator=clone(cfg["estimator"]),
            params=cfg["params"],
            cv=cv,
            scoring="neg_mean_absolute_error",
            search_cls=cfg["search"],
            search_kwargs=cfg["search_kwargs"],
            X_train=X_train,
            y_train=y_train,
        )

        candidate = search.best_estimator_
        pred = candidate.predict(X_test)
        metrics = _evaluate_regression(y_test.to_numpy(), pred)

        logger.info(
            "H%s REG %s -> CV_MAE=%.6f | TEST_MAE=%.6f | TEST_Pearson=%.4f | TEST_DA=%.4f",
            h,
            cfg["name"],
            -search.best_score_,
            metrics["MAE"],
            metrics["Pearson"],
            metrics["DA"],
        )

        all_results.append(
            {
                "candidate_name": cfg["name"],
                "best_params": search.best_params_,
                "cv_best_score": float(search.best_score_),
                "metrics_test": metrics,
                "estimator": candidate,
            }
        )

    # Guardar resultados completos de todos los experimentos
    _save_experiment_results(all_results, h, "regresion", artifact_dir)

    best = _pick_best_regression(all_results)
    best_model = best["estimator"]
    best.pop("estimator", None)

    return best_model, best, features


def _train_classification_for_h(h: int, artifact_dir: Path) -> tuple[Any, dict[str, Any], list[str]]:
    X_train, X_test, y_train, y_test = get_prepared_data(h, "clasificacion")
    features = X_train.columns.tolist()
    cv = TimeSeriesSplit(n_splits=5)

    all_results: list[dict[str, Any]] = []
    best_model = None

    for cfg in _classification_candidates(h):
        logger.info("H%s CLF -> buscando %s", h, cfg["name"])
        search = _run_search(
            estimator=clone(cfg["estimator"]),
            params=cfg["params"],
            cv=cv,
            scoring="roc_auc",
            search_cls=cfg["search"],
            search_kwargs=cfg["search_kwargs"],
            X_train=X_train,
            y_train=y_train,
        )

        best_base = search.best_estimator_
        calibrated = CalibratedClassifierCV(best_base, method="sigmoid", cv=cv)
        calibrated.fit(X_train, y_train)

        pred = calibrated.predict(X_test)
        prob = calibrated.predict_proba(X_test)[:, 1]
        metrics = _evaluate_classification(y_test.to_numpy(), pred, prob)

        logger.info(
            "H%s CLF %s -> CV_AUC=%.6f | TEST_AUC=%.4f | TEST_DA=%.4f",
            h,
            cfg["name"],
            search.best_score_,
            metrics["AUC"],
            metrics["DA"],
        )

        all_results.append(
            {
                "candidate_name": cfg["name"],
                "best_params": search.best_params_,
                "cv_best_score": float(search.best_score_),
                "metrics_test": metrics,
                "estimator": calibrated,
            }
        )

    # Guardar resultados completos de todos los experimentos
    _save_experiment_results(all_results, h, "clasificacion", artifact_dir)

    best = _pick_best_classification(all_results)
    best_model = best["estimator"]
    best.pop("estimator", None)

    return best_model, best, features


def train_exhaustive_and_save_best() -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    evidence_bundle_id = uuid.uuid4().hex

    metadata: dict[str, Any] = {
        "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "search_type": "grid+randomized",
        "cut_date": "2021-01-01",
        "evidence_bundle_id": evidence_bundle_id,
        "selected_models": {},
    }

    for h in HORIZONS:
        logger.info("===== H%s: inicio busqueda exhaustiva =====", h)

        reg_model, reg_info, reg_features = _train_regression_for_h(h, ARTIFACT_DIR)
        reg_model_path = ARTIFACT_DIR / f"datagia_best_h{h}_reg.joblib"
        joblib.dump(reg_model, reg_model_path)

        clf_model, clf_info, clf_features = _train_classification_for_h(h, ARTIFACT_DIR)
        clf_model_path = ARTIFACT_DIR / f"datagia_best_h{h}_clf.joblib"
        joblib.dump(clf_model, clf_model_path)

        metadata["selected_models"][f"H{h}"] = {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "regression": {
                "model_path": reg_model_path.name,
                "chosen_candidate": reg_info["candidate_name"],
                "best_params": reg_info["best_params"],
                "cv_best_score": reg_info["cv_best_score"],
                "metrics_test": reg_info["metrics_test"],
                "expected_columns": reg_features,
            },
            "classification": {
                "model_path": clf_model_path.name,
                "chosen_candidate": clf_info["candidate_name"],
                "best_params": clf_info["best_params"],
                "cv_best_score": clf_info["cv_best_score"],
                "metrics_test": clf_info["metrics_test"],
                "expected_columns": clf_features,
            },
        }

        logger.info(
            "H%s SELECCIONADOS -> REG=%s | CLF=%s",
            h,
            reg_info["candidate_name"],
            clf_info["candidate_name"],
        )

    metadata["run_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["evidence_manifest_sha256"] = _stable_metadata_digest(_metadata_for_digest(metadata))

    metadata_path = ARTIFACT_DIR / "model_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metadata), f, indent=2, ensure_ascii=False)

    logger.info("Metadata de seleccion guardada en: %s", metadata_path)
    return metadata


def print_summary(metadata: dict[str, Any]) -> None:
    logger.info("===== RESUMEN FINAL TEST 2021-2025 =====")
    for h in HORIZONS:
        selected = metadata["selected_models"][f"H{h}"]
        reg = selected["regression"]
        clf = selected["classification"]

        logger.info(
            "H%s REG (%s) -> MAE=%.6f | Pearson=%.4f | DA=%.4f",
            h,
            reg["chosen_candidate"],
            reg["metrics_test"]["MAE"],
            reg["metrics_test"]["Pearson"],
            reg["metrics_test"]["DA"],
        )
        logger.info(
            "H%s CLF (%s) -> AUC=%.4f | DA=%.4f",
            h,
            clf["chosen_candidate"],
            clf["metrics_test"]["AUC"],
            clf["metrics_test"]["DA"],
        )


def main() -> None:
    metadata = train_exhaustive_and_save_best()
    print_summary(metadata)


if __name__ == "__main__":
    main()
