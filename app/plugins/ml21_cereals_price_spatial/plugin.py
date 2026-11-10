"""Ml21CerealsPriceSpatialPlugin — ESP-CEREAL spatial cereal price prediction.

Predicts cereal prices across Spanish provinces at 1/2/3-month horizons using
ExtraTrees (H1/H2) + XGBoost (H3) for regression and LogisticRegression for
directional classification. Generates decision cards for buyers/sellers.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any, NamedTuple

import joblib
import numpy as np
import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.batch_rows import row_error_capture
from app.domain.services.exceptions import ModelNotLoadedError
from app.domain.services.mlflow_artifacts import removing_dir
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.domain.services.xai_values import xai_feature_values
from app.infrastructure.artifact_store import local_file_path
from app.infrastructure.base_model_publisher import persist_retrained_bundle
from app.plugins.ml21_cereals_price_spatial.constants import (
    ARTIFACT_FOLDER_NAME,
    DATASET_FILENAME,
    FRAMEWORK,
    GEO_RISK_DEFAULT_PROVINCES,
    MODEL_ID,
    TIMING_DELTA,
    TRAIN_RANDOM_STATE,
    VALID_HORIZONS,
    VERSION,
    XGB_H3_BASE_PARAMS,
    XGB_H3_N_ITER,
    XGB_H3_PARAM_DISTRIBUTIONS,
)
from app.plugins.ml21_cereals_price_spatial.explain import (
    extract_feature_importance,
    top_causal_drivers,
)
from app.plugins.ml21_cereals_price_spatial.model_loader import load_model_bundle
from app.plugins.ml21_cereals_price_spatial.mlflow_utils import download_user_model_from_mlflow
from app.plugins.ml21_cereals_price_spatial.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml21_cereals_price_spatial.preprocessing import (
    build_features_batch,
    build_features_from_row,
    get_selected_feature_columns,
    lookup_panel_row,
    signal_from_prob_return,
)
from app.plugins.ml21_cereals_price_spatial.train_dto import TrainResponse

logger = logging.getLogger(__name__)


def _coherence_message(signals: dict[int, str]) -> str:
    h1 = signals.get(1, "NEUTRAL/ESPERA")
    h3 = signals.get(3, "NEUTRAL/ESPERA")
    if (h1 == "ALCISTA" and h3 == "BAJISTA") or (h1 == "BAJISTA" and h3 == "ALCISTA"):
        return (
            "ATENCION: Volatilidad a corto plazo detectada. La tendencia a 90 dias "
            "contradice el movimiento inmediato. Riesgo de atrapamiento de inventario."
        )
    if h1 == h3 and h1 != "NEUTRAL/ESPERA":
        return "Tendencia solida confirmada en todos los horizontes analizados."
    return "Sin contradiccion estructural fuerte entre corto y largo plazo."


def _timing_decision(
    role: str,
    h1_ret: float,
    h2_ret: float,
    h3_ret: float,
) -> str:
    if role == "comprador":
        if h3_ret > h1_ret + TIMING_DELTA:
            return "COMPRA YA"
        if h2_ret < h1_ret - TIMING_DELTA or h3_ret < h1_ret - TIMING_DELTA:
            return "ESPERA PARA COMPRAR"
        return "SEGUIMIENTO ACTIVO"
    if h3_ret < h1_ret - TIMING_DELTA:
        return "VENDE YA"
    if h3_ret > h1_ret + TIMING_DELTA:
        return "ESPERA PARA VENDER"
    return "SEGUIMIENTO ACTIVO"


class _CardContext(NamedTuple):
    """What a decision card is about: the place, cereal, month and its 3-horizon predictions."""

    province: str
    cereal: str
    month: str
    predictions: dict[int, dict[str, Any]]
    geo_risk: bool


def _build_recommendation(role: str, ctx: _CardContext, timing_label: str | None = None) -> str:
    predictions, province, cereal, geo_risk = (
        ctx.predictions, ctx.province, ctx.cereal, ctx.geo_risk
    )
    h1_signal = predictions[1]["signal"]
    h2_signal = predictions[2]["signal"]
    h3_signal = predictions[3]["signal"]

    if role == "vendedor":
        if h3_signal == "BAJISTA":
            base = (
                "Se detecta presion bajista consistente en H3. Se recomienda cerrar ventas "
                "de cobertura en el corto plazo para proteger margen antes de una posible "
                "debilidad adicional de precios."
            )
        elif h3_signal == "ALCISTA":
            base = (
                "Recomendacion: Retener stock, el modelo detecta presiones al alza que "
                "mejoraran el margen en el trimestre actual."
            )
        else:
            base = (
                "Escenario mixto. Se recomienda mantener estrategia defensiva y ejecutar "
                "coberturas parciales hasta confirmar direccion en H2-H3."
            )
    else:
        if h3_signal == "ALCISTA":
            base = (
                "Fuerte presion al alza detectada. Se recomienda adelantar compras a 90 dias "
                "para reducir riesgo de encarecimiento en el horizonte estrategico."
            )
        elif h2_signal == "BAJISTA" or h3_signal == "BAJISTA":
            base = (
                "Recomendacion: Retrasar compras, se espera ventana de oportunidad con "
                "mejores precios en 60-90 dias."
            )
        else:
            base = (
                "Las senales no son concluyentes. Mantener compras fraccionadas y revisar "
                "confirmacion de tendencia en las proximas actualizaciones."
            )

    if timing_label:
        base = f"Timing Operativo: {timing_label}. " + base

    if h1_signal != h3_signal and h1_signal != "NEUTRAL/ESPERA" and h3_signal != "NEUTRAL/ESPERA":
        base += (
            " Existe divergencia entre corto y largo plazo, por lo que se sugiere tactica "
            "escalonada."
        )

    if geo_risk:
        base += (
            f" Riesgo Geografico Alto en {province}: elevar margen de seguridad y validar con "
            "informacion local antes de comprometer volumen."
        )

    base += f" Contexto evaluado para {province} ({cereal})."
    return base


def _build_card_text(
    ctx: _CardContext,
    *,
    recommendation: str,
    coherence_msg: str,
    causal_drivers: list[str],
) -> str:
    province, cereal, month_text, predictions = (
        ctx.province, ctx.cereal, ctx.month, ctx.predictions
    )
    risk_line = "SI" if ctx.geo_risk else "NO"

    def _line(h: int, days: int) -> str:
        p = predictions[h]
        return (
            f"HORIZONTE H{h} ({days} DIAS): {p['signal']} | "
            f"Confianza={p['confidence']:.1f}% | "
            f"Magnitud Esperada={p['expected_return'] * 100:.2f}%"
        )

    lines = [
        "FICHA DE DECISION DATAGIA v1.1",
        "",
        "[1] RESUMEN EJECUTIVO",
        f"Provincia: {province} | Cereal: {cereal} | Mes: {month_text}",
        f"Riesgo Geografico Alto: {risk_line}",
        f"Coherencia Temporal: {coherence_msg}",
        "",
        "[2] DETALLE POR HORIZONTE",
        _line(1, 30),
        _line(2, 60),
        _line(3, 90),
        "",
        "[3] FACTORES DE RIESGO Y CAUSALIDAD",
        "Motores de la Prediccion (Top 3 impacto aproximado):",
    ]
    for i, drv in enumerate(causal_drivers, start=1):
        lines.append(f"{i}. {drv}")

    lines.extend(
        [
            "",
            "[4] ACCION RECOMENDADA",
            recommendation,
        ]
    )
    return "\n".join(lines)


def _regression_search(h: int, cv):
    """Search of the delivered regressor for horizon h, as the AI team's train.py runs it.

    H3: XGBoost with RandomizedSearchCV (25 draws, seed 42) over the original 7-parameter grid,
    so a retrain can land on the delivered configuration (colsample_bytree included). H1/H2:
    ExtraTrees with GridSearchCV over the original grid.
    """
    # pylint: disable=import-outside-toplevel  # heavy training libs, not loaded when serving
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from xgboost import XGBRegressor
    # pylint: enable=import-outside-toplevel

    if h == 3:
        estimator = XGBRegressor(
            objective="reg:squarederror",
            random_state=TRAIN_RANDOM_STATE, n_jobs=-1,
            **XGB_H3_BASE_PARAMS,
        )
        return RandomizedSearchCV(
            estimator=estimator, param_distributions=XGB_H3_PARAM_DISTRIBUTIONS,
            n_iter=XGB_H3_N_ITER, random_state=TRAIN_RANDOM_STATE,
            cv=cv, scoring="neg_mean_absolute_error",
            n_jobs=-1, refit=True, verbose=0,
        )
    estimator = ExtraTreesRegressor(random_state=TRAIN_RANDOM_STATE, n_jobs=-1)
    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [2, 3, 4],
        "min_samples_leaf": [1, 3, 5],
    }
    return GridSearchCV(
        estimator=estimator, param_grid=param_grid,
        cv=cv, scoring="neg_mean_absolute_error",
        n_jobs=-1, refit=True, verbose=0,
    )


def _regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Test MAE, Pearson r (nan with fewer than 2 rows) and directional accuracy."""
    # pylint: disable=import-outside-toplevel  # heavy training libs, not loaded when serving
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_absolute_error
    # pylint: enable=import-outside-toplevel

    mae = float(mean_absolute_error(y_true, y_pred))
    pearson = (
        float(pearsonr(y_true, y_pred)[0]) if len(y_true) >= 2
        else float("nan")
    )
    da = float(np.mean((y_true > 0).astype(int) == (y_pred > 0).astype(int)))
    return mae, pearson, da


def _train_regression(h: int, split: tuple, cv, tmp_dir: str) -> tuple[dict, dict]:
    """Fit, score and save the H{h} regressor; return (metrics, its selected_models block)."""
    x_train, x_test, y_train, y_test = split
    features = x_train.columns.tolist()
    search = _regression_search(h, cv)
    search.fit(x_train, y_train)
    best_model = search.best_estimator_
    mae, pearson, da = _regression_scores(y_test.to_numpy(), best_model.predict(x_test))

    fname = f"datagia_best_h{h}_reg.joblib"
    joblib.dump(best_model, os.path.join(tmp_dir, fname))

    return {
        f"mae_h{h}": mae, f"pearson_h{h}": pearson, f"da_h{h}": da, f"auc_h{h}": float("nan"),
    }, {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "regression": {
            "model_path": fname,
            "chosen_candidate": "extra_trees" if h != 3 else "xgb",
            "best_params": search.best_params_,
            "cv_best_score": float(search.best_score_),
            "metrics_test": {
                "MAE": mae, "Pearson": pearson, "DA": da, "AUC": float("nan"),
            },
            "expected_columns": features,
            "feature_importance": extract_feature_importance(
                best_model, features
            ),
        },
    }


def _fit_classifier(x_train: pd.DataFrame, y_train: pd.Series, cv):
    """Grid-searched logistic regression, then sigmoid-calibrated; returns (search, model)."""
    # pylint: disable=import-outside-toplevel  # heavy training libs, not loaded when serving
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    # pylint: enable=import-outside-toplevel

    # random_state: the original leaves it unset, so liblinear's coordinate-descent shuffling made
    # every retrain differ slightly (test AUC ±0.003 on the same data). Fixed to the original's
    # seed so a retrain is reproducible; same C/penalty chosen and AUC within that range.
    estimator = LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=TRAIN_RANDOM_STATE
    )
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
    }
    search = GridSearchCV(
        estimator=estimator, param_grid=param_grid,
        cv=cv, scoring="roc_auc",
        n_jobs=-1, refit=True, verbose=0,
    )
    search.fit(x_train, y_train)
    best_base = search.best_estimator_
    calibrated = CalibratedClassifierCV(best_base, method="sigmoid", cv=cv)
    calibrated.fit(x_train, y_train)
    return search, calibrated


def _classification_scores(calibrated, x_test: pd.DataFrame, y_test: pd.Series):
    """Test directional accuracy and ROC AUC (nan when the test set has a single class)."""
    from sklearn.metrics import roc_auc_score  # pylint: disable=import-outside-toplevel

    y_pred = calibrated.predict(x_test)
    y_prob = calibrated.predict_proba(x_test)[:, 1]
    y_true = y_test.to_numpy()

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    da = float(np.mean(y_true == y_pred))
    return da, auc


def _train_classification(h: int, split: tuple, cv, tmp_dir: str) -> tuple[dict, dict]:
    """Fit, calibrate, score and save the H{h} classifier; return (metrics, its block)."""
    x_train, x_test, y_train, y_test = split
    features = x_train.columns.tolist()
    search, calibrated = _fit_classifier(x_train, y_train, cv)
    da, auc = _classification_scores(calibrated, x_test, y_test)

    fname = f"datagia_best_h{h}_clf.joblib"
    joblib.dump(calibrated, os.path.join(tmp_dir, fname))

    block = {
        "model_path": fname,
        "chosen_candidate": "logistic",
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
        "metrics_test": {
            "MAE": float("nan"), "Pearson": float("nan"), "DA": da, "AUC": auc,
        },
        "expected_columns": features,
        "feature_importance": extract_feature_importance(calibrated, features),
    }
    return {f"da_h{h}": da, f"auc_h{h}": auc}, block


def _write_metadata(metadata: dict[str, Any], tmp_dir: str) -> None:
    """Close the run (finish time + evidence hash) and write model_metadata.json."""
    metadata["run_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload = {k: v for k, v in metadata.items() if k != "evidence_manifest_sha256"}
    metadata["evidence_manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()

    meta_path = os.path.join(tmp_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False, default=str)


def _train_and_save_models(df: pd.DataFrame, tmp_dir: str) -> dict[str, Any]:
    """Train the 6 models into tmp_dir with model_metadata.json; return the test metrics."""
    # pylint: disable=import-outside-toplevel
    from sklearn.model_selection import TimeSeriesSplit

    # Lazy so tests can monkeypatch preprocessing.prepare_train_test.
    from app.plugins.ml21_cereals_price_spatial.preprocessing import prepare_train_test
    # pylint: enable=import-outside-toplevel

    all_metrics: dict[str, Any] = {}
    metadata: dict[str, Any] = {
        "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": TRAIN_RANDOM_STATE,
        "search_type": "grid+randomized",
        "cut_date": "2021-01-01",
        "evidence_bundle_id": uuid.uuid4().hex,
        "selected_models": {},
    }
    for h in VALID_HORIZONS:
        for task in ("regresion", "clasificacion"):
            split = prepare_train_test(df, h, task)
            cv = TimeSeriesSplit(n_splits=5)
            if task == "regresion":
                metrics, block = _train_regression(h, split, cv, tmp_dir)
                all_metrics.update(metrics)
                metadata["selected_models"][f"H{h}"] = block
            else:
                metrics, block = _train_classification(h, split, cv, tmp_dir)
                all_metrics.update(metrics)
                if f"H{h}" not in metadata["selected_models"]:
                    metadata["selected_models"][f"H{h}"] = {}
                metadata["selected_models"][f"H{h}"]["classification"] = block

    _write_metadata(metadata, tmp_dir)
    return all_metrics


def _upload_run_to_mlflow(tracker: BaseMLflowTracker, all_metrics: dict, tmp_dir: str) -> None:
    """Log params/metrics and upload the models + metadata (not the training CSV) to the run."""
    tracker.log_params({
        "model_type": "extra_trees+xgb+logistic",
        "horizons": "1,2,3",
        "cut_date": "2021-01-01",
        "random_state": str(TRAIN_RANDOM_STATE),
    })
    tracker.log_metrics(
        {
            k: v for k, v in all_metrics.items()
            if not (isinstance(v, float) and np.isnan(v))
        }
    )
    # The run keeps only the models + metadata; the training CSV goes to the base version.
    run_dir = tempfile.mkdtemp(prefix="ml21_run_")
    with removing_dir(run_dir):
        shutil.copytree(tmp_dir, run_dir, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(DATASET_FILENAME))
        tracker.upload_artifacts(run_dir, artifact_path="model")


def _train_response(all_metrics: dict[str, Any], df: pd.DataFrame, persisted) -> TrainResponse:
    """TrainResponse with the per-horizon test metrics and the published base version."""
    return TrainResponse(
        detail=(
            "Training completado — 6 modelos entrenados (3H × reg+clf). "
            f"Modelo base {persisted.base_version['version']}."
        ),
        mae_h1=all_metrics.get("mae_h1"),
        mae_h2=all_metrics.get("mae_h2"),
        mae_h3=all_metrics.get("mae_h3"),
        pearson_h1=all_metrics.get("pearson_h1"),
        pearson_h2=all_metrics.get("pearson_h2"),
        pearson_h3=all_metrics.get("pearson_h3"),
        da_h1=all_metrics.get("da_h1"),
        da_h2=all_metrics.get("da_h2"),
        da_h3=all_metrics.get("da_h3"),
        auc_h1=all_metrics.get("auc_h1"),
        auc_h2=all_metrics.get("auc_h2"),
        auc_h3=all_metrics.get("auc_h3"),
        n_train=len(df[df["date"] < pd.Timestamp("2021-01-01")]),
        n_test=len(df[df["date"] >= pd.Timestamp("2021-01-01")]),
        upload_warning=persisted.upload_warning,
        base_version=persisted.base_version,
    )


class Ml21CerealsPriceSpatialPlugin(PluginRuntimeMixin, ModelPluginPort):
    """ESP-CEREAL plugin for spatial cereal price prediction across Spanish provinces."""

    BASE_ARTIFACT_FOLDER = ARTIFACT_FOLDER_NAME   # every /train publishes the new base model

    def __init__(self) -> None:
        self._models: dict[str, dict[str, Any]] | None = None
        self._metadata: dict[str, Any] | None = None
        self._panel_df: pd.DataFrame | None = None
        # (mean, std) of the H3-regression features over the base panel (causal drivers)
        self._h3_reference: tuple[pd.Series | None, pd.Series | None] = (None, None)

    def load(self) -> None:
        self._models, self._metadata, self._panel_df = load_model_bundle()
        self._ensure_feature_importance(self._metadata, self._models)
        self._h3_reference = self._compute_h3_reference_stats(
            self._panel_df, self._metadata
        )
        logger.info("Ml21CerealsPriceSpatialPlugin loaded: %s", MODEL_ID)

    @staticmethod
    def _ensure_feature_importance(metadata: dict, models: dict[str, dict[str, Any]]) -> None:
        """Backfill feature_importance into metadata (in-memory only — never written back
        to the fixed S3 artifact) when the shipped model_metadata.json lacks it.

        Mirrors _ensure_importance_in_metadata from the original src/predict/predict_v1.py.
        """
        selected = metadata.get("selected_models", {})
        for h_key, block in selected.items():
            reg = block.get("regression", {})
            if reg.get("expected_columns") and not reg.get("feature_importance"):
                reg["feature_importance"] = extract_feature_importance(
                    models[h_key]["reg"], reg["expected_columns"]
                )
            clf = block.get("classification", {})
            if clf.get("expected_columns") and not clf.get("feature_importance"):
                clf["feature_importance"] = extract_feature_importance(
                    models[h_key]["clf"], clf["expected_columns"]
                )

    @staticmethod
    def _compute_h3_reference_stats(
        panel_df: pd.DataFrame, metadata: dict
    ) -> tuple[pd.Series, pd.Series]:
        """Mean/std of the H3-regression feature set over the base panel — the background
        distribution used to z-score each prediction's causal drivers (explain.py).

        The original computes this from X_train (pre-CUT_DATE rows) specifically; using the
        full panel here is a deliberate simplification to avoid re-deriving the train/test
        split at serving time. It only affects the *ranking* of causal drivers shown in the
        decision card, never the prediction itself.
        """
        expected_r = get_selected_feature_columns(metadata, 3, "regresion")
        feat = build_features_batch(panel_df, 3).reindex(columns=expected_r, fill_value=0)
        return feat.mean(axis=0), feat.std(axis=0)

    def is_loaded(self) -> bool:
        return self._models is not None

    def _require_loaded(self) -> None:
        # own message (no accent), kept as the API always returned it
        if self._models is None:
            raise ModelNotLoadedError("El modelo no esta cargado.")

    def _get_geo_risk_provinces(self, metadata: dict) -> set[str]:
        _ = metadata
        return GEO_RISK_DEFAULT_PROVINCES.copy()

    def _predict_single(
        self,
        raw_row: pd.DataFrame,
        metadata: dict,
        models: dict[str, dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        """Run prediction for a single row across all 3 horizons."""
        predictions: dict[int, dict[str, Any]] = {}

        for h in VALID_HORIZONS:
            expected_r = get_selected_feature_columns(metadata, h, "regresion")
            expected_c = get_selected_feature_columns(metadata, h, "clasificacion")

            row_r = build_features_from_row(raw_row, expected_r)
            row_c = build_features_from_row(raw_row, expected_c)

            reg_model = models[f"H{h}"]["reg"]
            clf_model = models[f"H{h}"]["clf"]

            expected_ret = float(reg_model.predict(row_r)[0])
            prob_up = float(clf_model.predict_proba(row_c)[:, 1][0])
            signal = signal_from_prob_return(prob_up, expected_ret)

            predictions[h] = {
                "horizon": h,
                "signal": signal,
                "confidence": max(prob_up, 1.0 - prob_up) * 100.0,
                "expected_return": expected_ret,
                "prob_up": prob_up,
            }

        return predictions

    def _resolve_model(self, mlflow_run_id: str) -> tuple[dict, dict, str | None]:
        """Return (models, metadata, temp_dir) to serve this request.

        The user's retrained model (if mlflow_run_id) is returned, never assigned to self:
        the plugin instance is shared by concurrent requests, so swapping it into self leaked
        a user's model to other requests — and, with two overlapping runs, left it in place
        as the "base" model. temp_dir is None for the base model.
        """
        if not mlflow_run_id:
            self._require_loaded()
            return self._models, self._metadata, None
        return download_user_model_from_mlflow(mlflow_run_id)

    def _raw_row(self, features: dict, province: str, cereal: str, month: str) -> pd.DataFrame:
        """The caller's fields over the base-panel row of (province, cereal, month).

        The caller only supplies provincia/cereal_predominante/date (manifest inputs.fixed) —
        the ~90 engineered columns the models were trained on (lat_centroide, month_sin/cos,
        fase_*, price lags, climate, MAPA indices...) come from the base panel dataset, same as
        run_single() in the original src/predict/predict_v1.py. Any field the caller DOES supply
        explicitly (the optional lat_centroide/dist_puerto_min_km/... overrides in the DTO)
        wins over the panel lookup.
        """
        panel_row: dict[str, Any] = {}
        if self._panel_df is not None:
            panel_row = lookup_panel_row(self._panel_df, province, cereal, month).to_dict()
        merged = {**panel_row, **{k: v for k, v in features.items() if v is not None}}
        return pd.DataFrame([merged])

    def _causal_drivers(
        self, raw_row: pd.DataFrame, metadata: dict, expected_r: list[str]
    ) -> list[str]:
        """Top-3 drivers of the H3 regression (z-scored against the base panel)."""
        h3_mean, h3_std = self._h3_reference
        if h3_mean is None or h3_std is None:
            return ["No hay importancias disponibles para explicar esta prediccion."]
        row_r_h3 = build_features_from_row(raw_row, expected_r)
        h3_importance = (
            metadata.get("selected_models", {})
            .get("H3", {})
            .get("regression", {})
            .get("feature_importance", [])
        )
        return top_causal_drivers(
            row_features=row_r_h3,
            feature_mean=h3_mean,
            feature_std=h3_std,
            importance_records=h3_importance,
            top_n=3,
        )

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> Any:
        """Single-row inference: province + cereal + month → 3-horizon predictions."""
        if mlflow_run_id:
            logger.info("predict_inline — using user model from MLflow run_id=%s", mlflow_run_id)
        models, metadata, user_temp_dir = self._resolve_model(mlflow_run_id)
        with removing_dir(user_temp_dir):
            assert models is not None and metadata is not None

            t0 = time.perf_counter()

            province = str(features.get("provincia", ""))
            cereal = str(features.get("cereal_predominante", ""))
            month = str(features.get("date", ""))
            raw_row = self._raw_row(features, province, cereal, month)

            ctx = _CardContext(
                province=province, cereal=cereal, month=month,
                predictions=self._predict_single(raw_row, metadata, models),
                geo_risk=province in self._get_geo_risk_provinces(metadata),
            )
            response_fields = self._decision_card(raw_row, ctx, metadata)

            self._record_latency((time.perf_counter() - t0) * 1000)

            return PredictInlineResponse(**response_fields)

    def _decision_card(self, raw_row: pd.DataFrame, ctx: _CardContext, metadata: dict) -> dict:
        """PredictInlineResponse fields: timing, causal drivers, decision card and XAI values."""
        predictions = ctx.predictions
        coherence_msg = _coherence_message(
            {h: predictions[h]["signal"] for h in VALID_HORIZONS}
        )
        timing_label = _timing_decision(
            "comprador",
            predictions[1]["expected_return"],
            predictions[2]["expected_return"],
            predictions[3]["expected_return"],
        )

        expected_r = get_selected_feature_columns(metadata, 3, "regresion")
        causal_drivers = self._causal_drivers(raw_row, metadata, expected_r)

        recommendation = _build_recommendation("comprador", ctx, timing_label=timing_label)

        card_text = _build_card_text(
            ctx,
            recommendation=recommendation,
            coherence_msg=coherence_msg,
            causal_drivers=causal_drivers,
        )

        # The H3 regression model's own input row (one-hot cereal/provincia columns included),
        # not the raw panel row: the explainability service feeds these values straight to
        # datagia_best_h3_reg, so any column missing here would be explained as 0.
        xai_values = xai_feature_values(
            build_features_from_row(raw_row, expected_r).iloc[0], expected_r
        ) or None

        return {
            "model_id": MODEL_ID,
            "province": ctx.province,
            "cereal": ctx.cereal,
            "month": ctx.month,
            "geo_risk": ctx.geo_risk,
            "timing_label": timing_label,
            "causal_drivers": causal_drivers,
            "card_text": card_text,
            "predictions": {f"H{h}": predictions[h] for h in VALID_HORIZONS},
            "model_version": VERSION,
            "xai_feature_values": xai_values,
        }

    def _batch_row(self, idx, row: pd.Series, metadata: dict, models: dict) -> dict[str, Any]:
        """Predictions of one CSV row (predict_batch turns a failure into that row's error)."""
        raw_row = row.to_frame().T
        predictions = self._predict_single(
            raw_row, metadata, models
        )
        return {
            "row": int(idx),
            "provincia": str(row.get("provincia", "")),
            "cereal_predominante": str(row.get("cereal_predominante", "")),
            **{f"ret_h{h}": predictions[h]["expected_return"] for h in VALID_HORIZONS},
            **{f"prob_up_h{h}": predictions[h]["prob_up"] for h in VALID_HORIZONS},
            **{f"signal_h{h}": predictions[h]["signal"] for h in VALID_HORIZONS},
            **{
                f"confidence_h{h}": predictions[h]["confidence"]
                for h in VALID_HORIZONS
            },
        }

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> Any:
        """Batch inference: CSV with raw panel rows → predictions for all province×cereal."""
        models, metadata, user_temp_dir = self._resolve_model(mlflow_run_id)
        with removing_dir(user_temp_dir):
            assert models is not None and metadata is not None

            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")

            t0 = time.perf_counter()
            results: list[dict[str, Any]] = []
            for idx, row in df.iterrows():
                # one bad row never aborts the batch
                with row_error_capture(results, {"row": int(idx)}, what=f"fila {idx}"):
                    results.append(self._batch_row(idx, row, metadata, models))

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._record_latency(elapsed_ms)

            logger.info(
                "predict_batch done — %d rows in %.1fms count=%d",
                len(results), elapsed_ms, self._predict_count,
            )
            return PredictBatchResponse(
                model_id=MODEL_ID,
                predictions=results,
                output_path=None,
            )

    def train(self, *, data_path: str = "", mlflow_run_id: str = "") -> Any:
        """Train 6 models (3 horizons × reg+clf) from a CSV and publish them as the new base
        version.

        The bundle is also uploaded to the MLflow run when *mlflow_run_id* is given.

        Training CSV must contain raw panel rows with columns:
          date, provincia, cereal_predominante, precio_provincial_lag_1,
          precio_provincial_TARGET_H1/H2/H3, plus all raw feature columns.

        The 6 models, model_metadata.json and the training CSV itself (as the new
        dataset_entrenamiento_final.csv panel, so predict_inline finds the new months) are
        published as the new base model version; if mlflow_run_id is given they are also
        saved to that run. The CSV must have every column of the current panel (400 if not).
        """
        t0 = time.perf_counter()

        tmp_dir = tempfile.mkdtemp(prefix="ml21_train_")
        with removing_dir(tmp_dir):
            df = self._load_training_frame(data_path, tmp_dir)
            tracker = BaseMLflowTracker(mlflow_run_id) if mlflow_run_id else None
            try:
                all_metrics = _train_and_save_models(df, tmp_dir)
                persisted = persist_retrained_bundle(
                    tmp_dir, artifact_folder=ARTIFACT_FOLDER_NAME, data_path=data_path,
                    metrics=all_metrics, mlflow_run_id=mlflow_run_id,
                    upload_to_mlflow=(
                        (lambda: _upload_run_to_mlflow(tracker, all_metrics, tmp_dir))
                        if tracker else None
                    ),
                )

                elapsed = time.perf_counter() - t0
                logger.info(
                    "train() done — MAE H1=%.4f H2=%.4f H3=%.4f elapsed=%.1fs mlflow=%s",
                    all_metrics.get("mae_h1", 0), all_metrics.get("mae_h2", 0),
                    all_metrics.get("mae_h3", 0), elapsed, bool(mlflow_run_id),
                )

                # The container swaps in a fresh plugin on the new base; train() never reloads
                # itself.
                return _train_response(all_metrics, df, persisted)

            except ValueError:
                raise
            except Exception as exc:
                logger.error("train() failed: %s", exc, exc_info=True)
                raise

    def _load_training_frame(self, data_path: str, tmp_dir: str) -> pd.DataFrame:
        """Read the training CSV (copied into tmp_dir as the new panel), check it and sort it."""
        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)
            shutil.copyfile(local_path, os.path.join(tmp_dir, DATASET_FILENAME))

        if self._panel_df is not None:
            missing = [c for c in self._panel_df.columns if c not in df.columns]
            if missing:
                raise ValueError(
                    f"El CSV de entrenamiento pasa a ser el panel del modelo base y le faltan "
                    f"columnas del panel vigente: {missing}"
                )

        if "date" not in df.columns:
            raise ValueError("El CSV debe contener la columna 'date'.")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()
        df = df.sort_values(["date", "provincia", "cereal_predominante"], kind="mergesort")
        return df.reset_index(drop=True)

    def _served_metrics(self) -> dict[str, Any]:
        """Model families per horizon plus the served models' test metrics (model_metadata)."""
        metrics: dict[str, Any] = {
            "horizons": list(VALID_HORIZONS),
            "models": {
                "H1_regression": "ExtraTrees",
                "H1_classification": "LogisticRegression",
                "H2_regression": "ExtraTrees",
                "H2_classification": "LogisticRegression",
                "H3_regression": "XGBoost",
                "H3_classification": "LogisticRegression",
            },
        }
        if self._metadata:
            selected = self._metadata.get("selected_models", {})
            for h_key in ("H1", "H2", "H3"):
                block = selected.get(h_key, {})
                if block:
                    reg_m = block.get("regression", {}).get("metrics_test", {})
                    clf_m = block.get("classification", {}).get("metrics_test", {})
                    if reg_m:
                        metrics[f"{h_key}_reg_MAE"] = reg_m.get("MAE")
                        metrics[f"{h_key}_reg_Pearson"] = reg_m.get("Pearson")
                    if clf_m:
                        metrics[f"{h_key}_clf_DA"] = clf_m.get("DA")
                        metrics[f"{h_key}_clf_AUC"] = clf_m.get("AUC")
        return metrics

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        metrics = self._served_metrics()

        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Modelo espacial de prediccion de precios de cereales en provincias espanolas "
                "a horizontes de 1/2/3 meses. Combina regresion (retorno esperado) y "
                "clasificacion (direccion alcista/bajista) por horizonte. Genera fichas de "
                "decision para compradores y vendedores."
            ),
            task_type="regression_classification",
            framework=FRAMEWORK,
            inputs=[
                InputField(
                    name="provincia",
                    type="str",
                    description="Provincia espanola (e.g., 'Burgos', 'Zamora')",
                ),
                InputField(
                    name="cereal_predominante",
                    type="str",
                    description="Tipo de cereal ('trigo', 'cebada', 'maiz')",
                ),
                InputField(
                    name="date",
                    type="str",
                    description="Mes de prediccion (YYYY-MM)",
                ),
            ],
            outputs=[
                OutputField(
                    name="predictions",
                    type="dict",
                    description=(
                        "Predicciones por horizonte H1/H2/H3 con signal, confidence, "
                        "expected_return"
                    ),
                ),
                OutputField(
                    name="card_text",
                    type="str",
                    description="Ficha de decision completa en texto plano",
                ),
            ],
            metrics=metrics,
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=self._avg_latency_ms(1),
            ),
        )
        # modelo 43-44-style audit follow-up: stats(mlflow_run_id=...) accepted the
        # parameter but never used it — every retrain's /stats always reported this
        # plugin's own static/served metrics, no matter which trained run was asked
        # about, even though train() already logs real metrics to MLflow.
        if mlflow_run_id:
            _add_run_metrics(base, mlflow_run_id)
        return base


def _add_run_metrics(base: StatsResponse, mlflow_run_id: str) -> None:
    """Best-effort: the run's params/metrics under "mlflow", its flat metrics on top-level
    and the legacy H{h}_* tiles overwritten with the run's real values."""
    try:
        tracker = BaseMLflowTracker(mlflow_run_id)
        mlflow_metrics = tracker.get_metrics()
        base.metrics["mlflow"] = {"params": tracker.get_params(), "metrics": mlflow_metrics}
        for metric_name, metric_value in mlflow_metrics.items():
            if isinstance(metric_value, (int, float)):
                base.metrics[metric_name] = metric_value

        # Regression (modelo 43-44-45 audit, Fase 5 follow-up): the loop above
        # only ADDS train()'s own flat keys (mae_h{h}/pearson_h{h}/da_h{h}/
        # auc_h{h}) — it never touches the base dict's own H{h}_reg_MAE/
        # H{h}_reg_Pearson/H{h}_clf_DA/H{h}_clf_AUC keys (built above from
        # self._metadata), because the two use different naming conventions.
        # Every retrain's /stats kept showing the served model's fixed
        # per-horizon numbers regardless of which run was asked about.
        for h_key, h_num in (("H1", 1), ("H2", 2), ("H3", 3)):
            legacy_aliases = {
                f"{h_key}_reg_MAE": f"mae_h{h_num}",
                f"{h_key}_reg_Pearson": f"pearson_h{h_num}",
                f"{h_key}_clf_DA": f"da_h{h_num}",
                f"{h_key}_clf_AUC": f"auc_h{h_num}",
            }
            for legacy_key, real_key in legacy_aliases.items():
                real_value = mlflow_metrics.get(real_key)
                if isinstance(real_value, (int, float)):
                    base.metrics[legacy_key] = real_value
    except Exception as exc:  # pylint: disable=broad-exception-caught  # /stats stays usable
        logger.warning("Could not fetch MLflow stats for run_id=%s: %s", mlflow_run_id, exc)
