"""M21CerealPriceSpatialPlugin — ESP-CEREAL spatial cereal price prediction.

Predicts cereal prices across Spanish provinces at 1/2/3-month horizons using
ExtraTrees (H1/H2) + XGBoost (H3) for regression and LogisticRegression for
directional classification. Generates decision cards for buyers/sellers.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError
from app.infrastructure.artifact_store import local_file_path
from app.plugins.m21_cereal_price_spatial.constants import (
    FRAMEWORK,
    GEO_RISK_DEFAULT_PROVINCES,
    MODEL_ID,
    TIMING_DELTA,
    VALID_HORIZONS,
    VERSION,
)
from app.plugins.m21_cereal_price_spatial.model_loader import load_model_bundle
from app.plugins.m21_cereal_price_spatial.mlflow_utils import download_user_model_from_mlflow
from app.plugins.m21_cereal_price_spatial.preprocessing import (
    build_features_from_row,
    get_selected_feature_columns,
    signal_from_prob_return,
)

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


def _build_recommendation(
    role: str,
    predictions: dict[int, dict[str, Any]],
    province: str,
    cereal: str,
    geo_risk: bool,
    timing_label: str | None = None,
) -> str:
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
        base += " Existe divergencia entre corto y largo plazo, por lo que se sugiere tactica escalonada."

    if geo_risk:
        base += (
            f" Riesgo Geografico Alto en {province}: elevar margen de seguridad y validar con "
            "informacion local antes de comprometer volumen."
        )

    base += f" Contexto evaluado para {province} ({cereal})."
    return base


def _build_card_text(
    province: str,
    cereal: str,
    month_text: str,
    predictions: dict[int, dict[str, Any]],
    recommendation: str,
    geo_risk: bool,
    coherence_msg: str,
    causal_drivers: list[str],
) -> str:
    risk_line = "SI" if geo_risk else "NO"

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


def _xai_values_from_row(row: pd.Series, feature_cols: list[str]) -> dict[str, float]:
    """Build xai_feature_values from a DataFrame row."""
    xai: dict[str, float] = {}
    for col in feature_cols:
        if col in row.index:
            val = row[col]
            try:
                fv = float(val)
                if not np.isnan(fv):
                    xai[col] = fv
            except (TypeError, ValueError):
                pass
    return xai


class M21CerealPriceSpatialPlugin(ModelPluginPort):
    """ESP-CEREAL plugin for spatial cereal price prediction across Spanish provinces."""

    def __init__(self) -> None:
        self._models: dict[str, dict[str, Any]] | None = None
        self._metadata: dict[str, Any] | None = None
        self._predict_count: int = 0
        self._total_latency_ms: float = 0.0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        self._models, self._metadata = load_model_bundle()
        logger.info("M21CerealPriceSpatialPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        return self._models is not None

    def _require_loaded(self) -> None:
        if self._models is None:
            raise ModelNotLoadedError("El modelo no esta cargado.")

    def _record(self, elapsed_ms: float) -> None:
        self._predict_count += 1
        self._total_latency_ms += elapsed_ms
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _get_geo_risk_provinces(self, metadata: dict) -> set[str]:
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

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> Any:
        """Single-row inference: province + cereal + month → 3-horizon predictions."""
        user_temp_dir = None
        saved_models = self._models
        saved_metadata = self._metadata

        if mlflow_run_id:
            logger.info("predict_inline — using user model from MLflow run_id=%s", mlflow_run_id)
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._models, self._metadata, user_temp_dir = loaded

        try:
            self._require_loaded()
            assert self._models is not None and self._metadata is not None

            t0 = time.perf_counter()

            raw_row = pd.DataFrame([features])

            predictions = self._predict_single(raw_row, self._metadata, self._models)

            h1_ret = predictions[1]["expected_return"]
            h2_ret = predictions[2]["expected_return"]
            h3_ret = predictions[3]["expected_return"]

            province = str(features.get("provincia", ""))
            cereal = str(features.get("cereal_predominante", ""))
            month = str(features.get("date", ""))

            geo_risk = province in self._get_geo_risk_provinces(self._metadata)
            coherence_msg = _coherence_message(
                {h: predictions[h]["signal"] for h in VALID_HORIZONS}
            )
            timing_label = _timing_decision("comprador", h1_ret, h2_ret, h3_ret)

            causal_drivers = ["No hay importancias disponibles para explicar esta prediccion."]

            recommendation = _build_recommendation(
                "comprador",
                predictions,
                province,
                cereal,
                geo_risk,
                timing_label=timing_label,
            )

            card_text = _build_card_text(
                province=province,
                cereal=cereal,
                month_text=month,
                predictions=predictions,
                recommendation=recommendation,
                geo_risk=geo_risk,
                coherence_msg=coherence_msg,
                causal_drivers=causal_drivers,
            )

            expected_r = get_selected_feature_columns(self._metadata, 3, "regresion")
            xai_values = _xai_values_from_row(raw_row.iloc[0], expected_r) or None

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._record(elapsed_ms)

            return {
                "model_id": MODEL_ID,
                "province": province,
                "cereal": cereal,
                "month": month,
                "geo_risk": geo_risk,
                "timing_label": timing_label,
                "causal_drivers": causal_drivers,
                "card_text": card_text,
                "predictions": {
                    f"H{h}": predictions[h] for h in VALID_HORIZONS
                },
                "model_version": VERSION,
                "xai_feature_values": xai_values,
            }

        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)
                self._models = saved_models
                self._metadata = saved_metadata

    def predict_batch(
        self, *, data_path: str, mlflow_run_id: str = ""
    ) -> Any:
        """Batch inference: CSV with raw panel rows → predictions for all province×cereal."""
        user_temp_dir = None
        saved_models = self._models
        saved_metadata = self._metadata

        if mlflow_run_id:
            loaded = download_user_model_from_mlflow(mlflow_run_id)
            if loaded:
                self._models, self._metadata, user_temp_dir = loaded

        try:
            self._require_loaded()
            assert self._models is not None and self._metadata is not None

            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")

            t0 = time.perf_counter()
            results: list[dict[str, Any]] = []

            for idx, row in df.iterrows():
                try:
                    raw_row = row.to_frame().T
                    predictions = self._predict_single(
                        raw_row, self._metadata, self._models
                    )
                    results.append({
                        "row": int(idx),
                        "provincia": str(row.get("provincia", "")),
                        "cereal_predominante": str(row.get("cereal_predominante", "")),
                        **{f"ret_h{h}": predictions[h]["expected_return"] for h in VALID_HORIZONS},
                        **{f"prob_up_h{h}": predictions[h]["prob_up"] for h in VALID_HORIZONS},
                        **{f"signal_h{h}": predictions[h]["signal"] for h in VALID_HORIZONS},
                        **{f"confidence_h{h}": predictions[h]["confidence"] for h in VALID_HORIZONS},
                    })
                except Exception as exc:
                    logger.warning("Error en fila %s: %s", idx, exc)
                    results.append({"row": int(idx), "error": str(exc)})

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._record(elapsed_ms)

            logger.info(
                "predict_batch done — %d rows in %.1fms count=%d",
                len(results), elapsed_ms, self._predict_count,
            )
            return {
                "model_id": MODEL_ID,
                "predictions": results,
                "output_path": None,
            }

        finally:
            if user_temp_dir:
                shutil.rmtree(user_temp_dir, ignore_errors=True)
                self._models = saved_models
                self._metadata = saved_metadata

    def train(self, *, data_path: str = "", mlflow_run_id: str = "") -> Any:
        """Train 6 models (3 horizons × reg+clf) from a CSV and upload to MLflow.

        Training CSV must contain raw panel rows with columns:
          date, provincia, cereal_predominante, precio_provincial_lag_1,
          precio_provincial_TARGET_H1/H2/H3, plus all raw feature columns.
        """
        import hashlib
        import json
        import tempfile
        import uuid
        from datetime import datetime, timezone

        import joblib
        from scipy.stats import pearsonr
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import mean_absolute_error, roc_auc_score
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        from xgboost import XGBRegressor

        from app.domain.services.mlflow_tracker import BaseMLflowTracker
        from app.plugins.m21_cereal_price_spatial.preprocessing import prepare_train_test

        t0 = time.perf_counter()

        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)

        if "date" not in df.columns:
            raise ValueError("El CSV debe contener la columna 'date'.")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()
        df = df.sort_values(["date", "provincia", "cereal_predominante"], kind="mergesort")
        df = df.reset_index(drop=True)

        tracker: BaseMLflowTracker | None = None
        if mlflow_run_id:
            tracker = BaseMLflowTracker(mlflow_run_id)

        tmp_dir = tempfile.mkdtemp(prefix="m21_train_")
        all_metrics: dict[str, Any] = {}
        metadata: dict[str, Any] = {
            "run_started_at_utc": datetime.now(timezone.utc).isoformat(),
            "random_state": 42,
            "search_type": "grid",
            "cut_date": "2021-01-01",
            "evidence_bundle_id": uuid.uuid4().hex,
            "selected_models": {},
        }

        try:
            for h in VALID_HORIZONS:
                for task in ("regresion", "clasificacion"):
                    X_train, X_test, y_train, y_test = prepare_train_test(df, h, task)
                    features = X_train.columns.tolist()
                    cv = TimeSeriesSplit(n_splits=5)

                    if task == "regresion":
                        if h == 3:
                            estimator = XGBRegressor(
                                objective="reg:squarederror",
                                random_state=42, n_jobs=-1,
                                reg_lambda=10.0, reg_alpha=0.5,
                            )
                            param_grid = {
                                "n_estimators": [250, 350],
                                "learning_rate": [0.03, 0.05],
                                "max_depth": [2, 3],
                                "subsample": [0.7, 0.85],
                            }
                            search = GridSearchCV(
                                estimator=estimator, param_grid=param_grid,
                                cv=cv, scoring="neg_mean_absolute_error",
                                n_jobs=-1, refit=True, verbose=0,
                            )
                        else:
                            estimator = ExtraTreesRegressor(random_state=42, n_jobs=-1)
                            param_grid = {
                                "n_estimators": [300, 500],
                                "max_depth": [2, 3, 4],
                                "min_samples_leaf": [1, 3, 5],
                            }
                            search = GridSearchCV(
                                estimator=estimator, param_grid=param_grid,
                                cv=cv, scoring="neg_mean_absolute_error",
                                n_jobs=-1, refit=True, verbose=0,
                            )

                        search.fit(X_train, y_train)
                        best_model = search.best_estimator_
                        y_pred = best_model.predict(X_test)
                        y_true = y_test.to_numpy()

                        mae = float(mean_absolute_error(y_true, y_pred))
                        pearson = float(pearsonr(y_true, y_pred)[0]) if len(y_true) >= 2 else float("nan")
                        da = float(np.mean((y_true > 0).astype(int) == (y_pred > 0).astype(int)))
                        auc = float("nan")

                        all_metrics[f"mae_h{h}"] = mae
                        all_metrics[f"pearson_h{h}"] = pearson
                        all_metrics[f"da_h{h}"] = da
                        all_metrics[f"auc_h{h}"] = auc

                        fname = f"datagia_best_h{h}_reg.joblib"
                        joblib.dump(best_model, os.path.join(tmp_dir, fname))

                        metadata["selected_models"][f"H{h}"] = {
                            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                            "regression": {
                                "model_path": fname,
                                "chosen_candidate": "extra_trees" if h != 3 else "xgb",
                                "best_params": search.best_params_,
                                "cv_best_score": float(search.best_score_),
                                "metrics_test": {"MAE": mae, "Pearson": pearson, "DA": da, "AUC": float("nan")},
                                "expected_columns": features,
                            },
                        }

                    else:
                        estimator = LogisticRegression(max_iter=3000, class_weight="balanced")
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
                        search.fit(X_train, y_train)
                        best_base = search.best_estimator_
                        calibrated = CalibratedClassifierCV(best_base, method="sigmoid", cv=cv)
                        calibrated.fit(X_train, y_train)

                        y_pred = calibrated.predict(X_test)
                        y_prob = calibrated.predict_proba(X_test)[:, 1]
                        y_true = y_test.to_numpy()

                        try:
                            auc = float(roc_auc_score(y_true, y_prob))
                        except ValueError:
                            auc = float("nan")
                        da = float(np.mean(y_true == y_pred))

                        all_metrics[f"da_h{h}"] = da
                        all_metrics[f"auc_h{h}"] = auc

                        fname = f"datagia_best_h{h}_clf.joblib"
                        joblib.dump(calibrated, os.path.join(tmp_dir, fname))

                        if f"H{h}" not in metadata["selected_models"]:
                            metadata["selected_models"][f"H{h}"] = {}
                        metadata["selected_models"][f"H{h}"]["classification"] = {
                            "model_path": fname,
                            "chosen_candidate": "logistic",
                            "best_params": search.best_params_,
                            "cv_best_score": float(search.best_score_),
                            "metrics_test": {"MAE": float("nan"), "Pearson": float("nan"), "DA": da, "AUC": auc},
                            "expected_columns": features,
                        }

            metadata["run_finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            payload = {k: v for k, v in metadata.items() if k != "evidence_manifest_sha256"}
            metadata["evidence_manifest_sha256"] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
            ).hexdigest()

            meta_path = os.path.join(tmp_dir, "model_metadata.json")
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2, ensure_ascii=False, default=str)

            upload_warning = None
            if tracker:
                tracker.log_params({
                    "model_type": "extra_trees+xgb+logistic",
                    "horizons": "1,2,3",
                    "cut_date": "2021-01-01",
                    "random_state": "42",
                })
                flat_metrics = {k: v for k, v in all_metrics.items() if not (isinstance(v, float) and np.isnan(v))}
                tracker.log_metrics(flat_metrics)
                try:
                    tracker.upload_artifacts(tmp_dir, artifact_path="model")
                except Exception as exc:
                    logger.error("MLflow artifact upload failed: %s", exc)
                    upload_warning = f"Artifacts no subidos a MLflow: {exc}"

            elapsed = time.perf_counter() - t0
            logger.info(
                "train() done — MAE H1=%.4f H2=%.4f H3=%.4f elapsed=%.1fs mlflow=%s",
                all_metrics.get("mae_h1", 0), all_metrics.get("mae_h2", 0),
                all_metrics.get("mae_h3", 0), elapsed, bool(mlflow_run_id),
            )

            self.load()

            from app.plugins.m21_cereal_price_spatial.train_dto import TrainResponse

            return TrainResponse(
                detail="Training completado — 6 modelos entrenados (3H × reg+clf)",
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
                upload_warning=upload_warning,
            )

        except ValueError:
            raise
        except Exception as exc:
            logger.error("train() failed: %s", exc, exc_info=True)
            raise
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        avg = self._total_latency_ms / self._predict_count if self._predict_count else None
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

        return StatsResponse(
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
                    description="Predicciones por horizonte H1/H2/H3 con signal, confidence, expected_return",
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
                avg_latency_ms=round(avg, 1) if avg is not None else None,
            ),
        )
