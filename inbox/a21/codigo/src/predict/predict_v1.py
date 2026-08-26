"""Strategic inference engine for DATAGIA v1.

Builds a decision card for buyers/sellers by combining directional confidence
(classification) and expected magnitude (return regression) across H1/H2/H3.

Umbrales estratégicos:
    - PROB_BULL (0.65): Probabilidad mínima para señal alcista
    - PROB_BEAR (0.35): Probabilidad máxima para señal bajista
    - RET_BULL (0.015): Retorno mínimo esperado para señal alcista (1.5%)
    - RET_BEAR (-0.015): Retorno máximo esperado para señal bajista (-1.5%)
    - SPREAD_MIN (0.01): Diferencial mínimo entre provincias para arbitraje (1%)
    - CONFIDENCE_HIGH_UP (0.60): Umbral superior para alta fiabilidad
    - CONFIDENCE_HIGH_DOWN (0.40): Umbral inferior para alta fiabilidad
    - TIMING_DELTA (0.01): Diferencial para decisiones de timing
    - BENCHMARK_BREAK_DELTA (0.01): Desviación de media para precios caros/baratos

Todos los umbrales están definidos en config/config.py y pueden ajustarse sin
modificar el código de inferencia.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config.config import (
    BENCHMARK_BREAK_DELTA,
    CONFIDENCE_HIGH_DOWN,
    CONFIDENCE_HIGH_UP,
    DATA_PROCESSED_DIR,
    MAPA_ADMIN_LAG,
    MODEL_ARTIFACTS_DIR,
    MODEL_METRICS_DIR,
    PREDICTIONS_DIR,
    PROJECT_ROOT,
    PROB_BEAR,
    PROB_BULL,
    RET_BEAR,
    RET_BULL,
    SPREAD_MIN,
    TIMING_DELTA,
)
from src.data_processing.prepare_data import BLACKLIST, CUT_DATE, VALID_HORIZONS, get_prepared_data
from src.utils.logging import get_logger

logger = get_logger(__name__)

ARTIFACTS_DIR = MODEL_ARTIFACTS_DIR
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
BASE_DATASET_PATH = DATA_PROCESSED_DIR / "dataset_entrenamiento_final.csv"
GEO_RISK_PATH = MODEL_METRICS_DIR / "error_analysis_h3_top10.csv"


@dataclass
class HorizonPrediction:
    horizon: int
    signal: str
    confidence: float
    expected_return: float
    prob_up: float


def _save_metadata(metadata: dict[str, Any]) -> None:
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def _extract_feature_importance(model: Any, feature_names: list[str]) -> list[dict[str, float]]:
    importances: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_, dtype=float).ravel())
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
        return []

    pairs = [
        {"feature": str(f), "importance": float(i)}
        for f, i in zip(feature_names, importances)
        if float(i) > 0
    ]
    pairs.sort(key=lambda x: x["importance"], reverse=True)
    return pairs


def _ensure_importance_in_metadata(
    metadata: dict[str, Any],
    models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected = metadata.get("selected_models", {})
    changed = False

    for h in VALID_HORIZONS:
        h_key = f"H{h}"
        block = selected.get(h_key, {})
        if not block:
            continue

        reg_cols = block.get("regression", {}).get("expected_columns", [])
        clf_cols = block.get("classification", {}).get("expected_columns", [])

        if reg_cols and not block.get("regression", {}).get("feature_importance"):
            reg_imp = _extract_feature_importance(models[h_key]["reg"], reg_cols)
            block["regression"]["feature_importance"] = reg_imp
            changed = True

        if clf_cols and not block.get("classification", {}).get("feature_importance"):
            clf_imp = _extract_feature_importance(models[h_key]["clf"], clf_cols)
            block["classification"]["feature_importance"] = clf_imp
            changed = True

    if changed:
        metadata["selected_models"] = selected
        _save_metadata(metadata)
        logger.info("Importancias guardadas en model_metadata.json")

    return metadata


def _feature_to_human(feature: str) -> str:
    pretty = feature.replace("_", " ")
    pretty = pretty.replace("prepag1 urea 46", "Urea")
    pretty = pretty.replace("wheat intl eur", "Trigo Internacional")
    pretty = pretty.replace("corn intl eur", "Maiz Internacional")
    pretty = pretty.replace("eur usd", "EUR/USD")
    pretty = pretty.replace("idx ", "Indice ")
    return pretty


def _top_causal_drivers(
    row_features: pd.DataFrame,
    train_features: pd.DataFrame,
    importance_records: list[dict[str, float]],
    top_n: int = 3,
) -> list[str]:
    if not importance_records:
        return ["No hay importancias disponibles para explicar esta prediccion."]

    row = row_features.iloc[0]
    mu = train_features.mean(axis=0)
    sigma = train_features.std(axis=0).replace(0, np.nan)

    scored: list[tuple[str, float, float]] = []
    for rec in importance_records:
        f = rec["feature"]
        imp = float(rec["importance"])
        if f not in row.index:
            continue
        z = (float(row[f]) - float(mu.get(f, 0.0))) / float(sigma.get(f, 1.0)) if pd.notna(sigma.get(f, np.nan)) else 0.0
        score = abs(z) * imp
        scored.append((f, score, z))

    if not scored:
        return ["No se pudieron calcular motores causales para este registro."]

    scored.sort(key=lambda x: x[1], reverse=True)
    out: list[str] = []
    for f, _, z in scored[:top_n]:
        direction = "al alza" if z >= 0 else "a la baja"
        out.append(
            f"{_feature_to_human(f)} ({direction}, intensidad relativa z={z:.2f})"
        )
    return out


def _load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"No existe metadata de modelos: {METADATA_PATH}")
    with METADATA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _selected_model_paths(metadata: dict[str, Any]) -> dict[str, dict[str, str]]:
    selected = metadata.get("selected_models", {})
    out: dict[str, dict[str, str]] = {}

    for h in VALID_HORIZONS:
        key = f"H{h}"
        if key in selected:
            out[key] = {
                "reg": selected[key]["regression"]["model_path"],
                "clf": selected[key]["classification"]["model_path"],
            }
        else:
            out[key] = {
                "reg": str(ARTIFACTS_DIR / f"datagia_best_h{h}_reg.joblib"),
                "clf": str(ARTIFACTS_DIR / f"datagia_best_h{h}_clf.joblib"),
            }
    return out


def _resolve_model_path(path_value: str) -> Path:
    r"""Resolve model path in a cross-platform way.

    Handles absolute Windows paths (C:\Users\...) stored in metadata by
    extracting only the filename and searching in ARTIFACTS_DIR.
    """
    p = Path(path_value)

    # Always try ARTIFACTS_DIR first using just the filename
    # This handles Windows absolute paths on Linux/Mac
    candidate = ARTIFACTS_DIR / p.name
    if candidate.exists():
        return candidate

    # Fallback: try other locations
    if p.is_absolute():
        # Absolute path from same system - try directly
        if p.exists():
            return p
        # Cross-platform absolute path - use filename only
        candidate = ARTIFACTS_DIR / p.name
        if candidate.exists():
            return candidate

    candidates = [
        PROJECT_ROOT / p,
        p,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return ARTIFACTS_DIR / p.name


def _selected_feature_columns(metadata: dict[str, Any], horizon: int, task: str) -> list[str]:
    key = f"H{horizon}"
    block = metadata.get("selected_models", {}).get(key, {})
    if not block:
        raise ValueError(f"No se encontro bloque {key} en model_metadata.json")

    if task == "regresion":
        cols = block["regression"].get("expected_columns", [])
    else:
        cols = block["classification"].get("expected_columns", [])

    if not cols:
        raise ValueError(f"No se encontraron expected_columns para {key} {task}")
    return cols


def _load_models(model_paths: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for h_key, paths in model_paths.items():
        reg_path = _resolve_model_path(paths["reg"])
        clf_path = _resolve_model_path(paths["clf"])

        if not reg_path.exists() or not clf_path.exists():
            raise FileNotFoundError(f"Modelos no encontrados para {h_key}: {reg_path}, {clf_path}")

        loaded[h_key] = {
            "reg": joblib.load(reg_path),
            "clf": joblib.load(clf_path),
        }
    return loaded


def _load_base_filtered_for_h(horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns aligned train/test raw rows for horizon-specific valid subset."""
    df = pd.read_csv(BASE_DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values(["date", "provincia", "cereal_predominante"], kind="mergesort")
    df = df.reset_index(drop=True)

    base_col = "precio_provincial_lag_1"
    tgt_col = f"precio_provincial_TARGET_H{horizon}"

    valid = pd.to_numeric(df[base_col], errors="coerce").notna() & pd.to_numeric(
        df[base_col], errors="coerce"
    ).ne(0) & pd.to_numeric(df[tgt_col], errors="coerce").notna()

    filtered = df.loc[valid].copy().reset_index(drop=True)
    train_raw = filtered.loc[filtered["date"] < CUT_DATE].reset_index(drop=True)
    test_raw = filtered.loc[filtered["date"] >= CUT_DATE].reset_index(drop=True)
    return train_raw, test_raw


def _source_coverage_report() -> str:
    """Best-effort per-source freshness check, for actionable 'no hay panel' errors.

    Diagnoses which of the many upstream feeds (MAPA prices/costes, clima,
    mercados, EUR/USD) is the binding constraint on how far the panel reaches,
    instead of forcing the user to guess (see auditor report, mes 2026-05).
    """
    utils_dir = DATA_PROCESSED_DIR / "auto" / "utils"
    checks: list[tuple[str, Path, str | None]] = [
        ("Precios agricultores MAPA (prepag2_total)", utils_dir / "prepag2_total.csv", "date"),
        ("Costes insumos MAPA (prepag1_total)", utils_dir / "prepag1_total.csv", "date"),
        ("Indices MAPA (Indpag1_total)", utils_dir / "Indpag1_total.csv", "date"),
        ("Indices MAPA (Indpag2_total)", utils_dir / "Indpag2_total.csv", "date"),
        ("Clima GEE (climate_monthly_provinces)", utils_dir / "climate_monthly_provinces.csv", None),
        ("Mercados intl. (mercados_internacionales)", utils_dir / "mercados_internacionales.csv", "week_start"),
        ("EUR/USD (eur_usd_weekly)", utils_dir / "eur_usd_weekly.csv", "week_start"),
    ]
    lines = []
    for label, path, datecol in checks:
        if not path.exists():
            lines.append(f"  - {label}: fichero no encontrado ({path})")
            continue
        try:
            df = pd.read_csv(path)
            if datecol is None and {"year", "month"}.issubset(df.columns):
                dates = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
            else:
                dates = pd.to_datetime(df[datecol], errors="coerce")
            # Treat 0.0 as "not yet published" for MAPA price/index series, matching
            # the convention used by load_raw.load_targets (0 -> NaN).
            value_cols = [c for c in df.columns if c not in (datecol, "year", "month")]
            if value_cols:
                non_placeholder = (df[value_cols].replace(0, np.nan).notna().any(axis=1))
                dates = dates.where(non_placeholder)
            lines.append(f"  - {label}: hasta {dates.max():%Y-%m}")
        except Exception as exc:
            lines.append(f"  - {label}: no se pudo determinar ({exc})")
    return (
        "Cobertura actual por fuente (ultimo dato real disponible, no placeholder):\n"
        + "\n".join(lines)
        + "\nLa fuente con la fecha mas antigua es la que limita el panel; "
        "actualiza esa fuente en data/raw/ y reejecuta build_dataset."
    )


def _load_panel_all_months() -> pd.DataFrame:
    """Load base panel without filtering by target availability.

    Unlike _load_base_filtered_for_h, this does NOT require future price targets
    to be non-null. Used for production inference where targets don't exist yet
    (e.g. June 2029 with MAPA data only through March 2029).
    Only requires a valid base price (precio_provincial_lag_1).
    """
    df = pd.read_csv(BASE_DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values(
        ["date", "provincia", "cereal_predominante"], kind="mergesort"
    ).reset_index(drop=True)
    base_col = "precio_provincial_lag_1"
    valid = pd.to_numeric(df[base_col], errors="coerce").notna() & pd.to_numeric(
        df[base_col], errors="coerce"
    ).ne(0)
    return df.loc[valid].copy().reset_index(drop=True)


def _build_features_from_raw_rows(
    rows: pd.DataFrame,
    expected_cols: list[str],
) -> pd.DataFrame:
    """Build model-ready feature matrix from raw panel rows, aligned to training schema.

    Mirrors prepare_data._build_features but does NOT require target columns.
    Used in production inference mode where future targets are unavailable.
    Columns not present in the inference batch (e.g. one-hot dummies for absent
    provinces) are filled with 0 to match the training schema.
    """
    drop = set(BLACKLIST) | {"date"}
    for h in VALID_HORIZONS:
        drop.add(f"precio_provincial_TARGET_H{h}")
        drop.add(f"target_return_h{h}")
        drop.add(f"target_class_h{h}")

    keep = [c for c in rows.columns if c not in drop]
    feat = rows[keep].copy()

    obj_cols = feat.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in obj_cols:
        feat[col] = feat[col].astype(str)
    feat = pd.get_dummies(feat, columns=obj_cols, drop_first=False)
    feat = feat.reindex(sorted(feat.columns), axis=1)

    return feat.reindex(columns=expected_cols, fill_value=0)


def _validate_expected_columns(actual_cols: list[str], expected_cols: list[str], horizon: int, task: str) -> None:
    missing = sorted(set(expected_cols) - set(actual_cols))
    extra = sorted(set(actual_cols) - set(expected_cols))

    if missing or extra:
        raise ValueError(
            f"Column mismatch en H{horizon} {task}. Missing={missing[:10]} Extra={extra[:10]}"
        )


def _get_geo_risk_provinces() -> set[str]:
    fallback = {"Cuenca", "Lleida"}
    if not GEO_RISK_PATH.exists():
        return fallback

    try:
        df = pd.read_csv(GEO_RISK_PATH)
        if "provincia" in df.columns:
            dynamic = set(df["provincia"].dropna().astype(str).unique().tolist())
            return fallback.union(dynamic)
    except Exception:
        return fallback
    return fallback


def _signal_from_prob_return(prob_up: float, expected_return: float) -> str:
    if prob_up > PROB_BULL and expected_return > RET_BULL:
        return "ALCISTA"
    if prob_up < PROB_BEAR and expected_return < RET_BEAR:
        return "BAJISTA"
    return "NEUTRAL/ESPERA"


def _build_recommendation(
    role: str,
    preds: dict[int, HorizonPrediction],
    province: str,
    cereal: str,
    geo_risk: bool,
    timing_label: str | None = None,
) -> str:
    h1 = preds[1]
    h2 = preds[2]
    h3 = preds[3]

    if role == "vendedor":
        if h3.signal == "BAJISTA":
            base = (
                "Se detecta presion bajista consistente en H3. Se recomienda cerrar ventas "
                "de cobertura en el corto plazo para proteger margen antes de una posible "
                "debilidad adicional de precios."
            )
        elif h3.signal == "ALCISTA":
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
        if h3.signal == "ALCISTA":
            base = (
                "Fuerte presion al alza detectada. Se recomienda adelantar compras a 90 dias "
                "para reducir riesgo de encarecimiento en el horizonte estrategico."
            )
        elif h2.signal == "BAJISTA" or h3.signal == "BAJISTA":
            base = (
                "Recomendacion: Retrasar compras, se espera ventana de oportunidad con "
                "mejores precios en 60-90 dias."
            )
        else:
            base = (
                "Las señales no son concluyentes. Mantener compras fraccionadas y revisar "
                "confirmacion de tendencia en las proximas actualizaciones."
            )

    if timing_label:
        base = f"Timing Operativo: {timing_label}. " + base

    if h1.signal != h3.signal and h1.signal != "NEUTRAL/ESPERA" and h3.signal != "NEUTRAL/ESPERA":
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
    preds: dict[int, HorizonPrediction],
    recommendation: str,
    geo_risk: bool,
    coherence_msg: str,
    causal_drivers: list[str],
) -> str:
    risk_line = "SI" if geo_risk else "NO"

    def _line(h: int, days: int) -> str:
        p = preds[h]
        return (
            f"HORIZONTE H{h} ({days} DIAS): {p.signal} | "
            f"Confianza={p.confidence:.1f}% | "
            f"Magnitud Esperada={p.expected_return * 100:.2f}%"
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


def _coherence_message(preds: dict[int, HorizonPrediction]) -> str:
    h1 = preds[1].signal
    h3 = preds[3].signal
    if (h1 == "ALCISTA" and h3 == "BAJISTA") or (h1 == "BAJISTA" and h3 == "ALCISTA"):
        return (
            "ATENCION: Volatilidad a corto plazo detectada. La tendencia a 90 dias "
            "contradice el movimiento inmediato. Riesgo de atrapamiento de inventario."
        )
    if h1 == h3 and h1 != "NEUTRAL/ESPERA":
        return "Tendencia solida confirmada en todos los horizontes analizados."
    return "Sin contradiccion estructural fuerte entre corto y largo plazo."


def _pick_index(
    test_raw: pd.DataFrame,
    province: str | None,
    cereal: str | None,
    month: str | None,
) -> int | None:
    """Return row label index in test_raw, or None if the month is not present.

    Returns None only when a specific month was requested but is absent from the
    target-filtered test set (production inference mode). When no month is given,
    falls back to the last available test row so interactive use keeps working.
    """
    candidate = test_raw.copy()

    if province:
        candidate = candidate[candidate["provincia"].astype(str).str.lower() == province.lower()]
    if cereal:
        candidate = candidate[candidate["cereal_predominante"].astype(str).str.lower() == cereal.lower()]
    if month:
        month_ts = pd.to_datetime(f"{month}-01", errors="coerce")
        if pd.notna(month_ts):
            candidate = candidate[candidate["date"].dt.to_period("M") == month_ts.to_period("M")]

    if candidate.empty:
        if month is not None:
            # Specific month requested but not in target-filtered set → production inference
            return None
        # No month specified: fall back to last available test row
        candidate = test_raw

    idx = candidate.sort_values("date").index[-1]
    return int(idx)


def _row_pos_from_index(df: pd.DataFrame, idx: int) -> int:
    """Convert row label index to positional index for iloc-safe selection."""
    return int(df.index.get_loc(idx))


def _is_high_reliability(prob_up: float) -> bool:
    return prob_up >= CONFIDENCE_HIGH_UP or prob_up <= CONFIDENCE_HIGH_DOWN


def _reliability_badge(prob_up: float) -> str:
    return " [ALTA FIABILIDAD]" if _is_high_reliability(prob_up) else ""


def _market_position_label(ret_value: float, mean_value: float) -> str:
    if ret_value < mean_value:
        return "barata"
    if ret_value > mean_value:
        return "cara"
    return "en media"


def _timing_decision(
    role: str,
    h1_ret: float,
    h2_ret: float,
    h3_ret: float,
    is_h1_extreme: bool,
) -> str:
    if role == "comprador":
        if is_h1_extreme and h3_ret > h1_ret + TIMING_DELTA:
            return "COMPRA YA"
        if h2_ret < h1_ret - TIMING_DELTA or h3_ret < h1_ret - TIMING_DELTA:
            return "ESPERA PARA COMPRAR"
        return "SEGUIMIENTO ACTIVO"

    if is_h1_extreme and h3_ret < h1_ret - TIMING_DELTA:
        return "VENDE YA"
    if h3_ret > h1_ret + TIMING_DELTA:
        return "ESPERA PARA VENDER"
    return "SEGUIMIENTO ACTIVO"


def _compute_month_horizon_stats(month: str, cereal: str) -> dict[int, dict[str, float]]:
    month_ts = pd.to_datetime(f"{month}-01", errors="coerce")
    if pd.isna(month_ts):
        return {}

    out: dict[int, dict[str, float]] = {}
    for h in VALID_HORIZONS:
        _, test_raw = _load_base_filtered_for_h(h)
        mask = (
            (test_raw["date"].dt.to_period("M") == month_ts.to_period("M"))
            & (test_raw["cereal_predominante"].astype(str).str.lower() == cereal.lower())
        )
        month_df = test_raw.loc[mask].copy()
        if month_df.empty:
            continue

        ret_col = f"ret_h{h}"
        prob_col = f"prob_up_h{h}"

        X_train_r, X_test_r, _, _ = get_prepared_data(h, "regresion")
        X_train_c, X_test_c, _, _ = get_prepared_data(h, "clasificacion")
        metadata = _load_metadata()
        expected_r = _selected_feature_columns(metadata, h, "regresion")
        expected_c = _selected_feature_columns(metadata, h, "clasificacion")
        model_paths = _selected_model_paths(metadata)
        models = _load_models(model_paths)

        idxs = month_df.index.tolist()
        pos = [_row_pos_from_index(test_raw, i) for i in idxs]
        Xr = X_test_r.iloc[pos][expected_r]
        Xc = X_test_c.iloc[pos][expected_c]

        reg_model = models[f"H{h}"]["reg"]
        clf_model = models[f"H{h}"]["clf"]
        month_df[ret_col] = reg_model.predict(Xr)
        month_df[prob_col] = clf_model.predict_proba(Xc)[:, 1]

        out[h] = {
            "ret_min": float(month_df[ret_col].min()),
            "ret_max": float(month_df[ret_col].max()),
            "ret_mean": float(month_df[ret_col].mean()),
        }

    return out


def generate_tactical_bulletin(
    batch_df: pd.DataFrame,
    month: str | None = None,
    role: str = "comprador",
) -> Path:
    """Generate tactical buy/sell bulletin by cereal and horizon from batch ranking."""
    if batch_df.empty:
        raise ValueError("No se puede generar boletin tactico con batch vacio.")

    required_cols = {"provincia", "cereal_predominante", "risk_geo_alto"}
    for h in VALID_HORIZONS:
        required_cols.update({f"ret_h{h}", f"prob_up_h{h}"})

    missing = sorted(required_cols - set(batch_df.columns))
    if missing:
        raise ValueError(f"Faltan columnas para boletin tactico: {missing}")

    if month:
        month_text = month
    else:
        first_date = pd.to_datetime(batch_df["date"], errors="coerce").dropna()
        month_text = first_date.iloc[0].strftime("%Y-%m") if not first_date.empty else "desconocido"

    cereals = sorted(batch_df["cereal_predominante"].dropna().astype(str).unique().tolist())

    report_lines: list[str] = [
        "INFORME TACTICO DATAGIA v1.3",
        f"FECHA DE ANALISIS: {month_text}",
        "",
    ]
    console_lines: list[str] = [
        f"RESUMEN TACTICO ({month_text})",
    ]
    risk_notes: list[str] = []

    timing_breaks: dict[str, list[str]] = {}

    for h in VALID_HORIZONS:
        ret_col = f"ret_h{h}"
        prob_col = f"prob_up_h{h}"

        report_lines.append(f"=== HORIZONTE H{h} ===")

        horizon_spreads: list[float] = []
        for cereal in cereals:
            cereal_df = batch_df[
                batch_df["cereal_predominante"].astype(str).str.lower() == cereal.lower()
            ].copy()
            cereal_df = cereal_df.dropna(subset=[ret_col, prob_col])
            if cereal_df.empty:
                continue

            mean_ret = float(cereal_df[ret_col].astype(float).mean())
            cheap_break = cereal_df[cereal_df[ret_col].astype(float) < (mean_ret - BENCHMARK_BREAK_DELTA)]
            expensive_break = cereal_df[cereal_df[ret_col].astype(float) > (mean_ret + BENCHMARK_BREAK_DELTA)]
            break_key = f"H{h}::{cereal}"
            timing_breaks[break_key] = []
            if not cheap_break.empty:
                cheap_names = sorted(cheap_break["provincia"].astype(str).unique().tolist())
                timing_breaks[break_key].append(
                    "Baratas vs media: " + ", ".join(cheap_names)
                )
            if not expensive_break.empty:
                exp_names = sorted(expensive_break["provincia"].astype(str).unique().tolist())
                timing_breaks[break_key].append(
                    "Caras vs media: " + ", ".join(exp_names)
                )

            buy_idx = cereal_df[ret_col].astype(float).idxmin()
            sell_idx = cereal_df[ret_col].astype(float).idxmax()

            buy_row = cereal_df.loc[buy_idx]
            sell_row = cereal_df.loc[sell_idx]

            buy_ret = float(buy_row[ret_col])
            sell_ret = float(sell_row[ret_col])
            spread = sell_ret - buy_ret
            horizon_spreads.append(spread)

            buy_conf = (1.0 - float(buy_row[prob_col])) * 100.0
            sell_conf = float(sell_row[prob_col]) * 100.0
            buy_prob = float(buy_row[prob_col])
            sell_prob = float(sell_row[prob_col])

            # Filter out low-quality tactical blocks where both extremes are neutral.
            if not _is_high_reliability(buy_prob) and not _is_high_reliability(sell_prob):
                if spread > SPREAD_MIN:
                    spread_msg = (
                        f"Spread de Arbitraje: Existe un diferencial de oportunidad del "
                        f"{spread * 100:.1f}% entre comprar en {buy_row['provincia']} "
                        f"y vender en {sell_row['provincia']}."
                    )
                else:
                    spread_msg = "Spread de Arbitraje: Mercado en equilibrio geografico."

                report_lines.extend(
                    [
                        f"Cereal: {cereal}",
                        "Sin señal tactica robusta: probabilidades cercanas al 50% (ruido neutral).",
                        spread_msg,
                        "",
                    ]
                )
                continue

            is_h1_extreme = False
            if h == 1:
                if role == "comprador":
                    is_h1_extreme = buy_row["provincia"] == cereal_df.loc[buy_idx, "provincia"]
                else:
                    is_h1_extreme = sell_row["provincia"] == cereal_df.loc[sell_idx, "provincia"]

            timing_tag = _timing_decision(
                role=role,
                h1_ret=float(buy_row[ret_col] if role == "comprador" else sell_row[ret_col]),
                h2_ret=float(cereal_df["ret_h2"].mean()) if "ret_h2" in cereal_df.columns else float(buy_row[ret_col]),
                h3_ret=float(cereal_df["ret_h3"].mean()) if "ret_h3" in cereal_df.columns else float(sell_row[ret_col]),
                is_h1_extreme=is_h1_extreme,
            )

            buy_rel = _market_position_label(buy_ret, mean_ret)
            sell_rel = _market_position_label(sell_ret, mean_ret)

            report_lines.extend(
                [
                    f"Cereal: {cereal}",
                    (
                        f"🛒 COMPRA TACTICA: {buy_row['provincia']} "
                        f"(Retorno esperado: {buy_ret * 100:.1f}% | Confianza: {buy_conf:.1f}%"
                        f"{_reliability_badge(buy_prob)} | {buy_rel} vs media nacional {mean_ret * 100:.1f}%)"
                    ),
                    (
                        f"💰 VENTA TACTICA: {sell_row['provincia']} "
                        f"(Retorno esperado: {sell_ret * 100:.1f}% | Confianza: {sell_conf:.1f}%"
                        f"{_reliability_badge(sell_prob)} | {sell_rel} vs media nacional {mean_ret * 100:.1f}%)"
                    ),
                    (f"Timing Operativo ({role}): {timing_tag}"),
                    "",
                ]
            )

            if spread > SPREAD_MIN:
                report_lines.insert(
                    len(report_lines) - 1,
                    (
                        f"Spread de Arbitraje: Existe un diferencial de oportunidad del "
                        f"{spread * 100:.1f}% entre comprar en {buy_row['provincia']} "
                        f"y vender en {sell_row['provincia']}."
                    ),
                )
            else:
                report_lines.insert(
                    len(report_lines) - 1,
                    "Spread de Arbitraje: Mercado en equilibrio geografico.",
                )

            if bool(buy_row.get("risk_geo_alto", False)):
                risk_notes.append(f"H{h} {cereal} COMPRA -> {buy_row['provincia']}")
            if bool(sell_row.get("risk_geo_alto", False)):
                risk_notes.append(f"H{h} {cereal} VENTA -> {sell_row['provincia']}")

        if horizon_spreads:
            console_lines.append(f"H{h}: spread max observado {max(horizon_spreads) * 100:.1f}%")
        else:
            console_lines.append(f"H{h}: sin datos tacticos para calcular spread")

    report_lines.append("NOTAS DE RIESGO:")
    if risk_notes:
        for note in sorted(set(risk_notes)):
            report_lines.append(f"- {note} con risk_geo_alto=True")
    else:
        report_lines.append("- Sin alertas de riesgo geografico alto en recomendaciones tacticas.")

    report_lines.extend(["", "ANALISIS DE TIMING NACIONAL:"])
    if timing_breaks:
        for key in sorted(timing_breaks):
            report_lines.append(f"- {key}")
            lines = timing_breaks[key]
            if lines:
                for ln in lines:
                    report_lines.append(f"  {ln}")
            else:
                report_lines.append("  Sin provincias rompiendo la media nacional.")
    else:
        report_lines.append("- Sin datos suficientes para benchmark nacional.")

    output_path = PREDICTIONS_DIR / f"informe_tactico_{month_text}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines), encoding="utf-8")

    logger.info("Informe tactico V1.3 guardado en: %s", output_path)
    print("\n" + "\n".join(console_lines))
    print(f"Informe tactico guardado en: {output_path}")

    return output_path


def run_single(
    role: str,
    province: str | None,
    cereal: str | None,
    month: str | None,
    output_path: Path | None,
) -> dict[str, Any]:
    metadata = _load_metadata()
    model_paths = _selected_model_paths(metadata)
    models = _load_models(model_paths)
    metadata = _ensure_importance_in_metadata(metadata, models)
    geo_risk_set = _get_geo_risk_provinces()

    if month:
        try:
            _month_ts = pd.to_datetime(f"{month}-01")
            _mapa_cutoff = _month_ts - pd.DateOffset(months=MAPA_ADMIN_LAG)
            logger.info(
                "Inferencia single %s — datos MAPA disponibles hasta %s (desfase %d meses)",
                month, _mapa_cutoff.strftime("%Y-%m"), MAPA_ADMIN_LAG,
            )
        except Exception:
            pass

    predictions: dict[int, HorizonPrediction] = {}

    selected_province = None
    selected_cereal = None
    selected_month_text = None
    h3_row_for_explain: pd.DataFrame | None = None
    h3_train_for_explain: pd.DataFrame | None = None

    for h in VALID_HORIZONS:
        X_train_r, X_test_r, _, _ = get_prepared_data(h, "regresion")
        X_train_c, X_test_c, _, _ = get_prepared_data(h, "clasificacion")

        expected_r = _selected_feature_columns(metadata, h, "regresion")
        expected_c = _selected_feature_columns(metadata, h, "clasificacion")

        _validate_expected_columns(X_train_r.columns.tolist(), expected_r, h, "regresion")
        _validate_expected_columns(X_train_c.columns.tolist(), expected_c, h, "clasificacion")

        _, test_raw = _load_base_filtered_for_h(h)
        idx = _pick_index(test_raw, province, cereal, month)

        if idx is None:
            # Production inference mode: the requested month has no future targets yet.
            # Build feature vector directly from the full panel (no target filter).
            full_panel = _load_panel_all_months()
            candidate = full_panel[full_panel["date"] >= CUT_DATE].copy()
            if province:
                candidate = candidate[
                    candidate["provincia"].astype(str).str.lower() == province.lower()
                ]
            if cereal:
                candidate = candidate[
                    candidate["cereal_predominante"].astype(str).str.lower() == cereal.lower()
                ]
            if month:
                m_ts = pd.to_datetime(f"{month}-01", errors="coerce")
                if pd.notna(m_ts):
                    candidate = candidate[
                        candidate["date"].dt.to_period("M") == m_ts.to_period("M")
                    ]
            if candidate.empty:
                raise ValueError(
                    f"Sin datos de panel para {month} — actualiza datos crudos "
                    f"y ejecuta build_dataset.\n{_source_coverage_report()}"
                )
            selected_raw = candidate.sort_values("date").iloc[[-1]]
            selected_row = selected_raw.iloc[0]
            row_r = _build_features_from_raw_rows(selected_raw, expected_r)
            row_c = _build_features_from_raw_rows(selected_raw, expected_c)
            logger.warning(
                "H%d single: mes %s sin targets — modo inferencia de producción.", h, month
            )
        else:
            selected_row = test_raw.loc[idx]
            row_pos = _row_pos_from_index(test_raw, idx)
            row_r = X_test_r.iloc[[row_pos]][expected_r]
            row_c = X_test_c.iloc[[row_pos]][expected_c]

        selected_province = str(selected_row["provincia"])
        selected_cereal = str(selected_row["cereal_predominante"])
        selected_month_text = pd.to_datetime(selected_row["date"]).strftime("%Y-%m")

        reg_model = models[f"H{h}"]["reg"]
        clf_model = models[f"H{h}"]["clf"]

        expected_ret = float(reg_model.predict(row_r)[0])
        prob_up = float(clf_model.predict_proba(row_c)[:, 1][0])
        signal = _signal_from_prob_return(prob_up, expected_ret)

        predictions[h] = HorizonPrediction(
            horizon=h,
            signal=signal,
            confidence=max(prob_up, 1.0 - prob_up) * 100.0,
            expected_return=expected_ret,
            prob_up=prob_up,
        )

        if h == 3:
            h3_row_for_explain = row_r.copy()
            h3_train_for_explain = X_train_r[expected_r].copy()

    if selected_province is None or selected_cereal is None or selected_month_text is None:
        raise RuntimeError("No se pudo determinar el registro para la ficha.")

    geo_risk = selected_province in geo_risk_set
    coherence_msg = _coherence_message(predictions)

    h3_imp = metadata.get("selected_models", {}).get("H3", {}).get("regression", {}).get("feature_importance", [])
    if h3_row_for_explain is None or h3_train_for_explain is None:
        causal_drivers = ["No se pudo construir atribucion causal para H3."]
    else:
        causal_drivers = _top_causal_drivers(
            row_features=h3_row_for_explain,
            train_features=h3_train_for_explain,
            importance_records=h3_imp,
            top_n=3,
        )

    month_stats = _compute_month_horizon_stats(selected_month_text, selected_cereal)
    h1_stats = month_stats.get(1, {})
    h1_ret = float(predictions[1].expected_return)
    h2_ret = float(predictions[2].expected_return)
    h3_ret = float(predictions[3].expected_return)

    if role == "comprador":
        is_h1_extreme = bool(h1_stats) and h1_ret <= float(h1_stats.get("ret_min", h1_ret))
    else:
        is_h1_extreme = bool(h1_stats) and h1_ret >= float(h1_stats.get("ret_max", h1_ret))

    timing_label = _timing_decision(
        role=role,
        h1_ret=h1_ret,
        h2_ret=h2_ret,
        h3_ret=h3_ret,
        is_h1_extreme=is_h1_extreme,
    )

    recommendation = _build_recommendation(
        role,
        predictions,
        selected_province,
        selected_cereal,
        geo_risk,
        timing_label=timing_label,
    )

    card_text = _build_card_text(
        province=selected_province,
        cereal=selected_cereal,
        month_text=selected_month_text,
        preds=predictions,
        recommendation=recommendation,
        geo_risk=geo_risk,
        coherence_msg=coherence_msg,
        causal_drivers=causal_drivers,
    )

    if output_path is None:
        output_path = PREDICTIONS_DIR / f"ficha_decision_{selected_province}_{selected_cereal}_{selected_month_text}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(card_text, encoding="utf-8")

    logger.info("Ficha de decision guardada en: %s", output_path)
    print(card_text)

    return {
        "province": selected_province,
        "cereal": selected_cereal,
        "month": selected_month_text,
        "geo_risk": geo_risk,
        "coherence": coherence_msg,
        "timing_label": timing_label,
        "causal_drivers": causal_drivers,
        "predictions": {
            f"H{h}": {
                "signal": predictions[h].signal,
                "prob_up": predictions[h].prob_up,
                "confidence": predictions[h].confidence,
                "expected_return": predictions[h].expected_return,
            }
            for h in VALID_HORIZONS
        },
        "card_path": str(output_path),
    }


def run_batch(month: str, role: str, output_csv: Path | None) -> pd.DataFrame:
    metadata = _load_metadata()
    model_paths = _selected_model_paths(metadata)
    models = _load_models(model_paths)
    geo_risk_set = _get_geo_risk_provinces()

    month_ts = pd.to_datetime(f"{month}-01", errors="coerce")
    if pd.isna(month_ts):
        raise ValueError("--month debe tener formato YYYY-MM")

    mapa_cutoff = month_ts - pd.DateOffset(months=MAPA_ADMIN_LAG)
    logger.info(
        "Inferencia batch %s — datos MAPA disponibles hasta %s (desfase %d meses)",
        month, mapa_cutoff.strftime("%Y-%m"), MAPA_ADMIN_LAG,
    )

    h_frames: dict[int, pd.DataFrame] = {}
    for h in VALID_HORIZONS:
        X_train_r, X_test_r, _, _ = get_prepared_data(h, "regresion")
        X_train_c, X_test_c, _, _ = get_prepared_data(h, "clasificacion")

        expected_r = _selected_feature_columns(metadata, h, "regresion")
        expected_c = _selected_feature_columns(metadata, h, "clasificacion")

        _validate_expected_columns(X_train_r.columns.tolist(), expected_r, h, "regresion")
        _validate_expected_columns(X_train_c.columns.tolist(), expected_c, h, "clasificacion")

        _, test_raw = _load_base_filtered_for_h(h)
        mask = test_raw["date"].dt.to_period("M") == month_ts.to_period("M")
        raw_month = test_raw.loc[mask].copy()

        if raw_month.empty:
            # Production inference mode: month has no future targets yet.
            # Load from full panel (no target filter) and build features from scratch.
            full_panel = _load_panel_all_months()
            mask_full = full_panel["date"].dt.to_period("M") == month_ts.to_period("M")
            raw_month = full_panel.loc[mask_full].copy()
            if raw_month.empty:
                raise ValueError(
                    f"Sin datos de panel para {month}. "
                    f"Actualiza datos crudos y ejecuta build_dataset.\n"
                    f"{_source_coverage_report()}"
                )
            logger.warning(
                "H%d batch: mes %s sin targets — modo inferencia de producción. "
                "MAPA disponible hasta %s.",
                h, month, mapa_cutoff.strftime("%Y-%m"),
            )
            Xr = _build_features_from_raw_rows(raw_month, expected_r)
            Xc = _build_features_from_raw_rows(raw_month, expected_c)
        else:
            idxs = raw_month.index.tolist()
            pos = [_row_pos_from_index(test_raw, i) for i in idxs]
            Xr = X_test_r.iloc[pos][expected_r]
            Xc = X_test_c.iloc[pos][expected_c]

        reg_model = models[f"H{h}"]["reg"]
        clf_model = models[f"H{h}"]["clf"]

        ret = reg_model.predict(Xr)
        prob = clf_model.predict_proba(Xc)[:, 1]

        out = raw_month[["date", "provincia", "cereal_predominante"]].copy()
        out[f"ret_h{h}"] = ret
        out[f"prob_up_h{h}"] = prob
        out[f"signal_h{h}"] = [
            _signal_from_prob_return(float(p), float(r)) for p, r in zip(prob, ret)
        ]
        h_frames[h] = out

    merged = h_frames[1].merge(
        h_frames[2], on=["date", "provincia", "cereal_predominante"], how="outer"
    ).merge(h_frames[3], on=["date", "provincia", "cereal_predominante"], how="outer")

    merged["score_h1"] = np.abs(merged["ret_h1"]) * (2 * np.abs(merged["prob_up_h1"] - 0.5))
    merged["score_h2"] = np.abs(merged["ret_h2"]) * (2 * np.abs(merged["prob_up_h2"] - 0.5))
    merged["score_h3"] = np.abs(merged["ret_h3"]) * (2 * np.abs(merged["prob_up_h3"] - 0.5))

    merged["arbitrage_score"] = 0.2 * merged["score_h1"] + 0.3 * merged["score_h2"] + 0.5 * merged["score_h3"]
    merged["risk_geo_alto"] = merged["provincia"].isin(geo_risk_set)

    if role == "comprador":
        merged["orientacion"] = np.where(
            merged["ret_h3"] > RET_BULL,
            "COMPRA_ANTICIPADA",
            np.where(merged["ret_h3"] < RET_BEAR, "ESPERAR_COMPRA", "NEUTRAL"),
        )
    else:
        merged["orientacion"] = np.where(
            merged["ret_h3"] < RET_BEAR,
            "VENTA_ANTICIPADA",
            np.where(merged["ret_h3"] > RET_BULL, "RETRASAR_VENTA", "NEUTRAL"),
        )

    ranking = merged.sort_values("arbitrage_score", ascending=False).reset_index(drop=True)

    if output_csv is None:
        output_csv = PREDICTIONS_DIR / f"ranking_oportunidades_arbitraje_{month}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_csv, index=False)

    logger.info("Ranking batch guardado en: %s", output_csv)
    print(ranking.head(10)[[
        "date",
        "provincia",
        "cereal_predominante",
        "arbitrage_score",
        "orientacion",
        "risk_geo_alto",
    ]].to_string(index=False))

    generate_tactical_bulletin(ranking, month=month, role=role)
    return ranking


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Motor de inferencia estrategica DATAGIA v1")
    p.add_argument("--mode", choices=["single", "batch"], default="single")
    p.add_argument("--role", choices=["comprador", "vendedor"], default="comprador")
    p.add_argument("--province", type=str, default=None)
    p.add_argument("--cereal", type=str, default=None)
    p.add_argument("--month", type=str, default=None, help="Formato YYYY-MM")
    p.add_argument("--output", type=str, default=None)
    return p


def main() -> None:
    args = _parser().parse_args()
    output = Path(args.output) if args.output else None

    if args.mode == "single":
        run_single(
            role=args.role,
            province=args.province,
            cereal=args.cereal,
            month=args.month,
            output_path=output,
        )
    else:
        if not args.month:
            raise ValueError("En modo batch debes pasar --month YYYY-MM")
        run_batch(month=args.month, role=args.role, output_csv=output)


if __name__ == "__main__":
    main()
