from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.common import save_json
from src.utils.paths import resolve_saved_path, to_repo_relative_path


KNOWN_DESCRIPTIONS = {
    "timestamp": "Marca temporal de la observación.",
    "asset_id": "Identificador del activo.",
    "sequence_id": "Identificador de la secuencia/ciclo usado para generar ventanas.",
    "cycle_id": "Identificador del ciclo de producción.",
    "phase": "Fase operativa del activo.",
    "flow_kg_s": "Caudal instantáneo.",
    "dP_kPa": "Caída de presión estimada.",
    "vibration_mm_s": "Nivel de vibración.",
    "Th_in_C": "Temperatura de entrada del circuito caliente.",
    "Tc_in_C": "Temperatura de entrada del circuito frío/producto.",
    "Th_out_C": "Temperatura de salida del circuito caliente.",
    "Tc_out_C": "Temperatura de salida del circuito frío/producto.",
    "thermal_eff_proxy": "Proxy de eficiencia térmica derivado del balance térmico.",
    "heat_proxy": "Proxy de transferencia térmica.",
    "Rf_m2K_W": "Severidad física de fouling (resistencia térmica).",
    "fouling_stage": "Clase de stage informada.",
    "fouling_stage_physical": "Clase de stage derivada de los umbrales físicos de severidad.",
    "fouling_onset_event": "Evento puntual de entrada en fouling incipiente.",
    "clog_onset_event": "Evento puntual de inicio de obstrucción.",
    "ttm_to_planned_cip_min": "Tiempo hasta la siguiente CIP planificada.",
    "ttm_to_unplanned_event_min": "Tiempo hasta la siguiente intervención no planificada.",
    "time_to_fouling_onset_min": "Tiempo restante hasta el siguiente inicio de fouling.",
    "time_to_clog_onset_min": "Tiempo restante hasta el siguiente inicio de obstrucción.",
}


TARGET_KEYWORDS = (
    "fouling_onset_within_",
    "clog_onset_within_",
    "unplanned_fouling_within_",
    "time_to_",
    "ttm_to_",
    "pred_",
    "p_",
)


def infer_role(column: str) -> str:
    if column in {"timestamp", "asset_id", "sequence_id", "cycle_id", "episode_id", "batch_id"}:
        return "metadata"
    if column in {"Rf_m2K_W", "fouling_stage", "fouling_stage_physical", "fouling_onset_event", "clog_onset_event", "clog_event"}:
        return "target_or_label"
    if any(column.startswith(prefix) for prefix in TARGET_KEYWORDS):
        return "target_or_prediction"
    if column.startswith(("resid_", "z_", "phase_", "milk_", "family_", "lastmaint_")):
        return "engineered_feature"
    return "feature"


def infer_description(column: str) -> str:
    if column in KNOWN_DESCRIPTIONS:
        return KNOWN_DESCRIPTIONS[column]
    if column.startswith("resid_"):
        return f"Residual respecto a baseline sano para {column.removeprefix('resid_')}."
    if column.startswith("z_"):
        return f"Versión robustamente escalada de {column.removeprefix('z_')}."
    if column.endswith("_slope15"):
        return f"Cambio frente a una ventana histórica corta para {column.removesuffix('_slope15')}."
    if column.endswith("_mean15") or column.endswith("_mean60"):
        return f"Media móvil de {column.rsplit('_', 1)[0]}."
    if column.endswith("_std15"):
        return f"Desviación estándar móvil de {column.removesuffix('_std15')}."
    if column.startswith("phase_"):
        return "Codificación one-hot de la fase operativa."
    if column.startswith("milk_"):
        return "Codificación one-hot del tipo de leche/producto."
    if column.startswith("family_"):
        return "Codificación one-hot de la familia del equipo."
    if column.startswith("lastmaint_"):
        return "Codificación one-hot del último tipo de mantenimiento."
    if column.endswith("_min"):
        return "Variable temporal expresada en minutos."
    if column.endswith("_kPa"):
        return "Variable hidráulica expresada en kPa."
    if column.endswith("_C"):
        return "Variable térmica expresada en grados Celsius."
    return "Descripción heurística generada automáticamente."


def build_column_catalog(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "role": infer_role(column),
                "description": infer_description(column),
                "missing_pct": float(series.isna().mean() * 100.0),
                "n_unique": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["role", "column"]).reset_index(drop=True)


def collect_model_summary(artifacts_dir: str | Path, metrics_dir: str | Path) -> dict[str, Any]:
    artifacts_dir = resolve_saved_path(artifacts_dir)
    metrics_dir = resolve_saved_path(metrics_dir)
    summary: dict[str, Any] = {
        "model_manifest": None,
        "ablation_summary": None,
        "selected_metrics": None,
    }
    manifest_path = artifacts_dir / "model_manifest.json"
    if manifest_path.exists():
        summary["model_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected = summary["model_manifest"].get("selected_scenario")
        if selected:
            metrics_path = metrics_dir / f"{selected}_test_window_metrics.json"
            if metrics_path.exists():
                summary["selected_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    ablation_path = metrics_dir / "ablation_summary.json"
    if ablation_path.exists():
        summary["ablation_summary"] = json.loads(ablation_path.read_text(encoding="utf-8"))
    return summary


def write_column_catalog(df: pd.DataFrame, out_csv: str | Path, artifacts_dir: str | Path, metrics_dir: str | Path) -> dict[str, Any]:
    out_csv = resolve_saved_path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    catalog = build_column_catalog(df)
    catalog.to_csv(out_csv, index=False)

    summary = collect_model_summary(artifacts_dir, metrics_dir)
    summary_path = out_csv.with_suffix(".model_summary.json")
    save_json(summary_path, summary)
    return {
        "column_catalog": to_repo_relative_path(out_csv),
        "model_summary": to_repo_relative_path(summary_path),
    }
