from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from scripts.build_data_manifest import SOURCE_SPECS, build_source_manifest
from src.reproducibility.mixed_context import leakage_audit, modeling_summary, quantity_feature_columns, trigger_feature_columns
from src.reproducibility.runtime import ensure_optional_dependency, official_paths, repo_root
from src.utils import ensure_directory, read_json, write_json

ensure_optional_dependency("matplotlib", repo_root_path=repo_root())
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPORTS_ROOT = repo_root() / "reports"
FIGURES_DIR = REPORTS_ROOT / "figures" / "eda"
TABLES_DIR = REPORTS_ROOT / "tables" / "eda"
NOTEBOOKS_DIR = REPORTS_ROOT / "notebooks"
EDA_DIR = REPORTS_ROOT / "eda"
SCOPE = "mixed_context"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def _ensure_report_dirs() -> None:
    for directory in [FIGURES_DIR, TABLES_DIR, NOTEBOOKS_DIR, EDA_DIR]:
        ensure_directory(directory)


def _figure_path(name: str) -> Path:
    _ensure_report_dirs()
    return FIGURES_DIR / name


def _table_path(name: str) -> Path:
    _ensure_report_dirs()
    return TABLES_DIR / name


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) or np.isinf(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _artifact_rel(path: Path) -> str:
    return path.resolve().relative_to(repo_root().resolve()).as_posix()


def _save_plot(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return _artifact_rel(path)


def _save_table(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return _artifact_rel(path)


def _execution_metadata() -> dict[str, Any]:
    return {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": SCOPE,
        "commit": _git_commit(),
    }


def _source_manifests() -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for source_id in SOURCE_SPECS:
        manifest_path = repo_root() / SOURCE_SPECS[source_id]["raw_dir"] / "source_manifest.json"
        if manifest_path.exists():
            manifests.append(read_json(manifest_path))
        else:
            manifests.append(build_source_manifest(source_id, write=False))
    return manifests


def _safe_sheet_profile(path: Path) -> dict[str, Any]:
    ensure_optional_dependency("openpyxl", repo_root_path=repo_root())
    try:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            sheet_name = None
        else:
            frame = pd.read_excel(path, sheet_name=0)
            sheet_name = "sheet_0"
    except Exception as exc:
        return {
            "path": _artifact_rel(path),
            "sheet_name": None,
            "row_count": None,
            "column_count": None,
            "columns": [],
            "dtypes": {},
            "missing_pct": {},
            "date_min": None,
            "date_max": None,
            "error": str(exc),
        }

    date_min = None
    date_max = None
    for candidate in frame.columns:
        if "date" in str(candidate).lower() or "period" in str(candidate).lower():
            parsed = pd.to_datetime(frame[candidate], errors="coerce")
            if parsed.notna().any():
                date_min = str(parsed.min().date())
                date_max = str(parsed.max().date())
                break
    return {
        "path": _artifact_rel(path),
        "sheet_name": sheet_name,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_pct": {str(column): float(frame[column].isna().mean()) for column in frame.columns},
        "date_min": date_min,
        "date_max": date_max,
        "error": None,
    }


def run_data_sources_audit(scope: str = SCOPE) -> dict[str, Any]:
    manifests = _source_manifests()
    rows = []
    for manifest in manifests:
        raw_paths = [entry["path"] for entry in manifest.get("raw_files", [])]
        derived = manifest.get("derived_artifacts", [])
        rows.append(
            {
                "source_id": manifest["source_id"],
                "organization": manifest["organization"],
                "status": manifest["evidence_status"],
                "official_url": manifest["official_url"],
                "download_url_or_endpoint": manifest["download_url_or_endpoint"],
                "license_or_terms_url": manifest["license_or_terms_url"],
                "raw_path": "; ".join(raw_paths),
                "processed_artifacts": "; ".join(derived),
                "role": manifest["role"],
                "limitations": manifest["limitations"],
                "raw_exists": all((repo_root() / path).exists() for path in raw_paths) if raw_paths else False,
                "processed_exists": all((repo_root() / path).exists() for path in derived) if derived else False,
                "hash_available": all(bool(entry.get("sha256")) for entry in manifest.get("raw_files", [])),
            }
        )
    audit_df = pd.DataFrame(rows).sort_values(["status", "source_id"]).reset_index(drop=True)
    table_rel = _save_table(audit_df, _table_path("data_sources_audit__mixed_context.csv"))

    counts = audit_df["status"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", color=["#2f6f4f", "#7f8c8d", "#c98f2b", "#b04c4c"], ax=ax)
    ax.set_title("CU28 Source Count by Evidence Status")
    ax.set_xlabel("evidence_status")
    ax.set_ylabel("source_count")
    figure_rel = _save_plot(fig, _figure_path("data_sources_status_counts__mixed_context.png"))

    return {
        **_execution_metadata(),
        "notebook": "00_data_sources_audit.ipynb",
        "outputs": [table_rel, figure_rel],
        "findings": [
            "INE_CPI and MAPA_SLAUGHTER_MAPA are the active contextual/proxy sources for mixed_context.",
            "MAPA_PRICES_OM remains traced only and is not defended as an active weekly feed.",
            "Candidate sources do not feed the official mixed_context pipeline.",
        ],
        "limitations": [
            "Source status reflects the defended route, not every exploratory source ever evaluated.",
        ],
    }


def run_raw_data_profile(scope: str = SCOPE) -> dict[str, Any]:
    manifests = _source_manifests()
    inventory_rows = []
    quality_rows = []
    coverage_rows = []

    for manifest in manifests:
        for raw_file in manifest.get("raw_files", []):
            path = repo_root() / raw_file["path"]
            inventory_rows.append(
                {
                    "source_id": manifest["source_id"],
                    "path": raw_file["path"],
                    "size_bytes": raw_file["size_bytes"],
                    "sha256": raw_file["sha256"],
                    "access_date": manifest["access_date"],
                    "retrieval_method": manifest["retrieval_method"],
                }
            )
            profile = _safe_sheet_profile(path)
            quality_rows.append(
                {
                    "source_id": manifest["source_id"],
                    "path": profile["path"],
                    "sheet_name": profile["sheet_name"],
                    "row_count": profile["row_count"],
                    "column_count": profile["column_count"],
                    "columns": json.dumps(profile["columns"], ensure_ascii=False),
                    "date_min": profile["date_min"],
                    "date_max": profile["date_max"],
                    "error": profile["error"],
                }
            )
            if profile["missing_pct"]:
                for column, missing_pct in profile["missing_pct"].items():
                    coverage_rows.append(
                        {
                            "source_id": manifest["source_id"],
                            "path": profile["path"],
                            "column": column,
                            "missing_pct": missing_pct,
                        }
                    )

    inventory_df = pd.DataFrame(inventory_rows)
    quality_df = pd.DataFrame(quality_rows)
    coverage_df = pd.DataFrame(coverage_rows)

    inventory_rel = _save_table(inventory_df, _table_path("raw_file_inventory__mixed_context.csv"))
    quality_rel = _save_table(quality_df, _table_path("raw_data_quality__mixed_context.csv"))

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    if not coverage_df.empty:
        coverage_df.groupby("column", as_index=True)["missing_pct"].mean().sort_values(ascending=False).head(15).plot(
            kind="bar", ax=ax1, color="#b75d32"
        )
    ax1.set_title("Average Raw Missing Values by Column")
    ax1.set_ylabel("missing_pct")
    missing_rel = _save_plot(fig1, _figure_path("raw_missing_values__mixed_context.png"))

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    temporal_df = quality_df.dropna(subset=["date_min", "date_max"]).copy()
    if not temporal_df.empty:
        temporal_df["date_min"] = pd.to_datetime(temporal_df["date_min"])
        temporal_df["date_max"] = pd.to_datetime(temporal_df["date_max"])
        temporal_df = temporal_df.sort_values("date_min").reset_index(drop=True)
        for index, row in temporal_df.iterrows():
            ax2.hlines(index, row["date_min"], row["date_max"], linewidth=6, color="#3c7dc4")
        ax2.set_yticks(range(len(temporal_df)))
        ax2.set_yticklabels(temporal_df["source_id"])
    ax2.set_title("Raw Temporal Coverage by Source")
    ax2.set_xlabel("date")
    coverage_rel = _save_plot(fig2, _figure_path("raw_temporal_coverage__mixed_context.png"))

    return {
        **_execution_metadata(),
        "notebook": "01_raw_data_profile.ipynb",
        "outputs": [inventory_rel, quality_rel, missing_rel, coverage_rel],
        "findings": [
            "Raw snapshots are external contextual files, not plant-level inventory or purchase history.",
            "The profile step documents file structure, completeness and temporal reach before ETL.",
        ],
        "limitations": [
            "Spreadsheet profiling uses the first sheet as a reproducible audit entrypoint.",
        ],
    }


def run_external_context_eda(scope: str = SCOPE) -> dict[str, Any]:
    external_long = pd.read_csv(repo_root() / "data/processed/external/context/external_long.csv")
    context_weekly = pd.read_csv(repo_root() / "data/processed/external/context/context_weekly_for_simulation.csv")
    external_long["date"] = pd.to_datetime(external_long["date"], errors="coerce")
    context_weekly["date"] = pd.to_datetime(context_weekly["date"], errors="coerce")
    context_weekly["demand_supply_gap"] = context_weekly["demand_index"] - context_weekly["supply_index"]

    summary_long = (
        external_long.groupby(["source", "dataset", "subseries"], dropna=False)
        .agg(variable=("unit", "first"), min_date=("date", "min"), max_date=("date", "max"), observations=("date", "size"), missing_rate=("value", lambda s: float(pd.to_numeric(s, errors="coerce").isna().mean())))
        .reset_index()
    )
    context_summary = context_weekly.describe(include="all").transpose().reset_index().rename(columns={"index": "variable"})
    combined_summary = pd.concat(
        [
            summary_long.assign(section="external_long"),
            context_summary.assign(section="context_weekly"),
        ],
        ignore_index=True,
        sort=False,
    )
    summary_rel = _save_table(combined_summary, _table_path("external_context_summary__mixed_context.csv"))

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    context_weekly.set_index("date")[["demand_index", "supply_index", "demand_supply_gap"]].plot(ax=ax1)
    ax1.set_title("External Context Weekly Signals")
    ax1.set_ylabel("index")
    timeseries_rel = _save_plot(fig1, _figure_path("external_context_timeseries__mixed_context.png"))

    corr = context_weekly[["demand_index", "supply_index", "purchase_price_index", "demand_supply_gap"]].corr(numeric_only=True)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    heat = ax2.imshow(corr.values, cmap="Blues")
    ax2.set_xticks(range(len(corr.columns)))
    ax2.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax2.set_yticks(range(len(corr.columns)))
    ax2.set_yticklabels(corr.columns)
    ax2.set_title("External Context Correlation")
    fig2.colorbar(heat, ax=ax2)
    correlation_rel = _save_plot(fig2, _figure_path("external_context_correlation__mixed_context.png"))

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    coverage = context_weekly.notna().mean().sort_values(ascending=False)
    coverage.plot(kind="bar", ax=ax3, color="#4d7f6f")
    ax3.set_title("Weekly Coverage by Context Variable")
    ax3.set_ylabel("coverage_rate")
    coverage_rel = _save_plot(fig3, _figure_path("external_context_coverage__mixed_context.png"))

    return {
        **_execution_metadata(),
        "notebook": "02_external_context_eda.ipynb",
        "outputs": [summary_rel, timeseries_rel, correlation_rel, coverage_rel],
        "findings": [
            "External signals act as contextual proxies and do not replace internal plant data.",
            "purchase_price_index remains documented as a traced/fallback signal when weekly price extraction is unavailable.",
        ],
        "limitations": [
            "The active defended route uses CPI and slaughter as contextual evidence; MAPA_PRICES_OM is traced only.",
        ],
    }


def run_synthetic_plant_layer_eda(scope: str = SCOPE) -> dict[str, Any]:
    synthetic_path = repo_root() / "data/processed/synthetic/plant/synthetic_plant_layer__mixed_context.csv"
    metadata_path = repo_root() / "data/processed/synthetic/plant/synthetic_plant_metadata__mixed_context.json"
    df = pd.read_csv(synthetic_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    metadata = read_json(metadata_path) if metadata_path.exists() else {}

    variables = [
        "current_inventory_tons",
        "expected_requirement_tons",
        "lead_time_days",
        "safety_coverage_days",
        "expected_yield_rate",
        "expected_waste_rate",
        "synthetic_procurement_need",
    ]
    summary_rows = []
    for variable in variables:
        series = pd.to_numeric(df[variable], errors="coerce")
        summary_rows.append(
            {
                "variable": variable,
                "definition": metadata.get("column_definitions", {}).get(variable, variable),
                "unit": "tons" if "tons" in variable or "inventory" in variable or "requirement" in variable else "ratio_or_days",
                "observed_min": float(series.min()),
                "observed_max": float(series.max()),
                "origin": "synthetic" if variable not in {"destination_profile"} else "synthetic_context",
                "role": "operational_input" if variable != "synthetic_procurement_need" else "upstream_signal",
            }
        )
    variable_df = pd.DataFrame(summary_rows)
    by_profile = (
        df.groupby("destination_profile")[variables]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    by_profile.columns = ["__".join([str(part) for part in column if part]).strip("_") for column in by_profile.columns.to_flat_index()]

    variable_rel = _save_table(variable_df, _table_path("synthetic_layer_summary__mixed_context.csv"))
    by_profile_rel = _save_table(by_profile, _table_path("synthetic_layer_by_profile__mixed_context.csv"))

    def _boxplot(column: str, filename: str, title: str) -> str:
        fig, ax = plt.subplots(figsize=(8, 4))
        df.boxplot(column=column, by="destination_profile", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("destination_profile")
        ax.set_ylabel(column)
        fig.suptitle("")
        return _save_plot(fig, _figure_path(filename))

    requirement_rel = _boxplot(
        "expected_requirement_tons",
        "synthetic_requirement_by_profile__mixed_context.png",
        "Expected Requirement by Destination Profile",
    )
    inventory_rel = _boxplot(
        "current_inventory_tons",
        "synthetic_inventory_distribution__mixed_context.png",
        "Current Inventory by Destination Profile",
    )
    lead_rel = _boxplot("lead_time_days", "synthetic_lead_time_by_profile__mixed_context.png", "Lead Time by Destination Profile")

    fig_yw, ax_yw = plt.subplots(figsize=(8, 4))
    grouped = df.groupby("destination_profile")[["expected_yield_rate", "expected_waste_rate"]].mean()
    grouped.plot(kind="bar", ax=ax_yw)
    ax_yw.set_title("Yield and Waste by Destination Profile")
    ax_yw.set_ylabel("rate")
    yield_waste_rel = _save_plot(fig_yw, _figure_path("synthetic_yield_waste_by_profile__mixed_context.png"))

    procurement_rel = _boxplot(
        "synthetic_procurement_need",
        "synthetic_procurement_need_by_profile__mixed_context.png",
        "Synthetic Procurement Need by Destination Profile",
    )

    corr = df[variables].corr(numeric_only=True)
    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
    heat = ax_corr.imshow(corr.values, cmap="Greens")
    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax_corr.set_yticks(range(len(corr.columns)))
    ax_corr.set_yticklabels(corr.columns)
    ax_corr.set_title("Synthetic Operational Correlation")
    fig_corr.colorbar(heat, ax=ax_corr)
    corr_rel = _save_plot(fig_corr, _figure_path("synthetic_layer_correlation__mixed_context.png"))

    return {
        **_execution_metadata(),
        "notebook": "03_synthetic_plant_layer_eda.ipynb",
        "outputs": [
            variable_rel,
            by_profile_rel,
            requirement_rel,
            inventory_rel,
            lead_rel,
            yield_waste_rel,
            procurement_rel,
            corr_rel,
        ],
        "findings": [
            "destination_profile represents expected productive destination, not the purchased product itself.",
            "synthetic_procurement_need remains an upstream pressure signal, not the final recommended quantity.",
        ],
        "limitations": [
            "Inventory, lead time, waste and yield remain synthetic unless replaced with customer data.",
        ],
    }


def _feature_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        if column in {"date", "destination_profile", "raw_material_id"}:
            stage = "identifier"
            origin = "calculated"
        elif column.startswith("synthetic_"):
            stage = "upstream_or_operational"
            origin = "synthetic"
        elif column.startswith(("demand_", "supply_", "purchase_price_")):
            stage = "external_context"
            origin = "proxy"
        elif "__" in column:
            stage = "encoded_context"
            origin = "calculated"
        elif column.endswith(("_lag_1", "_lag_2", "_lag_4", "_lag_8")):
            stage = "lag_feature"
            origin = "calculated"
        elif "_roll_mean_" in column:
            stage = "rolling_feature"
            origin = "calculated"
        else:
            stage = "operational_input"
            origin = "synthetic"
        rows.append(
            {
                "name": column,
                "dtype": str(df[column].dtype),
                "stage": stage,
                "origin": origin,
                "role": "target" if column in {"synthetic_procurement_need", "purchase_trigger_label", "quantity_optimizer_target_tons"} else "input",
                "is_lag": "_lag_" in column,
                "is_rolling": "_roll_mean_" in column,
                "is_contemporaneous": not ("_lag_" in column or "_roll_mean_" in column),
                "excluded_by_leakage": column in {"order_quantity_tons", "purchase_trigger_flag", "purchase_trigger_proba"},
            }
        )
    return pd.DataFrame(rows)


def run_feature_engineering_audit(scope: str = SCOPE) -> dict[str, Any]:
    dataset_path = repo_root() / "data/processed/baseline/feature_engineering_modeling__mixed_context.csv"
    df = pd.read_csv(dataset_path)
    inventory_df = _feature_inventory(df)
    trigger_audit = leakage_audit(df, trigger_feature_columns(df), stage="trigger")
    quantity_audit = leakage_audit(df, quantity_feature_columns(df), stage="quantity_optimizer")
    leakage_df = pd.concat([trigger_audit, quantity_audit], ignore_index=True)

    inventory_rel = _save_table(inventory_df, _table_path("feature_inventory__mixed_context.csv"))
    leakage_rel = _save_table(leakage_df, _table_path("leakage_audit__mixed_context.csv"))

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    inventory_df["stage"].value_counts().plot(kind="bar", ax=ax1, color="#4a6da7")
    ax1.set_title("Features by Family")
    ax1.set_ylabel("feature_count")
    family_rel = _save_plot(fig1, _figure_path("features_by_family__mixed_context.png"))

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    inventory_df["origin"].value_counts().plot(kind="bar", ax=ax2, color="#b8860b")
    ax2.set_title("Features by Origin")
    ax2.set_ylabel("feature_count")
    origin_rel = _save_plot(fig2, _figure_path("features_by_origin__mixed_context.png"))

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    df.isna().mean().sort_values(ascending=False).head(20).plot(kind="bar", ax=ax3, color="#b04c4c")
    ax3.set_title("Missing Values After Feature Engineering")
    ax3.set_ylabel("missing_pct")
    missing_rel = _save_plot(fig3, _figure_path("feature_missing_values__mixed_context.png"))

    corr_columns = [
        column
        for column in [
            "synthetic_procurement_need",
            "expected_requirement_tons",
            "current_inventory_tons",
            "lead_time_days",
            "safety_coverage_days",
            "demand_index",
            "supply_index",
            "demand_supply_gap",
        ]
        if column in df.columns
    ]
    corr = df[corr_columns].corr(numeric_only=True)
    fig4, ax4 = plt.subplots(figsize=(6, 5))
    heat = ax4.imshow(corr.values, cmap="Purples")
    ax4.set_xticks(range(len(corr.columns)))
    ax4.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax4.set_yticks(range(len(corr.columns)))
    ax4.set_yticklabels(corr.columns)
    ax4.set_title("Feature Correlation Audit")
    fig4.colorbar(heat, ax=ax4)
    correlation_rel = _save_plot(fig4, _figure_path("feature_correlation__mixed_context.png"))

    return {
        **_execution_metadata(),
        "notebook": "04_feature_engineering_audit.ipynb",
        "outputs": [inventory_rel, leakage_rel, family_rel, origin_rel, missing_rel, correlation_rel],
        "findings": [
            "The upstream predictor excludes downstream decision outputs and trigger outputs.",
            "The quantity optimizer only consumes trigger-stage outputs plus operational/context variables.",
        ],
        "limitations": [
            "Leakage checks are rule-based and documented for auditability.",
        ],
    }


def run_modeling_dataset_eda(scope: str = SCOPE) -> dict[str, Any]:
    dataset_path = repo_root() / "data/processed/baseline/feature_engineering_modeling__mixed_context.csv"
    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    summary_payload = modeling_summary(df)
    summary_df = pd.DataFrame([{"metric": key, "value": json.dumps(value) if isinstance(value, list) else value} for key, value in summary_payload.items()])
    quality_df = pd.DataFrame(
        {
            "column": df.columns,
            "missing_pct": [float(df[column].isna().mean()) for column in df.columns],
            "nunique": [int(df[column].nunique(dropna=True)) for column in df.columns],
            "is_constant": [bool(df[column].nunique(dropna=False) <= 1) for column in df.columns],
        }
    )
    target_by_profile = (
        df.groupby("destination_profile")["synthetic_procurement_need"]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    trigger_balance = (
        df.groupby("destination_profile")["purchase_trigger_label"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "trigger_rate", "sum": "trigger_positive_rows"})
    )

    summary_rel = _save_table(summary_df, _table_path("modeling_dataset_summary__mixed_context.csv"))
    quality_rel = _save_table(quality_df, _table_path("modeling_dataset_quality__mixed_context.csv"))
    target_rel = _save_table(target_by_profile, _table_path("target_by_profile__mixed_context.csv"))
    trigger_rel = _save_table(trigger_balance, _table_path("trigger_balance__mixed_context.csv"))

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    df["synthetic_procurement_need"].plot(kind="hist", bins=30, ax=ax1, color="#52796f")
    ax1.set_title("Synthetic Procurement Need Distribution")
    ax1.set_xlabel("synthetic_procurement_need")
    target_dist_rel = _save_plot(fig1, _figure_path("target_distribution__mixed_context.png"))

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    df.boxplot(column="synthetic_procurement_need", by="destination_profile", ax=ax2)
    ax2.set_title("Synthetic Procurement Need by Destination Profile")
    ax2.set_xlabel("destination_profile")
    ax2.set_ylabel("synthetic_procurement_need")
    fig2.suptitle("")
    target_profile_rel = _save_plot(fig2, _figure_path("target_by_profile__mixed_context.png"))

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    df["purchase_trigger_label"].value_counts().sort_index().plot(kind="bar", ax=ax3, color="#c97c5d")
    ax3.set_title("Trigger Label Balance")
    ax3.set_xlabel("purchase_trigger_label")
    trigger_balance_rel = _save_plot(fig3, _figure_path("trigger_balance__mixed_context.png"))

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    trigger_balance.set_index("destination_profile")["trigger_rate"].plot(kind="bar", ax=ax4, color="#7f5539")
    ax4.set_title("Trigger Rate by Destination Profile")
    ax4.set_ylabel("trigger_rate")
    trigger_profile_rel = _save_plot(fig4, _figure_path("trigger_by_profile__mixed_context.png"))

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    df.set_index("date")[["synthetic_procurement_need", "expected_requirement_tons", "current_inventory_tons"]].plot(ax=ax5)
    ax5.set_title("Modeling Dataset Weekly Signals")
    ax5.set_ylabel("tons")
    timeseries_rel = _save_plot(fig5, _figure_path("modeling_dataset_timeseries__mixed_context.png"))

    corr_cols = ["synthetic_procurement_need", "expected_requirement_tons", "current_inventory_tons", "lead_time_days", "demand_index", "supply_index"]
    corr_cols = [column for column in corr_cols if column in df.columns]
    corr = df[corr_cols].corr(numeric_only=True)
    fig6, ax6 = plt.subplots(figsize=(6, 5))
    heat = ax6.imshow(corr.values, cmap="Oranges")
    ax6.set_xticks(range(len(corr.columns)))
    ax6.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax6.set_yticks(range(len(corr.columns)))
    ax6.set_yticklabels(corr.columns)
    ax6.set_title("Modeling Dataset Correlation")
    fig6.colorbar(heat, ax=ax6)
    correlation_rel = _save_plot(fig6, _figure_path("modeling_dataset_correlation__mixed_context.png"))

    return {
        **_execution_metadata(),
        "notebook": "05_modeling_dataset_eda.ipynb",
        "outputs": [summary_rel, quality_rel, target_rel, trigger_rel, target_dist_rel, target_profile_rel, trigger_balance_rel, trigger_profile_rel, timeseries_rel, correlation_rel],
        "findings": [
            "The official modeling dataset is weekly and mixed/derived from contextual proxies plus a synthetic plant layer.",
            "Trigger balance varies by destination_profile and is evaluated chronologically.",
        ],
        "limitations": [
            "The dataset is not a plant historical purchase ledger; it remains a defended mixed-context training environment.",
        ],
    }


def run_split_validation_and_leakage_audit(scope: str = SCOPE) -> dict[str, Any]:
    split_dir = repo_root() / "data/splits/baseline/default__mixed_context"
    split_metadata = read_json(split_dir / "split_metadata.json")
    frames = {
        "train": pd.read_csv(split_dir / "train.csv"),
        "validation": pd.read_csv(split_dir / "validation.csv"),
        "test": pd.read_csv(split_dir / "test.csv"),
    }
    for frame in frames.values():
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    summary_rows = []
    for name, frame in frames.items():
        summary_rows.append(
            {
                "split": name,
                "date_start": str(frame["date"].min().date()),
                "date_end": str(frame["date"].max().date()),
                "rows": int(len(frame)),
                "target_mean": float(frame["synthetic_procurement_need"].mean()),
                "trigger_rate": float(frame["purchase_trigger_label"].mean()),
            }
        )
    checks = [
        {"check": "max(train.date) < min(validation.date)", "status": bool(frames["train"]["date"].max() < frames["validation"]["date"].min())},
        {"check": "max(validation.date) < min(test.date)", "status": bool(frames["validation"]["date"].max() < frames["test"]["date"].min())},
        {"check": "no duplicated rows across splits", "status": bool(pd.concat(frames.values()).duplicated().sum() == 0)},
    ]
    summary_df = pd.DataFrame(summary_rows)
    checks_df = pd.DataFrame(checks)

    summary_rel = _save_table(summary_df, _table_path("split_summary__mixed_context.csv"))
    checks_rel = _save_table(checks_df, _table_path("split_validation_checks__mixed_context.csv"))

    fig1, ax1 = plt.subplots(figsize=(10, 3))
    colors = {"train": "#52796f", "validation": "#e9c46a", "test": "#e76f51"}
    for index, row in summary_df.iterrows():
        ax1.hlines(index, pd.to_datetime(row["date_start"]), pd.to_datetime(row["date_end"]), linewidth=8, color=colors[row["split"]])
    ax1.set_yticks(range(len(summary_df)))
    ax1.set_yticklabels(summary_df["split"])
    ax1.set_title("Chronological Split Timeline")
    timeline_rel = _save_plot(fig1, _figure_path("split_timeline__mixed_context.png"))

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    summary_df.set_index("split")["target_mean"].plot(kind="bar", ax=ax2, color="#4d6cfa")
    ax2.set_title("Target Mean by Split")
    target_rel = _save_plot(fig2, _figure_path("target_by_split__mixed_context.png"))

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    summary_df.set_index("split")["trigger_rate"].plot(kind="bar", ax=ax3, color="#8d5a97")
    ax3.set_title("Trigger Rate by Split")
    trigger_rel = _save_plot(fig3, _figure_path("trigger_by_split__mixed_context.png"))

    profile_coverage = []
    for name, frame in frames.items():
        counts = frame["destination_profile"].value_counts(normalize=True)
        for profile, value in counts.items():
            profile_coverage.append({"split": name, "destination_profile": profile, "coverage": float(value)})
    coverage_df = pd.DataFrame(profile_coverage)
    fig4, ax4 = plt.subplots(figsize=(9, 4))
    for split_name, group in coverage_df.groupby("split"):
        ax4.bar(group["destination_profile"] + f" ({split_name})", group["coverage"], label=split_name)
    ax4.set_title("Destination Profile Coverage by Split")
    ax4.set_ylabel("coverage")
    ax4.tick_params(axis="x", rotation=45)
    coverage_rel = _save_plot(fig4, _figure_path("profile_coverage_by_split__mixed_context.png"))

    return {
        **_execution_metadata(),
        "notebook": "06_split_validation_and_leakage_audit.ipynb",
        "outputs": [summary_rel, checks_rel, timeline_rel, target_rel, trigger_rel, coverage_rel],
        "findings": [
            "Splits are chronological and test is held out from model selection.",
            "Validation is the explicit selection surface for the defended mixed_context route.",
        ],
        "limitations": [
            "The split audit is deterministic and tied to the current modeling dataset hash.",
        ],
    }


def run_training_and_policy_results_eda(scope: str = SCOPE) -> dict[str, Any]:
    summary_dir = repo_root() / "models/metrics/summary"
    predictions_path = repo_root() / "data/predictions/predictions_latest__mixed_context.csv"
    baseline = read_json(summary_dir / "baseline_comparison_latest__mixed_context.json")
    neuro = read_json(summary_dir / "neuroevolution_comparison_latest__mixed_context.json")
    trigger = read_json(summary_dir / "trigger_metrics_latest__mixed_context.json")
    quantity = read_json(summary_dir / "quantity_optimizer_latest__mixed_context.json")
    policy = read_json(summary_dir / "policy_simulation_latest__mixed_context.json")
    metrics_summary = read_json(summary_dir / "metrics_summary__mixed_context.json")
    predictions = pd.read_csv(predictions_path)

    training_metrics_df = pd.DataFrame(
        [
            {"section": "upstream_baseline", "metric": "validation_rmse", "value": baseline.get("best_baseline_run", {}).get("validation_rmse")},
            {"section": "upstream_baseline", "metric": "test_rmse", "value": baseline.get("best_baseline_run", {}).get("test_rmse")},
            {"section": "upstream_neuroevolution", "metric": "validation_rmse", "value": neuro.get("best_neuroevolution_run", {}).get("validation_rmse")},
            {"section": "upstream_neuroevolution", "metric": "test_rmse", "value": neuro.get("best_neuroevolution_run", {}).get("test_rmse")},
            {"section": "trigger", "metric": "accuracy", "value": trigger.get("test", {}).get("accuracy")},
            {"section": "trigger", "metric": "false_negative_rate", "value": trigger.get("test", {}).get("false_negative_rate")},
            {"section": "quantity_optimizer", "metric": "test_rmse", "value": quantity.get("test", {}).get("rmse")},
        ]
    )
    policy_metrics_df = pd.DataFrame(
        [
            {"metric": "aggregate_excess_reduction_pct", "value": policy.get("aggregate_excess_reduction_pct")},
            {"metric": "aggregate_stockout_change_pct", "value": policy.get("aggregate_stockout_change_pct")},
            {"metric": "stockout_guardrail_pass", "value": policy.get("stockout_guardrail_pass")},
        ]
    )
    training_rel = _save_table(training_metrics_df, _table_path("training_metrics_summary__mixed_context.csv"))
    policy_rel = _save_table(policy_metrics_df, _table_path("policy_metrics_summary__mixed_context.csv"))

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    if "synthetic_procurement_need" in predictions.columns:
        ax1.scatter(predictions["synthetic_procurement_need"], predictions["synthetic_procurement_need_pred"], alpha=0.5, color="#355070")
        max_value = float(max(predictions["synthetic_procurement_need"].max(), predictions["synthetic_procurement_need_pred"].max()))
        ax1.plot([0, max_value], [0, max_value], linestyle="--", color="#999999")
    ax1.set_title("Upstream Prediction vs Actual")
    ax1.set_xlabel("actual synthetic_procurement_need")
    ax1.set_ylabel("predicted synthetic_procurement_need")
    upstream_rel = _save_plot(fig1, _figure_path("upstream_prediction_vs_actual__mixed_context.png"))

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    predictions["purchase_trigger_proba"].plot(kind="hist", bins=20, ax=ax2, color="#6d597a")
    ax2.set_title("Purchase Trigger Probability Distribution")
    trigger_proba_rel = _save_plot(fig2, _figure_path("trigger_probability_distribution__mixed_context.png"))

    fig3, ax3 = plt.subplots(figsize=(4, 4))
    cm = trigger.get("test", {}).get("confusion_matrix", {})
    matrix = np.array(
        [
            [cm.get("true_negative", 0), cm.get("false_positive", 0)],
            [cm.get("false_negative", 0), cm.get("true_positive", 0)],
        ]
    )
    heat = ax3.imshow(matrix, cmap="Reds")
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["pred_0", "pred_1"])
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["actual_0", "actual_1"])
    ax3.set_title("Trigger Confusion Matrix")
    fig3.colorbar(heat, ax=ax3)
    cm_rel = _save_plot(fig3, _figure_path("trigger_confusion_matrix__mixed_context.png"))

    fig4, ax4 = plt.subplots(figsize=(9, 4))
    ax4.scatter(predictions["baseline_order_quantity_tons"], predictions["order_quantity_tons"], alpha=0.5, color="#588157")
    ax4.set_title("Order Quantity vs Baseline")
    ax4.set_xlabel("baseline_order_quantity_tons")
    ax4.set_ylabel("order_quantity_tons")
    order_rel = _save_plot(fig4, _figure_path("order_quantity_vs_baseline__mixed_context.png"))

    fig5, ax5 = plt.subplots(figsize=(6, 4))
    pd.Series(
        {"baseline_excess_tons": policy.get("baseline_excess_tons", 0.0), "policy_excess_tons": policy.get("policy_excess_tons", 0.0)}
    ).plot(kind="bar", ax=ax5, color=["#bc4749", "#386641"])
    ax5.set_title("Excess by Policy")
    excess_rel = _save_plot(fig5, _figure_path("excess_by_policy__mixed_context.png"))

    fig6, ax6 = plt.subplots(figsize=(6, 4))
    pd.Series(
        {"baseline_stockout_tons": policy.get("baseline_stockout_tons", 0.0), "policy_stockout_tons": policy.get("stockout_tons", 0.0)}
    ).plot(kind="bar", ax=ax6, color=["#4d908e", "#277da1"])
    ax6.set_title("Stockout by Policy")
    stockout_rel = _save_plot(fig6, _figure_path("stockout_by_policy__mixed_context.png"))

    profile_excess = predictions.groupby("destination_profile").apply(
        lambda frame: max(frame["baseline_order_quantity_tons"].sum() - frame["order_quantity_tons"].sum(), 0.0)
    )
    fig7, ax7 = plt.subplots(figsize=(8, 4))
    profile_excess.plot(kind="bar", ax=ax7, color="#f4a261")
    ax7.set_title("Excess Reduction by Scenario/Profile")
    ax7.set_ylabel("tons")
    excess_profile_rel = _save_plot(fig7, _figure_path("excess_reduction_by_scenario__mixed_context.png"))

    fig8, ax8 = plt.subplots(figsize=(10, 4))
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    predictions.set_index("date")[["baseline_order_quantity_tons", "order_quantity_tons"]].plot(ax=ax8)
    ax8.set_title("Policy Time Series")
    ax8.set_ylabel("tons")
    timeseries_rel = _save_plot(fig8, _figure_path("policy_timeseries__mixed_context.png"))

    return {
        **_execution_metadata(),
        "notebook": "07_training_and_policy_results_eda.ipynb",
        "outputs": [training_rel, policy_rel, upstream_rel, trigger_proba_rel, cm_rel, order_rel, excess_rel, stockout_rel, excess_profile_rel, timeseries_rel],
        "findings": [
            "The KPI is evaluated against the operational baseline and framed as functional simulation evidence.",
            "order_quantity_tons is treated as a calculated policy output rather than an observed purchase record.",
        ],
        "limitations": [
            "This is not presented as final industrial validation over complete plant history.",
        ],
    }


NOTEBOOK_FUNCTIONS: dict[str, Callable[[str], dict[str, Any]]] = {
    "00_data_sources_audit.ipynb": run_data_sources_audit,
    "01_raw_data_profile.ipynb": run_raw_data_profile,
    "02_external_context_eda.ipynb": run_external_context_eda,
    "03_synthetic_plant_layer_eda.ipynb": run_synthetic_plant_layer_eda,
    "04_feature_engineering_audit.ipynb": run_feature_engineering_audit,
    "05_modeling_dataset_eda.ipynb": run_modeling_dataset_eda,
    "06_split_validation_and_leakage_audit.ipynb": run_split_validation_and_leakage_audit,
    "07_training_and_policy_results_eda.ipynb": run_training_and_policy_results_eda,
}


def build_eda_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    manifests = _source_manifests()
    modeling_df = pd.read_csv(repo_root() / "data/processed/baseline/feature_engineering_modeling__mixed_context.csv")
    active_count = sum(1 for item in manifests if item["evidence_status"] == "active")
    traced_sources = [item["source_id"] for item in manifests if item["evidence_status"] == "traced"]
    raw_files_available = sum(len(item.get("raw_files", [])) for item in manifests)
    excluded_by_leakage = [
        "order_quantity_tons",
        "quantity_optimizer_recommendation_tons",
        "purchase_trigger_flag",
        "purchase_trigger_proba",
    ]
    summary = {
        **_execution_metadata(),
        "active_sources_count": active_count,
        "traced_sources": traced_sources,
        "raw_files_available": raw_files_available,
        "temporal_range": {
            "date_min": str(pd.to_datetime(modeling_df["date"], errors="coerce").min().date()),
            "date_max": str(pd.to_datetime(modeling_df["date"], errors="coerce").max().date()),
        },
        "modeling_dataset_rows": int(len(modeling_df)),
        "modeling_dataset_columns": int(len(modeling_df.columns)),
        "variable_groups": {
            "proxy": [column for column in modeling_df.columns if column.startswith(("demand_", "supply_", "purchase_price_"))],
            "synthetic": [column for column in modeling_df.columns if column.startswith("synthetic_")],
            "calculated": [column for column in modeling_df.columns if column.endswith(("_lag_1", "_lag_2", "_lag_4", "_lag_8")) or "_roll_mean_" in column],
        },
        "excluded_by_leakage": excluded_by_leakage,
        "splits": read_json(repo_root() / "data/splits/baseline/default__mixed_context/split_metadata.json").get("splits", {}),
        "findings": [finding for result in results for finding in result.get("findings", [])],
        "limitations": [limitation for result in results for limitation in result.get("limitations", [])],
        "figure_paths": sorted(_artifact_rel(path) for path in FIGURES_DIR.glob("*.png")),
        "table_paths": sorted(_artifact_rel(path) for path in TABLES_DIR.glob("*.csv")),
        "notebook_html_paths": sorted(_artifact_rel(path) for path in NOTEBOOKS_DIR.glob("*.html")),
    }

    write_json(EDA_DIR / "eda_summary__mixed_context.json", summary)
    markdown_lines = [
        "# CU28 mixed_context EDA summary",
        "",
        f"- active_sources_count: {summary['active_sources_count']}",
        f"- traced_sources: {', '.join(summary['traced_sources']) or 'none'}",
        f"- raw_files_available: {summary['raw_files_available']}",
        f"- temporal_range: {summary['temporal_range']['date_min']} -> {summary['temporal_range']['date_max']}",
        f"- modeling_dataset_rows: {summary['modeling_dataset_rows']}",
        f"- modeling_dataset_columns: {summary['modeling_dataset_columns']}",
        "",
        "## Findings",
        *[f"- {item}" for item in summary["findings"]],
        "",
        "## Limitations",
        *[f"- {item}" for item in summary["limitations"]],
        "",
        "## Figures",
        *[f"- {item}" for item in summary["figure_paths"]],
        "",
        "## Tables",
        *[f"- {item}" for item in summary["table_paths"]],
        "",
        "## Notebook HTML",
        *[f"- {item}" for item in summary["notebook_html_paths"]],
    ]
    (EDA_DIR / "eda_summary__mixed_context.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return summary
