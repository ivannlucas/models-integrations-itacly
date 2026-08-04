from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [dedent(text).strip() + "\n"],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [dedent(text).strip() + "\n"],
    }


def notebook_payload(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.x",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_IMPORTS = """
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.reproducibility.notebook_support import (
    detect_temporal_columns,
    ensure_eda_dirs,
    execution_metadata,
    first_valid_temporal_range,
    load_source_manifests,
    load_tabular_file,
    parse_markdown_table,
    print_frame,
    print_series,
    project_root,
    read_json,
    relative_to_root,
    save_figure,
    save_table,
    sha256_file,
)
"""


def base_cells(title: str, objective: str, inputs: list[str], outputs: list[str], notebook_name: str, extra_imports: str = "") -> list[dict]:
    inputs_text = "\n".join(f"- `{item}`" for item in inputs)
    outputs_text = "\n".join(f"- `{item}`" for item in outputs)
    return [
        md(
            f"""
            # {title}

            Notebook narrativo de auditoria para el scope `mixed_context`.
            """
        ),
        md(
            f"""
            ## Objetivo

            {objective}
            """
        ),
        md(
            """
            ## Alcance

            Este analisis describe la ruta oficial reproducible `mixed_context`. Las senales externas se tratan como contexto/proxy. Las variables internas de planta siguen siendo sinteticas salvo carga posterior de cliente.
            """
        ),
        md(
            f"""
            ## Inputs

            {inputs_text}
            """
        ),
        md(
            f"""
            ## Outputs esperados

            {outputs_text}
            """
        ),
        md(
            """
            ## Limitaciones

            Este notebook documenta evidencia reproducible del pipeline oficial, pero no sustituye la revision de codigo, la auditoria de datos de origen ni una certificacion operacional de planta.
            """
        ),
        code(COMMON_IMPORTS + "\n" + dedent(extra_imports).strip()),
        code(
            f"""
            NOTEBOOK_NAME = "{notebook_name}"
            PROJECT_ROOT = project_root()
            SCOPE = globals().get("scope", "mixed_context")
            REPORT_DIRS = ensure_eda_dirs()
            META = execution_metadata(SCOPE)
            FIGURES = []
            TABLES = []
            print(json.dumps(META, indent=2))
            """
        ),
        md(
            """
            ## Carga de datos

            Las siguientes celdas cargan los artefactos de entrada y muestran verificaciones intermedias antes de producir tablas y graficas.
            """
        ),
    ]


def nb00() -> list[dict]:
    cells = base_cells(
        "CU28 mixed_context - Data Sources Audit",
        "Revisar de forma explicita las fuentes activas, trazadas y candidatas, y verificar su trazabilidad local mediante manifests, URLs oficiales, licencias y artefactos derivados.",
        [
            "docs/data_sources_registry.md",
            "data/raw/external/*/source_manifest.json",
            "data/raw/external/raw_manifest__mixed_context.json",
            "data_blob_manifest.json",
        ],
        [
            "reports/tables/eda/data_sources_audit__mixed_context.csv",
            "reports/tables/eda/data_sources_inconsistencies__mixed_context.csv",
            "reports/figures/eda/data_sources_status_counts__mixed_context.png",
            "reports/figures/eda/data_sources_type_counts__mixed_context.png",
            "reports/figures/eda/data_sources_raw_file_counts__mixed_context.png",
        ],
        "00_data_sources_audit.ipynb",
    )
    cells.extend(
        [
            code(
                """
                registry_path = PROJECT_ROOT / "docs" / "data_sources_registry.md"
                registry_table = parse_markdown_table(registry_path)
                for column in registry_table.columns:
                    registry_table[column] = registry_table[column].astype(str).str.replace("`", "", regex=False).str.strip()
                print_frame("Registry markdown table", registry_table, rows=10)
                display(registry_table.head(10))
                """
            ),
            code(
                """
                raw_root = PROJECT_ROOT / "data" / "raw" / "external"
                source_manifests = load_source_manifests(raw_root)
                manifest_df = pd.DataFrame(
                    [
                        {
                            "source_id": item["source_id"],
                            "organization": item["organization"],
                            "source_type": item["source_type"],
                            "evidence_status": item["evidence_status"],
                            "official_url": item["official_url"],
                            "download_url_or_endpoint": item["download_url_or_endpoint"],
                            "license_or_terms_url": item["license_or_terms_url"],
                            "access_date": item["access_date"],
                            "retrieval_method": item["retrieval_method"],
                            "role": item["role"],
                            "limitations": item["limitations"],
                            "raw_files_count": len(item.get("raw_files", [])),
                            "derived_artifacts_count": len(item.get("derived_artifacts", [])),
                        }
                        for item in source_manifests
                    ]
                )
                print_frame("Manifest summary", manifest_df, rows=10)
                display(manifest_df)
                """
            ),
            md(
                """
                ## Inspeccion inicial

                Se cruza la tabla de documentacion con los manifests reales para comprobar si ambas vistas coinciden en estado, URLs y rutas locales.
                """
            ),
            code(
                """
                source_overview = registry_table.merge(
                    manifest_df,
                    left_on="source_id",
                    right_on="source_id",
                    how="left",
                    suffixes=("_registry", "_manifest"),
                )
                source_overview["status_match"] = source_overview["status"].eq(source_overview["evidence_status"])
                print_frame(
                    "Overview after joining registry and manifests",
                    source_overview[["source_id", "status", "evidence_status", "source_type", "raw_files_count", "derived_artifacts_count", "status_match"]],
                    rows=10,
                )
                display(source_overview[["source_id", "status", "evidence_status", "source_type", "status_match"]])
                """
            ),
            code(
                """
                url_checks = source_overview[[
                    "source_id",
                    "official_url_registry",
                    "download_url_or_endpoint_registry",
                    "license_or_terms_url_registry",
                ]].copy()
                url_checks["official_url_ok"] = url_checks["official_url_registry"].astype(str).str.startswith("http")
                url_checks["download_url_ok"] = url_checks["download_url_or_endpoint_registry"].astype(str).str.startswith("http")
                url_checks["license_url_ok"] = url_checks["license_or_terms_url_registry"].astype(str).str.startswith("http")
                print_frame("URL checks", url_checks, rows=10)
                display(url_checks)
                """
            ),
            code(
                """
                source_to_manifest = {item["source_id"]: item for item in source_manifests}
                existence_rows = []
                for source_id, manifest in source_to_manifest.items():
                    raw_paths = [entry["path"] for entry in manifest.get("raw_files", [])]
                    derived_paths = manifest.get("derived_artifacts", [])
                    existence_rows.append(
                        {
                            "source_id": source_id,
                            "raw_exists": all((PROJECT_ROOT / raw_path).exists() for raw_path in raw_paths) if raw_paths else False,
                            "processed_exists": all((PROJECT_ROOT / derived_path).exists() for derived_path in derived_paths) if derived_paths else False,
                            "hash_available": all(bool(entry.get("sha256")) for entry in manifest.get("raw_files", [])),
                            "raw_path_count": len(raw_paths),
                            "derived_path_count": len(derived_paths),
                        }
                    )
                existence_df = pd.DataFrame(existence_rows)
                print_frame("Existence checks", existence_df, rows=10)
                display(existence_df)
                """
            ),
            md(
                """
                ## Interpretacion intermedia

                Las fuentes `active` deben tener URL oficial, licencia, raw local y artefactos procesados. La fuente `traced` puede quedar como evidencia de contexto siempre que no se defienda como feed semanal activo.
                """
            ),
            code(
                """
                inconsistencies_df = source_overview[[
                    "source_id",
                    "status",
                    "evidence_status",
                    "status_match",
                ]].merge(existence_df, on="source_id", how="left").merge(
                    url_checks[["source_id", "official_url_ok", "download_url_ok", "license_url_ok"]],
                    on="source_id",
                    how="left",
                )
                inconsistencies_df["has_inconsistency"] = ~(
                    inconsistencies_df["status_match"]
                    & inconsistencies_df["official_url_ok"]
                    & inconsistencies_df["license_url_ok"]
                    & (
                        (inconsistencies_df["status"] != "active")
                        | (inconsistencies_df["raw_exists"] & inconsistencies_df["processed_exists"])
                    )
                )
                print_frame("Inconsistency table", inconsistencies_df, rows=10)
                display(inconsistencies_df)
                """
            ),
            code(
                """
                status_counts = source_overview["evidence_status"].fillna("missing_manifest").value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(7, 4))
                status_counts.plot(kind="bar", color=["#2f6f4f", "#c98f2b", "#7f8c8d"], ax=ax)
                ax.set_title("Sources by evidence status")
                ax.set_xlabel("evidence_status")
                ax.set_ylabel("source_count")
                FIGURES.append(save_figure(fig, "data_sources_status_counts__mixed_context.png"))
                plt.close(fig)
                print(status_counts.to_string())
                """
            ),
            md(
                """
                ### Interpretacion de la figura

                La barra de estados debe mostrar que solo INE_CPI y MAPA_SLAUGHTER_MAPA quedan activos en la ruta oficial. MAPA_PRICES_OM se conserva como trazado y no como serie semanal defendible.
                """
            ),
            code(
                """
                type_counts = manifest_df["source_type"].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(8, 4))
                type_counts.plot(kind="bar", color="#3c7dc4", ax=ax)
                ax.set_title("Sources by source_type")
                ax.set_xlabel("source_type")
                ax.set_ylabel("source_count")
                FIGURES.append(save_figure(fig, "data_sources_type_counts__mixed_context.png"))
                plt.close(fig)
                print(type_counts.to_string())
                """
            ),
            md(
                """
                ### Interpretacion de la figura

                La mezcla de CSV oficial y bundle de hojas de calculo confirma que el pipeline parte de snapshots heterogeneos. Por eso el manifest por fuente es necesario para una auditoria reproducible.
                """
            ),
            code(
                """
                raw_file_counts = manifest_df.set_index("source_id")["raw_files_count"].sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(7, 4))
                raw_file_counts.plot(kind="bar", color="#7a4b94", ax=ax)
                ax.set_title("Raw files restored per source")
                ax.set_xlabel("source_id")
                ax.set_ylabel("raw_files_count")
                FIGURES.append(save_figure(fig, "data_sources_raw_file_counts__mixed_context.png"))
                plt.close(fig)
                print(raw_file_counts.to_string())
                """
            ),
            md(
                """
                ## Hallazgos parciales

                - `INE_CPI` y `MAPA_SLAUGHTER_MAPA` aparecen como fuentes activas con trazabilidad local.
                - `MAPA_PRICES_OM` queda trazada como referencia de respaldo y no como feed semanal activo.
                - Las fuentes candidatas no deben entrar en el blob oficial ni en el pipeline mixto defendible.
                """
            ),
            code(
                """
                for source_id in ["INE_CPI", "MAPA_SLAUGHTER_MAPA", "MAPA_PRICES_OM"]:
                    subset = source_overview[source_overview["source_id"] == source_id]
                    print(f"Source focus: {source_id}")
                    if subset.empty:
                        print("  source not found")
                        continue
                    row = subset.iloc[0]
                    print(f"  status documented: {row['status']}")
                    print(f"  status manifest: {row['evidence_status']}")
                    print(f"  official_url: {row['official_url_registry']}")
                    print(f"  limitations: {row['limitations_registry']}")
                """
            ),
            code(
                """
                TABLES.append(save_table(source_overview, "data_sources_audit__mixed_context.csv"))
                TABLES.append(save_table(inconsistencies_df, "data_sources_inconsistencies__mixed_context.csv"))
                RESULT = {
                    "notebook": NOTEBOOK_NAME,
                    "tables": TABLES,
                    "figures": FIGURES,
                    "findings": [
                        "Active sources keep official URLs, license URLs and local raw snapshots.",
                        "MAPA_PRICES_OM remains traced only and is not treated as active weekly evidence.",
                        "Candidate sources remain outside the defended mixed_context route.",
                    ],
                    "limitations": [
                        "The markdown registry is a curated summary and must stay aligned with source manifests.",
                    ],
                }
                print(json.dumps(RESULT, indent=2))
                """
            ),
            md(
                """
                ## Limitaciones

                Este notebook verifica coherencia documental y presencia local de artefactos. No evalua por si solo la calidad estadistica de las series ni sustituye la inspeccion de raw y procesados.
                """
            ),
            md(
                """
                ## Concluson final

                La capa de inventario documental queda defendible cuando el estado de cada fuente coincide entre el registro markdown y el manifest JSON, y cuando los raw y derivados exigidos existen localmente.
                """
            ),
        ]
    )
    return cells


def nb01() -> list[dict]:
    cells = base_cells(
        "CU28 mixed_context - Raw Data Profile",
        "Perfilar los snapshots raw oficiales antes del ETL para mostrar su estructura, cobertura y limitaciones como senales externas/proxy.",
        [
            "data/raw/external/INE_CPI/",
            "data/raw/external/MAPA_SLAUGHTER_MAPA/",
            "data/raw/external/MAPA_PRICES_OM/",
            "data/raw/external/*/source_manifest.json",
        ],
        [
            "reports/tables/eda/raw_file_inventory__mixed_context.csv",
            "reports/tables/eda/raw_data_quality__mixed_context.csv",
            "reports/tables/eda/raw_temporal_ranges__mixed_context.csv",
            "reports/figures/eda/raw_file_sizes__mixed_context.png",
            "reports/figures/eda/raw_record_counts__mixed_context.png",
            "reports/figures/eda/raw_temporal_coverage__mixed_context.png",
            "reports/figures/eda/raw_missing_values__mixed_context.png",
        ],
        "01_raw_data_profile.ipynb",
    )
    cells.extend(
        [
            code(
                """
                raw_root = PROJECT_ROOT / "data" / "raw" / "external"
                source_manifests = load_source_manifests(raw_root)
                if not source_manifests:
                    raise FileNotFoundError(f"No source manifests found under {raw_root}")
                raw_inventory_rows = []
                for manifest in source_manifests:
                    for raw_file in manifest.get("raw_files", []):
                        file_path = PROJECT_ROOT / raw_file["path"]
                        raw_inventory_rows.append(
                            {
                                "source_id": manifest["source_id"],
                                "path": raw_file["path"],
                                "size_bytes": file_path.stat().st_size if file_path.exists() else raw_file.get("size_bytes"),
                                "sha256": sha256_file(file_path) if file_path.exists() else raw_file.get("sha256"),
                                "access_date": manifest["access_date"],
                                "retrieval_method": manifest["retrieval_method"],
                            }
                        )
                if not raw_inventory_rows:
                    raise FileNotFoundError(f"No raw files declared in source manifests under {raw_root}")
                raw_inventory = pd.DataFrame(raw_inventory_rows).sort_values(["source_id", "path"]).reset_index(drop=True)
                print_frame("Raw inventory", raw_inventory, rows=20)
                display(raw_inventory)
                """
            ),
            code(
                """
                profiles = []
                previews = {}
                tail_previews = {}
                for row in raw_inventory.itertuples(index=False):
                    file_path = PROJECT_ROOT / row.path
                    frame = load_tabular_file(file_path)
                    temporal = first_valid_temporal_range(frame)
                    previews[row.path] = frame.head(3)
                    tail_previews[row.path] = frame.tail(3)
                    profiles.append(
                        {
                            "source_id": row.source_id,
                            "path": row.path,
                            "row_count": int(len(frame)),
                            "column_count": int(len(frame.columns)),
                            "columns": ", ".join(str(column) for column in frame.columns[:12]),
                            "dtypes": ", ".join(f"{column}:{dtype}" for column, dtype in frame.dtypes.astype(str).items()),
                            "missing_pct_mean": float(frame.isna().mean().mean()),
                            "duplicate_rows": int(frame.duplicated().sum()),
                            "temporal_column": temporal["column"],
                            "date_min": temporal["date_min"],
                            "date_max": temporal["date_max"],
                        }
                    )
                raw_quality = pd.DataFrame(profiles).sort_values(["source_id", "path"]).reset_index(drop=True)
                print_frame("Raw profile", raw_quality, rows=20)
                display(raw_quality[["source_id", "path", "row_count", "column_count", "temporal_column", "date_min", "date_max"]])
                """
            ),
            md(
                """
                ## Inspeccion inicial

                En esta fase se comprueban shape, columnas, tipos, nulos y duplicados. La lectura usa el primer sheet reproducible de cada snapshot tabular.
                """
            ),
            code(
                """
                shape_summary = raw_quality[["source_id", "path", "row_count", "column_count"]].copy()
                print_frame("Shape by raw file", shape_summary, rows=20)
                display(shape_summary)
                """
            ),
            code(
                """
                column_and_dtype_summary = raw_quality[["source_id", "path", "columns", "dtypes"]].copy()
                print_frame("Columns and dtypes", column_and_dtype_summary, rows=20)
                display(column_and_dtype_summary.head(10))
                """
            ),
            code(
                """
                quality_summary = raw_quality[["source_id", "path", "missing_pct_mean", "duplicate_rows"]].copy()
                quality_summary["missing_pct_mean"] = quality_summary["missing_pct_mean"].round(4)
                print_frame("Missing values and duplicates", quality_summary, rows=20)
                display(quality_summary)
                """
            ),
            code(
                """
                for path, preview in previews.items():
                    print(f"First rows for {path}")
                    print(preview.to_string(index=False))
                    print("")
                """
            ),
            code(
                """
                for path, preview in tail_previews.items():
                    print(f"Last rows for {path}")
                    print(preview.to_string(index=False))
                    print("")
                """
            ),
            code(
                """
                temporal_ranges = raw_quality[["source_id", "path", "temporal_column", "date_min", "date_max"]].copy()
                print_frame("Temporal range by raw file", temporal_ranges, rows=20)
                display(temporal_ranges)
                """
            ),
            md(
                """
                ## Interpretacion intermedia

                Los snapshots raw son insumos contextuales externos. No representan inventario observado, compras observadas ni produccion observada de una planta concreta.
                """
            ),
            code(
                """
                size_by_source = raw_inventory.groupby("source_id", as_index=False)["size_bytes"].sum().sort_values("size_bytes", ascending=False)
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(size_by_source["source_id"], size_by_source["size_bytes"], color="#355c7d")
                ax.set_title("Raw size by source")
                ax.set_ylabel("bytes")
                FIGURES.append(save_figure(fig, "raw_file_sizes__mixed_context.png"))
                plt.close(fig)
                print_frame("Size by source", size_by_source)
                """
            ),
            md(
                """
                ### Interpretacion de la figura

                El tamano por fuente ayuda a entender el peso del blob y confirma que el repositorio no necesita contener datos masivos si el empaquetado externo esta bien trazado.
                """
            ),
            code(
                """
                records_by_source = raw_quality.groupby("source_id", as_index=False)["row_count"].sum().sort_values("row_count", ascending=False)
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(records_by_source["source_id"], records_by_source["row_count"], color="#c06c84")
                ax.set_title("Raw record count by source")
                ax.set_ylabel("rows")
                FIGURES.append(save_figure(fig, "raw_record_counts__mixed_context.png"))
                plt.close(fig)
                print_frame("Records by source", records_by_source)
                """
            ),
            md(
                """
                ### Interpretacion de la figura

                El conteo de filas ilustra que las fuentes aportan granularidades distintas y que el ETL posterior debe armonizar frecuencia y estructura.
                """
            ),
            code(
                """
                temporal_plot_df = temporal_ranges.dropna(subset=["date_min", "date_max"]).copy()
                temporal_plot_df["date_min"] = pd.to_datetime(temporal_plot_df["date_min"])
                temporal_plot_df["date_max"] = pd.to_datetime(temporal_plot_df["date_max"])
                fig, ax = plt.subplots(figsize=(10, 4))
                for idx, row in temporal_plot_df.reset_index(drop=True).iterrows():
                    ax.hlines(idx, row["date_min"], row["date_max"], linewidth=6, color="#6c5b7b")
                ax.set_yticks(range(len(temporal_plot_df)))
                ax.set_yticklabels(temporal_plot_df["source_id"])
                ax.set_title("Temporal coverage by raw file")
                ax.set_xlabel("date")
                FIGURES.append(save_figure(fig, "raw_temporal_coverage__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                missing_by_file = raw_quality[["path", "missing_pct_mean"]].sort_values("missing_pct_mean", ascending=False)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(missing_by_file["path"], missing_by_file["missing_pct_mean"], color="#f67280")
                ax.set_title("Average missing rate by raw file")
                ax.set_ylabel("missing_pct_mean")
                ax.tick_params(axis="x", rotation=75)
                FIGURES.append(save_figure(fig, "raw_missing_values__mixed_context.png"))
                plt.close(fig)
                print_frame("Missing rate by raw file", missing_by_file, rows=20)
                """
            ),
            md(
                """
                ## Hallazgos parciales

                - El inventario raw queda identificado por ruta, hash y metodo de obtencion.
                - La cobertura temporal es heterogenea y confirma la necesidad de un ETL que armonice frecuencia.
                - Las hojas tabulares no contienen variables internas observadas de planta.
                """
            ),
            code(
                """
                TABLES.append(save_table(raw_inventory, "raw_file_inventory__mixed_context.csv"))
                TABLES.append(save_table(raw_quality, "raw_data_quality__mixed_context.csv"))
                TABLES.append(save_table(temporal_ranges, "raw_temporal_ranges__mixed_context.csv"))
                RESULT = {
                    "notebook": NOTEBOOK_NAME,
                    "tables": TABLES,
                    "figures": FIGURES,
                    "findings": [
                        "Raw files are external contextual signals and do not contain internal observed plant variables.",
                        "Temporal coverage differs by source and motivates the weekly harmonisation step.",
                        "Each raw file remains traceable through size, hash and access metadata.",
                    ],
                    "limitations": [
                        "Spreadsheet profiling reads the first sheet only, which is the defended audit entrypoint for these snapshots.",
                    ],
                }
                print(json.dumps(RESULT, indent=2))
                """
            ),
            md(
                """
                ## Limitaciones

                El perfil raw no sustituye la inspeccion sectorial de contenido. Su objetivo es dejar constancia reproducible de estructura, completitud y cobertura antes del ETL.
                """
            ),
            md(
                """
                ## Concluson final

                Los raw oficiales quedan caracterizados como snapshots externos/proxy listos para transformacion, sin insinuar que sean historicos internos de aprovisionamiento o produccion.
                """
            ),
        ]
    )
    return cells


def nb02() -> list[dict]:
    cells = base_cells(
        "CU28 mixed_context - External Context EDA",
        "Analizar las series procesadas de contexto externo para entender que variables proxy alimentan el pipeline oficial y cuales son sus limitaciones.",
        [
            "data/processed/external/context/external_long.csv",
            "data/processed/external/context/context_weekly_for_simulation.csv",
            "data/processed/external/context/context_proxy_limitations.json",
        ],
        [
            "reports/tables/eda/external_context_summary__mixed_context.csv",
            "reports/tables/eda/external_context_weekly_snapshot__mixed_context.csv",
            "reports/tables/eda/external_context_limitations__mixed_context.csv",
            "reports/figures/eda/external_context_demand_index__mixed_context.png",
            "reports/figures/eda/external_context_supply_index__mixed_context.png",
            "reports/figures/eda/external_context_gap__mixed_context.png",
            "reports/figures/eda/external_context_price_index__mixed_context.png",
            "reports/figures/eda/external_context_correlation__mixed_context.png",
            "reports/figures/eda/external_context_coverage__mixed_context.png",
        ],
        "02_external_context_eda.ipynb",
    )
    cells.extend(
        [
            code(
                """
                external_long_path = PROJECT_ROOT / "data/processed/external/context/external_long.csv"
                context_weekly_path = PROJECT_ROOT / "data/processed/external/context/context_weekly_for_simulation.csv"
                limitations_path = PROJECT_ROOT / "data/processed/external/context/context_proxy_limitations.json"
                external_long = pd.read_csv(external_long_path)
                external_long["date"] = pd.to_datetime(external_long["date"], errors="coerce")
                print(external_long.shape)
                print(external_long.columns.tolist())
                print_frame("external_long preview", external_long.head(10))
                """
            ),
            code(
                """
                context_weekly = pd.read_csv(context_weekly_path)
                context_weekly["date"] = pd.to_datetime(context_weekly["date"], errors="coerce")
                context_weekly["demand_supply_gap"] = context_weekly["demand_index"] - context_weekly["supply_index"]
                limitations_payload = read_json(limitations_path)
                limitations_df = pd.DataFrame({"limitation": limitations_payload.get("limitations", [])})
                print(context_weekly.shape)
                print(context_weekly.columns.tolist())
                print_frame("context_weekly preview", context_weekly.head(10))
                """
            ),
            md(
                """
                ## Inspeccion inicial

                Primero se comprueba la estructura de ambos datasets procesados y el rango temporal disponible para la simulacion semanal.
                """
            ),
            code(
                """
                external_summary_shape = pd.DataFrame(
                    [{"dataset": "external_long", "rows": len(external_long), "columns": len(external_long.columns)}]
                )
                print_frame("Shape of external_long", external_summary_shape)
                display(external_summary_shape)
                """
            ),
            code(
                """
                weekly_summary_shape = pd.DataFrame(
                    [{"dataset": "context_weekly", "rows": len(context_weekly), "columns": len(context_weekly.columns)}]
                )
                print_frame("Shape of context_weekly", weekly_summary_shape)
                display(weekly_summary_shape)
                """
            ),
            code(
                """
                temporal_range = pd.DataFrame(
                    [
                        {
                            "dataset": "external_long",
                            "date_min": str(external_long["date"].min().date()),
                            "date_max": str(external_long["date"].max().date()),
                        },
                        {
                            "dataset": "context_weekly",
                            "date_min": str(context_weekly["date"].min().date()),
                            "date_max": str(context_weekly["date"].max().date()),
                        },
                    ]
                )
                variable_table = pd.DataFrame({"variable": context_weekly.columns})
                print_frame("Temporal range", temporal_range)
                print_frame("Variables in context_weekly", variable_table, rows=20)
                """
            ),
            code(
                """
                missing_summary = context_weekly.isna().mean().reset_index()
                missing_summary.columns = ["variable", "missing_pct"]
                missing_summary["missing_pct"] = missing_summary["missing_pct"].round(4)
                print_frame("Missing values in context_weekly", missing_summary, rows=20)
                display(missing_summary)
                """
            ),
            md(
                """
                ## Tablas intermedias

                Se resume `external_long` por fuente, dataset y subserie para ver observaciones, cobertura y huecos antes de analizar las series agregadas semanales.
                """
            ),
            code(
                """
                external_context_summary = (
                    external_long.groupby(["source", "dataset", "subseries"], dropna=False)
                    .agg(
                        variable=("unit", "first"),
                        min_date=("date", "min"),
                        max_date=("date", "max"),
                        observations=("value", "size"),
                        missing_rate=("value", lambda s: float(pd.to_numeric(s, errors="coerce").isna().mean())),
                    )
                    .reset_index()
                )
                print_frame("Summary by source, dataset and subseries", external_context_summary, rows=20)
                display(external_context_summary.head(20))
                """
            ),
            code(
                """
                weekly_snapshot = context_weekly[["date", "demand_index", "supply_index", "demand_supply_gap", "purchase_price_index"]].head(12).copy()
                print_frame("Weekly snapshot", weekly_snapshot, rows=12)
                display(weekly_snapshot)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(context_weekly["date"], context_weekly["demand_index"], color="#355c7d")
                ax.set_title("Demand index over time")
                ax.set_ylabel("index")
                FIGURES.append(save_figure(fig, "external_context_demand_index__mixed_context.png"))
                plt.close(fig)
                """
            ),
            md(
                """
                ### Interpretacion de la figura

                `demand_index` captura presion de demanda sectorial agregada. Es una senal proxy de contexto y no una observacion directa de pedidos internos de planta.
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(context_weekly["date"], context_weekly["supply_index"], color="#6c5b7b")
                ax.set_title("Supply index over time")
                ax.set_ylabel("index")
                FIGURES.append(save_figure(fig, "external_context_supply_index__mixed_context.png"))
                plt.close(fig)
                """
            ),
            md(
                """
                ### Interpretacion de la figura

                `supply_index` resume un proxy de oferta sectorial. Su lectura debe hacerse como contexto macro, no como disponibilidad confirmada para una planta concreta.
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(context_weekly["date"], context_weekly["demand_supply_gap"], color="#c06c84")
                ax.axhline(0.0, linestyle="--", color="black", linewidth=1)
                ax.set_title("Demand supply gap over time")
                ax.set_ylabel("gap")
                FIGURES.append(save_figure(fig, "external_context_gap__mixed_context.png"))
                plt.close(fig)
                """
            ),
            md(
                """
                ### Interpretacion de la figura

                La brecha `demand_supply_gap` representa tension relativa entre demanda y oferta. Se usa como contexto de riesgo, no como decision de compra final.
                """
            ),
            code(
                """
                purchase_price_constant = context_weekly["purchase_price_index"].nunique() == 1
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(context_weekly["date"], context_weekly["purchase_price_index"], color="#f67280")
                ax.set_title("Purchase price index")
                ax.set_ylabel("index")
                FIGURES.append(save_figure(fig, "external_context_price_index__mixed_context.png"))
                plt.close(fig)
                print(f"purchase_price_index uses fallback constant: {purchase_price_constant}")
                """
            ),
            code(
                """
                corr = context_weekly[["demand_index", "supply_index", "purchase_price_index", "demand_supply_gap"]].corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(6, 5))
                image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.index)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                ax.set_yticklabels(corr.index)
                ax.set_title("Correlation between external signals")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                FIGURES.append(save_figure(fig, "external_context_correlation__mixed_context.png"))
                plt.close(fig)
                print(corr.to_string())
                """
            ),
            code(
                """
                coverage_df = context_weekly.notna().mean().reset_index()
                coverage_df.columns = ["variable", "coverage_rate"]
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(coverage_df["variable"], coverage_df["coverage_rate"], color="#2a9d8f")
                ax.set_title("Weekly coverage by variable")
                ax.set_ylabel("coverage_rate")
                ax.tick_params(axis="x", rotation=30)
                FIGURES.append(save_figure(fig, "external_context_coverage__mixed_context.png"))
                plt.close(fig)
                print_frame("Coverage table", coverage_df)
                """
            ),
            md(
                """
                ## Limitaciones proxy

                La siguiente tabla se deriva del JSON de limitaciones y debe leerse junto con la evidencia de que `purchase_price_index` puede operar como valor constante de respaldo.
                """
            ),
            code(
                """
                print_frame("Proxy limitations", limitations_df, rows=20)
                display(limitations_df)
                """
            ),
            code(
                """
                TABLES.append(save_table(external_context_summary, "external_context_summary__mixed_context.csv"))
                TABLES.append(save_table(weekly_snapshot, "external_context_weekly_snapshot__mixed_context.csv"))
                TABLES.append(save_table(limitations_df, "external_context_limitations__mixed_context.csv"))
                RESULT = {
                    "notebook": NOTEBOOK_NAME,
                    "tables": TABLES,
                    "figures": FIGURES,
                    "findings": [
                        "External processed signals remain contextual proxies and do not replace internal plant telemetry.",
                        "The weekly dataset is compact and reproducible, with demand, supply and gap signals aligned on a common calendar.",
                        "purchase_price_index must be interpreted as fallback evidence when it remains constant.",
                    ],
                    "limitations": [
                        "Proxy limitations are structural and already documented in the dedicated JSON contract.",
                    ],
                }
                print(json.dumps(RESULT, indent=2))
                """
            ),
            md(
                """
                ## Concluson final

                `external_long` documenta procedencia y cobertura de las series fuente, mientras que `context_weekly_for_simulation.csv` concentra el contexto semanal que alimenta la reconstruccion `mixed_context`.
                """
            ),
        ]
    )
    return cells


def nb03() -> list[dict]:
    cells = base_cells(
        "CU28 mixed_context - Synthetic Plant Layer EDA",
        "Validar de forma plausible la capa operativa sintetica que aproxima inventario, requirement, lead time, yield y waste para la ruta mixed_context.",
        [
            "data/processed/synthetic/plant/synthetic_plant_layer__mixed_context.csv",
            "data/processed/synthetic/plant/synthetic_plant_metadata__mixed_context.json",
            "config/manufacturing_profiles.yaml",
            "docs/simulation_assumptions.md",
            "docs/simulation_data_basis.md",
        ],
        [
            "reports/tables/eda/synthetic_layer_summary__mixed_context.csv",
            "reports/tables/eda/synthetic_layer_by_profile__mixed_context.csv",
            "reports/tables/eda/synthetic_layer_variable_ranges__mixed_context.csv",
            "reports/figures/eda/synthetic_requirement_by_profile__mixed_context.png",
            "reports/figures/eda/synthetic_inventory_distribution__mixed_context.png",
            "reports/figures/eda/synthetic_lead_time_by_profile__mixed_context.png",
            "reports/figures/eda/synthetic_yield_waste_by_profile__mixed_context.png",
            "reports/figures/eda/synthetic_procurement_need_by_profile__mixed_context.png",
            "reports/figures/eda/synthetic_layer_correlation__mixed_context.png",
            "reports/figures/eda/synthetic_requirement_inventory_timeseries__mixed_context.png",
        ],
        "03_synthetic_plant_layer_eda.ipynb",
        extra_imports="import yaml",
    )
    cells.extend(
        [
            code(
                """
                synthetic_path = PROJECT_ROOT / "data/processed/synthetic/plant/synthetic_plant_layer__mixed_context.csv"
                synthetic_meta_path = PROJECT_ROOT / "data/processed/synthetic/plant/synthetic_plant_metadata__mixed_context.json"
                profiles_path = PROJECT_ROOT / "config" / "manufacturing_profiles.yaml"
                synthetic_df = pd.read_csv(synthetic_path)
                synthetic_df["date"] = pd.to_datetime(synthetic_df["date"], errors="coerce")
                synthetic_meta = read_json(synthetic_meta_path)
                print(synthetic_df.shape)
                print_frame("Synthetic layer preview", synthetic_df.head(10))
                """
            ),
            code(
                """
                manufacturing_profiles = yaml.safe_load(profiles_path.read_text(encoding="utf-8"))
                selected_profile_keys = list((manufacturing_profiles or {}).keys())
                print({"profile_count": len(selected_profile_keys), "profiles": selected_profile_keys[:10]})
                print(json.dumps({k: synthetic_meta[k] for k in ["environment_name", "time_granularity", "validated_base_horizon_label"] if k in synthetic_meta}, indent=2))
                """
            ),
            md(
                """
                ## Inspeccion inicial

                Se revisan shape, columnas, tipos y nulos para dejar claro que esta capa es sintetica y que su funcion es dar contexto operativo reproducible al pipeline mixto.
                """
            ),
            code(
                """
                shape_summary = pd.DataFrame([{"rows": len(synthetic_df), "columns": len(synthetic_df.columns)}])
                column_summary = pd.DataFrame({"column": synthetic_df.columns})
                print_frame("Shape summary", shape_summary)
                print_frame("Column list", column_summary, rows=30)
                """
            ),
            code(
                """
                dtype_summary = synthetic_df.dtypes.astype(str).reset_index()
                dtype_summary.columns = ["column", "dtype"]
                print_frame("Dtype summary", dtype_summary, rows=30)
                display(dtype_summary.head(30))
                """
            ),
            code(
                """
                null_summary = synthetic_df.isna().mean().reset_index()
                null_summary.columns = ["column", "missing_pct"]
                null_summary["missing_pct"] = null_summary["missing_pct"].round(4)
                print_frame("Null summary", null_summary.sort_values("missing_pct", ascending=False), rows=30)
                display(null_summary.sort_values("missing_pct", ascending=False).head(30))
                """
            ),
            code(
                """
                variables_of_interest = [
                    "current_inventory_tons",
                    "expected_requirement_tons",
                    "lead_time_days",
                    "safety_coverage_days",
                    "expected_yield_rate",
                    "expected_waste_rate",
                    "synthetic_procurement_need",
                ]
                variable_ranges = synthetic_df[variables_of_interest].agg(["min", "max", "mean"]).transpose().reset_index()
                variable_ranges.columns = ["variable", "min", "max", "mean"]
                print_frame("Observed ranges", variable_ranges, rows=20)
                display(variable_ranges)
                """
            ),
            md(
                """
                ## Analisis por variable

                Las siguientes celdas revisan la plausibilidad operativa de requirement, inventory, lead time, coverage, yield, waste y la senal upstream sintetica.
                """
            ),
            code(
                """
                requirement_summary = synthetic_df.groupby("destination_profile")["expected_requirement_tons"].agg(["count", "mean", "median", "min", "max"]).reset_index()
                print_frame("Requirement by destination_profile", requirement_summary, rows=20)
                display(requirement_summary)
                """
            ),
            code(
                """
                inventory_summary = synthetic_df.groupby("destination_profile")["current_inventory_tons"].agg(["mean", "median", "min", "max"]).reset_index()
                print_frame("Inventory by destination_profile", inventory_summary, rows=20)
                display(inventory_summary)
                """
            ),
            code(
                """
                lead_coverage_summary = synthetic_df.groupby("destination_profile")[["lead_time_days", "safety_coverage_days"]].mean().reset_index()
                print_frame("Lead time and safety coverage by profile", lead_coverage_summary, rows=20)
                display(lead_coverage_summary)
                """
            ),
            code(
                """
                yield_waste_summary = synthetic_df.groupby("destination_profile")[["expected_yield_rate", "expected_waste_rate"]].mean().reset_index()
                print_frame("Yield and waste by profile", yield_waste_summary, rows=20)
                display(yield_waste_summary)
                """
            ),
            code(
                """
                profile_summary = synthetic_df.groupby("destination_profile").agg(
                    rows=("destination_profile", "size"),
                    requirement_mean=("expected_requirement_tons", "mean"),
                    inventory_mean=("current_inventory_tons", "mean"),
                    procurement_need_mean=("synthetic_procurement_need", "mean"),
                ).reset_index()
                print_frame("Overall summary by profile", profile_summary, rows=20)
                display(profile_summary)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(requirement_summary["destination_profile"], requirement_summary["mean"], color="#355c7d")
                ax.set_title("Expected requirement by destination profile")
                ax.set_ylabel("tons")
                FIGURES.append(save_figure(fig, "synthetic_requirement_by_profile__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(synthetic_df["current_inventory_tons"], bins=20, color="#6c5b7b", edgecolor="white")
                ax.set_title("Current inventory distribution")
                ax.set_xlabel("tons")
                FIGURES.append(save_figure(fig, "synthetic_inventory_distribution__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                lead_time_plot = synthetic_df.groupby("destination_profile")["lead_time_days"].mean().reset_index()
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(lead_time_plot["destination_profile"], lead_time_plot["lead_time_days"], color="#c06c84")
                ax.set_title("Lead time by destination profile")
                ax.set_ylabel("days")
                FIGURES.append(save_figure(fig, "synthetic_lead_time_by_profile__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(8, 4))
                width = 0.35
                positions = np.arange(len(yield_waste_summary))
                ax.bar(positions - width / 2, yield_waste_summary["expected_yield_rate"], width=width, label="yield_rate")
                ax.bar(positions + width / 2, yield_waste_summary["expected_waste_rate"], width=width, label="waste_rate")
                ax.set_xticks(positions)
                ax.set_xticklabels(yield_waste_summary["destination_profile"], rotation=15)
                ax.set_title("Yield and waste by destination profile")
                ax.legend()
                FIGURES.append(save_figure(fig, "synthetic_yield_waste_by_profile__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                procurement_need_summary = synthetic_df.groupby("destination_profile")["synthetic_procurement_need"].mean().reset_index()
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(procurement_need_summary["destination_profile"], procurement_need_summary["synthetic_procurement_need"], color="#2a9d8f")
                ax.set_title("Synthetic procurement need by profile")
                ax.set_ylabel("proxy tons")
                FIGURES.append(save_figure(fig, "synthetic_procurement_need_by_profile__mixed_context.png"))
                plt.close(fig)
                print_frame("Synthetic procurement need by profile", procurement_need_summary, rows=20)
                """
            ),
            code(
                """
                operational_cols = [
                    "current_inventory_tons",
                    "expected_requirement_tons",
                    "lead_time_days",
                    "safety_coverage_days",
                    "expected_yield_rate",
                    "expected_waste_rate",
                    "synthetic_procurement_need",
                ]
                corr = synthetic_df[operational_cols].corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(7, 6))
                image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.index)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                ax.set_yticklabels(corr.index)
                ax.set_title("Operational variable correlation")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                FIGURES.append(save_figure(fig, "synthetic_layer_correlation__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                weekly_profile = synthetic_df.groupby("date")[["expected_requirement_tons", "current_inventory_tons"]].mean().reset_index()
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(weekly_profile["date"], weekly_profile["expected_requirement_tons"], label="expected_requirement_tons")
                ax.plot(weekly_profile["date"], weekly_profile["current_inventory_tons"], label="current_inventory_tons")
                ax.set_title("Weekly requirement and inventory")
                ax.set_ylabel("tons")
                ax.legend()
                FIGURES.append(save_figure(fig, "synthetic_requirement_inventory_timeseries__mixed_context.png"))
                plt.close(fig)
                """
            ),
            md(
                """
                ## Interpretacion

                `destination_profile` representa destino productivo previsto y no una compra observada. `synthetic_procurement_need` es una senal upstream de presion y no la cantidad final recomendada. La decision final de cantidad aparece despues como `order_quantity_tons`.
                """
            ),
            code(
                """
                TABLES.append(save_table(variable_ranges, "synthetic_layer_variable_ranges__mixed_context.csv"))
                TABLES.append(save_table(profile_summary, "synthetic_layer_by_profile__mixed_context.csv"))
                TABLES.append(save_table(null_summary, "synthetic_layer_summary__mixed_context.csv"))
                RESULT = {
                    "notebook": NOTEBOOK_NAME,
                    "tables": TABLES,
                    "figures": FIGURES,
                    "findings": [
                        "The synthetic layer produces plausible operational ranges by destination profile.",
                        "The upstream signal synthetic_procurement_need behaves as a pressure indicator and not as a final order quantity.",
                        "Inventory, requirement, lead time, yield and waste remain synthetic unless the customer uploads observed values.",
                    ],
                    "limitations": [
                        "This notebook validates plausibility only; it does not convert synthetic variables into observed plant evidence.",
                    ],
                }
                print(json.dumps(RESULT, indent=2))
                """
            ),
            md(
                """
                ## Limitaciones

                La capa sintetica es una decision metodologica declarada del caso de uso. Sirve para reproducibilidad y simulacion controlada, no para afirmar observacion directa de operaciones internas.
                """
            ),
            md(
                """
                ## Concluson final

                La evidencia visual confirma que la capa sintetica es coherente con el relato oficial: soporte batch/offline, datos externos de contexto y variables operativas sinteticas salvo carga futura de cliente.
                """
            ),
        ]
    )
    return cells


def nb04() -> list[dict]:
    cells = base_cells(
        "CU28 mixed_context - Feature Engineering Audit",
        "Mostrar de forma explicita como se estructura el dataset modelable, que features entran en cada etapa y como se controla el leakage.",
        [
            "data/processed/baseline/feature_engineering_modeling__mixed_context.csv",
            "data/processed/baseline/modeling_metadata__mixed_context.json",
            "data/processed/baseline/feature_contract__mixed_context.csv",
            "data/processed/baseline/feature_roles_metadata__mixed_context.json",
            "docs/input_contract.md",
            "docs/feature_engineering.md",
            "docs/leakage_policy.md",
        ],
        [
            "reports/tables/eda/feature_inventory__mixed_context.csv",
            "reports/tables/eda/leakage_audit__mixed_context.csv",
            "reports/tables/eda/feature_sets_allowed__mixed_context.csv",
            "reports/figures/eda/features_by_family__mixed_context.png",
            "reports/figures/eda/features_by_origin__mixed_context.png",
            "reports/figures/eda/feature_missing_values__mixed_context.png",
            "reports/figures/eda/feature_correlation__mixed_context.png",
            "reports/figures/eda/feature_target_correlation__mixed_context.png",
        ],
        "04_feature_engineering_audit.ipynb",
        extra_imports="""
from src.reproducibility.mixed_context import quantity_feature_columns, trigger_feature_columns
""",
    )
    cells.extend(
        [
            code(
                """
                modeling_path = PROJECT_ROOT / "data/processed/baseline/feature_engineering_modeling__mixed_context.csv"
                metadata_path = PROJECT_ROOT / "data/processed/baseline/modeling_metadata__mixed_context.json"
                feature_contract_path = PROJECT_ROOT / "data/processed/baseline/feature_contract__mixed_context.csv"
                feature_roles_path = PROJECT_ROOT / "data/processed/baseline/feature_roles_metadata__mixed_context.json"
                input_contract_path = PROJECT_ROOT / "docs" / "input_contract.md"
                modeling_df = pd.read_csv(modeling_path)
                metadata = read_json(metadata_path)
                feature_contract = pd.read_csv(feature_contract_path)
                feature_roles = read_json(feature_roles_path)
                input_contract_excerpt = input_contract_path.read_text(encoding="utf-8").splitlines()[:20]
                print(modeling_df.shape)
                print_frame("Feature contract preview", feature_contract.head(12))
                """
            ),
            code(
                """
                print("Input contract excerpt")
                for line in input_contract_excerpt:
                    print(line)
                print(json.dumps({"recommended_feature_set": feature_roles.get("recommended_feature_set")}, indent=2))
                """
            ),
            md(
                """
                ## Inspeccion inicial

                A continuacion se separan targets, outputs, inputs permitidos y columnas excluidas para dejar trazabilidad explicita de la fase de feature engineering.
                """
            ),
            code(
                """
                shape_summary = pd.DataFrame([{"rows": len(modeling_df), "columns": len(modeling_df.columns)}])
                column_summary = pd.DataFrame({"feature_name": modeling_df.columns})
                print_frame("Modeling dataset shape", shape_summary)
                print_frame("Modeling columns", column_summary, rows=30)
                """
            ),
            code(
                """
                targets_df = feature_contract[feature_contract["output_role"] == "predictive_target"].copy()
                outputs_df = feature_contract[feature_contract["output_role"] == "decision_output"].copy()
                inputs_df = feature_contract[feature_contract["allowed_model_input"] == "yes"].copy()
                excluded_df = feature_contract[feature_contract["allowed_model_input"] == "no"].copy()
                print_frame("Targets", targets_df)
                print_frame("Decision outputs", outputs_df)
                print_frame("Allowed model inputs", inputs_df.head(20))
                """
            ),
            code(
                """
                lag_features = feature_contract[feature_contract["feature_name"].str.contains("_lag_", na=False)].copy()
                rolling_features = feature_contract[feature_contract["feature_name"].str.contains("_roll_mean_", na=False)].copy()
                calendar_features = feature_contract[feature_contract["feature_origin"] == "calendar_derived"].copy()
                profile_features = feature_contract[
                    feature_contract["feature_name"].str.startswith(
                        ("product_family__", "recipe_profile__", "shelf_life_class__", "manufacturing_context_profile__"),
                        na=False,
                    )
                ].copy()
                print_frame("Lag features", lag_features.head(20))
                print_frame("Rolling features", rolling_features.head(20))
                print_frame("Calendar features", calendar_features.head(20))
                print_frame("Profile features", profile_features.head(20))
                """
            ),
            code(
                """
                missing_summary = modeling_df.isna().mean().reset_index()
                missing_summary.columns = ["feature_name", "missing_pct"]
                missing_summary = missing_summary.sort_values("missing_pct", ascending=False)
                constant_features = pd.DataFrame(
                    {"feature_name": [column for column in modeling_df.columns if modeling_df[column].nunique(dropna=False) <= 1]}
                )
                print_frame("Missing summary", missing_summary.head(20))
                print_frame("Constant features", constant_features.head(20))
                """
            ),
            code(
                """
                feature_inventory = feature_contract[[
                    "feature_name",
                    "feature_origin",
                    "feature_type",
                    "system_layer",
                    "temporal_relation_to_target",
                    "output_role",
                    "allowed_model_input",
                    "leakage_risk",
                ]].copy()
                print_frame("Feature inventory", feature_inventory.head(20))
                display(feature_inventory.head(20))
                """
            ),
            md(
                """
                ## Auditoria de leakage

                La politica oficial excluye salidas downstream y señales de decision final como inputs indebidos en la fase upstream y en la fase trigger.
                """
            ),
            code(
                """
                prohibited_features = [
                    "order_quantity_tons",
                    "quantity_optimizer_recommendation_tons",
                    "quantity_optimizer_target_tons",
                    "excess_tons",
                    "stockout_tons",
                    "purchase_trigger_flag",
                    "purchase_trigger_proba",
                ]
                trigger_allowed = set(trigger_feature_columns(modeling_df))
                quantity_allowed = set(quantity_feature_columns(modeling_df))
                upstream_allowed = set(feature_roles.get("official_extended_inputs", []))
                leakage_rows = []
                for feature_name in prohibited_features:
                    leakage_rows.append(
                        {
                            "feature_name": feature_name,
                            "present_in_dataset": feature_name in modeling_df.columns,
                            "in_upstream": feature_name in upstream_allowed,
                            "in_trigger": feature_name in trigger_allowed,
                            "in_quantity_optimizer": feature_name in quantity_allowed,
                            "status": "fail" if (feature_name in upstream_allowed or feature_name in trigger_allowed) else "pass",
                        }
                    )
                leakage_audit_df = pd.DataFrame(leakage_rows)
                print_frame("Leakage audit", leakage_audit_df, rows=20)
                display(leakage_audit_df)
                """
            ),
            code(
                """
                feature_sets_allowed = pd.DataFrame(
                    [
                        {"stage": "upstream_predictor", "feature_name": feature_name}
                        for feature_name in sorted(upstream_allowed)
                    ]
                    + [
                        {"stage": "purchase_trigger", "feature_name": feature_name}
                        for feature_name in sorted(trigger_allowed)
                    ]
                    + [
                        {"stage": "quantity_optimizer", "feature_name": feature_name}
                        for feature_name in sorted(quantity_allowed)
                    ]
                )
                print_frame("Allowed feature sets by stage", feature_sets_allowed.head(30))
                display(feature_sets_allowed.head(30))
                """
            ),
            code(
                """
                origin_counts = feature_contract["feature_origin"].value_counts().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(origin_counts.index, origin_counts.values, color="#355c7d")
                ax.set_title("Features by origin")
                ax.set_ylabel("count")
                ax.tick_params(axis="x", rotation=45)
                FIGURES.append(save_figure(fig, "features_by_origin__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                family_counts = feature_contract["feature_type"].value_counts().sort_values(ascending=False).head(12)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(family_counts.index, family_counts.values, color="#6c5b7b")
                ax.set_title("Features by family")
                ax.set_ylabel("count")
                ax.tick_params(axis="x", rotation=45)
                FIGURES.append(save_figure(fig, "features_by_family__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                missing_plot = missing_summary.head(20).copy()
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(missing_plot["feature_name"], missing_plot["missing_pct"], color="#f67280")
                ax.set_title("Top missing values after feature engineering")
                ax.set_ylabel("missing_pct")
                ax.tick_params(axis="x", rotation=75)
                FIGURES.append(save_figure(fig, "feature_missing_values__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                numeric_candidates = modeling_df.select_dtypes(include=["number"]).copy()
                correlation_columns = ["synthetic_procurement_need", "purchase_trigger_label", "demand_index", "supply_index", "current_inventory_tons", "expected_requirement_tons"]
                correlation_columns = [column for column in correlation_columns if column in numeric_candidates.columns]
                corr = numeric_candidates[correlation_columns].corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(7, 5))
                image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.index)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                ax.set_yticklabels(corr.index)
                ax.set_title("Key feature correlation")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                FIGURES.append(save_figure(fig, "feature_correlation__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                target_corr = numeric_candidates.corr(numeric_only=True)["synthetic_procurement_need"].dropna().sort_values(key=lambda s: s.abs(), ascending=False).head(15)
                target_corr_df = target_corr.reset_index()
                target_corr_df.columns = ["feature_name", "correlation_with_target"]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(target_corr_df["feature_name"], target_corr_df["correlation_with_target"], color="#2a9d8f")
                ax.set_title("Correlation with synthetic_procurement_need")
                ax.tick_params(axis="x", rotation=75)
                FIGURES.append(save_figure(fig, "feature_target_correlation__mixed_context.png"))
                plt.close(fig)
                print_frame("Correlation with synthetic_procurement_need", target_corr_df, rows=20)
                """
            ),
            md(
                """
                ## Interpretacion

                La auditoria confirma que las salidas downstream no entran como inputs indebidos en upstream ni trigger. `purchase_trigger_label` y su probabilidad solo aparecen como entradas legitimas del quantity optimizer.
                """
            ),
            code(
                """
                TABLES.append(save_table(feature_inventory, "feature_inventory__mixed_context.csv"))
                TABLES.append(save_table(leakage_audit_df, "leakage_audit__mixed_context.csv"))
                TABLES.append(save_table(feature_sets_allowed, "feature_sets_allowed__mixed_context.csv"))
                RESULT = {
                    "notebook": NOTEBOOK_NAME,
                    "tables": TABLES,
                    "figures": FIGURES,
                    "findings": [
                        "Feature engineering is traceable through the feature contract and roles metadata.",
                        "Lagged, rolling and profile-derived variables are visible as separate families.",
                        "Downstream decision outputs are excluded from upstream and trigger inputs.",
                    ],
                    "limitations": [
                        "The notebook audits the exported feature contract; it does not replace code review of the full feature pipeline.",
                    ],
                }
                print(json.dumps(RESULT, indent=2))
                """
            ),
            md(
                """
                ## Limitaciones

                El notebook audita los artefactos exportados de feature engineering. No sustituye la lectura del codigo completo ni la inspeccion manual de todas las transformaciones intermedias.
                """
            ),
            md(
                """
                ## Concluson final

                La fase de feature engineering queda defendible cuando los conjuntos permitidos por etapa son visibles, las exclusiones por leakage son explicitas y el dataset modelable puede reconstruirse a partir de artefactos versionados.
                """
            ),
        ]
    )
    return cells


def nb05() -> list[dict]:
    cells = base_cells(
        "CU28 mixed_context - Modeling Dataset EDA",
        "Realizar el EDA central del dataset final usado para entrenamiento y evaluacion, distinguiendo target upstream, trigger label y senales operativas.",
        [
            "data/processed/baseline/feature_engineering_modeling__mixed_context.csv",
            "data/processed/baseline/modeling_metadata__mixed_context.json",
        ],
        [
            "reports/tables/eda/modeling_dataset_summary__mixed_context.csv",
            "reports/tables/eda/modeling_dataset_quality__mixed_context.csv",
            "reports/tables/eda/target_by_profile__mixed_context.csv",
            "reports/tables/eda/trigger_balance__mixed_context.csv",
            "reports/figures/eda/target_distribution__mixed_context.png",
            "reports/figures/eda/target_by_profile__mixed_context.png",
            "reports/figures/eda/trigger_balance__mixed_context.png",
            "reports/figures/eda/trigger_by_profile__mixed_context.png",
            "reports/figures/eda/modeling_dataset_timeseries__mixed_context.png",
            "reports/figures/eda/modeling_dataset_correlation__mixed_context.png",
        ],
        "05_modeling_dataset_eda.ipynb",
    )
    cells.extend(
        [
            code(
                """
                modeling_path = PROJECT_ROOT / "data/processed/baseline/feature_engineering_modeling__mixed_context.csv"
                metadata_path = PROJECT_ROOT / "data/processed/baseline/modeling_metadata__mixed_context.json"
                modeling_df = pd.read_csv(modeling_path)
                modeling_df["date"] = pd.to_datetime(modeling_df["date"], errors="coerce")
                modeling_meta = read_json(metadata_path)
                print(modeling_df.shape)
                print_frame("Modeling dataset preview", modeling_df.head(10))
                """
            ),
            code(
                """
                dataset_summary = pd.DataFrame(
                    [
                        {
                            "rows": len(modeling_df),
                            "columns": len(modeling_df.columns),
                            "date_min": str(modeling_df["date"].min().date()),
                            "date_max": str(modeling_df["date"].max().date()),
                            "granularity": "weekly",
                            "destination_profiles": int(modeling_df["destination_profile"].nunique()),
                        }
                    ]
                )
                column_summary = pd.DataFrame({"column": modeling_df.columns})
                print_frame("Dataset summary", dataset_summary)
                print_frame("Column summary", column_summary, rows=30)
                """
            ),
            md(
                """
                ## Calidad de datos

                Se revisan nulos, duplicados, columnas constantes y outliers basicos antes de interpretar el target upstream y la etiqueta trigger.
                """
            ),
            code(
                """
                null_summary = modeling_df.isna().mean().reset_index()
                null_summary.columns = ["column", "missing_pct"]
                duplicate_count = int(modeling_df.duplicated().sum())
                constant_columns = [column for column in modeling_df.columns if modeling_df[column].nunique(dropna=False) <= 1]
                quality_summary = pd.DataFrame(
                    [
                        {"metric": "duplicate_rows", "value": duplicate_count},
                        {"metric": "constant_columns", "value": len(constant_columns)},
                    ]
                )
                print_frame("Null summary", null_summary.sort_values("missing_pct", ascending=False).head(20))
                print_frame("Quality summary", quality_summary)
                """
            ),
            code(
                """
                outlier_rows = []
                for column in ["synthetic_procurement_need", "current_inventory_tons", "expected_requirement_tons"]:
                    q1 = modeling_df[column].quantile(0.25)
                    q3 = modeling_df[column].quantile(0.75)
                    iqr = q3 - q1
                    upper = q3 + 1.5 * iqr
                    lower = q1 - 1.5 * iqr
                    outlier_rows.append(
                        {
                            "column": column,
                            "lower_bound": lower,
                            "upper_bound": upper,
                            "outlier_rows": int(((modeling_df[column] < lower) | (modeling_df[column] > upper)).sum()),
                        }
                    )
                outlier_df = pd.DataFrame(outlier_rows)
                print_frame("Basic outlier audit", outlier_df)
                display(outlier_df)
                """
            ),
            code(
                """
                target_distribution = modeling_df["synthetic_procurement_need"].describe().reset_index()
                target_distribution.columns = ["metric", "value"]
                target_by_profile = modeling_df.groupby("destination_profile")["synthetic_procurement_need"].agg(["count", "mean", "median", "min", "max"]).reset_index()
                print_frame("Target distribution", target_distribution, rows=20)
                print_frame("Target by profile", target_by_profile, rows=20)
                """
            ),
            code(
                """
                trigger_balance = modeling_df["purchase_trigger_label"].value_counts(normalize=True).reset_index()
                trigger_balance.columns = ["purchase_trigger_label", "share"]
                trigger_by_profile = modeling_df.groupby("destination_profile")["purchase_trigger_label"].mean().reset_index()
                print_frame("Trigger balance", trigger_balance, rows=20)
                print_frame("Trigger by profile", trigger_by_profile, rows=20)
                """
            ),
            code(
                """
                trigger_vs_inventory = modeling_df.groupby(pd.cut(modeling_df["current_inventory_tons"], bins=10))["purchase_trigger_label"].mean().reset_index()
                trigger_vs_requirement = modeling_df.groupby(pd.cut(modeling_df["expected_requirement_tons"], bins=10))["purchase_trigger_label"].mean().reset_index()
                print_frame("Trigger vs inventory bins", trigger_vs_inventory, rows=20)
                print_frame("Trigger vs requirement bins", trigger_vs_requirement, rows=20)
                """
            ),
            md(
                """
                ## Graficas

                Las graficas siguientes muestran comportamiento del target upstream, balance trigger y senales temporales semanales.
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(modeling_df["synthetic_procurement_need"], bins=25, color="#355c7d", edgecolor="white")
                ax.set_title("synthetic_procurement_need distribution")
                ax.set_xlabel("tons")
                FIGURES.append(save_figure(fig, "target_distribution__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(target_by_profile["destination_profile"], target_by_profile["mean"], color="#6c5b7b")
                ax.set_title("synthetic_procurement_need by destination profile")
                ax.set_ylabel("mean tons")
                FIGURES.append(save_figure(fig, "target_by_profile__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(trigger_balance["purchase_trigger_label"].astype(str), trigger_balance["share"], color="#c06c84")
                ax.set_title("Trigger label balance")
                ax.set_ylabel("share")
                FIGURES.append(save_figure(fig, "trigger_balance__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(trigger_by_profile["destination_profile"], trigger_by_profile["purchase_trigger_label"], color="#f67280")
                ax.set_title("Trigger rate by destination profile")
                ax.set_ylabel("trigger_rate")
                FIGURES.append(save_figure(fig, "trigger_by_profile__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                weekly_signals = modeling_df.groupby("date")[["synthetic_procurement_need", "current_inventory_tons", "expected_requirement_tons"]].mean().reset_index()
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(weekly_signals["date"], weekly_signals["synthetic_procurement_need"], label="synthetic_procurement_need")
                ax.plot(weekly_signals["date"], weekly_signals["expected_requirement_tons"], label="expected_requirement_tons")
                ax.plot(weekly_signals["date"], weekly_signals["current_inventory_tons"], label="current_inventory_tons")
                ax.set_title("Weekly modeling signals")
                ax.legend()
                FIGURES.append(save_figure(fig, "modeling_dataset_timeseries__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                corr_columns = [
                    "synthetic_procurement_need",
                    "purchase_trigger_label",
                    "current_inventory_tons",
                    "expected_requirement_tons",
                    "lead_time_days",
                    "safety_coverage_days",
                ]
                corr = modeling_df[corr_columns].corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(7, 5))
                image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.index)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                ax.set_yticklabels(corr.index)
                ax.set_title("Reduced correlation matrix")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                FIGURES.append(save_figure(fig, "modeling_dataset_correlation__mixed_context.png"))
                plt.close(fig)
                """
            ),
            md(
                """
                ## Interpretacion

                La granularidad semanal permite combinar proxies externos con una capa sintetica de planta sin pretender que el dataset sea un historico observado de compras. `synthetic_procurement_need` sigue siendo el target upstream y no la cantidad final recomendada.
                """
            ),
            code(
                """
                TABLES.append(save_table(dataset_summary, "modeling_dataset_summary__mixed_context.csv"))
                TABLES.append(save_table(outlier_df, "modeling_dataset_quality__mixed_context.csv"))
                TABLES.append(save_table(target_by_profile, "target_by_profile__mixed_context.csv"))
                TABLES.append(save_table(trigger_balance, "trigger_balance__mixed_context.csv"))
                RESULT = {
                    "notebook": NOTEBOOK_NAME,
                    "tables": TABLES,
                    "figures": FIGURES,
                    "findings": [
                        "The modeling dataset is weekly and combines proxy, synthetic and derived variables.",
                        "synthetic_procurement_need varies materially by destination profile and supports upstream modeling.",
                        "purchase_trigger_label tracks operational stress through inventory, requirement and lead-time relationships.",
                    ],
                    "limitations": [
                        "The dataset remains mixed and derived; it should not be read as an observed purchase ledger.",
                    ],
                }
                print(json.dumps(RESULT, indent=2))
                """
            ),
            md(
                """
                ## Limitaciones

                La calidad del dataset final depende de decisiones de simulacion y armonizacion semanal. El objetivo del notebook es hacer visibles esas decisiones y sus efectos estadisticos basicos.
                """
            ),
            md(
                """
                ## Concluson final

                Este es el EDA central del dataset modelable: resume cobertura temporal, calidad, dispersion por perfil y comportamiento conjunto de target upstream y trigger.
                """
            ),
        ]
    )
    return cells


def nb06() -> list[dict]:
    cells = base_cells(
        "CU28 mixed_context - Split Validation and Leakage Audit",
        "Demostrar que los splits train, validation y test son cronologicos, no se mezclan temporalmente y reservan test para evaluacion final.",
        [
            "data/splits/baseline/default__mixed_context/train.csv",
            "data/splits/baseline/default__mixed_context/validation.csv",
            "data/splits/baseline/default__mixed_context/test.csv",
            "data/splits/baseline/default__mixed_context/split_metadata.json",
            "data/processed/baseline/feature_engineering_modeling__mixed_context.csv",
        ],
        [
            "reports/tables/eda/split_summary__mixed_context.csv",
            "reports/tables/eda/split_validation_checks__mixed_context.csv",
            "reports/tables/eda/split_target_distribution__mixed_context.csv",
            "reports/figures/eda/split_timeline__mixed_context.png",
            "reports/figures/eda/target_by_split__mixed_context.png",
            "reports/figures/eda/trigger_by_split__mixed_context.png",
            "reports/figures/eda/profile_coverage_by_split__mixed_context.png",
        ],
        "06_split_validation_and_leakage_audit.ipynb",
    )
    cells.extend(
        [
            code(
                """
                split_root = PROJECT_ROOT / "data/splits/baseline/default__mixed_context"
                train_df = pd.read_csv(split_root / "train.csv")
                validation_df = pd.read_csv(split_root / "validation.csv")
                test_df = pd.read_csv(split_root / "test.csv")
                for frame in [train_df, validation_df, test_df]:
                    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
                split_meta = read_json(split_root / "split_metadata.json")
                print({"train_rows": len(train_df), "validation_rows": len(validation_df), "test_rows": len(test_df)})
                """
            ),
            code(
                """
                split_summary = pd.DataFrame(
                    [
                        {
                            "split": "train",
                            "rows": len(train_df),
                            "date_min": str(train_df["date"].min().date()),
                            "date_max": str(train_df["date"].max().date()),
                        },
                        {
                            "split": "validation",
                            "rows": len(validation_df),
                            "date_min": str(validation_df["date"].min().date()),
                            "date_max": str(validation_df["date"].max().date()),
                        },
                        {
                            "split": "test",
                            "rows": len(test_df),
                            "date_min": str(test_df["date"].min().date()),
                            "date_max": str(test_df["date"].max().date()),
                        },
                    ]
                )
                print_frame("Split summary", split_summary, rows=10)
                display(split_summary)
                """
            ),
            md(
                """
                ## Validaciones cronologicas

                Se comprueba explicitamente que train termina antes de validation y que validation termina antes de test.
                """
            ),
            code(
                """
                chronology_checks = pd.DataFrame(
                    [
                        {"check": "max_train_before_min_validation", "pass": bool(train_df["date"].max() < validation_df["date"].min())},
                        {"check": "max_validation_before_min_test", "pass": bool(validation_df["date"].max() < test_df["date"].min())},
                    ]
                )
                print_frame("Chronology checks", chronology_checks)
                display(chronology_checks)
                """
            ),
            code(
                """
                train_keys = set(zip(train_df["date"], train_df["destination_profile"], train_df["raw_material_id"]))
                validation_keys = set(zip(validation_df["date"], validation_df["destination_profile"], validation_df["raw_material_id"]))
                test_keys = set(zip(test_df["date"], test_df["destination_profile"], test_df["raw_material_id"]))
                overlap_counts = pd.DataFrame(
                    [
                        {"pair": "train_validation", "overlap_rows": len(train_keys & validation_keys)},
                        {"pair": "validation_test", "overlap_rows": len(validation_keys & test_keys)},
                        {"pair": "train_test", "overlap_rows": len(train_keys & test_keys)},
                    ]
                )
                print_frame("Duplicate keys across splits", overlap_counts)
                display(overlap_counts)
                """
            ),
            code(
                """
                combined = pd.concat(
                    [
                        train_df.assign(split="train"),
                        validation_df.assign(split="validation"),
                        test_df.assign(split="test"),
                    ],
                    ignore_index=True,
                )
                target_by_split = combined.groupby("split")["synthetic_procurement_need"].agg(["mean", "median", "min", "max"]).reset_index()
                trigger_by_split = combined.groupby("split")["purchase_trigger_label"].mean().reset_index()
                profile_coverage = combined.groupby(["split", "destination_profile"]).size().reset_index(name="rows")
                print_frame("Target by split", target_by_split)
                print_frame("Trigger rate by split", trigger_by_split)
                print_frame("Profile coverage", profile_coverage, rows=20)
                """
            ),
            md(
                """
                ## Visualizacion de splits

                Las graficas siguientes deben dejar claro que la separacion es temporal, que la distribucion del target cambia de forma razonable y que ningun split usa informacion futura para seleccion.
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(10, 4))
                offsets = {"train": 0, "validation": 1, "test": 2}
                colors = {"train": "#355c7d", "validation": "#6c5b7b", "test": "#c06c84"}
                for split_name, split_df in [("train", train_df), ("validation", validation_df), ("test", test_df)]:
                    ax.scatter(split_df["date"], np.full(len(split_df), offsets[split_name]), s=8, alpha=0.5, color=colors[split_name], label=split_name)
                ax.set_yticks([0, 1, 2])
                ax.set_yticklabels(["train", "validation", "test"])
                ax.set_title("Timeline by split")
                ax.legend()
                FIGURES.append(save_figure(fig, "split_timeline__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(target_by_split["split"], target_by_split["mean"], color=["#355c7d", "#6c5b7b", "#c06c84"])
                ax.set_title("Average target by split")
                ax.set_ylabel("synthetic_procurement_need")
                FIGURES.append(save_figure(fig, "target_by_split__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(trigger_by_split["split"], trigger_by_split["purchase_trigger_label"], color=["#355c7d", "#6c5b7b", "#c06c84"])
                ax.set_title("Trigger rate by split")
                ax.set_ylabel("trigger_rate")
                FIGURES.append(save_figure(fig, "trigger_by_split__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                pivot_profile = profile_coverage.pivot(index="destination_profile", columns="split", values="rows").fillna(0)
                fig, ax = plt.subplots(figsize=(8, 4))
                pivot_profile.plot(kind="bar", ax=ax)
                ax.set_title("Profile coverage by split")
                ax.set_ylabel("rows")
                FIGURES.append(save_figure(fig, "profile_coverage_by_split__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                checks_df = chronology_checks.copy()
                checks_df["detail"] = [
                    "train end before validation start",
                    "validation end before test start",
                ]
                checks_df = pd.concat(
                    [
                        checks_df,
                        pd.DataFrame([{"check": "no_overlap_between_splits", "pass": int(overlap_counts["overlap_rows"].sum()) == 0, "detail": "no duplicated key tuple across splits"}]),
                    ],
                    ignore_index=True,
                )
                print_frame("Pass fail checks", checks_df, rows=20)
                display(checks_df)
                """
            ),
            md(
                """
                ## Interpretacion

                `validation` se usa para seleccion y calibracion. `test` queda reservado a evaluacion final. La grafica temporal debe mostrar bloques ordenados sin mezcla entre periodos.
                """
            ),
            code(
                """
                TABLES.append(save_table(split_summary, "split_summary__mixed_context.csv"))
                TABLES.append(save_table(checks_df, "split_validation_checks__mixed_context.csv"))
                TABLES.append(save_table(target_by_split, "split_target_distribution__mixed_context.csv"))
                RESULT = {
                    "notebook": NOTEBOOK_NAME,
                    "tables": TABLES,
                    "figures": FIGURES,
                    "findings": [
                        "Train, validation and test remain chronologically ordered.",
                        "No duplicate key tuples were found across splits in the defended dataset.",
                        "Validation is the selection surface while test is reserved for final evaluation.",
                    ],
                    "limitations": [
                        "Coverage by profile can still differ across splits because the scenario mix evolves over time.",
                    ],
                }
                print(json.dumps(RESULT, indent=2))
                """
            ),
            md(
                """
                ## Limitaciones

                Un split cronologico no elimina toda deriva temporal. Solo garantiza que la evaluacion final no vea filas posteriores durante la seleccion de configuracion.
                """
            ),
            md(
                """
                ## Concluson final

                La evidencia conjunta de fechas, no solapamiento y uso diferenciado de validation/test permite defender la estrategia de particionado del pipeline mixed_context.
                """
            ),
        ]
    )
    return cells


def nb07() -> list[dict]:
    cells = base_cells(
        "CU28 mixed_context - Training and Policy Results EDA",
        "Visualizar resultados de entrenamiento, trigger, optimizador y simulacion de politica para la ruta mixed_context, distinguiendo claramente metricas de validacion funcional parcial en simulacion.",
        [
            "models/metrics/summary/baseline_comparison_latest__mixed_context.json",
            "models/metrics/summary/neuroevolution_comparison_latest__mixed_context.json",
            "models/metrics/summary/trigger_metrics_latest__mixed_context.json",
            "models/metrics/summary/quantity_optimizer_latest__mixed_context.json",
            "models/metrics/summary/policy_simulation_latest__mixed_context.json",
            "models/metrics/summary/metrics_summary__mixed_context.json",
            "data/predictions/predictions_latest__mixed_context.csv",
            "data/splits/baseline/default__mixed_context/test.csv",
        ],
        [
            "reports/tables/eda/training_metrics_summary__mixed_context.csv",
            "reports/tables/eda/policy_metrics_summary__mixed_context.csv",
            "reports/tables/eda/trigger_confusion_matrix__mixed_context.csv",
            "reports/figures/eda/upstream_prediction_vs_actual__mixed_context.png",
            "reports/figures/eda/trigger_probability_distribution__mixed_context.png",
            "reports/figures/eda/trigger_confusion_matrix__mixed_context.png",
            "reports/figures/eda/order_quantity_vs_baseline__mixed_context.png",
            "reports/figures/eda/excess_by_policy__mixed_context.png",
            "reports/figures/eda/stockout_by_policy__mixed_context.png",
            "reports/figures/eda/policy_timeseries__mixed_context.png",
        ],
        "07_training_and_policy_results_eda.ipynb",
    )
    cells.extend(
        [
            code(
                """
                metrics_root = PROJECT_ROOT / "models/metrics/summary"
                baseline_metrics = read_json(metrics_root / "baseline_comparison_latest__mixed_context.json")
                neuro_metrics = read_json(metrics_root / "neuroevolution_comparison_latest__mixed_context.json")
                trigger_metrics = read_json(metrics_root / "trigger_metrics_latest__mixed_context.json")
                quantity_metrics = read_json(metrics_root / "quantity_optimizer_latest__mixed_context.json")
                policy_metrics = read_json(metrics_root / "policy_simulation_latest__mixed_context.json")
                metrics_summary = read_json(metrics_root / "metrics_summary__mixed_context.json")
                predictions = pd.read_csv(PROJECT_ROOT / "data/predictions/predictions_latest__mixed_context.csv")
                test_df = pd.read_csv(PROJECT_ROOT / "data/splits/baseline/default__mixed_context/test.csv")
                predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
                test_df["date"] = pd.to_datetime(test_df["date"], errors="coerce")
                print(predictions.shape)
                print_frame("Predictions preview", predictions.head(10))
                """
            ),
            code(
                """
                merged = predictions.merge(
                    test_df[["date", "raw_material_id", "destination_profile", "synthetic_procurement_need", "purchase_trigger_label", "quantity_optimizer_target_tons"]],
                    on=["date", "raw_material_id", "destination_profile", "synthetic_procurement_need"],
                    how="left",
                )
                upstream_summary = pd.DataFrame(
                    [
                        {
                            "model_family": metrics_summary["upstream"]["baseline_reference_run"]["model_family"],
                            "feature_set": metrics_summary["upstream"]["baseline_reference_run"]["feature_set"],
                            "validation_rmse": metrics_summary["upstream"]["baseline_reference_run"]["validation_rmse"],
                            "test_rmse": metrics_summary["upstream"]["baseline_reference_run"]["test_rmse"],
                            "recommendation": metrics_summary["upstream"]["recommendation"],
                        }
                    ]
                )
                print_frame("Upstream summary", upstream_summary)
                display(upstream_summary)
                """
            ),
            code(
                """
                trigger_summary = pd.DataFrame(
                    [
                        {
                            "split": split_name,
                            "accuracy": trigger_metrics[split_name]["accuracy"],
                            "precision": trigger_metrics[split_name]["precision"],
                            "recall": trigger_metrics[split_name]["recall"],
                            "false_negative_rate": trigger_metrics[split_name]["false_negative_rate"],
                        }
                        for split_name in ["train", "validation", "test"]
                    ]
                )
                quantity_summary = pd.DataFrame(
                    [
                        {
                            "split": split_name,
                            "mae": quantity_metrics[split_name]["mae"],
                            "rmse": quantity_metrics[split_name]["rmse"],
                            "r2": quantity_metrics[split_name]["r2"],
                        }
                        for split_name in ["train", "validation", "test"]
                    ]
                )
                print_frame("Trigger summary", trigger_summary)
                print_frame("Quantity summary", quantity_summary)
                """
            ),
            md(
                """
                ## Validaciones funcionales

                Antes de leer las figuras, se revisan resumenes de metrica por etapa y se comprueba que las predicciones se pueden alinear con el test split para una evaluacion final separada de la seleccion.
                """
            ),
            md(
                """
                ## Analisis de predicciones y trigger

                En esta seccion se relacionan probabilidades, etiquetas observadas y cantidades recomendadas para evaluar la politica frente al baseline.
                """
            ),
            code(
                """
                probability_summary = predictions["purchase_trigger_proba"].describe().reset_index()
                probability_summary.columns = ["metric", "value"]
                trigger_activation = pd.DataFrame(
                    [
                        {
                            "activation_rate": float(predictions["purchase_trigger_flag"].mean()),
                            "zero_orders_when_no_trigger": bool((predictions.loc[predictions["purchase_trigger_flag"] == 0, "order_quantity_tons"] == 0.0).all()),
                        }
                    ]
                )
                print_frame("Probability summary", probability_summary)
                print_frame("Trigger activation summary", trigger_activation)
                """
            ),
            code(
                """
                confusion = pd.crosstab(
                    merged["purchase_trigger_label"].fillna(-1),
                    merged["purchase_trigger_flag"],
                    rownames=["actual_label"],
                    colnames=["predicted_flag"],
                    dropna=False,
                )
                confusion_df = confusion.reset_index()
                print_frame("Confusion matrix table", confusion_df)
                display(confusion_df)
                """
            ),
            code(
                """
                order_summary = predictions[["order_quantity_tons", "baseline_order_quantity_tons", "quantity_optimizer_recommendation_tons"]].describe().transpose().reset_index()
                order_summary.columns = ["metric_group", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
                policy_summary = pd.DataFrame([policy_metrics])
                print_frame("Order quantity summary", order_summary)
                print_frame("Policy summary", policy_summary)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(merged["synthetic_procurement_need"], merged["synthetic_procurement_need_pred"], alpha=0.5, color="#355c7d")
                diagonal_min = min(merged["synthetic_procurement_need"].min(), merged["synthetic_procurement_need_pred"].min())
                diagonal_max = max(merged["synthetic_procurement_need"].max(), merged["synthetic_procurement_need_pred"].max())
                ax.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], linestyle="--", color="black")
                ax.set_title("Upstream prediction vs actual")
                ax.set_xlabel("actual synthetic_procurement_need")
                ax.set_ylabel("predicted synthetic_procurement_need")
                FIGURES.append(save_figure(fig, "upstream_prediction_vs_actual__mixed_context.png"))
                plt.close(fig)
                """
            ),
            md(
                """
                ### Interpretacion de la figura

                La diagonal sirve como referencia de ajuste. El objetivo sigue siendo upstream y no debe confundirse con una cantidad final de compra.
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(predictions["purchase_trigger_proba"], bins=20, color="#6c5b7b", edgecolor="white")
                ax.set_title("Trigger probability distribution")
                ax.set_xlabel("purchase_trigger_proba")
                FIGURES.append(save_figure(fig, "trigger_probability_distribution__mixed_context.png"))
                plt.close(fig)
                """
            ),
            md(
                """
                ### Interpretacion de la figura

                La distribucion de probabilidades ayuda a identificar si el trigger esta saturado o si mantiene separacion entre casos de compra y no compra.
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(5, 4))
                image = ax.imshow(confusion.values, cmap="Blues")
                ax.set_xticks(range(len(confusion.columns)))
                ax.set_yticks(range(len(confusion.index)))
                ax.set_xticklabels(confusion.columns.tolist())
                ax.set_yticklabels(confusion.index.tolist())
                ax.set_title("Trigger confusion matrix")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                FIGURES.append(save_figure(fig, "trigger_confusion_matrix__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(predictions["baseline_order_quantity_tons"], predictions["order_quantity_tons"], alpha=0.5, color="#c06c84")
                ax.set_title("Recommended order vs baseline order")
                ax.set_xlabel("baseline_order_quantity_tons")
                ax.set_ylabel("order_quantity_tons")
                FIGURES.append(save_figure(fig, "order_quantity_vs_baseline__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                excess_df = pd.DataFrame(
                    [
                        {"policy": policy_metrics["baseline_policy_name"], "excess_tons": policy_metrics["baseline_excess_tons"]},
                        {"policy": policy_metrics["proposed_policy_name"], "excess_tons": policy_metrics["policy_excess_tons"]},
                    ]
                )
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(excess_df["policy"], excess_df["excess_tons"], color=["#999999", "#2a9d8f"])
                ax.set_title("Excess tons by policy")
                ax.set_ylabel("tons")
                FIGURES.append(save_figure(fig, "excess_by_policy__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                stockout_df = pd.DataFrame(
                    [
                        {"policy": policy_metrics["baseline_policy_name"], "stockout_tons": policy_metrics["baseline_stockout_tons"]},
                        {"policy": policy_metrics["proposed_policy_name"], "stockout_tons": policy_metrics["stockout_tons"]},
                    ]
                )
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(stockout_df["policy"], stockout_df["stockout_tons"], color=["#999999", "#e76f51"])
                ax.set_title("Stockout tons by policy")
                ax.set_ylabel("tons")
                FIGURES.append(save_figure(fig, "stockout_by_policy__mixed_context.png"))
                plt.close(fig)
                """
            ),
            code(
                """
                policy_ts = predictions.groupby("date")[["order_quantity_tons", "baseline_order_quantity_tons"]].sum().reset_index()
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(policy_ts["date"], policy_ts["order_quantity_tons"], label="order_quantity_tons")
                ax.plot(policy_ts["date"], policy_ts["baseline_order_quantity_tons"], label="baseline_order_quantity_tons")
                ax.set_title("Policy quantities over time")
                ax.set_ylabel("tons")
                ax.legend()
                FIGURES.append(save_figure(fig, "policy_timeseries__mixed_context.png"))
                plt.close(fig)
                """
            ),
            md(
                """
                ## Interpretacion

                El KPI de reduccion de exceso se lee frente al baseline y dentro de una validacion funcional parcial en simulacion. `order_quantity_tons` es una salida calculada de politica, no un registro observado de compra. El guardrail exige no empeorar stockout.
                """
            ),
            code(
                """
                training_metrics_summary = pd.concat(
                    [
                        upstream_summary.assign(metric_group="upstream"),
                        trigger_summary.assign(metric_group="trigger"),
                        quantity_summary.assign(metric_group="quantity_optimizer"),
                    ],
                    ignore_index=True,
                    sort=False,
                )
                policy_metrics_summary = pd.concat([policy_summary, excess_df, stockout_df], axis=1)
                TABLES.append(save_table(training_metrics_summary, "training_metrics_summary__mixed_context.csv"))
                TABLES.append(save_table(policy_metrics_summary, "policy_metrics_summary__mixed_context.csv"))
                TABLES.append(save_table(confusion_df, "trigger_confusion_matrix__mixed_context.csv"))
                RESULT = {
                    "notebook": NOTEBOOK_NAME,
                    "tables": TABLES,
                    "figures": FIGURES,
                    "findings": [
                        "The proposed policy reduces excess relative to the operational baseline while preserving the stockout guardrail.",
                        "The trigger stage respects the zero-order rule when purchase_trigger_flag equals zero.",
                        "Upstream, trigger and quantity metrics remain traceable through validation and test summaries.",
                    ],
                    "limitations": [
                        "Results come from simulation-based functional validation and must not be presented as plant-wide industrial proof.",
                    ],
                }
                print(json.dumps(RESULT, indent=2))
                """
            ),
            md(
                """
                ## Limitaciones

                Las metricas aqui mostradas no son garantia industrial. Reflejan comportamiento del pipeline en entorno reproducible y controlado, con variables internas sinteticas y contexto externo proxy.
                """
            ),
            md(
                """
                ## Concluson final

                El resultado defendible del pipeline mixed_context es una politica batch/offline de soporte a la decision con trigger, optimizador y simulacion comparada contra baseline, no un sistema de compra automatica ni una promesa de rendimiento industrial cerrado.
                """
            ),
        ]
    )
    return cells


def build_notebooks() -> dict[str, list[dict]]:
    return {
        "00_data_sources_audit.ipynb": nb00(),
        "01_raw_data_profile.ipynb": nb01(),
        "02_external_context_eda.ipynb": nb02(),
        "03_synthetic_plant_layer_eda.ipynb": nb03(),
        "04_feature_engineering_audit.ipynb": nb04(),
        "05_modeling_dataset_eda.ipynb": nb05(),
        "06_split_validation_and_leakage_audit.ipynb": nb06(),
        "07_training_and_policy_results_eda.ipynb": nb07(),
    }


def main() -> int:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebook_map = build_notebooks()
    for filename, cells in notebook_map.items():
        payload = notebook_payload(cells)
        (NOTEBOOK_DIR / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: len(cells) for name, cells in notebook_map.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
