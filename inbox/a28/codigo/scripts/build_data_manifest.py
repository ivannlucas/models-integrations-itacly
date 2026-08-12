from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data_blob_manifest.json"
SCOPE = "mixed_context"

OFFICIAL_PROCESSED_FILES = [
    "data/processed/external/context/external_long.csv",
    "data/processed/external/context/context_weekly_for_simulation.csv",
    "data/processed/external/context/context_proxy_limitations.json",
    "data/processed/external/context/download_registry.json",
    "data/processed/synthetic/plant/synthetic_plant_layer__mixed_context.csv",
    "data/processed/synthetic/plant/synthetic_plant_metadata__mixed_context.json",
    "data/processed/baseline/feature_engineering_modeling__mixed_context.csv",
    "data/processed/baseline/modeling_metadata__mixed_context.json",
    "data/processed/baseline/modeling_weekly__mixed_context.csv",
]

OFFICIAL_SPLIT_DIRS = [
    "data/splits/baseline/default__mixed_context",
]

OFFICIAL_METRIC_FILES = [
    "models/metrics/summary/baseline_comparison_latest__mixed_context.csv",
    "models/metrics/summary/baseline_comparison_latest__mixed_context.json",
    "models/metrics/summary/neuroevolution_comparison_latest__mixed_context.csv",
    "models/metrics/summary/neuroevolution_comparison_latest__mixed_context.json",
    "models/metrics/summary/trigger_metrics_latest__mixed_context.json",
    "models/metrics/summary/quantity_optimizer_latest__mixed_context.json",
    "models/metrics/summary/quantity_optimizer_baseline_comparison_latest__mixed_context.csv",
    "models/metrics/summary/quantity_optimizer_baseline_comparison_latest__mixed_context.json",
    "models/metrics/summary/policy_simulation_latest__mixed_context.csv",
    "models/metrics/summary/policy_simulation_latest__mixed_context.json",
    "models/metrics/summary/metrics_summary__mixed_context.csv",
    "models/metrics/summary/metrics_summary__mixed_context.json",
]

OFFICIAL_MODEL_FILES = [
    "models/artifacts/upstream_predictor_latest__mixed_context.pkl",
    "models/artifacts/purchase_trigger_latest__mixed_context.pkl",
    "models/artifacts/quantity_optimizer_latest__mixed_context.pkl",
    "models/artifacts/model_manifest__mixed_context.json",
]

OFFICIAL_PREDICTION_FILES = [
    "data/predictions/predictions_latest__mixed_context.csv",
]

OFFICIAL_REPRODUCIBILITY_FILES = [
    "reproducibility_manifest__mixed_context.json",
    "data/raw/external/raw_manifest__mixed_context.json",
]

OFFICIAL_DOCS = [
    "README.md",
    "DELIVERY_README_CU28.md",
    "docs/README.md",
    "docs/reproducibility.md",
    "docs/repository_outputs.md",
    "docs/data_lineage.md",
    "docs/data_sources_registry.md",
    "docs/data_blob_inventory.md",
    "docs/etl_pipeline.md",
    "docs/feature_engineering.md",
    "docs/input_contract.md",
    "docs/output_contract.md",
    "docs/leakage_policy.md",
    "docs/model_card_cu28.md",
    "docs/platform_usage.md",
    "docs/simulation_assumptions.md",
    "docs/simulation_data_basis.md",
]

DEMO_FILES = [
    "data/demo/customer_upload_example.csv",
]

OFFICIAL_CONFIG_FILES = [
    "config/config.yaml",
    "config/platform_config.yaml",
    "config/manufacturing_profiles.yaml",
]

RAW_FILE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "INE_CPI": {
        "76128.csv": "INE CPI table 76128 cached CSV snapshot used as contextual inflation proxy.",
    },
    "MAPA_SLAUGHTER_MAPA": {
        "boletin--diciembre-2025-web.xlsx": "MAPA monthly slaughter bulletin snapshot used for contextual supply proxy extraction.",
        "sacrificio_anual_2024.xlsx": "MAPA definitive slaughter census 2024 snapshot used for contextual supply traceability.",
        "sacrificio_exhaustivo_2023.xlsx": "MAPA definitive slaughter census 2023 snapshot used for contextual supply traceability.",
        "web-mapa-1-junio-24-25-web-0.xlsx": "MAPA slaughter monthly snapshot for 2024-2025 contextual supply traceability.",
        "web-mapa-2-diciembre-24-25-web.xlsx": "MAPA slaughter monthly snapshot for 2024-2025 contextual supply traceability.",
    },
    "MAPA_PRICES_OM": {
        "s502025rv0.xlsx": "Cached MAPA origen-destino prices workbook preserved only as traced fallback evidence.",
    },
}

SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "INE_CPI": {
        "source_name": "Indice de Precios de Consumo (IPC), tabla 76128",
        "organization": "Instituto Nacional de Estadistica (INE)",
        "source_type": "official_open_data_csv",
        "evidence_status": "active",
        "official_url": "https://www.ine.es/dyngs/DAB/index.htm?cid=1722",
        "download_url_or_endpoint": "https://www.ine.es/jaxiT3/files/t/csv_bdsc/76128.csv",
        "access_date": "2026-05-18",
        "license_or_terms_url": "https://www.ine.es/dyngs/AYU/en/index.htm?cid=125",
        "redistribution_allowed": "yes_with_attribution_cc_by_4_0",
        "retrieval_method": "cached_snapshot",
        "raw_dir": "data/raw/external/INE_CPI",
        "derived_artifacts": [
            "data/processed/external/context/external_long.csv",
            "data/processed/external/context/context_weekly_for_simulation.csv",
            "data/processed/baseline/feature_engineering_modeling__mixed_context.csv",
        ],
        "role": "Contextual inflation and price-pressure proxy for the official mixed_context route.",
        "limitations": "Proxy context only. It is not a plant-level meat raw-material purchase series and does not contain internal factory variables.",
    },
    "MAPA_SLAUGHTER_MAPA": {
        "source_name": "Encuesta de sacrificio de ganado",
        "organization": "Ministerio de Agricultura, Pesca y Alimentacion (MAPA)",
        "source_type": "official_statistics_spreadsheet_bundle",
        "evidence_status": "active",
        "official_url": "https://www.mapa.gob.es/es/estadistica/temas/estadisticas-agrarias/ganaderia/encuestas-sacrificio-ganado/",
        "download_url_or_endpoint": "https://www.mapa.gob.es/es/estadistica/temas/estadisticas-agrarias/ganaderia/encuestas-sacrificio-ganado/",
        "access_date": "2026-05-18",
        "license_or_terms_url": "https://www.mapa.gob.es/es/atencion-al-ciudadano/aviso-legal",
        "redistribution_allowed": "yes_with_source_citation_unless_third_party_rights_apply",
        "retrieval_method": "cached_snapshot",
        "raw_dir": "data/raw/external/MAPA_SLAUGHTER_MAPA",
        "derived_artifacts": [
            "data/processed/external/context/external_long.csv",
            "data/processed/external/context/context_weekly_for_simulation.csv",
            "data/processed/baseline/feature_engineering_modeling__mixed_context.csv",
        ],
        "role": "Contextual slaughter/supply proxy for the official mixed_context route.",
        "limitations": "Used as macro proxy only. It is not an observed weekly purchase ledger for a meat plant and remains external contextual evidence.",
    },
    "MAPA_PRICES_OM": {
        "source_name": "Sistema de precios origen-destino (Observatorio de la Cadena)",
        "organization": "Ministerio de Agricultura, Pesca y Alimentacion (MAPA)",
        "source_type": "official_spreadsheet_snapshot",
        "evidence_status": "traced",
        "official_url": "https://servicio.mapa.gob.es/es/alimentacion/temas/observatorio-cadena/cadenas-valor/sistema-de-precios-om",
        "download_url_or_endpoint": "https://servicio.mapa.gob.es/dam/mapa/contenido/alimentacion/servicios/observatorio-de-precios-de-los-alimentos/sistema-de-informacion-de-precios-origen---destino/s132026rv0.xlsx",
        "access_date": "2026-05-18",
        "license_or_terms_url": "https://www.mapa.gob.es/es/atencion-al-ciudadano/aviso-legal",
        "redistribution_allowed": "yes_with_source_citation_unless_third_party_rights_apply",
        "retrieval_method": "cached_snapshot",
        "raw_dir": "data/raw/external/MAPA_PRICES_OM",
        "derived_artifacts": [
            "data/processed/external/context/context_weekly_for_simulation.csv",
            "data/processed/external/context/context_proxy_limitations.json",
        ],
        "role": "Traced fallback constant reference only; not an active weekly price feed in the defended route.",
        "limitations": "The current processed route uses purchase_price_index as fallback_constant=100. This source must not be defended as an active weekly signal.",
    },
}


def repo_path(relative_path: str | Path) -> Path:
    return REPO_ROOT / Path(relative_path)


def to_posix(path: str | Path) -> str:
    return Path(path).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_file_entry(path: Path, description: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if description:
        entry["description"] = description
    return entry


def list_files(relative_dir: str | Path, *, exclude_names: Iterable[str] | None = None) -> list[Path]:
    directory = repo_path(relative_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Missing required directory: {directory}")
    excluded = set(exclude_names or [])
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.name not in excluded)


def ensure_required_paths(paths: Iterable[str]) -> None:
    missing = [path for path in paths if not repo_path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            cwd=REPO_ROOT,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def build_source_manifest(source_id: str, *, write: bool = True) -> dict[str, Any]:
    spec = SOURCE_SPECS[source_id]
    raw_dir = repo_path(spec["raw_dir"])
    raw_files = list_files(spec["raw_dir"], exclude_names={"source_manifest.json"})
    if spec["evidence_status"] in {"active", "traced"} and not raw_files:
        raise FileNotFoundError(f"Source {source_id} has no raw snapshot files under {raw_dir}")

    descriptions = RAW_FILE_DESCRIPTIONS.get(source_id, {})
    manifest = {
        "source_id": source_id,
        "source_name": spec["source_name"],
        "organization": spec["organization"],
        "source_type": spec["source_type"],
        "evidence_status": spec["evidence_status"],
        "official_url": spec["official_url"],
        "download_url_or_endpoint": spec["download_url_or_endpoint"],
        "access_date": spec["access_date"],
        "license_or_terms_url": spec["license_or_terms_url"],
        "redistribution_allowed": spec["redistribution_allowed"],
        "retrieval_method": spec["retrieval_method"],
        "role": spec["role"],
        "raw_files": [
            collect_file_entry(path, description=descriptions.get(path.name, f"Cached raw snapshot for {source_id}."))
            for path in raw_files
        ],
        "derived_artifacts": spec["derived_artifacts"],
        "limitations": spec["limitations"],
    }

    if write:
        output_path = raw_dir / "source_manifest.json"
        output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _collect_entries_for_paths(paths: Iterable[str]) -> list[dict[str, Any]]:
    ensure_required_paths(paths)
    return [collect_file_entry(repo_path(path)) for path in sorted(paths)]


def _collect_entries_for_directories(directories: Iterable[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for directory in directories:
        files = list_files(directory)
        if not files:
            raise FileNotFoundError(f"Directory declared as official but empty: {directory}")
        entries.extend(collect_file_entry(path) for path in files)
    return entries


def build_manifest(
    *,
    output_path: Path | None = DEFAULT_MANIFEST_PATH,
    write_source_manifests: bool = True,
    generation_command: str | None = None,
) -> dict[str, Any]:
    source_manifests = [build_source_manifest(source_id, write=write_source_manifests) for source_id in SOURCE_SPECS]

    processed_entries = _collect_entries_for_paths(OFFICIAL_PROCESSED_FILES)
    split_entries = _collect_entries_for_directories(OFFICIAL_SPLIT_DIRS)
    metric_entries = _collect_entries_for_paths(OFFICIAL_METRIC_FILES)
    model_entries = _collect_entries_for_paths(OFFICIAL_MODEL_FILES)
    prediction_entries = _collect_entries_for_paths(OFFICIAL_PREDICTION_FILES)
    reproducibility_entries = _collect_entries_for_paths(OFFICIAL_REPRODUCIBILITY_FILES)
    docs_entries = _collect_entries_for_paths(OFFICIAL_DOCS)
    config_entries = _collect_entries_for_paths(OFFICIAL_CONFIG_FILES)
    demo_entries = _collect_entries_for_paths(DEMO_FILES)
    raw_entries = []
    for source_manifest in source_manifests:
        raw_entries.extend(source_manifest["raw_files"])

    source_manifest_paths = [f"{SOURCE_SPECS[source_id]['raw_dir']}/source_manifest.json" for source_id in SOURCE_SPECS]
    source_manifest_entries = _collect_entries_for_paths(source_manifest_paths)

    manifest = {
        "schema_version": 1,
        "scope": SCOPE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "generation_command": generation_command or "python scripts/build_data_manifest.py --output data_blob_manifest.json",
        "configuration": {
            "paths": OFFICIAL_CONFIG_FILES,
            "entries": config_entries,
        },
        "generated_outputs": {
            "raw": [
                SOURCE_SPECS[source_id]["raw_dir"]
                for source_id, spec in SOURCE_SPECS.items()
                if spec["evidence_status"] in {"active", "traced"}
            ],
            "processed": OFFICIAL_PROCESSED_FILES,
            "splits": OFFICIAL_SPLIT_DIRS,
            "metrics": OFFICIAL_METRIC_FILES,
            "models": OFFICIAL_MODEL_FILES,
            "predictions": OFFICIAL_PREDICTION_FILES,
            "reproducibility": OFFICIAL_REPRODUCIBILITY_FILES,
            "docs_snapshot": OFFICIAL_DOCS,
            "demo": DEMO_FILES,
        },
        "sources": source_manifests,
        "required_paths": {
            "raw": [
                SOURCE_SPECS[source_id]["raw_dir"]
                for source_id, spec in SOURCE_SPECS.items()
                if spec["evidence_status"] in {"active", "traced"}
            ],
            "processed": OFFICIAL_PROCESSED_FILES,
            "splits": OFFICIAL_SPLIT_DIRS,
            "metrics": OFFICIAL_METRIC_FILES,
            "models": OFFICIAL_MODEL_FILES,
            "predictions": OFFICIAL_PREDICTION_FILES,
            "reproducibility": OFFICIAL_REPRODUCIBILITY_FILES,
            "docs_snapshot": OFFICIAL_DOCS,
            "demo": DEMO_FILES,
        },
        "files": {
            "raw": raw_entries,
            "processed": processed_entries,
            "splits": split_entries,
            "metrics": metric_entries,
            "models": model_entries,
            "predictions": prediction_entries,
            "reproducibility": reproducibility_entries,
            "docs_snapshot": docs_entries,
            "configuration": config_entries,
            "demo": demo_entries,
            "source_manifests": source_manifest_entries,
        },
    }

    if output_path is not None:
        output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the auditable CU28 mixed_context data-blob manifest.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path where data_blob_manifest.json will be written.",
    )
    args = parser.parse_args(argv)
    manifest = build_manifest(
        output_path=Path(args.output),
        write_source_manifests=True,
        generation_command=f"python scripts/build_data_manifest.py --output {to_posix(args.output)}",
    )
    print(json.dumps({"manifest_path": str(Path(args.output)), "scope": manifest["scope"], "commit": manifest["commit"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
