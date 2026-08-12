from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.reproducibility.hashes import describe_existing_files, sha256_file
from src.reproducibility.runtime import ensure_optional_dependency, official_paths
from src.utils import ensure_directory, write_json


def _mixed_context_reference_date(config: dict[str, Any]) -> pd.Timestamp:
    reference_value = (
        config.get("project", {}).get("reference_date")
        or config.get("data", {}).get("mixed_context_end_date")
        or config.get("data_processing", {}).get("source_refresh", {}).get("end_date")
    )
    if not reference_value or str(reference_value).strip().lower() == "today":
        return pd.Timestamp.utcnow().tz_localize(None).normalize()

    reference_date = pd.Timestamp(reference_value)
    if reference_date.tzinfo is not None:
        reference_date = reference_date.tz_convert(None)
    return reference_date.normalize()


def _prepare_legacy_raw_cache(repo_root: Path) -> Path:
    cache_root = ensure_directory(repo_root / "data" / "interim" / "external" / "legacy_raw_cache")
    mapping = {
        repo_root / "data" / "raw" / "external" / "INE_CPI": cache_root / "ine_ipc",
        repo_root / "data" / "raw" / "external" / "MAPA_SLAUGHTER_MAPA": cache_root / "mapa_sacrificio",
        repo_root / "data" / "raw" / "external" / "MAPA_PRICES_OM": cache_root / "mapa_precios_om",
    }
    for source_dir, target_dir in mapping.items():
        ensure_directory(target_dir)
        for source_file in source_dir.glob("*"):
            if source_file.is_file() and source_file.name != "source_manifest.json":
                shutil.copy2(source_file, target_dir / source_file.name)
    return cache_root


def build_external_long(config: dict[str, Any], *, force_download: bool = False) -> dict[str, Any]:
    repo_root = Path(config["project"]["repo_root"])
    ensure_optional_dependency("openpyxl", repo_root_path=repo_root)
    ensure_optional_dependency("requests", repo_root_path=repo_root)
    import requests

    from src.data_processing.cu04_external_ingestion import (
        ingest_ine_cpi,
        ingest_mapa_prices_om,
        ingest_mapa_slaughter,
    )

    raw_cache = _prepare_legacy_raw_cache(repo_root)
    output_paths = official_paths(config)
    interim_dir = ensure_directory(repo_root / "data" / "interim" / "external")

    start_month = pd.Timestamp("2004-01-01")
    end_raw = _mixed_context_reference_date(config)
    end_month = end_raw.to_period("M").to_timestamp(how="start")
    registry: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "CU28 Mixed Context Reproducibility/1.0"})

    cpi_long, demand_monthly = ingest_ine_cpi(
        session,
        raw_dir=raw_cache,
        start_month=start_month,
        end_month=end_month,
        force_download=force_download,
        retries=2,
        timeout=60,
        backoff=1.5,
        cpi_codes=["01.1.2.3", "01.1.2.5"],
        registry=registry,
    )
    slaughter_long, supply_monthly = ingest_mapa_slaughter(
        session,
        raw_dir=raw_cache,
        start_month=start_month,
        end_month=end_month,
        force_download=force_download,
        retries=2,
        timeout=60,
        backoff=1.5,
        species_keywords=["bov", "porc", "ovin", "capr", "aviar", "cun"],
        registry=registry,
    )
    prices_long, purchase_price_weekly = ingest_mapa_prices_om(
        session,
        raw_dir=raw_cache,
        start_month=start_month,
        end_raw=end_raw,
        force_download=force_download,
        retries=2,
        timeout=60,
        backoff=1.5,
        price_keywords=["vacuno", "ternera", "porcino", "cerdo", "pollo", "pavo", "elaborados"],
        registry=registry,
    )

    if purchase_price_weekly.empty:
        fallback_index = pd.date_range(
            start=start_month - pd.to_timedelta(start_month.weekday(), unit="D"),
            end=end_raw - pd.to_timedelta(end_raw.weekday(), unit="D"),
            freq="W-MON",
        )
        purchase_price_weekly = pd.DataFrame({"date": fallback_index, "value": 100.0})
        prices_long = pd.DataFrame(
            {
                "date": fallback_index,
                "source": "MAPA",
                "dataset": "PRICES_OM",
                "subseries": "fallback_constant",
                "value": 100.0,
                "unit": "index",
                "notes": "fallback_no_prices_extracted",
            }
        )

    external_long = pd.concat([frame for frame in [cpi_long, slaughter_long, prices_long] if not frame.empty], ignore_index=True)
    external_long["date"] = pd.to_datetime(external_long["date"], errors="coerce")
    external_long = external_long.sort_values(["date", "source", "dataset", "subseries"]).reset_index(drop=True)
    external_long.to_csv(output_paths["external_long"], index=False)

    demand_monthly_path = interim_dir / "demand_index_monthly__mixed_context.csv"
    supply_monthly_path = interim_dir / "supply_index_monthly__mixed_context.csv"
    purchase_price_weekly_path = interim_dir / "purchase_price_weekly__mixed_context.csv"
    download_registry_path = output_paths["context_weekly"].parent / "download_registry.json"

    demand_monthly.to_csv(demand_monthly_path, index=False)
    supply_monthly.to_csv(supply_monthly_path, index=False)
    purchase_price_weekly.to_csv(purchase_price_weekly_path, index=False)
    write_json(download_registry_path, registry)

    metadata_path = interim_dir / "build_external_long__mixed_context.json"
    metadata = {
        "scope": "mixed_context",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_date": str(end_raw.date()),
        "inputs": describe_existing_files(
            [
                repo_root / "data" / "raw" / "external" / "INE_CPI" / "76128.csv",
                *sorted((repo_root / "data" / "raw" / "external" / "MAPA_SLAUGHTER_MAPA").glob("*.xlsx")),
                *sorted((repo_root / "data" / "raw" / "external" / "MAPA_PRICES_OM").glob("*.xlsx")),
            ],
            repo_root=repo_root,
        ),
        "outputs": describe_existing_files(
            [output_paths["external_long"], demand_monthly_path, supply_monthly_path, purchase_price_weekly_path, download_registry_path],
            repo_root=repo_root,
        ),
        "row_count": int(len(external_long)),
        "columns": external_long.columns.tolist(),
        "date_min": str(external_long["date"].min().date()) if not external_long.empty else None,
        "date_max": str(external_long["date"].max().date()) if not external_long.empty else None,
    }
    write_json(metadata_path, metadata)
    return {
        "external_long_path": str(output_paths["external_long"]),
        "metadata_path": str(metadata_path),
        "download_registry_path": str(download_registry_path),
    }
