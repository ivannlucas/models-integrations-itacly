"""Data processing package exports with lazy loading."""

from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, str] = {
    "run_cu04_external_pipeline": ".cu04_external_ingestion",
    "run_external_context_pipeline": ".cu04_external_ingestion",
    "apply_manual_feature_decisions": ".feature_report",
    "build_feature_catalog": ".feature_report",
    "build_feature_contract_from_config": ".feature_report",
    "build_feature_roles_metadata_from_config": ".feature_report",
    "build_feature_selection_export": ".feature_report",
    "write_feature_outputs": ".feature_report",
    "build_base_url": ".ine_ingestion",
    "build_default_dataset_config": ".ine_ingestion",
    "build_default_stable_table_fallbacks": ".ine_ingestion",
    "discover_operation": ".ine_ingestion",
    "discover_table": ".ine_ingestion",
    "download_table_data": ".ine_ingestion",
    "extract_id": ".ine_ingestion",
    "extract_name": ".ine_ingestion",
    "find_repo_root": ".ine_ingestion",
    "get_table_groups": ".ine_ingestion",
    "get_table_values": ".ine_ingestion",
    "get_tables_for_operation": ".ine_ingestion",
    "ine_get": ".ine_ingestion",
    "monthly_to_weekly_ffill": ".ine_ingestion",
    "normalize_ine_series": ".ine_ingestion",
    "normalize_text": ".ine_ingestion",
    "run_ine_ingestion_pipeline": ".ine_ingestion",
    "safe_slug": ".ine_ingestion",
    "search_operations": ".ine_ingestion",
    "prepare_modeling_frame": ".pipeline",
    "run_data_processing": ".pipeline",
    "build_synthetic_plant_dataset": ".synthetic_plant",
    "write_synthetic_plant_outputs": ".synthetic_plant",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
