"""Shared utilities for the project pipeline."""

from .config import load_config
from .io import find_latest_file, read_json, read_tabular, write_json
from .logging_utils import setup_stage_logger
from .manufacturing_profiles import (
    build_recipe_runtime_configs,
    current_runtime_scope_token,
    current_recipe_context,
    current_recipe_profile,
    current_recipe_slug,
    ensure_runtime_context_resolved,
    filter_frame_to_recipe,
    load_manufacturing_profile_registry,
    runtime_stage_prefix,
    scope_value_for_runtime,
    scope_value_for_recipe,
)
from .model_selection import (
    build_best_by_target,
    canonical_target_column,
    resolve_reference_record,
    select_best_record,
    select_preferred_record,
    selection_policy_description,
)
from .project import ensure_directory, find_repo_root, make_run_id, resolve_repo_path, to_repo_relative_path, utc_timestamp

__all__ = [
    "build_best_by_target",
    "build_recipe_runtime_configs",
    "canonical_target_column",
    "current_runtime_scope_token",
    "current_recipe_context",
    "current_recipe_profile",
    "current_recipe_slug",
    "ensure_runtime_context_resolved",
    "ensure_directory",
    "filter_frame_to_recipe",
    "find_latest_file",
    "find_repo_root",
    "load_config",
    "load_manufacturing_profile_registry",
    "make_run_id",
    "read_json",
    "read_tabular",
    "resolve_reference_record",
    "resolve_repo_path",
    "runtime_stage_prefix",
    "select_best_record",
    "select_preferred_record",
    "selection_policy_description",
    "scope_value_for_runtime",
    "scope_value_for_recipe",
    "setup_stage_logger",
    "to_repo_relative_path",
    "utc_timestamp",
    "write_json",
]
