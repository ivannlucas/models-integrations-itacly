"""Manufacturing-profile registry and runtime recipe helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .project import resolve_repo_path, slugify, to_repo_relative_path

RECIPE_CORE_FIELDS = [
    "profile_name",
    "recipe_profile",
    "product_family",
    "process_type",
    "formulation_class",
    "expected_yield",
    "expected_waste",
    "process_lead_time_days",
    "shelf_life_class",
    "priority_level",
    "cost_sensitivity",
]
RECIPE_SIMULATOR_FIELDS = [
    "production_multiplier",
    "coverage_adjustment_days",
    "lead_time_multiplier",
    "pressure_buffer_adjustment_days",
    "requirement_pressure_multiplier",
    "waste_stress_multiplier",
    "yield_stress_multiplier",
    "policy_target_coverage_adjustment_days",
    "policy_excess_penalty_multiplier",
    "policy_need_multiplier",
    "policy_max_order_multiplier",
    "selection_bias",
    "demand_sensitivity",
    "supply_stress_sensitivity",
    "seasonal_quarter_weights",
]
UNRESOLVED_SELECTION_MODE = "unresolved"
EXPLICIT_RECIPE_SELECTION_MODE = "single_recipe"
DEFAULT_RECIPE_SELECTION_MODE = "default_recipe_profile"
MIXED_CONTEXT_SELECTION_MODE = "mixed_context"
MIXED_CONTEXT_SCOPE_TOKEN = "mixed_context"


def _registry_path_from_config(config: dict[str, Any], recipe_config_path: str | Path | None) -> str | Path | None:
    if recipe_config_path:
        return recipe_config_path
    registry_cfg = config.get("manufacturing_profiles", {})
    if registry_cfg.get("registry_path"):
        return registry_cfg["registry_path"]
    return (
        config.get("synthetic_data", {})
        .get("simulation_parameters", {})
        .get("manufacturing_context", {})
        .get("registry_path")
    )


def _validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    normalised = dict(profile)
    missing = [field for field in [*RECIPE_CORE_FIELDS, *RECIPE_SIMULATOR_FIELDS] if field not in normalised]
    if missing:
        raise ValueError(f"Manufacturing profile is missing required fields: {missing}")

    for key in [
        "expected_yield",
        "expected_waste",
        "process_lead_time_days",
        "production_multiplier",
        "coverage_adjustment_days",
        "lead_time_multiplier",
        "pressure_buffer_adjustment_days",
        "requirement_pressure_multiplier",
        "waste_stress_multiplier",
        "yield_stress_multiplier",
        "policy_target_coverage_adjustment_days",
        "policy_excess_penalty_multiplier",
        "policy_need_multiplier",
        "policy_max_order_multiplier",
        "selection_bias",
        "demand_sensitivity",
        "supply_stress_sensitivity",
    ]:
        normalised[key] = float(normalised[key])

    seasonal_weights = dict(normalised.get("seasonal_quarter_weights", {}))
    if not seasonal_weights:
        raise ValueError(
            f"Manufacturing profile '{normalised['profile_name']}' must define seasonal_quarter_weights."
        )
    normalised["seasonal_quarter_weights"] = {
        str(key): float(value) for key, value in seasonal_weights.items()
    }
    return normalised


def _load_registry_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manufacturing profile registry not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Manufacturing profile registry must contain a mapping: {path}")
    return payload


def load_manufacturing_profile_registry(
    config: dict[str, Any],
    *,
    recipe_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the manufacturing-profile registry and inject it into config."""
    repo_root = Path(config["project"]["repo_root"])
    registry_cfg = config.setdefault("manufacturing_profiles", {})
    registry_cfg.setdefault("selection_key", "recipe_profile")
    registry_cfg.setdefault("profile_name_key", "profile_name")

    registry_path_value = _registry_path_from_config(config, recipe_config_path)
    inline_profiles = (
        config.get("synthetic_data", {})
        .get("simulation_parameters", {})
        .get("manufacturing_context", {})
        .get("profiles", [])
    )

    registry_metadata: dict[str, Any] = {}
    profiles_payload: list[dict[str, Any]]
    if registry_path_value:
        resolved_registry_path = resolve_repo_path(registry_path_value, repo_root)
        payload = _load_registry_payload(resolved_registry_path)
        raw_profiles = payload.get("profiles", [])
        if not isinstance(raw_profiles, list):
            raise ValueError(
                f"Manufacturing profile registry must expose a list under 'profiles': {resolved_registry_path}"
            )
        profiles_payload = [dict(profile) for profile in raw_profiles if isinstance(profile, dict)]
        registry_metadata = {key: value for key, value in payload.items() if key != "profiles"}
        registry_cfg["registry_path"] = to_repo_relative_path(resolved_registry_path, repo_root)
    else:
        if not isinstance(inline_profiles, list) or not inline_profiles:
            raise ValueError(
                "No manufacturing profile registry is configured. Define manufacturing_profiles.registry_path "
                "or provide synthetic_data.simulation_parameters.manufacturing_context.profiles."
            )
        profiles_payload = [dict(profile) for profile in inline_profiles if isinstance(profile, dict)]
        registry_cfg["registry_path"] = None

    profiles = [_validate_profile(profile) for profile in profiles_payload]
    if not profiles:
        raise ValueError("The manufacturing profile registry must contain at least one profile.")

    by_recipe: dict[str, dict[str, Any]] = {}
    by_profile_name: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        recipe_profile = str(profile["recipe_profile"])
        profile_name = str(profile["profile_name"])
        if recipe_profile in by_recipe:
            raise ValueError(f"Duplicated recipe_profile in manufacturing registry: {recipe_profile}")
        if profile_name in by_profile_name:
            raise ValueError(f"Duplicated profile_name in manufacturing registry: {profile_name}")
        by_recipe[recipe_profile] = profile
        by_profile_name[profile_name] = profile

    registry_cfg["registry_metadata"] = registry_metadata
    registry_cfg["profiles"] = deepcopy(profiles)
    registry_cfg["available_recipe_profiles"] = [profile["recipe_profile"] for profile in profiles]
    registry_cfg["available_profile_names"] = [profile["profile_name"] for profile in profiles]

    manufacturing_context_cfg = (
        config.setdefault("synthetic_data", {})
        .setdefault("simulation_parameters", {})
        .setdefault("manufacturing_context", {})
    )
    manufacturing_context_cfg["profiles"] = deepcopy(profiles)
    if registry_cfg.get("registry_path") is not None:
        manufacturing_context_cfg["registry_path"] = registry_cfg["registry_path"]

    config.setdefault("runtime", {})
    config["runtime"]["recipe_context"] = {
        "selection_mode": UNRESOLVED_SELECTION_MODE,
        "mode_resolution": UNRESOLVED_SELECTION_MODE,
        "scope_type": None,
        "scope_token": None,
        "output_suffix": None,
        "recipe_profile": None,
        "manufacturing_context_profile": None,
        "recipe_slug": None,
        "recipe_registry_path": registry_cfg.get("registry_path"),
        "available_recipe_profiles": list(registry_cfg["available_recipe_profiles"]),
        "available_manufacturing_profiles": list(registry_cfg["available_profile_names"]),
    }
    return config


def _recipe_suffix(recipe_slug: str) -> str:
    return f"__recipe_{recipe_slug}"


def _mixed_context_suffix() -> str:
    return f"__{MIXED_CONTEXT_SCOPE_TOKEN}"


def _scope_value(value: str | Path | None, suffix: str | None) -> str | None:
    if value is None or not suffix:
        return value if value is None else str(Path(value).as_posix())
    path = Path(value)
    if suffix in path.name:
        return path.as_posix()
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}{path.suffix}").as_posix()
    return path.parent.joinpath(f"{path.name}{suffix}").as_posix()


def scope_value_for_recipe(value: str | Path | None, recipe_slug: str | None) -> str | None:
    """Append a stable recipe suffix to a filename or directory-like path."""
    return _scope_value(value, _recipe_suffix(recipe_slug) if recipe_slug else None)


def scope_value_for_runtime(value: str | Path | None, runtime_context: dict[str, Any]) -> str | None:
    """Append the resolved runtime suffix to a filename or directory-like path."""
    return _scope_value(value, runtime_context.get("output_suffix"))


def current_recipe_context(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("runtime", {}).get("recipe_context", {}))


def current_recipe_profile(config: dict[str, Any]) -> str | None:
    return current_recipe_context(config).get("recipe_profile")


def current_recipe_slug(config: dict[str, Any]) -> str | None:
    return current_recipe_context(config).get("recipe_slug")


def current_runtime_scope_token(config: dict[str, Any]) -> str | None:
    return current_recipe_context(config).get("scope_token")


def runtime_stage_prefix(stage_name: str, config: dict[str, Any]) -> str:
    runtime_context = current_recipe_context(config)
    scope_type = runtime_context.get("scope_type")
    scope_token = runtime_context.get("scope_token")
    recipe_slug = runtime_context.get("recipe_slug")
    if scope_type == "recipe" and recipe_slug:
        return f"{stage_name}_recipe_{recipe_slug}"
    if not scope_token:
        return stage_name
    return f"{stage_name}_{scope_token}"


def filter_frame_to_recipe(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    stage_name: str,
    recipe_column: str = "recipe_profile",
    logger=None,
    reset_index: bool = True,
) -> pd.DataFrame:
    """Filter a frame to the selected recipe_profile when runtime selection is active."""
    selected_recipe = current_recipe_profile(config)
    if not selected_recipe or recipe_column not in df.columns:
        return df

    filtered = df[df[recipe_column].astype(str) == str(selected_recipe)].copy()
    if filtered.empty:
        raise ValueError(
            f"Stage '{stage_name}' selected recipe_profile='{selected_recipe}', but the active dataframe "
            f"contains no rows for that recipe."
        )
    if logger is not None and len(filtered) != len(df):
        logger.info(
            "Applied runtime recipe filter stage=%s recipe_profile=%s rows_before=%s rows_after=%s",
            stage_name,
            selected_recipe,
            len(df),
            len(filtered),
        )
    return filtered.reset_index(drop=True) if reset_index else filtered


def _recipe_runtime_context(
    registry_cfg: dict[str, Any],
    *,
    recipe_profile: str,
    profile: dict[str, Any],
    selection_mode: str,
    mode_resolution: str,
) -> dict[str, Any]:
    recipe_slug = slugify(recipe_profile)
    return {
        "selection_mode": selection_mode,
        "mode_resolution": mode_resolution,
        "scope_type": "recipe",
        "scope_token": f"recipe_{recipe_slug}",
        "output_suffix": _recipe_suffix(recipe_slug),
        "recipe_profile": recipe_profile,
        "manufacturing_context_profile": str(profile["profile_name"]),
        "recipe_slug": recipe_slug,
        "recipe_registry_path": registry_cfg.get("registry_path"),
        "available_recipe_profiles": list(registry_cfg.get("available_recipe_profiles", [])),
        "available_manufacturing_profiles": list(registry_cfg.get("available_profile_names", [])),
    }


def _mixed_runtime_context(registry_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_mode": MIXED_CONTEXT_SELECTION_MODE,
        "mode_resolution": "cli_mixed_context_flag",
        "scope_type": MIXED_CONTEXT_SELECTION_MODE,
        "scope_token": MIXED_CONTEXT_SCOPE_TOKEN,
        "output_suffix": _mixed_context_suffix(),
        "recipe_profile": None,
        "manufacturing_context_profile": None,
        "recipe_slug": None,
        "recipe_registry_path": registry_cfg.get("registry_path"),
        "available_recipe_profiles": list(registry_cfg.get("available_recipe_profiles", [])),
        "available_manufacturing_profiles": list(registry_cfg.get("available_profile_names", [])),
    }


def _scoped_stage_config(
    config: dict[str, Any],
    *,
    runtime_context: dict[str, Any],
    active_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scoped = deepcopy(config)

    scoped_path_fields = {
        "paths": ["processed_dataset_path", "processed_metadata_path", "splits_dir"],
        "synthetic_data": [
            "output_dataset_path",
            "column_lineage_path",
            "simulation_parameters_path",
            "environment_summary_path",
        ],
        "feature_selection": [
            "feature_catalog_path",
            "feature_contract_path",
            "feature_roles_metadata_path",
            "feature_config_export_path",
            "prepared_dataset_path",
        ],
        "policy_simulation": [
            "input_dataset_path",
        ],
    }
    for section_name, keys in scoped_path_fields.items():
        section = scoped.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for key in keys:
            if section.get(key):
                section[key] = scope_value_for_runtime(section[key], runtime_context)

    # The prepared dataset becomes the default input for recipe-scoped prediction and policy.
    prepared_dataset_path = scoped.get("feature_selection", {}).get("prepared_dataset_path")
    if prepared_dataset_path:
        scoped.setdefault("prediction", {})["input_path"] = prepared_dataset_path
        scoped.setdefault("policy_simulation", {})["input_dataset_path"] = prepared_dataset_path

    if scoped.get("prediction", {}).get("output_filename"):
        scoped["prediction"]["output_filename"] = scope_value_for_runtime(
            scoped["prediction"]["output_filename"],
            runtime_context,
        )

    scoped_name_fields = {
        "training": ["comparison_summary_csv_name", "comparison_summary_json_name"],
        "neuroevolution": ["comparison_summary_csv_name", "comparison_summary_json_name"],
        "policy_simulation": ["summary_csv_name", "summary_json_name"],
        "get_stats": ["summary_csv_name", "summary_json_name"],
    }
    for section_name, keys in scoped_name_fields.items():
        section = scoped.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for key in keys:
            if section.get(key):
                section[key] = scope_value_for_runtime(section[key], runtime_context)

    reference_runs_cfg = scoped.get("policy_simulation", {}).get("reference_runs", {})
    if isinstance(reference_runs_cfg, dict):
        for key in ["baseline_summary_json", "neuro_summary_json"]:
            if reference_runs_cfg.get(key):
                reference_runs_cfg[key] = scope_value_for_runtime(reference_runs_cfg[key], runtime_context)

    comparison_name = scoped.get("training", {}).get("comparison_name")
    if comparison_name and runtime_context.get("scope_token"):
        scoped["training"]["comparison_name"] = f"{comparison_name}_{runtime_context['scope_token']}"

    policy_comparison_name = scoped.get("policy_simulation", {}).get("comparison_name")
    if policy_comparison_name and runtime_context.get("scope_token"):
        scoped["policy_simulation"]["comparison_name"] = f"{policy_comparison_name}_{runtime_context['scope_token']}"

    registry_cfg = scoped.setdefault("manufacturing_profiles", {})
    manufacturing_context_cfg = (
        scoped.setdefault("synthetic_data", {})
        .setdefault("simulation_parameters", {})
        .setdefault("manufacturing_context", {})
    )
    if active_profile is not None:
        manufacturing_context_cfg["profiles"] = [deepcopy(active_profile)]
        registry_cfg["active_profile"] = deepcopy(active_profile)
    else:
        registry_cfg.pop("active_profile", None)
    scoped.setdefault("runtime", {})
    scoped["runtime"]["recipe_context"] = dict(runtime_context)
    return scoped


def build_recipe_runtime_configs(
    config: dict[str, Any],
    *,
    recipe_profile: str | None = None,
    all_recipes: bool = False,
    mixed_context: bool = False,
) -> list[dict[str, Any]]:
    """Expand the base config into one or multiple recipe-scoped runtime configs."""
    if recipe_profile and all_recipes:
        raise ValueError("Use either --recipe-profile or --all-recipes, not both.")
    if mixed_context and (recipe_profile or all_recipes):
        raise ValueError("Use --mixed-context on its own, not together with --recipe-profile or --all-recipes.")

    registry_cfg = config.get("manufacturing_profiles", {})
    profiles = [dict(profile) for profile in registry_cfg.get("profiles", []) if isinstance(profile, dict)]
    if not profiles:
        return [config]

    recipe_lookup = {
        str(profile["recipe_profile"]): profile
        for profile in profiles
    }
    name_lookup = {
        str(profile["profile_name"]): profile
        for profile in profiles
    }

    if all_recipes:
        return [
            _scoped_stage_config(
                config,
                runtime_context=_recipe_runtime_context(
                    registry_cfg,
                    recipe_profile=str(profile["recipe_profile"]),
                    profile=profile,
                    selection_mode=EXPLICIT_RECIPE_SELECTION_MODE,
                    mode_resolution="cli_recipe_profile",
                ),
                active_profile=profile,
            )
            for profile in profiles
        ]

    if mixed_context:
        return [
            _scoped_stage_config(
                config,
                runtime_context=_mixed_runtime_context(registry_cfg),
            )
        ]

    if recipe_profile:
        selected_profile = recipe_lookup.get(str(recipe_profile)) or name_lookup.get(str(recipe_profile))
        if selected_profile is None:
            available = ", ".join(sorted(recipe_lookup))
            raise ValueError(
                f"Unknown recipe_profile='{recipe_profile}'. Available recipe profiles: {available}"
            )
        return [
            _scoped_stage_config(
                config,
                runtime_context=_recipe_runtime_context(
                    registry_cfg,
                    recipe_profile=str(selected_profile["recipe_profile"]),
                    profile=selected_profile,
                    selection_mode=EXPLICIT_RECIPE_SELECTION_MODE,
                    mode_resolution="cli_recipe_profile",
                ),
                active_profile=selected_profile,
            )
        ]

    default_recipe_profile = registry_cfg.get("default_recipe_profile")
    if default_recipe_profile:
        selected_profile = recipe_lookup.get(str(default_recipe_profile)) or name_lookup.get(str(default_recipe_profile))
        if selected_profile is None:
            available = ", ".join(sorted(recipe_lookup))
            raise ValueError(
                "manufacturing_profiles.default_recipe_profile is invalid. "
                f"Received '{default_recipe_profile}'. Available recipe profiles: {available}"
            )
        return [
            _scoped_stage_config(
                config,
                runtime_context=_recipe_runtime_context(
                    registry_cfg,
                    recipe_profile=str(selected_profile["recipe_profile"]),
                    profile=selected_profile,
                    selection_mode=DEFAULT_RECIPE_SELECTION_MODE,
                    mode_resolution="config.default_recipe_profile",
                ),
                active_profile=selected_profile,
            )
        ]

    raise ValueError(
        "No runtime manufacturing context was selected. Use --recipe-profile <recipe_profile>, "
        "--mixed-context, or define manufacturing_profiles.default_recipe_profile in config/config.yaml."
    )


def ensure_runtime_context_resolved(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve default runtime scope when a stage is called directly without src.main."""
    runtime_context = current_recipe_context(config)
    if runtime_context.get("selection_mode") and runtime_context.get("selection_mode") != UNRESOLVED_SELECTION_MODE:
        return config
    return build_recipe_runtime_configs(config)[0]
