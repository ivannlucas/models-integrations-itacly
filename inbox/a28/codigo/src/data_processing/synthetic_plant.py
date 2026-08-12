"""Synthetic plant-layer generation with explicit manufacturing context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import ensure_directory, resolve_repo_path, to_repo_relative_path, write_json

REQUIRED_CONTEXT_COLUMNS = ["date", "demand_index", "supply_index", "purchase_price_index"]
MANUFACTURING_CONTEXT_PUBLIC_COLUMNS = [
    "manufacturing_context_profile",
    "product_family",
    "process_type",
    "recipe_profile",
    "formulation_class",
    "expected_yield",
    "expected_waste",
    "process_lead_time_days",
    "shelf_life_class",
    "priority_level",
    "cost_sensitivity",
]
REQUIRED_PROCUREMENT_TARGET_PARAMETERS = (
    "canonical_variant",
    "forward_requirement_weights",
    "available_coverage_current_weight",
    "available_coverage_trailing_weight",
    "pressure_buffer_base_days",
    "pressure_lead_time_weight",
    "pressure_supply_stress_weight",
    "pressure_growth_weight",
    "pressure_ratio_offset",
    "pressure_scaling_divisor",
    "pressure_context_days_weight",
    "pressure_context_multiplier_weight",
    "pressure_snapshot_blend_weight",
    "pressure_coverage_gap_blend_weight",
    "coverage_gap_buffer_base_days",
    "coverage_gap_lead_time_weight",
    "coverage_gap_supply_stress_weight",
    "coverage_gap_growth_weight",
    "coverage_gap_future_requirement_weight",
    "coverage_gap_trailing_requirement_weight",
    "hybrid_buffer_base_days",
    "hybrid_lead_time_weight",
    "hybrid_supply_stress_weight",
    "hybrid_growth_weight",
    "hybrid_ratio_offset",
    "hybrid_scaling_divisor",
    "hybrid_growth_softplus_scale_tons",
    "hybrid_growth_boost_scale",
    "hybrid_growth_boost_weight",
    "hybrid_snapshot_penalty_weight",
)
DEFAULT_MANUFACTURING_CONTEXT = {
    "context_window_weeks": 4,
    "assignment_noise_std": 0.03,
    "persistence_bonus": 0.08,
    "profiles": [],
}
PRIORITY_LEVEL_SCORES = {"planned_campaign": 0.45, "standard_service": 0.65, "service_critical": 0.95}
COST_SENSITIVITY_SCORES = {"high": 0.85, "medium": 0.55, "low": 0.25}
SHELF_LIFE_FACTORS = {"short": 0.70, "medium": 1.00, "long": 1.35}


def _portable_path(path: str | Path, repo_root: Path) -> str:
    return to_repo_relative_path(path, repo_root)

def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Synthetic plant generation requires columns: {missing}")


def _clip(series: pd.Series, *, lower: float, upper: float) -> pd.Series:
    return series.clip(lower=lower, upper=upper)


def _softplus(values: pd.Series, *, scale: float) -> pd.Series:
    scaled = np.clip(pd.to_numeric(values, errors="coerce").fillna(0.0) / max(scale, 1e-6), -60.0, 60.0)
    return pd.Series(np.log1p(np.exp(scaled)), index=values.index)


def _weighted_forward_average(series: pd.Series, weights: list[float]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    weighted_sum = pd.Series(0.0, index=numeric.index, dtype=float)
    weight_sum = pd.Series(0.0, index=numeric.index, dtype=float)
    for horizon, weight in enumerate(weights, start=1):
        shifted = numeric.shift(-horizon)
        valid = shifted.notna().astype(float)
        weighted_sum = weighted_sum.add(shifted.fillna(0.0) * float(weight), fill_value=0.0)
        weight_sum = weight_sum.add(valid * float(weight), fill_value=0.0)
    return (weighted_sum / weight_sum.replace(0.0, np.nan)).fillna(numeric)


def _rolling_mean(series: pd.Series, *, window: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rolling(window=window, min_periods=1).mean()


def _quarter_weight(profile: dict[str, Any], quarter: int) -> float:
    weights = dict(profile.get("seasonal_quarter_weights", {}))
    return float(weights.get(f"q{quarter}", weights.get(str(quarter), 0.0)))


def _merge_procurement_target_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parameters.get("procurement_target_parameters", {}))
    missing = [key for key in REQUIRED_PROCUREMENT_TARGET_PARAMETERS if key not in merged]
    if missing:
        raise ValueError(
            "synthetic_data.simulation_parameters.procurement_target_parameters "
            f"is missing canonical config keys: {missing}"
        )
    weights = [float(weight) for weight in merged["forward_requirement_weights"]]
    if not weights or sum(weights) <= 0:
        raise ValueError("procurement_target_parameters.forward_requirement_weights must contain positive weights.")
    merged["forward_requirement_weights"] = weights
    return merged


def _merge_manufacturing_context(parameters: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "context_window_weeks": int(DEFAULT_MANUFACTURING_CONTEXT["context_window_weeks"]),
        "assignment_noise_std": float(DEFAULT_MANUFACTURING_CONTEXT["assignment_noise_std"]),
        "persistence_bonus": float(DEFAULT_MANUFACTURING_CONTEXT["persistence_bonus"]),
        "profiles": [dict(profile) for profile in DEFAULT_MANUFACTURING_CONTEXT["profiles"]],
    }
    override = dict(parameters.get("manufacturing_context", {}))
    if "profiles" in override:
        merged["profiles"] = [dict(profile) for profile in override.get("profiles", [])]
    for key in ["context_window_weeks", "assignment_noise_std", "persistence_bonus"]:
        if key in override:
            merged[key] = override[key]
    if not merged["profiles"]:
        raise ValueError("manufacturing_context.profiles must contain at least one profile.")
    return merged


def _assign_manufacturing_context(
    df: pd.DataFrame,
    *,
    demand_factor: pd.Series,
    supply_factor: pd.Series,
    context_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, Any]]:
    public_rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    previous_profile_name: str | None = None
    window = max(1, int(context_cfg.get("context_window_weeks", 4)))
    for start in range(0, len(df), window):
        stop = min(start + window, len(df))
        quarter = int(df.iloc[start]["date"].quarter)
        demand_pressure = float(pd.to_numeric(demand_factor.iloc[start:stop], errors="coerce").mean() - 1.0)
        supply_stress = float(max(1.0 - pd.to_numeric(supply_factor.iloc[start:stop], errors="coerce").mean(), 0.0))
        best_profile = None
        best_score = -np.inf
        for profile in context_cfg["profiles"]:
            score = (
                _quarter_weight(profile, quarter)
                + float(profile.get("selection_bias", 0.0))
                + float(profile["demand_sensitivity"]) * demand_pressure
                + float(profile["supply_stress_sensitivity"]) * supply_stress
                + rng.normal(0.0, float(context_cfg.get("assignment_noise_std", 0.03)))
            )
            if previous_profile_name == profile["profile_name"]:
                score += float(context_cfg.get("persistence_bonus", 0.08))
            if score > best_score:
                best_score = score
                best_profile = profile
        if best_profile is None:
            raise ValueError("No manufacturing context profile could be assigned.")
        block_rows.append({
            "date_start": str(pd.to_datetime(df.iloc[start]["date"]).date()),
            "date_end": str(pd.to_datetime(df.iloc[stop - 1]["date"]).date()),
            "profile_name": best_profile["profile_name"],
            "product_family": best_profile["product_family"],
            "process_type": best_profile["process_type"],
            "quarter": quarter,
            "assignment_score": round(float(best_score), 6),
        })
        public_payload = {
            "manufacturing_context_profile": best_profile["profile_name"],
            "product_family": best_profile["product_family"],
            "process_type": best_profile["process_type"],
            "recipe_profile": best_profile["recipe_profile"],
            "formulation_class": best_profile["formulation_class"],
            "expected_yield": best_profile["expected_yield"],
            "expected_waste": best_profile["expected_waste"],
            "process_lead_time_days": best_profile["process_lead_time_days"],
            "shelf_life_class": best_profile["shelf_life_class"],
            "priority_level": best_profile["priority_level"],
            "cost_sensitivity": best_profile["cost_sensitivity"],
        }
        internal_payload = {
            "production_multiplier": best_profile["production_multiplier"],
            "coverage_adjustment_days": best_profile["coverage_adjustment_days"],
            "lead_time_multiplier": best_profile["lead_time_multiplier"],
            "pressure_buffer_adjustment_days": best_profile["pressure_buffer_adjustment_days"],
            "requirement_pressure_multiplier": best_profile["requirement_pressure_multiplier"],
            "waste_stress_multiplier": best_profile["waste_stress_multiplier"],
            "yield_stress_multiplier": best_profile["yield_stress_multiplier"],
            "policy_target_coverage_adjustment_days": best_profile["policy_target_coverage_adjustment_days"],
            "policy_excess_penalty_multiplier": best_profile["policy_excess_penalty_multiplier"],
            "policy_need_multiplier": best_profile["policy_need_multiplier"],
            "policy_max_order_multiplier": best_profile["policy_max_order_multiplier"],
        }
        for _ in range(start, stop):
            public_rows.append(dict(public_payload))
            internal_rows.append(dict(internal_payload))
        previous_profile_name = best_profile["profile_name"]
    public_df = pd.DataFrame(public_rows, index=df.index)
    internal_df = pd.DataFrame(internal_rows, index=df.index)
    internal_series = {column: pd.to_numeric(internal_df[column], errors="coerce").astype(float) for column in internal_df.columns}
    return public_df, internal_series, {"context_window_weeks": window, "profiles": context_cfg["profiles"], "blocks": block_rows}


def _build_lineage_rows(canonical_variant: str) -> list[dict[str, Any]]:
    rows = [
        {"column_name": "date", "origin_type": "derived", "source": "weekly_index", "description": "Weekly temporal reference.", "notes": "Not a plant ERP timestamp."},
        {"column_name": "demand_index", "origin_type": "proxy", "source": "external_proxy", "description": "External demand proxy.", "notes": "Not internal orders."},
        {"column_name": "supply_index", "origin_type": "proxy", "source": "external_proxy", "description": "External supply proxy.", "notes": "Not inbound receipts."},
        {"column_name": "purchase_price_index", "origin_type": "proxy", "source": "external_proxy", "description": "External price proxy.", "notes": "Weak explanatory value in current data."},
    ]
    for column_name in MANUFACTURING_CONTEXT_PUBLIC_COLUMNS:
        rows.append({"column_name": column_name, "origin_type": "manufacturing_context", "source": "context_profile_assignment", "description": "Manufacturing-context field published for conditioning, EDA and contract tracing.", "notes": "Used to avoid treating all processed-meat manufacturing modes as equivalent."})
    for column_name in ["synthetic_plant_production_volume", "synthetic_plant_capacity_utilization", "synthetic_yield_rate", "synthetic_waste_rate", "synthetic_raw_material_requirement", "synthetic_inventory_coverage_days", "synthetic_inventory_level", "synthetic_lead_time_days", "synthetic_procurement_need_pressure", "synthetic_procurement_need_coverage_gap", "synthetic_procurement_need_hybrid", "synthetic_procurement_need_refined", "synthetic_procurement_need", "synthetic_planned_orders"]:
        origin_type = "synthetic"
        notes = "Published synthetic operational signal."
        if column_name == "synthetic_procurement_need":
            origin_type = "synthetic_target_alias"
            notes = f"Canonical target alias backed by the '{canonical_variant}' formulation."
        elif column_name == "synthetic_procurement_need_refined":
            origin_type = "synthetic_target_alias"
            notes = "Refined comparison-only alias."
        elif column_name.startswith("synthetic_procurement_need_"):
            origin_type = "synthetic_target_variant"
            notes = "Explicit target formulation retained for traceability."
        rows.append({"column_name": column_name, "origin_type": origin_type, "source": "synthetic_plant_simulator", "description": "Synthetic signal generated from proxy context plus manufacturing context.", "notes": notes})
    return rows


def _build_procurement_target_formulations_payload(target_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_variant": target_cfg["canonical_variant"],
        "shared_derived_terms": {
            "future_requirement_weighted": "Forward weighted average of synthetic_raw_material_requirement over t+1..t+4.",
            "future_lead_time_weighted": "Forward weighted average of synthetic_lead_time_days over t+1..t+4.",
            "manufacturing_context_adjustments": "Manufacturing context shifts yield, waste, coverage, lead time and pressure buffers before target computation.",
        },
        "variants": {"pressure": {"column": "synthetic_procurement_need_pressure"}, "coverage_gap": {"column": "synthetic_procurement_need_coverage_gap"}, "hybrid": {"column": "synthetic_procurement_need_hybrid"}},
    }

def _build_assumptions_payload(parameters: dict[str, Any], synthetic_config: dict[str, Any], context_payload: dict[str, Any]) -> dict[str, Any]:
    target_roles = synthetic_config.get("target_roles", {})
    return {
        "environment_name": synthetic_config.get("environment_name", "procurement_training_environment"),
        "environment_role": synthetic_config.get("environment_role"),
        "time_granularity": str(synthetic_config.get("time_granularity", "weekly")),
        "validated_base_horizon_weeks": int(synthetic_config.get("validated_base_horizon_weeks", 1)),
        "validated_base_horizon_label": str(synthetic_config.get("validated_base_horizon_label", "W+1")),
        "target_projection_window_weeks": int(synthetic_config.get("target_projection_window_weeks", 4)),
        "target_roles": {
            "canonical_target_column": target_roles.get("canonical_target_column", "synthetic_procurement_need"),
            "canonical_variant": target_roles.get("canonical_variant", "pressure"),
            "requirement_target_column": target_roles.get("target_requirement_column", "synthetic_raw_material_requirement"),
            "decision_output_column": target_roles.get("decision_output_column", "order_quantity_tons"),
            "decision_role": target_roles.get("decision_role", "downstream_policy_layer"),
            "optimal_quantity_role": target_roles.get("optimal_quantity_role", "downstream_decision_or_optimization_stage"),
        },
        "parameters": parameters,
        "manufacturing_context": {
            "context_columns": list(synthetic_config.get("manufacturing_context_columns", MANUFACTURING_CONTEXT_PUBLIC_COLUMNS)),
            "assignment_rules": context_payload,
            "recipe_runtime_context": synthetic_config.get("recipe_runtime_context"),
            "recipe_registry_path": synthetic_config.get("recipe_registry_path"),
            "statement": "Procurement is conditioned by product family, process and recipe context inside a controlled synthetic environment.",
        },
        "problem_framing": {
            "target_vs_decision": "synthetic_procurement_need is the operational target; order_quantity_tons belongs to the downstream policy layer.",
            "manufacturing_context_statement": "The simulator no longer treats processed meats as a single generic manufacturing mode.",
            "synthetic_environment_statement": "This remains controlled simulation, not industrial validation.",
        },
    }


def _group_metric_summary(df: pd.DataFrame, group_column: str, metrics: list[str]) -> list[dict[str, Any]]:
    if group_column not in df.columns:
        return []
    available_metrics = [metric for metric in metrics if metric in df.columns]
    if not available_metrics:
        return []
    grouped = df.groupby(group_column, dropna=False)[available_metrics].mean().reset_index()
    rows = []
    for _, row in grouped.iterrows():
        payload = {group_column: row[group_column]}
        for metric in available_metrics:
            payload[f"{metric}_mean"] = float(row[metric])
        rows.append(payload)
    return rows


def _build_environment_summary_payload(df: pd.DataFrame, synthetic_columns: list[str], synthetic_config: dict[str, Any]) -> dict[str, Any]:
    key_columns = [column for column in ["synthetic_raw_material_requirement", "synthetic_inventory_coverage_days", "synthetic_lead_time_days", "synthetic_procurement_need"] if column in df.columns]
    return {
        "environment_name": synthetic_config.get("environment_name"),
        "environment_role": synthetic_config.get("environment_role"),
        "time_granularity": str(synthetic_config.get("time_granularity", "weekly")),
        "validated_base_horizon_weeks": int(synthetic_config.get("validated_base_horizon_weeks", 1)),
        "validated_base_horizon_label": str(synthetic_config.get("validated_base_horizon_label", "W+1")),
        "target_projection_window_weeks": int(synthetic_config.get("target_projection_window_weeks", 4)),
        "dataset_summary": {
            "rows": int(len(df)),
            "date_min": str(df["date"].min().date()) if not df.empty else None,
            "date_max": str(df["date"].max().date()) if not df.empty else None,
            "synthetic_columns": synthetic_columns,
            "manufacturing_context_columns": [column for column in synthetic_config.get("manufacturing_context_columns", MANUFACTURING_CONTEXT_PUBLIC_COLUMNS) if column in df.columns],
            "active_recipe_profile": synthetic_config.get("recipe_runtime_context", {}).get("recipe_profile"),
            "recipe_registry_path": synthetic_config.get("recipe_registry_path"),
        },
        "manufacturing_context_summary": {
            "profile_mix": df["manufacturing_context_profile"].value_counts(dropna=False).to_dict() if "manufacturing_context_profile" in df.columns else {},
            "product_family_mix": df["product_family"].value_counts(dropna=False).to_dict() if "product_family" in df.columns else {},
            "process_type_mix": df["process_type"].value_counts(dropna=False).to_dict() if "process_type" in df.columns else {},
            "recipe_profile_mix": df["recipe_profile"].value_counts(dropna=False).to_dict() if "recipe_profile" in df.columns else {},
            "by_product_family": _group_metric_summary(df, "product_family", key_columns),
            "by_process_type": _group_metric_summary(df, "process_type", key_columns),
            "by_recipe_profile": _group_metric_summary(df, "recipe_profile", key_columns),
        },
        "target_roles": synthetic_config.get("target_roles", {}),
    }


def build_synthetic_plant_dataset(
    context_df: pd.DataFrame,
    synthetic_config: dict[str, Any],
    *,
    recipe_runtime_context: dict[str, Any] | None = None,
    recipe_registry_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate a synthetic plant layer from the external weekly proxy context."""
    _require_columns(context_df, REQUIRED_CONTEXT_COLUMNS)
    parameters = dict(synthetic_config.get("simulation_parameters", {}))
    if not parameters:
        raise ValueError("synthetic_data.simulation_parameters is required to build the synthetic dataset.")
    parameters["procurement_target_parameters"] = _merge_procurement_target_parameters(parameters)
    parameters["manufacturing_context"] = _merge_manufacturing_context(parameters)
    target_cfg = parameters["procurement_target_parameters"]

    rng = np.random.default_rng(int(synthetic_config.get("simulation_seed", 42)))
    df = context_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].sort_values("date").reset_index(drop=True)
    for column in ["demand_index", "purchase_price_index"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").ffill().bfill()
    df["supply_index"] = pd.to_numeric(df["supply_index"], errors="coerce").ffill()
    supply_index_for_simulation = df["supply_index"].fillna(100.0)

    week_of_year = df["date"].dt.isocalendar().week.astype(int)
    seasonal_factor = 1.0 + float(parameters["seasonality_amplitude"]) * np.sin(2.0 * np.pi * (week_of_year - 1) / 52.0)
    demand_median = float(df["demand_index"].median()) if not df["demand_index"].dropna().empty else 1.0
    supply_median = float(supply_index_for_simulation.median()) if not supply_index_for_simulation.dropna().empty else 100.0
    demand_factor = df["demand_index"] / max(demand_median, 1e-6)
    supply_factor = supply_index_for_simulation / max(supply_median, 1e-6)

    context_public_df, context_internal, context_payload = _assign_manufacturing_context(df, demand_factor=demand_factor, supply_factor=supply_factor, context_cfg=parameters["manufacturing_context"], rng=rng)
    df = pd.concat([df, context_public_df], axis=1)
    priority_score = df["priority_level"].map(PRIORITY_LEVEL_SCORES).fillna(0.60)
    cost_score = df["cost_sensitivity"].map(COST_SENSITIVITY_SCORES).fillna(0.50)
    shelf_life_factor = df["shelf_life_class"].map(SHELF_LIFE_FACTORS).fillna(1.0)

    production_driver = float(parameters["demand_weight"]) * demand_factor + float(parameters["supply_weight"]) * supply_factor
    context_volume_multiplier = context_internal["production_multiplier"] * (1.0 + 0.05 * (priority_score - 0.5) - 0.04 * (cost_score - 0.5))
    production_noise = rng.normal(1.0, float(parameters["production_noise_std"]), size=len(df))
    synthetic_plant_production_volume = float(parameters["base_weekly_output_tons"]) * production_driver * seasonal_factor * context_volume_multiplier * production_noise
    synthetic_plant_production_volume = _clip(pd.Series(synthetic_plant_production_volume, index=df.index), lower=float(parameters["min_weekly_output_tons"]), upper=float(parameters["nominal_capacity_tons"]) * 0.99)

    synthetic_plant_capacity_utilization = _clip(synthetic_plant_production_volume / float(parameters["nominal_capacity_tons"]), lower=float(parameters["min_capacity_utilization"]), upper=float(parameters["max_capacity_utilization"]))
    utilization_excess = (synthetic_plant_capacity_utilization - float(parameters["preferred_capacity_utilization"])).clip(lower=0.0)
    yield_noise = rng.normal(0.0, float(parameters["yield_noise_std"]), size=len(df))
    synthetic_yield_rate = _clip(pd.to_numeric(df["expected_yield"], errors="coerce") - float(parameters["yield_penalty_per_utilization_point"]) * context_internal["yield_stress_multiplier"] * utilization_excess + yield_noise, lower=float(parameters["min_yield_rate"]), upper=float(parameters["max_yield_rate"]))
    waste_noise = rng.normal(0.0, float(parameters["waste_noise_std"]), size=len(df))
    synthetic_waste_rate = _clip(pd.to_numeric(df["expected_waste"], errors="coerce") + float(parameters["waste_penalty_per_utilization_point"]) * context_internal["waste_stress_multiplier"] * utilization_excess + waste_noise, lower=float(parameters["min_waste_rate"]), upper=float(parameters["max_waste_rate"]))
    synthetic_raw_material_requirement = (synthetic_plant_production_volume / synthetic_yield_rate) * (1.0 + synthetic_waste_rate)

    inventory_noise = rng.normal(0.0, float(parameters["inventory_coverage_noise_std"]), size=len(df))
    coverage_context = context_internal["coverage_adjustment_days"] + 1.4 * (priority_score - 0.5) + 1.6 * (shelf_life_factor - 1.0) - 1.1 * (cost_score - 0.5)
    synthetic_inventory_coverage_days = _clip(float(parameters["base_inventory_coverage_days"]) + float(parameters["inventory_supply_bonus_days"]) * (supply_factor - 1.0) - float(parameters["inventory_demand_penalty_days"]) * (demand_factor - 1.0) + coverage_context + inventory_noise, lower=float(parameters["min_inventory_coverage_days"]), upper=float(parameters["max_inventory_coverage_days"]))
    synthetic_inventory_level = synthetic_raw_material_requirement * (synthetic_inventory_coverage_days / 7.0)

    lead_time_noise = rng.normal(0.0, float(parameters["lead_time_noise_std"]), size=len(df))
    synthetic_lead_time_days = _clip(pd.to_numeric(df["process_lead_time_days"], errors="coerce") + float(parameters["lead_time_supply_penalty_days"]) * (1.0 - supply_factor) * context_internal["lead_time_multiplier"] + 0.5 * (cost_score - 0.5) + lead_time_noise, lower=float(parameters["min_lead_time_days"]), upper=float(parameters["max_lead_time_days"])).round(0)

    target_stock_days = synthetic_lead_time_days + float(parameters["safety_stock_days"]) + 0.35 * context_internal["pressure_buffer_adjustment_days"]
    legacy_snapshot_need = (synthetic_raw_material_requirement * (target_stock_days / 7.0) - synthetic_inventory_level).clip(lower=0.0)
    forward_weights = [float(weight) for weight in target_cfg["forward_requirement_weights"]]
    future_requirement_weighted = _weighted_forward_average(synthetic_raw_material_requirement, forward_weights)
    future_lead_time_weighted = _weighted_forward_average(synthetic_lead_time_days.astype(float), forward_weights)
    future_supply_weighted = _weighted_forward_average(supply_index_for_simulation, forward_weights)
    trailing_requirement_mean = _rolling_mean(synthetic_raw_material_requirement, window=4)
    trailing_coverage_mean = _rolling_mean(synthetic_inventory_coverage_days, window=4)
    blended_available_coverage_days = (float(target_cfg["available_coverage_current_weight"]) * synthetic_inventory_coverage_days + float(target_cfg["available_coverage_trailing_weight"]) * trailing_coverage_mean).clip(lower=0.5)
    demand_growth_ratio = (((future_requirement_weighted - trailing_requirement_mean) / trailing_requirement_mean.replace(0.0, np.nan)).clip(lower=0.0)).fillna(0.0)
    supply_stress_ratio = ((supply_median - future_supply_weighted) / max(supply_median, 1e-6)).clip(lower=0.0)
    context_pressure_days = context_internal["pressure_buffer_adjustment_days"] + 0.8 * (priority_score - 0.5) - 0.4 * (shelf_life_factor - 1.0) + 0.25 * (cost_score - 0.5)

    projected_coverage_target_days = float(target_cfg["coverage_gap_buffer_base_days"]) + float(target_cfg["coverage_gap_lead_time_weight"]) * future_lead_time_weighted + float(target_cfg["coverage_gap_supply_stress_weight"]) * supply_stress_ratio + float(target_cfg["coverage_gap_growth_weight"]) * demand_growth_ratio + 0.6 * context_pressure_days
    blended_requirement = float(target_cfg["coverage_gap_future_requirement_weight"]) * future_requirement_weighted + float(target_cfg["coverage_gap_trailing_requirement_weight"]) * trailing_requirement_mean
    projected_coverage_gap_days = (projected_coverage_target_days - blended_available_coverage_days).clip(lower=0.0)
    synthetic_procurement_need_coverage_gap = (blended_requirement * projected_coverage_gap_days / 7.0).clip(lower=0.0)

    effective_context_pressure_days = float(target_cfg["pressure_context_days_weight"]) * context_pressure_days
    effective_requirement_pressure_multiplier = 1.0 + float(target_cfg["pressure_context_multiplier_weight"]) * (
        context_internal["requirement_pressure_multiplier"] - 1.0
    )
    projected_pressure_days = float(target_cfg["pressure_buffer_base_days"]) + float(target_cfg["pressure_lead_time_weight"]) * future_lead_time_weighted + float(target_cfg["pressure_supply_stress_weight"]) * supply_stress_ratio + float(target_cfg["pressure_growth_weight"]) * demand_growth_ratio + effective_context_pressure_days
    pressure_core = (future_requirement_weighted * effective_requirement_pressure_multiplier * (((projected_pressure_days / blended_available_coverage_days) - float(target_cfg["pressure_ratio_offset"])).clip(lower=0.0)) / float(target_cfg["pressure_scaling_divisor"])).clip(lower=0.0)
    pressure_snapshot_blend_weight = float(target_cfg["pressure_snapshot_blend_weight"])
    pressure_coverage_gap_blend_weight = float(target_cfg["pressure_coverage_gap_blend_weight"])
    pressure_core_weight = max(1.0 - pressure_snapshot_blend_weight - pressure_coverage_gap_blend_weight, 0.0)
    pressure_weight_total = pressure_core_weight + pressure_snapshot_blend_weight + pressure_coverage_gap_blend_weight
    if pressure_weight_total <= 0:
        synthetic_procurement_need_pressure = pressure_core.copy()
    else:
        synthetic_procurement_need_pressure = (
            pressure_core_weight * pressure_core
            + pressure_snapshot_blend_weight * legacy_snapshot_need
            + pressure_coverage_gap_blend_weight * synthetic_procurement_need_coverage_gap
        ) / pressure_weight_total
        synthetic_procurement_need_pressure = synthetic_procurement_need_pressure.clip(lower=0.0)

    hybrid_projected_pressure_days = float(target_cfg["hybrid_buffer_base_days"]) + float(target_cfg["hybrid_lead_time_weight"]) * future_lead_time_weighted + float(target_cfg["hybrid_supply_stress_weight"]) * supply_stress_ratio + float(target_cfg["hybrid_growth_weight"]) * demand_growth_ratio + context_pressure_days
    hybrid_pressure_core = (future_requirement_weighted * context_internal["requirement_pressure_multiplier"] * (((hybrid_projected_pressure_days / blended_available_coverage_days) - float(target_cfg["hybrid_ratio_offset"])).clip(lower=0.0)) / float(target_cfg["hybrid_scaling_divisor"])).clip(lower=0.0)
    growth_delta_tons = (future_requirement_weighted - trailing_requirement_mean).clip(lower=0.0)
    smoothed_growth_boost = float(target_cfg["hybrid_growth_boost_scale"]) * _softplus(growth_delta_tons, scale=float(target_cfg["hybrid_growth_softplus_scale_tons"]))
    synthetic_procurement_need_hybrid = (hybrid_pressure_core + float(target_cfg["hybrid_growth_boost_weight"]) * smoothed_growth_boost - float(target_cfg["hybrid_snapshot_penalty_weight"]) * (legacy_snapshot_need - hybrid_pressure_core).clip(lower=0.0)).clip(lower=0.0)

    canonical_variant = str(target_cfg["canonical_variant"]).strip().lower()
    target_variant_map = {"pressure": synthetic_procurement_need_pressure, "coverage_gap": synthetic_procurement_need_coverage_gap, "hybrid": synthetic_procurement_need_hybrid}
    if canonical_variant not in target_variant_map:
        raise ValueError(f"Unsupported synthetic procurement target variant: {canonical_variant}")
    synthetic_procurement_need = target_variant_map[canonical_variant].copy()
    synthetic_procurement_need_refined = synthetic_procurement_need_hybrid.copy()
    planned_ratio = float(parameters["planned_orders_buffer_ratio"]) * (1.0 + 0.15 * (priority_score - 0.5) - 0.12 * (cost_score - 0.5))
    synthetic_planned_orders = synthetic_procurement_need * (1.0 + planned_ratio.clip(lower=0.0))

    synthetic_columns = {
        "synthetic_plant_production_volume": synthetic_plant_production_volume,
        "synthetic_plant_capacity_utilization": synthetic_plant_capacity_utilization,
        "synthetic_yield_rate": synthetic_yield_rate,
        "synthetic_waste_rate": synthetic_waste_rate,
        "synthetic_raw_material_requirement": synthetic_raw_material_requirement,
        "synthetic_inventory_coverage_days": synthetic_inventory_coverage_days,
        "synthetic_inventory_level": synthetic_inventory_level,
        "synthetic_lead_time_days": synthetic_lead_time_days.astype(int),
        "synthetic_procurement_need_pressure": synthetic_procurement_need_pressure,
        "synthetic_procurement_need_coverage_gap": synthetic_procurement_need_coverage_gap,
        "synthetic_procurement_need_hybrid": synthetic_procurement_need_hybrid,
        "synthetic_procurement_need_refined": synthetic_procurement_need_refined,
        "synthetic_procurement_need": synthetic_procurement_need,
        "synthetic_planned_orders": synthetic_planned_orders,
    }
    for name, values in synthetic_columns.items():
        df[name] = values

    lineage_df = pd.DataFrame(_build_lineage_rows(canonical_variant))
    synthetic_config = dict(synthetic_config)
    synthetic_config["recipe_runtime_context"] = dict(recipe_runtime_context or {})
    synthetic_config["recipe_registry_path"] = recipe_registry_path
    assumptions_payload = _build_assumptions_payload(parameters, synthetic_config, context_payload)
    assumptions_payload["input_context"] = {"rows": int(len(context_df)), "date_min": str(df["date"].min().date()) if not df.empty else None, "date_max": str(df["date"].max().date()) if not df.empty else None, "columns": context_df.columns.tolist()}
    assumptions_payload["context_imputation"] = {
        "supply_index_public_column": "preserved_as_nan_before_first_mapa_observation",
        "simulator_internal_fill_value": 100.0,
        "simulator_internal_fill_scope": "synthetic_signal_generation_only",
        "modeling_imputation": "train_median_with_neutral_fallback_inside_training_and_inference",
        "backfill_applied_to_supply_index": False,
    }
    assumptions_payload["synthetic_columns"] = list(synthetic_columns)
    assumptions_payload["manufacturing_context_columns"] = list(synthetic_config.get("manufacturing_context_columns", MANUFACTURING_CONTEXT_PUBLIC_COLUMNS))
    assumptions_payload["available_target_columns"] = [column for column in synthetic_config.get("candidate_target_columns", []) if column in df.columns]
    assumptions_payload["procurement_target_formulations"] = _build_procurement_target_formulations_payload(target_cfg)
    assumptions_payload["simulator_environment_summary"] = _build_environment_summary_payload(df, list(synthetic_columns), synthetic_config)
    return df, lineage_df, assumptions_payload

def write_synthetic_plant_outputs(
    fused_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    assumptions_payload: dict[str, Any],
    synthetic_config: dict[str, Any],
    repo_root: Path,
) -> dict[str, str]:
    """Persist the synthetic plant dataset and its traceability metadata."""
    output_dataset_path = resolve_repo_path(synthetic_config["output_dataset_path"], repo_root)
    lineage_path = resolve_repo_path(synthetic_config["column_lineage_path"], repo_root)
    assumptions_path = resolve_repo_path(synthetic_config["simulation_parameters_path"], repo_root)
    ensure_directory(output_dataset_path.parent)
    ensure_directory(lineage_path.parent)
    ensure_directory(assumptions_path.parent)
    fused_df.to_csv(output_dataset_path, index=False)
    lineage_df.to_csv(lineage_path, index=False)
    write_json(assumptions_path, assumptions_payload)
    output_paths = {
        "output_dataset_path": _portable_path(output_dataset_path, repo_root),
        "column_lineage_path": _portable_path(lineage_path, repo_root),
        "simulation_parameters_path": _portable_path(assumptions_path, repo_root),
    }
    summary_payload = assumptions_payload.get("simulator_environment_summary")
    summary_path_value = synthetic_config.get("environment_summary_path")
    if summary_path_value and summary_payload:
        summary_path = resolve_repo_path(summary_path_value, repo_root)
        ensure_directory(summary_path.parent)
        write_json(summary_path, summary_payload)
        output_paths["environment_summary_path"] = _portable_path(summary_path, repo_root)
    return output_paths
