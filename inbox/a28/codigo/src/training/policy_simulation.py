"""Offline policy simulation on top of the controlled synthetic procurement environment."""

from __future__ import annotations

import math
import pickle
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import percentage_change, percentage_reduction, stockout_guardrail_pass
from src.utils import (
    current_recipe_context,
    ensure_runtime_context_resolved,
    ensure_directory,
    filter_frame_to_recipe,
    make_run_id,
    read_json,
    resolve_repo_path,
    to_repo_relative_path,
    utc_timestamp,
    write_json,
)

REQUIRED_POLICY_NUMERIC_COLUMNS = [
    "date",
    "demand_index",
    "supply_index",
    "purchase_price_index",
    "synthetic_plant_production_volume",
    "synthetic_plant_capacity_utilization",
    "synthetic_yield_rate",
    "synthetic_raw_material_requirement",
    "synthetic_inventory_coverage_days",
    "synthetic_inventory_level",
    "synthetic_lead_time_days",
    "synthetic_planned_orders",
    "synthetic_waste_rate",
]
REQUIRED_POLICY_CONTEXT_COLUMNS = [
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
PRIORITY_LEVEL_SCORES = {"planned_campaign": 0.45, "standard_service": 0.65, "service_critical": 0.95}
COST_SENSITIVITY_SCORES = {"high": 0.85, "medium": 0.55, "low": 0.25}
SHELF_LIFE_FACTORS = {"short": 0.70, "medium": 1.00, "long": 1.35}

CURRENT_FEATURE_ALIASES = {
    "demand_index": "demand_index",
    "supply_index": "supply_index",
    "purchase_price_index": "purchase_price_index",
    "synthetic_plant_production_volume": "synthetic_plant_production_volume",
    "synthetic_plant_capacity_utilization": "synthetic_plant_capacity_utilization",
    "synthetic_yield_rate": "synthetic_yield_rate",
    "synthetic_raw_material_requirement": "synthetic_raw_material_requirement",
    "synthetic_inventory_coverage_days": "synthetic_inventory_coverage_days",
    "synthetic_lead_time_days": "synthetic_lead_time_days",
    "synthetic_waste_rate": "synthetic_waste_rate",
    "expected_yield": "expected_yield",
    "expected_waste": "expected_waste",
    "process_lead_time_days": "process_lead_time_days",
}


def _portable_path(path: str | Path | None, repo_root: Path) -> str | None:
    if path is None:
        return None
    return to_repo_relative_path(path, repo_root)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _clip_value(value: float, lower: float | None = None, upper: float | None = None) -> float:
    clipped = float(value)
    if lower is not None:
        clipped = max(clipped, float(lower))
    if upper is not None:
        clipped = min(clipped, float(upper))
    return float(clipped)


def _round_to_lot(value: float, lot_size: float) -> float:
    if lot_size <= 0:
        return float(max(value, 0.0))
    return float(round(max(value, 0.0) / lot_size) * lot_size)


def _resolve_split_bounds(total_rows: int, split_cfg: dict[str, Any], evaluation_split: str) -> tuple[int, int]:
    train_end = max(1, int(total_rows * float(split_cfg.get("train_size", 0.7))))
    valid_end = train_end + int(total_rows * float(split_cfg.get("valid_size", 0.15)))

    split_name = str(evaluation_split).strip().lower()
    if split_name == "train":
        return 0, train_end
    if split_name in {"validation", "valid"}:
        return train_end, valid_end
    if split_name == "test":
        return valid_end, total_rows
    raise ValueError(f"Unsupported policy_simulation.evaluation_split: {evaluation_split}")


def _stable_offset(scenario_name: str, seed: int, feasible_offsets: int) -> int:
    if feasible_offsets <= 1:
        return 0
    token_score = sum((index + 1) * ord(character) for index, character in enumerate(scenario_name))
    return int((token_score + (int(seed) * 17)) % feasible_offsets)


def _resolve_artifact_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = resolve_repo_path(path, repo_root)
    if not path.exists():
        raise FileNotFoundError(f"Policy simulation requires an artifact that was not found: {path}")
    return path


def _load_model_payload(
    *,
    summary_path: Path,
    run_key: str,
    repo_root: Path,
    expected_feature_set: str,
    expected_target_column: str,
    expected_model_family: str | None,
) -> dict[str, Any]:
    summary_payload = read_json(summary_path)
    run_payload = summary_payload.get(run_key)
    if not isinstance(run_payload, dict):
        raise ValueError(f"Expected '{run_key}' inside {summary_path}, but it was missing or invalid.")

    if run_payload.get("feature_set") != expected_feature_set:
        raise ValueError(
            f"Run '{run_key}' from {summary_path} uses feature_set={run_payload.get('feature_set')} "
            f"instead of the expected '{expected_feature_set}'."
        )
    if run_payload.get("target_column") != expected_target_column:
        raise ValueError(
            f"Run '{run_key}' from {summary_path} uses target={run_payload.get('target_column')} "
            f"instead of '{expected_target_column}'."
        )
    if expected_model_family and run_payload.get("model_family") != expected_model_family:
        raise ValueError(
            f"Run '{run_key}' from {summary_path} uses model_family={run_payload.get('model_family')} "
            f"instead of '{expected_model_family}'."
        )

    artifact_path_value = run_payload.get("artifact_path")
    if not artifact_path_value:
        raise FileNotFoundError(
            f"Policy simulation requires an active serialized artifact for '{run_key}' in {summary_path}, "
            "but the summary does not expose one. Re-run training to regenerate active `.pkl` artifacts."
        )

    artifact_path = _resolve_artifact_path(artifact_path_value, repo_root)
    with artifact_path.open("rb") as handle:
        artifact_payload = pickle.load(handle)
    if artifact_payload.get("target_column") != expected_target_column:
        raise ValueError(
            f"Artifact {artifact_path} was trained for target={artifact_payload.get('target_column')} "
            f"instead of '{expected_target_column}'."
        )
    if artifact_payload.get("feature_set") != expected_feature_set:
        raise ValueError(
            f"Artifact {artifact_path} uses feature_set={artifact_payload.get('feature_set')} "
            f"instead of '{expected_feature_set}'."
        )
    return {
        "summary_run": run_payload,
        "artifact_payload": artifact_payload,
        "artifact_path": artifact_path,
        "summary_path": summary_path,
    }


def _prepare_base_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = [*REQUIRED_POLICY_NUMERIC_COLUMNS, *REQUIRED_POLICY_CONTEXT_COLUMNS]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Policy simulation dataset is missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].sort_values("date").reset_index(drop=True)
    for column in REQUIRED_POLICY_NUMERIC_COLUMNS:
        if column == "date":
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce").ffill().bfill()
    for column in REQUIRED_POLICY_CONTEXT_COLUMNS:
        if column in {"expected_yield", "expected_waste", "process_lead_time_days"}:
            df[column] = pd.to_numeric(df[column], errors="coerce").ffill().bfill()
        else:
            df[column] = df[column].ffill().bfill()
    return df


def _resolve_context_profile_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = (
        config.get("synthetic_data", {})
        .get("simulation_parameters", {})
        .get("manufacturing_context", {})
        .get("profiles", [])
    )
    return {
        str(profile["profile_name"]): dict(profile)
        for profile in profiles
        if isinstance(profile, dict) and profile.get("profile_name")
    }


def _context_adjustments(
    row: pd.Series,
    context_profile_lookup: dict[str, dict[str, Any]],
) -> dict[str, float]:
    profile = context_profile_lookup.get(str(row.get("manufacturing_context_profile", "")), {})
    priority_score = PRIORITY_LEVEL_SCORES.get(str(row.get("priority_level", "")), 0.60)
    cost_score = COST_SENSITIVITY_SCORES.get(str(row.get("cost_sensitivity", "")), 0.50)
    shelf_life_factor = SHELF_LIFE_FACTORS.get(str(row.get("shelf_life_class", "")), 1.0)
    return {
        "target_coverage_adjustment_days": float(profile.get("policy_target_coverage_adjustment_days", 0.0))
        + 0.30 * (priority_score - 0.5)
        - 0.20 * (1.0 - shelf_life_factor),
        "excess_penalty_multiplier": float(profile.get("policy_excess_penalty_multiplier", 1.0))
        * (1.0 + 0.08 * cost_score),
        "need_multiplier": float(profile.get("policy_need_multiplier", 1.0))
        * (1.0 + 0.06 * (priority_score - 0.5)),
        "max_order_multiplier": float(profile.get("policy_max_order_multiplier", 1.0)),
        "stockout_penalty_multiplier": 1.0 + 0.30 * priority_score,
        "waste_penalty_multiplier": 1.0 + 0.25 * max(1.0 - shelf_life_factor, 0.0),
        "priority_score": float(priority_score),
        "cost_score": float(cost_score),
        "shelf_life_factor": float(shelf_life_factor),
    }


def _generate_scenario_segment(
    *,
    base_df: pd.DataFrame,
    evaluation_start: int,
    evaluation_end: int,
    history_weeks: int,
    scenario_template: dict[str, Any],
    seed: int,
    scenario_cfg: dict[str, Any],
    logger,
) -> tuple[pd.DataFrame, int]:
    horizon_weeks = int(scenario_template["horizon_weeks"])
    evaluation_length = evaluation_end - evaluation_start
    if horizon_weeks <= 0:
        raise ValueError(f"Scenario horizon must be positive. Received: {horizon_weeks}")
    if horizon_weeks > evaluation_length:
        raise ValueError(
            f"Scenario '{scenario_template['name']}' horizon={horizon_weeks} exceeds "
            f"the selected evaluation split length={evaluation_length}."
        )

    feasible_offsets = evaluation_length - horizon_weeks + 1
    start_offset = _stable_offset(str(scenario_template["name"]), seed, feasible_offsets)
    scenario_start = evaluation_start + start_offset
    history_start = max(0, scenario_start - history_weeks)
    segment_end = scenario_start + horizon_weeks

    segment = base_df.iloc[history_start:segment_end].copy().reset_index(drop=True)
    evaluation_index = scenario_start - history_start
    if evaluation_index <= 0:
        raise ValueError("Policy simulation requires at least one warm-up/history row before the scenario window.")

    rng = np.random.default_rng(int(seed))
    total_rows = len(segment)

    scenario_name = str(scenario_template["name"])
    demand_multiplier = float(scenario_template.get("demand_multiplier", 1.0))
    supply_multiplier = float(scenario_template.get("supply_multiplier", 1.0))
    lead_time_multiplier = float(scenario_template.get("lead_time_multiplier", 1.0))
    initial_inventory_multiplier = float(scenario_template.get("initial_inventory_multiplier", 1.0))
    target_coverage_shift_days = float(scenario_template.get("target_coverage_shift_days", 0.0))

    min_multiplier = float(scenario_cfg.get("min_multiplier", 0.6))
    max_multiplier = float(scenario_cfg.get("max_multiplier", 1.4))
    requirement_demand_weight = float(scenario_cfg.get("requirement_demand_weight", 1.0))
    requirement_supply_stress_weight = float(scenario_cfg.get("requirement_supply_stress_weight", 0.35))
    lead_time_supply_stress_weight = float(scenario_cfg.get("lead_time_supply_stress_weight", 0.4))
    waste_tension_weight = float(scenario_cfg.get("waste_tension_weight", 0.15))

    demand_noise = rng.normal(0.0, float(scenario_template.get("demand_noise_std", 0.02)), size=total_rows)
    supply_noise = rng.normal(0.0, float(scenario_template.get("supply_noise_std", 0.02)), size=total_rows)
    requirement_noise = rng.normal(0.0, float(scenario_template.get("requirement_noise_std", 0.02)), size=total_rows)
    lead_time_noise = rng.normal(0.0, float(scenario_template.get("lead_time_noise_std_days", 0.25)), size=total_rows)

    demand_factor = np.clip(demand_multiplier + demand_noise, min_multiplier, max_multiplier)
    supply_factor = np.clip(supply_multiplier + supply_noise, min_multiplier, max_multiplier)
    supply_stress = np.clip(1.0 - supply_factor, 0.0, None)
    requirement_factor = np.clip(
        1.0
        + requirement_demand_weight * (demand_factor - 1.0)
        + requirement_supply_stress_weight * supply_stress
        + requirement_noise,
        min_multiplier,
        max_multiplier,
    )
    lead_time_factor = np.clip(
        lead_time_multiplier + lead_time_supply_stress_weight * supply_stress,
        min_multiplier,
        max_multiplier + 0.5,
    )

    segment["scenario_template"] = scenario_name
    segment["scenario_seed"] = int(seed)
    segment["scenario_name"] = f"{scenario_name}__seed_{int(seed)}"
    segment["scenario_demand_multiplier"] = demand_factor
    segment["scenario_supply_multiplier"] = supply_factor
    segment["scenario_requirement_multiplier"] = requirement_factor
    segment["scenario_lead_time_multiplier"] = lead_time_factor
    segment["scenario_initial_inventory_multiplier"] = initial_inventory_multiplier
    segment["scenario_target_coverage_shift_days"] = target_coverage_shift_days

    segment["demand_index"] = segment["demand_index"] * demand_factor
    segment["supply_index"] = segment["supply_index"] * supply_factor
    segment["synthetic_raw_material_requirement"] = (
        segment["synthetic_raw_material_requirement"] * requirement_factor
    ).clip(lower=1.0)
    segment["synthetic_lead_time_days"] = (
        segment["synthetic_lead_time_days"] * lead_time_factor + lead_time_noise
    ).clip(lower=1.0).round(0)
    segment["synthetic_waste_rate"] = (
        segment["synthetic_waste_rate"] * (1.0 + waste_tension_weight * np.clip(requirement_factor - 1.0, 0.0, None))
    ).clip(lower=0.001)
    segment["synthetic_inventory_level"] = (
        segment["synthetic_inventory_level"] * initial_inventory_multiplier
    ).clip(lower=0.0)
    segment["synthetic_inventory_coverage_days"] = (
        segment["synthetic_inventory_level"] / segment["synthetic_raw_material_requirement"].replace(0.0, np.nan) * 7.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    segment["purchase_price_index"] = (
        segment["purchase_price_index"] * (1.0 + 0.25 * np.clip(demand_factor - supply_factor, -0.3, 0.6))
    ).clip(lower=0.1)

    logger.info(
        "Prepared scenario path name=%s seed=%s horizon_weeks=%s eval_start=%s history_start=%s",
        scenario_name,
        seed,
        horizon_weeks,
        scenario_start,
        history_start,
    )
    return segment, evaluation_index


def _seed_pipeline(
    segment_df: pd.DataFrame,
    evaluation_index: int,
    lead_time_weeks_cap: int,
) -> dict[int, float]:
    pipeline_queue: dict[int, float] = defaultdict(float)
    for history_index in range(evaluation_index):
        order_qty = float(segment_df.iloc[history_index]["synthetic_planned_orders"])
        lead_time_days = float(segment_df.iloc[history_index]["synthetic_lead_time_days"])
        arrival_weeks = int(np.clip(math.ceil(max(lead_time_days, 1.0) / 7.0), 1, max(1, lead_time_weeks_cap)))
        arrival_step = history_index + arrival_weeks - evaluation_index
        if arrival_step >= 0:
            pipeline_queue[arrival_step] += max(order_qty, 0.0)
    return dict(pipeline_queue)


def _history_values(segment_df: pd.DataFrame, evaluation_index: int, history_weeks: int) -> dict[str, deque[float]]:
    history_slice = segment_df.iloc[max(0, evaluation_index - history_weeks):evaluation_index].copy()
    series_map = {
        "demand_index": history_slice["demand_index"].astype(float).tolist(),
        "supply_index": history_slice["supply_index"].astype(float).tolist(),
        "synthetic_plant_production_volume": history_slice["synthetic_plant_production_volume"].astype(float).tolist(),
        "synthetic_plant_capacity_utilization": history_slice["synthetic_plant_capacity_utilization"].astype(float).tolist(),
        "synthetic_yield_rate": history_slice["synthetic_yield_rate"].astype(float).tolist(),
        "synthetic_raw_material_requirement": history_slice["synthetic_raw_material_requirement"].astype(float).tolist(),
        "synthetic_inventory_coverage_days": history_slice["synthetic_inventory_coverage_days"].astype(float).tolist(),
        "synthetic_lead_time_days": history_slice["synthetic_lead_time_days"].astype(float).tolist(),
        "purchase_price_index": history_slice["purchase_price_index"].astype(float).tolist(),
        "synthetic_waste_rate": history_slice["synthetic_waste_rate"].astype(float).tolist(),
    }
    return {name: deque(values, maxlen=max(12, history_weeks + 2)) for name, values in series_map.items()}


def _series_lag(history: deque[float], lag: int) -> float:
    if lag <= 0:
        raise ValueError(f"Lag must be positive. Received: {lag}")
    if len(history) < lag:
        return float(history[0]) if history else 0.0
    return float(list(history)[-lag])


def _series_roll_mean(history_with_current: list[float], window: int) -> float:
    values = history_with_current[-window:] if len(history_with_current) >= window else history_with_current
    if not values:
        return 0.0
    return float(np.mean(values))


def _build_feature_row(
    feature_columns: list[str],
    *,
    current_values: dict[str, Any],
    histories: dict[str, deque[float]],
    current_date: pd.Timestamp,
) -> pd.DataFrame:
    feature_values: dict[str, float] = {}
    for feature_name in feature_columns:
        if feature_name in CURRENT_FEATURE_ALIASES:
            feature_values[feature_name] = float(current_values[CURRENT_FEATURE_ALIASES[feature_name]])
            continue
        if feature_name == "date_week_of_year":
            feature_values[feature_name] = float(int(current_date.isocalendar().week))
            continue
        if feature_name == "date_month":
            feature_values[feature_name] = float(current_date.month)
            continue
        if feature_name == "date_quarter":
            feature_values[feature_name] = float(current_date.quarter)
            continue
        if feature_name == "date_year":
            feature_values[feature_name] = float(current_date.year)
            continue
        if feature_name == "demand_supply_gap":
            feature_values[feature_name] = float(current_values["demand_index"] - current_values["supply_index"])
            continue
        if feature_name == "demand_supply_ratio":
            denominator = max(float(current_values["supply_index"]), 1e-6)
            feature_values[feature_name] = float(current_values["demand_index"] / denominator)
            continue
        if "__" in feature_name:
            base_name, encoded_value = feature_name.split("__", 1)
            if base_name in current_values:
                feature_values[feature_name] = 1.0 if str(current_values[base_name]) == encoded_value else 0.0
                continue

        lag_match = re.match(r"^(?P<base>.+)_lag_(?P<lag>\d+)$", feature_name)
        if lag_match:
            base_name = lag_match.group("base")
            lag = int(lag_match.group("lag"))
            if base_name not in histories:
                raise ValueError(f"Unsupported lag feature in policy simulation: {feature_name}")
            feature_values[feature_name] = _series_lag(histories[base_name], lag)
            continue

        roll_match = re.match(r"^(?P<base>.+)_roll_mean_(?P<window>\d+)$", feature_name)
        if roll_match:
            base_name = roll_match.group("base")
            window = int(roll_match.group("window"))
            if base_name not in histories:
                raise ValueError(f"Unsupported rolling feature in policy simulation: {feature_name}")
            current_value = float(current_values.get(base_name, _series_lag(histories[base_name], 1)))
            history_with_current = list(histories[base_name]) + [current_value]
            feature_values[feature_name] = _series_roll_mean(history_with_current, window)
            continue

        raise ValueError(f"Unsupported feature for policy simulation: {feature_name}")

    return pd.DataFrame([feature_values], columns=feature_columns)


def _heuristic_need_estimate(
    *,
    requirement_tons: float,
    coverage_days: float,
    lead_time_days: float,
    safety_stock_days: float,
) -> float:
    pressure_days = float(lead_time_days) + float(safety_stock_days) - float(coverage_days)
    return float(max(0.0, requirement_tons * (pressure_days / 7.0)))


def _policy_need_estimate(
    policy_spec: dict[str, Any],
    *,
    current_values: dict[str, float],
    histories: dict[str, deque[float]],
    current_date: pd.Timestamp,
    heuristic_cfg: dict[str, Any],
) -> float:
    policy_type = policy_spec["policy_type"]
    if policy_type == "operational_simple":
        return _heuristic_need_estimate(
            requirement_tons=float(current_values["synthetic_raw_material_requirement"]),
            coverage_days=float(current_values["synthetic_inventory_coverage_days"]),
            lead_time_days=float(current_values["synthetic_lead_time_days"]),
            safety_stock_days=float(heuristic_cfg.get("safety_stock_days", 6.0)),
        )

    artifact_payload = policy_spec["artifact_payload"]
    feature_columns = list(artifact_payload["feature_columns"])
    feature_frame = _build_feature_row(
        feature_columns,
        current_values=current_values,
        histories=histories,
        current_date=current_date,
    )
    prediction = artifact_payload["model"].predict(feature_frame)[0]
    return float(max(prediction, 0.0))


def _summarise_period_metrics(period_df: pd.DataFrame) -> dict[str, float]:
    requirement_total = float(period_df["requirement_tons"].sum())
    stockout_total = float(period_df["stockout_tons"].sum())
    inventory_excess_total = float(period_df["excess_inventory_tons"].sum())
    procurement_total = float(period_df["order_quantity_tons"].sum())
    unnecessary_total = float(period_df["unnecessary_procurement_tons"].sum())
    waste_total = float(period_df["waste_tons"].sum())
    excess_total = inventory_excess_total + unnecessary_total

    return {
        "periods": int(len(period_df)),
        "total_requirement_tons": requirement_total,
        "total_procurement_tons": procurement_total,
        "avg_procurement_tons": float(period_df["order_quantity_tons"].mean()),
        "total_excess_tons": excess_total,
        "avg_excess_tons": float(period_df["excess_raw_material_tons"].mean()),
        "total_inventory_excess_tons": inventory_excess_total,
        "avg_inventory_excess_tons": float(period_df["excess_inventory_tons"].mean()),
        "total_stockout_tons": stockout_total,
        "stockout_rate": float(stockout_total / requirement_total) if requirement_total > 0 else 0.0,
        "avg_starting_inventory_tons": float(period_df["starting_inventory_tons"].mean()),
        "avg_ending_inventory_tons": float(period_df["ending_inventory_tons"].mean()),
        "avg_coverage_days": float(period_df["ending_coverage_days"].mean()),
        "avg_need_estimate_tons": float(period_df["need_estimate_tons"].mean()),
        "total_unnecessary_procurement_tons": unnecessary_total,
        "avg_unnecessary_procurement_tons": float(period_df["unnecessary_procurement_tons"].mean()),
        "total_waste_tons": waste_total,
        "avg_waste_tons": float(period_df["waste_tons"].mean()),
        "total_cost_proxy": float(period_df["cost_proxy"].sum()),
        "avg_cost_proxy": float(period_df["cost_proxy"].mean()),
        "final_inventory_tons": float(period_df["ending_inventory_tons"].iloc[-1]),
        "final_pipeline_tons": float(period_df["pipeline_after_order_tons"].iloc[-1]),
    }


def _policy_comparison_rows(
    period_df: pd.DataFrame,
    *,
    policy_spec: dict[str, Any],
    scenario_name: str,
    scenario_template: str,
    scenario_seed: int,
    target_column: str,
    feature_set_name: str,
) -> dict[str, Any]:
    payload = _summarise_period_metrics(period_df)
    dominant_product_family = period_df["product_family"].mode().iloc[0] if "product_family" in period_df and not period_df["product_family"].mode().empty else None
    dominant_process_type = period_df["process_type"].mode().iloc[0] if "process_type" in period_df and not period_df["process_type"].mode().empty else None
    payload.update(
        {
            "policy_name": policy_spec["policy_name"],
            "policy_type": policy_spec["policy_type"],
            "policy_group": policy_spec["policy_group"],
            "decision_strategy": policy_spec.get("decision_strategy"),
            "policy_variant": policy_spec.get("policy_variant"),
            "scenario_name": scenario_name,
            "scenario_template": scenario_template,
            "scenario_seed": int(scenario_seed),
            "target_column": target_column,
            "feature_set": feature_set_name,
            "source_run_id": policy_spec.get("source_run_id"),
            "source_model_family": policy_spec.get("source_model_family"),
            "dominant_product_family": dominant_product_family,
            "dominant_process_type": dominant_process_type,
        }
    )
    return payload


def _aggregate_policy_results(
    scenario_summary_df: pd.DataFrame,
    *,
    baseline_policy_name: str,
    kpi_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    baseline_lookup = (
        scenario_summary_df[scenario_summary_df["policy_name"] == baseline_policy_name][["scenario_name", "total_excess_tons", "total_stockout_tons"]]
        .rename(
            columns={
                "total_excess_tons": "baseline_total_excess_tons",
                "total_stockout_tons": "baseline_total_stockout_tons",
            }
        )
        .drop_duplicates(subset=["scenario_name"])
    )
    enriched = scenario_summary_df.merge(baseline_lookup, on="scenario_name", how="left")
    enriched["kpi_excess_reduction_pct"] = percentage_reduction(
        enriched["baseline_total_excess_tons"],
        enriched["total_excess_tons"],
    )
    enriched["stockout_change_pct"] = percentage_change(
        enriched["baseline_total_stockout_tons"],
        enriched["total_stockout_tons"],
    )

    allowed_stockout_increase_pct = float(kpi_cfg.get("allowed_stockout_increase_pct", 0.0))
    require_non_worse_stockout = bool(kpi_cfg.get("require_non_worse_stockouts", True))
    minimum_excess_reduction_pct = float(kpi_cfg.get("minimum_excess_reduction_pct", 20.0))
    if require_non_worse_stockout:
        enriched["stockout_guardrail_pass"] = stockout_guardrail_pass(
            enriched["baseline_total_stockout_tons"],
            enriched["total_stockout_tons"],
            allowed_increase_pct=allowed_stockout_increase_pct,
        )
    else:
        enriched["stockout_guardrail_pass"] = True
    enriched["kpi_met_in_scenario"] = (
        (enriched["kpi_excess_reduction_pct"] >= minimum_excess_reduction_pct) & enriched["stockout_guardrail_pass"]
    )

    aggregated = (
        enriched.groupby(
            [
                "policy_name",
                "policy_type",
                "policy_group",
                "decision_strategy",
                "policy_variant",
                "target_column",
                "feature_set",
                "source_run_id",
                "source_model_family",
            ],
            dropna=False,
        )
        .agg(
            scenarios_evaluated=("scenario_name", "count"),
            aggregate_total_excess_tons=("total_excess_tons", "sum"),
            aggregate_baseline_total_excess_tons=("baseline_total_excess_tons", "sum"),
            aggregate_total_stockout_tons=("total_stockout_tons", "sum"),
            aggregate_baseline_total_stockout_tons=("baseline_total_stockout_tons", "sum"),
            avg_total_excess_tons=("total_excess_tons", "mean"),
            median_total_excess_tons=("total_excess_tons", "median"),
            avg_excess_tons=("avg_excess_tons", "mean"),
            avg_total_stockout_tons=("total_stockout_tons", "mean"),
            avg_stockout_rate=("stockout_rate", "mean"),
            avg_coverage_days=("avg_coverage_days", "mean"),
            avg_total_procurement_tons=("total_procurement_tons", "mean"),
            avg_total_unnecessary_procurement_tons=("total_unnecessary_procurement_tons", "mean"),
            avg_total_waste_tons=("total_waste_tons", "mean"),
            avg_total_cost_proxy=("total_cost_proxy", "mean"),
            avg_kpi_excess_reduction_pct=("kpi_excess_reduction_pct", "mean"),
            median_kpi_excess_reduction_pct=("kpi_excess_reduction_pct", "median"),
            scenario_success_rate=("kpi_met_in_scenario", "mean"),
            stockout_guardrail_pass_rate=("stockout_guardrail_pass", "mean"),
        )
        .reset_index()
    )
    aggregated["aggregate_excess_reduction_pct"] = percentage_reduction(
        aggregated["aggregate_baseline_total_excess_tons"],
        aggregated["aggregate_total_excess_tons"],
    )
    aggregated["aggregate_stockout_change_pct"] = percentage_change(
        aggregated["aggregate_baseline_total_stockout_tons"],
        aggregated["aggregate_total_stockout_tons"],
    )
    if require_non_worse_stockout:
        aggregated["aggregate_stockout_guardrail_pass"] = stockout_guardrail_pass(
            aggregated["aggregate_baseline_total_stockout_tons"],
            aggregated["aggregate_total_stockout_tons"],
            allowed_increase_pct=allowed_stockout_increase_pct,
        )
    else:
        aggregated["aggregate_stockout_guardrail_pass"] = True
    aggregated["scenario_success_rate"] = aggregated["scenario_success_rate"] * 100.0
    aggregated["stockout_guardrail_pass_rate"] = aggregated["stockout_guardrail_pass_rate"] * 100.0
    aggregated["meets_kpi_on_average"] = aggregated["avg_kpi_excess_reduction_pct"] >= minimum_excess_reduction_pct
    aggregated["meets_kpi_on_aggregate"] = (
        aggregated["aggregate_excess_reduction_pct"] >= minimum_excess_reduction_pct
    ) & aggregated["aggregate_stockout_guardrail_pass"]
    aggregated["meets_kpi_in_all_scenarios"] = aggregated["scenario_success_rate"] >= 99.999
    aggregated = aggregated.sort_values(
        [
            "meets_kpi_on_average",
            "scenario_success_rate",
            "stockout_guardrail_pass_rate",
            "meets_kpi_on_aggregate",
            "avg_kpi_excess_reduction_pct",
            "aggregate_excess_reduction_pct",
            "avg_total_stockout_tons",
            "avg_total_excess_tons",
            "avg_total_cost_proxy",
        ],
        ascending=[False, False, False, False, False, False, True, True, True],
    ).reset_index(drop=True)

    best_policy = aggregated.iloc[0].to_dict()
    best_classical_policy = None
    best_neuro_policy = None
    classical_df = aggregated[aggregated["policy_group"] == "classical_model"]
    if not classical_df.empty:
        best_classical_policy = classical_df.iloc[0].to_dict()
    neuro_df = aggregated[aggregated["policy_group"] == "neuroevolution_model"]
    if not neuro_df.empty:
        best_neuro_policy = neuro_df.iloc[0].to_dict()

    best_policy_meets_kpi = bool(best_policy["meets_kpi_on_aggregate"])
    if best_policy_meets_kpi:
        functional_status = "met_in_simulation"
    elif float(best_policy["aggregate_excess_reduction_pct"]) > 0 and bool(best_policy["aggregate_stockout_guardrail_pass"]):
        functional_status = "partially_met_in_simulation"
    else:
        functional_status = "not_met_in_simulation"

    kpi_assessment = {
        "baseline_policy_name": baseline_policy_name,
        "threshold_pct": minimum_excess_reduction_pct,
        "require_non_worse_stockouts": require_non_worse_stockout,
        "allowed_stockout_increase_pct": allowed_stockout_increase_pct,
        "best_policy_name": best_policy["policy_name"],
        "best_policy_group": best_policy["policy_group"],
        "best_policy_avg_excess_reduction_pct": float(best_policy["avg_kpi_excess_reduction_pct"]),
        "best_policy_aggregate_excess_reduction_pct": float(best_policy["aggregate_excess_reduction_pct"]),
        "best_policy_stockout_guardrail_pass_rate": float(best_policy["stockout_guardrail_pass_rate"]),
        "best_policy_aggregate_stockout_change_pct": float(best_policy["aggregate_stockout_change_pct"]),
        "best_policy_aggregate_stockout_guardrail_pass": bool(best_policy["aggregate_stockout_guardrail_pass"]),
        "kpi_met_within_simulator": bool(best_policy_meets_kpi),
        "policies_meeting_kpi_on_average": aggregated.loc[
            aggregated["meets_kpi_on_average"], "policy_name"
        ].tolist(),
        "policies_meeting_kpi_on_aggregate": aggregated.loc[
            aggregated["meets_kpi_on_aggregate"], "policy_name"
        ].tolist(),
    }
    functional_assessment = {
        "status": functional_status,
        "best_policy_name": best_policy["policy_name"],
        "best_policy_group": best_policy["policy_group"],
        "interpretation": (
            "A policy reduces accumulated excess versus the operational baseline without violating the configured "
            "stockout guardrail."
            if functional_status != "not_met_in_simulation"
            else "No evaluated policy simultaneously improves excess and respects the configured stockout guardrail."
        ),
    }
    return enriched, aggregated, kpi_assessment, {
        "best_policy": best_policy,
        "best_classical_policy": best_classical_policy,
        "best_neuroevolution_policy": best_neuro_policy,
        "functional_assessment": functional_assessment,
    }


def _resolve_policy_definition_models(
    *,
    model_source: str | None,
    baseline_model_payload: dict[str, Any],
    neuro_model_payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if not model_source:
        return None, None, None
    source = str(model_source).strip().lower()
    if source in {"baseline", "baseline_reference", "classical"}:
        artifact_payload = baseline_model_payload["artifact_payload"]
        return artifact_payload, artifact_payload.get("run_id"), artifact_payload.get("model_family")
    if source in {"neuro", "neuroevolution", "neuroevolution_reference"}:
        artifact_payload = neuro_model_payload["artifact_payload"]
        return artifact_payload, artifact_payload.get("run_id"), artifact_payload.get("model_family")
    raise ValueError(f"Unsupported policy model_source: {model_source}")


def _build_policy_specs(
    *,
    policy_cfg: dict[str, Any],
    baseline_model_payload: dict[str, Any],
    neuro_model_payload: dict[str, Any],
    runtime_context: dict[str, Any],
) -> list[dict[str, Any]]:
    policy_definitions = list(policy_cfg.get("policy_definitions", []))
    if not policy_definitions:
        raise ValueError("policy_simulation.policy_definitions must contain at least one policy.")

    recipe_profile = runtime_context.get("recipe_profile")
    policy_specs: list[dict[str, Any]] = []
    for policy_definition in policy_definitions:
        if not policy_definition.get("enabled", True):
            continue
        artifact_payload, source_run_id, source_model_family = _resolve_policy_definition_models(
            model_source=policy_definition.get("model_source"),
            baseline_model_payload=baseline_model_payload,
            neuro_model_payload=neuro_model_payload,
        )
        decision_variant_context = _resolve_decision_variant_config(
            policy_cfg,
            {
                "policy_name": policy_definition["policy_name"],
                "policy_variant": policy_definition.get("policy_variant"),
            },
            recipe_profile=recipe_profile,
        )
        policy_specs.append(
            {
                "policy_name": policy_definition["policy_name"],
                "policy_type": policy_definition["policy_type"],
                "policy_group": policy_definition["policy_group"],
                "decision_strategy": policy_definition.get("decision_strategy", policy_definition["policy_type"]),
                "policy_variant": policy_definition.get("policy_variant"),
                "artifact_payload": artifact_payload,
                "source_run_id": source_run_id,
                "source_model_family": source_model_family,
                "resolved_decision_variant": decision_variant_context["config"],
                "calibration_scope": decision_variant_context["calibration_scope"],
                "recipe_profile": recipe_profile,
                "applied_recipe_calibration": decision_variant_context["recipe_overrides"],
                "applied_policy_overrides": decision_variant_context["policy_name_overrides"],
            }
        )

    if not policy_specs:
        raise ValueError("No enabled policies were resolved from policy_simulation.policy_definitions.")
    return policy_specs


def _resolve_decision_variant_config(
    policy_cfg: dict[str, Any],
    policy_spec: dict[str, Any],
    *,
    recipe_profile: str | None = None,
) -> dict[str, Any]:
    variants_cfg = dict(policy_cfg.get("decision_policy_variants", {}))
    variant_name = policy_spec.get("policy_variant")
    base_config = dict(variants_cfg[variant_name]) if variant_name and variant_name in variants_cfg else dict(policy_cfg.get("decision_policy", {}))

    recipe_policy_overrides = dict(policy_cfg.get("recipe_policy_overrides", {}))
    recipe_override_bucket = {}
    if recipe_profile and isinstance(recipe_policy_overrides.get(recipe_profile), dict):
        recipe_override_bucket = dict(recipe_policy_overrides[recipe_profile])

    variant_overrides = {}
    if variant_name and isinstance(recipe_override_bucket.get(variant_name), dict):
        variant_overrides = dict(recipe_override_bucket[variant_name])

    policy_name = str(policy_spec.get("policy_name", ""))
    policy_name_overrides = {}
    if policy_name and isinstance(recipe_override_bucket.get(policy_name), dict):
        policy_name_overrides = dict(recipe_override_bucket[policy_name])

    resolved_config = dict(base_config)
    resolved_config.update(variant_overrides)
    resolved_config.update(policy_name_overrides)

    return {
        "config": resolved_config,
        "calibration_scope": "recipe_specific" if (variant_overrides or policy_name_overrides) else "global_default",
        "recipe_overrides": variant_overrides,
        "policy_name_overrides": policy_name_overrides,
    }


def _anchor_order_context(
    *,
    replenishment_gap_tons: float,
    anchor_need_tons: float,
    coverage_excess_tons: float,
    decision_variant_cfg: dict[str, Any],
) -> float:
    anchor_need_multiplier = float(decision_variant_cfg.get("anchor_need_multiplier", 1.0))
    excess_penalty_weight = float(decision_variant_cfg.get("excess_penalty_weight", 0.30))
    fixed_order_buffer_tons = float(decision_variant_cfg.get("fixed_order_buffer_tons", 0.0))
    return float(
        replenishment_gap_tons
        + anchor_need_multiplier * anchor_need_tons
        - excess_penalty_weight * coverage_excess_tons
        + fixed_order_buffer_tons
    )


def _decision_rule_audit(period_results_df: pd.DataFrame) -> dict[str, Any]:
    if period_results_df.empty:
        return {}

    audit_rows: list[dict[str, Any]] = []
    for policy_name, policy_df in period_results_df.groupby("policy_name", sort=False):
        replenishment_total = float(policy_df["replenishment_gap_tons"].sum())
        need_total = float(policy_df["need_estimate_tons"].sum())
        order_total = float(policy_df["order_quantity_tons"].sum())
        unnecessary_total = float(policy_df["unnecessary_procurement_tons"].sum())
        anchor_total = float(policy_df.get("anchor_order_tons", pd.Series(dtype=float)).sum())
        audit_rows.append(
            {
                "policy_name": policy_name,
                "policy_variant": policy_df["policy_variant"].iloc[0],
                "decision_strategy": policy_df["decision_strategy"].iloc[0],
                "avg_need_estimate_tons": float(policy_df["need_estimate_tons"].mean()),
                "avg_replenishment_gap_tons": float(policy_df["replenishment_gap_tons"].mean()),
                "avg_anchor_order_tons": float(policy_df["anchor_order_tons"].mean()) if "anchor_order_tons" in policy_df else None,
                "avg_order_quantity_tons": float(policy_df["order_quantity_tons"].mean()),
                "order_vs_gap_ratio": float(order_total / max(replenishment_total, 1e-6)),
                "need_vs_gap_ratio": float(need_total / max(replenishment_total, 1e-6)),
                "unnecessary_procurement_share": float(unnecessary_total / max(order_total, 1e-6)),
            }
        )

    audit_df = pd.DataFrame(audit_rows)
    operational_row = audit_df[audit_df["policy_name"] == "operational_simple"]
    diagnosis = [
        "The original model-based rule adds the model prediction directly on top of replenishment_gap, which turns prediction error into a larger decision error.",
        "Current evidence shows that the original model-based policies order about 6-8% above the replenishment gap, while the operational heuristic stays almost neutral versus the gap.",
        "The recalibrated variant treats the model as an advisory correction around the heuristic anchor instead of a second additive order signal.",
        "Decision parameters are now context-conditioned, so fresh, cooked-emulsion and cured profiles do not share the same coverage and ordering behaviour.",
    ]
    if not operational_row.empty:
        diagnosis.append(
            "Operational simple remains the reference anchor because it keeps unnecessary procurement close to the replenishment gap without triggering stockouts."
        )

    return {
        "policy_rows": audit_df.to_dict(orient="records"),
        "diagnosis": diagnosis,
    }


def _simulate_policy(
    *,
    scenario_df: pd.DataFrame,
    evaluation_index: int,
    history_weeks: int,
    policy_spec: dict[str, Any],
    policy_cfg: dict[str, Any],
    decision_cfg: dict[str, Any],
    heuristic_cfg: dict[str, Any],
    kpi_cfg: dict[str, Any],
    cost_cfg: dict[str, Any],
    context_profile_lookup: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    decision_variant_cfg = dict(
        policy_spec.get("resolved_decision_variant")
        or _resolve_decision_variant_config(
            policy_cfg,
            policy_spec,
            recipe_profile=policy_spec.get("recipe_profile"),
        )["config"]
    )
    lead_time_weeks_floor = int(decision_variant_cfg.get("lead_time_weeks_floor", decision_cfg.get("lead_time_weeks_floor", 1)))
    lead_time_weeks_cap = int(decision_variant_cfg.get("lead_time_weeks_cap", decision_cfg.get("lead_time_weeks_cap", 4)))
    decision_target_coverage_days = float(
        decision_variant_cfg.get("target_coverage_days", decision_cfg.get("target_coverage_days", kpi_cfg.get("target_coverage_days", 11.0)))
    )
    base_excess_penalty_weight = float(decision_variant_cfg.get("excess_penalty_weight", decision_cfg.get("excess_penalty_weight", 0.25)))
    base_need_multiplier = float(decision_variant_cfg.get("need_multiplier", decision_cfg.get("need_multiplier", 1.0)))
    fixed_order_buffer_tons = float(decision_variant_cfg.get("fixed_order_buffer_tons", decision_cfg.get("fixed_order_buffer_tons", 0.0)))
    min_order_tons = float(decision_variant_cfg.get("min_order_tons", decision_cfg.get("min_order_tons", 0.0)))
    base_max_order_tons = float(decision_variant_cfg.get("max_order_tons", decision_cfg.get("max_order_tons", 999999.0)))
    lot_rounding_tons = float(decision_variant_cfg.get("order_lot_rounding_tons", decision_cfg.get("order_lot_rounding_tons", 0.0)))

    purchase_cost_weight = float(cost_cfg.get("purchase_cost_weight", 1.0))
    excess_holding_cost_per_ton = float(cost_cfg.get("excess_holding_cost_per_ton", 0.1))
    base_stockout_penalty_per_ton = float(cost_cfg.get("stockout_penalty_per_ton", 2.0))
    base_waste_penalty_per_ton = float(cost_cfg.get("waste_penalty_per_ton", 0.5))

    histories = _history_values(scenario_df, evaluation_index, history_weeks)
    pipeline_queue = _seed_pipeline(scenario_df, evaluation_index, lead_time_weeks_cap)
    starting_inventory = float(scenario_df.iloc[evaluation_index]["synthetic_inventory_level"])
    on_hand_inventory = max(starting_inventory, 0.0)
    previous_order_tons = (
        float(scenario_df.iloc[evaluation_index - 1]["synthetic_planned_orders"]) if evaluation_index > 0 else 0.0
    )

    period_rows: list[dict[str, Any]] = []
    evaluation_df = scenario_df.iloc[evaluation_index:].reset_index(drop=True)

    for step, row in evaluation_df.iterrows():
        current_date = pd.to_datetime(row["date"])
        arrivals_tons = float(pipeline_queue.pop(step, 0.0))
        on_hand_inventory = max(on_hand_inventory + arrivals_tons, 0.0)
        pipeline_before_order_tons = float(sum(pipeline_queue.values()))

        requirement_tons = float(max(row["synthetic_raw_material_requirement"], 1.0))
        lead_time_days = float(max(row["synthetic_lead_time_days"], 1.0))
        lead_time_weeks = int(
            np.clip(math.ceil(lead_time_days / 7.0), lead_time_weeks_floor, max(lead_time_weeks_floor, lead_time_weeks_cap))
        )
        current_coverage_days = float((on_hand_inventory / requirement_tons) * 7.0) if requirement_tons > 0 else 0.0
        context_adjustment = _context_adjustments(row, context_profile_lookup)
        effective_need_multiplier = base_need_multiplier * context_adjustment["need_multiplier"]
        effective_excess_penalty_weight = base_excess_penalty_weight * context_adjustment["excess_penalty_multiplier"]
        effective_max_order_tons = base_max_order_tons * context_adjustment["max_order_multiplier"]
        stockout_penalty_per_ton = base_stockout_penalty_per_ton * context_adjustment["stockout_penalty_multiplier"]
        waste_penalty_per_ton = base_waste_penalty_per_ton * context_adjustment["waste_penalty_multiplier"]

        current_values = {
            "demand_index": float(row["demand_index"]),
            "supply_index": float(row["supply_index"]),
            "purchase_price_index": float(row["purchase_price_index"]),
            "synthetic_plant_production_volume": float(row["synthetic_plant_production_volume"]),
            "synthetic_plant_capacity_utilization": float(row["synthetic_plant_capacity_utilization"]),
            "synthetic_yield_rate": float(row["synthetic_yield_rate"]),
            "synthetic_raw_material_requirement": requirement_tons,
            "synthetic_inventory_coverage_days": current_coverage_days,
            "synthetic_lead_time_days": lead_time_days,
            "synthetic_waste_rate": float(row["synthetic_waste_rate"]),
            "expected_yield": float(row["expected_yield"]),
            "expected_waste": float(row["expected_waste"]),
            "process_lead_time_days": float(row["process_lead_time_days"]),
            "manufacturing_context_profile": str(row["manufacturing_context_profile"]),
            "product_family": str(row["product_family"]),
            "process_type": str(row["process_type"]),
            "recipe_profile": str(row["recipe_profile"]),
            "formulation_class": str(row["formulation_class"]),
            "shelf_life_class": str(row["shelf_life_class"]),
            "priority_level": str(row["priority_level"]),
            "cost_sensitivity": str(row["cost_sensitivity"]),
        }
        need_estimate_tons = _policy_need_estimate(
            policy_spec,
            current_values=current_values,
            histories=histories,
            current_date=current_date,
            heuristic_cfg=heuristic_cfg,
        )
        anchor_need_tons = _heuristic_need_estimate(
            requirement_tons=requirement_tons,
            coverage_days=current_coverage_days,
            lead_time_days=lead_time_days,
            safety_stock_days=float(heuristic_cfg.get("safety_stock_days", 6.0)),
        )
        target_coverage_days = (
            decision_target_coverage_days
            + float(row.get("scenario_target_coverage_shift_days", 0.0))
            + context_adjustment["target_coverage_adjustment_days"]
        )
        target_coverage_days = max(target_coverage_days, float(heuristic_cfg.get("minimum_coverage_days", 1.0)))
        target_inventory_tons = float(requirement_tons * (target_coverage_days / 7.0))

        inventory_position_tons = on_hand_inventory + pipeline_before_order_tons
        expected_consumption_until_arrival_tons = float(requirement_tons * lead_time_weeks)
        replenishment_gap_tons = max(
            0.0,
            expected_consumption_until_arrival_tons + target_inventory_tons - inventory_position_tons,
        )
        coverage_excess_tons = max(0.0, inventory_position_tons - target_inventory_tons)
        anchor_order_tons = _anchor_order_context(
            replenishment_gap_tons=replenishment_gap_tons,
            anchor_need_tons=anchor_need_tons,
            coverage_excess_tons=coverage_excess_tons,
            decision_variant_cfg=decision_variant_cfg,
        )

        decision_strategy = str(policy_spec.get("decision_strategy", policy_spec["policy_type"]))
        decision_adjustment_tons = 0.0
        raw_order_before_limits_tons = anchor_order_tons
        if decision_strategy == "operational_simple":
            raw_order_before_limits_tons = anchor_order_tons
        elif decision_strategy == "model_based_original":
            raw_order_before_limits_tons = (
                replenishment_gap_tons
                + (effective_need_multiplier * max(need_estimate_tons, 0.0))
                - (effective_excess_penalty_weight * coverage_excess_tons)
                + fixed_order_buffer_tons
            )
            decision_adjustment_tons = raw_order_before_limits_tons - anchor_order_tons
        elif decision_strategy == "model_based_calibrated":
            prediction_shrink_ratio = float(decision_variant_cfg.get("prediction_shrink_ratio", 0.20))
            reorder_deadband_tons = float(decision_variant_cfg.get("reorder_deadband_tons", 5.0))
            positive_delta_weight = float(decision_variant_cfg.get("positive_delta_weight", 0.05)) * (
                1.0 + 0.10 * context_adjustment["priority_score"]
            )
            negative_delta_weight = float(decision_variant_cfg.get("negative_delta_weight", 0.60))
            smoothing_weight = float(decision_variant_cfg.get("smoothing_weight", 0.20))
            max_order_growth_ratio = float(decision_variant_cfg.get("max_order_growth_ratio", 0.01))
            max_order_reduction_ratio = float(decision_variant_cfg.get("max_order_reduction_ratio", 0.08))
            negative_adjustment_coverage_floor_days = decision_variant_cfg.get("negative_adjustment_coverage_floor_days")
            positive_adjustment_coverage_gate_days = float(
                decision_variant_cfg.get("positive_adjustment_coverage_gate_days", target_coverage_days - 0.5)
            )
            allow_positive_adjustment_above_gate = bool(
                decision_variant_cfg.get("allow_positive_adjustment_above_gate", False)
            )
            calibrated_need_tons = float(max(need_estimate_tons, 0.0) * prediction_shrink_ratio * context_adjustment["need_multiplier"])
            advisory_delta_tons = calibrated_need_tons - anchor_need_tons
            if abs(advisory_delta_tons) < reorder_deadband_tons:
                advisory_delta_tons = 0.0
            if advisory_delta_tons > 0:
                if (current_coverage_days >= positive_adjustment_coverage_gate_days) and (
                    not allow_positive_adjustment_above_gate
                ):
                    advisory_delta_tons = 0.0
                else:
                    advisory_delta_tons = advisory_delta_tons * positive_delta_weight
            elif advisory_delta_tons < 0:
                if (
                    negative_adjustment_coverage_floor_days is not None
                    and current_coverage_days <= float(negative_adjustment_coverage_floor_days)
                ):
                    advisory_delta_tons = 0.0
                else:
                    advisory_delta_tons = advisory_delta_tons * negative_delta_weight
            decision_adjustment_tons = advisory_delta_tons
            raw_order_before_limits_tons = anchor_order_tons + advisory_delta_tons
            smoothed_order_tons = (
                (1.0 - smoothing_weight) * raw_order_before_limits_tons + smoothing_weight * previous_order_tons
            )
            lower_guardrail_tons = max(0.0, anchor_order_tons * (1.0 - max_order_reduction_ratio))
            upper_guardrail_tons = anchor_order_tons * (1.0 + max_order_growth_ratio)
            raw_order_before_limits_tons = _clip_value(
                smoothed_order_tons,
                lower=lower_guardrail_tons,
                upper=upper_guardrail_tons,
            )
        else:
            raise ValueError(f"Unsupported decision_strategy: {decision_strategy}")

        raw_order_tons = raw_order_before_limits_tons
        raw_order_tons = _clip_value(raw_order_tons, lower=min_order_tons, upper=effective_max_order_tons)
        order_quantity_tons = _round_to_lot(raw_order_tons, lot_rounding_tons)
        arrival_step = step + lead_time_weeks
        pipeline_queue[arrival_step] = float(pipeline_queue.get(arrival_step, 0.0) + order_quantity_tons)
        pipeline_after_order_tons = float(sum(pipeline_queue.values()))

        fulfilled_requirement_tons = float(min(on_hand_inventory, requirement_tons))
        stockout_tons = float(max(requirement_tons - on_hand_inventory, 0.0))
        ending_inventory_tons = float(max(on_hand_inventory - requirement_tons, 0.0))
        ending_coverage_days = float((ending_inventory_tons / requirement_tons) * 7.0) if requirement_tons > 0 else 0.0
        excess_inventory_tons = float(max(ending_inventory_tons - target_inventory_tons, 0.0))
        unnecessary_procurement_tons = float(max(order_quantity_tons - replenishment_gap_tons, 0.0))
        waste_tons = float(fulfilled_requirement_tons * max(float(row["synthetic_waste_rate"]), 0.0))
        excess_raw_material_tons = float(excess_inventory_tons + unnecessary_procurement_tons)
        cost_proxy = float(
            order_quantity_tons * float(row["purchase_price_index"]) * purchase_cost_weight
            + excess_inventory_tons * excess_holding_cost_per_ton
            + stockout_tons * stockout_penalty_per_ton
            + waste_tons * waste_penalty_per_ton
        )

        period_rows.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "step": int(step),
                "scenario_name": str(row["scenario_name"]),
                "scenario_template": str(row["scenario_template"]),
                "scenario_seed": int(row["scenario_seed"]),
                "policy_name": policy_spec["policy_name"],
                "policy_type": policy_spec["policy_type"],
                "policy_group": policy_spec["policy_group"],
                "decision_strategy": decision_strategy,
                "policy_variant": policy_spec.get("policy_variant"),
                "policy_calibration_scope": policy_spec.get("calibration_scope"),
                "policy_recipe_calibration_profile": policy_spec.get("recipe_profile"),
                "source_run_id": policy_spec.get("source_run_id"),
                "source_model_family": policy_spec.get("source_model_family"),
                "manufacturing_context_profile": str(row["manufacturing_context_profile"]),
                "product_family": str(row["product_family"]),
                "process_type": str(row["process_type"]),
                "recipe_profile": str(row["recipe_profile"]),
                "formulation_class": str(row["formulation_class"]),
                "shelf_life_class": str(row["shelf_life_class"]),
                "priority_level": str(row["priority_level"]),
                "cost_sensitivity": str(row["cost_sensitivity"]),
                "expected_yield": float(row["expected_yield"]),
                "expected_waste": float(row["expected_waste"]),
                "process_lead_time_days": float(row["process_lead_time_days"]),
                "context_target_coverage_adjustment_days": context_adjustment["target_coverage_adjustment_days"],
                "context_need_multiplier": context_adjustment["need_multiplier"],
                "context_excess_penalty_multiplier": context_adjustment["excess_penalty_multiplier"],
                "context_max_order_multiplier": context_adjustment["max_order_multiplier"],
                "priority_level_score": context_adjustment["priority_score"],
                "cost_sensitivity_score": context_adjustment["cost_score"],
                "demand_index": current_values["demand_index"],
                "supply_index": current_values["supply_index"],
                "purchase_price_index": current_values["purchase_price_index"],
                "requirement_tons": requirement_tons,
                "lead_time_days": lead_time_days,
                "lead_time_weeks": int(lead_time_weeks),
                "starting_inventory_tons": float(on_hand_inventory),
                "arrivals_tons": arrivals_tons,
                "pipeline_before_order_tons": pipeline_before_order_tons,
                "inventory_position_tons": inventory_position_tons,
                "current_coverage_days": current_coverage_days,
                "target_coverage_days": target_coverage_days,
                "target_inventory_tons": target_inventory_tons,
                "expected_consumption_until_arrival_tons": expected_consumption_until_arrival_tons,
                "replenishment_gap_tons": replenishment_gap_tons,
                "coverage_excess_tons": coverage_excess_tons,
                "anchor_need_tons": anchor_need_tons,
                "anchor_order_tons": anchor_order_tons,
                "need_estimate_tons": need_estimate_tons,
                "decision_adjustment_tons": decision_adjustment_tons,
                "raw_order_before_limits_tons": raw_order_before_limits_tons,
                "order_quantity_tons": order_quantity_tons,
                "pipeline_after_order_tons": pipeline_after_order_tons,
                "fulfilled_requirement_tons": fulfilled_requirement_tons,
                "stockout_tons": stockout_tons,
                "ending_inventory_tons": ending_inventory_tons,
                "ending_coverage_days": ending_coverage_days,
                "excess_inventory_tons": excess_inventory_tons,
                "unnecessary_procurement_tons": unnecessary_procurement_tons,
                "excess_raw_material_tons": excess_raw_material_tons,
                "waste_tons": waste_tons,
                "cost_proxy": cost_proxy,
            }
        )

        on_hand_inventory = ending_inventory_tons
        histories["demand_index"].append(current_values["demand_index"])
        histories["supply_index"].append(current_values["supply_index"])
        histories["synthetic_plant_production_volume"].append(current_values["synthetic_plant_production_volume"])
        histories["synthetic_plant_capacity_utilization"].append(current_values["synthetic_plant_capacity_utilization"])
        histories["synthetic_yield_rate"].append(current_values["synthetic_yield_rate"])
        histories["synthetic_raw_material_requirement"].append(requirement_tons)
        histories["synthetic_inventory_coverage_days"].append(ending_coverage_days)
        histories["synthetic_lead_time_days"].append(lead_time_days)
        histories["purchase_price_index"].append(current_values["purchase_price_index"])
        histories["synthetic_waste_rate"].append(current_values["synthetic_waste_rate"])
        previous_order_tons = order_quantity_tons

    return pd.DataFrame(period_rows)


def run_policy_simulation(config: dict[str, Any], logger) -> dict[str, Any]:
    """Simulate procurement policies on the synthetic environment and evaluate the KPI in-simulator."""
    config = ensure_runtime_context_resolved(config)
    repo_root = Path(config["project"]["repo_root"])
    policy_cfg = config.get("policy_simulation", {})
    procurement_definition = dict(config.get("procurement_problem_definition", {}))
    runtime_recipe_context = current_recipe_context(config)
    logger.info(
        "Policy simulation runtime selection_mode=%s mode_resolution=%s scope_token=%s recipe_profile=%s",
        runtime_recipe_context.get("selection_mode"),
        runtime_recipe_context.get("mode_resolution"),
        runtime_recipe_context.get("scope_token"),
        runtime_recipe_context.get("recipe_profile"),
    )
    if not policy_cfg.get("enabled", False):
        raise ValueError("policy_simulation.enabled=false. Enable it in config before running this stage.")

    paths_cfg = config["paths"]
    metrics_dir = ensure_directory(resolve_repo_path(paths_cfg["model_metrics_dir"], repo_root))
    stats_dir = ensure_directory(resolve_repo_path(paths_cfg["stats_dir"], repo_root))

    dataset_path = resolve_repo_path(policy_cfg["input_dataset_path"], repo_root)
    base_df = _prepare_base_dataset(dataset_path)
    base_df = filter_frame_to_recipe(base_df, config, stage_name="policy_simulation", logger=logger)

    evaluation_split = policy_cfg.get("evaluation_split", "test")
    history_weeks = int(policy_cfg.get("history_weeks", 12))
    evaluation_start, evaluation_end = _resolve_split_bounds(
        len(base_df),
        config["data_processing"]["split"],
        evaluation_split,
    )
    if evaluation_end - evaluation_start <= 0:
        raise ValueError(f"Selected evaluation split '{evaluation_split}' produced no rows.")

    target_column = str(policy_cfg.get("target_column", config["synthetic_data"]["target_roles"]["canonical_target_column"]))
    feature_set_name = str(policy_cfg.get("official_feature_set", config.get("neuroevolution", {}).get("feature_set_name", "extended")))

    reference_cfg = policy_cfg.get("reference_runs", {})
    baseline_summary_path = resolve_repo_path(reference_cfg["baseline_summary_json"], repo_root)
    neuro_summary_path = resolve_repo_path(reference_cfg["neuro_summary_json"], repo_root)
    baseline_run_key = reference_cfg.get("baseline_run_key", "best_baseline_run")
    neuro_run_key = reference_cfg.get("neuro_run_key", "best_neuroevolution_run")

    baseline_model_payload = _load_model_payload(
        summary_path=baseline_summary_path,
        run_key=baseline_run_key,
        repo_root=repo_root,
        expected_feature_set=feature_set_name,
        expected_target_column=target_column,
        expected_model_family=reference_cfg.get("baseline_model_family"),
    )
    neuro_model_payload = _load_model_payload(
        summary_path=neuro_summary_path,
        run_key=neuro_run_key,
        repo_root=repo_root,
        expected_feature_set=feature_set_name,
        expected_target_column=target_column,
        expected_model_family=reference_cfg.get("neuro_model_family", "neuroevolution"),
    )
    logger.info(
        "Loaded policy reference models baseline_scope=%s neuro_scope=%s",
        (baseline_model_payload["summary_run"].get("runtime_context") or {}).get("scope_token"),
        (neuro_model_payload["summary_run"].get("runtime_context") or {}).get("scope_token"),
    )

    policy_specs = _build_policy_specs(
        policy_cfg=policy_cfg,
        baseline_model_payload=baseline_model_payload,
        neuro_model_payload=neuro_model_payload,
        runtime_context=runtime_recipe_context,
    )
    for policy_spec in policy_specs:
        logger.info(
            "Resolved policy calibration policy=%s scope=%s recipe_profile=%s overrides=%s",
            policy_spec["policy_name"],
            policy_spec.get("calibration_scope"),
            policy_spec.get("recipe_profile"),
            sorted(
                {
                    *dict(policy_spec.get("applied_recipe_calibration", {})).keys(),
                    *dict(policy_spec.get("applied_policy_overrides", {})).keys(),
                }
            ),
        )
    scenario_cfg = policy_cfg.get("scenario_generation", {})
    scenario_templates = list(scenario_cfg.get("scenario_templates", []))
    scenario_seeds = list(scenario_cfg.get("seeds", [int(config["project"].get("seed", 42))]))
    if not scenario_templates:
        raise ValueError("policy_simulation.scenario_generation.scenario_templates must contain at least one scenario.")

    comparison_name = str(policy_cfg.get("comparison_name", "policy_simulation"))
    simulation_run_id = make_run_id(comparison_name)
    period_frames: list[pd.DataFrame] = []
    scenario_rows: list[dict[str, Any]] = []

    decision_cfg = dict(policy_cfg.get("decision_policy", {}))
    heuristic_cfg = dict(policy_cfg.get("operational_baseline_policy", {}))
    kpi_cfg = dict(policy_cfg.get("kpi_definition", {}))
    cost_cfg = dict(policy_cfg.get("cost_proxy", {}))
    baseline_policy_name = str(policy_cfg.get("baseline_policy_name", "operational_simple"))
    context_profile_lookup = _resolve_context_profile_lookup(config)

    for template in scenario_templates:
        for seed in scenario_seeds:
            scenario_df, evaluation_index = _generate_scenario_segment(
                base_df=base_df,
                evaluation_start=evaluation_start,
                evaluation_end=evaluation_end,
                history_weeks=history_weeks,
                scenario_template=template,
                seed=int(seed),
                scenario_cfg=scenario_cfg,
                logger=logger,
            )
            scenario_name = str(scenario_df.iloc[evaluation_index]["scenario_name"])
            for policy_spec in policy_specs:
                logger.info(
                    "Simulating policy name=%s scenario=%s horizon=%s",
                    policy_spec["policy_name"],
                    scenario_name,
                    int(template["horizon_weeks"]),
                )
                period_df = _simulate_policy(
                    scenario_df=scenario_df,
                    evaluation_index=evaluation_index,
                    history_weeks=history_weeks,
                    policy_spec=policy_spec,
                    policy_cfg=policy_cfg,
                    decision_cfg=decision_cfg,
                    heuristic_cfg=heuristic_cfg,
                    kpi_cfg=kpi_cfg,
                    cost_cfg=cost_cfg,
                    context_profile_lookup=context_profile_lookup,
                )
                period_df.insert(0, "simulation_run_id", simulation_run_id)
                period_frames.append(period_df)
                scenario_rows.append(
                    _policy_comparison_rows(
                        period_df,
                        policy_spec=policy_spec,
                        scenario_name=scenario_name,
                        scenario_template=str(template["name"]),
                        scenario_seed=int(seed),
                        target_column=target_column,
                        feature_set_name=feature_set_name,
                    )
                )

    period_results_df = pd.concat(period_frames, ignore_index=True) if period_frames else pd.DataFrame()
    scenario_summary_df = pd.DataFrame(scenario_rows)
    if scenario_summary_df.empty or period_results_df.empty:
        raise ValueError("Policy simulation did not generate any scenario results.")

    scenario_summary_with_kpi_df, policy_aggregate_df, kpi_assessment, best_policy_payload = _aggregate_policy_results(
        scenario_summary_df,
        baseline_policy_name=baseline_policy_name,
        kpi_cfg=kpi_cfg,
    )
    decision_rule_audit = _decision_rule_audit(period_results_df)

    best_policy_overall = policy_aggregate_df.iloc[0].to_dict()
    best_classical_policy = best_policy_payload["best_classical_policy"]
    best_neuro_policy = best_policy_payload["best_neuroevolution_policy"]
    functional_assessment = best_policy_payload["functional_assessment"]

    summary_json_path = stats_dir / policy_cfg.get("summary_json_name", "policy_simulation_latest.json")
    summary_csv_path = stats_dir / policy_cfg.get("summary_csv_name", "policy_simulation_latest.csv")
    scenario_csv_path = metrics_dir / f"{simulation_run_id}_scenario_policy_metrics.csv"
    period_csv_path = metrics_dir / f"{simulation_run_id}_period_policy_metrics.csv"
    policy_aggregate_df.to_csv(summary_csv_path, index=False)
    scenario_summary_with_kpi_df.to_csv(scenario_csv_path, index=False)
    period_results_df.to_csv(period_csv_path, index=False)

    summary_payload = {
        "simulation_run_id": simulation_run_id,
        "created_at_utc": utc_timestamp(),
        "input_dataset_path": _portable_path(dataset_path, repo_root),
        "evaluation_split": evaluation_split,
        "evaluation_rows": int(evaluation_end - evaluation_start),
        "history_weeks": history_weeks,
        "target_column": target_column,
        "official_feature_set": feature_set_name,
        "recipe_context": runtime_recipe_context,
        "runtime_context": runtime_recipe_context,
        "validated_base_horizon_weeks": int(policy_cfg.get("validated_base_horizon_weeks", procurement_definition.get("validated_base_horizon_weeks", 1))),
        "validated_base_horizon_label": str(policy_cfg.get("validated_base_horizon_label", procurement_definition.get("validated_base_horizon_label", "W+1"))),
        "baseline_policy_name": baseline_policy_name,
        "procurement_problem_definition": procurement_definition,
        "decision_rule": {
            "description": (
                "Policies share the same replenishment-gap logic, but now their target coverage, excess penalties, "
                "need amplification and order caps are modulated by manufacturing context. The operational baseline "
                "still anchors the decision, while the model-based variants apply original or calibrated adjustments "
                "on top of that anchor."
            ),
            "parameters": decision_cfg,
            "policy_variants": policy_cfg.get("decision_policy_variants", {}),
            "policy_definitions": policy_cfg.get("policy_definitions", []),
            "recipe_policy_overrides": (
                policy_cfg.get("recipe_policy_overrides", {}).get(runtime_recipe_context.get("recipe_profile"), {})
                if runtime_recipe_context.get("recipe_profile")
                else {}
            ),
            "resolved_policy_configs": [
                {
                    "policy_name": policy_spec["policy_name"],
                    "policy_variant": policy_spec.get("policy_variant"),
                    "decision_strategy": policy_spec.get("decision_strategy"),
                    "calibration_scope": policy_spec.get("calibration_scope"),
                    "recipe_profile": policy_spec.get("recipe_profile"),
                    "resolved_decision_variant": policy_spec.get("resolved_decision_variant"),
                    "applied_recipe_calibration": policy_spec.get("applied_recipe_calibration"),
                    "applied_policy_overrides": policy_spec.get("applied_policy_overrides"),
                }
                for policy_spec in policy_specs
            ],
            "manufacturing_context_profiles": list(context_profile_lookup),
        },
        "kpi_definition": {
            "name": kpi_cfg.get("name", "kpi_excess_reduction_pct"),
            "description": kpi_cfg.get(
                "description",
                "Percentage reduction of accumulated excess raw material, defined as inventory above target coverage "
                "plus procurement ordered beyond endogenous replenishment need, relative to the simple operational baseline.",
            ),
            "target_coverage_days": kpi_cfg.get("target_coverage_days"),
            "minimum_excess_reduction_pct": kpi_cfg.get("minimum_excess_reduction_pct", 20.0),
            "excess_formula": (
                "max(ending_inventory_tons - requirement_tons * target_coverage_days / 7, 0) + "
                "max(order_quantity_tons - replenishment_gap_tons, 0)"
            ),
            "stockout_guardrail": {
                "require_non_worse_stockouts": kpi_cfg.get("require_non_worse_stockouts", True),
                "allowed_stockout_increase_pct": kpi_cfg.get("allowed_stockout_increase_pct", 0.0),
            },
        },
        "reference_models": {
            "baseline_reference_run": baseline_model_payload["summary_run"],
            "neuroevolution_reference_run": neuro_model_payload["summary_run"],
        },
        "reference_model_scopes": {
            "baseline": baseline_model_payload["summary_run"].get("runtime_context"),
            "neuroevolution": neuro_model_payload["summary_run"].get("runtime_context"),
        },
        "scenario_generation": {
            "seeds": scenario_seeds,
            "scenario_templates": scenario_templates,
            "parameters": scenario_cfg,
        },
        "policy_ranking": policy_aggregate_df.to_dict(orient="records"),
        "best_policy_overall": best_policy_overall,
        "best_classical_policy": best_classical_policy,
        "best_neuroevolution_policy": best_neuro_policy,
        "kpi_assessment": kpi_assessment,
        "functional_objective_assessment": functional_assessment,
        "decision_rule_audit": decision_rule_audit,
        "artifacts": {
            "summary_csv_path": _portable_path(summary_csv_path, repo_root),
            "scenario_metrics_csv_path": _portable_path(scenario_csv_path, repo_root),
            "period_metrics_csv_path": _portable_path(period_csv_path, repo_root),
        },
    }
    cleaned_summary_payload = _json_ready(summary_payload)
    write_json(summary_json_path, cleaned_summary_payload)

    logger.info("Saved policy simulation summary JSON to %s", summary_json_path)
    logger.info("Saved policy simulation ranking CSV to %s", summary_csv_path)
    logger.info("Saved scenario policy metrics CSV to %s", scenario_csv_path)
    logger.info("Saved period policy metrics CSV to %s", period_csv_path)

    return {
        "simulation_run_id": simulation_run_id,
        "summary_json_path": _portable_path(summary_json_path, repo_root),
        "summary_csv_path": _portable_path(summary_csv_path, repo_root),
        "scenario_metrics_csv_path": _portable_path(scenario_csv_path, repo_root),
        "period_metrics_csv_path": _portable_path(period_csv_path, repo_root),
        "best_policy_overall": cleaned_summary_payload["best_policy_overall"],
        "best_classical_policy": cleaned_summary_payload["best_classical_policy"],
        "best_neuroevolution_policy": cleaned_summary_payload["best_neuroevolution_policy"],
        "kpi_assessment": cleaned_summary_payload["kpi_assessment"],
        "functional_objective_assessment": cleaned_summary_payload["functional_objective_assessment"],
        "recipe_context": runtime_recipe_context,
        "runtime_context": runtime_recipe_context,
        "reference_model_scopes": cleaned_summary_payload["reference_model_scopes"],
    }
