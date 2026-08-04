"""Feature engineering reporting helpers for notebooks and pipeline design."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _source_group_for_feature(feature_name: str, datetime_column: str) -> str:
    if feature_name == datetime_column:
        return datetime_column
    if "__" in feature_name:
        return feature_name.split("__", 1)[0]
    if feature_name.startswith("date_"):
        return datetime_column
    if "_lag_" in feature_name:
        return feature_name.split("_lag_")[0]
    if "_roll_" in feature_name:
        return feature_name.split("_roll_")[0]
    if feature_name.endswith("_gap"):
        return feature_name.replace("_gap", "")
    if feature_name.endswith("_ratio"):
        return feature_name.replace("_ratio", "")
    return feature_name


def _feature_origin_for_feature(
    feature_name: str,
    *,
    datetime_column: str,
    raw_feature_columns: list[str],
    context_columns: list[str],
) -> str:
    source_group = _source_group_for_feature(feature_name, datetime_column)
    if feature_name == datetime_column:
        return "temporal_index"
    if feature_name.startswith("date_"):
        return "calendar_derived"
    if source_group in context_columns:
        return "manufacturing_context"
    if feature_name.endswith("_gap") or feature_name.endswith("_ratio"):
        return "derived_ratio_gap"
    if "_lag_" in feature_name or "_roll_" in feature_name:
        return "autoregressive"
    if source_group.startswith("synthetic_") or feature_name.startswith("synthetic_"):
        return "synthetic_internal"
    if source_group in raw_feature_columns:
        return "external_proxy"
    return "derived"


def _feature_type_for_column(
    feature_name: str,
    series: pd.Series,
    *,
    datetime_column: str,
    raw_feature_columns: list[str],
    context_columns: list[str],
    categorical_threshold: int,
) -> str:
    source_group = _source_group_for_feature(feature_name, datetime_column)
    if feature_name == datetime_column:
        return "temporal"
    if feature_name.startswith("date_"):
        return "derived_temporal"
    if source_group in context_columns:
        if "__" in feature_name:
            return "categorical_context"
        if pd.api.types.is_numeric_dtype(series):
            return "numerical_context"
        return "categorical_context"
    if feature_name in raw_feature_columns:
        if pd.api.types.is_numeric_dtype(series):
            return "numerical"
        unique_count = series.nunique(dropna=True)
        return "categorical" if unique_count <= categorical_threshold else "review"
    if "_lag_" in feature_name or "_roll_" in feature_name or feature_name.endswith("_gap") or feature_name.endswith("_ratio"):
        return "derived"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    return "categorical"


def _default_leakage_risk(feature_name: str, *, target_column: str, target_base_column: str, datetime_column: str) -> str:
    if feature_name == target_column:
        return "high_target_column"
    if "order_quantity_tons" in feature_name:
        return "high_downstream_decision"
    if feature_name.startswith("synthetic_procurement_need") and feature_name != target_column:
        return "high_target_family"
    if feature_name == datetime_column:
        return "low"
    if feature_name == target_base_column or feature_name.startswith(f"{target_base_column}_"):
        return "low_autoregressive"
    return "low"


def _feature_layer_for_feature(
    feature_name: str,
    *,
    target_column: str,
    datetime_column: str,
    raw_feature_columns: list[str],
    context_columns: list[str],
    decision_output_columns: list[str],
    alternative_target_columns: list[str],
) -> str:
    source_group = _source_group_for_feature(feature_name, datetime_column)
    if feature_name == datetime_column or feature_name.startswith("date_"):
        return "temporal_reference"
    if feature_name == target_column:
        return "target_supervision"
    if feature_name in decision_output_columns or source_group in decision_output_columns:
        return "downstream_decision"
    if feature_name.startswith("synthetic_procurement_need") or source_group in alternative_target_columns:
        return "target_like_diagnostic"
    if source_group in context_columns:
        return "manufacturing_context"
    if source_group.startswith("synthetic_") or feature_name.startswith("synthetic_"):
        return "synthetic_operational_state"
    if source_group in raw_feature_columns:
        return "external_proxy_upstream"
    return "engineered_feature"


def _temporal_relation_for_feature(feature_name: str, *, target_column: str, datetime_column: str) -> str:
    if feature_name == datetime_column:
        return "temporal_index"
    if feature_name == target_column:
        return "W_plus_1_supervision"
    if "_t_plus_" in feature_name:
        return "future_not_allowed"
    if "_lag_" in feature_name:
        return "historical_lag"
    if "_roll_" in feature_name:
        return "historical_rolling"
    if feature_name.startswith("date_"):
        return "calendar_at_t"
    if "order_quantity_tons" in feature_name:
        return "downstream_after_prediction"
    return "current_t_context"


def _output_role_for_feature(feature_name: str, *, target_column: str, decision_output_columns: list[str]) -> str:
    if feature_name == target_column:
        return "predictive_target"
    if feature_name in decision_output_columns or any(feature_name.startswith(f"{column}_") for column in decision_output_columns):
        return "decision_output"
    return "input_or_diagnostic"


def build_feature_catalog(
    df: pd.DataFrame,
    *,
    target_column: str,
    target_base_column: str,
    datetime_column: str,
    raw_feature_columns: list[str],
    context_columns: list[str] | None = None,
    decision_output_columns: list[str] | None = None,
    alternative_target_columns: list[str] | None = None,
    categorical_threshold: int = 20,
    review_corr_threshold: float = 0.97,
    high_corr_threshold: float = 0.995,
) -> pd.DataFrame:
    """Build a traceable feature catalog with basic statistics and heuristics."""
    context_columns = list(context_columns or [])
    decision_output_columns = list(decision_output_columns or [])
    alternative_target_columns = list(alternative_target_columns or [])
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    corr_matrix = numeric_df.corr().abs() if not numeric_df.empty else pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for feature_name in df.columns:
        series = df[feature_name]
        null_ratio = float(series.isna().mean())
        cardinality = int(series.nunique(dropna=True)) if feature_name != datetime_column else int(series.nunique(dropna=True))
        top_freq_ratio = float(series.value_counts(dropna=False, normalize=True).iloc[0]) if len(series) else 0.0
        feature_type = _feature_type_for_column(
            feature_name,
            series,
            datetime_column=datetime_column,
            raw_feature_columns=raw_feature_columns,
            context_columns=context_columns,
            categorical_threshold=categorical_threshold,
        )
        feature_layer = _feature_layer_for_feature(
            feature_name,
            target_column=target_column,
            datetime_column=datetime_column,
            raw_feature_columns=raw_feature_columns,
            context_columns=context_columns,
            decision_output_columns=decision_output_columns,
            alternative_target_columns=alternative_target_columns,
        )
        temporal_relation = _temporal_relation_for_feature(
            feature_name,
            target_column=target_column,
            datetime_column=datetime_column,
        )
        output_role = _output_role_for_feature(
            feature_name,
            target_column=target_column,
            decision_output_columns=decision_output_columns,
        )

        correlation_flag = "not_applicable"
        notes = ""
        if feature_name in corr_matrix.columns:
            correlations = corr_matrix[feature_name].drop(labels=[feature_name]).dropna().sort_values(ascending=False)
            if not correlations.empty:
                top_feature = correlations.index[0]
                top_corr = float(correlations.iloc[0])
                notes = f"top_abs_corr={top_feature}:{top_corr:.4f}"
                if top_corr >= high_corr_threshold:
                    correlation_flag = f"high_redundancy_with:{top_feature}"
                elif top_corr >= review_corr_threshold:
                    correlation_flag = f"review_redundancy_with:{top_feature}"
                else:
                    correlation_flag = f"ok_top_corr:{top_feature}:{top_corr:.4f}"

        leakage_risk = _default_leakage_risk(
            feature_name,
            target_column=target_column,
            target_base_column=target_base_column,
            datetime_column=datetime_column,
        )

        if feature_name == target_column:
            status = "drop"
            justification = "Future target reserved for supervision only; exclude from predictors."
        elif feature_name == datetime_column:
            status = "drop"
            justification = "Reference temporal index only; prefer derived calendar features."
        elif cardinality <= 1 or top_freq_ratio >= 0.995:
            status = "drop"
            justification = "Constant or quasi-constant column without usable predictive signal."
        elif null_ratio > 0.50:
            status = "review"
            justification = "High missingness; review before using in baseline models."
        elif correlation_flag.startswith("high_redundancy_with:"):
            status = "review"
            justification = "Potentially informative but highly redundant with another feature."
        else:
            status = "keep"
            if feature_type == "numerical":
                justification = "Observed numerical signal available in the processed context."
            elif feature_type == "derived_temporal":
                justification = "Calendar feature derived from the temporal index."
            elif feature_type == "derived":
                justification = "Derived feature built from historical context without future leakage."
            else:
                justification = "Candidate feature kept for baseline evaluation."

        rows.append(
            {
                "feature_name": feature_name,
                "source_column_or_source_group": _source_group_for_feature(feature_name, datetime_column),
                "feature_origin": _feature_origin_for_feature(
                    feature_name,
                    datetime_column=datetime_column,
                    raw_feature_columns=raw_feature_columns,
                    context_columns=context_columns,
                ),
                "feature_type": feature_type,
                "system_layer": feature_layer,
                "temporal_relation_to_target": temporal_relation,
                "output_role": output_role,
                "allowed_model_input": "candidate",
                "status": status,
                "justification": justification,
                "null_ratio": round(null_ratio, 6),
                "cardinality": cardinality,
                "correlation_flag_or_redundancy_flag": correlation_flag,
                "leakage_risk": leakage_risk,
                "notes": notes,
            }
        )

    return pd.DataFrame(rows)


def apply_manual_feature_decisions(
    catalog: pd.DataFrame,
    decisions: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Apply notebook-level status and justification overrides to the catalog."""
    updated = catalog.copy()
    for feature_name, overrides in decisions.items():
        mask = updated["feature_name"] == feature_name
        for field_name, value in overrides.items():
            if field_name in updated.columns:
                updated.loc[mask, field_name] = value
    return updated


def build_feature_selection_export(
    catalog: pd.DataFrame,
    *,
    target_column: str,
    datetime_column: str,
    include_generic_keep_in_extended: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a config-friendly export from the final feature catalog."""
    minimal_catalog = catalog[catalog["status"] == "keep_minimal"].copy()
    extended_statuses = ["keep_minimal", "keep_extended", "keep"] if include_generic_keep_in_extended else ["keep_minimal", "keep_extended"]
    extended_catalog = catalog[catalog["status"].isin(extended_statuses)].copy()
    review_catalog = catalog[catalog["status"] == "review"].copy()
    drop_catalog = catalog[catalog["status"] == "drop"].copy()

    selected_features_minimal = [
        feature
        for feature in minimal_catalog["feature_name"].tolist()
        if feature not in {target_column, datetime_column}
    ]
    selected_features_extended_only = [
        feature
        for feature in extended_catalog["feature_name"].tolist()
        if feature not in {target_column, datetime_column}
    ]
    selected_features_extended = list(dict.fromkeys(selected_features_minimal + selected_features_extended_only))

    export = {
        "target": target_column,
        "selected_features_minimal": selected_features_minimal,
        "selected_features_extended": selected_features_extended,
        "selected_features": selected_features_extended,
        "excluded_features": [
            {
                "feature_name": row["feature_name"],
                "reason": row["justification"],
            }
            for _, row in drop_catalog.iterrows()
        ],
        "review_features": [
            {
                "feature_name": row["feature_name"],
                "reason": row["justification"],
            }
            for _, row in review_catalog.iterrows()
        ],
        "review_pool_features": [
            {
                "feature_name": row["feature_name"],
                "reason": row["justification"],
            }
            for _, row in review_catalog.iterrows()
        ],
        "numerical_features": extended_catalog[extended_catalog["feature_type"] == "numerical"]["feature_name"].tolist(),
        "categorical_features": extended_catalog[extended_catalog["feature_type"] == "categorical"]["feature_name"].tolist(),
        "temporal_features": extended_catalog[
            extended_catalog["feature_type"].isin(["temporal", "derived_temporal"])
        ]["feature_name"].tolist(),
        "derived_features": extended_catalog[extended_catalog["feature_type"] == "derived"]["feature_name"].tolist(),
    }
    if metadata:
        export.update(metadata)
    return export


def write_feature_outputs(
    catalog: pd.DataFrame,
    feature_export: dict[str, Any],
    *,
    catalog_path: str | Path,
    export_yaml_path: str | Path,
) -> tuple[Path, Path]:
    """Persist the feature catalog and the config-oriented export."""
    catalog_target = Path(catalog_path)
    export_target = Path(export_yaml_path)
    catalog_target.parent.mkdir(parents=True, exist_ok=True)
    export_target.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(catalog_target, index=False)
    export_target.write_text(yaml.safe_dump(feature_export, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return catalog_target, export_target


def _feature_reason_map(entries: list[Any] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in entries or []:
        if isinstance(entry, str):
            mapping[str(entry)] = ""
            continue
        if not isinstance(entry, dict):
            continue
        feature_name = entry.get("feature_name") or entry.get("name")
        if not feature_name:
            continue
        reason = entry.get("reason") or entry.get("justification") or entry.get("notes") or ""
        mapping[str(feature_name)] = str(reason)
    return mapping


def build_feature_contract_from_config(
    config: dict[str, Any],
    *,
    modeling_df: pd.DataFrame | None = None,
    modeling_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build a notebook-friendly feature contract directly from config selections.

    This is used as a fallback when the exported feature-contract artifacts are not
    available on disk but the notebook still needs a traceable, readable contract.
    """
    feature_cfg = dict(config.get("feature_selection", {}))
    data_dataset_cfg = dict(config.get("data_processing", {}).get("dataset", {}))
    procurement_cfg = dict(config.get("procurement_problem_definition", {}))
    synthetic_cfg = dict(config.get("synthetic_data", {}))
    target_roles_cfg = dict(synthetic_cfg.get("target_roles", {}))

    datetime_column = str(data_dataset_cfg.get("datetime_column", "date"))
    raw_feature_columns = list(data_dataset_cfg.get("feature_columns", []))
    target_column = str(
        feature_cfg.get("target_primary")
        or feature_cfg.get("target")
        or procurement_cfg.get("target_column")
        or target_roles_cfg.get("canonical_target_column")
        or "synthetic_procurement_need"
    )
    alternative_target_columns = list(
        dict.fromkeys(
            [
                target_column,
                *feature_cfg.get("target_alternatives", []),
                procurement_cfg.get("refined_target_column"),
                procurement_cfg.get("requirement_signal_column"),
            ]
        )
    )
    alternative_target_columns = [column for column in alternative_target_columns if column]

    context_columns = list(
        procurement_cfg.get("manufacturing_context_columns")
        or synthetic_cfg.get("manufacturing_context_columns")
        or []
    )
    decision_output_columns = list(
        dict.fromkeys(
            [
                procurement_cfg.get("decision_output_column"),
                target_roles_cfg.get("decision_output_column"),
            ]
        )
    )
    decision_output_columns = [column for column in decision_output_columns if column]

    minimal_features = {str(feature) for feature in feature_cfg.get("selected_features_minimal", [])}
    extended_features = {str(feature) for feature in feature_cfg.get("selected_features_extended", feature_cfg.get("selected_features", []))}
    review_features = _feature_reason_map(feature_cfg.get("review_pool_features", feature_cfg.get("review_features", [])))
    excluded_features = _feature_reason_map(feature_cfg.get("excluded_features", []))
    loaded_columns = [str(column) for column in (modeling_columns or [])]
    if modeling_df is not None:
        loaded_columns = [str(column) for column in modeling_df.columns.tolist()]

    feature_names = list(
        dict.fromkeys(
            [
                *loaded_columns,
                *minimal_features,
                *extended_features,
                *review_features.keys(),
                *excluded_features.keys(),
                *context_columns,
                *decision_output_columns,
                *alternative_target_columns,
                datetime_column,
            ]
        )
    )

    rows: list[dict[str, Any]] = []
    for feature_name in feature_names:
        source_group = _source_group_for_feature(feature_name, datetime_column)
        feature_origin = _feature_origin_for_feature(
            feature_name,
            datetime_column=datetime_column,
            raw_feature_columns=raw_feature_columns,
            context_columns=context_columns,
        )
        if modeling_df is not None and feature_name in modeling_df.columns:
            feature_type = _feature_type_for_column(
                feature_name,
                modeling_df[feature_name],
                datetime_column=datetime_column,
                raw_feature_columns=raw_feature_columns,
                context_columns=context_columns,
                categorical_threshold=20,
            )
        else:
            if feature_name == datetime_column:
                feature_type = "temporal"
            elif feature_name.startswith("date_"):
                feature_type = "derived_temporal"
            elif "__" in feature_name:
                feature_type = "categorical_context"
            elif (
                feature_name.endswith("_gap")
                or feature_name.endswith("_ratio")
                or "_lag_" in feature_name
                or "_roll_" in feature_name
            ):
                feature_type = "derived"
            elif source_group in context_columns:
                feature_type = "numerical_context"
            else:
                feature_type = "numerical"
        system_layer = _feature_layer_for_feature(
            feature_name,
            target_column=target_column,
            datetime_column=datetime_column,
            raw_feature_columns=raw_feature_columns,
            context_columns=context_columns,
            decision_output_columns=decision_output_columns,
            alternative_target_columns=alternative_target_columns,
        )
        temporal_relation = _temporal_relation_for_feature(
            feature_name,
            target_column=target_column,
            datetime_column=datetime_column,
        )
        output_role = _output_role_for_feature(
            feature_name,
            target_column=target_column,
            decision_output_columns=decision_output_columns,
        )
        leakage_risk = _default_leakage_risk(
            feature_name,
            target_column=target_column,
            target_base_column=target_column,
            datetime_column=datetime_column,
        )
        present_in_loaded_dataset = "yes" if feature_name in loaded_columns else "no"

        if feature_name in minimal_features:
            official_baseline_status = "keep_minimal"
            allowed_model_input = "yes"
            included_in_minimal_baseline = "yes"
            included_in_extended_baseline = "yes"
            justification = "Included in the official minimal baseline feature set."
        elif feature_name in extended_features:
            official_baseline_status = "keep_extended"
            allowed_model_input = "yes"
            included_in_minimal_baseline = "no"
            included_in_extended_baseline = "yes"
            justification = "Included only in the official extended baseline feature set."
        elif feature_name in review_features:
            official_baseline_status = "review"
            allowed_model_input = "review"
            included_in_minimal_baseline = "no"
            included_in_extended_baseline = "no"
            justification = review_features[feature_name] or "Review pool feature retained for diagnostic evaluation only."
        elif (
            feature_name in excluded_features
            or feature_name == datetime_column
            or output_role == "predictive_target"
            or output_role == "decision_output"
            or feature_name.startswith("synthetic_procurement_need")
        ):
            official_baseline_status = "drop"
            allowed_model_input = "no"
            included_in_minimal_baseline = "no"
            included_in_extended_baseline = "no"
            justification = excluded_features.get(feature_name) or "Excluded from the official baseline feature sets."
        else:
            official_baseline_status = "drop"
            allowed_model_input = "no"
            included_in_minimal_baseline = "no"
            included_in_extended_baseline = "no"
            if feature_name in loaded_columns:
                justification = "Available in the dataset, but excluded from the official baseline and review pool."
            else:
                justification = "Configured for contract tracing, but not present in the dataset currently loaded."

        notes = ""
        if present_in_loaded_dataset == "no":
            notes = "Not present in the loaded notebook dataset."

        rows.append(
            {
                "feature_name": feature_name,
                "source_column_or_source_group": source_group,
                "feature_origin": feature_origin,
                "feature_type": feature_type,
                "system_layer": system_layer,
                "temporal_relation_to_target": temporal_relation,
                "output_role": output_role,
                "allowed_model_input": allowed_model_input,
                "official_baseline_status": official_baseline_status,
                "included_in_minimal_baseline": included_in_minimal_baseline,
                "included_in_extended_baseline": included_in_extended_baseline,
                "leakage_risk": leakage_risk,
                "present_in_loaded_dataset": present_in_loaded_dataset,
                "justification": justification,
                "notes": notes,
            }
        )

    contract_df = pd.DataFrame(rows)
    status_order = [
        "keep_minimal",
        "keep_extended",
        "review",
        "drop",
    ]
    contract_df["official_baseline_status"] = pd.Categorical(
        contract_df["official_baseline_status"],
        categories=status_order,
        ordered=True,
    )
    return contract_df.sort_values(["official_baseline_status", "feature_name"]).reset_index(drop=True)


def build_feature_roles_metadata_from_config(
    config: dict[str, Any],
    feature_contract_df: pd.DataFrame,
    *,
    dataset_path: str | None = None,
    available_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Build notebook-friendly feature-role metadata from config and contract."""
    feature_cfg = dict(config.get("feature_selection", {}))
    procurement_cfg = dict(config.get("procurement_problem_definition", {}))
    synthetic_cfg = dict(config.get("synthetic_data", {}))
    target_roles_cfg = dict(synthetic_cfg.get("target_roles", {}))

    target_column = str(
        feature_cfg.get("target_primary")
        or feature_cfg.get("target")
        or procurement_cfg.get("target_column")
        or target_roles_cfg.get("canonical_target_column")
        or "synthetic_procurement_need"
    )
    configured_feature_sets = feature_cfg.get("feature_sets", {})
    if not isinstance(configured_feature_sets, dict):
        configured_feature_sets = {}
    available = set(available_columns or feature_contract_df["feature_name"].astype(str).tolist())

    return {
        "target_column": target_column,
        "target_alternatives": [
            value
            for value in dict.fromkeys(
                [
                    *feature_cfg.get("target_alternatives", []),
                    procurement_cfg.get("refined_target_column"),
                ]
            )
            if value
        ],
        "requirement_signal_column": procurement_cfg.get(
            "requirement_signal_column",
            target_roles_cfg.get("target_requirement_column", "synthetic_raw_material_requirement"),
        ),
        "decision_output_column": procurement_cfg.get(
            "decision_output_column",
            target_roles_cfg.get("decision_output_column", "order_quantity_tons"),
        ),
        "validated_base_horizon_weeks": int(
            procurement_cfg.get(
                "validated_base_horizon_weeks",
                synthetic_cfg.get("validated_base_horizon_weeks", 1),
            )
        ),
        "validated_base_horizon_label": str(
            procurement_cfg.get(
                "validated_base_horizon_label",
                synthetic_cfg.get("validated_base_horizon_label", "W+1"),
            )
        ),
        "manufacturing_context_columns": list(
            procurement_cfg.get("manufacturing_context_columns")
            or synthetic_cfg.get("manufacturing_context_columns")
            or []
        ),
        "official_minimal_inputs": feature_contract_df[
            feature_contract_df["included_in_minimal_baseline"].eq("yes")
        ]["feature_name"].tolist(),
        "official_extended_inputs": feature_contract_df[
            feature_contract_df["included_in_extended_baseline"].eq("yes")
        ]["feature_name"].tolist(),
        "feature_sets": {
            str(name): [
                str(feature)
                for feature in features
                if isinstance(feature, str) and str(feature) in available
            ]
            for name, features in configured_feature_sets.items()
            if isinstance(features, list)
        },
        "recommended_feature_set": str(feature_cfg.get("recommended_feature_set", "extended")),
        "review_pool_features": feature_contract_df[
            feature_contract_df["official_baseline_status"].astype(str).eq("review")
        ]["feature_name"].tolist(),
        "loaded_dataset_path": dataset_path,
        "recipe_registry_path": config.get("manufacturing_profiles", {}).get("registry_path"),
        "recipe_context": dict(config.get("runtime", {}).get("recipe_context", {})),
    }
