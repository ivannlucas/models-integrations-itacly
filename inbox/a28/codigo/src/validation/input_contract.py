"""Validation helpers for the CU28 customer input contract."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "raw_material_id",
    "current_inventory_tons",
    "expected_requirement_tons",
    "lead_time_days",
    "safety_coverage_days",
    "expected_yield_rate",
    "expected_waste_rate",
    "unit_purchase_cost",
    "shelf_life_days",
    "destination_profile",
]

OPTIONAL_COLUMNS: list[str] = []


def _append_numeric_validation_errors(
    report: dict[str, object],
    df: pd.DataFrame,
    column: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    numeric_values = pd.to_numeric(df[column], errors="coerce")
    if numeric_values.isna().any():
        report["errors"].append(f"Column '{column}' contains non-numeric values.")  # type: ignore[index]
        return
    if min_value is not None and (numeric_values < min_value).any():
        report["errors"].append(f"Column '{column}' contains values below {min_value}.")  # type: ignore[index]
    if max_value is not None and (numeric_values > max_value).any():
        report["errors"].append(f"Column '{column}' contains values above {max_value}.")  # type: ignore[index]


def validate_input_dataframe(df: pd.DataFrame) -> dict[str, object]:
    """Validate the customer input frame against the official contract."""

    report: dict[str, object] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "required_columns": REQUIRED_COLUMNS,
        "missing_columns": [],
    }

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    report["missing_columns"] = missing_columns
    if missing_columns:
        report["errors"].append(f"Missing required columns: {', '.join(missing_columns)}.")  # type: ignore[index]
        report["valid"] = False
        return report

    null_columns = [column for column in REQUIRED_COLUMNS if df[column].isna().any()]
    if null_columns:
        report["errors"].append(f"Required columns contain null values: {', '.join(null_columns)}.")  # type: ignore[index]

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        report["errors"].append("Column 'date' contains non-parseable values.")  # type: ignore[index]

    _append_numeric_validation_errors(report, df, "current_inventory_tons", min_value=0.0)
    _append_numeric_validation_errors(report, df, "expected_requirement_tons", min_value=0.0)
    _append_numeric_validation_errors(report, df, "lead_time_days", min_value=0.0)
    _append_numeric_validation_errors(report, df, "safety_coverage_days", min_value=0.0)
    _append_numeric_validation_errors(report, df, "expected_yield_rate", min_value=0.0, max_value=1.0)
    _append_numeric_validation_errors(report, df, "expected_waste_rate", min_value=0.0, max_value=1.0)

    if "unit_purchase_cost" in df.columns:
        _append_numeric_validation_errors(report, df, "unit_purchase_cost", min_value=0.0)
    if "shelf_life_days" in df.columns:
        _append_numeric_validation_errors(report, df, "shelf_life_days", min_value=0.0)

    if not df["destination_profile"].astype(str).str.len().gt(0).all():
        report["errors"].append("Column 'destination_profile' contains empty values.")  # type: ignore[index]
    if not df["raw_material_id"].astype(str).str.len().gt(0).all():
        report["errors"].append("Column 'raw_material_id' contains empty values.")  # type: ignore[index]

    unexpected_columns = sorted(set(df.columns) - set(REQUIRED_COLUMNS) - set(OPTIONAL_COLUMNS))
    if unexpected_columns:
        report["warnings"].append(f"Unexpected columns will be carried through but are not part of the official contract: {', '.join(unexpected_columns)}.")  # type: ignore[index]

    report["valid"] = len(report["errors"]) == 0
    return report


def validate_input_file(path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and validate a customer input CSV."""

    input_path = Path(path)
    dataframe = pd.read_csv(input_path)
    report = validate_input_dataframe(dataframe)
    return dataframe, report


def write_validation_report(report: dict[str, object], output_path: str | Path) -> str:
    """Persist a JSON validation report and return its path."""

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(output_file)
