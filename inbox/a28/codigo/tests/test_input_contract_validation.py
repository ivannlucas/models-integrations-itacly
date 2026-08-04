from __future__ import annotations

import pandas as pd

from src.validation import validate_input_dataframe, validate_input_file


def test_valid_csv_passes() -> None:
    _, report = validate_input_file("data/demo/customer_upload_example.csv")
    assert report["valid"] is True
    assert report["missing_columns"] == []


def test_missing_required_column_fails() -> None:
    dataframe = pd.read_csv("data/demo/customer_upload_example.csv").drop(columns=["destination_profile"])
    report = validate_input_dataframe(dataframe)
    assert report["valid"] is False
    assert "destination_profile" in report["missing_columns"]


def test_negative_values_fail() -> None:
    dataframe = pd.read_csv("data/demo/customer_upload_example.csv")
    dataframe.loc[0, "current_inventory_tons"] = -1
    report = validate_input_dataframe(dataframe)
    assert report["valid"] is False
    assert any("current_inventory_tons" in error for error in report["errors"])


def test_yield_and_waste_out_of_range_fail() -> None:
    dataframe = pd.read_csv("data/demo/customer_upload_example.csv")
    dataframe.loc[0, "expected_yield_rate"] = 1.2
    dataframe.loc[1, "expected_waste_rate"] = -0.1
    report = validate_input_dataframe(dataframe)
    assert report["valid"] is False
    assert any("expected_yield_rate" in error for error in report["errors"])
    assert any("expected_waste_rate" in error for error in report["errors"])
